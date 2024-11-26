import collections.abc
import socket
import threading
import time
import types
import typing

import numpy

from . import enums, events_stream


class Receiver:

    def target(self):
        while self.running:
            try:
                read = self.socket.recv_into(self.buffer)
                if read > 0:
                    self.queue.append(bytes(self.buffer[0:read]))
            except TimeoutError:
                pass

    def __init__(
        self,
        address: typing.Union[
            tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
        ],
    ):
        ipv6 = len(address) == 4
        if ipv6:
            if address[2] is None and address[3] is None:
                self.address = (address[0], address[1])
            elif address[3] is None:
                self.address = (address[0], address[1], address[2])  # type: ignore
        self.socket = socket.socket(
            socket.AF_INET6 if ipv6 else socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.socket.settimeout(0.1)
        self.socket.bind(address)
        self.buffer = bytearray(65536)
        self.queue: collections.deque[bytes] = collections.deque()
        self.running = True
        self.thread = threading.Thread(target=self.target, daemon=True)
        self.thread.start()

    def __enter__(self) -> "Receiver":
        return self

    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool:
        self.running = False
        self.thread.join()
        return False

    def next(self) -> bytes:
        while True:
            try:
                return self.queue.popleft()
            except IndexError:
                time.sleep(0.02)


class Decoder(events_stream.EventsStream):
    def __init__(
        self,
        dimensions: tuple[int, int],
        address: typing.Union[
            tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
        ],
        format: enums.UdpFormat = "t64_x16_y16_on8",
    ):
        super().__init__()
        self.inner_dimensions = dimensions
        self.address = address

        self.format = format

    def dimensions(self) -> tuple[int, int]:
        return self.inner_dimensions

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        previous_t = 0
        if self.format == "t64_x16_y16_on8":
            with Receiver(self.address) as receiver:
                while True:
                    raw_bytes = receiver.next()
                    events = numpy.frombuffer(
                        raw_bytes[0 : (len(raw_bytes) // 13) * 13],
                        dtype=events_stream.EVENTS_DTYPE,
                    )
                    if len(events) > 0:
                        if events["t"][0] >= previous_t:
                            previous_t = events["t"][-1]
                            yield events
        elif self.format == "t32_x16_y15_on1":
            with Receiver(self.address) as receiver:
                dtype = numpy.dtype([("t", "<u4"), ("x", "<u2"), ("y+on", "<u2")])
                while True:
                    raw_bytes = receiver.next()
                    raw_events = numpy.frombuffer(
                        raw_bytes[0 : (len(raw_bytes) // 8) * 8],
                        dtype=dtype,
                    )
                    if len(raw_events) > 0:
                        events = numpy.ndarray(
                            len(raw_events),
                            dtype=events_stream.EVENTS_DTYPE,
                        )
                        events["t"] = raw_events["t"]
                        events["x"] = raw_events["x"]
                        events["y"] = raw_events["y+on"] >> 1
                        events["y"] = raw_events["on"] & 1
                        events["t"] += offset
                        if events["t"][0] >= previous_t:
                            previous_t = events["t"][-1]
                            yield events
                        elif previous_t - events["t"][0] > (1 << 31):
                            # if the time difference between the current packet and the previous packet
                            # is more than 2**31, assume a timestamp wrap around rather than a packet drop
                            offset += 1 << 32
                            events["t"] += 1 << 32
                            yield events
        else:
            raise Exception(f'unknown format "{self.format}"')
