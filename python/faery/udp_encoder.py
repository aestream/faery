import collections.abc
import socket
import typing

import numpy

from . import enums, events_stream_state


def encode(
    stream: collections.abc.Iterable[numpy.ndarray],
    address: typing.Union[
        tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
    ],
    events_per_packet: typing.Optional[int] = None,
    format: enums.UdpFormat = "t64_x16_y16_on8",
    on_progress: typing.Callable[
        [events_stream_state.EventsStreamState], None
    ] = lambda _: None,
):
    if events_per_packet is None:
        events_per_packet = 100

    ipv6 = len(address) == 4
    if ipv6:
        if address[2] is None and address[3] is None:
            address = (address[0], address[1])
        elif address[3] is None:
            address = (address[0], address[1], address[2])  # type: ignore
    udp_socket = socket.socket(
        socket.AF_INET6 if ipv6 else socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    state_manager = events_stream_state.StateManager(
        stream=stream, on_progress=on_progress
    )
    if format == "t64_x16_y16_on8":
        state_manager.start()
        for events in stream:
            for index in range(0, len(events), events_per_packet):
                udp_socket.sendto(
                    events[index : index + events_per_packet].tobytes(), address
                )
            state_manager.commit(events=events)
        state_manager.end()
    elif format == "t32_x16_y15_on1":
        buffer = numpy.zeros(
            events_per_packet,
            [("t", "<u4"), ("x", "<u2"), ("y+on", "<u2")],
        )
        state_manager.start()
        for events in stream:
            for index in range(0, len(events), events_per_packet):
                selection = events[index : index + events_per_packet]
                buffer[0 : len(selection)]["t"] = selection["t"] & 0xFFFFFFFF
                buffer[0 : len(selection)]["x"] = selection["x"]
                buffer[0 : len(selection)]["y+on"] = (
                    (selection["y"] << 1) & 0x7FFF
                ) | (selection["on"] & 1)
                udp_socket.sendto(buffer[0 : len(selection)].tobytes(), address)
            state_manager.commit(events=events)
        state_manager.end()
    else:
        raise Exception(f'unknown format "{format}"')
