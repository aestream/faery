import dataclasses
import typing

import numpy

from . import timestamp


@dataclasses.dataclass
class PacketState:
    """
    The index and time range of an events packet in a stream.

    The events' timestamps in the packet are guaranteed to be larger than
    or equal to time_range[0] and strictly smaller than time_range[1].
    """

    index: int
    time_range: tuple[timestamp.Time, timestamp.Time]


@dataclasses.dataclass
class EventsStreamState:
    """
    Metadata that identifies an events packet in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a packet is fully processed.
    """

    packet: typing.Union[PacketState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first packet)

    "end" indicates the end of the stream (after reading the last packet)
    """


@dataclasses.dataclass
class FiniteEventsStreamState:
    """
    Metadata that identifies an events packet in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a packet is fully processed.
    """

    packet: typing.Union[PacketState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first packet)

    "end" indicates the end of the stream (after reading the last packet)
    """
    stream_time_range: tuple[timestamp.Time, timestamp.Time]
    """
    The stream's time range in microseconds.
    """
    progress: float
    """
    Ratio of processed packets in the range [0, 1].
    """


@dataclasses.dataclass
class RegularEventsStreamState:
    """
    Metadata that identifies an events packet in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a packet is fully processed.
    """

    packet: typing.Union[PacketState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first packet)

    "end" indicates the end of the stream (after reading the last packet)
    """
    frequency_hz: float
    """
    The stream's frequency in Hertz.
    """


@dataclasses.dataclass
class FiniteRegularEventsStreamState:
    """
    Metadata that identifies an events packet in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a packet is fully processed.
    """

    packet: typing.Union[PacketState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first packet)

    "end" indicates the end of the stream (after reading the last packet)
    """
    stream_time_range: tuple[timestamp.Time, timestamp.Time]
    """
    The stream's time range in microseconds.
    """
    frequency_hz: float
    """
    The stream's frequency in Hertz.
    """
    progress: float
    """
    Ratio of processed packets in the range [0, 1].
    """
    packet_count: int
    """
    Total number of packets in the stream.
    """


class StateManager:
    """
    Keeps track of the number of frames processed by the stream and calls `on_progress`.
    """

    def __init__(
        self,
        stream: typing.Any,
        on_progress: typing.Callable[[typing.Any], None],
    ):
        self.index = 0
        try:
            self.time_range: typing.Optional[tuple[timestamp.Time, timestamp.Time]] = (
                stream.time_range()
            )
        except (AttributeError, NotImplementedError):
            self.time_range = None
        try:
            self.frequency_hz = stream.frequency_hz()
        except (AttributeError, NotImplementedError):
            self.frequency_hz = None
        self.on_progress = on_progress
        if self.time_range is None or self.frequency_hz is None:
            self.packet_count = None
        else:
            self.packet_count = 1
            period_us = 1e6 / self.frequency_hz
            while True:
                end = timestamp.Time(
                    microseconds=int(
                        round(
                            self.time_range[0].to_microseconds()
                            + self.packet_count * period_us
                        )
                    )
                )
                if end >= self.time_range[1]:
                    break
                self.packet_count += 1

    def start(self):
        """
        Must be called by the stream consumer just before starting the iteration (before the first commit call).
        """
        if self.time_range is None:
            if self.frequency_hz is None:
                self.on_progress(EventsStreamState(packet="start"))
            else:
                assert self.packet_count is not None
                self.on_progress(
                    RegularEventsStreamState(
                        packet="start",
                        frequency_hz=self.frequency_hz,
                    )
                )
        else:
            if self.frequency_hz is None:
                self.on_progress(
                    FiniteEventsStreamState(
                        packet="start",
                        stream_time_range=self.time_range,
                        progress=0.0,
                    )
                )
            else:
                assert self.packet_count is not None
                self.on_progress(
                    FiniteRegularEventsStreamState(
                        packet="start",
                        stream_time_range=self.time_range,
                        frequency_hz=self.frequency_hz,
                        progress=0.0,
                        packet_count=self.packet_count,
                    )
                )

    def commit(self, events: numpy.ndarray):
        """
        Must be called by the stream consumer after processing an events packet.
        """
        if len(events) == 0:
            if self.frequency_hz is None:
                return
            if self.time_range is None:
                pass
            else:
                pass
        else:
            if self.frequency_hz is None:
                if self.time_range is None:
                    self.on_progress(
                        EventsStreamState(
                            packet=PacketState(
                                index=self.index,
                                time_range=(
                                    timestamp.Time(microseconds=int(events["t"][0])),
                                    timestamp.Time(
                                        microseconds=int(events["t"][-1]) + 1
                                    ),
                                ),
                            )
                        )
                    )
                else:
                    self.on_progress(
                        FiniteEventsStreamState(
                            packet=PacketState(
                                index=self.index,
                                time_range=(
                                    timestamp.Time(microseconds=int(events["t"][0])),
                                    timestamp.Time(
                                        microseconds=int(events["t"][-1]) + 1
                                    ),
                                ),
                            ),
                            stream_time_range=self.time_range,
                            progress=(
                                events["t"][-1]
                                + 1
                                - self.time_range[0].to_microseconds()
                            )
                            / (
                                self.time_range[1].to_microseconds()
                                - self.time_range[0].to_microseconds()
                            ),
                        )
                    )
            else:
                if self.time_range is None:
                    self.on_progress(
                        RegularEventsStreamState(
                            packet=PacketState(
                                index=self.index,
                                time_range=(
                                    timestamp.Time(microseconds=int(events["t"][0])),
                                    timestamp.Time(
                                        microseconds=int(events["t"][-1]) + 1
                                    ),
                                ),
                            ),
                            frequency_hz=self.frequency_hz,
                        )
                    )
                else:
                    assert self.packet_count is not None
                    self.on_progress(
                        FiniteRegularEventsStreamState(
                            packet=PacketState(
                                index=self.index,
                                time_range=(
                                    timestamp.Time(microseconds=int(events["t"][0])),
                                    timestamp.Time(
                                        microseconds=int(events["t"][-1]) + 1
                                    ),
                                ),
                            ),
                            stream_time_range=self.time_range,
                            frequency_hz=self.frequency_hz,
                            progress=(
                                events["t"][-1]
                                + 1
                                - self.time_range[0].to_microseconds()
                            )
                            / (
                                self.time_range[1].to_microseconds()
                                - self.time_range[0].to_microseconds()
                            ),
                            packet_count=self.packet_count,
                        )
                    )
        self.index += 1

    def end(self):
        """
        Must be called by the stream consumer after processing the last events packet (after the last commit call).
        """

        if self.time_range is None:
            if self.frequency_hz is None:
                self.on_progress(EventsStreamState(packet="end"))
            else:
                assert self.packet_count is not None
                self.on_progress(
                    RegularEventsStreamState(
                        packet="end",
                        frequency_hz=self.frequency_hz,
                    )
                )
        else:
            if self.frequency_hz is None:
                self.on_progress(
                    FiniteEventsStreamState(
                        packet="end",
                        stream_time_range=self.time_range,
                        progress=1.0,
                    )
                )
            else:
                assert self.packet_count is not None
                self.on_progress(
                    FiniteRegularEventsStreamState(
                        packet="end",
                        stream_time_range=self.time_range,
                        frequency_hz=self.frequency_hz,
                        progress=1.0,
                        packet_count=self.packet_count,
                    )
                )
