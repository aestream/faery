import dataclasses
import typing

from . import timestamp

if typing.TYPE_CHECKING:
    from .frame_stream import Frame


@dataclasses.dataclass
class FrameState:
    """
    The index and time of a frame in a stream.
    """

    index: int
    t: timestamp.Time


@dataclasses.dataclass
class FrameStreamState:
    """
    Metadata that identifies a frame in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a frame is fully processed.
    """

    frame: typing.Union[FrameState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first frame)

    "end" indicates the end of the stream (after reading the last frame)
    """


@dataclasses.dataclass
class FiniteFrameStreamState:
    """
    Metadata that identifies a frame in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a frame is fully processed.
    """

    frame: typing.Union[FrameState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first frame)

    "end" indicates the end of the stream (after reading the last frame)
    """
    stream_time_range: tuple[timestamp.Time, timestamp.Time]
    """
    The stream's time range in microseconds.
    """
    progress: float
    """
    Ratio of processed frames in the range [0, 1].
    """


@dataclasses.dataclass
class RegularFrameStreamState:
    """
    Metadata that identifies a frame in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a frame is fully processed.
    """

    frame: typing.Union[FrameState, typing.Literal["start", "end"]]
    """
    "start" indicates the beginning of the stream (before reading the first frame)

    "end" indicates the end of the stream (after reading the last frame)
    """
    frequency_hz: float
    """
    The stream's frequency in Hertz.
    """


@dataclasses.dataclass
class FiniteRegularFrameStreamState:
    """
    Metadata that identifies a frame in a stream, intended to display progress.

    Provided to the user via the `on_progress` callback *after* a frame is fully processed.
    """

    frame: typing.Union[FrameState, typing.Literal["start", "end"]]
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
    Ratio of processed frames in the range [0, 1].
    """
    frame_count: int
    """
    Total number of frames in the stream.
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
            self.time_range = stream.time_range()
        except (AttributeError, NotImplementedError):
            self.time_range = None
        try:
            self.frequency_hz = stream.frequency_hz()
        except (AttributeError, NotImplementedError):
            self.frequency_hz = None
        self.on_progress = on_progress
        if self.time_range is None or self.frequency_hz is None:
            self.frame_count = None
        else:
            self.frame_count = 1
            period_us = 1e6 / self.frequency_hz
            while True:
                end = timestamp.Time(
                    microseconds=int(
                        round(
                            self.time_range[0].to_microseconds()
                            + self.frame_count * period_us
                        )
                    )
                )
                if end >= self.time_range[1]:
                    break
                self.frame_count += 1

    def start(self):
        """
        Must be called by the stream consumer just before starting the iteration (before the first commit call).
        """

        if self.time_range is None:
            if self.frequency_hz is None:
                self.on_progress(FrameStreamState(frame="start"))
            else:
                assert self.frame_count is not None
                self.on_progress(
                    RegularFrameStreamState(
                        frame="start",
                        frequency_hz=self.frequency_hz,
                    )
                )
        else:
            if self.frequency_hz is None:
                self.on_progress(
                    FiniteFrameStreamState(
                        frame="start",
                        stream_time_range=self.time_range,
                        progress=0.0,
                    )
                )
            else:
                assert self.frame_count is not None
                self.on_progress(
                    FiniteRegularFrameStreamState(
                        frame="start",
                        stream_time_range=self.time_range,
                        frequency_hz=self.frequency_hz,
                        progress=0.0,
                        frame_count=self.frame_count,
                    )
                )

    def commit(self, frame: "Frame"):
        """
        Must be called by the stream consumer after processing a frame.
        """

        if self.frequency_hz is None:
            if self.time_range is None:
                self.on_progress(
                    FrameStreamState(
                        frame=FrameState(
                            index=self.index,
                            t=frame.t,
                        )
                    )
                )
            else:
                self.on_progress(
                    FiniteFrameStreamState(
                        frame=FrameState(
                            index=self.index,
                            t=frame.t,
                        ),
                        stream_time_range=self.time_range,
                        progress=(frame.t - self.time_range[0])
                        / (
                            self.time_range[1].to_microseconds()
                            - self.time_range[0].to_microseconds()
                        ),
                    )
                )
        else:
            if self.time_range is None:
                self.on_progress(
                    RegularFrameStreamState(
                        frame=FrameState(
                            index=self.index,
                            t=frame.t,
                        ),
                        frequency_hz=self.frequency_hz,
                    )
                )
            else:
                assert self.frame_count is not None
                self.on_progress(
                    FiniteRegularFrameStreamState(
                        frame=FrameState(
                            index=self.index,
                            t=frame.t,
                        ),
                        stream_time_range=self.time_range,
                        frequency_hz=self.frequency_hz,
                        progress=(
                            frame.t.to_microseconds()
                            + 1
                            - self.time_range[0].to_microseconds()
                        )
                        / (
                            self.time_range[1].to_microseconds()
                            - self.time_range[0].to_microseconds()
                        ),
                        frame_count=self.frame_count,
                    )
                )
        self.index += 1

    def end(self):
        """
        Must be called by the stream consumer after processing the last frame (after the last commit call).
        """

        if self.time_range is None:
            if self.frequency_hz is None:
                self.on_progress(FrameStreamState(frame="end"))
            else:
                assert self.frame_count is not None
                self.on_progress(
                    RegularFrameStreamState(
                        frame="end",
                        frequency_hz=self.frequency_hz,
                    )
                )
        else:
            if self.frequency_hz is None:
                self.on_progress(
                    FiniteFrameStreamState(
                        frame="end",
                        stream_time_range=self.time_range,
                        progress=1.0,
                    )
                )
            else:
                assert self.frame_count is not None
                self.on_progress(
                    FiniteRegularFrameStreamState(
                        frame="end",
                        stream_time_range=self.time_range,
                        frequency_hz=self.frequency_hz,
                        progress=1.0,
                        frame_count=self.frame_count,
                    )
                )
