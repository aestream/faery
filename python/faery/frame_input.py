import typing

import numpy.typing

from . import frame_stream, timestamp


def frame_stream_from_list(
    start_t: timestamp.TimeOrTimecode,
    frequency_hz: float,
    frames: list[numpy.typing.NDArray[numpy.uint8]],
) -> frame_stream.FrameList:
    return frame_stream.FrameList(
        start_t=start_t,
        frequency_hz=frequency_hz,
        frames=frames,
    )


def frame_stream_from_function(
    start_t: timestamp.TimeOrTimecode,
    frequency_hz: float,
    dimensions: tuple[int, int],
    frame_count: int,
    get_frame: typing.Callable[[timestamp.Time], numpy.typing.NDArray[numpy.uint8]],
) -> frame_stream.FrameFunction:
    return frame_stream.FrameFunction(
        start_t=start_t,
        frequency_hz=frequency_hz,
        dimensions=dimensions,
        frame_count=frame_count,
        get_frame=get_frame,
    )
