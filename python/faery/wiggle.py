import dataclasses
import typing

import numpy

from . import enums, timestamp


@dataclasses.dataclass(frozen=True)
class WiggleParameters:
    frequency_hz: float
    decay: enums.Decay
    tau: timestamp.Time
    frame_rate: float
    rewind: bool
    skip: int


def wiggle_parameters(
    time_range: tuple[timestamp.Time, timestamp.Time],
    rewind: bool = True,
    decay: enums.Decay = "exponential",
    output_duration: timestamp.TimeOrTimecode = 3.0 * timestamp.s,
    tau_frames: float = 5.0,
    frame_rate: float = 30.0,
    skip: typing.Optional[int] = None,
) -> "WiggleParameters":
    output_frames = int(
        numpy.ceil(frame_rate * timestamp.parse_time(output_duration).to_seconds())
    )
    if skip is None:
        if decay == "exponential":
            skip = int(numpy.ceil(3.0 * tau_frames))
        elif decay == "linear":
            skip = int(numpy.ceil(2.0 * tau_frames))
        else:
            skip = int(numpy.ceil(tau_frames))
    if rewind and output_frames >= 3:
        output_frames = int(numpy.ceil((output_frames + 2) / 2))
    output_frames += skip
    frequency_hz = output_frames / (time_range[1] - time_range[0]).to_seconds()
    return WiggleParameters(
        frequency_hz=frequency_hz,
        decay=decay,
        tau=tau_frames / frequency_hz * timestamp.s,
        frame_rate=frame_rate,
        rewind=rewind,
        skip=skip,
    )
