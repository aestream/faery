import dataclasses
import typing

import numpy

from . import enums, timestamp


@dataclasses.dataclass(frozen=True, init=False)
class WiggleParameters:
    """
    Values that must be passed to different filters to generate a wiggle plot.

    These parameters can be derived from a (more) human-fridenly description of the wiggle plot.

    The __init__ function of this class uses such a description, with reasonnable defaults.
    """

    frequency_hz: float
    decay: enums.Decay
    tau: timestamp.Time
    frame_rate: float
    rewind: bool
    skip: int

    def __init__(
        self,
        time_range: tuple[timestamp.Time, timestamp.Time],
        rewind: bool = True,
        decay: enums.Decay = "exponential",
        output_duration: timestamp.TimeOrTimecode = 3.0 * timestamp.s,
        tau_frames: float = 5.0,
        frame_rate: float = 30.0,
        skip: typing.Optional[int] = None,
    ):
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
        # we use a frozen class here to discourage direct modification
        # of the parameters and encourage the use of __init__.
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "tau", tau_frames / frequency_hz * timestamp.s)
        object.__setattr__(self, "frame_rate", frame_rate)
        object.__setattr__(self, "rewind", rewind)
        object.__setattr__(self, "skip", skip)
