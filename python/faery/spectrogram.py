import collections.abc
import math
import typing

import numpy.typing

from . import color as color_module
from . import enums, events_stream_state, timestamp

if typing.TYPE_CHECKING:
    from .types import image  # type: ignore
else:
    from .extension import image


class Spectrogram:

    def __init__(
        self,
        time_range: tuple[timestamp.Time, timestamp.Time],
        frequency_range: tuple[float, float],
        normalized_activities: numpy.typing.NDArray[numpy.float64],
    ):
        self._time_range = time_range
        self.normalized_activities = normalized_activities
        self._frequency_range = frequency_range

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        """
        Times of the kinectograph's start and end.

        Returns:
            tuple[int, int]: First and one-past-last times.
        """
        return self._time_range

    def frequency_range(self) -> tuple[float, float]:
        """
        Minimum and maximum frequencies of the spectrogram, in Hz.

        Returns:
            tuple[int, int]: First and one-past-last frequencies in Hz.
        """
        return self._frequency_range

    def dimensions(self) -> tuple[int, int]:
        """
        Spectrogram dimensions in pixels.

        Returns:
            tuple[int, int]: Width (left-right direction) and height (top-bottom direction) in pixels.
        """
        return (
            self.normalized_activities.shape[1],
            self.normalized_activities.shape[0],
        )

    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "Spectrogram":
        dimensions = self.dimensions()
        if isinstance(factor_or_minimum_dimensions, (float, int)):
            factor = factor_or_minimum_dimensions
        else:
            factor = max(
                1.0,
                math.ceil(factor_or_minimum_dimensions[0] / dimensions[0]),
                math.ceil(factor_or_minimum_dimensions[1] / dimensions[1]),
            )
        if factor == 1.0:
            return self
        new_dimensions = (
            int(round(dimensions[0] * factor)),
            int(round(dimensions[1] * factor)),
        )
        return Spectrogram(
            time_range=self._time_range,
            frequency_range=self._frequency_range,
            normalized_activities=image.resize(
                frame=self.normalized_activities,
                new_dimensions=new_dimensions,
                sampling_filter=sampling_filter,
            ),
        )

    @classmethod
    def from_events(
        cls,
        stream: collections.abc.Iterable[numpy.ndarray],
        dimensions: tuple[int, int],
        time_range: tuple[timestamp.Time, timestamp.Time],
        frequency_range: tuple[float, float],
        spectrogram_dimensions: tuple[int, int],
        on_progress: typing.Callable[
            [events_stream_state.EventsStreamState], None
        ] = lambda _: None,
    ):
        assert time_range[0] < time_range[1]
        if time_range[1].to_microseconds() == time_range[0].to_microseconds() + 1:
            slope = 0.0
            intercept = 0.5
        else:
            slope = 1.0 / (
                time_range[1].to_microseconds() - (time_range[0].to_microseconds() + 1)
            )
            intercept = -slope * time_range[0].to_microseconds()

        normalized_activities = numpy.zeros(
            (spectrogram_dimensions[1], spectrogram_dimensions[0], 2),
            dtype=numpy.float64,
        )
        state_manager = events_stream_state.StateManager(
            stream=stream, on_progress=on_progress
        )
        state_manager.start()
        for events in stream:
            # @DEV compute spectrogram
            state_manager.commit(events=events)
        state_manager.end()
        return cls(
            time_range=time_range,
            frequency_range=frequency_range,
            normalized_activities=normalized_activities,
        )
