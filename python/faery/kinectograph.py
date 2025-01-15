import collections.abc
import math
import typing

import numpy
import numpy.typing

from . import color as color_module
from . import enums, events_stream_state, frame_stream, timestamp

if typing.TYPE_CHECKING:
    from .types import image  # type: ignore
else:
    from .extension import image

LEGEND_MAXIMUM_RESOLUTION: int = 16384
FONT_WIDTH_RATIO: float = 0.6
FONT_HEIGHT_RATIO: float = 1.31885


class Kinectograph:

    def __init__(
        self,
        time_range: tuple[timestamp.Time, timestamp.Time],
        normalized_times_and_opacities: numpy.typing.NDArray[numpy.float64],
        legend: numpy.typing.NDArray[numpy.float64],
    ):
        self._time_range = time_range
        self.normalized_times_and_opacities = normalized_times_and_opacities
        self.legend = legend

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        """
        Times of the kinectograph's start and end.

        Returns:
            tuple[int, int]: First and one-past-last times.
        """
        return self._time_range

    def dimensions(self) -> tuple[int, int]:
        """
        Kinectograph dimensions in pixels.

        Returns:
            tuple[int, int]: Width (left-right direction) and height (top-bottom direction) in pixels.
        """
        return (
            self.normalized_times_and_opacities.shape[1],
            self.normalized_times_and_opacities.shape[0],
        )

    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "Kinectograph":
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
        return Kinectograph(
            time_range=self._time_range,
            normalized_times_and_opacities=image.resize(
                frame=self.normalized_times_and_opacities,
                new_dimensions=new_dimensions,
                sampling_filter=sampling_filter,
            ),
            legend=self.legend,
        )

    def render(
        self,
        color_theme: color_module.ColorTheme = color_module.DARK_COLOR_THEME,
        legend: bool = True,
        legend_bar_width: int = 30,
        legend_gap: int = 16,
        legend_padding_left: int = 20,
        legend_padding_right: int = 20,
        legend_padding_top: int = 0,
        legend_padding_bottom: int = 0,
        legend_font_size: int = 20,
    ) -> frame_stream.Frame:
        colormap_data = numpy.round(color_theme.colormap.rgba * 255.0).astype(
            dtype=numpy.uint8
        )
        width = self.normalized_times_and_opacities.shape[1]
        if legend:
            start_timecode = self._time_range[0].to_timecode()
            end_timecode = self._time_range[1].to_timecode()
            width += int(
                round(
                    (
                        legend_padding_left
                        + legend_bar_width
                        + legend_gap
                        + max(len(start_timecode), len(end_timecode))
                        * legend_font_size
                        * FONT_WIDTH_RATIO
                        + legend_padding_right
                    )
                )
            )
        frame = numpy.zeros(
            (
                self.normalized_times_and_opacities.shape[0],
                width,
                4,
            ),
            dtype=numpy.uint8,
        )
        frame[:, :] = color_module.color_to_ints(color_theme.background)
        if legend:
            colorbar_height = (
                self.normalized_times_and_opacities.shape[0]
                - legend_padding_top
                - legend_padding_bottom
            )
            if colorbar_height > 1:
                legend_left = (
                    self.normalized_times_and_opacities.shape[1] + legend_padding_left
                )
                colorbar_points = numpy.arange(
                    0, colorbar_height, dtype=numpy.float64
                ) / (colorbar_height - 1)
                colormap_points = numpy.arange(
                    0, color_theme.colormap.rgba.shape[0], dtype=numpy.float64
                ) / (color_theme.colormap.rgba.shape[0] - 1)
                colorbar = numpy.tile(
                    numpy.column_stack(
                        (
                            numpy.interp(
                                colorbar_points,
                                colormap_points,
                                color_theme.colormap.rgba[:, 0],
                            )
                            * 255.0,
                            numpy.interp(
                                colorbar_points,
                                colormap_points,
                                color_theme.colormap.rgba[:, 1],
                            )
                            * 255.0,
                            numpy.interp(
                                colorbar_points,
                                colormap_points,
                                color_theme.colormap.rgba[:, 2],
                            )
                            * 255.0,
                            numpy.interp(
                                colorbar_points,
                                colormap_points,
                                color_theme.colormap.rgba[:, 3],
                            )
                            * 255.0,
                        )
                    ),
                    (legend_bar_width, 1, 1),
                )
                colorbar = numpy.flip(numpy.transpose(colorbar, axes=(1, 0, 2)), axis=0)
                numpy.round(colorbar, out=colorbar)
                frame[
                    legend_padding_top : legend_padding_top + colorbar_height,
                    legend_left : legend_left + legend_bar_width,
                ] = colorbar.astype(numpy.uint8)
                color = color_module.color_to_ints(color_theme.labels)
                image.annotate(
                    frame=frame,
                    text=start_timecode,
                    x=legend_left + legend_bar_width + legend_gap,
                    y=int(
                        round(
                            legend_padding_top
                            + colorbar_height
                            - legend_font_size * FONT_HEIGHT_RATIO
                        )
                    ),
                    size=legend_font_size,
                    color=color,
                )
                image.annotate(
                    frame=frame,
                    text=end_timecode,
                    x=legend_left + legend_bar_width + legend_gap,
                    y=legend_padding_top,
                    size=legend_font_size,
                    color=color,
                )

        normalized_times = self.normalized_times_and_opacities[:, :, 0].copy()
        numpy.multiply(
            normalized_times,
            colormap_data.shape[0] - 1,
            out=normalized_times,
        )
        numpy.round(normalized_times, out=normalized_times)
        overlay = colormap_data[normalized_times.astype(numpy.uint32)]
        overlay[:, :, 3] = numpy.round(
            overlay[:, :, 3].astype(numpy.float64)
            * self.normalized_times_and_opacities[:, :, 1]
        ).astype(numpy.uint8)
        image.overlay(
            frame=frame,
            overlay=overlay,
            x=0,
            y=0,
            new_dimensions=(overlay.shape[1], overlay.shape[0]),
            sampling_filter="nearest",
        )
        return frame_stream.Frame(t=self._time_range[1], pixels=frame)

    @classmethod
    def from_events(
        cls,
        stream: collections.abc.Iterable[numpy.ndarray],
        dimensions: tuple[int, int],
        time_range: tuple[timestamp.Time, timestamp.Time],
        threshold_quantile: float = 0.9,
        normalized_times_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda normalized_times: normalized_times,
        opacities_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda opacities_gamma: opacities_gamma,
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
        normalized_times_and_opacities = numpy.zeros(
            (dimensions[1], dimensions[0], 2), dtype=numpy.float64
        )
        state_manager = events_stream_state.StateManager(
            stream=stream, on_progress=on_progress
        )
        state_manager.start()
        for events in stream:
            numpy.add.at(
                normalized_times_and_opacities[:, :, 0],
                (events["y"], events["x"]),
                events["t"].astype(numpy.float64) * slope + intercept,
            )
            numpy.add.at(
                normalized_times_and_opacities[:, :, 1], (events["y"], events["x"]), 1.0
            )
            state_manager.commit(events=events)
        mask = normalized_times_and_opacities[:, :, 1] > 0
        nonzero_counts = normalized_times_and_opacities[:, :, 1][mask]
        if len(nonzero_counts) == 0:
            count_threshold = 0.0
        else:
            count_threshold = numpy.quantile(nonzero_counts, threshold_quantile)
            normalized_times_and_opacities[:, :, 1] /= count_threshold
            normalized_times_and_opacities[:, :, 1][
                normalized_times_and_opacities[:, :, 1] > 1.0
            ] = 1.0
        normalized_times_and_opacities[:, :, 0][mask] /= nonzero_counts
        state_manager.end()
        normalized_times_and_opacities[:, :, 0] = normalized_times_gamma(
            normalized_times_and_opacities[:, :, 0]
        )
        normalized_times_and_opacities[:, :, 1] = opacities_gamma(
            normalized_times_and_opacities[:, :, 1]
        )
        return cls(
            time_range=time_range,
            normalized_times_and_opacities=normalized_times_and_opacities,
            legend=normalized_times_gamma(
                numpy.linspace(0.0, 1.0, LEGEND_MAXIMUM_RESOLUTION)
            ),
        )
