import collections.abc
import pathlib
import typing

import numpy.typing

from . import color as color_module
from . import enums, events_stream_state, file_encoder, frame_stream, svg, timestamp

if typing.TYPE_CHECKING:
    from .types import image, raster  # type: ignore
else:
    from .extension import image, raster

CHARACTER_TO_SUPERSCRIPT: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
}
FONT_WIDTH_RATIO: float = 0.6
FONT_HEIGHT_RATIO: float = 1.318848
FONT_OFFSET_RATIO: float = 1.0
AXIS_THICKNESS: int = 4
LINE_THICKNESS: int = 4


def superscript(number: int, use_tspan: bool, font_size: int) -> str:
    number_as_string = str(number)
    if use_tspan:
        length = len(number_as_string) * FONT_WIDTH_RATIO
        return f'<tspan font-size="{int(round(font_size * 0.75))}px" baseline-shift="super" textLength="{length}px">{number_as_string}</tspan>'
    else:
        return "".join(
            CHARACTER_TO_SUPERSCRIPT[character] for character in number_as_string
        )


class EventRate:
    def __init__(
        self,
        time_range: tuple[timestamp.Time, timestamp.Time],
        samples: numpy.typing.NDArray[numpy.float64],
        timestamps: numpy.typing.NDArray[numpy.uint64],
    ):
        self._time_range = time_range
        self.samples = samples
        self.timestamps = timestamps

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        """
        Times of the kinectograph's start and end.

        Returns:
            tuple[Time, Time]: First and one-past-last times.
        """
        return self._time_range

    @classmethod
    def from_events(
        cls,
        stream: collections.abc.Iterable[numpy.ndarray],
        time_range: tuple[timestamp.Time, timestamp.Time],
        samples: int = 1600,
        on_progress: typing.Callable[
            [events_stream_state.EventsStreamState], None
        ] = lambda _: None,
    ):
        assert samples > 0
        state_manager = events_stream_state.StateManager(
            stream=stream, on_progress=on_progress
        )
        state_manager.start()
        buckets = numpy.zeros(samples, dtype=numpy.float64)
        scale = samples / (
            time_range[1].to_microseconds() - time_range[0].to_microseconds()
        )
        for events in stream:
            numpy.add.at(
                buckets,
                numpy.floor(
                    (
                        events["t"].astype(numpy.float64)
                        - time_range[0].to_microseconds()
                    )
                    * scale
                ).astype(numpy.uint64),
                1,
            )
            state_manager.commit(events=events)
        timestamps = (
            numpy.floor(
                numpy.arange(start=0, stop=samples, step=1, dtype=numpy.float64) / scale
            ).astype(numpy.uint64)
            + time_range[0].to_microseconds()
        ).astype(numpy.uint64)
        deltas = (
            numpy.diff(
                numpy.floor(
                    numpy.arange(start=0, stop=samples + 1, step=1, dtype=numpy.float64)
                    / scale
                )
            )
            * 1e-6
        )
        deltas[deltas == 0.0] = 1.0
        buckets /= deltas
        state_manager.end()
        return cls(
            time_range=time_range,
            samples=buckets,
            timestamps=timestamps,
        )

    def filtered_samples(
        self, window: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        assert len(window) > 0
        window = window / numpy.sum(window)
        if len(window) == 1:
            return self.samples * window[0]

        extended_samples = numpy.concatenate(
            (
                numpy.full(len(window) // 2, self.samples[0], dtype=numpy.float64),
                self.samples,
                numpy.full(
                    (len(window) - 1) // 2, self.samples[-1], dtype=numpy.float64
                ),
            )
        )
        sliding_window_view = numpy.lib.stride_tricks.sliding_window_view(
            extended_samples, len(window)
        )
        return numpy.sum(sliding_window_view * window, axis=1)

    def to_svg_string(
        self,
        color_theme: color_module.ColorTheme = color_module.LIGHT_COLOR_THEME,
        hamming_windows_sizes: typing.Iterable[int] = (1, 20),
        x_ticks: typing.Optional[list[tuple[timestamp.Time, str]]] = None,
        y_range: tuple[typing.Optional[float], typing.Optional[float]] = (None, None),
        y_ticks: typing.Optional[list[tuple[float, str]]] = None,
        y_log_scale: bool = True,
        y_range_padding_ratio: float = 0.1,
        graph_width: typing.Optional[int] = None,
        graph_height: int = 1000,
        font_size: int = 30,
        use_tspan_for_superscripts: bool = False,
        indent: str = "    ",
        line_breaks: bool = True,
    ) -> str:
        assert y_range_padding_ratio > 0.0
        if y_range[0] is not None and y_range[1] is not None:
            assert y_range[0] < y_range[1]
        if graph_width is None:
            graph_width = len(self.samples)
        hamming_windows_sizes = list(hamming_windows_sizes)
        series: list[numpy.typing.NDArray[numpy.float64]] = [
            self.filtered_samples(numpy.hamming(hamming_window_size))
            for hamming_window_size in hamming_windows_sizes
        ]

        # choose the graph's y axis minimum and maximum
        # in "auto" mode (y_range[0] is None / y_range[1] is None),
        # the extrema are calculated from the data with a margin (y_range_padding_ratio)
        if y_range[0] is None or y_range[1] is None:
            if y_range[0] is None:
                if y_log_scale:
                    nonnull_series = [
                        filtered_samples[filtered_samples > 0.0]
                        for filtered_samples in series
                    ]
                    if all(
                        len(nonnull_filtered_samples) == 0
                        for nonnull_filtered_samples in nonnull_series
                    ):
                        y_minimum = 1.0
                    else:
                        y_minimum = min(
                            *(
                                float(nonnull_filtered_samples.min())
                                for nonnull_filtered_samples in nonnull_series
                                if len(nonnull_filtered_samples) > 0
                            )
                        )
                else:
                    y_minimum = min(
                        *(float(filtered_samples.min()) for filtered_samples in series)
                    )
            else:
                y_minimum = y_range[0]
            if y_range[1] is None:
                y_maximum = max(
                    *(float(filtered_samples.max()) for filtered_samples in series)
                )
            else:
                y_maximum = y_range[1]
            if y_log_scale:
                if y_minimum == y_maximum:
                    if y_range[0] is None and y_range[1] is None:
                        y_minimum /= 1.0 + 1e-6
                        y_maximum *= 1.0 + 1e-6
                    elif y_range[0] is None:
                        y_minimum = min(y_minimum, y_maximum) / (1.0 + 1e-6)
                    else:
                        y_maximum = max(y_minimum, y_maximum) * (1.0 + 1e-6)
                else:
                    offset = (
                        numpy.log(y_maximum) - numpy.log(y_minimum)
                    ) * y_range_padding_ratio
                    if y_range[0] is None:
                        y_minimum = float(y_minimum * numpy.exp(-offset))
                        for filtered_samples in series:
                            filtered_samples[filtered_samples == 0.0] = y_minimum
                    if y_range[1] is None:
                        y_maximum = float(y_maximum * numpy.exp(offset))
            else:
                if y_minimum == y_maximum:
                    if y_range[0] is None and y_range[1] is None:
                        y_minimum = max(0.0, y_minimum - 1.0)
                        y_maximum += 1.0
                    elif y_range[0] is None:
                        y_minimum = min(y_minimum, y_maximum) - 1.0
                    else:
                        y_maximum = max(y_minimum, y_maximum) + 1.0
                else:
                    offset = (y_maximum - y_minimum) * y_range_padding_ratio
                    if y_range[0] is None:
                        y_minimum -= offset
                        if y_minimum < 0.0:
                            y_minimum = 0.0
                    if y_range[1] is None:
                        y_maximum += offset
        else:
            y_minimum = y_range[0]
            y_maximum = y_range[1]

        # declare the y mapping function
        padding = int(numpy.ceil(FONT_WIDTH_RATIO * font_size))
        if y_log_scale:
            y_to_position = lambda y: (
                -(graph_height - LINE_THICKNESS)
                / numpy.log10(y_maximum / y_minimum)
                * (numpy.log10(y / y_minimum))
                + (padding * 2 + graph_height - LINE_THICKNESS / 2)
            )
        else:
            # @DEV implement linear scale
            y_to_position = lambda y: y

        # find visible y ticks
        y_ticks_labels_maximum_length: int = 0
        if y_ticks is None:
            y_ticks_height = FONT_HEIGHT_RATIO * font_size
            if y_log_scale:
                base_delta = y_to_position(y_minimum) - y_to_position(10.0 * y_minimum)
                if False:  # base_delta >= 5 * y_ticks_height:
                    # @DEV: implement sub-decade log labels (linear)
                    y_ticks_values = []
                    y_ticks_labels = []
                else:
                    exponent_step = int(numpy.ceil((2 * y_ticks_height) / base_delta))
                    exponents = numpy.arange(
                        start=numpy.ceil(numpy.log10(y_minimum)),
                        stop=numpy.floor(numpy.log10(y_maximum)) + 1,
                        step=exponent_step,
                        dtype=numpy.int64,
                    )
                    y_ticks_values = (10.0 ** exponents.astype(numpy.float64)).tolist()
                    y_ticks_labels = [
                        f"10{superscript(exponent, use_tspan=use_tspan_for_superscripts, font_size=font_size)}"
                        for exponent in exponents.tolist()
                    ]
                    y_ticks_labels_maximum_length = max(
                        len(f"10{exponent}") for exponent in exponents.tolist()
                    )
                    if exponent_step == 1:
                        y_subgrid = []
                        for exponent in [exponents[0] - 1] + exponents.tolist():
                            for multiplier in range(2, 10):
                                value = multiplier * (10.0**exponent)
                                if value >= y_minimum and value <= y_maximum:
                                    y_subgrid.append(value)

                    else:
                        subgrid_exponents = numpy.arange(
                            start=numpy.ceil(numpy.log10(y_minimum)),
                            stop=numpy.floor(numpy.log10(y_maximum)) + 1,
                            step=1,
                            dtype=numpy.int64,
                        )
                        y_subgrid = (
                            10.0 ** subgrid_exponents.astype(numpy.float64)
                        ).tolist()
            else:
                # @DEV: implement linear scale
                y_ticks_values = []
                y_ticks_labels = []
                y_subgrid = []
        else:
            y_ticks_values = [
                value
                for value, _ in y_ticks
                if value >= y_minimum and value <= y_maximum
            ]
            y_ticks_labels = [
                label
                for value, label in y_ticks
                if value >= y_minimum and value <= y_maximum
            ]
            y_ticks_labels_maximum_length = max(len(label) for label in y_ticks_labels)
            y_subgrid = []

        # calculate the width of the y labels
        y_ticks_labels_maximum_length = max(
            y_ticks_labels_maximum_length, len("events/s")
        )
        y_ticks_labels_width = int(
            numpy.ceil(FONT_WIDTH_RATIO * font_size * y_ticks_labels_maximum_length)
        )

        # declare the x mapping function
        x_to_position = lambda x: (
            (
                (
                    graph_width
                    / (
                        self._time_range[1].to_microseconds()
                        - 1
                        - self._time_range[0].to_microseconds()
                    )
                    * (x - self._time_range[0].to_microseconds())
                )
                if self._time_range[1].to_microseconds() - 1
                > self._time_range[0].to_microseconds()
                else x * 0.0
            )
            + padding
            + y_ticks_labels_width
            + padding
            + AXIS_THICKNESS
        )

        # find visible x ticks
        time_range = self.time_range()
        x_ticks_labels_width = len("00:00:00.000000") * font_size * FONT_WIDTH_RATIO
        if x_ticks is None:
            x_ticks_step: int = 1
            increase_counter = 0
            while True:
                if (
                    x_to_position(self._time_range[0].to_microseconds() + x_ticks_step)
                    - x_to_position(self._time_range[0].to_microseconds())
                    >= x_ticks_labels_width + 1.5 * font_size * FONT_WIDTH_RATIO
                ):
                    break

                if increase_counter == 0:
                    x_ticks_step *= 2
                    increase_counter = 1
                elif increase_counter == 1:
                    x_ticks_step //= 2
                    x_ticks_step *= 5
                    increase_counter = 2
                else:
                    x_ticks_step *= 2
                    increase_counter = 0
            x_ticks_values = list(
                range(
                    int(
                        numpy.ceil(self._time_range[0].to_microseconds() / x_ticks_step)
                    )
                    * x_ticks_step,
                    int(
                        numpy.floor(
                            self._time_range[1].to_microseconds() / x_ticks_step
                        )
                        + 1
                    )
                    * x_ticks_step,
                    x_ticks_step,
                )
            )
            x_ticks_labels = [
                timestamp.Time(microseconds=value).to_timecode()
                for value in x_ticks_values
            ]
            x_subgrid_step = int(x_ticks_step / 10.0) if x_ticks_step >= 10 else 1
            x_subgrid = list(
                range(
                    int(
                        numpy.ceil(
                            self._time_range[0].to_microseconds() / x_subgrid_step
                        )
                    )
                    * x_subgrid_step,
                    int(
                        numpy.floor(
                            self._time_range[1].to_microseconds() / x_subgrid_step
                        )
                        + 1
                    )
                    * x_subgrid_step,
                    x_subgrid_step,
                )
            )
        else:
            parsed_x_ticks = [
                (value.to_microseconds(), label) for value, label in x_ticks
            ]
            x_ticks_values = [
                value
                for value, _ in parsed_x_ticks
                if value >= self._time_range[0].to_microseconds()
                and value <= self._time_range[1].to_microseconds()
            ]
            x_ticks_labels = [
                label
                for value, label in parsed_x_ticks
                if value >= self._time_range[0].to_microseconds()
                and value <= self._time_range[1].to_microseconds()
            ]
            x_subgrid = []

        # create the figure
        width = (
            padding
            + y_ticks_labels_width
            + padding
            + AXIS_THICKNESS
            + graph_width
            + padding * 2
        )
        height = (
            padding * 2
            + graph_height
            + AXIS_THICKNESS
            + padding
            + int(numpy.ceil(FONT_HEIGHT_RATIO * font_size))
            + padding
        )
        figure = svg.Svg(
            width=width,
            height=height,
        )
        figure.node(
            "rect",
            {
                "id": "background",
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "fill": color_module.color_to_hex_string(color_theme.background),
            },
        )

        # data
        group = figure.node("g", {"id": "data"})
        data_x_positions = x_to_position(self.timestamps)
        x_position_minimum = padding + y_ticks_labels_width + padding + AXIS_THICKNESS
        x_position_maximum = x_position_minimum + graph_width
        y_position_minimum = padding * 2 + LINE_THICKNESS / 2
        y_position_maximum = y_position_minimum + graph_height - LINE_THICKNESS
        for index, (window_size, filtered_samples) in enumerate(
            zip(hamming_windows_sizes, series)
        ):
            data_y_positions = y_to_position(filtered_samples)
            d = ""
            tracing = False
            for x_position, y_position in zip(data_x_positions, data_y_positions):
                if (
                    x_position >= x_position_minimum
                    and x_position <= x_position_maximum
                    and y_position >= y_position_minimum
                    and y_position <= y_position_maximum
                ):
                    if tracing:
                        d += f" L{x_position},{y_position}"
                    elif len(d) == 0:
                        d += f"M{x_position},{y_position}"
                    else:
                        d += f" M{x_position},{y_position}"
                    tracing = True
                else:
                    tracing = False
            group.node(
                "path",
                {
                    "id": f"hamming-{window_size}",
                    "d": d,
                    "fill": "none",
                    "stroke": color_module.color_to_hex_string(
                        color_theme.lines[index % len(color_theme.lines)]
                    ),
                    "stroke-linejoin": "round",
                    "stroke-width": LINE_THICKNESS,
                },
            )

        # x labels and grid
        group = figure.node("g", {"id": "x-labels"})
        x_title_position = (
            padding
            + y_ticks_labels_width
            + padding
            + AXIS_THICKNESS
            + graph_width
            + padding
        )
        x_title_minimum_distance = (
            x_ticks_labels_width / 2 + len("   time") * font_size * FONT_WIDTH_RATIO
        )
        x_grid_positions: list[float] = []
        for value, label in zip(x_ticks_values, x_ticks_labels):
            position = position = float(numpy.floor(x_to_position(value))) + 0.5
            if abs(x_title_position - position) >= x_title_minimum_distance:
                x_grid_positions.append(position)
                group.node(
                    "text",
                    {
                        "x": position,
                        "y": padding * 2
                        + graph_height
                        + AXIS_THICKNESS
                        + padding
                        + FONT_OFFSET_RATIO * font_size,
                        "font-family": "Roboto Mono, monospace",
                        "font-size": f"{font_size}px",
                        "fill": color_module.color_to_hex_string(color_theme.labels),
                        "text-anchor": "middle",
                    },
                ).text(label)
        group.node(
            "text",
            {
                "x": x_title_position,
                "y": padding * 2
                + graph_height
                + AXIS_THICKNESS
                + padding
                + FONT_OFFSET_RATIO * font_size,
                "font-family": "Roboto Mono, monospace",
                "font-size": f"{font_size}px",
                "fill": color_module.color_to_hex_string(color_theme.labels),
                "text-anchor": "end",
            },
        ).text("time")
        if len(x_subgrid) > 0:
            group = figure.node("g", {"id": "x-subgrid"})
            for x in x_subgrid:
                position = float(numpy.floor(x_to_position(x))) + 0.5
                group.node(
                    "line",
                    {
                        "x1": position,
                        "y1": padding,
                        "x2": position,
                        "y2": padding * 2 + graph_height,
                        "stroke": color_module.color_to_hex_string(color_theme.subgrid),
                        "stroke-width": 1.0,
                    },
                )
        group = figure.node("g", {"id": "x-grid"})
        for position in x_grid_positions:
            group.node(
                "line",
                {
                    "x1": position,
                    "y1": padding,
                    "x2": position,
                    "y2": padding * 2 + graph_height,
                    "stroke": color_module.color_to_hex_string(color_theme.grid),
                    "stroke-width": 1.0,
                },
            )

        # y labels and grid
        group = figure.node("g", {"id": "y-labels"})
        y_title_position = padding + FONT_OFFSET_RATIO * font_size
        y_title_minimum_distance = 1.5 * y_ticks_height
        y_grid_positions: list[float] = []
        for value, label in zip(y_ticks_values, y_ticks_labels):
            position = float(numpy.floor(y_to_position(value))) + 0.5
            text_position = (
                position + (FONT_OFFSET_RATIO - FONT_HEIGHT_RATIO / 2) * font_size
            )
            if abs(y_title_position - text_position) >= y_title_minimum_distance:
                y_grid_positions.append(position)
                group.node(
                    "text",
                    {
                        "x": padding + y_ticks_labels_width,
                        "y": text_position,
                        "font-family": "Roboto Mono, monospace",
                        "font-size": f"{font_size}px",
                        "fill": color_module.color_to_hex_string(color_theme.labels),
                        "text-anchor": "end",
                    },
                ).text(label)
        group.node(
            "text",
            {
                "x": padding + y_ticks_labels_width,
                "y": padding + FONT_OFFSET_RATIO * font_size,
                "font-family": "Roboto Mono, monospace",
                "font-size": f"{font_size}px",
                "fill": color_module.color_to_hex_string(color_theme.labels),
                "text-anchor": "end",
            },
        ).text("events/s")
        if len(y_subgrid) > 0:
            group = figure.node("g", {"id": "y-subgrid"})
            for y in y_subgrid:
                position = float(numpy.floor(y_to_position(y))) + 0.5
                group.node(
                    "line",
                    {
                        "x1": padding + y_ticks_labels_width + padding + AXIS_THICKNESS,
                        "y1": position,
                        "x2": padding
                        + y_ticks_labels_width
                        + padding
                        + AXIS_THICKNESS
                        + graph_width
                        + padding,
                        "y2": position,
                        "stroke": color_module.color_to_hex_string(color_theme.subgrid),
                        "stroke-width": 1.0,
                    },
                )
        group = figure.node("g", {"id": "y-grid"})
        for position in y_grid_positions:
            group.node(
                "line",
                {
                    "x1": padding + y_ticks_labels_width + padding + AXIS_THICKNESS,
                    "y1": position,
                    "x2": padding
                    + y_ticks_labels_width
                    + padding
                    + AXIS_THICKNESS
                    + graph_width
                    + padding,
                    "y2": position,
                    "stroke": color_module.color_to_hex_string(color_theme.grid),
                    "stroke-width": 1.0,
                },
            )

        # axes
        figure.node(
            "rect",
            {
                "id": "x-axis",
                "x": padding + y_ticks_labels_width + padding,
                "y": padding * 2 + graph_height,
                "width": AXIS_THICKNESS + graph_width + padding,
                "height": AXIS_THICKNESS,
                "fill": color_module.color_to_hex_string(color_theme.axes),
            },
        )
        figure.node(
            "rect",
            {
                "id": "y-axis",
                "x": padding + y_ticks_labels_width + padding,
                "y": padding,
                "width": AXIS_THICKNESS,
                "height": graph_height + padding,
                "fill": color_module.color_to_hex_string(color_theme.axes),
            },
        )

        return figure.to_string(indent=indent, line_breaks=line_breaks)

    def render(
        self,
        color_theme: color_module.ColorTheme = color_module.LIGHT_COLOR_THEME,
        hamming_windows_sizes: typing.Iterable[int] = (1, 20),
        x_ticks: typing.Optional[list[tuple[timestamp.Time, str]]] = None,
        y_range: tuple[typing.Optional[float], typing.Optional[float]] = (None, None),
        y_ticks: typing.Optional[list[tuple[float, str]]] = None,
        y_log_scale: bool = True,
        y_range_padding_ratio: float = 0.1,
        graph_width: typing.Optional[int] = None,
        graph_height: int = 1000,
        font_size: int = 30,
    ) -> frame_stream.Frame:
        svg_string = self.to_svg_string(
            color_theme=color_theme,
            hamming_windows_sizes=hamming_windows_sizes,
            x_ticks=x_ticks,
            y_range=y_range,
            y_ticks=y_ticks,
            y_log_scale=y_log_scale,
            y_range_padding_ratio=y_range_padding_ratio,
            graph_width=graph_width,
            graph_height=graph_height,
            font_size=font_size,
            use_tspan_for_superscripts=True,
        )
        frame = raster.render(svg_string)
        return frame_stream.Frame(t=self._time_range[1], pixels=frame)

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        color_theme: color_module.ColorTheme = color_module.LIGHT_COLOR_THEME,
        hamming_windows_sizes: typing.Iterable[int] = (1, 20),
        x_ticks: typing.Optional[list[tuple[timestamp.Time, str]]] = None,
        y_range: tuple[typing.Optional[float], typing.Optional[float]] = (None, None),
        y_ticks: typing.Optional[list[tuple[float, str]]] = None,
        y_log_scale: bool = True,
        y_range_padding_ratio: float = 0.1,
        graph_width: typing.Optional[int] = None,
        graph_height: int = 1000,
        font_size: int = 30,
        use_tspan_for_superscripts: bool = False,
        indent: str = "    ",
        line_breaks: bool = True,
        compression_level: enums.ImageFileCompressionLevel = "best",
        file_type: typing.Optional[enums.GraphFileType] = None,
        use_write_suffix: bool = True,
    ):
        path = pathlib.Path(path)
        if use_write_suffix:
            write_path = file_encoder.with_write_suffix(path=path)
        else:
            write_path = None
        path.parent.mkdir(exist_ok=True, parents=True)
        compression_level = enums.validate_image_file_compression_level(
            compression_level
        )
        if file_type is None:
            file_type = enums.graph_file_type_guess(path)
        else:
            file_type = enums.validate_graph_file_type(file_type)
        svg_string = self.to_svg_string(
            color_theme=color_theme,
            hamming_windows_sizes=hamming_windows_sizes,
            x_ticks=x_ticks,
            y_range=y_range,
            y_ticks=y_ticks,
            y_log_scale=y_log_scale,
            y_range_padding_ratio=y_range_padding_ratio,
            graph_width=graph_width,
            graph_height=graph_height,
            font_size=font_size,
            indent=indent,
            line_breaks=line_breaks,
            use_tspan_for_superscripts=(
                use_tspan_for_superscripts if file_type == "svg" else True
            ),
        )
        if file_type == "svg":
            with open(path if write_path is None else write_path, "w") as output:
                output.write(svg_string)
        elif file_type == "png":
            frame = raster.render(svg_string)
            data = image.encode(frame=frame, compression_level=compression_level)
            with open(path if write_path is None else write_path, "wb") as output:
                output.write(data)
        else:
            raise Exception(f"file type {file_type} not implemented")
        if write_path is not None:
            write_path.replace(path)
