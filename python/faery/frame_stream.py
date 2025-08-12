import collections.abc
import dataclasses
import pathlib
import typing

import numpy
import numpy.typing

from . import color as color_module
from . import enums, frame_stream_state, stream, timestamp

if typing.TYPE_CHECKING:
    from .types import image  # type: ignore
else:
    from .extension import image


@dataclasses.dataclass
class Frame:
    """
    A frame with 4 channels per pixels (red, green, blue, alpha), with values in the range [0, 255]
    """

    t: timestamp.Time
    pixels: numpy.typing.NDArray[numpy.uint8]

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        compression_level: enums.ImageFileCompressionLevel = "fast",
        file_type: typing.Optional[enums.ImageFileType] = None,
        use_write_suffix: bool = True,
    ):
        from . import file_encoder

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
            file_type = enums.image_file_type_guess(path)
        else:
            file_type = enums.validate_image_file_type(file_type)
        with open(path if write_path is None else write_path, "wb") as output:
            output.write(
                image.encode(frame=self.pixels, compression_level=compression_level)
            )
        if write_path is not None:
            write_path.replace(path)


OutputState = typing.TypeVar("OutputState")


class FrameOutput(typing.Generic[OutputState]):
    def __iter__(self) -> collections.abc.Iterator[Frame]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        raise NotImplementedError()

    def to_files(
        self,
        path_pattern: typing.Union[pathlib.Path, str],
        compression_level: enums.ImageFileCompressionLevel = "fast",
        file_type: typing.Optional[enums.ImageFileType] = None,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ):
        from . import file_encoder

        file_encoder.frames_to_files(
            stream=self,
            path_pattern=path_pattern,
            compression_level=compression_level,
            file_type=file_type,
            use_write_suffix=True,
            on_progress=on_progress,
        )

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        frame_rate: float = 60.0,
        crf: float = 17.0,
        preset: enums.VideoFilePreset = "medium",
        tune: enums.VideoFileTune = "none",
        profile: enums.VideoFileProfile = "baseline",
        quality: int = 100,
        rewind: bool = False,
        skip: int = 0,
        file_type: typing.Optional[enums.VideoFileType] = None,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ):
        """
        See .file_encoder.frames_to_file
        """

        from . import file_encoder

        try:
            self.time_range()  # type: ignore
            use_write_suffix = True
        except (AttributeError, NotImplementedError):
            use_write_suffix = False
        file_encoder.frames_to_file(
            stream=self,
            path=path,
            dimensions=self.dimensions(),
            frame_rate=frame_rate,
            crf=crf,
            preset=preset,
            tune=tune,
            profile=profile,
            quality=quality,
            rewind=rewind,
            skip=skip,
            file_type=file_type,
            use_write_suffix=use_write_suffix,
            on_progress=on_progress,
        )


class FrameStream(
    stream.Stream[Frame], FrameOutput[frame_stream_state.FrameStreamState]
):
    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ) -> "FrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
    ) -> "FrameStream": ...

    def add_overlay(
        self,
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FrameStream": ...


class FiniteFrameStream(
    stream.FiniteStream[Frame],
    FrameOutput[frame_stream_state.FiniteFrameStreamState],
):
    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FiniteFrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ) -> "FiniteFrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
    ) -> "FiniteFrameStream": ...

    def add_overlay(
        self,
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FiniteFrameStream": ...


class RegularFrameStream(
    stream.RegularStream[Frame],
    FrameOutput[frame_stream_state.RegularFrameStreamState],
):
    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "RegularFrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ) -> "RegularFrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
        output_frame_rate: typing.Optional[float] = 60.0,
    ) -> "RegularFrameStream": ...

    def add_overlay(
        self,
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "RegularFrameStream": ...


class FiniteRegularFrameStream(
    stream.FiniteRegularStream[Frame],
    FrameOutput[frame_stream_state.FiniteRegularFrameStreamState],
):
    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FiniteRegularFrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ) -> "FiniteRegularFrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
        output_frame_rate: typing.Optional[float] = 60.0,
    ) -> "FiniteRegularFrameStream": ...

    def add_overlay(
        self,
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ) -> "FiniteRegularFrameStream": ...


def bind(prefix: typing.Literal["", "Finite", "Regular", "FiniteRegular"]):

    def scale(
        self,
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}Scale"](
            parent=self,
            factor_or_minimum_dimensions=factor_or_minimum_dimensions,
            sampling_filter=sampling_filter,
        )

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}Annotate"](
            parent=self,
            text=text,
            x=x,
            y=y,
            size=size,
            color=color,
        )

    if prefix == "" or prefix == "Finite":

        def add_timecode(
            self,
            x: int = 21,
            y: int = 15,
            size: int = 30,
            color: color_module.Color = "#FFFFFF",
        ):
            from .frame_filter import FILTERS

            return FILTERS[f"{prefix}AddTimecode"](
                parent=self,
                x=x,
                y=y,
                size=size,
                color=color,
            )

        add_timecode.filter_return_annotation = f"{prefix}FrameStream"
        globals()[f"{prefix}FrameStream"].add_timecode = add_timecode
    else:

        def add_timecode_and_speedup(
            self,
            x: int = 21,
            y: int = 15,
            size: int = 30,
            color: color_module.Color = "#FFFFFF",
            output_frame_rate: typing.Optional[float] = 60.0,
        ):
            from .frame_filter import FILTERS

            return FILTERS[f"{prefix}AddTimecodeAndSpeedup"](
                parent=self,
                x=x,
                y=y,
                size=size,
                color=color,
                output_frame_rate=output_frame_rate,
            )

        add_timecode_and_speedup.filter_return_annotation = f"{prefix}FrameStream"
        globals()[f"{prefix}FrameStream"].add_timecode = add_timecode_and_speedup

    def add_overlay(
        self,
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}AddOverlay"](
            parent=self,
            overlay=overlay,
            x=x,
            y=y,
            scale_factor=scale_factor,
            scale_filter=scale_filter,
        )

    def map(
        self,
        function: collections.abc.Callable[
            [numpy.typing.NDArray[numpy.uint8]], numpy.typing.NDArray[numpy.uint8]
        ],
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}Map"](
            parent=self,
            function=function,
        )

    scale.filter_return_annotation = f"{prefix}FrameStream"
    annotate.filter_return_annotation = f"{prefix}FrameStream"

    globals()[f"{prefix}FrameStream"].scale = scale
    globals()[f"{prefix}FrameStream"].annotate = annotate
    globals()[f"{prefix}FrameStream"].add_overlay = add_overlay
    globals()[f"{prefix}FrameStream"].map = map


for prefix in ("", "Finite", "Regular", "FiniteRegular"):
    bind(prefix=prefix)


class List(FiniteRegularFrameStream):
    def __init__(
        self,
        start_t: timestamp.TimeOrTimecode,
        frequency_hz: float,
        frames: list[numpy.typing.NDArray[numpy.uint8]],
    ):
        super().__init__()
        self.start_t = timestamp.parse_time(start_t)
        self.inner_frequency_hz = frequency_hz
        assert len(self.frames) > 0
        assert frames[0].shape[2] == 4
        dimensions = (frames[0].shape[1], frames[0].shape[0])
        for frame in frames[1:]:
            assert frame.shape[1] == dimensions[0]
            assert frame.shape[0] == dimensions[1]
            assert frame.shape[2] == 4
        self.frames = frames

    def __iter__(self) -> collections.abc.Iterator[Frame]:
        for index, frame in enumerate(self.frames):
            yield Frame(
                t=round(
                    self.start_t.microseconds + index * 1e6 / self.inner_frequency_hz
                )
                * timestamp.us,
                pixels=frame.copy(),
            )

    def dimensions(self) -> tuple[int, int]:
        return (self.frames[0].shape[1], self.frames[0].shape[0])

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return (
            self.start_t,
            round(
                self.start_t.microseconds
                + len(self.frames) * 1e6 / self.inner_frequency_hz
            )
            * timestamp.us,
        )

    def frequency_hz(self) -> float:
        return self.inner_frequency_hz


class Function(FiniteRegularFrameStream):
    def __init__(
        self,
        start_t: timestamp.TimeOrTimecode,
        frequency_hz: float,
        dimensions: tuple[int, int],
        frame_count: int,
        get_frame: typing.Callable[[timestamp.Time], numpy.typing.NDArray[numpy.uint8]],
    ):
        self.inner_dimensions = dimensions
        self.start_t = timestamp.parse_time(start_t)
        self.end_t = timestamp.parse_time(start_t)
        self.inner_frequency_hz = frequency_hz
        self.frame_count = frame_count
        self.get_frame = get_frame

    def __iter__(self) -> collections.abc.Iterator[Frame]:
        for index in range(0, self.frame_count):
            # this formula uses "index + 1" to match the convention used by events
            # specifically, for a time range [t0, t1) and a frequency f,
            # rendered event frames have labels in [t0 + 1e6 / f, t1)
            # where the frame with label t is made from events in [t - 1/f, t)
            t = (
                round(
                    self.start_t.microseconds
                    + (index + 1) * 1e6 / self.inner_frequency_hz
                )
                * timestamp.us
            )
            pixels = self.get_frame(t)
            yield Frame(pixels=pixels.copy(), t=t)

    def dimensions(self) -> tuple[int, int]:
        return self.inner_dimensions

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return (
            self.start_t,
            round(
                self.start_t.microseconds
                + self.frame_count * 1e6 / self.inner_frequency_hz
            )
            * timestamp.us,
        )

    def frequency_hz(self) -> float:
        return self.inner_frequency_hz


class FrameFilter(
    FrameStream,
    stream.Filter[Frame],
):
    pass


class FiniteFrameFilter(
    FiniteFrameStream,
    stream.FiniteFilter[Frame],
):
    pass


class RegularFrameFilter(
    RegularFrameStream,
    stream.RegularFilter[Frame],
):
    pass


class FiniteRegularFrameFilter(
    FiniteRegularFrameStream,
    stream.FiniteRegularFilter[Frame],
):
    pass
