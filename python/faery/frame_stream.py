import collections.abc
import dataclasses
import pathlib
import typing

import numpy
import numpy.typing

from . import enums, stream
from .colormaps._base import Color, Colormap

if typing.TYPE_CHECKING:
    from .types import image  # type: ignore
else:
    from .extension import image


@dataclasses.dataclass
class Float64Frame:
    """
    A frame with one channel per pixel, with values in the range [-1.0, 1.0]
    """

    t: int
    pixels: numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass
class Rgba8888Frame:
    """
    A frame with 4 channels per pixels (red, green, blue, alpha), with values in the range [0, 255]
    """

    t: int
    pixels: numpy.typing.NDArray[numpy.uint8]

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        compression_level: enums.ImageFileCompressionLevel = "fast",
        file_type: typing.Optional[enums.ImageFileType] = None,
    ):
        path = pathlib.Path(path)
        compression_level = enums.validate_image_file_compression_level(
            compression_level
        )
        if file_type is None:
            file_type = enums.image_file_type_guess(path)
        else:
            file_type = enums.validate_image_file_type(file_type)
        with open(path, "wb") as output:
            output.write(image.encode(self.pixels, compression_level=compression_level))


class Rgba8888Output:
    def __iter__(self) -> collections.abc.Iterator[Rgba8888Frame]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        raise NotImplementedError()

    def to_files(
        self,
        path_pattern: typing.Union[pathlib.Path, str],
        compression_level: enums.ImageFileCompressionLevel = "fast",
        file_type: typing.Optional[enums.ImageFileType] = None,
    ):
        from . import file_encoder

        file_encoder.frames_to_files(
            stream=self,
            path_pattern=path_pattern,
            compression_level=compression_level,
            file_type=file_type,
        )

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        frequency_hz: float = 60,
        crf: float = 15.0,
        preset: enums.VideoFilePreset = "ultrafast",
        tune: enums.VideoFileTune = "none",
        profile: enums.VideoFileProfile = "baseline",
        file_type: typing.Optional[enums.VideoFileType] = None,
    ):
        from . import file_encoder

        file_encoder.frames_to_file(
            stream=self,
            path=path,
            dimensions=self.dimensions(),
            frequency_hz=frequency_hz,
            crf=crf,
            preset=preset,
            tune=tune,
            profile=profile,
            file_type=file_type,
        )


class Float64FrameStream(stream.Stream[Float64Frame]):
    def colorize(self, colormap: Colormap) -> "Rgba8888FrameStream": ...


class FiniteFloat64FrameStream(stream.FiniteStream[Float64Frame]):
    def colorize(self, colormap: Colormap) -> "FiniteRgba8888FrameStream": ...


class RegularFloat64FrameStream(stream.RegularStream[Float64Frame]):
    def colorize(self, colormap: Colormap) -> "RegularRgba8888FrameStream": ...


class FiniteRegularFloat64FrameStream(stream.FiniteRegularStream[Float64Frame]):
    def colorize(self, colormap: Colormap) -> "FiniteRegularRgba8888FrameStream": ...


class Rgba8888FrameStream(stream.Stream[Rgba8888Frame], Rgba8888Output):
    def scale(
        self,
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ) -> "Rgba8888FrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ) -> "Rgba8888FrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: Color = "#FFFFFF",
    ) -> "Rgba8888FrameStream": ...


class FiniteRgba8888FrameStream(stream.FiniteStream[Rgba8888Frame], Rgba8888Output):
    def scale(
        self,
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ) -> "FiniteRgba8888FrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ) -> "FiniteRgba8888FrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: Color = "#FFFFFF",
    ) -> "FiniteRgba8888FrameStream": ...


class RegularRgba8888FrameStream(stream.RegularStream[Rgba8888Frame], Rgba8888Output):
    def scale(
        self,
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ) -> "RegularRgba8888FrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ) -> "RegularRgba8888FrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: Color = "#FFFFFF",
        speed_up_label_output_frequency_hz: typing.Optional[int] = 60,
    ) -> "RegularRgba8888FrameStream": ...


class FiniteRegularRgba8888FrameStream(
    stream.FiniteRegularStream[Rgba8888Frame], Rgba8888Output
):
    def scale(
        self,
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ) -> "FiniteRegularRgba8888FrameStream": ...

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ) -> "FiniteRegularRgba8888FrameStream": ...

    def add_timecode(
        self,
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: Color = "#FFFFFF",
        speed_up_label_output_frequency_hz: typing.Optional[int] = 60,
    ) -> "FiniteRegularRgba8888FrameStream": ...


def bind_float64(prefix: typing.Literal["", "Finite", "Regular", "FiniteRegular"]):

    def colorize(
        self,
        colormap: Colormap,
    ):
        from .render import FILTERS

        return FILTERS[f"{prefix}Rgba8888Colorize"](
            parent=self,
            colormap=colormap,
        )

    colorize.filter_return_annotation = f"{prefix}Rgba8888FrameStream"

    globals()[f"{prefix}Float64FrameStream"].colorize = colorize


for prefix in ("", "Finite", "Regular", "FiniteRegular"):
    bind_float64(prefix=prefix)


def bind_rgb8888(prefix: typing.Literal["", "Finite", "Regular", "FiniteRegular"]):

    def scale(
        self,
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}Rgba8888Scale"](
            parent=self,
            factor=factor,
            filter=filter,
        )

    def annotate(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ):
        from .frame_filter import FILTERS

        return FILTERS[f"{prefix}Rgba8888Annotate"](
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
            color: Color = "#FFFFFF",
        ):
            from .frame_filter import FILTERS

            return FILTERS[f"{prefix}Rgba8888AddTimecode"](
                parent=self,
                x=x,
                y=y,
                size=size,
                color=color,
            )

        add_timecode.filter_return_annotation = f"{prefix}Rgba8888FrameStream"
        globals()[f"{prefix}Rgba8888FrameStream"].add_timecode = add_timecode
    else:

        def add_timecode_and_speedup(
            self,
            x: int = 21,
            y: int = 15,
            size: int = 30,
            color: Color = "#FFFFFF",
            speed_up_label_output_frequency_hz: typing.Optional[int] = 60,
        ):
            from .frame_filter import FILTERS

            return FILTERS[f"{prefix}Rgba8888AddTimecodeAndSpeedup"](
                parent=self,
                x=x,
                y=y,
                size=size,
                color=color,
                speed_up_label_output_frequency_hz=speed_up_label_output_frequency_hz,
            )

        add_timecode_and_speedup.filter_return_annotation = (
            f"{prefix}Rgba8888FrameStream"
        )
        globals()[
            f"{prefix}Rgba8888FrameStream"
        ].add_timecode = add_timecode_and_speedup

    scale.filter_return_annotation = f"{prefix}Rgba8888FrameStream"
    annotate.filter_return_annotation = f"{prefix}Rgba8888FrameStream"

    globals()[f"{prefix}Rgba8888FrameStream"].scale = scale
    globals()[f"{prefix}Rgba8888FrameStream"].annotate = annotate


for prefix in ("", "Finite", "Regular", "FiniteRegular"):
    bind_rgb8888(prefix=prefix)


class Float64FrameFilter(
    Float64FrameStream,
    stream.Filter[Float64Frame],
):
    pass


class FiniteFloat64FrameFilter(
    FiniteFloat64FrameStream,
    stream.FiniteFilter[Float64Frame],
):
    pass


class RegularFloat64FrameFilter(
    RegularFloat64FrameStream,
    stream.RegularFilter[Float64Frame],
):
    pass


class FiniteRegularFloat64FrameFilter(
    FiniteRegularFloat64FrameStream,
    stream.FiniteRegularFilter[Float64Frame],
):
    pass


class Rgba8888FrameFilter(
    Rgba8888FrameStream,
    stream.Filter[Rgba8888Frame],
):
    pass


class FiniteRgba8888FrameFilter(
    FiniteRgba8888FrameStream,
    stream.FiniteFilter[Rgba8888Frame],
):
    pass


class RegularRgba8888FrameFilter(
    RegularRgba8888FrameStream,
    stream.RegularFilter[Rgba8888Frame],
):
    pass


class FiniteRegularRgba8888FrameFilter(
    FiniteRegularRgba8888FrameStream,
    stream.FiniteRegularFilter[Rgba8888Frame],
):
    pass
