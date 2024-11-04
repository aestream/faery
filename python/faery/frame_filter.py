import collections.abc
import pathlib
import types
import typing

import numpy
import numpy.typing

from . import enums, frame_stream, stream, timestamp
from .colormaps._base import Color, parse_color

if typing.TYPE_CHECKING:
    from .types import image  # type: ignore
else:
    from .extension import image


SPEED_UP_PRECISION: float = 1e-3


def number_to_string(number: typing.Union[int, float], precision: float) -> str:
    """
    Converts a number larger than one to a string with float or integer representation depending on the value.
    """
    assert number >= 1.0
    absolute_value = abs(number)
    for ndigits in range(0, int(-numpy.log10(precision)) + 1):
        rounded_value = round(absolute_value, ndigits=ndigits)
        if number < rounded_value * (1.0 + precision) and number > rounded_value * (
            1.0 - precision
        ):
            return f"{number:.{ndigits}f}"
        ndigits += 1
    return str(number)


def restrict(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_filter(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(filter_class):
        attributes = [
            name
            for name, item in filter_class.__dict__.items()
            if isinstance(item, types.FunctionType)
        ]
        for prefix in prefixes:

            class Generated(getattr(frame_stream, f"{prefix}FrameFilter")):
                pass

            for attribute in attributes:
                method = getattr(filter_class, attribute)
                if not hasattr(method, "_prefixes") or prefix in method._prefixes:
                    setattr(Generated, attribute, getattr(filter_class, attribute))
            Generated.__name__ = f"{prefix}{filter_class.__name__}"
            Generated.__qualname__ = Generated.__name__
            FILTERS[Generated.__name__] = Generated
        return None

    return decorator


@typed_filter(
    {
        "Rgba8888",
        "FiniteRgba8888",
        "RegularRgba8888",
        "FiniteRegularRgba8888",
    }
)
class Scale(frame_stream.FiniteRegularRgba8888FrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Rgba8888Frame],
        factor: float,
        filter: enums.ImageResizeFilter = "nearest",
    ):
        self.init(parent=parent)
        self.factor = factor
        self.filter: enums.ImageResizeFilter = enums.validate_image_resize_filter(
            filter
        )

    def dimensions(self) -> tuple[int, int]:
        parent_dimensions = self.parent.dimensions()
        return (
            int(round(parent_dimensions[0] * self.factor)),
            int(round(parent_dimensions[1] * self.factor)),
        )

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        new_dimensions = self.dimensions()
        for frame in self.parent:
            frame.pixels = image.resize(
                frame=frame.pixels,
                new_dimensions=new_dimensions,
                filter=self.filter,
            )
            yield frame


@typed_filter(
    {"Rgba8888", "FiniteRgba8888", "RegularRgba8888", "FiniteRegularRgba8888"}
)
class Annotate(frame_stream.FiniteRegularRgba8888FrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Rgba8888Frame],
        text: str,
        x: int,
        y: int,
        size: int,
        color: Color,
    ):
        self.init(parent=parent)
        self.text = text
        self.x = x
        self.y = y
        self.size = size
        color = parse_color(color)
        self.color = (
            int(round(color[0] * 255.0)),
            int(round(color[1] * 255.0)),
            int(round(color[2] * 255.0)),
            int(round(color[3] * 255.0)),
        )

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        for frame in self.parent:
            image.annotate(
                frame=frame.pixels,
                text=self.text,
                x=self.x,
                y=self.y,
                size=self.size,
                color=self.color,
            )
            yield frame


@typed_filter({"Rgba8888", "FiniteRgba8888"})
class AddTimecode(frame_stream.RegularRgba8888FrameFilter):
    def __init__(
        self,
        parent: stream.RegularStream[frame_stream.Rgba8888Frame],
        x: int = 20,
        y: int = 20,
        size: int = 30,
        color: Color = "#FFFFFF",
    ):
        self.init(parent=parent)
        self.x = x
        self.y = y
        self.size = size
        color = parse_color(color)
        self.color = (
            int(round(color[0] * 255.0)),
            int(round(color[1] * 255.0)),
            int(round(color[2] * 255.0)),
            int(round(color[3] * 255.0)),
        )

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        for frame in self.parent:
            image.annotate(
                frame=frame.pixels,
                text=timestamp.timestamp_to_timecode(frame.t),
                x=self.x,
                y=self.y,
                size=self.size,
                color=self.color,
            )
            yield frame


@typed_filter({"RegularRgba8888", "FiniteRegularRgba8888"})
class AddTimecodeAndSpeedup(frame_stream.FiniteRegularRgba8888FrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Rgba8888Frame],
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: Color = "#FFFFFF",
        output_frame_rate: typing.Optional[float] = 60.0,
    ):
        self.init(parent=parent)
        self.x = x
        self.y = y
        self.size = size
        color = parse_color(color)
        self.color = (
            int(round(color[0] * 255.0)),
            int(round(color[1] * 255.0)),
            int(round(color[2] * 255.0)),
            int(round(color[3] * 255.0)),
        )
        self.output_frame_rate = output_frame_rate

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        if self.output_frame_rate is not None:
            frequency_hz = self.frequency_hz()
            speed_up = self.output_frame_rate / frequency_hz
            if (
                speed_up < 1.0 + SPEED_UP_PRECISION
                and speed_up > 1.0 - SPEED_UP_PRECISION
            ):
                speedup_label = "Real-time"
            elif speed_up < 1.0:
                inverse_speedup = 1.0 / speed_up
                speedup_label = (
                    f"× 1/{number_to_string(inverse_speedup, SPEED_UP_PRECISION)}"
                )
            else:
                speedup_label = f"× {number_to_string(speed_up, SPEED_UP_PRECISION)}"
        else:
            speedup_label = None
        for frame in self.parent:
            image.annotate(
                frame=frame.pixels,
                text=timestamp.timestamp_to_timecode(frame.t),
                x=self.x,
                y=self.y,
                size=self.size,
                color=self.color,
            )
            if speedup_label is not None:
                image.annotate(
                    frame=frame.pixels,
                    text=speedup_label,
                    x=self.x,
                    y=self.y + round(self.size * 1.2),
                    size=self.size,
                    color=self.color,
                )
            yield frame


@typed_filter(
    {"Rgba8888", "FiniteRgba8888", "RegularRgba8888", "FiniteRegularRgba8888"}
)
class AddOverlay(frame_stream.FiniteRegularRgba8888FrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Rgba8888Frame],
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeFilter = "nearest",
    ):
        self.init(parent=parent)
        self.overlay = overlay
        self.x = x
        self.y = y
        self.scale_factor = scale_factor
        self.scale_filter: enums.ImageResizeFilter = scale_filter

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        if isinstance(self.overlay, (pathlib.Path, str)):
            with open(self.overlay, "rb") as overlay_file:
                overlay = image.decode(overlay_file.read())
        else:
            overlay = self.overlay
        for frame in self.parent:
            image.overlay(
                frame=frame.pixels,
                overlay=overlay,
                x=self.x,
                y=self.y,
                new_dimensions=(
                    int(round(overlay.shape[0] * self.scale_factor)),
                    int(round(overlay.shape[1] * self.scale_factor)),
                ),
                filter=self.scale_filter,
            )
            yield frame
