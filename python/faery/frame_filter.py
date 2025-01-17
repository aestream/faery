import collections.abc
import math
import pathlib
import types
import typing

import numpy
import numpy.typing

from . import color as color_module
from . import enums, frame_stream, stream, timestamp

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


def restrict(prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]]):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_filter(
    prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]]
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


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Scale(frame_stream.FiniteRegularFrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Frame],
        factor_or_minimum_dimensions: typing.Union[float, tuple[int, int]] = (960, 720),
        sampling_filter: enums.ImageResizeSamplingFilter = "nearest",
    ):
        self.init(parent=parent)
        self.factor_or_minimum_dimensions = factor_or_minimum_dimensions
        self.sampling_filter: enums.ImageResizeSamplingFilter = (
            enums.validate_image_resize_samplin_filter(sampling_filter)
        )

    def dimensions(self) -> tuple[int, int]:
        parent_dimensions = self.parent.dimensions()
        if isinstance(self.factor_or_minimum_dimensions, (float, int)):
            factor = self.factor_or_minimum_dimensions
        else:
            factor = max(
                1.0,
                math.ceil(self.factor_or_minimum_dimensions[0] / parent_dimensions[0]),
                math.ceil(self.factor_or_minimum_dimensions[1] / parent_dimensions[1]),
            )
        return (
            int(round(parent_dimensions[0] * factor)),
            int(round(parent_dimensions[1] * factor)),
        )

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
        parent_dimensions = self.parent.dimensions()
        if isinstance(self.factor_or_minimum_dimensions, (float, int)):
            factor = self.factor_or_minimum_dimensions
        else:
            factor = max(
                1.0,
                math.ceil(960 / parent_dimensions[0]),
                math.ceil(720 / parent_dimensions[1]),
            )
        if factor == 1.0:
            for frame in self.parent:
                yield frame
        else:
            new_dimensions = (
                int(round(parent_dimensions[0] * factor)),
                int(round(parent_dimensions[1] * factor)),
            )
            for frame in self.parent:
                frame.pixels = image.resize(
                    frame=frame.pixels,
                    new_dimensions=new_dimensions,
                    sampling_filter=self.sampling_filter,
                )
                yield frame


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Annotate(frame_stream.FiniteRegularFrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Frame],
        text: str,
        x: int,
        y: int,
        size: int,
        color: color_module.Color,
    ):
        self.init(parent=parent)
        self.text = text
        self.x = x
        self.y = y
        self.size = size
        self.color = color_module.color_to_ints(color)

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
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


@typed_filter({"", "Finite"})
class AddTimecode(frame_stream.RegularFrameFilter):
    def __init__(
        self,
        parent: stream.RegularStream[frame_stream.Frame],
        x: int = 20,
        y: int = 20,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
    ):
        self.init(parent=parent)
        self.x = x
        self.y = y
        self.size = size
        self.color = color_module.color_to_ints(color)

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
        for frame in self.parent:
            image.annotate(
                frame=frame.pixels,
                text=frame.t.to_timecode(),
                x=self.x,
                y=self.y,
                size=self.size,
                color=self.color,
            )
            yield frame


@typed_filter({"Regular", "FiniteRegular"})
class AddTimecodeAndSpeedup(frame_stream.FiniteRegularFrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Frame],
        x: int = 21,
        y: int = 15,
        size: int = 30,
        color: color_module.Color = "#FFFFFF",
        output_frame_rate: typing.Optional[float] = 60.0,
    ):
        self.init(parent=parent)
        self.x = x
        self.y = y
        self.size = size
        self.color = color_module.color_to_ints(color)
        self.output_frame_rate = output_frame_rate

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
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
                text=frame.t.to_timecode(),
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


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class AddOverlay(frame_stream.FiniteRegularFrameFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Frame],
        overlay: typing.Union[pathlib.Path, str, numpy.typing.NDArray[numpy.uint8]],
        x: int = 0,
        y: int = 0,
        scale_factor: float = 1.0,
        scale_filter: enums.ImageResizeSamplingFilter = "nearest",
    ):
        self.init(parent=parent)
        self.overlay = overlay
        self.x = x
        self.y = y
        self.scale_factor = scale_factor
        self.scale_sampling_filter: enums.ImageResizeSamplingFilter = scale_filter

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
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
                    int(round(overlay.shape[1] * self.scale_factor)),
                    int(round(overlay.shape[0] * self.scale_factor)),
                ),
                sampling_filter=self.scale_sampling_filter,
            )
            yield frame
