import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing

from . import stream


@dataclasses.dataclass(frozen=True)
class Float64Frame:
    """
    A frame with one channel per pixel, with values in the range [-1.0, 1.0]
    """

    index: int
    timestamp: int
    pixels: numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True)
class Rgba8888Frame:
    """
    A frame with 4 channels per pixels, with values in the range [0, 255]
    """

    index: int
    timestamp: int
    pixels: numpy.typing.NDArray[numpy.uint8]


@dataclasses.dataclass(frozen=True)
class Rgb888Frame:
    """
    A frame with 3 channels per pixels, with values in the range [0, 255]
    """

    index: int
    timestamp: int
    pixels: numpy.typing.NDArray[numpy.uint8]
