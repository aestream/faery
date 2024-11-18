import typing

import numpy.typing

def decode(
    bytes: bytes,
) -> numpy.typing.NDArray[numpy.uint8]: ...
def encode(
    frame: numpy.typing.NDArray[numpy.uint8],
    compression_level: typing.Literal["default", "fast", "best"],
) -> bytes: ...
def annotate(
    frame: numpy.typing.NDArray[numpy.uint8],
    text: str,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int, int],
) -> None: ...
@typing.overload
def resize(
    frame: numpy.typing.NDArray[numpy.uint8],
    new_dimensions: tuple[int, int],
    sampling_filter: typing.Literal[
        "nearest", "triangle", "catmull_rom", "gaussian", "lanczos3"
    ],
) -> numpy.typing.NDArray[numpy.uint8]: ...
@typing.overload
def resize(
    frame: numpy.typing.NDArray[numpy.float64],
    new_dimensions: tuple[int, int],
    sampling_filter: typing.Literal[
        "nearest", "triangle", "catmull_rom", "gaussian", "lanczos3"
    ],
) -> numpy.typing.NDArray[numpy.float64]: ...
def overlay(
    frame: numpy.typing.NDArray[numpy.uint8],
    overlay: numpy.typing.NDArray[numpy.uint8],
    x: int,
    y: int,
    new_dimensions: tuple[int, int],
    sampling_filter: typing.Literal[
        "nearest", "triangle", "catmull_rom", "gaussian", "lanczos3"
    ],
) -> None: ...
