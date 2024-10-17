import typing

import numpy.typing

def encode(
    frame: numpy.typing.NDArray[numpy.uint8],
    compression_level: typing.Literal["default", "fast", "best"],
) -> bytes: ...
def annotate(
    frame: numpy.typing.NDArray[numpy.uint8],
    text: str,
    x_offset: int,
    y_offset: int,
    scale: int,
    color: tuple[int, int, int, int],
) -> None: ...
