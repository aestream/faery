import pathlib
import types
import typing

import numpy.typing

class Encoder:
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        dimensions: tuple[int, int],
        frame_rate: float,
        quality: int,
        fast: bool,
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def write(
        self,
        frame: numpy.typing.NDArray[numpy.uint8],
    ): ...
