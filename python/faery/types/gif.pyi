import pathlib
import types

import numpy.typing

class Encoder:
    def __init__(
        self,
        path: pathlib.Path | str,
        dimensions: tuple[int, int],
        frame_rate: float,
        quality: int,
        fast: bool,
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def write(
        self,
        frame: numpy.typing.NDArray[numpy.uint8],
    ): ...
