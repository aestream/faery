import pathlib
import types

import numpy

class Decoder:
    dimensions: tuple[int, int]

    def __init__(
        self,
        path: pathlib.Path | str | None,
        dimensions: tuple[int, int],
        has_header: bool,
        separator: int,
        t_index: int,
        x_index: int,
        y_index: int,
        on_index: int,
        t_scale: float,
        t0: int,
        on_value: bytes,
        off_value: bytes,
        skip_errors: bool,
    ): ...
    def __enter__(self) -> Decoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def __iter__(self) -> Decoder: ...
    def __next__(self) -> numpy.ndarray: ...

class Encoder:
    def __init__(
        self,
        path: pathlib.Path | str | None,
        separator: int,
        header: bool,
        dimensions: tuple[int, int],
        enforce_monotonic: bool,
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def write(self, events: numpy.ndarray): ...
