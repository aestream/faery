import pathlib
import types
import typing

import numpy

class Decoder:
    version: typing.Literal["dat1", "dat2"]
    event_type: typing.Literal["2d", "cd", "trigger"]
    dimensions: tuple[int, int] | None

    def __init__(
        self,
        path: pathlib.Path | str,
        dimensions_fallback: tuple[int, int] | None,
        version_fallback: typing.Literal["dat1", "dat2"] | None,
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
    @typing.overload
    def __init__(
        self,
        path: pathlib.Path | str,
        version: typing.Literal["dat1", "dat2"],
        event_type: typing.Literal["2d"],
        zero_t0: bool,
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self,
        path: pathlib.Path | str,
        version: typing.Literal["dat1", "dat2"],
        event_type: typing.Literal["cd"],
        zero_t0: bool,
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self,
        path: pathlib.Path | str,
        version: typing.Literal["dat1", "dat2"],
        event_type: typing.Literal["trigger"],
        zero_t0: bool,
        dimensions: None,
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def t0(self) -> int | None: ...
    def write(self, packet: numpy.ndarray): ...
