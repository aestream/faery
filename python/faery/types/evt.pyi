import pathlib
import types
import typing

import numpy

class Decoder:
    version: typing.Literal["evt2", "evt2.1", "evt3"]
    dimensions: tuple[int, int]

    def __init__(
        self,
        path: pathlib.Path | str,
        dimensions_fallback: tuple[int, int] | None,
        version_fallback: typing.Literal["evt2", "evt2.1", "evt3"] | None,
    ): ...
    def __enter__(self) -> Decoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def __iter__(self) -> Decoder: ...
    def __next__(self) -> dict[typing.Literal["events", "triggers"], numpy.ndarray]: ...

class Encoder:
    def __init__(
        self,
        path: pathlib.Path | str,
        version: typing.Literal["evt2", "evt2.1", "evt3"],
        zero_t0: bool,
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
    def t0(self) -> int | None: ...
    def write(
        self, packet: dict[typing.Literal["events", "triggers"], numpy.ndarray]
    ): ...
