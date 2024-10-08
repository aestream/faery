import pathlib
import typing
import types

import numpy

class Decoder:
    version: typing.Literal["evt2", "evt2.1", "evt3"]
    dimensions: tuple[int, int]

    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        dimensions_fallback: typing.Optional[tuple[int, int]] = None,
        version_fallback: typing.Optional[
            typing.Literal["evt2", "evt2.1", "evt3"]
        ] = None,
    ): ...
    def __enter__(self) -> Decoder: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def __iter__(self) -> Decoder: ...
    def __next__(self) -> dict[typing.Literal["events", "triggers"], numpy.ndarray]: ...

class Encoder:
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        version: typing.Literal["evt2", "evt2.1", "evt3"],
        zero_t0: bool,
        dimensions: tuple[int, int],
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def t0(self) -> typing.Optional[int]: ...
    def write(
        self, packet: dict[typing.Literal["events", "triggers"], numpy.ndarray]
    ): ...
