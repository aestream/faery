import pathlib
import types
import typing

import numpy

class Decoder:
    version: str
    event_type: typing.Literal["generic", "dvs", "atis", "color"]
    dimensions: typing.Optional[tuple[int, int]]

    def __init__(self, path: typing.Union[pathlib.Path, str], t0: int): ...
    def __enter__(self) -> Decoder: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def __iter__(self) -> Decoder: ...
    def __next__(self) -> numpy.ndarray: ...

class Encoder:
    @typing.overload
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        event_type: typing.Literal["generic"],
        zero_t0: bool,
        dimensions: None,
    ): ...
    @typing.overload
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        event_type: typing.Literal["dvs"],
        zero_t0: bool,
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        event_type: typing.Literal["atis"],
        zero_t0: bool,
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        event_type: typing.Literal["color"],
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
    def write(self, packet: numpy.ndarray): ...
