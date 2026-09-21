import pathlib
import types
import typing

import numpy

LZ4_FASTEST: tuple[typing.Literal["lz4"], typing.Literal[1]]
LZ4_DEFAULT: tuple[typing.Literal["lz4"], typing.Literal[1]]
LZ4_HIGHEST: tuple[typing.Literal["lz4"], typing.Literal[12]]
ZSTD_FASTEST: tuple[typing.Literal["zstd"], typing.Literal[1]]
ZSTD_DEFAULT: tuple[typing.Literal["zstd"], typing.Literal[3]]
ZSTD_HIGHEST: tuple[typing.Literal["zstd"], typing.Literal[22]]

class Track:
    id: int
    data_type: typing.Literal["events", "frame", "imus", "triggers"]
    dimensions: tuple[int, int] | None

    @typing.overload
    def __init__(
        self,
        id: int,
        data_type: typing.Literal["events"],
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self,
        id: int,
        data_type: typing.Literal["frame"],
        dimensions: tuple[int, int],
    ): ...
    @typing.overload
    def __init__(
        self, id: int, data_type: typing.Literal["imus"], dimensions: None
    ): ...
    @typing.overload
    def __init__(
        self, id: int, data_type: typing.Literal["triggers"], dimensions: None
    ): ...

class Frame:
    t: int
    start_t: int
    end_t: int
    exposure_start_t: int
    exposure_end_t: int
    format: typing.Literal["L", "I;16", "RGB", "RGBA"]
    offset_x: int
    offset_y: int
    pixels: numpy.ndarray

class FileDataDefinition:
    byte_offset: int
    track_id: int
    size: int
    elements_count: int
    start_t: int
    end_t: int

class DescriptionAttribute:
    attribute_type: typing.Literal["string", "int", "long"]
    value: str | int

    def __init__(
        self,
        attribute_type: typing.Literal["string", "int", "long"],
        value: str | int,
    ): ...

class DescriptionNode:
    name: str
    path: str | None
    attributes: dict[str, DescriptionAttribute]
    nodes: list[DescriptionNode]

    def __init__(
        self,
        name: str,
        path: str | None,
        attributes: dict[str, DescriptionAttribute],
        nodes: list[DescriptionNode],
    ): ...

class Decoder:
    def __init__(
        self,
        path: pathlib.Path | str,
    ): ...
    def __enter__(self) -> Decoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def __iter__(self) -> Decoder: ...
    def __next__(
        self,
    ) -> tuple[
        Track,
        numpy.ndarray | Frame,
    ]: ...
    def tracks(self) -> list[Track]: ...
    def description(self) -> list[DescriptionNode]: ...
    def file_data_definitions(self) -> list[FileDataDefinition]: ...

class Encoder:
    def __init__(
        self,
        path: pathlib.Path | str,
        description: list[DescriptionNode],
        compression: tuple[typing.Literal["lz4", "zstd"], int] | None,
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool: ...
    def write(self, track_id: int, packet: numpy.ndarray | Frame): ...
