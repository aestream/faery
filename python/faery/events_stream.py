import collections.abc
import dataclasses
import pathlib
import typing

import numpy

from . import enums, events_stream_state, frame_stream, stream, timestamp

if typing.TYPE_CHECKING:
    from .types import aedat  # type: ignore
else:
    from .extension import aedat

EVENTS_DTYPE: numpy.dtype = numpy.dtype(
    [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), (("p", "on"), "?")]
)

# A type puzzle
# =============
#
# We support four types of streams:
# - default (possibly infinite stream with packets of arbitrary duration)
# - finite (finite stream with packets of arbitrary duration)
# - regular (possibly infinite stream with packets of fixed duration)
# - finite regular (finite stream with packets of fixed duration)
#
# Most functions (for instance `crop`) are available to all stream types
# and have the same implementation for all stream types.
#
# Some functions (for instance `to_array`, which collects the stream into a single array)
# are only available to specific stream types (finite streams in the case of `to_array`).
#
# Some functions (for instance `regularize`) behave differently depending on the stream type.
#
# *All* functions must properly transmit (or transform) the stream type so that they
# can be chained.
# For instance, `crop` on a regular stream must return a regular stream,
# but `crop` on a finite stream must return a finite stream.
# `regularize` on a finite stream must return a finite regular stream,
# but `regularize` on a default stream must return a regular stream.
#
# (a) We want the *static* type system to accurately represent stream types
# so that IDEs can suggest the right functions whilst writing pipelines.
#
# (b) Python's *dynamic* runtime needs to know the stream type
# to use the right implementation when several are available and raise
# appropriate errors when a function is called on the wrong stream type.
#
#
# Current implementation
# ----------------------
#
# The present file contains four classes to represent the four stream types,
# an explicit list of the filters available for each stream type,
# and the type that these filters return.
#
# The filters' implementation is in *events_filter.py*. The decorator defined in that file
# generates four classes for each filter so that objects can have the right runtime type.
# Implementations are dynamically bound to the methods defined here (see `bind`) to minimize boilerplate.
#
# The current implementation is verbose (methods are declared four times) but it solves (a) and (b). The
# biggest drawback is that documentation needs to be duplicated four times. This is not a major problem
# for custom filters (written by users, see `apply`) since they would typically target a specific stream type.
#
# Ideas to improve the current design are welcome.
#
#
# Considered solutions
# --------------------
#
# Since most functions are identical, it is tempting to use Python's typing.Generic to reduce
# code duplication. However, since typing is optional in Python, it is not possible to retrieve the actual type at
# runtime when using typing.Generic. This is an issue for (b).
#
# Dynamic type inference (for instance using virtual methods) with a single class would solve (b) but would not allow for (a).
#
# @typing.overload with a single class does not work here because we are encoding the stream type in the class,
# not the function parameters.
#
# Templates would be a viable solution (another program would generate the Python source code).
# However, this would require that library contributors write non-quite-Python code and pre-process
# the code before testing it.

OutputState = typing.TypeVar("OutputState")


class Output(typing.Generic[OutputState]):
    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        raise NotImplementedError()

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        version: typing.Optional[enums.EventsFileVersion] = None,
        zero_t0: bool = True,
        compression: typing.Optional[
            typing.Tuple[enums.EventsFileCompression, int]
        ] = aedat.LZ4_DEFAULT,
        file_type: typing.Optional[enums.EventsFileType] = None,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> str:
        """
        Writes the stream to an event file (supports .aedat4, .es, .raw, and .dat).

        version is only used if the file type is EVT (.raw) or DAT.

        zero_t0 is only used if the file type is ES, EVT (.raw) or DAT.
        The original t0 is stored in the header of EVT and DAT files, and is discarded if the file type is ES.

        compression is only used if the file type is AEDAT.

        Args:
            stream: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            path: Path of the output event file.
            dimensions: Width and height of the sensor.
            version: Version for EVT (.raw) and DAT files. Defaults to "dat2" for DAT and "evt3" for EVT.
            zero_t0: Whether to normalize timestamps and write the offset in the header for EVT (.raw) and DAT files. Defaults to True.
            compression: Compression for aedat files. Defaults to ("lz4", 1).
            file_type: Override the type determination algorithm. Defaults to None.

        Returns:
            The original t0 as a timecode if the file type is ES, EVT (.raw) or DAT, and if `zero_t0` is true. 0 as a timecode otherwise.
            To reconstruct the original timestamps when decoding ES files with Faery, pass the returned value to `faery.stream_from_file`.
            EVT (.raw) and DAT files do not need this (t0 is written in their header), but it is returned here anyway for compatibility
            with software than do not support the t0 header field.
        """
        from . import file_encoder

        return file_encoder.events_to_file(
            stream=self,
            path=path,
            dimensions=self.dimensions(),
            version=version,
            zero_t0=zero_t0,
            compression=compression,
            file_type=file_type,
            on_progress=on_progress,  # type: ignore
        )

    def to_stdout(
        self,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> str:
        from . import file_encoder

        return file_encoder.events_to_file(
            stream=self,
            path=None,
            dimensions=self.dimensions(),
            file_type="csv",
            on_progress=on_progress,  # type: ignore
        )

    def to_udp(
        self,
        address: typing.Union[
            tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
        ],
        payload_length: typing.Optional[int] = None,
        format: typing.Literal[
            "t64_x16_y16_on8", "t32_x16_y15_on1"
        ] = "t64_x16_y16_on8",
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> None:
        """
        Sends the stream to the given UDP address and port.

        The address format defines whether IPv4 or IPv6 is used. If it has two items (host, port),
        IPv4 is used. It it has four items (host, port, flowinfo, scope_id), IPv6 is used.
        To force IPv6 but ommit scope_id and/or flowinfo, set them to None.
        See https://docs.python.org/3/library/socket.html for details.

        To maximize throughput, consider re-arranging the stream in packets of exactly `payload_length / format_size` events
        with `.chunks(payload_length / format_size)`. By default:
        - for format "t64_x16_y16_on8", payload_length is 1209 and payload size is 13.
        - for format "t32_x16_y15_on1", payload_length is 1208 and payload size is 8.

        Args:
            address: the UDP address as (host, port) for IPv4 and (host, port, flowinfo, scope_id) for IPv6.
            payload_length: maximum payload size in bytes. It must be a multiple of the event size
            (13 for format="t64_x16_y16_on8" and 8 for format="t64_x16_y16_on8"). Defaults to 1209 if format is "t64_x16_y16_on8"
            and 1208 if format is "t32_x16_y15_on1".
            format: Event encoding format. Defaults to "t64_x16_y16_on8".
        """
        from . import udp_encoder

        return udp_encoder.encode(
            stream=self,
            address=address,
            payload_length=payload_length,
            format=format,
        )


class EventsStream(
    stream.Stream[numpy.ndarray], Output[events_stream_state.EventsStreamState]
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ) -> "RegularEventsStream":
        """
        Converts the stream to a regular stream with the given frequency (or packet rate).

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            frequency: Number of packets per second.
            start: Optional starting time of the first packet. If None (default), the timestamp of the first event is used.
        """
        ...

    def chunks(self, chunk_length: int) -> "EventsStream": ...

    def time_slice(
        self,
        start: timestamp.Time,
        end: timestamp.Time,
        zero: bool = False,
    ) -> "FiniteEventsStream": ...

    def event_slice(self, start: int, end: int) -> "FiniteEventsStream": ...

    def remove_on_events(self) -> "EventsStream": ...

    def remove_off_events(self) -> "EventsStream": ...

    def crop(self, left: int, right: int, top: int, bottom: int) -> "EventsStream": ...

    def mask(self, array: numpy.ndarray) -> "EventsStream": ...

    def transpose(self, action: enums.TransposeAction) -> "EventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "EventsStream": ...

    def apply(
        self, filter_class: type["EventsFilter"], *args, **kwargs
    ) -> "EventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def envelope(
        self,
        decay: enums.Decay,
        tau: timestamp.Time,
    ) -> frame_stream.Float64FrameStream: ...


class FiniteEventsStream(
    stream.FiniteStream[numpy.ndarray],
    Output[events_stream_state.FiniteEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ) -> "FiniteRegularEventsStream":
        """
        Converts the stream to a regular stream with the given frequency (or packet rate).

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            frequency: Number of packets per second.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range_us()[0]`) is used.
        """
        ...

    def chunks(self, chunk_length: int) -> "FiniteEventsStream":
        from .events_filter import Chunks

        return Chunks(
            parent=self,  # type: ignore (see "Note on filter types" in events_filter)
            chunk_length=chunk_length,
        )

    def time_slice(
        self,
        start: timestamp.Time,
        end: timestamp.Time,
        zero: bool = False,
    ) -> "FiniteEventsStream": ...

    def event_slice(self, start: int, end: int) -> "FiniteEventsStream": ...

    def remove_on_events(self) -> "FiniteEventsStream": ...

    def remove_off_events(self) -> "FiniteEventsStream": ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> "FiniteEventsStream": ...

    def mask(self, array: numpy.ndarray) -> "FiniteEventsStream": ...

    def transpose(self, action: enums.TransposeAction) -> "FiniteEventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "FiniteEventsStream": ...

    def apply(
        self, filter_class: type["FiniteEventsFilter"], *args, **kwargs
    ) -> "FiniteEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def to_array(self) -> numpy.ndarray:
        return numpy.concatenate(list(self))

    def envelope(
        self,
        decay: enums.Decay,
        tau: timestamp.Time,
    ) -> frame_stream.FiniteFloat64FrameStream: ...


class RegularEventsStream(
    stream.RegularStream[numpy.ndarray],
    Output[events_stream_state.RegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ) -> "RegularEventsStream":
        """
        Changes the frequency of the stream.

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            period: Time interval covered by each packet.
            start: Optional starting time of the first packet. If None (default), the timestamp of the first event is used.
        """
        ...

    def chunks(self, chunk_length: int) -> "EventsStream":
        from .events_filter import Chunks

        return Chunks(
            parent=self,  # type: ignore (see "Note on filter types" in events_filter)
            chunk_length=chunk_length,
        )

    def packet_slice(
        self,
        start: int,
        end: int,
        zero: bool = False,
    ) -> "FiniteRegularEventsStream": ...

    def event_slice(self, start: int, end: int) -> "FiniteRegularEventsStream": ...

    def remove_on_events(self) -> "RegularEventsStream": ...

    def remove_off_events(self) -> "RegularEventsStream": ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> "RegularEventsStream": ...

    def mask(self, array: numpy.ndarray) -> "RegularEventsStream": ...

    def transpose(self, action: enums.TransposeAction) -> "RegularEventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "RegularEventsStream": ...

    def apply(
        self, filter_class: type["RegularEventsFilter"], *args, **kwargs
    ) -> "RegularEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def envelope(
        self,
        decay: enums.Decay,
        tau: timestamp.Time,
    ) -> frame_stream.RegularFloat64FrameStream: ...


class FiniteRegularEventsStream(
    stream.FiniteRegularStream[numpy.ndarray],
    Output[events_stream_state.FiniteRegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ) -> "FiniteRegularEventsStream":
        """
        Changes the frequency of the stream.

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            period: Time interval covered by each packet.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range_us()[0]`) is used.
        """
        ...

    def chunks(self, chunk_length: int) -> "FiniteEventsStream": ...

    def packet_slice(
        self,
        start: int,
        end: int,
        zero: bool = False,
    ) -> "FiniteRegularEventsStream": ...

    def event_slice(self, start: int, end: int) -> "FiniteRegularEventsStream": ...

    def remove_on_events(self) -> "FiniteRegularEventsStream": ...

    def remove_off_events(self) -> "FiniteRegularEventsStream": ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> "FiniteRegularEventsStream": ...

    def mask(self, array: numpy.ndarray) -> "FiniteRegularEventsStream": ...

    def transpose(
        self, action: enums.TransposeAction
    ) -> "FiniteRegularEventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "FiniteRegularEventsStream": ...

    def apply(
        self, filter_class: type["FiniteRegularEventsFilter"], *args, **kwargs
    ) -> "FiniteRegularEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def to_array(self) -> numpy.ndarray:
        return numpy.concatenate(list(self))

    def envelope(
        self,
        decay: enums.Decay,
        tau: timestamp.Time,
    ) -> frame_stream.FiniteRegularFloat64FrameStream: ...


def bind(prefix: typing.Literal["", "Finite", "Regular", "FiniteRegular"]):
    regularize_prefix = (
        "Regular" if prefix == "" or prefix == "Regular" else "FiniteRegular"
    )
    chunks_prefix = "" if prefix == "" or prefix == "Regular" else "Finite"
    event_slice_prefix = (
        "Finite" if prefix == "" or prefix == "Finite" else "FiniteRegular"
    )

    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{regularize_prefix}Regularize"](
            parent=self,
            frequency_hz=frequency_hz,
            start=start,
        )

    def chunks(
        self,
        chunk_length: int,
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{chunks_prefix}Chunks"](parent=self, chunk_length=chunk_length)

    if prefix == "" or prefix == "Finite":

        def time_slice(
            self, start: timestamp.Time, end: timestamp.Time, zero: bool = False
        ):
            from .events_filter import FILTERS

            return FILTERS[f"FiniteTimeSlice"](
                parent=self,
                start=start,
                end=end,
                zero=zero,
            )

        time_slice.filter_return_annotation = f"FiniteEventsStream"
        globals()[f"{prefix}EventsStream"].time_slice = time_slice

    else:

        def packet_slice(
            self,
            start: int,
            end: int,
            zero: bool = False,
        ):
            from .events_filter import FILTERS

            return FILTERS[f"FiniteRegularPacketSlice"](
                parent=self,
                start=start,
                end=end,
                zero=zero,
            )

        packet_slice.filter_return_annotation = f"FiniteRegularPacketSlice"
        globals()[f"{prefix}EventsStream"].packet_slice = packet_slice

    def event_slice(
        self,
        start: int,
        end: int,
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{event_slice_prefix}EventSlice"](
            parent=self,
            start=start,
            end=end,
        )

    def remove_on_events(self):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Map"](
            parent=self, function=lambda events: events[numpy.logical_not(events["on"])]
        )

    def remove_off_events(self):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Map"](
            parent=self, function=lambda events: events[events["on"]]
        )

    def crop(self, left: int, right: int, top: int, bottom: int):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Crop"](
            parent=self,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

    def mask(self, array: numpy.ndarray):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Mask"](
            parent=self,
            array=array,
        )

    def transpose(self, action: enums.TransposeAction):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Transpose"](
            parent=self,
            action=action,
        )

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{prefix}Map"](
            parent=self,
            function=function,
        )

    def envelope(
        self,
        decay: enums.Decay,
        tau: timestamp.Time,
    ):
        from .render import FILTERS

        return FILTERS[f"{prefix}Float64Envelope"](parent=self, decay=decay, tau=tau)

    regularize.filter_return_annotation = f"{regularize_prefix}EventsStream"
    chunks.filter_return_annotation = f"{chunks_prefix}EventsStream"
    event_slice.filter_return_annotation = f"{event_slice_prefix}EventsStream"
    remove_on_events.filter_return_annotation = f"{prefix}EventsStream"
    remove_off_events.filter_return_annotation = f"{prefix}EventsStream"
    crop.filter_return_annotation = f"{prefix}EventsStream"
    mask.filter_return_annotation = f"{prefix}EventsStream"
    transpose.filter_return_annotation = f"{prefix}EventsStream"
    map.filter_return_annotation = f"{prefix}EventsStream"
    envelope.filter_return_annotation = f"{prefix}FrameStream"

    globals()[f"{prefix}EventsStream"].regularize = regularize
    globals()[f"{prefix}EventsStream"].chunks = chunks
    globals()[f"{prefix}EventsStream"].event_slice = event_slice
    globals()[f"{prefix}EventsStream"].remove_on_events = remove_on_events
    globals()[f"{prefix}EventsStream"].remove_off_events = remove_off_events
    globals()[f"{prefix}EventsStream"].crop = crop
    globals()[f"{prefix}EventsStream"].mask = mask
    globals()[f"{prefix}EventsStream"].transpose = transpose
    globals()[f"{prefix}EventsStream"].map = map
    globals()[f"{prefix}EventsStream"].envelope = envelope


for prefix in ("", "Finite", "Regular", "FiniteRegular"):
    bind(prefix=prefix)


class Array(FiniteEventsStream):
    def __init__(self, events: numpy.ndarray, dimensions: tuple[int, int]):
        super().__init__()
        assert self.events.dtype == EVENTS_DTYPE
        self.events = events
        self.inner_dimensions = dimensions

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        yield self.events.copy()

    def dimensions(self) -> tuple[int, int]:
        return self.inner_dimensions

    def time_range_us(self) -> tuple[int, int]:
        if len(self.events) == 0:
            return (0, 1)
        return (int(self.events["t"][0]), int(self.events["t"][-1]) + 1)


class EventsFilter(
    EventsStream,
    stream.Filter[numpy.ndarray],
):
    pass


class FiniteEventsFilter(
    FiniteEventsStream,
    stream.FiniteFilter[numpy.ndarray],
):
    pass


class RegularEventsFilter(
    RegularEventsStream,
    stream.RegularFilter[numpy.ndarray],
):
    pass


class FiniteRegularEventsFilter(
    FiniteRegularEventsStream,
    stream.FiniteRegularFilter[numpy.ndarray],
):
    pass
