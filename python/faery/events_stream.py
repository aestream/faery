import collections.abc
import pathlib
import typing

import numpy
import numpy.typing

from . import color, enums, events_stream_state, frame_stream, stream, timestamp

if typing.TYPE_CHECKING:
    from . import event_rate, kinectograph
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
        csv_separator: bytes = b",",
        csv_header: bool = True,
        file_type: typing.Optional[enums.EventsFileType] = None,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> str:
        """
        Writes the stream to an event file (supports .aedat4, .es, .raw, and .dat).

        version is only used if the file type is EVT (.raw) or DAT.

        zero_t0 is only used if the file type is ES, EVT (.raw) or DAT.
        The original t0 is stored in the header of EVT and DAT files, and is discarded if the file type is ES.

        compression is only used if the file type is AEDAT.

        csv_separator and csv_header are only used if the file type is CSV.

        Args:
            stream: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            path: Path of the output event file.
            dimensions: Width and height of the sensor.
            version: Version for EVT (.raw) and DAT files. Defaults to "dat2" for DAT and "evt3" for EVT.
            zero_t0: Whether to normalize timestamps and write the offset in the header for EVT (.raw) and DAT files. Defaults to True.
            compression: Compression for aedat files. Defaults to ("lz4", 1).
            csv_separator: Separator between CSV fields. Defaults to b",".
            csv_header: Whether to generate a CSV header. Defaults to True.
            file_type: Override the type determination algorithm. Defaults to None.

        Returns:
            The original t0 as a timecode if the file type is ES, EVT (.raw) or DAT, and if `zero_t0` is true. 0 as a timecode otherwise.
            To reconstruct the original timestamps when decoding ES files with Faery, pass the returned value to `faery.stream_from_file`.
            EVT (.raw) and DAT files do not need this (t0 is written in their header), but it is returned here anyway for compatibility
            with software than do not support the t0 header field.
        """
        from . import file_encoder

        try:
            self.time_range()  # type: ignore
            use_write_suffix = True
        except (AttributeError, NotImplementedError):
            use_write_suffix = False
        return file_encoder.events_to_file(
            stream=self,
            path=path,
            dimensions=self.dimensions(),
            version=version,
            zero_t0=zero_t0,
            compression=compression,
            csv_separator=csv_separator,
            csv_header=csv_header,
            file_type=file_type,
            use_write_suffix=use_write_suffix,
            on_progress=on_progress,  # type: ignore
        )

    def to_stdout(
        self,
        csv_separator: bytes = b",",
        csv_header: bool = True,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> str:
        from . import file_encoder

        return file_encoder.events_to_file(
            stream=self,
            path=None,
            dimensions=self.dimensions(),
            csv_separator=csv_separator,
            csv_header=csv_header,
            file_type="csv",
            on_progress=on_progress,  # type: ignore
        )

    def to_udp(
        self,
        address: typing.Union[
            tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
        ],
        payload_length: typing.Optional[int] = None,
        format: enums.UdpFormat = "t64_x16_y16_on8",
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
            on_progress=on_progress,  # type: ignore
        )


class EventsStream(
    stream.Stream[numpy.ndarray], Output[events_stream_state.EventsStreamState]
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
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
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
        zero: bool = False,
    ) -> "FiniteEventsStream": ...

    def event_slice(self, start: int, end: int) -> "FiniteEventsStream": ...

    def remove_on_events(self) -> "EventsStream": ...

    def remove_off_events(self) -> "EventsStream": ...

    def crop(self, left: int, right: int, top: int, bottom: int) -> "EventsStream": ...

    def mask(self, array: numpy.ndarray) -> "EventsStream": ...

    def transpose(self, action: enums.TransposeAction) -> "EventsStream": ...

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> "EventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "EventsStream": ...

    def apply(
        self, filter_class: type["EventsFilter"], *args, **kwargs
    ) -> "EventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
    ) -> frame_stream.FrameStream: ...


class FiniteEventsStream(
    stream.FiniteStream[numpy.ndarray],
    Output[events_stream_state.FiniteEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> "FiniteRegularEventsStream":
        """
        Converts the stream to a regular stream with the given frequency (or packet rate).

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            frequency: Number of packets per second.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range()[0]`) is used.
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
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
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

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> "FiniteEventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "FiniteEventsStream": ...

    def apply(
        self, filter_class: type["FiniteEventsFilter"], *args, **kwargs
    ) -> "FiniteEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def to_array(
        self, on_progress: typing.Callable[[OutputState], None] = lambda _: None
    ) -> numpy.ndarray:
        events_buffers = []
        state_manager = events_stream_state.StateManager(
            stream=self, on_progress=on_progress
        )
        state_manager.start()
        for events in self:
            events_buffers.append(events)
            state_manager.commit(events=events)
        if len(events_buffers) == 0:
            result = numpy.array([], dtype=EVENTS_DTYPE)
        else:
            result = numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
        state_manager.end()
        return result

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
    ) -> frame_stream.FiniteFrameStream: ...

    def to_kinectograph(
        self,
        threshold_quantile: float = 0.9,
        normalized_times_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda normalized_times: normalized_times,
        opacities_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda opacities_gamma: opacities_gamma,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> "kinectograph.Kinectograph":

        from . import kinectograph

        return kinectograph.Kinectograph.from_events(
            stream=self,
            dimensions=self.dimensions(),
            time_range=self.time_range(),
            threshold_quantile=threshold_quantile,
            normalized_times_gamma=normalized_times_gamma,
            opacities_gamma=opacities_gamma,
            on_progress=on_progress,  # type: ignore
        )

    def to_event_rate(
        self,
        samples: int = 1600,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> "event_rate.EventRate":

        from . import event_rate

        return event_rate.EventRate.from_events(
            stream=self,
            time_range=self.time_range(),
            samples=samples,
            on_progress=on_progress,  # type: ignore
        )


class RegularEventsStream(
    stream.RegularStream[numpy.ndarray],
    Output[events_stream_state.RegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
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

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> "EventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "RegularEventsStream": ...

    def apply(
        self, filter_class: type["RegularEventsFilter"], *args, **kwargs
    ) -> "RegularEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
    ) -> frame_stream.RegularFrameStream: ...


class FiniteRegularEventsStream(
    stream.FiniteRegularStream[numpy.ndarray],
    Output[events_stream_state.FiniteRegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> "FiniteRegularEventsStream":
        """
        Changes the frequency of the stream.

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            period: Time interval covered by each packet.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range()[0]`) is used.
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

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> "FiniteEventsStream": ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> "FiniteRegularEventsStream": ...

    def apply(
        self, filter_class: type["FiniteRegularEventsFilter"], *args, **kwargs
    ) -> "FiniteRegularEventsStream":
        return filter_class(self, *args, **kwargs)  # type: ignore

    def to_array(
        self, on_progress: typing.Callable[[OutputState], None] = lambda _: None
    ) -> numpy.ndarray:
        events_buffers = []
        state_manager = events_stream_state.StateManager(
            stream=self, on_progress=on_progress
        )
        state_manager.start()
        for events in self:
            events_buffers.append(events)
            state_manager.commit(events=events)
        if len(events_buffers) == 0:
            result = numpy.array([], dtype=EVENTS_DTYPE)
        else:
            result = numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
        state_manager.end()
        return result

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
    ) -> frame_stream.FiniteRegularFrameStream: ...

    def to_kinectograph(
        self,
        threshold_quantile: float = 0.9,
        normalized_times_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda normalized_times: normalized_times,
        opacities_gamma: typing.Callable[
            [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
        ] = lambda opacities_gamma: opacities_gamma,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> "kinectograph.Kinectograph":

        from . import kinectograph

        return kinectograph.Kinectograph.from_events(
            stream=self,
            dimensions=self.dimensions(),
            time_range=self.time_range(),
            threshold_quantile=threshold_quantile,
            normalized_times_gamma=normalized_times_gamma,
            opacities_gamma=opacities_gamma,
            on_progress=on_progress,  # type: ignore
        )

    def to_event_rate(
        self,
        samples: int = 1600,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> "event_rate.EventRate":

        from . import event_rate

        return event_rate.EventRate.from_events(
            stream=self,
            time_range=self.time_range(),
            samples=samples,
            on_progress=on_progress,  # type: ignore
        )


def bind(prefix: typing.Literal["", "Finite", "Regular", "FiniteRegular"]):
    regularize_prefix = (
        "Regular" if prefix == "" or prefix == "Regular" else "FiniteRegular"
    )
    unregularize_prefix = "" if prefix == "" or prefix == "Regular" else "Finite"
    finitize_prefix = (
        "Finite" if prefix == "" or prefix == "Finite" else "FiniteRegular"
    )

    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
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

        return FILTERS[f"{unregularize_prefix}Chunks"](
            parent=self, chunk_length=chunk_length
        )

    if prefix == "" or prefix == "Finite":

        def time_slice(
            self,
            start: timestamp.TimeOrTimecode,
            end: timestamp.TimeOrTimecode,
            zero: bool = False,
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

        return FILTERS[f"{finitize_prefix}EventSlice"](
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

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{unregularize_prefix}FilterArbiterSaturationLines"](
            parent=self,
            maximum_line_fill_ratio=maximum_line_fill_ratio,
            filter_orientation=filter_orientation,
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

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
    ):
        from .events_render import FILTERS

        return FILTERS[f"{prefix}Render"](
            parent=self, decay=decay, tau=tau, colormap=colormap
        )

    regularize.filter_return_annotation = f"{regularize_prefix}EventsStream"
    chunks.filter_return_annotation = f"{unregularize_prefix}EventsStream"
    event_slice.filter_return_annotation = f"{finitize_prefix}EventsStream"
    remove_on_events.filter_return_annotation = f"{prefix}EventsStream"
    remove_off_events.filter_return_annotation = f"{prefix}EventsStream"
    crop.filter_return_annotation = f"{prefix}EventsStream"
    mask.filter_return_annotation = f"{prefix}EventsStream"
    transpose.filter_return_annotation = f"{prefix}EventsStream"
    filter_arbiter_saturation_lines.filter_return_annotation = (
        f"{unregularize_prefix}EventsStream"
    )
    map.filter_return_annotation = f"{prefix}EventsStream"
    render.filter_return_annotation = f"{prefix}FrameStream"

    globals()[f"{prefix}EventsStream"].regularize = regularize
    globals()[f"{prefix}EventsStream"].chunks = chunks
    globals()[f"{prefix}EventsStream"].event_slice = event_slice
    globals()[f"{prefix}EventsStream"].remove_on_events = remove_on_events
    globals()[f"{prefix}EventsStream"].remove_off_events = remove_off_events
    globals()[f"{prefix}EventsStream"].crop = crop
    globals()[f"{prefix}EventsStream"].mask = mask
    globals()[f"{prefix}EventsStream"].transpose = transpose
    globals()[
        f"{prefix}EventsStream"
    ].filter_arbiter_saturation_lines = filter_arbiter_saturation_lines
    globals()[f"{prefix}EventsStream"].map = map
    globals()[f"{prefix}EventsStream"].render = render


for prefix in ("", "Finite", "Regular", "FiniteRegular"):
    bind(prefix=prefix)


class Array(FiniteEventsStream):
    def __init__(self, events: numpy.ndarray, dimensions: tuple[int, int]):
        super().__init__()
        assert events.dtype == EVENTS_DTYPE
        self.events = events
        self.inner_dimensions = dimensions

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        yield self.events.copy()

    def dimensions(self) -> tuple[int, int]:
        return self.inner_dimensions

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        if len(self.events) == 0:
            return (timestamp.Time(microseconds=0), timestamp.Time(microseconds=1))
        return (
            timestamp.Time(microseconds=int(self.events["t"][0])),
            timestamp.Time(microseconds=int(self.events["t"][-1]) + 1),
        )


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
