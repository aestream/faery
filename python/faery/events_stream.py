from __future__ import annotations

import collections.abc
import pathlib
import typing

import numpy
import numpy.typing

from . import color, enums, events_stream_state, frame_stream, stream, timestamp

if typing.TYPE_CHECKING:
    from . import event_rate, kinectograph, spectrogram
    from .types import aedat, array
else:
    from .extension import aedat, array

EVENTS_DTYPE: numpy.dtype = numpy.dtype(
    [("t", "=u8"), ("x", "=u2"), ("y", "=u2"), (("p", "on"), "?")]
)


def concatenate_events(events_buffers: list[numpy.ndarray]) -> numpy.ndarray:
    """Joins event packets into one new contiguous array of EVENTS_DTYPE.

    numpy before 2.5 copies structured records field by field, which made
    joining packets (in regularize, chunks, or to_array) most of the cost of
    windowing a recording. The extension copies each packet with one memcpy,
    and falls back to numpy.concatenate for packets of another dtype.
    """
    result = array.concatenate(events_buffers, EVENTS_DTYPE)
    if result is None:
        return numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
    return result


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


_PREFETCH_DONE = object()


def _check_prefetch_buffers(
    out: typing.Union[numpy.ndarray, collections.abc.Sequence[numpy.ndarray], None],
    prefetch: int,
) -> None:
    if out is None:
        return
    buffers = 1 if isinstance(out, numpy.ndarray) else len(out)
    if buffers < prefetch + 2:
        raise ValueError(
            f"prefetch={prefetch} needs at least {prefetch + 2} out buffers "
            f"(got {buffers}): the background thread writes up to "
            f"prefetch + 1 arrays ahead of the current one"
        )


def _prefetched(
    source: collections.abc.Iterator[typing.Any], prefetch: int
) -> collections.abc.Iterator[typing.Any]:
    """Runs `source` in a background thread, up to `prefetch` items ahead.

    The faery DLPack exporters release the GIL while they walk a packet, so
    the next packet is prepared while the caller works on the current one
    (e.g. queues GPU work). Exceptions raised by `source` are re-raised here.
    Closing this generator (break, garbage collection) stops the thread and
    closes `source`.
    """
    import queue
    import threading

    items: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def produce():
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        items.put((item, None), timeout=0.05)
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    return
            item, error = _PREFETCH_DONE, None
        except BaseException as exception:
            # BaseException too: the consumer would otherwise wait forever.
            item, error = _PREFETCH_DONE, exception
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        while not stop.is_set():
            try:
                items.put((item, error), timeout=0.05)
                return
            except queue.Full:
                pass

    thread = threading.Thread(target=produce, name="faery-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item, error = items.get()
            if error is not None:
                raise error
            if item is _PREFETCH_DONE:
                return
            yield item
    finally:
        stop.set()
        thread.join()


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
            tuple[enums.EventsFileCompression, int]
        ] = aedat.LZ4_DEFAULT,
        csv_separator: bytes = b",",
        csv_header: bool = True,
        file_type: typing.Optional[enums.EventsFileType] = None,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
        enforce_monotonic_timestamps: bool = True,
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
            enforce_monotonic_timestamps: Whether to enforce that timestamps are monotonically increasing. Note that some formats
                (such as AEDAT, ES, and DAT) do not support non-monotonic timestamps. Defaults to True.

        Returns:
            The original t0 as a timecode if the file type is ES, EVT (.raw) or DAT, and if `zero_t0` is true. 0 as a timecode otherwise.
            To reconstruct the original timestamps when decoding ES files with Faery, pass the returned value to `faery.stream_from_file`.
            EVT (.raw) and DAT files do not need this (t0 is written in their header), but it is returned here anyway for compatibility
            with software than do not support the t0 header field.
        """
        from . import file_encoder

        try:
            self.time_range()
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
            enforce_monotonic_timestamps=enforce_monotonic_timestamps,
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
            enforce_monotonic_timestamps=False,
        )

    def to_udp(
        self,
        address: typing.Union[
            tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
        ],
        events_per_packet: typing.Optional[int] = None,
        format: enums.UdpFormat = "t64_x16_y16_on8",
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> None:
        """
        Sends the stream to the given UDP address and port.

        The address format defines whether IPv4 or IPv6 is used. If it has two items (host, port),
        IPv4 is used. It has four items (host, port, flowinfo, scope_id), IPv6 is used.
        To force IPv6 but omit scope_id and/or flowinfo, set them to None.
        See https://docs.python.org/3/library/socket.html for details.

        To maximize throughput, consider re-arranging the stream in packets of exactly `events_per_packet` events
        with `.chunks(events_per_packet)`.

        Args:
            address: the UDP address as (host, port) for IPv4 and (host, port, flowinfo, scope_id) for IPv6.
            events_per_packet: number of events per UDP packet. Defaults to 100 for both formats.
            format: Event encoding format, either "t64_x16_y16_on8" or "t32_x16_y15_on1". Defaults to "t64_x16_y16_on8".
        """
        from . import udp_encoder

        return udp_encoder.encode(
            stream=self,
            address=address,
            events_per_packet=events_per_packet,
            format=format,
            on_progress=on_progress,  # type: ignore
        )

    def _dlpack_batches(
        self, batch_events: typing.Optional[int]
    ) -> collections.abc.Iterator[list[numpy.ndarray]]:
        """Consecutive packets grouped until each group holds >= batch_events.

        Without batch_events, every packet is its own group. Batching only
        re-partitions the stream, which changes nothing about a non-regular
        stream (its packet boundaries carry no meaning). A regular stream's
        packets are time bins, and merging them would change the result, so
        batching a regular stream is refused.
        """
        if batch_events is None:
            for events in self:
                yield [events]
            return
        if batch_events < 1:
            raise ValueError(f"batch_events must be at least 1 (got {batch_events})")
        if isinstance(self, (RegularEventsStream, FiniteRegularEventsStream)):
            raise ValueError(
                "batch_events would merge the packets (time bins) of a regular stream; "
                "batch a non-regular stream, e.g. before regularize()"
            )
        pending: list[numpy.ndarray] = []
        pending_events = 0
        for events in self:
            if len(events) == 0:
                continue
            pending.append(events)
            pending_events += len(events)
            if pending_events >= batch_events:
                yield pending
                pending = []
                pending_events = 0
        if len(pending) > 0:
            yield pending

    def _dlpack_window_batches(
        self, windows_per_batch: int
    ) -> collections.abc.Iterator[list[numpy.ndarray]]:
        """Consecutive packets (time windows) of a regular stream, K at a time.

        Unlike _dlpack_batches, empty packets are kept: each is a frame.
        """
        pending: list[numpy.ndarray] = []
        for events in self:
            pending.append(events)
            if len(pending) == windows_per_batch:
                yield pending
                pending = []
        if len(pending) > 0:
            yield pending

    def to_dlpack_sparse(
        self,
        fields: collections.abc.Sequence[typing.Literal["t", "x", "y", "p"]] = (
            "t",
            "x",
            "y",
            "p",
        ),
        batch_events: typing.Optional[int] = None,
    ) -> collections.abc.Iterator[dict[str, numpy.ndarray]]:
        """
        Yields events per packet as a dict of contiguous arrays (by default {"t", "x", "y", "p"}).

        Each value is a numpy array that exposes `__dlpack__`, so it can be passed
        to any ML framework that supports the DLPack protocol
        (e.g. `torch.from_dlpack`, `jax.dlpack.from_dlpack`).

        Fields are copied out of the structured event packet into contiguous arrays
        so that DLPack export does not require strided support on the consumer side.
        Dtypes match the event packet:
            t: uint64, x: uint16, y: uint16, p: bool

        Each yielded array is one GPU upload. Packets decoded from a file are
        small (8192 events from DAT), and a transfer from pageable memory
        blocks, so per-packet uploads are dominated by their fixed cost:
        `batch_events` concatenates consecutive packets until each yielded
        array holds at least that many events. About 262144 (2^18) was fastest
        on a 1280x720 recording; one upload per packet was 1.8x slower. Only
        non-regular streams can be batched (see `_dlpack_batches`).

        Args:
            fields: Fields to extract, any subset of ("t", "x", "y", "p").
                Fields not listed are not copied. A continuously streaming GPU
                consumer typically only needs ("x", "y", "p") — timestamps are
                implicit in the packet cadence.
            batch_events: Minimum events per yielded array, or None (default)
                for one array per packet.
        """
        key_map = {"t": "t", "x": "x", "y": "y", "p": "on"}
        for field in fields:
            if field not in key_map:
                raise ValueError(
                    f'unknown field "{field}" (expected "t", "x", "y", or "p")'
                )
        for batch in self._dlpack_batches(batch_events):
            if len(batch) == 1:
                # .copy(), not ascontiguousarray: numpy flags any array with at
                # most one element as contiguous regardless of its stride, so for
                # 0- or 1-event packets ascontiguousarray returns the strided
                # field view itself (stride = record size), which
                # torch.from_numpy rejects.
                yield {field: batch[0][key_map[field]].copy() for field in fields}
            else:
                # Concatenating the field views copies each event once, into a
                # new contiguous array.
                yield {
                    field: numpy.concatenate(
                        [events[key_map[field]] for events in batch]
                    )
                    for field in fields
                }

    def to_dlpack_indices(
        self,
        batch_events: typing.Optional[int] = None,
        out: typing.Union[
            numpy.ndarray, collections.abc.Sequence[numpy.ndarray], None
        ] = None,
        prefetch: int = 0,
        windows_per_batch: typing.Optional[int] = None,
    ) -> collections.abc.Iterator[
        typing.Union[numpy.ndarray, tuple[numpy.ndarray, int]]
    ]:
        """
        Yields per-packet flat frame indices as 1-D int32 numpy arrays.

        Each event becomes `p * height * width + y * width + x` (p = 0 for OFF,
        1 for ON), an index into a flattened `(2, height, width)` frame. This
        is the most compact way to ship a packet to a GPU (4 bytes per event, a
        single transfer), and one scatter rebuilds the frame there:

            indices = torch.from_dlpack(indices_np).to("cuda")
            frame = torch.zeros(2 * height * width, device="cuda")
            frame.index_add_(0, indices, torch.ones(len(indices), device="cuda"))
            frame = frame.view(2, height, width)

        index_add_ benchmarks faster than torch.bincount, which returns int64
        counts that still need converting.

        Arrays expose `__dlpack__`. Events outside the sensor raise a ValueError.

        `batch_events` concatenates consecutive packets until each yielded
        array holds at least that many events, as for `to_dlpack_sparse`: fewer,
        larger uploads. Only non-regular streams can be batched.

        By default every array is newly allocated. Pass `out` to write into
        buffers instead: one array, or a sequence of arrays used in turn. Each
        must be a writeable, C-contiguous 1-D int32 array with room for the
        largest packet (or batch), and the yielded array is a view of its
        start, overwritten when the buffer's turn comes again. Pinned buffers
        make the upload a DMA transfer:

            buffers = [torch.empty(capacity, dtype=torch.int32, pin_memory=True)
                       for _ in range(2)]
            out = [buffer.numpy() for buffer in buffers]
            for index, indices in enumerate(stream.to_dlpack_indices(out=out)):
                gpu = buffers[index % 2][: len(indices)].to("cuda", non_blocking=True)
                ...  # wait for that copy before the buffer's next turn

        `prefetch` prepares up to that many arrays ahead in a background
        thread, overlapping the Rust work (which releases the GIL) with
        whatever the caller does with the current array, such as queueing GPU
        work. 0 (default) prepares each array on demand, in the caller's
        thread. Combine it with pinned `out` buffers: without them, the
        background thread writes into memory that the upload has just read
        on another core, and on CPUs with several L3 caches (AMD Ryzen and
        Threadripper, multi-socket systems) moving those cache lines made
        prefetching 3.5x slower than not prefetching. As for
        `to_dlpack_frame`, `out` needs at least `prefetch + 2` buffers.

        `windows_per_batch` groups the packets of a regular stream (its time
        windows) instead, K at a time, and yields `(indices, windows)`
        tuples: window j of the group has its indices offset by
        `j * 2 * height * width`, so one scatter builds every frame of the
        group:

            for indices, windows in stream.to_dlpack_indices(windows_per_batch=16):
                gpu = torch.from_dlpack(indices).to("cuda")
                frames = torch.zeros(windows * 2 * height * width, device="cuda")
                frames.index_add_(0, gpu, torch.ones(len(gpu), device="cuda"))
                for frame in frames.view(windows, 2, height, width):
                    ...

        Every upload and scatter has a fixed cost (about 0.1 ms with PyTorch
        on an RTX 3090), which dominates at high frame rates: grouping 16
        windows of 1 ms halved the time from file to frames. Frames arrive in
        groups, so a window waits for up to K - 1 later windows: at video
        rates, where per-window costs are small, the added latency
        outweighs the gain. `windows` is K except for the last group, and
        empty windows still count. Only regular streams can be grouped this
        way, and it excludes `batch_events`.

        Args:
            batch_events: Minimum events per yielded array, or None (default)
                for one array per packet.
            out: Optional buffer, or sequence of buffers, to write into.
            prefetch: Arrays to prepare ahead in a background thread
                (0 disables the thread).
            windows_per_batch: Windows of a regular stream per yielded
                array, or None (default) for one array per packet.
        """
        if prefetch < 0:
            raise ValueError(f"prefetch must be at least 0 (got {prefetch})")
        if windows_per_batch is not None:
            if batch_events is not None:
                raise ValueError("batch_events and windows_per_batch are exclusive")
            if windows_per_batch < 1:
                raise ValueError(
                    f"windows_per_batch must be at least 1 (got {windows_per_batch})"
                )
            if not isinstance(self, (RegularEventsStream, FiniteRegularEventsStream)):
                raise ValueError(
                    "windows_per_batch groups the time windows of a regular stream; "
                    "call regularize() first"
                )
        if prefetch > 0:
            _check_prefetch_buffers(out, prefetch)
            yield from _prefetched(
                self.to_dlpack_indices(
                    batch_events, out, windows_per_batch=windows_per_batch
                ),
                prefetch,
            )
            return
        from .extension import dlpack

        width, height = self.dimensions()
        if windows_per_batch is None:
            batches = self._dlpack_batches(batch_events)
        else:
            batches = self._dlpack_window_batches(windows_per_batch)
        buffers = None
        if out is not None:
            buffers = [out] if isinstance(out, numpy.ndarray) else list(out)
            if len(buffers) == 0:
                raise ValueError("out must contain at least one buffer")
        for index, batch in enumerate(batches):
            if windows_per_batch is None and buffers is None and len(batch) == 1:
                yield dlpack.linear_indices(batch[0], width, height)
                continue
            total = sum(len(events) for events in batch)
            if buffers is None:
                buffer = numpy.empty(total, dtype=numpy.int32)
            else:
                buffer = buffers[index % len(buffers)]
                if (
                    not isinstance(buffer, numpy.ndarray)
                    or buffer.ndim != 1
                    or len(buffer) < total
                ):
                    raise ValueError(
                        f"out must be a writeable, C-contiguous 1-D numpy array with "
                        f"dtype int32 and at least {total} elements"
                    )
            offset = 0
            for frame, events in enumerate(batch):
                dlpack.linear_indices(
                    events,
                    width,
                    height,
                    buffer[offset : offset + len(events)],
                    frame if windows_per_batch is not None else 0,
                )
                offset += len(events)
            if windows_per_batch is None:
                yield buffer[:total]
            else:
                yield buffer[:total], len(batch)

    def to_dlpack_frame(
        self,
        dtype: typing.Literal["u16", "u32", "f32"] = "u16",
        out: typing.Union[
            numpy.ndarray, collections.abc.Sequence[numpy.ndarray], None
        ] = None,
        prefetch: int = 0,
    ) -> collections.abc.Iterator[numpy.ndarray]:
        """
        Yields per-packet `(2, height, width)` frames as numpy arrays.

        Each frame is a polarity-split event count histogram where
        `frame[p, y, x]` is the number of events at pixel `(x, y)` with polarity `p`
        (0 = OFF, 1 = ON) inside the packet. Output is a numpy array that exposes
        `__dlpack__`, so it can be passed to any ML framework that supports DLPack.

        u16 saturates at 65535 (hot pixels in long packets may saturate);
        u32 and f32 are safe from saturation.

        By default every packet allocates a new frame. Pass `out` to reuse
        buffers instead: one array, or a sequence of arrays used in turn.
        Each must be a writeable, C-contiguous `(2, height, width)` array of
        the requested dtype. A yielded frame *is* one of these buffers, and is
        overwritten when its turn comes again, so finish with it (or copy it)
        by then. Two pinned host buffers allow asynchronous GPU uploads:

            # torch has limited uint16 support, so stage u16 frames as int16
            # (exact for counts below 32768).
            buffers = [torch.empty((2, h, w), dtype=torch.int16, pin_memory=True)
                       for _ in range(2)]
            out = [buffer.numpy().view(numpy.uint16) for buffer in buffers]
            for index, frame in enumerate(stream.to_dlpack_frame(out=out)):
                device_frame.copy_(buffers[index % 2], non_blocking=True)
                ...  # wait for the copy from buffer (index + 1) % 2 before
                     # the next iteration overwrites it (e.g. a CUDA event)

        `prefetch` rasterizes up to that many frames ahead in a background
        thread, overlapping the Rust work (which releases the GIL) with
        whatever the caller does with the current frame. 0 (default)
        rasterizes each frame on demand, in the caller's thread. With `out`,
        the thread writes up to `prefetch + 1` frames ahead of the one the
        caller holds, so `out` needs at least `prefetch + 2` buffers, and one
        more if the previous frame is still being read asynchronously (a
        non-blocking upload) when the next one is requested.

        Args:
            dtype: Output dtype, one of "u16" (default), "u32", or "f32".
            out: Optional buffer, or sequence of buffers, to rasterize into.
            prefetch: Frames to rasterize ahead in a background thread
                (0 disables the thread).
        """
        if prefetch < 0:
            raise ValueError(f"prefetch must be at least 0 (got {prefetch})")
        if prefetch > 0:
            _check_prefetch_buffers(out, prefetch)
            yield from _prefetched(self.to_dlpack_frame(dtype, out), prefetch)
            return
        from .extension import dlpack

        width, height = self.dimensions()
        if out is None:
            for events in self:
                yield dlpack.rasterize_to_frame(events, width, height, dtype)
            return
        buffers = [out] if isinstance(out, numpy.ndarray) else list(out)
        if len(buffers) == 0:
            raise ValueError("out must contain at least one buffer")
        for index, events in enumerate(self):
            yield dlpack.rasterize_to_frame(
                events, width, height, dtype, buffers[index % len(buffers)]
            )


class EventsStream(
    stream.Stream[numpy.ndarray], Output[events_stream_state.EventsStreamState]
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> RegularEventsStream:
        """
        Converts the stream to a regular stream with the given frequency (or packet rate).

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            frequency: Number of packets per second.
            start: Optional starting time of the first packet. If None (default), the timestamp of the first event is used.
        """

    def chunks(self, chunk_length: int) -> EventsStream: ...

    def time_slice(
        self,
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
        zero: bool = False,
    ) -> FiniteEventsStream: ...

    def event_slice(self, start: int, end: int) -> FiniteEventsStream: ...

    def remove_on_events(self) -> EventsStream: ...

    def remove_off_events(self) -> EventsStream: ...

    def crop(self, left: int, right: int, top: int, bottom: int) -> EventsStream: ...

    def mask(self, array: numpy.ndarray) -> EventsStream: ...

    def transpose(self, action: enums.TransposeAction) -> EventsStream: ...

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> EventsStream: ...

    def filter_hot_pixels(
        self,
        maximum_relative_event_count: float,
    ) -> EventsStream: ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> EventsStream: ...

    def apply(self, filter_class: type[EventsFilter], *args, **kwargs) -> EventsStream:
        return filter_class(self, *args, **kwargs)  # type: ignore

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
    ) -> frame_stream.FrameStream: ...


class FiniteEventsStream(
    stream.FiniteStream[numpy.ndarray],
    Output[events_stream_state.FiniteEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> FiniteRegularEventsStream:
        """
        Converts the stream to a regular stream with the given frequency (or packet rate).

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            frequency: Number of packets per second.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range()[0]`) is used.
        """

    def chunks(self, chunk_length: int) -> FiniteEventsStream:
        from .events_filter import Chunks

        return Chunks(  # ty: ignore[invalid-return-type]
            parent=self,  # type: ignore (see "Note on filter types" in events_filter)
            chunk_length=chunk_length,
        )

    def time_slice(
        self,
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
        zero: bool = False,
    ) -> FiniteEventsStream: ...

    def event_slice(self, start: int, end: int) -> FiniteEventsStream: ...

    def remove_on_events(self) -> FiniteEventsStream: ...

    def remove_off_events(self) -> FiniteEventsStream: ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> FiniteEventsStream: ...

    def mask(self, array: numpy.ndarray) -> FiniteEventsStream: ...

    def transpose(self, action: enums.TransposeAction) -> FiniteEventsStream: ...

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> FiniteEventsStream: ...

    def filter_hot_pixels(
        self,
        maximum_relative_event_count: float,
    ) -> FiniteEventsStream: ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> FiniteEventsStream: ...

    def apply(
        self, filter_class: type[FiniteEventsFilter], *args, **kwargs
    ) -> FiniteEventsStream:
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
        result = concatenate_events(events_buffers)
        state_manager.end()
        return result

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
    ) -> frame_stream.FiniteFrameStream: ...

    def reverse(self) -> FiniteEventsStream:
        """
        Reverses time: the same events in reverse order, with polarity flipped.

        Timestamps are mirrored within the time range ([start, end) becomes
        start + (end - 1) - t), so time still increases along the stream and the
        time range is unchanged. The first output event is the last input event,
        so the whole stream is buffered: memory grows with its length.
        """
        from .events_filter import FILTERS

        return FILTERS["FiniteReverse"](parent=self)  # ty: ignore[invalid-return-type]

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
    ) -> kinectograph.Kinectograph:

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

    def to_spectrogram(
        self,
        frequency_range: tuple[float, float] = (10.0, 4000.0),
        bins_per_octave: int = 12,
        columns: int = 1600,
        polarity: spectrogram.Polarity = "on_minus_off",
        sampling_rate: float | None = None,
        gamma: float | None = 0.0,
        filter_scale: float = 1.0,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> spectrogram.Spectrogram:

        from . import spectrogram

        return spectrogram.Spectrogram.from_events(
            stream=self,
            time_range=self.time_range(),
            frequency_range=frequency_range,
            bins_per_octave=bins_per_octave,
            columns=columns,
            polarity=polarity,
            sampling_rate=sampling_rate,
            gamma=gamma,
            filter_scale=filter_scale,
            on_progress=on_progress,  # type: ignore
        )

    def to_event_rate(
        self,
        samples: int = 1600,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> event_rate.EventRate:

        from . import event_rate

        return event_rate.EventRate.from_events(
            stream=self,
            time_range=self.time_range(),
            samples=samples,
            on_progress=on_progress,  # type: ignore
        )


class RegularEventsStream(
    stream.Stream[numpy.ndarray],
    Output[events_stream_state.RegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> RegularEventsStream:
        """
        Changes the frequency of the stream.

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            period: Time interval covered by each packet.
            start: Optional starting time of the first packet. If None (default), the timestamp of the first event is used.
        """

    def chunks(self, chunk_length: int) -> EventsStream:
        from .events_filter import Chunks

        return Chunks(  # ty: ignore[invalid-return-type]
            parent=self,  # type: ignore (see "Note on filter types" in events_filter)
            chunk_length=chunk_length,
        )

    def packet_slice(
        self,
        start: int,
        end: int,
        zero: bool = False,
    ) -> FiniteRegularEventsStream: ...

    def event_slice(self, start: int, end: int) -> FiniteRegularEventsStream: ...

    def remove_on_events(self) -> RegularEventsStream: ...

    def remove_off_events(self) -> RegularEventsStream: ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> RegularEventsStream: ...

    def mask(self, array: numpy.ndarray) -> RegularEventsStream: ...

    def transpose(self, action: enums.TransposeAction) -> RegularEventsStream: ...

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> EventsStream: ...

    def filter_hot_pixels(
        self,
        maximum_relative_event_count: float,
    ) -> RegularEventsStream: ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> RegularEventsStream: ...

    def apply(
        self, filter_class: type[RegularEventsFilter], *args, **kwargs
    ) -> RegularEventsStream:
        return filter_class(self, *args, **kwargs)  # type: ignore

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
    ) -> frame_stream.RegularFrameStream: ...


class FiniteRegularEventsStream(
    stream.FiniteRegularStream[numpy.ndarray],
    Output[events_stream_state.FiniteRegularEventsStreamState],
):
    def regularize(
        self,
        frequency_hz: float,
        start: typing.Optional[timestamp.TimeOrTimecode] = None,
    ) -> FiniteRegularEventsStream:
        """
        Changes the frequency of the stream.

        Args:
            parent: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
            period: Time interval covered by each packet.
            start: Optional starting time of the first packet. If None (default), the start of the time range (`parent.time_range()[0]`) is used.
        """

    def chunks(self, chunk_length: int) -> FiniteEventsStream: ...

    def packet_slice(
        self,
        start: int,
        end: int,
        zero: bool = False,
    ) -> FiniteRegularEventsStream: ...

    def event_slice(self, start: int, end: int) -> FiniteRegularEventsStream: ...

    def remove_on_events(self) -> FiniteRegularEventsStream: ...

    def remove_off_events(self) -> FiniteRegularEventsStream: ...

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> FiniteRegularEventsStream: ...

    def mask(self, array: numpy.ndarray) -> FiniteRegularEventsStream: ...

    def transpose(self, action: enums.TransposeAction) -> FiniteRegularEventsStream: ...

    def filter_arbiter_saturation_lines(
        self,
        maximum_line_fill_ratio: float,
        filter_orientation: enums.FilterOrientation = "row",
    ) -> FiniteEventsStream: ...

    def filter_hot_pixels(
        self,
        maximum_relative_event_count: float,
    ) -> FiniteRegularEventsStream: ...

    def map(
        self,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ) -> FiniteRegularEventsStream: ...

    def apply(
        self, filter_class: type[FiniteRegularEventsFilter], *args, **kwargs
    ) -> FiniteRegularEventsStream:
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
        result = concatenate_events(events_buffers)
        state_manager.end()
        return result

    def render(
        self,
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color.Colormap,
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
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
    ) -> kinectograph.Kinectograph:

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

    def to_spectrogram(
        self,
        frequency_range: tuple[float, float] = (10.0, 4000.0),
        bins_per_octave: int = 12,
        columns: int = 1600,
        polarity: spectrogram.Polarity = "on_minus_off",
        sampling_rate: float | None = None,
        gamma: float | None = 0.0,
        filter_scale: float = 1.0,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> spectrogram.Spectrogram:

        from . import spectrogram

        return spectrogram.Spectrogram.from_events(
            stream=self,
            time_range=self.time_range(),
            frequency_range=frequency_range,
            bins_per_octave=bins_per_octave,
            columns=columns,
            polarity=polarity,
            sampling_rate=sampling_rate,
            gamma=gamma,
            filter_scale=filter_scale,
            on_progress=on_progress,  # type: ignore
        )

    def to_event_rate(
        self,
        samples: int = 1600,
        on_progress: typing.Callable[[OutputState], None] = lambda _: None,
    ) -> event_rate.EventRate:

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

            return FILTERS["FiniteTimeSlice"](
                parent=self,
                start=start,
                end=end,
                zero=zero,
            )

        time_slice.filter_return_annotation = "FiniteEventsStream"
        globals()[f"{prefix}EventsStream"].time_slice = time_slice

    else:

        def packet_slice(
            self,
            start: int,
            end: int,
            zero: bool = False,
        ):
            from .events_filter import FILTERS

            return FILTERS["FiniteRegularPacketSlice"](
                parent=self,
                start=start,
                end=end,
                zero=zero,
            )

        packet_slice.filter_return_annotation = "FiniteRegularPacketSlice"
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

    def filter_hot_pixels(
        self,
        maximum_relative_event_count: float,
    ):
        from .events_filter import FILTERS

        return FILTERS[f"{unregularize_prefix}FilterHotPixels"](
            parent=self,
            maximum_relative_event_count=maximum_relative_event_count,
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
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
    ):
        from .events_render import FILTERS

        return FILTERS[f"{prefix}Render"](
            parent=self,
            decay=decay,
            tau=tau,
            colormap=colormap,
            minimum_clip=minimum_clip,
            maximum_clip=maximum_clip,
            gamma=gamma,
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
    filter_hot_pixels.filter_return_annotation = f"{prefix}EventsStream"
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
    globals()[f"{prefix}EventsStream"].filter_hot_pixels = filter_hot_pixels
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
