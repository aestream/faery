import numpy
import pytest

from faery.events_stream import EVENTS_DTYPE, Output


def _make_packet():
    events = numpy.array(
        [
            (10, 1, 2, False),
            (20, 1, 2, True),
            (30, 1, 2, True),
            (40, 0, 0, False),
            (50, 9, 7, True),
        ],
        dtype=EVENTS_DTYPE,
    )
    return events


class _FixedStream(Output):
    """A minimal finite events stream stand-in for unit testing the DLPack outputs.

    Implements just the surface that to_dlpack_sparse / to_dlpack_frame need:
    an iterator of structured event packets and a dimensions() method.
    """

    def __init__(self, packets, dimensions):
        self._packets = packets
        self._dimensions = dimensions

    def __iter__(self):
        return iter(self._packets)

    def dimensions(self):
        return self._dimensions


def test_to_dlpack_sparse_yields_field_dicts():
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    fields = list(stream.to_dlpack_sparse())
    assert len(fields) == 1
    out = fields[0]
    assert set(out) == {"t", "x", "y", "p"}
    assert out["t"].dtype == numpy.uint64
    assert out["x"].dtype == numpy.uint16
    assert out["y"].dtype == numpy.uint16
    assert out["p"].dtype == numpy.bool_
    numpy.testing.assert_array_equal(out["t"], packet["t"])
    numpy.testing.assert_array_equal(out["x"], packet["x"])
    numpy.testing.assert_array_equal(out["y"], packet["y"])
    numpy.testing.assert_array_equal(out["p"], packet["on"])
    for arr in out.values():
        assert arr.flags["C_CONTIGUOUS"]
        assert hasattr(arr, "__dlpack__")


def test_to_dlpack_sparse_field_subset():
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    out = next(iter(stream.to_dlpack_sparse(fields=("x", "y", "p"))))
    assert set(out) == {"x", "y", "p"}
    numpy.testing.assert_array_equal(out["x"], packet["x"])
    numpy.testing.assert_array_equal(out["p"], packet["on"])


def test_to_dlpack_sparse_unknown_field():
    stream = _FixedStream([_make_packet()], dimensions=(10, 8))
    with pytest.raises(ValueError):
        next(iter(stream.to_dlpack_sparse(fields=("x", "polarity"))))  # type: ignore[arg-type]


def test_to_dlpack_frame_counts_polarities():
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    frames = list(stream.to_dlpack_frame())
    assert len(frames) == 1
    frame = frames[0]
    assert frame.shape == (2, 8, 10)
    assert frame.dtype == numpy.uint16
    # Pixel (1, 2) has 1 OFF and 2 ON events.
    assert frame[0, 2, 1] == 1
    assert frame[1, 2, 1] == 2
    # Pixel (0, 0) has 1 OFF event.
    assert frame[0, 0, 0] == 1
    # Pixel (9, 7) has 1 ON event.
    assert frame[1, 7, 9] == 1
    # Total counts.
    assert frame.sum() == len(packet)


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [("u16", numpy.uint16), ("u32", numpy.uint32), ("f32", numpy.float32)],
)
def test_to_dlpack_frame_dtype_selection(dtype, np_dtype):
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    frame = next(iter(stream.to_dlpack_frame(dtype=dtype)))
    assert frame.dtype == np_dtype
    assert frame.sum() == len(packet)


def test_to_dlpack_frame_rejects_out_of_bounds():
    bad = numpy.array([(0, 20, 0, True)], dtype=EVENTS_DTYPE)
    stream = _FixedStream([bad], dimensions=(10, 8))
    with pytest.raises(ValueError):
        list(stream.to_dlpack_frame())


def test_to_dlpack_frame_invalid_dtype():
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    with pytest.raises(ValueError):
        list(stream.to_dlpack_frame(dtype="bad"))  # type: ignore[arg-type]


def test_dlpack_capsule_roundtrip_via_numpy():
    """numpy.from_dlpack should round-trip a frame produced by to_dlpack_frame."""
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    frame = next(iter(stream.to_dlpack_frame()))
    round_tripped = numpy.from_dlpack(frame)
    assert round_tripped.shape == frame.shape
    numpy.testing.assert_array_equal(round_tripped, frame)


def test_to_dlpack_frame_negative_stride():
    packet = _make_packet()
    stream = _FixedStream([packet[::-1]], dimensions=(10, 8))
    reference = next(iter(_FixedStream([packet], (10, 8)).to_dlpack_frame()))
    frame = next(iter(stream.to_dlpack_frame()))
    numpy.testing.assert_array_equal(frame, reference)


def test_to_dlpack_frame_non_canonical_polarity_bytes():
    """Bool bytes other than 0/1 count as ON and never index past the frame."""
    packet = _make_packet()
    raw = packet.view(numpy.uint8).reshape(len(packet), EVENTS_DTYPE.itemsize)
    raw[:, EVENTS_DTYPE.fields["on"][1]] *= 200  # True -> 200, False stays 0
    stream = _FixedStream([packet], dimensions=(10, 8))
    frame = next(iter(stream.to_dlpack_frame()))
    assert frame.sum() == len(packet)
    assert frame[1, 2, 1] == 2
    assert frame[0, 2, 1] == 1


def test_to_dlpack_indices_matches_frame():
    packet = _make_packet()
    stream = _FixedStream([packet], dimensions=(10, 8))
    indices = next(iter(stream.to_dlpack_indices()))
    assert indices.dtype == numpy.int32
    assert indices.shape == (len(packet),)
    numpy.testing.assert_array_equal(
        indices,
        packet["on"].astype(numpy.int32) * 80 + packet["y"] * 10 + packet["x"],
    )
    frame = next(iter(stream.to_dlpack_frame(dtype="u32")))
    numpy.testing.assert_array_equal(
        numpy.bincount(indices, minlength=2 * 8 * 10).reshape(2, 8, 10), frame
    )


def test_to_dlpack_indices_empty_and_reversed():
    packet = _make_packet()
    empty = next(iter(_FixedStream([packet[:0]], (10, 8)).to_dlpack_indices()))
    assert empty.shape == (0,)
    forward = next(iter(_FixedStream([packet], (10, 8)).to_dlpack_indices()))
    reverse = next(iter(_FixedStream([packet[::-1]], (10, 8)).to_dlpack_indices()))
    numpy.testing.assert_array_equal(reverse, forward[::-1])


def test_to_dlpack_indices_rejects_out_of_bounds():
    bad = numpy.array([(0, 3, 9, True)], dtype=EVENTS_DTYPE)
    with pytest.raises(ValueError):
        list(_FixedStream([bad], dimensions=(10, 8)).to_dlpack_indices())


def test_to_dlpack_frame_u16_saturates():
    hot = numpy.zeros(70_000, dtype=EVENTS_DTYPE)
    hot["x"], hot["y"], hot["on"] = 3, 4, True
    frame = next(iter(_FixedStream([hot], dimensions=(10, 8)).to_dlpack_frame()))
    assert frame[1, 4, 3] == 65535
    assert frame.sum() == 65535


def test_to_dlpack_frame_out_reuses_buffers():
    packets = [_make_packet(), _make_packet()[:2], _make_packet()[3:]]
    stream = _FixedStream(packets, dimensions=(10, 8))
    expected = [frame.copy() for frame in stream.to_dlpack_frame(dtype="f32")]
    buffers = [numpy.full((2, 8, 10), 7, dtype=numpy.float32) for _ in range(2)]
    for index, frame in enumerate(stream.to_dlpack_frame(dtype="f32", out=buffers)):
        assert frame is buffers[index % 2]
        numpy.testing.assert_array_equal(frame, expected[index])
    single = numpy.empty((2, 8, 10), dtype=numpy.float32)
    frames = [f.sum() for f in stream.to_dlpack_frame(dtype="f32", out=single)]
    assert frames == [len(p) for p in packets]


@pytest.mark.parametrize(
    "buffer",
    [
        numpy.zeros((2, 8, 10), dtype=numpy.float32),  # wrong dtype for u16
        numpy.zeros((2, 10, 8), dtype=numpy.uint16),  # wrong shape
        numpy.zeros((2, 8, 20), dtype=numpy.uint16)[:, :, ::2],  # not contiguous
        numpy.zeros((2, 8, 10), dtype=numpy.uint16, order="F"),  # not C order
    ],
)
def test_to_dlpack_frame_out_rejects_bad_buffers(buffer):
    stream = _FixedStream([_make_packet()], dimensions=(10, 8))
    with pytest.raises(ValueError):
        list(stream.to_dlpack_frame(out=buffer))


def test_to_dlpack_frame_out_rejects_read_only():
    buffer = numpy.zeros((2, 8, 10), dtype=numpy.uint16)
    buffer.flags.writeable = False
    stream = _FixedStream([_make_packet()], dimensions=(10, 8))
    with pytest.raises(ValueError):
        list(stream.to_dlpack_frame(out=buffer))


def _batch_packets():
    packet = _make_packet()
    return [packet, packet[:2], packet[:0], packet[3:], packet[1:4]]


@pytest.mark.parametrize("batch_events", [1, 4, 7, 100])
def test_to_dlpack_sparse_batches_concatenate_packets(batch_events):
    stream = _FixedStream(_batch_packets(), dimensions=(10, 8))
    unbatched = list(stream.to_dlpack_sparse())
    batched = list(stream.to_dlpack_sparse(batch_events=batch_events))
    for field in ("t", "x", "y", "p"):
        numpy.testing.assert_array_equal(
            numpy.concatenate([b[field] for b in batched]),
            numpy.concatenate([u[field] for u in unbatched]),
        )
        assert all(b[field].flags.c_contiguous for b in batched)
    # Every array but the last reaches batch_events; empty packets are dropped.
    sizes = [len(b["t"]) for b in batched]
    assert all(size >= batch_events for size in sizes[:-1])
    assert all(size > 0 for size in sizes)


@pytest.mark.parametrize("batch_events", [1, 6, 100])
def test_to_dlpack_indices_batches_concatenate_packets(batch_events):
    stream = _FixedStream(_batch_packets(), dimensions=(10, 8))
    unbatched = numpy.concatenate(list(stream.to_dlpack_indices()))
    batched = list(stream.to_dlpack_indices(batch_events=batch_events))
    numpy.testing.assert_array_equal(numpy.concatenate(batched), unbatched)
    assert all(b.dtype == numpy.int32 for b in batched)


def test_dlpack_batching_refuses_regular_streams():
    import faery

    regular = faery.events_stream_from_array(
        _make_packet(), dimensions=(10, 8)
    ).regularize(frequency_hz=1e5)
    with pytest.raises(ValueError, match="regular"):
        next(regular.to_dlpack_sparse(batch_events=4))
    with pytest.raises(ValueError, match="regular"):
        next(regular.to_dlpack_indices(batch_events=4))
    # Without batching, regular streams are unaffected.
    assert sum(len(p["t"]) for p in regular.to_dlpack_sparse()) == 5


def test_dlpack_batching_rejects_non_positive():
    stream = _FixedStream(_batch_packets(), dimensions=(10, 8))
    with pytest.raises(ValueError):
        next(stream.to_dlpack_indices(batch_events=0))


def _prefetch_threads():
    import threading

    return [t for t in threading.enumerate() if t.name == "faery-prefetch"]


def _many_packets(count=50):
    rng = numpy.random.default_rng(0)
    packets = []
    for _ in range(count):
        packet = numpy.zeros(int(rng.integers(0, 200)), dtype=EVENTS_DTYPE)
        packet["x"] = rng.integers(0, 10, len(packet))
        packet["y"] = rng.integers(0, 8, len(packet))
        packet["on"] = rng.integers(0, 2, len(packet)).astype(bool)
        packets.append(packet)
    return packets


@pytest.mark.parametrize("prefetch", [1, 3])
def test_prefetch_matches_on_demand(prefetch):
    stream = _FixedStream(_many_packets(), dimensions=(10, 8))
    frames = list(stream.to_dlpack_frame(dtype="u32"))
    prefetched = list(stream.to_dlpack_frame(dtype="u32", prefetch=prefetch))
    assert len(prefetched) == len(frames)
    for a, b in zip(prefetched, frames):
        numpy.testing.assert_array_equal(a, b)
    indices = list(stream.to_dlpack_indices())
    prefetched = list(stream.to_dlpack_indices(prefetch=prefetch))
    assert len(prefetched) == len(indices)
    for a, b in zip(prefetched, indices):
        numpy.testing.assert_array_equal(a, b)
    assert _prefetch_threads() == []


@pytest.mark.parametrize("prefetch", [1, 2])
def test_prefetch_out_buffers_not_overwritten_while_held(prefetch):
    import time

    stream = _FixedStream(_many_packets(20), dimensions=(10, 8))
    expected = [frame.copy() for frame in stream.to_dlpack_frame()]
    buffers = [numpy.empty((2, 8, 10), dtype=numpy.uint16) for _ in range(prefetch + 2)]
    for index, frame in enumerate(stream.to_dlpack_frame(out=buffers, prefetch=prefetch)):
        assert frame is buffers[index % len(buffers)]
        # Give the producer time to run ahead as far as it can.
        time.sleep(0.005)
        numpy.testing.assert_array_equal(frame, expected[index])


def test_prefetch_rejects_too_few_out_buffers():
    stream = _FixedStream([_make_packet()], dimensions=(10, 8))
    buffers = [numpy.empty((2, 8, 10), dtype=numpy.uint16) for _ in range(3)]
    with pytest.raises(ValueError, match="at least 4"):
        next(stream.to_dlpack_frame(out=buffers, prefetch=2))
    with pytest.raises(ValueError):
        next(stream.to_dlpack_frame(prefetch=-1))


def test_prefetch_propagates_errors():
    bad = numpy.array([(0, 20, 0, True)], dtype=EVENTS_DTYPE)
    stream = _FixedStream([_make_packet(), bad, _make_packet()], dimensions=(10, 8))
    frames = stream.to_dlpack_frame(prefetch=2)
    assert next(frames).sum() == 5
    with pytest.raises(ValueError, match="out of bounds"):
        next(frames)
    with pytest.raises(ValueError, match="out of bounds"):
        list(stream.to_dlpack_indices(prefetch=1))
    assert _prefetch_threads() == []


def test_prefetch_stops_thread_on_break():
    closed = []

    class _EndlessStream(_FixedStream):
        def __iter__(self):
            try:
                while True:
                    yield _make_packet()
            finally:
                closed.append(True)

    stream = _EndlessStream([], dimensions=(10, 8))
    for index, _ in enumerate(stream.to_dlpack_indices(prefetch=2)):
        if index == 3:
            break
    assert _prefetch_threads() == []
    assert closed == [True]
    frames = stream.to_dlpack_frame(prefetch=2)
    next(frames)
    frames.close()
    assert _prefetch_threads() == []
    assert closed == [True, True]


def test_to_dlpack_indices_out_reuses_buffers():
    stream = _FixedStream(_batch_packets(), dimensions=(10, 8))
    expected = list(stream.to_dlpack_indices())
    buffers = [numpy.full(16, -1, dtype=numpy.int32) for _ in range(2)]
    for index, indices in enumerate(stream.to_dlpack_indices(out=buffers)):
        assert indices.base is buffers[index % 2]
        numpy.testing.assert_array_equal(indices, expected[index])
    batched = list(stream.to_dlpack_indices(batch_events=6, out=numpy.empty(16, numpy.int32)))
    assert sum(map(len, batched)) == sum(map(len, expected))


@pytest.mark.parametrize(
    "buffer",
    [
        numpy.zeros(16, dtype=numpy.int64),  # wrong dtype
        numpy.zeros(4, dtype=numpy.int32),  # too small for 5 events
        numpy.zeros(32, dtype=numpy.int32)[::2],  # not contiguous
        numpy.zeros((2, 8), dtype=numpy.int32),  # not 1-D
    ],
)
def test_to_dlpack_indices_out_rejects_bad_buffers(buffer):
    stream = _FixedStream([_make_packet()], dimensions=(10, 8))
    with pytest.raises(ValueError):
        list(stream.to_dlpack_indices(out=buffer))


@pytest.mark.parametrize("prefetch", [1, 2])
def test_prefetch_indices_out_buffers_not_overwritten_while_held(prefetch):
    import time

    stream = _FixedStream(_many_packets(20), dimensions=(10, 8))
    expected = list(stream.to_dlpack_indices())
    buffers = [numpy.empty(200, dtype=numpy.int32) for _ in range(prefetch + 2)]
    for index, indices in enumerate(stream.to_dlpack_indices(out=buffers, prefetch=prefetch)):
        time.sleep(0.005)
        numpy.testing.assert_array_equal(indices, expected[index])
    with pytest.raises(ValueError, match=f"at least {prefetch + 2}"):
        next(stream.to_dlpack_indices(out=buffers[:-1], prefetch=prefetch))


def test_linear_indices_frame_offset():
    from faery.extension import dlpack

    packet = _make_packet()
    base = dlpack.linear_indices(packet, 10, 8)
    numpy.testing.assert_array_equal(
        dlpack.linear_indices(packet, 10, 8, frame=3), base + 3 * 2 * 10 * 8
    )
    out = numpy.zeros(len(packet), dtype=numpy.int32)
    dlpack.linear_indices(packet, 10, 8, out, 2)
    numpy.testing.assert_array_equal(out, base + 2 * 2 * 10 * 8)


def test_linear_indices_frame_offset_overflow():
    from faery.extension import dlpack

    # 2 x 1280 x 720 = 1,843,200 elements: 1,165 frames fit in int32, 1,166 do not.
    packet = numpy.zeros(1, dtype=EVENTS_DTYPE)
    last = dlpack.linear_indices(packet, 1280, 720, frame=1164)
    assert last[0] == 1164 * 1_843_200
    with pytest.raises(ValueError, match="int32"):
        dlpack.linear_indices(packet, 1280, 720, frame=1165)


def _windowed_stream():
    """A regular stream of 1 ms windows over 20 ms, with empty windows."""
    import faery

    generator = numpy.random.default_rng(7)
    events = numpy.zeros(2_000, dtype=EVENTS_DTYPE)
    t = numpy.sort(generator.integers(0, 20_000, len(events)))
    t[0] = 0  # regularize starts the first window at the first event
    t = t[(t < 5_000) | (t >= 8_000)]  # windows 5, 6 and 7 are empty
    events = events[: len(t)]
    events["t"] = t
    events["x"] = generator.integers(0, 10, len(events))
    events["y"] = generator.integers(0, 8, len(events))
    events["p"] = generator.integers(0, 2, len(events)).astype(bool)
    return (
        faery.events_stream_from_array(events, dimensions=(10, 8))
        .chunks(97)
        .regularize(frequency_hz=1000.0)
    )


def _frames_from_batches(batches):
    frames = []
    for indices, windows in batches:
        assert indices.dtype == numpy.int32
        counts = numpy.bincount(indices, minlength=windows * 2 * 10 * 8)
        assert len(counts) == windows * 2 * 10 * 8
        frames.extend(counts.reshape(windows, 2, 8, 10))
    return frames


@pytest.mark.parametrize("windows_per_batch", [1, 3, 7, 100])
def test_windows_per_batch_matches_frames(windows_per_batch):
    stream = _windowed_stream()
    expected = [frame.astype(numpy.int64) for frame in stream.to_dlpack_frame(dtype="u32")]
    batches = list(stream.to_dlpack_indices(windows_per_batch=windows_per_batch))
    assert [windows for _, windows in batches[:-1]] == [windows_per_batch] * (len(batches) - 1)
    frames = _frames_from_batches(batches)
    assert len(frames) == len(expected)
    assert sum(frame.sum() for frame in frames[5:8]) == 0
    for frame, reference in zip(frames, expected):
        numpy.testing.assert_array_equal(frame, reference)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_windows_per_batch_out_buffers(prefetch):
    stream = _windowed_stream()
    expected = _frames_from_batches(stream.to_dlpack_indices(windows_per_batch=4))
    out = [numpy.empty(4_000, dtype=numpy.int32) for _ in range(prefetch + 2)]
    batches = stream.to_dlpack_indices(windows_per_batch=4, out=out, prefetch=prefetch)
    # Count each batch before the generator may reuse its buffer.
    frames = []
    for batch in batches:
        frames.extend(_frames_from_batches([batch]))
    assert len(frames) == len(expected)
    for frame, reference in zip(frames, expected):
        numpy.testing.assert_array_equal(frame, reference)


def test_windows_per_batch_refusals():
    stream = _FixedStream(_batch_packets(), dimensions=(10, 8))
    with pytest.raises(ValueError, match="regular"):
        next(stream.to_dlpack_indices(windows_per_batch=2))
    regular = _windowed_stream()
    with pytest.raises(ValueError, match="exclusive"):
        next(regular.to_dlpack_indices(batch_events=4, windows_per_batch=2))
    with pytest.raises(ValueError, match="at least 1"):
        next(regular.to_dlpack_indices(windows_per_batch=0))
