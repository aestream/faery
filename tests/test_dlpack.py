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
