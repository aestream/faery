import numpy
import pytest

import faery
from faery.events_stream import EVENTS_DTYPE, concatenate_events
from faery.extension import array


def _events(length, seed):
    generator = numpy.random.default_rng(seed)
    events = numpy.zeros(length, dtype=EVENTS_DTYPE)
    events["t"] = numpy.sort(generator.integers(0, 1 << 40, length))
    events["x"] = generator.integers(0, 1280, length)
    events["y"] = generator.integers(0, 720, length)
    events["p"] = generator.integers(0, 2, length).astype(bool)
    return events


def _assert_same(result, expected):
    assert result.dtype == EVENTS_DTYPE
    assert result.flags.c_contiguous and result.flags.owndata
    assert numpy.array_equal(result, expected)


@pytest.mark.parametrize("lengths", [[], [0], [5], [0, 0], [3, 0, 8192, 1, 7]])
def test_concatenate_matches_numpy(lengths):
    parts = [_events(length, seed) for seed, length in enumerate(lengths)]
    result = array.concatenate(parts, EVENTS_DTYPE)
    assert result is not None
    if parts:
        _assert_same(result, numpy.concatenate(parts))
    else:
        _assert_same(result, numpy.array([], dtype=EVENTS_DTYPE))


def test_concatenate_copies():
    part = _events(10, 0)
    result = array.concatenate([part], EVENTS_DTYPE)
    assert result is not None
    assert not numpy.shares_memory(result, part)


def test_concatenate_strided_views():
    events = _events(100, 1)
    parts = [events[::3], events[50:10:-1], events[5:6], events[90:]]
    result = array.concatenate(parts, EVENTS_DTYPE)
    assert result is not None
    _assert_same(result, numpy.concatenate(parts))


@pytest.mark.parametrize(
    "part",
    [
        numpy.zeros(4, dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")]),
        numpy.zeros(4, dtype=numpy.dtype(EVENTS_DTYPE.descr, align=True)),
        numpy.zeros(4, dtype=EVENTS_DTYPE.newbyteorder()),
        numpy.zeros((2, 2), dtype=EVENTS_DTYPE),
        [(0, 0, 0, False)],
    ],
    ids=["names", "aligned", "byteorder", "2d", "list"],
)
def test_concatenate_declines_other_parts(part):
    assert array.concatenate([_events(3, 0), part], EVENTS_DTYPE) is None


def test_concatenate_declines_object_dtypes():
    parts = [numpy.array([1, "a"], dtype=object)]
    assert array.concatenate(parts, numpy.dtype(object)) is None


def test_concatenate_events_falls_back_to_numpy():
    other = numpy.zeros(2, dtype=numpy.dtype(EVENTS_DTYPE.descr, align=True))
    other["t"] = [7, 8]
    result = concatenate_events([_events(3, 0), other])
    _assert_same(result, numpy.concatenate([_events(3, 0), other.astype(EVENTS_DTYPE)]))


def test_regularize_windows_match_numpy():
    events = _events(50_000, 2)
    events["t"] = numpy.arange(len(events)) * 3
    # Packets of 1,351 events, so most 1 ms windows (333 events) sit inside
    # a packet and the rest span two.
    stream = faery.events_stream_from_array(events, dimensions=(1280, 720)).chunks(1351)
    windows = list(stream.regularize(frequency_hz=1000.0))
    assert all(window.dtype == EVENTS_DTYPE for window in windows)
    _assert_same(numpy.concatenate(windows), events)
    period_us = 1000
    for index, window in enumerate(windows):
        assert numpy.all(window["t"] // period_us == index)


def test_chunks_match_numpy():
    events = _events(10_000, 3)
    stream = faery.events_stream_from_array(events, dimensions=(1280, 720))
    chunks = list(stream.chunks(997).chunks(4096))
    assert [len(chunk) for chunk in chunks] == [4096, 4096, 1808]
    _assert_same(numpy.concatenate(chunks), events)
