import numpy

import faery
from faery.events_stream import EVENTS_DTYPE


def _events(length):
    generator = numpy.random.default_rng(5)
    events = numpy.zeros(length, dtype=EVENTS_DTYPE)
    events["t"] = numpy.sort(generator.integers(0, 2_000_000, length))
    events["x"] = generator.integers(0, 64, length)
    events["y"] = generator.integers(0, 48, length)
    events["p"] = generator.integers(0, 2, length).astype(bool)
    return events


def test_regularize_one_large_packet_matches_small_packets():
    events = _events(200_000)
    stream = faery.events_stream_from_array(events, dimensions=(64, 48))
    whole = list(stream.regularize(frequency_hz=1000.0))
    chunked = list(stream.chunks(8192).regularize(frequency_hz=1000.0))
    assert len(whole) == len(chunked)
    for a, b in zip(whole, chunked):
        numpy.testing.assert_array_equal(a, b)


def test_regularize_searchsorted_needs_no_copy(monkeypatch):
    # Regression: searchsorted on the strided events["t"] view copied the
    # whole column on every window boundary, so regularizing one large packet
    # took 19 s instead of 0.3 s. Every call must get a contiguous uint64
    # column and a uint64 key.
    keys = []
    searchsorted = numpy.searchsorted

    def recording(array, value, *args, **kwargs):
        assert array.flags.c_contiguous and array.dtype == numpy.uint64
        keys.append(value)
        return searchsorted(array, value, *args, **kwargs)

    monkeypatch.setattr(numpy, "searchsorted", recording)
    stream = faery.events_stream_from_array(_events(10_000), dimensions=(64, 48))
    windows = list(stream.regularize(frequency_hz=1000.0))
    assert sum(len(window) for window in windows) == 10_000
    assert len(keys) > 100
    assert all(isinstance(key, numpy.uint64) for key in keys)
