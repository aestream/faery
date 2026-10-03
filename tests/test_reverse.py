"""reverse: time reversal of a finite events stream.

Expected semantics (paper, "Bounding memory and reversing time"):
- reverse: S(E)_fin -> S(E)_fin. It exists on finite streams only.
- The output holds the same events in reverse order, with timestamps mirrored
  within the stream's time range, so time still increases along the stream:
  an event at t becomes start + (end - 1) - t, where [start, end) is
  time_range(). The time range itself is unchanged.
- Reversing time turns a brightness increase into a decrease, so polarity flips.
- It applies to any finite stream (files, filters, arrays) and composes with
  the rest of the algebra; reverse . reverse is the identity.
"""

from __future__ import annotations

import numpy
import pytest

import faery

from . import assets

DIMENSIONS = (10, 8)


def _events() -> numpy.ndarray:
    events = numpy.zeros(6, dtype=faery.EVENTS_DTYPE)
    events["t"] = [100, 105, 105, 130, 160, 199]
    events["x"] = [1, 2, 3, 4, 5, 9]
    events["y"] = [0, 1, 2, 3, 4, 7]
    events["on"] = [True, False, True, True, False, False]
    return events


def _collect(stream) -> numpy.ndarray:
    packets = [packet for packet in stream if len(packet) > 0]
    if len(packets) == 0:
        return numpy.array([], dtype=faery.EVENTS_DTYPE)
    return numpy.concatenate(packets)


def _expected_reverse(events, time_range) -> numpy.ndarray:
    start, end = (time.to_microseconds() for time in time_range)
    expected = events[::-1].copy()
    expected["t"] = start + (end - 1) - expected["t"]
    expected["on"] = ~expected["on"]
    return expected


def _streams():
    """Finite streams of different kinds, all with several packets."""
    array = faery.events_stream_from_array(_events(), dimensions=DIMENSIONS)
    yield pytest.param(array, id="array")
    yield pytest.param(array.chunks(4), id="array.chunks")
    yield pytest.param(array.chunks(1), id="one-event packets")
    yield pytest.param(array.remove_off_events(), id="filter")
    file = next(file for file in assets.files if file.format == "es-dvs")
    yield pytest.param(faery.events_stream_from_file(file.path), id="file")


@pytest.mark.parametrize("stream", _streams())
def test_reverse_mirrors_events_in_time(stream):
    events = _collect(stream)
    reversed_stream = stream.reverse()
    assert reversed_stream.time_range() == stream.time_range()
    assert reversed_stream.dimensions() == stream.dimensions()
    numpy.testing.assert_array_equal(
        _collect(reversed_stream), _expected_reverse(events, stream.time_range())
    )


@pytest.mark.parametrize("stream", _streams())
def test_reversed_time_increases(stream):
    t = _collect(stream.reverse())["t"].astype(numpy.int64)
    assert numpy.all(numpy.diff(t) >= 0)


@pytest.mark.parametrize("stream", _streams())
def test_reverse_is_an_involution(stream):
    numpy.testing.assert_array_equal(
        _collect(stream.reverse().reverse()), _collect(stream)
    )


def test_reverse_is_finite_and_composes():
    stream = faery.events_stream_from_array(_events(), dimensions=DIMENSIONS)
    reversed_stream = stream.reverse()
    assert isinstance(reversed_stream, faery.FiniteEventsStream)
    # Downstream operations see an ordinary finite stream.
    assert len(reversed_stream.to_array()) == len(_events())
    regular = reversed_stream.regularize(frequency_hz=1e5)
    assert sum(len(packet) for packet in regular) == len(_events())


def test_reverse_requires_a_finite_stream():
    stream = faery.events_stream_from_array(_events(), dimensions=DIMENSIONS)
    infinite_like = faery.EventsStream
    assert not hasattr(infinite_like, "reverse")
    assert hasattr(type(stream), "reverse")


def test_slice_and_reverse_do_not_commute():
    # Paper: slice . reverse != reverse . slice. Slicing the first 31 us keeps
    # the start of the recording; reversed first, it keeps the end.
    stream = faery.events_stream_from_array(_events(), dimensions=DIMENSIONS)
    window = (faery.Time(microseconds=100), faery.Time(microseconds=131))
    slice_then_reverse = _collect(stream.time_slice(*window).reverse())
    reverse_then_slice = _collect(stream.reverse().time_slice(*window))
    assert len(slice_then_reverse) > 0 and len(reverse_then_slice) > 0
    assert not numpy.array_equal(slice_then_reverse, reverse_then_slice)
