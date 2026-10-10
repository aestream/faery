from __future__ import annotations

import numpy
import pytest

import faery

from . import assets

FAST_TIME_RANGE_FORMATS = ("aedat4", "dat2")


def decoded_time_range(path) -> tuple[faery.Time, faery.Time]:
    """The reference: first and last events of a full decode."""
    start = None
    end = None
    for events in faery.file_decoder.Decoder(path, time_range_cache=None):
        if len(events) > 0:
            if start is None:
                start = int(events["t"][0])
            end = int(events["t"][-1])
    assert start is not None and end is not None
    return (
        faery.Time(microseconds=start),
        faery.Time(microseconds=end + 1),
    )


@pytest.mark.parametrize(
    "file",
    [file for file in assets.files if file.format in FAST_TIME_RANGE_FORMATS],
)
def test_time_range_without_decoding(file: assets.File):
    decoder = faery.file_decoder.Decoder(file.path, time_range_cache=None)
    start, end = decoder._time_range_without_decoding()
    assert start is not None and end is not None, "fast path not taken"
    assert decoder.time_range() == decoded_time_range(file.path)


@pytest.mark.parametrize(
    "file", [file for file in assets.files if file.format == "dat2"]
)
def test_dat_as_events(file: assets.File):
    with (
        faery.dat.Decoder(file.path, None, None) as raw,
        faery.dat.Decoder(file.path, None, None, as_events=True) as converted,
    ):
        for raw_events, events in zip(raw, converted, strict=True):
            assert events.dtype == faery.EVENTS_DTYPE
            numpy.clip(raw_events["payload"], 0, 1, raw_events["payload"])
            expected = raw_events.astype(faery.EVENTS_DTYPE, casting="unsafe")
            assert numpy.array_equal(events, expected)


def test_full_iteration_fills_the_cache():
    file = next(file for file in assets.files if file.format == "dat2")
    cache = faery.file_decoder.TimeRangeCache()
    decoder = faery.file_decoder.Decoder(file.path, time_range_cache=cache)
    for _ in decoder:
        pass
    path_hash = cache.path_hash(path=decoder.path)
    assert cache.get_time_range(path=decoder.path, path_hash=path_hash) == (
        decoded_time_range(file.path)
    )
