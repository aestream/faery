"""Benchmarks: reading a recording from a file.

Unlike test_handoff.py, decoding is the timed work here. Each round reads a
whole recording; divide by `extra_info["events"]` for a per-event figure.

Groups:
- decode: events_stream_from_file iterated to the end (the file's own packets).
- regularize: the same, sliced into 60 Hz packets.
- time_range: FileDecoder.time_range() with no cache, i.e. what a finite stream
  costs to describe before any data is read.
- cold: the first regularize in a fresh process (empty time range cache).

Recordings come from BENCH_DATA (a directory mounted by benchmarks/run.sh) and
are skipped when it is unset, since they are too large for the repository.
"""

import os
import pathlib

import pytest

import faery
from faery import file_decoder

FPS = 60.0
RECORDINGS = ("driving_sample.dat", "driving_sample.aedat4")
DATA = os.environ.get("BENCH_DATA")

recording = pytest.mark.parametrize(
    "workload",
    [
        pytest.param(
            pathlib.Path(DATA or "") / name,
            id=name,
            marks=pytest.mark.skipif(
                DATA is None or not (pathlib.Path(DATA) / name).exists(),
                reason=f"{name} not found (set BENCH_DATA)",
            ),
        )
        for name in RECORDINGS
    ],
)


def _round(benchmark, path, run, rounds=5):
    benchmark.extra_info["file"] = path.name
    benchmark.extra_info["bytes"] = path.stat().st_size
    events = benchmark.pedantic(run, rounds=rounds, warmup_rounds=1, iterations=1)
    if events is not None:
        benchmark.extra_info["events"] = events


def _count(packets) -> int:
    return sum(len(packet) for packet in packets)


@pytest.mark.benchmark(group="decode")
@recording
def test_decode(benchmark, workload):
    path = workload
    _round(benchmark, path, lambda: _count(faery.events_stream_from_file(path)))


@pytest.mark.benchmark(group="regularize")
@recording
def test_regularize(benchmark, workload):
    path = workload
    _round(
        benchmark,
        path,
        lambda: _count(faery.events_stream_from_file(path).regularize(frequency_hz=FPS)),
    )


@pytest.mark.benchmark(group="time_range")
@recording
def test_time_range(benchmark, workload):
    path = workload
    _round(
        benchmark,
        path,
        lambda: file_decoder.Decoder(path, time_range_cache=None).time_range() and None,
    )


@pytest.mark.benchmark(group="cold")
@recording
def test_regularize_cold(benchmark, workload):
    path = workload
    def run():
        decoder = file_decoder.Decoder(path, time_range_cache=file_decoder.TimeRangeCache())
        return _count(decoder.regularize(frequency_hz=FPS))

    _round(benchmark, path, run)
