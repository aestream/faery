"""Shared workloads for the GPU handoff benchmarks.

Every workload is a list of decoded event packets (structured arrays with
faery.EVENTS_DTYPE) plus sensor dimensions. Decoding happens once, at
collection time, so the timed sections cover only what this branch changes:
packet -> (Rust or numpy prep) -> host-to-device transfer -> GPU frame.

Workloads:
- dvs.es: the real 320x240 recording from tests/data, regularized at 60 Hz.
- synthetic-<N>: 1280x720 packets with N uniformly distributed events each,
  spanning sparse to dense activity so the sparse/dense crossover is visible.
"""

import dataclasses
import pathlib

import numpy
import pytest

import faery
from faery.events_stream import EVENTS_DTYPE, Output

ROOT = pathlib.Path(__file__).resolve().parent.parent
FPS = 60.0
SYNTHETIC_DIMENSIONS = (1280, 720)
SYNTHETIC_PACKETS = 30
SYNTHETIC_EVENTS_PER_PACKET = (1_000, 10_000, 100_000, 1_000_000)


@dataclasses.dataclass
class Workload:
    name: str
    packets: list[numpy.ndarray]
    dimensions: tuple[int, int]

    @property
    def events(self) -> int:
        return sum(len(packet) for packet in self.packets)

    def stream(self) -> "CachedStream":
        return CachedStream(self.packets, self.dimensions)


class CachedStream(Output):
    """Replays cached packets through faery's public Output methods."""

    def __init__(self, packets, dimensions):
        self._packets = packets
        self._dimensions = dimensions

    def __iter__(self):
        return iter(self._packets)

    def dimensions(self):
        return self._dimensions


def _recording() -> Workload:
    stream = faery.events_stream_from_file(ROOT / "tests" / "data" / "dvs.es")
    stream = stream.regularize(frequency_hz=FPS)
    return Workload("dvs.es", list(stream), stream.dimensions())


def _synthetic(events_per_packet: int) -> Workload:
    width, height = SYNTHETIC_DIMENSIONS
    rng = numpy.random.default_rng(events_per_packet)
    period_us = int(1e6 / FPS)
    packets = []
    for index in range(SYNTHETIC_PACKETS):
        packet = numpy.empty(events_per_packet, dtype=EVENTS_DTYPE)
        packet["t"] = numpy.sort(
            rng.integers(index * period_us, (index + 1) * period_us, events_per_packet)
        )
        packet["x"] = rng.integers(0, width, events_per_packet)
        packet["y"] = rng.integers(0, height, events_per_packet)
        packet["on"] = rng.integers(0, 2, events_per_packet).astype(bool)
        packets.append(packet)
    return Workload(f"synthetic-{events_per_packet}", packets, SYNTHETIC_DIMENSIONS)


def pytest_benchmark_generate_commit_info(config):
    """Commit info from benchmarks/run.sh (git is not in the container)."""
    import os

    return {
        "id": os.environ.get("BENCH_COMMIT", "unknown"),
        "branch": os.environ.get("BENCH_BRANCH", "unknown"),
        "dirty": os.environ.get("BENCH_DIRTY", "") == "1",
        "project": "faery",
    }


def pytest_benchmark_update_machine_info(config, machine_info):
    try:
        import torch
    except ImportError:
        return
    machine_info["torch"] = torch.__version__
    if torch.cuda.is_available():
        machine_info["gpu"] = torch.cuda.get_device_name(0)
        machine_info["cuda"] = torch.version.cuda


WORKLOAD_NAMES =["dvs.es"] + [f"synthetic-{n}" for n in SYNTHETIC_EVENTS_PER_PACKET]
_cache: dict[str, Workload] = {}


@pytest.fixture(params=WORKLOAD_NAMES)
def workload(request) -> Workload:
    name = request.param
    if name not in _cache:
        if name == "dvs.es":
            _cache[name] = _recording()
        else:
            _cache[name] = _synthetic(int(name.split("-")[1]))
    return _cache[name]
