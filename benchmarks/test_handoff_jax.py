"""Benchmarks: event packets -> (2, H, W) float32 frame on the GPU, in JAX.

The JAX counterpart of the gpu group in test_handoff.py, with the same
workloads and round structure: a round replays every packet and ends with
block_until_ready on the last frame (JAX runs device work in issue order).

JAX compiles a jitted function once per input shape, and packets have
variable lengths. Scattering an exact-length index array recompiles on
almost every packet (~58 ms per packet at 10k events on an RTX 3090, against
~1 ms with a cached compilation). The scatter variants therefore pad each
packet to a power-of-two bucket (at least MIN_BUCKET) with an out-of-range
sentinel that `mode="drop"` discards: at most 2x padding, and a few
compilations per workload, all during warmup.

Run through benchmarks/run.sh, which saves results for later comparison.
"""

import os

import numpy
import pytest

# JAX grabs 75% of GPU memory at startup by default, which starves PyTorch
# when both benchmark files run in one session.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


def _gpu():
    if jax is None:
        return None
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


GPU = _gpu()
pytestmark = pytest.mark.skipif(GPU is None, reason="requires JAX with a GPU")
MIN_BUCKET = 1024


def _bucket(length):
    return max(MIN_BUCKET, 1 << (length - 1).bit_length())


def _pad(array, fill):
    padded = numpy.full(_bucket(len(array)), fill, dtype=array.dtype)
    padded[: len(array)] = array
    return padded


def _round(benchmark, workload, run):
    benchmark.extra_info["packets"] = len(workload.packets)
    benchmark.extra_info["events"] = workload.events
    benchmark.extra_info["dimensions"] = list(workload.dimensions)
    benchmark.pedantic(run, rounds=10, warmup_rounds=2, iterations=1)


@jax.jit if jax is not None else (lambda f: f)
def _to_float(frame):
    return frame.astype(jnp.float32)


def _indices_scatter(width, height):
    size = 2 * height * width

    @jax.jit
    def scatter(indices):
        # Padding indices equal `size` and are dropped.
        flat = jnp.zeros(size, jnp.float32).at[indices].add(1.0, mode="drop")
        return flat.reshape(2, height, width)

    return scatter


def _sparse_scatter(width, height):
    @jax.jit
    def scatter(x, y, p):
        # Padding x values equal `width` and are dropped.
        frame = jnp.zeros((2, height, width), jnp.float32)
        return frame.at[p.astype(jnp.int32), y, x].add(1.0, mode="drop")

    return scatter


@pytest.mark.benchmark(group="jax")
def test_jax_frame(benchmark, workload):
    """Rasterize u16 on CPU, device_put, convert to float32 on the device."""
    stream = workload.stream()

    def run():
        frame = None
        for packet in stream.to_dlpack_frame(dtype="u16"):
            frame = _to_float(jax.device_put(packet, GPU))
        if frame is not None:
            frame.block_until_ready()

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="jax")
def test_jax_frame_dlpack(benchmark, workload):
    """As jax_frame, but importing through jax.dlpack.from_dlpack first.

    from_dlpack wraps the numpy frame as a CPU array (zero-copy); it still
    has to be moved to the GPU explicitly.
    """
    stream = workload.stream()

    def run():
        frame = None
        for packet in stream.to_dlpack_frame(dtype="u16"):
            cpu = jax.dlpack.from_dlpack(packet)
            frame = _to_float(jax.device_put(cpu, GPU))
        if frame is not None:
            frame.block_until_ready()

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="jax")
def test_jax_indices_bucketed(benchmark, workload):
    """One int32 index per event, padded to a bucket, jitted .at[].add."""
    stream = workload.stream()
    width, height = workload.dimensions
    scatter = _indices_scatter(width, height)
    sentinel = 2 * height * width

    def run():
        frame = None
        for indices in stream.to_dlpack_indices():
            frame = scatter(jax.device_put(_pad(indices, sentinel), GPU))
        if frame is not None:
            frame.block_until_ready()

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="jax")
def test_jax_sparse_bucketed(benchmark, workload):
    """x/y/p uploaded separately, padded to a bucket, jitted .at[p, y, x].add."""
    stream = workload.stream()
    width, height = workload.dimensions
    scatter = _sparse_scatter(width, height)

    def run():
        frame = None
        for fields in stream.to_dlpack_sparse(fields=("x", "y", "p")):
            x = jax.device_put(_pad(fields["x"], width), GPU)
            y = jax.device_put(_pad(fields["y"], 0), GPU)
            p = jax.device_put(_pad(fields["p"], 0), GPU)
            frame = scatter(x, y, p)
        if frame is not None:
            frame.block_until_ready()

    _round(benchmark, workload, run)


def test_jax_variants_agree(workload):
    """Guard: every jax variant must produce the same frame as numpy."""
    width, height = workload.dimensions
    packet = workload.packets[len(workload.packets) // 2]
    expected = numpy.zeros((2, height, width), dtype=numpy.float32)
    numpy.add.at(
        expected,
        (packet["on"].astype(numpy.intp), packet["y"], packet["x"]),
        1.0,
    )
    stream = type(workload.stream())([packet], workload.dimensions)

    frame = next(iter(stream.to_dlpack_frame(dtype="u16")))
    gpu = _to_float(jax.device_put(frame, GPU))
    assert gpu.devices() == {GPU}
    numpy.testing.assert_array_equal(numpy.asarray(gpu), expected)
    gpu = _to_float(jax.device_put(jax.dlpack.from_dlpack(frame), GPU))
    numpy.testing.assert_array_equal(numpy.asarray(gpu), expected)

    indices = next(iter(stream.to_dlpack_indices()))
    gpu = _indices_scatter(width, height)(
        jax.device_put(_pad(indices, 2 * height * width), GPU)
    )
    numpy.testing.assert_array_equal(numpy.asarray(gpu), expected)

    fields = next(iter(stream.to_dlpack_sparse(fields=("x", "y", "p"))))
    gpu = _sparse_scatter(width, height)(
        jax.device_put(_pad(fields["x"], width), GPU),
        jax.device_put(_pad(fields["y"], 0), GPU),
        jax.device_put(_pad(fields["p"], 0), GPU),
    )
    numpy.testing.assert_array_equal(numpy.asarray(gpu), expected)


def test_jax_empty_packet(workload):
    """An empty packet pads to a bucket of sentinels and scatters to zeros."""
    width, height = workload.dimensions
    indices = numpy.empty(0, dtype=numpy.int32)
    gpu = _indices_scatter(width, height)(
        jax.device_put(_pad(indices, 2 * height * width), GPU)
    )
    assert not numpy.asarray(gpu).any()
