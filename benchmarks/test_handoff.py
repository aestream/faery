"""Benchmarks: event packets -> (2, H, W) float32 frame on the GPU.

One benchmark round replays every packet of a workload and ends with a CUDA
synchronize, so a round's time is the end-to-end cost for that workload.
Divide by `extra_info["packets"]` for a per-packet figure.

Groups:
- cpu: the faery-side preparation alone (Rust rasterization, field copies).
  Runs without a GPU and isolates changes to src/dlpack.rs.
- gpu: preparation + host-to-device transfer + GPU work, until the frame is
  ready on the device.

Run through benchmarks/run.sh, which saves results for later comparison.
"""

import numpy
import pytest

try:
    import torch
except ImportError:
    torch = None

CUDA = torch is not None and torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="requires PyTorch with CUDA")
DEVICE = "cuda:0"


def _round(benchmark, workload, run):
    benchmark.extra_info["packets"] = len(workload.packets)
    benchmark.extra_info["events"] = workload.events
    benchmark.extra_info["dimensions"] = list(workload.dimensions)
    benchmark.pedantic(run, rounds=10, warmup_rounds=2, iterations=1)


# --- cpu --------------------------------------------------------------------


@pytest.mark.benchmark(group="cpu")
@pytest.mark.parametrize("dtype", ["u16", "f32"])
def test_cpu_frame(benchmark, workload, dtype):
    stream = workload.stream()

    def run():
        for _ in stream.to_dlpack_frame(dtype=dtype):
            pass

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="cpu")
@pytest.mark.parametrize("dtype", ["u16", "f32"])
def test_cpu_frame_out(benchmark, workload, dtype):
    """Rasterize into one reused buffer instead of allocating per packet."""
    stream = workload.stream()
    width, height = workload.dimensions
    numpy_dtype = {"u16": numpy.uint16, "f32": numpy.float32}[dtype]
    out = numpy.empty((2, height, width), dtype=numpy_dtype)

    def run():
        for _ in stream.to_dlpack_frame(dtype=dtype, out=out):
            pass

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="cpu")
def test_cpu_indices(benchmark, workload):
    stream = workload.stream()

    def run():
        for _ in stream.to_dlpack_indices():
            pass

    _round(benchmark, workload, run)


@pytest.mark.benchmark(group="cpu")
def test_cpu_sparse(benchmark, workload):
    stream = workload.stream()

    def run():
        for _ in stream.to_dlpack_sparse(fields=("x", "y", "p")):
            pass

    _round(benchmark, workload, run)


# --- gpu --------------------------------------------------------------------


@requires_cuda
@pytest.mark.benchmark(group="gpu")
@pytest.mark.parametrize("dtype", ["u16", "f32"])
def test_gpu_frame(benchmark, workload, dtype):
    """Rasterize on CPU, upload the frame, convert to float32 on the device."""
    stream = workload.stream()

    def run():
        for frame in stream.to_dlpack_frame(dtype=dtype):
            gpu = torch.from_dlpack(frame).to(DEVICE, non_blocking=True)
            if gpu.dtype != torch.float32:
                gpu = gpu.float()
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
@pytest.mark.benchmark(group="gpu")
def test_gpu_frame_out(benchmark, workload):
    """u16 into one reused pageable buffer, synchronous upload, float on device."""
    stream = workload.stream()
    width, height = workload.dimensions
    out = numpy.empty((2, height, width), dtype=numpy.uint16)

    def run():
        for frame in stream.to_dlpack_frame(out=out):
            gpu = torch.from_dlpack(frame).to(DEVICE).float()
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
@pytest.mark.benchmark(group="gpu")
def test_gpu_frame_pinned(benchmark, workload):
    """u16 rasterized straight into two pinned buffers in turn; async uploads.

    Rust writes packet N+1 into one pinned buffer while packet N's upload
    from the other is in flight. Buffers are staged as int16 (torch's uint16
    support is limited; exact below 32768 counts per pixel).
    """
    stream = workload.stream()
    width, height = workload.dimensions
    shape = (2, height, width)
    pinned = [torch.empty(shape, dtype=torch.int16, pin_memory=True) for _ in range(2)]
    staged = [torch.empty(shape, dtype=torch.int16, device=DEVICE) for _ in range(2)]
    uploaded = [torch.cuda.Event() for _ in range(2)]
    out = [buffer.numpy().view(numpy.uint16) for buffer in pinned]

    def run():
        for index, _ in enumerate(stream.to_dlpack_frame(out=out)):
            slot = index % 2
            staged[slot].copy_(pinned[slot], non_blocking=True)
            uploaded[slot].record()
            frame = staged[slot].float()
            # The next iteration rasterizes into the other buffer: wait for
            # the upload that is still reading it (issued last iteration).
            uploaded[1 - slot].synchronize()
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
@pytest.mark.benchmark(group="gpu")
def test_gpu_sparse_index_put(benchmark, workload):
    """Upload x/y/p separately, scatter with index_put_ (examples/ pattern)."""
    stream = workload.stream()
    width, height = workload.dimensions

    def run():
        for fields in stream.to_dlpack_sparse(fields=("x", "y", "p")):
            x = torch.from_dlpack(fields["x"]).to(DEVICE, non_blocking=True).long()
            y = torch.from_dlpack(fields["y"]).to(DEVICE, non_blocking=True).long()
            p = torch.from_dlpack(fields["p"]).to(DEVICE, non_blocking=True).long()
            frame = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
            frame.index_put_(
                (p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True
            )
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
@pytest.mark.benchmark(group="gpu")
def test_gpu_indices_bincount(benchmark, workload):
    """One int32 index per event, one upload, bincount on the device."""
    stream = workload.stream()
    width, height = workload.dimensions

    def run():
        for indices in stream.to_dlpack_indices():
            gpu = torch.from_dlpack(indices).to(DEVICE, non_blocking=True)
            frame = torch.bincount(gpu, minlength=2 * height * width)
            frame = frame.view(2, height, width).float()
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
@pytest.mark.benchmark(group="gpu")
def test_gpu_indices_index_add(benchmark, workload):
    """One int32 index per event, one upload, index_add_ into a flat frame."""
    stream = workload.stream()
    width, height = workload.dimensions
    ones = torch.ones(max(map(len, workload.packets), default=0), device=DEVICE)

    def run():
        for indices in stream.to_dlpack_indices():
            gpu = torch.from_dlpack(indices).to(DEVICE, non_blocking=True)
            frame = torch.zeros(2 * height * width, device=DEVICE)
            frame.index_add_(0, gpu, ones[: len(gpu)])
            frame = frame.view(2, height, width)
        torch.cuda.synchronize()

    _round(benchmark, workload, run)


@requires_cuda
def test_gpu_variants_agree(workload):
    """Guard: every gpu variant must produce the same frame as numpy."""
    width, height = workload.dimensions
    packet = workload.packets[len(workload.packets) // 2]
    expected = numpy.zeros((2, height, width), dtype=numpy.float32)
    numpy.add.at(
        expected,
        (packet["on"].astype(numpy.intp), packet["y"], packet["x"]),
        1.0,
    )
    stream = type(workload.stream())([packet], workload.dimensions)

    frame = next(iter(stream.to_dlpack_frame(dtype="f32")))
    numpy.testing.assert_array_equal(frame, expected)

    pinned = torch.empty((2, height, width), dtype=torch.int16, pin_memory=True)
    out = pinned.numpy().view(numpy.uint16)
    next(iter(stream.to_dlpack_frame(out=out)))
    staged = torch.empty((2, height, width), dtype=torch.int16, device=DEVICE)
    staged.copy_(pinned, non_blocking=True)
    numpy.testing.assert_array_equal(staged.float().cpu().numpy(), expected)

    fields = next(iter(stream.to_dlpack_sparse(fields=("x", "y", "p"))))
    x = torch.from_dlpack(fields["x"]).to(DEVICE).long()
    y = torch.from_dlpack(fields["y"]).to(DEVICE).long()
    p = torch.from_dlpack(fields["p"]).to(DEVICE).long()
    gpu = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
    gpu.index_put_((p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True)
    numpy.testing.assert_array_equal(gpu.cpu().numpy(), expected)

    indices = torch.from_dlpack(next(iter(stream.to_dlpack_indices()))).to(DEVICE)
    counts = torch.bincount(indices, minlength=2 * height * width)
    numpy.testing.assert_array_equal(
        counts.view(2, height, width).float().cpu().numpy(), expected
    )
    flat = torch.zeros(2 * height * width, device=DEVICE)
    flat.index_add_(0, indices, torch.ones(len(indices), device=DEVICE))
    numpy.testing.assert_array_equal(flat.view(2, height, width).cpu().numpy(), expected)
