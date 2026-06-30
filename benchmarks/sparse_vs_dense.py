"""Sparse vs dense: CPU rasterize + GPU upload, vs raw-events upload + GPU scatter.

Two paths that both produce a `(2, H, W)` polarity-split count frame on a GPU
tensor — the difference is *who* runs the scatter and what crosses PCIe:

- Dense (CPU rasterize -> GPU upload):
    faery.to_dlpack_frame() runs the scatter in Rust on CPU.
    torch.from_dlpack(frame).cuda(non_blocking=True) uploads the small frame.
    PCIe payload per packet: ~ 2*H*W*sizeof(dtype) bytes.

- Sparse (events upload -> GPU scatter):
    faery.to_dlpack_sparse() hands per-packet {t,x,y,p} fields.
    Each field is uploaded to CUDA and torch.Tensor.index_put_ scatters on GPU.
    PCIe payload per packet: ~ N_events * (8 + 2 + 2 + 1) bytes.

At sparse-typical event density the sparse path moves less data over PCIe
and should win; at dense density (above the crossover) the fixed-size frame
becomes smaller than the events array and the dense path should win.

Requires: PyTorch with CUDA. Skips gracefully otherwise.
"""

import pathlib
import sys
import time

import faery

try:
    import torch
except ImportError:
    print("This benchmark requires PyTorch. Install with: pip install torch")
    sys.exit(0)

if not torch.cuda.is_available():
    print("This benchmark requires CUDA. Falling back to CPU would defeat the point.")
    sys.exit(0)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH = ROOT / "tests" / "data" / "dvs.es"
FPS = 60.0
REPEATS = 5
DEVICE = torch.device("cuda")


def time_dense_upload() -> tuple[float, int]:
    stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=FPS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for frame_np in stream.to_dlpack_frame(dtype="u16"):
        _gpu = torch.from_dlpack(frame_np).to(DEVICE, non_blocking=True)
        n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def time_sparse_scatter() -> tuple[float, int]:
    stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=FPS)
    width, height = stream.dimensions()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for fields in stream.to_dlpack_sparse():
        x = torch.from_dlpack(fields["x"]).to(DEVICE, non_blocking=True).long()
        y = torch.from_dlpack(fields["y"]).to(DEVICE, non_blocking=True).long()
        p = torch.from_dlpack(fields["p"]).to(DEVICE, non_blocking=True).long()
        frame = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
        frame.index_put_((p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True)
        n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def count_events_and_dims() -> tuple[int, tuple[int, int]]:
    stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=FPS)
    dims = stream.dimensions()
    n = sum(len(packet) for packet in stream)
    return n, dims


def estimate_pcie_bytes(n_events: int, n_frames: int, dims: tuple[int, int]) -> tuple[int, int]:
    width, height = dims
    # Dense: one (2, H, W) uint16 frame per packet.
    dense_bytes = n_frames * 2 * height * width * 2
    # Sparse: 13 bytes per event (u64 t + u16 x + u16 y + u8 p).
    sparse_bytes = n_events * (8 + 2 + 2 + 1)
    return dense_bytes, sparse_bytes


def report(name: str, fn) -> None:
    fn()  # warm-up
    times = [fn()[0] for _ in range(REPEATS)]
    _, n_frames = fn()
    best = min(times) * 1000
    median = sorted(times)[len(times) // 2] * 1000
    per_frame_us = (median / n_frames) * 1000 if n_frames else 0
    print(
        f"{name:38s} median {median:6.1f} ms  best {best:6.1f} ms  "
        f"{n_frames:3d} frames ({per_frame_us:.0f} µs/frame)"
    )


def main() -> None:
    total_events, dims = count_events_and_dims()
    dense_bytes, sparse_bytes = estimate_pcie_bytes(total_events, 60, dims)
    print(f"file:        {PATH.name}")
    print(f"dims:        {dims} (w, h)")
    print(f"fps:         {FPS}")
    print(f"events:      {total_events}")
    print(f"PCIe dense:  {dense_bytes/1e6:.2f} MB total")
    print(f"PCIe sparse: {sparse_bytes/1e6:.2f} MB total")
    print(f"device:      {torch.cuda.get_device_name(DEVICE)}")
    print(f"runs:        {REPEATS} measured (1 warm-up)")
    print()

    report("dense (CPU rasterize + upload)", time_dense_upload)
    report("sparse (upload + GPU index_put_)", time_sparse_scatter)


if __name__ == "__main__":
    main()
