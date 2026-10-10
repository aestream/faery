"""Sparse vs dense GPU handoff: frame upload, vs raw-events upload + GPU scatter.

Two paths that both produce a `(2, H, W)` polarity-split count frame on a GPU
tensor — the difference is *who* runs the scatter and what crosses the
interconnect (PCIe on discrete GPUs; the same DRAM on APUs):

- Dense (CPU rasterize -> GPU upload):
    faery.to_dlpack_frame() runs the scatter in Rust on CPU.
    The (2, H, W) frame is uploaded per packet.
    Payload per packet: ~ 2*H*W*sizeof(dtype) bytes.

- Sparse (events upload -> GPU scatter):
    faery.to_dlpack_sparse() hands per-packet {x,y,p} fields (t stays on the
    host: the stream is continuous, so the consumer doesn't need timestamps).
    Each field is uploaded and torch.Tensor.index_put_ scatters on GPU.
    Payload per packet: ~ N_events * (2 + 2 + 1) bytes.

Each path comes in two flavors:

- naive: upload from pageable numpy memory on the default stream. The copy is
    effectively synchronous, so host, DMA, and kernel serialize.
- pinned (double-buffered): the torch analog of AEStream's TensorBuffer swap
    (https://github.com/aestream/aestream/blob/main/src/python/tensor_buffer.cpp).
    Two slots of pre-allocated pinned host + device buffers; uploads are truly
    asynchronous (`non_blocking=True` from pinned memory) and run on a side
    stream, so the host prepares packet N+1 while packet N is still in flight.
    CUDA events replace AEStream's buffer_lock: a slot is reused only after its
    previous scatter has completed.

Methodology: the file is decoded (and, for the dense path, rasterized) ONCE
into cached per-packet arrays before any timing, so the timed sections measure
only the GPU handoff — not faery's decode pipeline, which otherwise dominates
(the one-off decode floor is printed for reference). All HIP/CUDA kernels are
exercised before timing so no variant pays first-use compilation costs, and
each variant additionally gets one untimed warm-up run.

At sparse-typical event density the sparse path moves less data over the
interconnect and should win on discrete GPUs; at dense density (above the
crossover) the fixed-size frame becomes smaller than the events array and the
dense path should win. On unified-memory APUs the payload difference is
irrelevant and per-packet kernel/sync overhead dominates instead.

Requires: PyTorch with CUDA. Skips gracefully otherwise.
"""

import pathlib
import sys
import time

import numpy

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
# The test file holds ~1 s of data (60 packets at FPS); loop over the cached
# packets LOOPS times per measured run so each run covers thousands of packets.
LOOPS = 200
DEVICE = torch.device("cuda")


def make_stream():
    return faery.events_stream_from_file(PATH).regularize(frequency_hz=FPS)


def load_cached_packets() -> tuple[list, list, tuple[int, int]]:
    """Decode the file once: pre-rasterized frames, sparse fields, dimensions."""
    dims = make_stream().dimensions()
    frames = list(make_stream().to_dlpack_frame(dtype="u16"))
    fields_list = list(make_stream().to_dlpack_sparse(fields=("x", "y", "p")))
    return frames, fields_list, dims


def measure_decode_floor() -> float:
    """One untimed-section reference: µs per packet for decode + regularize."""
    t0 = time.perf_counter()
    n_frames = 0
    for _ in range(LOOPS):
        for _packet in make_stream():
            n_frames += 1
    return (time.perf_counter() - t0) / n_frames * 1e6


def exercise_kernels(frames, fields_list, dims) -> None:
    """Run every op used below a few times so kernel compilation, allocator
    pools, pinned allocations, and stream/event machinery are all warm
    before any timing starts."""
    width, height = dims
    copy_stream = torch.cuda.Stream()
    pinned_i16 = torch.empty(4096, dtype=torch.int16, pin_memory=True)
    pinned_u8 = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    dev_i16 = torch.empty(4096, dtype=torch.int16, device=DEVICE)
    dev_u8 = torch.empty(4096, dtype=torch.uint8, device=DEVICE)
    event = torch.cuda.Event()
    for _ in range(3):
        for frame_np in frames[:2]:
            torch.from_dlpack(frame_np).to(DEVICE, non_blocking=True)
            torch.from_numpy(frame_np.view(numpy.int16)).pin_memory().to(
                DEVICE, non_blocking=True
            )
        for fields in fields_list[:2]:
            x = torch.from_dlpack(fields["x"]).to(DEVICE, non_blocking=True).long()
            y = torch.from_dlpack(fields["y"]).to(DEVICE, non_blocking=True).long()
            p = torch.from_dlpack(fields["p"]).to(DEVICE, non_blocking=True).long()
            frame = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
            frame.index_put_(
                (p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True
            )
            frame.zero_()
        with torch.cuda.stream(copy_stream):
            dev_i16.copy_(pinned_i16, non_blocking=True)
            dev_u8.copy_(pinned_u8, non_blocking=True)
            event.record(copy_stream)
        torch.cuda.current_stream().wait_event(event)
        dev_i16.long()
        dev_u8.long()
        event.record()
        event.synchronize()
    torch.cuda.synchronize()


def time_dense_upload(frames) -> tuple[float, int]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for _ in range(LOOPS):
        for frame_np in frames:
            _gpu = torch.from_dlpack(frame_np).to(DEVICE, non_blocking=True)
            n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def time_dense_pinned(frames, dims) -> tuple[float, int]:
    width, height = dims
    # Frames are staged as int16 views of the u16 data (a free reinterpret;
    # counts >= 32768 would need a consumer-side cast back to u16).
    slots = [
        {
            "host": torch.empty((2, height, width), dtype=torch.int16, pin_memory=True),
            "dev": torch.empty((2, height, width), dtype=torch.int16, device=DEVICE),
            "free": torch.cuda.Event(),
        }
        for _ in range(2)
    ]
    for slot in slots:
        slot["free"].record()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for _ in range(LOOPS):
        for frame_np in frames:
            slot = slots[n_frames % 2]
            # Wait until this slot's previous upload has landed before
            # overwriting its pinned staging buffer (two packets ago).
            slot["free"].synchronize()
            slot["host"].copy_(torch.from_numpy(frame_np.view(numpy.int16)))
            # Pinned -> device is truly asynchronous: the host loops on to
            # stage the next packet while this frame is DMA'd.
            slot["dev"].copy_(slot["host"], non_blocking=True)
            slot["free"].record()
            n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def time_sparse_scatter(fields_list, dims) -> tuple[float, int]:
    width, height = dims
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for _ in range(LOOPS):
        for fields in fields_list:
            x = torch.from_dlpack(fields["x"]).to(DEVICE, non_blocking=True).long()
            y = torch.from_dlpack(fields["y"]).to(DEVICE, non_blocking=True).long()
            p = torch.from_dlpack(fields["p"]).to(DEVICE, non_blocking=True).long()
            frame = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
            frame.index_put_(
                (p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True
            )
            n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def time_sparse_pinned(fields_list, dims, max_events: int) -> tuple[float, int]:
    width, height = dims
    # x/y are staged as int16 views of the u16 data (free reinterpret,
    # valid because sensor dimensions stay below 32768).
    assert width < 2**15 and height < 2**15
    copy_stream = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()
    ones = torch.ones(max_events, device=DEVICE, dtype=torch.float32)
    slots = [
        {
            "x_host": torch.empty(max_events, dtype=torch.int16, pin_memory=True),
            "y_host": torch.empty(max_events, dtype=torch.int16, pin_memory=True),
            "p_host": torch.empty(max_events, dtype=torch.uint8, pin_memory=True),
            "x_dev": torch.empty(max_events, dtype=torch.int16, device=DEVICE),
            "y_dev": torch.empty(max_events, dtype=torch.int16, device=DEVICE),
            "p_dev": torch.empty(max_events, dtype=torch.uint8, device=DEVICE),
            "frame": torch.empty(
                (2, height, width), dtype=torch.float32, device=DEVICE
            ),
            "uploaded": torch.cuda.Event(),
            "free": torch.cuda.Event(),
        }
        for _ in range(2)
    ]
    for slot in slots:
        slot["free"].record()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_frames = 0
    for _ in range(LOOPS):
        for fields in fields_list:
            slot = slots[n_frames % 2]
            n = fields["x"].shape[0]
            # Wait until this slot's previous scatter (two packets ago) has
            # completed before overwriting its buffers. A real consumer would
            # extend this fence past its own use of slot["frame"].
            slot["free"].synchronize()
            slot["x_host"][:n].copy_(torch.from_numpy(fields["x"].view(numpy.int16)))
            slot["y_host"][:n].copy_(torch.from_numpy(fields["y"].view(numpy.int16)))
            slot["p_host"][:n].copy_(torch.from_numpy(fields["p"].view(numpy.uint8)))
            with torch.cuda.stream(copy_stream):
                slot["x_dev"][:n].copy_(slot["x_host"][:n], non_blocking=True)
                slot["y_dev"][:n].copy_(slot["y_host"][:n], non_blocking=True)
                slot["p_dev"][:n].copy_(slot["p_host"][:n], non_blocking=True)
                slot["uploaded"].record(copy_stream)
            # The scatter on the main stream starts once the upload lands; the
            # host does not wait, it loops on to stage the next packet.
            main_stream.wait_event(slot["uploaded"])
            frame = slot["frame"]
            frame.zero_()
            frame.index_put_(
                (
                    slot["p_dev"][:n].long(),
                    slot["y_dev"][:n].long(),
                    slot["x_dev"][:n].long(),
                ),
                ones[:n],
                accumulate=True,
            )
            slot["free"].record(main_stream)
            n_frames += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_frames


def estimate_payload_bytes(
    n_events: int, n_frames: int, dims: tuple[int, int]
) -> tuple[int, int]:
    width, height = dims
    # Dense: one (2, H, W) uint16 frame per packet.
    dense_bytes = n_frames * 2 * height * width * 2
    # Sparse: 5 bytes per event (u16 x + u16 y + u8 p); t is not shipped —
    # the stream is continuous, so the GPU consumer doesn't need timestamps.
    sparse_bytes = n_events * (2 + 2 + 1)
    return dense_bytes, sparse_bytes


def report(name: str, fn) -> None:
    fn()  # per-variant warm-up
    results = [fn() for _ in range(REPEATS)]
    times = [elapsed for elapsed, _ in results]
    n_frames = results[0][1]
    best = min(times) * 1000
    median = sorted(times)[len(times) // 2] * 1000
    per_frame_us = (median / n_frames) * 1000 if n_frames else 0
    print(
        f"{name:38s} median {median:8.1f} ms  best {best:8.1f} ms  "
        f"{n_frames:5d} frames ({per_frame_us:.0f} µs/frame)"
    )


def main() -> None:
    frames, fields_list, dims = load_cached_packets()
    lengths = [fields["x"].shape[0] for fields in fields_list]
    total_events = sum(lengths)
    max_events = max(lengths, default=0)
    n_packets = len(fields_list)
    dense_bytes, sparse_bytes = estimate_payload_bytes(
        total_events * LOOPS, n_packets * LOOPS, dims
    )
    decode_floor_us = measure_decode_floor()

    print(f"file:           {PATH.name} ({n_packets} packets, cached {LOOPS}x per run)")
    print(f"dims:           {dims} (w, h)")
    print(f"fps:            {FPS}")
    print(f"events:         {total_events * LOOPS} (max {max_events} per packet)")
    print(f"payload dense:  {dense_bytes/1e6:.2f} MB total")
    print(f"payload sparse: {sparse_bytes/1e6:.2f} MB total")
    print(f"device:         {torch.cuda.get_device_name(DEVICE)}")
    print(f"runs:           {REPEATS} measured (kernels exercised, 1 warm-up each)")
    print(
        f"decode floor:   {decode_floor_us:.0f} µs/frame "
        "(decode + regularize; excluded from the rows below)"
    )
    print()

    exercise_kernels(frames, fields_list, dims)

    report("dense (upload cached frames)", lambda: time_dense_upload(frames))
    report(
        "dense (pinned, double-buffered)",
        lambda: time_dense_pinned(frames, dims),
    )
    report(
        "sparse (upload + GPU index_put_)",
        lambda: time_sparse_scatter(fields_list, dims),
    )
    report(
        "sparse (pinned, double-buffered)",
        lambda: time_sparse_pinned(fields_list, dims, max_events),
    )


if __name__ == "__main__":
    main()
