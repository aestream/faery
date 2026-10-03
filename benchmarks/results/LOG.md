# GPU handoff benchmark log

One entry per saved run, newest last. Paste the output of
`python3 benchmarks/compare.py <previous> <this>` under each entry. Raw runs are
the JSON files next to this log; see _docs/dev.md_ for how to produce them.

## 0006_decode-before, 0005_decode-after

Decoding a 1280x720 recording (Prophesee `driving_sample`, 111,484,516 events,
12.5 s) from DAT and from an AEDAT 4 transcode, with `benchmarks/test_decode.py`
(`BENCH_DATA` pointing at the recordings). Before is 6e8528b; after adds the
DAT decoder writing `EVENTS_DTYPE` directly (no clip/astype, one memcpy per
packet) and `time_range()` without decoding (AEDAT 4 file data table, DAT
timestamp-only scan). Times are per round, one round = the whole recording
(the "ms/packet" header is compare.py's default label).

```
`0006_decode-before` (6e8528b9) → `0005_decode-after` (6e8528b9+dirty)

| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_regularize_cold[driving_sample.aedat4] | 3758.0096 | 2004.3264 | -46.7% |
| test_regularize_cold[driving_sample.dat] | 4092.8248 | 781.8833 | -80.9% |
| test_decode[driving_sample.aedat4] | 1775.1457 | 1822.8090 | +2.7% |
| test_decode[driving_sample.dat] | 1995.6216 | 484.8121 | -75.7% |
| test_regularize[driving_sample.aedat4] | 1968.2525 | 2033.3914 | +3.3% |
| test_regularize[driving_sample.dat] | 2065.4958 | 614.8083 | -70.2% |
| test_time_range[driving_sample.aedat4] | 1778.6633 | 14.3458 | -99.2% |
| test_time_range[driving_sample.dat] | 1930.1206 | 161.6581 | -91.6% |
```

## 0001_baseline, 0002_baseline-repeat — noise floor

Same commit twice on a quiet machine (load < 3, pinned to cores 2-5): CPU rows
repeat within ±2% (1M-event u16 ±14%), GPU rows within ±12%. Treat smaller
differences as noise.

## 0003_safety — regression

Safety fixes (raw polarity byte, signed stride) plus PyArray_Zeros. Frame
rasterization 23-112% slower; sparse rows unchanged (control).

```
`0001_baseline` (3c83edbc) → `0003_safety` (25f79632)
| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_cpu_frame[dvs.es-f32] | 0.0198 | 0.0191 | -3.4% |
| test_cpu_frame[dvs.es-u16] | 0.0160 | 0.0302 | +89.3% |
| test_cpu_frame[synthetic-1000-f32] | 0.2111 | 0.4472 | +111.9% |
| test_cpu_frame[synthetic-1000-u16] | 0.1477 | 0.1822 | +23.3% |
| test_cpu_frame[synthetic-10000-f32] | 0.2456 | 0.5167 | +110.4% |
| test_cpu_frame[synthetic-10000-u16] | 0.0676 | 0.0909 | +34.6% |
| test_cpu_frame[synthetic-100000-f32] | 0.6010 | 0.9634 | +60.3% |
| test_cpu_frame[synthetic-100000-u16] | 0.3112 | 0.5153 | +65.6% |
| test_cpu_frame[synthetic-1000000-f32] | 3.6183 | 3.3743 | ~-6.7% |
| test_cpu_frame[synthetic-1000000-u16] | 2.7059 | 4.7820 | +76.7% |
```

## 0004_indices — to_dlpack_indices

One int32 flat index per event, one upload, index_add_/bincount on the GPU.
index_add_ is 1.5-5.1x faster than the x/y/p index_put_ path and beats the
dense u16 frame on every 1280x720 workload (dvs.es 320x240: on par).

```
`0003_safety` (25f79632) → `0004_indices` (9ef64f33)
| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_gpu_indices_bincount[dvs.es] | — | 0.0973 | new |
| test_gpu_indices_bincount[synthetic-1000000] | — | 1.9272 | new |
| test_gpu_indices_bincount[synthetic-100000] | — | 0.2680 | new |
| test_gpu_indices_bincount[synthetic-10000] | — | 0.1026 | new |
| test_gpu_indices_bincount[synthetic-1000] | — | 0.0884 | new |
| test_gpu_indices_index_add[dvs.es] | — | 0.0655 | new |
| test_gpu_indices_index_add[synthetic-1000000] | — | 1.8831 | new |
| test_gpu_indices_index_add[synthetic-100000] | — | 0.2242 | new |
| test_gpu_indices_index_add[synthetic-10000] | — | 0.0693 | new |
| test_gpu_indices_index_add[synthetic-1000] | — | 0.0548 | new |
| test_gpu_sparse_index_put[dvs.es] | 0.3226 | 0.3212 | -0.4% |
| test_gpu_sparse_index_put[synthetic-1000000] | 2.8338 | 2.8274 | -0.2% |
| test_gpu_sparse_index_put[synthetic-100000] | 0.4992 | 0.4980 | -0.3% |
| test_gpu_sparse_index_put[synthetic-10000] | 0.3242 | 0.3220 | -0.7% |
| test_gpu_sparse_index_put[synthetic-1000] | 0.2840 | 0.2818 | -0.8% |
```

## 0007_zeroing-fix — memset instead of PyArray_Zeros

calloc'd multi-MB frames page-fault during the scatter. f32 recovers; u16
still 30-90% slower, growing with event count (loop, not allocation).

```
`0001_baseline` (3c83edbc) → `0007_zeroing-fix` (6ad7b586)
| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_cpu_frame[dvs.es-f32] | 0.0198 | 0.0189 | -4.5% |
| test_cpu_frame[dvs.es-u16] | 0.0160 | 0.0312 | +95.5% |
| test_cpu_frame[synthetic-1000-f32] | 0.2111 | 0.2300 | +8.9% |
| test_cpu_frame[synthetic-1000-u16] | 0.1477 | 0.1519 | +2.8% |
| test_cpu_frame[synthetic-10000-f32] | 0.2456 | 0.2572 | +4.7% |
| test_cpu_frame[synthetic-10000-u16] | 0.0676 | 0.0898 | +32.9% |
| test_cpu_frame[synthetic-100000-f32] | 0.6010 | 0.6033 | +0.4% |
| test_cpu_frame[synthetic-100000-u16] | 0.3112 | 0.5310 | +70.6% |
| test_cpu_frame[synthetic-1000000-f32] | 3.6183 | 3.2873 | ~-9.1% |
| test_cpu_frame[synthetic-1000000-u16] | 2.7059 | 5.1087 | +88.8% |
```

## 0008_saturate-branch — branch instead of saturating_add

After the walker refactor LLVM emitted a branchless select for u16
saturating_add (1M events: 4.73 ms vs 2.20 ms with a branch). Frame paths
back to baseline. Open: gpu_frame[dvs.es-u16] reads 0.17 ms in some runs and
0.055 in others with identical CPU time — GPU-side bimodality, not this change.

```
`0001_baseline` (3c83edbc) → `0008_saturate-branch` (ec466ac7)
| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_cpu_frame[dvs.es-f32] | 0.0198 | 0.0193 | -2.4% |
| test_cpu_frame[dvs.es-u16] | 0.0160 | 0.0155 | -2.9% |
| test_cpu_frame[synthetic-1000-f32] | 0.2111 | 0.2240 | +6.1% |
| test_cpu_frame[synthetic-1000-u16] | 0.1477 | 0.1524 | +3.1% |
| test_cpu_frame[synthetic-10000-f32] | 0.2456 | 0.2609 | +6.2% |
| test_cpu_frame[synthetic-10000-u16] | 0.0676 | 0.0665 | -1.5% |
| test_cpu_frame[synthetic-100000-f32] | 0.6010 | 0.5994 | ~-0.3% |
| test_cpu_frame[synthetic-100000-u16] | 0.3112 | 0.2955 | -5.0% |
| test_cpu_frame[synthetic-1000000-f32] | 3.6183 | 3.3427 | ~-7.6% |
| test_cpu_frame[synthetic-1000000-u16] | 2.7059 | 2.9977 | ~+10.8% |
| test_gpu_frame[dvs.es-f32] | 0.0604 | 0.0619 | +2.5% |
| test_gpu_frame[dvs.es-u16] | 0.0545 | 0.1693 | +210.7% |
| test_gpu_frame[synthetic-1000-f32] | 0.6344 | 0.6640 | +4.7% |
| test_gpu_frame[synthetic-1000-u16] | 0.2425 | 0.2575 | +6.2% |
| test_gpu_frame[synthetic-10000-f32] | 0.6883 | 0.7268 | +5.6% |
| test_gpu_frame[synthetic-10000-u16] | 0.2670 | 0.2819 | +5.6% |
| test_gpu_frame[synthetic-100000-f32] | 1.2103 | 1.2111 | ~+0.1% |
| test_gpu_frame[synthetic-100000-u16] | 0.5247 | 0.5663 | +7.9% |
| test_gpu_frame[synthetic-1000000-f32] | 4.3327 | 4.1968 | -3.1% |
| test_gpu_frame[synthetic-1000000-u16] | 3.1443 | 3.0073 | -4.4% |
```

## 0009_out-buffer — out= buffers, pinned double-buffering

rasterize_to_frame writes into caller buffers. CPU-side 2.6-3.4x faster at
low event counts (no per-packet allocation). On the GPU, reusing a pageable
buffer gains nothing (synchronous upload dominates); two pinned buffers in
turn are 24-38% faster than plain u16 uploads at 1k-100k events and hit the
PCIe floor (~0.18 ms for a 3.7 MB 1280x720 u16 frame). to_dlpack_indices
(0004) is still fastest on every 1280x720 workload; dense pinned wins only
on the 320x240 dvs.es (0.047 vs 0.066 ms). gpu_frame[dvs.es-u16] flipped
back to 0.054 ms with no change to that path: confirms GPU-side noise.

```
`0008_saturate-branch` (ec466ac7) → `0009_out-buffer` (5501ab98)
| benchmark | before (ms/packet) | after (ms/packet) | change |
|---|---:|---:|---:|
| test_cpu_frame[dvs.es-f32] | 0.0193 | 0.0192 | ~-0.5% |
| test_cpu_frame[dvs.es-u16] | 0.0155 | 0.0155 | ~+0.0% |
| test_cpu_frame[synthetic-1000-f32] | 0.2240 | 0.2307 | +3.0% |
| test_cpu_frame[synthetic-1000-u16] | 0.1524 | 0.1516 | ~-0.5% |
| test_cpu_frame[synthetic-10000-f32] | 0.2609 | 0.2626 | ~+0.6% |
| test_cpu_frame[synthetic-10000-u16] | 0.0665 | 0.0665 | ~-0.1% |
| test_cpu_frame[synthetic-100000-f32] | 0.5994 | 0.5892 | -1.7% |
| test_cpu_frame[synthetic-100000-u16] | 0.2955 | 0.2912 | -1.4% |
| test_cpu_frame[synthetic-1000000-f32] | 3.3427 | 3.1850 | -4.7% |
| test_cpu_frame[synthetic-1000000-u16] | 2.9977 | 2.5381 | -15.3% |
| test_cpu_frame_out[dvs.es-f32] | — | 0.0189 | new |
| test_cpu_frame_out[dvs.es-u16] | — | 0.0147 | new |
| test_cpu_frame_out[synthetic-1000-f32] | — | 0.0843 | new |
| test_cpu_frame_out[synthetic-1000-u16] | — | 0.0442 | new |
| test_cpu_frame_out[synthetic-10000-f32] | — | 0.1017 | new |
| test_cpu_frame_out[synthetic-10000-u16] | — | 0.0641 | new |
| test_cpu_frame_out[synthetic-100000-f32] | — | 0.3174 | new |
| test_cpu_frame_out[synthetic-100000-u16] | — | 0.3112 | new |
| test_cpu_frame_out[synthetic-1000000-f32] | — | 2.4166 | new |
| test_cpu_frame_out[synthetic-1000000-u16] | — | 2.7531 | new |
| test_gpu_frame[dvs.es-f32] | 0.0619 | 0.0602 | -2.7% |
| test_gpu_frame[dvs.es-u16] | 0.1693 | 0.0539 | -68.2% |
| test_gpu_frame[synthetic-1000-f32] | 0.6640 | 0.6149 | -7.4% |
| test_gpu_frame[synthetic-1000-u16] | 0.2575 | 0.2405 | -6.6% |
| test_gpu_frame[synthetic-10000-f32] | 0.7268 | 0.6668 | -8.3% |
| test_gpu_frame[synthetic-10000-u16] | 0.2819 | 0.2616 | -7.2% |
| test_gpu_frame[synthetic-100000-f32] | 1.2111 | 1.3129 | +8.4% |
| test_gpu_frame[synthetic-100000-u16] | 0.5663 | 0.5507 | -2.8% |
| test_gpu_frame[synthetic-1000000-f32] | 4.1968 | 4.0063 | -4.5% |
| test_gpu_frame[synthetic-1000000-u16] | 3.0073 | 2.9867 | -0.7% |
| test_gpu_frame_out[dvs.es] | — | 0.0679 | new |
| test_gpu_frame_out[synthetic-1000000] | — | 2.7656 | new |
| test_gpu_frame_out[synthetic-100000] | — | 0.5084 | new |
| test_gpu_frame_out[synthetic-10000] | — | 0.2910 | new |
| test_gpu_frame_out[synthetic-1000] | — | 0.2726 | new |
| test_gpu_frame_pinned[dvs.es] | — | 0.0471 | new |
| test_gpu_frame_pinned[synthetic-1000000] | — | 2.7838 | new |
| test_gpu_frame_pinned[synthetic-100000] | — | 0.3408 | new |
| test_gpu_frame_pinned[synthetic-10000] | — | 0.1841 | new |
| test_gpu_frame_pinned[synthetic-1000] | — | 0.1833 | new |
```

## 0011_jax-gpu — JAX on the GPU (benchmarks/test_handoff_jax.py)

jax[cuda13] 0.11.2 (matches torch 2.14.1+cu130) in the bench group. The
torch gpu group ran in the same session for comparison (on a tree dirty with
the in-progress GIL-release/prefetch work, which doesn't touch those paths;
all within noise of 0009/0010 except the noisy small-packet u16/pinned rows).

Recompilation: jit compiles per input shape, and packet lengths vary.
Scattering exact-length indices cost 57.6 ms/packet (10k events,
recompiling almost every packet) vs 1.05 ms with one compilation. The
scatter variants pad to power-of-two buckets (>= 1024) with an out-of-range
sentinel dropped by `mode="drop"`: <= 2x padding, a few compilations per
workload, all in warmup.

ms/packet, best JAX variant vs best PyTorch variant:

| workload | jax_frame | jax_frame_dlpack | jax_indices_bucketed | jax_sparse_bucketed | torch index_add_ | torch pinned |
|---|---:|---:|---:|---:|---:|---:|
| dvs.es | 0.259 | 0.321 | **0.184** | 0.345 | 0.069 | **0.049** |
| synthetic-1k | 1.056 | 1.257 | **0.166** | 0.350 | **0.058** | 0.085 |
| synthetic-10k | 1.074 | 1.285 | **0.186** | 0.350 | **0.073** | 0.113 |
| synthetic-100k | 1.283 | 1.541 | **0.371** | 0.694 | **0.230** | 0.366 |
| synthetic-1M | 3.634 | 4.111 | **2.789** | 5.745 | **1.899** | 2.880 |

- In JAX, indices + bucketed scatter wins everywhere: 1.4x faster than a
  dense frame on dvs.es, 1.3-6.4x at 1280x720.
- JAX is 1.5-2.8x slower than PyTorch's best: about 0.1 ms more fixed cost
  per packet (device_put from pageable numpy + dispatch + host-side padding),
  and a slower scatter at 1M events.
- jax.dlpack.from_dlpack + device_put is 13-24% slower than device_put of
  the numpy frame directly. from_dlpack yields a CPU array, and jit runs
  where its inputs are: the tutorial's JAX loop ran the convolution on the
  CPU. Fixed (device_put).
- jax_sparse_bucketed[synthetic-1M] is noisy (IQR 56 ms on a 172 ms round).

## 0010_gil-prefetch, 0012_prefetch-pinned-indices — GIL release, prefetch thread

The event walks in src/dlpack.rs run inside `Python::detach`, and
`to_dlpack_frame`/`to_dlpack_indices` take `prefetch=N`: a background thread
prepares up to N arrays ahead. `to_dlpack_indices` also takes `out=` buffers
(0012). New group gpu-consumer: each packet is followed by 0.86 ms of GPU work
(`torch.cuda._sleep`) and a blocking `.item()`, a stand-in for a model whose
result is read back per packet. Without a blocking consumer there's nothing to
overlap: GPU work is already asynchronous.

ms/packet from 0012 (consumer_only, the floor, is 0.862 everywhere):

| workload | indices pageable 0 → 2 | indices pinned 0 → 2 | frame pinned 0 → 2 |
|---|---:|---:|---:|
| dvs.es | 0.941 → 0.954 | 0.938 → 0.936 | 0.910 → 0.915 |
| synthetic-1k | 0.931 → 0.951 | 0.928 → 0.938 | 1.099 → 1.099 |
| synthetic-10k | 0.946 → 0.950 | 0.941 → 0.947 | 1.121 → 1.120 |
| synthetic-100k | 1.104 → 1.005 | 1.087 → **0.958** | 1.365 → **1.101** |
| synthetic-1M | 2.886 → *6.897* | 2.732 → **1.581** | 3.855 → **2.834** |

- Pinned indices + prefetch at 1M: -42%, within 0.08 ms of the CPU-side
  floor (cpu_indices: 1.52 ms). Frames: -26%. Below 100k events per packet
  there's nothing to hide; the thread costs about 0.01 ms per packet.
- Pageable indices + prefetch is 2.4x *slower* at 1M (0010 too). The
  producer's `linear_indices` took 6-7 ms CPU time (not GIL waiting: thread
  CPU time = wall time; setswitchinterval had no effect) instead of 1.5 ms.
  Cause: the 3960X has 8 L3s (3-core CCXs) and `--cpuset-cpus=2-5` spans
  two. The upload memcpy reads each index array on the consumer's core,
  malloc recycles that memory for the producer, and every write must
  invalidate the line in the other CCX's L3. Same script, 1M, no consumer:
  cpus 3-5 (one CCX) 1.95 → 1.64 ms with prefetch; cpus 2-5 1.93 → 6.82;
  unpinned 1.90 → 7.61. With pinned buffers the GPU reads by DMA, no remote
  CPU cache holds the lines, and the penalty disappears (1.91 ms on both
  core sets, pinned staging via a copy). Hence `out=` for indices, and the
  docstrings say to combine prefetch with pinned buffers.
- GIL release on the non-prefetch paths: +1-3% on most rows vs 0009 (one
  detach/reacquire per packet, ~0.5 µs; +17% = +0.4 µs on cpu_indices[1k]).
  0012's +12-22% on a few 1M dense-frame rows was load from another job
  (load average 2.9 at the start): two re-runs gave gpu_frame_out 2.82/2.88
  (0009: 2.77), gpu_frame_pinned 2.91/2.87 (2.78), cpu_frame_out f32
  2.31/2.29 (2.42).
