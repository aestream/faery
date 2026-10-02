# GPU handoff benchmark log

One entry per saved run, newest last. Paste the output of
`python3 benchmarks/compare.py <previous> <this>` under each entry. Raw runs are
the JSON files next to this log; see _docs/dev.md_ for how to produce them.

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
