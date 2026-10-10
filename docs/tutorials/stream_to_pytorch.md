---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Streaming events into PyTorch and JAX

Event cameras produce sparse, asynchronous streams of events rather than frames.
To feed them to a conventional deep learning model, we need to (1) chop the stream
into fixed-rate packets and (2) hand each packet to the ML framework without
copying data around. Faery does both: `regularize` produces fixed-duration
packets, and the DLPack exporters (`to_dlpack_frame`, `to_dlpack_indices`, and
`to_dlpack_sparse`) expose each packet through the [DLPack protocol](https://dmlc.github.io/dlpack/latest/)
that PyTorch, JAX, TensorFlow, and CuPy all understand.

In this tutorial we run a convolutional edge detector over a live event stream.
We use a file recording so the notebook runs anywhere; swapping in a camera is a
one-line change (see the end of the tutorial).

:::{tip}
This page is a Jupyter notebook stored as [jupytext](https://jupytext.readthedocs.io/)
Markdown. To run it locally, download the source
(`docs/tutorials/stream_to_pytorch.md`) and either open it directly in Jupyter
Lab with the jupytext extension installed, or convert it first:

```sh
pip install jupytext
jupytext --to ipynb stream_to_pytorch.md
```

You will also need `pip install faery torch` (and `jax` for the last section).
:::

## Open a regularized stream

`regularize` turns the raw stream into packets covering equal time slices —
here 60 packets per second, so each packet holds the events of one "frame".

```{code-cell} python
import pathlib
import urllib.request

import faery

# Use the recording from the faery repository, downloading it if this notebook
# runs outside of a repository checkout.
PATH = pathlib.Path("../../tests/data/dvs.es")
if not PATH.exists():
    PATH = pathlib.Path("dvs.es")
    if not PATH.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/aestream/faery/main/tests/data/dvs.es",
            PATH,
        )

stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
width, height = stream.dimensions()
width, height
```

## From packets to tensors

`to_dlpack_frame` rasterizes each packet in Rust into a `(2, height, width)`
count frame — channel 0 counts OFF events per pixel, channel 1 counts ON
events. The result exposes `__dlpack__`, so `torch.from_dlpack` wraps it
without copying:

```{code-cell} python
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

frame = next(iter(stream.to_dlpack_frame(dtype="u16")))
tensor = torch.from_dlpack(frame)
tensor.shape, tensor.dtype, int(tensor.sum())
```

The `dtype` argument selects the frame's element type: `"u8"` (saturates
at 255), `"u16"` (default, saturates at 65535), `"u32"`, or `"f32"`. A network wants float32, but it is
cheaper to upload the u16 frame and call `.float()` on the device than to ask
faery for `"f32"`: half the bytes cross the PCIe bus, which made it about 2.6x
faster in our benchmarks at low event counts. The cost is saturation at 65535
events per pixel per packet, which only matters for hot pixels in long packets.

## An edge-detection convolution

We use a smoothed horizontal difference-of-sigmoids convolutional filter and its transpose to track horizontal and vertical edges in the stream.

```{code-cell} python
kernel_size = 9
gaussian = torch.sigmoid(torch.linspace(-10, 10, kernel_size + 1))
kernel = (gaussian.diff() - 0.14).repeat(kernel_size, 1)
kernels = torch.stack((kernel, kernel.T))

convolution = torch.nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=kernel_size,
    padding=12,
    bias=False,
    dilation=3,
)
convolution.weight = torch.nn.Parameter(kernels.unsqueeze(1))
convolution = convolution.to(DEVICE)
```

## Stream frames through the network

Iterating `to_dlpack_frame` yields one frame per packet, indefinitely for a
live camera and until end-of-file for a recording.
Each frame goes zero-copy into PyTorch, onto the GPU if one is available, and through the convolution.
We process one second ($60\ \mathrm{frames}$), the whole of this recording;
`itertools.islice` is also how you would bound a live camera stream:

```{code-cell} python
import itertools

with torch.inference_mode():
    for index, frame in enumerate(
        itertools.islice(stream.to_dlpack_frame(dtype="u16"), 60)
    ):
        tensor = torch.from_dlpack(frame).to(DEVICE).float()
        # Merge OFF and ON counts into one input channel, add a batch dimension.
        tensor = tensor.sum(dim=0).view(1, 1, height, width)
        # filtered has shape (1, 2, H', W'): horizontal and vertical edge maps.
        filtered = convolution(tensor)
        if index % 30 == 0:
            print(
                f"frame {index:3d} on {tensor.device}: "
                f"{int(tensor.sum()):6d} events, "
                f"edge energy horizontal={filtered[0, 0].abs().sum():9.1f} "
                f"vertical={filtered[0, 1].abs().sum():9.1f}"
            )
```

Replace the `print` with whatever your model does — the loop structure is the
whole recipe: `regularize` → `to_dlpack_frame` → `from_dlpack` → forward pass.

## The same pipeline in JAX

The faery side is identical — only the consumer changes. One difference
from PyTorch: JAX runs a jitted function on the device its inputs live on, and
`jax.dlpack.from_dlpack` wraps a numpy frame as a *CPU* array, so the
convolution would silently run on the CPU. `jax.device_put` moves each frame to
the default device (the GPU, if JAX has one) instead; it accepts the numpy
frame directly, which was about 20% faster than going through `from_dlpack`
first in our benchmarks. `lax.conv_general_dilated` applies the same dilated
kernel:

```{code-cell} python
import jax
import jax.numpy as jnp
from jax import lax

# The same kernels, rebuilt as a JAX array of shape (2, 1, 9, 9).
kernels_jax = jnp.asarray(kernels.numpy())[:, None, :, :]


@jax.jit
def edge_filter(tensor):
    return lax.conv_general_dilated(
        tensor,
        kernels_jax,
        window_strides=(1, 1),
        padding=[(12, 12), (12, 12)],
        rhs_dilation=(3, 3),
    )


stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
for index, frame in enumerate(
    itertools.islice(stream.to_dlpack_frame(dtype="u16"), 60)
):
    tensor = jax.device_put(frame).astype(jnp.float32)
    tensor = tensor.sum(axis=0).reshape(1, 1, height, width)
    filtered = edge_filter(tensor)
    if index % 30 == 0:
        print(
            f"frame {index:3d} on {filtered.device}: "
            f"edge energy horizontal={jnp.abs(filtered[0, 0]).sum():9.1f} "
            f"vertical={jnp.abs(filtered[0, 1]).sum():9.1f}"
        )
```

## Sparse export

Rasterizing in Rust is convenient, but a packet usually holds far fewer events
than the frame has pixels, so it is often faster to ship the events and build
the frame on the GPU. `to_dlpack_indices` turns each event into a single int32
index into a flattened `(2, height, width)` frame
(`p * height * width + y * width + x`): one compact array per packet, and one
`index_add_` on the device rebuilds the frame:

```{code-cell} python
stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
indices_np = next(iter(stream.to_dlpack_indices()))

indices = torch.from_dlpack(indices_np).to(DEVICE)
sparse_frame = torch.zeros(2 * height * width, device=DEVICE)
sparse_frame.index_add_(0, indices, torch.ones(len(indices), device=DEVICE))
sparse_frame = sparse_frame.view(2, height, width)

dense_frame = torch.from_dlpack(next(iter(stream.to_dlpack_frame()))).float()
torch.equal(sparse_frame.cpu(), dense_frame)
```

In our benchmarks this was 1.5 to 5.1 times faster than uploading x, y, and p
separately and scattering with `index_put_`, and faster than uploading a dense
u16 frame on 1280x720 sensors (on par for this 320x240 recording). In a loop, preallocate the ones on the device, sized
to the largest packet, and slice them (`ones[: len(indices)]`).

### High frame rates: one scatter for several windows

Every upload and every `index_add_` has a fixed cost, about 0.1 ms with
PyTorch on an RTX 3090. At 60 Hz that is noise, but at 1 kHz it is paid
1,000 times per second, and it dominated our benchmark: building 1 ms frames
from a 1280x720 recording took twice as long as with a pre-decoded file.
`windows_per_batch` groups the windows instead. It yields `(indices, windows)`
pairs, with window `j` of each group offset by `j` frames, so a single upload
and a single `index_add_` build a stack of `windows` frames:

```{code-cell} python
fast_stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=1000.0)
ones = torch.ones(1 << 16, device=DEVICE)  # grown below if a group needs more

frames = []
for group, windows in fast_stream.to_dlpack_indices(windows_per_batch=16):
    if len(group) > len(ones):
        ones = torch.ones(len(group), device=DEVICE)
    group = torch.from_dlpack(group).to(DEVICE)
    stack = torch.zeros(windows * 2 * height * width, device=DEVICE)
    stack.index_add_(0, group, ones[: len(group)])
    frames.extend(stack.view(windows, 2, height, width))

dense = [torch.from_dlpack(frame).float() for frame in fast_stream.to_dlpack_frame()]
len(frames), all(torch.equal(a.cpu(), b) for a, b in zip(frames, dense))
```

With groups of 16 windows, the 1 kHz benchmark ran twice as fast, level with
the pre-decoded file. The price is latency: a frame waits for the rest of its
group (up to 15 ms here). At video rates, where the fixed costs are small,
leave `windows_per_batch` off: at 60 Hz, groups of 16 gained 5% and delayed
the first frame from 7 ms to 66 ms. `windows` is the group's size except for
the last group, and empty windows still count as frames.

### Pinned buffers and prefetching

For the fastest uploads, pass `out=` buffers in pinned memory: faery writes
the indices (or frames, with `to_dlpack_frame`) directly where the GPU's copy
engine reads them, and `.to(DEVICE, non_blocking=True)` returns at once. The
docstrings of `to_dlpack_indices` and `to_dlpack_frame` show the pattern,
including when a buffer may be reused. Both methods also take `prefetch=`,
which prepares the next packets in a background thread. It rarely pays:
faery's per-window work is now small, and in our benchmarks prefetching was
4% faster at 60 Hz and 13% slower at 1 kHz. Measure before you enable it.

Indices are also the fastest route in JAX: 1.4x faster than uploading the
dense frame for this recording, and 1.3 to 6 times faster at 1280x720 (the
gap narrows as packets fill up). There is one catch: JAX compiles a jitted
function once per input shape, and every packet has a different length. Scattering the raw indices recompiles on nearly every
packet (about 58 ms each on an RTX 3090). Pad each packet to a power-of-two
size with an out-of-range index instead, and let `mode="drop"` discard the
padding; there are then only a handful of shapes to compile:

```{code-cell} python
import numpy


@jax.jit
def scatter(indices):
    flat = jnp.zeros(2 * height * width, jnp.float32)
    flat = flat.at[indices].add(1.0, mode="drop")
    return flat.reshape(2, height, width)


def pad(indices):
    size = max(1024, 1 << (len(indices) - 1).bit_length())
    padded = numpy.full(size, 2 * height * width, dtype=numpy.int32)
    padded[: len(indices)] = indices
    return padded


jax_frame = scatter(jax.device_put(pad(indices_np)))
bool((jax_frame == jnp.asarray(dense_frame.numpy())).all())
```

Even so, PyTorch's `index_add_` was 1.5 to 3 times faster than this in our
benchmarks (about 0.1 ms more per packet at low event counts).

Some models — event GNNs, point-cloud networks, custom CUDA kernels — need the
raw event fields, timestamps included. `to_dlpack_sparse` yields each packet as
a dict of contiguous per-field arrays instead, again DLPack-compatible:

```{code-cell} python
packet = next(iter(stream.to_dlpack_sparse(fields=("t", "x", "y", "p"))))
{field: (torch.from_dlpack(array).shape, torch.from_dlpack(array).dtype)
 for field, array in packet.items()}
```

Only the listed fields are copied, so pass the subset you need.

## Choosing a method

Seconds to put every frame of a 12.5 s, 1280x720 recording (111 M events)
on an RTX 3090, and the time until the first frame arrives:

| method | 60 Hz | 1 kHz | first frame (60 Hz / 1 kHz) |
|---|---:|---:|---:|
| `to_dlpack_frame(dtype="u16")` | 1.10 | 3.35 | 7 / 2 ms |
| `to_dlpack_frame(dtype="u8")` | 0.97 | 1.88 | 6 / 2 ms |
| `to_dlpack_indices()` + `index_add_` | 0.92 | 2.17 | 7 / 2 ms |
| `to_dlpack_indices(windows_per_batch=16)` | 0.87 | 1.14 | 66 / 6 ms |

Scatter indices at video rates, and group windows at high rates if your model
can wait for a group. If you need frames from the CPU, `"u8"` is the fastest
`dtype` when no pixel sees more than 255 events per packet.

## Going live

To run this on a real event camera instead of a recording, replace the input
(requires the camera extra: `pip install faery[camera]`):

```python
stream = faery.events_stream_from_camera().regularize(frequency_hz=60.0)
```

Everything downstream — frame export, DLPack import, the model — stays exactly
the same. A standalone script version of this tutorial lives in the
[examples directory](https://github.com/aestream/faery/tree/main/examples).
