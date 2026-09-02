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
packets, and the DLPack exporters (`to_dlpack_frame` and `to_dlpack_sparse`)
expose each packet through the [DLPack protocol](https://dmlc.github.io/dlpack/latest/)
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

frame = next(iter(stream.to_dlpack_frame(dtype="f32")))
tensor = torch.from_dlpack(frame)
tensor.shape, tensor.dtype, int(tensor.sum())
```

The `dtype` argument selects the frame's element type: `"u16"` (default,
saturates at 65535), `"u32"`, or `"f32"`. For feeding a network, `"f32"` is
usually what you want — no cast needed on the framework side.

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
We process two seconds ($60\ \mathrm{frames} \times 2$) here:

```{code-cell} python
import itertools

with torch.inference_mode():
    for index, frame in enumerate(
        itertools.islice(stream.to_dlpack_frame(dtype="f32"), 120)
    ):
        tensor = torch.from_dlpack(frame).to(DEVICE, non_blocking=True)
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

DLPack is framework-agnostic, so the faery side is identical — only the
consumer changes. `jax.dlpack.from_dlpack` imports each frame, and
`lax.conv_general_dilated` applies the same dilated kernel:

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
    itertools.islice(stream.to_dlpack_frame(dtype="f32"), 120)
):
    tensor = jax.dlpack.from_dlpack(frame).sum(axis=0).reshape(1, 1, height, width)
    filtered = edge_filter(tensor)
    if index % 30 == 0:
        print(
            f"frame {index:3d}: "
            f"edge energy horizontal={jnp.abs(filtered[0, 0]).sum():9.1f} "
            f"vertical={jnp.abs(filtered[0, 1]).sum():9.1f}"
        )
```

## Sparse export

Rasterizing to frames is convenient for convolutions, but some models — event
GNNs, point-cloud networks, custom CUDA kernels — consume raw events.
`to_dlpack_sparse` yields each packet as a dict of contiguous per-field arrays
instead, again DLPack-compatible:

```{code-cell} python
stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
packet = next(iter(stream.to_dlpack_sparse(fields=("x", "y", "p"))))
{field: (torch.from_dlpack(array).shape, torch.from_dlpack(array).dtype)
 for field, array in packet.items()}
```

For a continuously streaming consumer, timestamps are implicit in the packet
cadence, so `fields=("x", "y", "p")` skips copying `t` entirely.

## Going live

To run this on a real event camera instead of a recording, replace the input
(requires the camera extra: `pip install faery[camera]`):

```python
stream = faery.events_stream_from_camera().regularize(frequency_hz=60.0)
```

Everything downstream — frame export, DLPack import, the model — stays exactly
the same. A standalone script version of this tutorial lives in the
[examples directory](https://github.com/aestream/faery/tree/main/examples).
