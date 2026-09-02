"""Tutorial: continuously run a convolution on an event stream.

This mirrors AEStream's usb_edgedetection.py example
(https://github.com/aestream/aestream/blob/main/example/usb_edgedetection.py)
using faery's DLPack frame export instead of AEStream's tensor reader.

The pipeline, step by step:

1. Open an event stream (a file here; swap in a camera below).
2. `regularize` chops the stream into fixed-duration packets, one per frame.
3. `to_dlpack_frame` rasterizes each packet in Rust to a (2, height, width)
   polarity-split count frame that exposes `__dlpack__`.
4. `torch.from_dlpack` wraps that frame zero-copy, and a Conv2d runs on it —
   on the GPU if one is available.

Requires: pip install torch
"""

import torch

import faery

# --- 1. Open a stream ---------------------------------------------------------
# For a live camera, use `faery.events_stream_from_camera()` instead
# (requires the "camera" extra: pip install faery[camera]).
PATH = faery.dirname.parent / "tests" / "data" / "dvs.es"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Fixed-rate packets: one packet per frame at 60 Hz ---------------------
stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
width, height = stream.dimensions()

# --- 3. Build the convolution (identical to the AEStream example) -------------
# Horizontal and vertical edge detectors: a smoothed difference-of-sigmoids
# kernel and its transpose, stacked as two output channels.
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

# --- 4. Stream frames through the convolution ---------------------------------
with torch.inference_mode():
    for index, frame_np in enumerate(stream.to_dlpack_frame(dtype="f32")):
        # frame has shape (2, height, width): OFF counts and ON counts.
        frame = torch.from_dlpack(frame_np).to(DEVICE, non_blocking=True)
        # Merge polarities into a single input channel, add a batch dimension.
        tensor = frame.sum(dim=0).view(1, 1, height, width)
        # filtered has shape (1, 2, H', W'): horizontal and vertical edge maps.
        filtered = convolution(tensor)
        horizontal = filtered[0, 0]
        vertical = filtered[0, 1]
        print(
            f"frame {index:3d} on {frame.device}: "
            f"{int(frame.sum()):6d} events, "
            f"edge energy horizontal={horizontal.abs().sum():9.1f} "
            f"vertical={vertical.abs().sum():9.1f}"
        )
