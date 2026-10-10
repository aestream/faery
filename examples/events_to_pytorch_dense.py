"""Stream pre-rasterized (P, H, W) event frames from faery into PyTorch via DLPack.

Faery decodes events and runs the scatter in Rust; the consumer receives a
(2, height, width) count frame per packet. Frames are exported as u16 and
converted to float32 on the device: that moves half the bytes over PCIe compared
to asking faery for "f32" (u16 saturates at 65535 events per pixel per packet).

Requires: pip install torch
"""

import torch

import faery

PATH = faery.dirname.parent / "tests" / "data" / "dvs.es"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)

for frame_np in stream.to_dlpack_frame(dtype="u16"):
    frame = torch.from_dlpack(frame_np).to(DEVICE).float()
    # frame.shape == (2, height, width); feed to a model from here.
    print(f"frame on {frame.device}: shape={tuple(frame.shape)}, events={int(frame.sum())}")
