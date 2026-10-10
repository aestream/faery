"""Stream events from faery into PyTorch via DLPack; rebuild frames on the GPU.

Faery turns each packet into one int32 flat index per event
(`p * height * width + y * width + x`). That is a single, compact array to
upload (4 bytes per event), and one index_add_ on the device rebuilds the
polarity-split (2, height, width) count frame. This is faster than uploading
x/y/p separately and scattering with index_put_.

If you need the raw fields (e.g. timestamps), use stream.to_dlpack_sparse().

Requires: pip install torch
"""

import torch

import faery

PATH = faery.dirname.parent / "tests" / "data" / "dvs.es"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
width, height = stream.dimensions()

for indices_np in stream.to_dlpack_indices():
    indices = torch.from_dlpack(indices_np).to(DEVICE)
    frame = torch.zeros(2 * height * width, device=DEVICE)
    frame.index_add_(0, indices, torch.ones(len(indices), device=DEVICE))
    frame = frame.view(2, height, width)
    print(f"frame on {frame.device}: shape={tuple(frame.shape)}, events={int(frame.sum())}")
