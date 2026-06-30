"""Stream raw event fields from faery into PyTorch via DLPack; scatter on GPU.

Faery hands over per-packet {t, x, y, p} as DLPack tensors. PyTorch uploads them
to the device and runs the scatter-add as a single index_put_ op, keeping the
rasterization on the consumer's GPU.

Requires: pip install torch
"""

import torch

import faery

PATH = faery.dirname.parent / "tests" / "data" / "dvs.es"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=60.0)
width, height = stream.dimensions()

for fields in stream.to_dlpack_sparse():
    x = torch.from_dlpack(fields["x"]).to(DEVICE, non_blocking=True).long()
    y = torch.from_dlpack(fields["y"]).to(DEVICE, non_blocking=True).long()
    p = torch.from_dlpack(fields["p"]).to(DEVICE, non_blocking=True).long()
    frame = torch.zeros((2, height, width), device=DEVICE, dtype=torch.float32)
    frame.index_put_((p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True)
    print(f"frame on {frame.device}: shape={tuple(frame.shape)}, events={int(frame.sum())}")
