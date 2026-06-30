"""Encode an event file to a video using faery's DLPack frame export + numpy rendering.

Each packet is rasterized to a (2, H, W) count histogram in Rust via
to_dlpack_frame, then mapped to RGB and written by imageio. This is *not* the
same visualization as faery.render() — there is no temporal decay, just raw
per-packet polarity counts. Use this pattern when you want a custom consumer-side
renderer (e.g. ML-style "frame", custom colormap); use faery.render() for the
classic decay-colormap event video.

Requires: pip install imageio[ffmpeg]
"""

import imageio.v2 as imageio
import numpy

import faery

PATH = faery.dirname.parent / "tests" / "data" / "dvs.es"
OUT = faery.dirname.parent / "tests" / "data_generated" / "dvs_dlpack.mp4"
FPS = 60.0

stream = faery.events_stream_from_file(PATH).regularize(frequency_hz=FPS)
OUT.parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8)

try:
    for frame in stream.to_dlpack_frame(dtype="u16"):
        # OFF channel -> red, ON channel -> blue.
        off = numpy.clip(frame[0].astype(numpy.float32) / 4.0, 0.0, 1.0)
        on = numpy.clip(frame[1].astype(numpy.float32) / 4.0, 0.0, 1.0)
        rgb = numpy.zeros((frame.shape[1], frame.shape[2], 3), dtype=numpy.uint8)
        rgb[..., 0] = (off * 255).astype(numpy.uint8)
        rgb[..., 2] = (on * 255).astype(numpy.uint8)
        writer.append_data(rgb)
finally:
    writer.close()

print(f"Wrote video to {OUT}")
