"""
Fork-join blob tracker: render events to frames, run a tracker on the same events,
and overlay the tracker output on the rendered frames.

                     |-> render ------------------> frames -|
    events -> regularize                                    |-> overlay_with(draw) -> video
                     |-> run_tracker -> tracker positions --|

Both branches are built from the same `regularize(frequency_hz=..., start=...)`
parameters, so packet i in each branch covers the same time window and the merge is a
positional zip (see `FrameStream.overlay_with`).

`run_tracker` is the boundary to an arbitrary backend. Here it is a trivial Python
placeholder; swap it for Greg's C++ tracker (the packet is a zero-copy structured-array
view -- hand its buffer to C++ via pybind11/DLPack and return the detections).
"""

import numpy

import faery

INPUT = faery.dirname.parent / "tests" / "data" / "dvs.es"
OUTPUT = faery.dirname.parent / "tests" / "data_generated" / "dvs_tracked.mp4"
FREQUENCY_HZ = 60.0
START = 0 * faery.us


# --- tracker branch --------------------------------------------------------------
# PLACEHOLDER. Replace the body with your detector (e.g. call into C++). It must return,
# per packet, an iterable of (x, y, radius) detections. Returning [] means "no object".
def run_tracker(packet: numpy.ndarray) -> list[tuple[int, int, int]]:
    if len(packet) == 0:
        return []
    x = int(round(packet["x"].mean()))
    y = int(round(packet["y"].mean()))
    spread = int(round(numpy.hypot(packet["x"].std(), packet["y"].std()))) + 5
    return [(x, y, spread)]


# --- draw / merge branch ---------------------------------------------------------
# Plotting only: rasterize hollow circles for each detection onto the RGBA frame.
def draw_circles(
    pixels: numpy.ndarray,
    detections: list[tuple[int, int, int]],
    color=(255, 255, 0, 255),
    thickness: float = 1.5,
) -> None:
    height, width = pixels.shape[:2]
    color_array = numpy.array(color, dtype=numpy.uint8)
    yy, xx = numpy.mgrid[0:height, 0:width]
    for cx, cy, radius in detections:
        distance = numpy.hypot(xx - cx, yy - cy)
        ring = numpy.abs(distance - radius) <= thickness
        pixels[ring] = color_array


def main() -> None:
    source = faery.events_stream_from_file(INPUT)

    frames = source.regularize(frequency_hz=FREQUENCY_HZ, start=START).render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    )
    tracks = (
        run_tracker(packet)
        for packet in source.regularize(frequency_hz=FREQUENCY_HZ, start=START)
    )

    frames.overlay_with(tracks, draw=draw_circles).to_file(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
