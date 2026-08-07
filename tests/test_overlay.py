"""
Tests for `FrameStream.overlay_with`, the "merge" step of a fork-join pipeline:
one event stream is used twice -- once to render frames, once to compute per-packet
sidecar data (e.g. tracker positions) -- and the two branches are recombined by
drawing the sidecar onto the frames.
"""

import numpy
import pytest

import faery

EVENTS = numpy.array(
    [
        (100, 5, 5, True),  # window 0 -> [0, 1000)
        (1100, 10, 8, True),  # window 1 -> [1000, 2000)
        # window 2 -> [2000, 3000) is intentionally empty
        (3100, 20, 20, False),  # window 3 -> [3000, 4000)
    ],
    dtype=faery.EVENTS_DTYPE,
)
DIMENSIONS = (32, 24)  # width, height
FREQUENCY_HZ = 1000.0  # period = 1000 us, so one window per event gap

RED = numpy.array([255, 0, 0, 255], dtype=numpy.uint8)


def centroid(packet: numpy.ndarray):
    """Toy 'tracker': returns the (x, y) centroid of a packet, or None if empty.

    This stands in for a foreign backend. `packet` is a zero-copy structured-array
    view, exactly what would be handed to a C++/Rust tracker through pybind/DLPack.
    """
    if len(packet) == 0:
        return None
    return (int(round(packet["x"].mean())), int(round(packet["y"].mean())))


def draw_marker(pixels: numpy.ndarray, position) -> None:
    """Draws the tracker position onto the frame in place (returns None)."""
    if position is None:
        return
    x, y = position
    pixels[y, x] = RED


def branches():
    source = faery.events_stream_from_array(EVENTS, dimensions=DIMENSIONS)
    # Two branches off the same source, pinned to the same regularize parameters so
    # that packet i in each branch covers the identical time window (index alignment).
    frames = source.regularize(frequency_hz=FREQUENCY_HZ, start=0 * faery.us).render(
        decay="exponential",
        tau="00:00:00.001000",
        colormap=faery.colormaps.managua,
    )
    tracks = (
        centroid(packet)
        for packet in source.regularize(frequency_hz=FREQUENCY_HZ, start=0 * faery.us)
    )
    return frames, tracks


def test_overlay_aligns_and_draws():
    frames, tracks = branches()
    video = frames.overlay_with(tracks, draw=draw_marker)

    rendered = list(video)

    # Independently recompute the expected per-packet centroids from the same source,
    # so a match confirms frame i was paired with the sidecar derived from packet i.
    expected = [
        centroid(packet)
        for packet in faery.events_stream_from_array(
            EVENTS, dimensions=DIMENSIONS
        ).regularize(frequency_hz=FREQUENCY_HZ, start=0 * faery.us)
    ]

    assert len(rendered) == len(expected)
    assert expected == [(5, 5), (10, 8), None, (20, 20)]

    for frame, position in zip(rendered, expected):
        if position is None:
            # empty packet -> nothing drawn -> no pure-red marker anywhere
            assert not numpy.any(numpy.all(frame.pixels == RED, axis=-1))
        else:
            x, y = position
            assert numpy.array_equal(frame.pixels[y, x], RED)


def test_overlay_replacement_return_value():
    """A draw callback that returns an array replaces the frame's pixels."""
    frames, tracks = branches()

    replacement = numpy.zeros((DIMENSIONS[1], DIMENSIONS[0], 4), dtype=numpy.uint8)

    def replace(pixels: numpy.ndarray, position):
        return replacement

    rendered = list(frames.overlay_with(tracks, draw=replace))
    for frame in rendered:
        assert numpy.array_equal(frame.pixels, replacement)


def test_overlay_raises_on_short_sidecar():
    frames, _ = branches()
    # Only one sidecar item, but the stream renders several frames.
    video = frames.overlay_with([None], draw=draw_marker)
    with pytest.raises(ValueError, match="not aligned"):
        list(video)
