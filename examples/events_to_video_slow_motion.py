import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=600.0)
    .envelope(
        decay="exponential",
        tau="00:00:00.200000",
    )
    .colorize(colormap=faery.colormaps.managua.flipped())
    .scale(factor=4.0)
    .add_timecode()
    .add_overlay(
        overlay=dirname.parent / "tests" / "data" / "logo.png",
        x=0,
        y=874,
    )
    .to_file(
        dirname.parent / "tests" / "data_generated" / "dvs_slow_motion_with_logo.mp4"
    )
)
