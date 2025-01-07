import faery

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=600.0)
    .render(
        decay="exponential",
        tau="00:00:00.050000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .scale()
    .add_timecode()
    .add_overlay(
        overlay=faery.dirname.parent / "tests" / "data" / "logo.png",
        x=0,
        y=874,
    )
    .to_file(
        faery.dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_slow_motion_with_logo.mp4"
    )
)
