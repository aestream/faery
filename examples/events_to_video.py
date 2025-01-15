import faery

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .to_file(faery.dirname.parent / "tests" / "data_generated" / "dvs.mp4")
)
