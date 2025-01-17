import faery

colormap = faery.colormaps.batlow.repeated(count=10, flip_odd_indices=True)
colormap.name = "batlow repeated"
colormap.to_file(
    faery.dirname.parent / "tests" / "data_generated" / "repeated_colormap.png"
)

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .render(decay="linear", tau="00:00:00.100000", colormap=colormap)
    .to_files(
        faery.dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_frames_repeated_colormap"
        / "{index:04}.png"
    )
)
