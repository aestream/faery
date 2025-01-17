import faery

colormap = faery.Colormap.diverging_from_triplet(
    name="The Son of Man",
    start="#FF0000",
    middle="#000000",
    end="#00FF00",
)
colormap.to_file(
    faery.dirname.parent / "tests" / "data_generated" / "custom_colormap.png"
)

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .render(decay="window", tau="00:00:00.020000", colormap=colormap)
    .to_files(
        faery.dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_frames_custom_colormap"
        / "{index:04}.png"
    )
)
