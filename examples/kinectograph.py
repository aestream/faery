import faery

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .to_kinectograph()
    .scale()
    .render(
        color_theme=faery.DARK_COLOR_THEME.replace(
            colormap=faery.colormaps.roma_o.rolled(shift=128).repeated(count=2),
        )
    )
    .to_file(faery.dirname.parent / "tests" / "data_generated" / "kinectograph.png")
)
