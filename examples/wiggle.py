import faery

stream = faery.events_stream_from_file(
    faery.dirname.parent / "tests" / "data" / "dvs.es",
)

wiggle_parameters = faery.WiggleParameters(time_range=stream.time_range())

(
    stream.regularize(frequency_hz=wiggle_parameters.frequency_hz)
    .render(
        decay=wiggle_parameters.decay,
        tau=wiggle_parameters.tau,
        colormap=faery.colormaps.managua.flipped(),
    )
    .scale()
    .add_timecode(output_frame_rate=wiggle_parameters.frame_rate)
    .to_file(
        path=faery.dirname.parent / "tests" / "data_generated" / "wiggle.gif",
        frame_rate=wiggle_parameters.frame_rate,
        rewind=wiggle_parameters.rewind,
        skip=wiggle_parameters.skip,
        on_progress=faery.progress_bar,
    )
)
