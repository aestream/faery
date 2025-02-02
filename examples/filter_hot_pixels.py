import faery

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=10.0)  # bin events for the hot pixel filter
    .filter_hot_pixels(  # removes pixels significantly more active than their most active neighbor
        maximum_relative_event_count=3.0,
    )
    .to_file(
        faery.dirname.parent / "tests" / "data_generated" / "dvs_without_hot_pixels.es"
    )
)
