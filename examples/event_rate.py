import faery

event_rate = faery.events_stream_from_file(
    faery.dirname.parent / "tests" / "data" / "dvs.es",
).to_event_rate()

event_rate.to_file(
    faery.dirname.parent / "tests" / "data_generated" / "dvs_event_rate.svg",
)
event_rate.to_file(
    faery.dirname.parent / "tests" / "data_generated" / "dvs_event_rate.png",
)
