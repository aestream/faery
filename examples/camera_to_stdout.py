import faery

(
    faery.events_stream_from_camera()  # Open an event camera, if available
    .crop(10, 110, 10, 110)            # Remove events outside the region (10, 10) to (110, 110)
    .to_stdout()                       # Print the events to stdout
)
