import faery

(
    faery.events_stream_from_camera("Inivation")  # Open an Inivation camera
    .crop(10, 110, 10, 110)  # Remove events outside the region (10, 10) to (110, 110)
    .to_file("some.aedat4")  # Save the events to a file
)
