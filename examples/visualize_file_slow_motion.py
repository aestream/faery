import faery

(
    faery.events_stream_from_file(     # Load events from a file
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=6e3)      # Regularize to 60000 Hz
    .render(tau="00:00:00.2",          # Render frames with exponential decaying pixels
        decay="exponential", 
        colormap=faery.colormaps.managua.flipped()
    )
    .view(30)                            # Display the frames in a gui
)
