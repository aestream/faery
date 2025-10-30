import faery

(
    faery.events_stream_from_camera()  # Open an event camera
    .regularize(frequency_hz=30)       # Regularize to 30 Hz
    .render(tau="00:00:00.2",          # Render frames with exponential decaying pixels
        decay="exponential", 
        colormap=faery.colormaps.starry_night)
    .view()                            # Display the frames in a gui
)
