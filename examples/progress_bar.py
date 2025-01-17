import faery

print("🎬 Render dvs.es as a slow-motion video")

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=600.0)
    .render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .scale()
    .add_timecode()
    .to_file(
        faery.dirname.parent / "tests" / "data_generated" / "dvs_slow_motion.mp4",
        on_progress=faery.progress_bar,
    )
)
