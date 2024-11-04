import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

print("🎬 Render dvs.es as a slow-motion video")

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=600.0)
    .envelope(
        decay="exponential",
        tau="00:00:00.200000",
    )
    .colorize(colormap=faery.colormaps.managua.flipped())
    .scale(factor=4.0)
    .add_timecode()
    .to_file(
        dirname.parent / "tests" / "data_generated" / "dvs_slow_motion.mp4",
        on_progress=faery.progress_bar,
    )
)
