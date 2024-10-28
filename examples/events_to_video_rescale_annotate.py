import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .envelope(
        decay="exponential",
        tau="00:00:00.200000",
    )
    .colorize(colormap=faery.colormaps.managua.flipped())
    .scale(factor=4.0)
    .add_timecode()
    .to_file(dirname.parent / "tests" / "data_generated" / "dvs_rescaled_annotated.mp4")
)
