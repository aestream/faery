import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .to_kinectograph()
    .scale(factor=4.0)
    .colorize(colormap=faery.colormaps.roma_o.rolled(shift=128).repeated(count=2))
    .to_file(dirname.parent / "tests" / "data_generated" / "kinectograph.png")
)
