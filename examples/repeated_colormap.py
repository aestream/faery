import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

colormap = faery.colormaps.batlow.repeated(count=10, flip_odd_indices=True)
colormap.to_file(
    dirname.parent / "tests" / "data_generated" / "repeated_colormap.png",
    "Repeated batlow",
)

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .envelope(
        decay="linear",
        tau="00:00:00.100000",
    )
    .colorize(colormap=colormap)
    .to_files(
        dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_frames_repeated_colormap"
        / "{index:04}.png"
    )
)
