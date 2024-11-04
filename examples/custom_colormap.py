import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

colormap = faery.Colormap.diverging_from_triplet(
    start="#FF0000",
    middle="#000000",
    end="#00FF00",
)
colormap.to_file(
    dirname.parent / "tests" / "data_generated" / "custom_colormap.png",
    "The Son of Man",
)

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .envelope(
        decay="window",
        tau="00:00:00.020000",
    )
    .colorize(colormap=colormap)
    .to_files(
        dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_frames_custom_colormap"
        / "{index:04}.png"
    )
)
