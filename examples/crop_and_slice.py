import pathlib

import faery

dirname = pathlib.Path(__file__).resolve().parent

(
    faery.events_stream_from_file(
        dirname.parent / "tests" / "data" / "dvs.es",
    )
    .crop(
        left=110,
        right=210,
        top=70,
        bottom=170,
    )
    .time_slice(
        start="00:00:00.400000",
        end="00:00:00.600000",
        zero=True,
    )
    .to_file(
        dirname.parent / "tests" / "data_generated" / "dvs_crop_and_slice.csv",
    )
)
