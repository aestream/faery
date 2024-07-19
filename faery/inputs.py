from typing import Union
from pathlib import Path

import faery.rusty_faery as rusty

from faery.csv import CsvFileEventStream
from faery.stream_event import EventStream


def read_file_dat(filename: str) -> EventStream:
    return rusty.dat.Decoder(filename)


def read_file(filename: Union[str, Path]):
    if isinstance(filename, str):
        filename = Path(filename)
    assert filename.exists(), f"File {filename} does not exist"

    if filename.suffix == ".csv":
        return CsvFileEventStream(filename)
    elif filename.suffix == ".dat":
        return rusty.dat.Decoder(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
