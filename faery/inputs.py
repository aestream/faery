from typing import Union
from pathlib import Path

import faery.rusty_faery as rusty

from faery.csv import CsvFileEventStream
from faery.stream_event import EventStream, EventStreamIterator


def read_file_dat(filename: str) -> EventStream:
    return rusty.dat.Decoder(filename)


class EventStreamFileEventStreamIterator(EventStreamIterator):

    def __init__(self, parent: EventStream):
        self.parent = parent
        self.iter = iter(parent)

    def __next__(self):
        return next(self.iter)["events"]


class EventStreamFileEventStream(EventStream):

    def __init__(self, parent: EventStream):
        self.parent = parent

    def __iter__(self):
        return EventStreamFileEventStreamIterator(self.parent)


def read_file(filename: Union[str, Path]):
    if isinstance(filename, str):
        filename = Path(filename)
    assert filename.exists(), f"File {filename} does not exist"

    if filename.suffix == ".csv":
        return CsvFileEventStream(filename)
    elif filename.suffix == ".dat":
        return rusty.dat.Decoder(filename)
    elif filename.suffix == ".es":
        return rusty.event_stream.Decoder(filename)
    elif filename.suffix == ".raw" or filename.suffix == ".evt":
        event_decoder = rusty.evt.Decoder(filename)
        return EventStreamFileEventStream(event_decoder)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
