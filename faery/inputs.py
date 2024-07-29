from typing import Callable, Optional, Union
from pathlib import Path

import faery.rusty_faery as rusty

from faery.csv import CsvEventStream
from faery.stream_event import EventStream, EventStreamIterator
from faery.stream_types import Events


def read_file_dat(filename: str) -> EventStream:
    return rusty.dat.Decoder(filename)


class EventStreamFileEventStreamIterator(EventStreamIterator):

    def __init__(
        self, parent: EventStream, fn: Optional[Callable[[Events], Events]] = None
    ):
        self.parent = parent
        self.iter = iter(parent)
        self.fn = fn

    def __next__(self):
        if self.fn is not None:
            return self.fn(next(self.iter))
        else:
            return next(self.iter)


class EventStreamFileEventStream(EventStream):

    def __init__(
        self, parent: EventStream, fn: Optional[Callable[[Events], Events]] = None
    ):
        self.parent = parent
        self.fn = fn

    def __iter__(self):
        return EventStreamFileEventStreamIterator(self.parent, self.fn)


def read_file(filename: Union[str, Path]):
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist")
    assert filename.exists(), f"File {filename} does not exist"

    if filename.suffix == ".csv":
        return CsvEventStream(filename)
    elif filename.suffix == ".dat":
        event_decoder = rusty.dat.Decoder(filename, version_fallback="dat2")
        return EventStreamFileEventStream(event_decoder)
    elif filename.suffix == ".es":
        event_decoder = rusty.event_stream.Decoder(filename)
        return EventStreamFileEventStream(event_decoder)
    elif filename.suffix == ".raw" or filename.suffix == ".evt":
        event_decoder = rusty.evt.Decoder(filename)
        return EventStreamFileEventStream(
            event_decoder, lambda events: events["events"]
        )
    else:
        raise ValueError(f"Unsupported file type: {filename}")
