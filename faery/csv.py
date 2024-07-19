from pathlib import Path
import re
import numpy

from .stream import StreamIterator
from .stream_event import EventStream, ChunkedEventStream
from .types import Event, Events
from .output import EventOutput
from typing import Optional


class CsvFileEventStreamIterator(StreamIterator):

    CSV_PATTERN = re.compile(r"(\d*),(\d*),(\d*),(\d)")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fp = open(self.path, "r")

    def __next__(self) -> Events:
        buffer: Events = []
        for line in self.fp:
            match = self.CSV_PATTERN.match(line)
            if match:
                buffer.append(
                    (
                        int(match.group(1)),  # t
                        int(match.group(2)),  # x
                        int(match.group(3)),  # y
                        int(match.group(4)),  # p
                    )
                )
            if len(buffer) >= self.BUFFER_SIZE:
                return numpy.array(buffer, dtype=Event)
        if len(buffer) > 0:
            return numpy.array(buffer, dtype=Event)
        raise StopIteration()


class CsvFileEventStream(EventStream):

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        assert self.path.is_file()

    def __iter__(self) -> "StreamIterator[Events]":
        return CsvFileEventStreamIterator(self.path)
    
    def chunk(self, dt: Optional[int] = None, n_events: Optional[int] = None) -> "ChunkedEventStream":
        return ChunkedEventStream(parent=iter(self), dt=dt, n_events=n_events)


class CsvOutput(EventOutput):

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.fp = open(self.path, "w")

    def close(self):
        self.fp.close()

    def apply(self, data: Events, *args, **kwargs):
        for event in data:
            self.fp.write(f"{event.t},{event.x},{event.y},{int(event.p)}\n")
