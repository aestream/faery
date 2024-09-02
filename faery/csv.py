from io import IOBase
from pathlib import Path
import re
import numpy
from typing import List, Union, Tuple

from faery.stream import StreamIterator
from faery.stream_event import EventStream
from faery.stream_types import Event, Events
from faery.output import EventOutput
from faery.stream_event import ChunkedEventStream


class CsvEventStreamIterator(StreamIterator):

    CSV_PATTERN = re.compile(r"(\d*),(\d*),(\d*),(\d)")

    buffer: List[Tuple[int, int, int, bool]]
    
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fp = open(self.path, "r")
        self.buffer = []

    def _vacate_buffer(self):
        data = numpy.array(self.buffer, dtype=Event)
        self.buffer = []
        return data

    def __next__(self) -> Events:
        for line in self.fp:
            match = self.CSV_PATTERN.match(line)
            if match:
                self.buffer.append(
                    (
                        int(match.group(1)),  # t
                        int(match.group(2)),  # x
                        int(match.group(3)),  # y
                        bool(int(match.group(4))),  # p
                    )
                )
            if len(self.buffer) >= self.BUFFER_SIZE:
                return self._vacate_buffer()
        if len(self.buffer) > 0:
            return self._vacate_buffer()
        raise StopIteration()


class CsvEventStream(EventStream):

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        assert self.path.is_file()

    def __iter__(self) -> "StreamIterator[Events]":
        return CsvEventStreamIterator(self.path)


class CsvEventOutput(EventOutput):

    def __init__(self, path: Union[str, Path, IOBase]) -> None:
        if isinstance(path, IOBase):
            self.fp = path
        else:
            path = Path(path)
            self.fp = open(path, "w")

    def close(self):
        self.fp.close()

    def apply(self, data: Events, *args, **kwargs):
        for event in data:
            self.fp.write(f"{event['t']},{event['x']},{event['y']},{int(event['p'])}\n")
