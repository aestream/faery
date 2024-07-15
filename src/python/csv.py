from pathlib import Path
import re

from stream import StreamIterator, EventStream, Event, Events


class CsvFileEventStreamIterator(StreamIterator[Events]):

    CSV_PATTERN = re.compile(r"(\d*),(\d*),(\d*),(\d)")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fp = open(self.path, "r")

    def __next__(self) -> Events:
        buffer: Events = []
        for line in self.fp:
            match = self.CSV_PATTERN.match(line)
            if match:
                buffer.append(Event(
                    t=int(match.group(1)),
                    x=int(match.group(2)),
                    y=int(match.group(3)),
                    p=bool(match.group(4))
                ))
            if len(buffer) >= self.BUFFER_SIZE:
                return buffer
        if len(buffer) > 0:
            return buffer
        raise StopIteration()

class CsvFileEventStream(EventStream):

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        assert self.path.is_file()

    def __iter__(self) -> "StreamIterator[Events]":
        return CsvFileEventStreamIterator(self.path)