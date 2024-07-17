from stream import StreamIterator, EventStream, Events


class DatFileEventStreamIterator(StreamIterator[Events]):

    def __init__(self, path: str) -> None:
        self.path = path
        # ...

class DatFileEventStream(EventStream):

    def __init__(self, path: str) -> None:
        self.path = path

    def __iter__(self) -> "StreamIterator[Events]":
        return DatFileEventStreamIterator(self.path)