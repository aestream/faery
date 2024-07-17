import dataclasses
import typing

T = typing.TypeVar("T")

class Sink():
    def process(self, data: "Events") -> None:
        raise NotImplementedError()

class Stream:

    def __iter__(self) -> "StreamIterator":
        raise NotImplementedError()

    def output(self, sink: Sink) -> None:
        for data in self:
            sink.process(data)

    # def save(self, path: str) -> None:
    #     if path.endswith(".csv"):
    #         self.output(

class StreamIterator:

    BUFFER_SIZE: int = 1024 * 64

    def __iter__(self) -> typing.Iterator:
        return self

    def __next__(self) -> typing.Any:
        raise NotImplementedError()

class EventStream(Stream):
    pass

class EventStreamIterator(StreamIterator):
    pass

class ChunkedEventStream(Stream):
    pass

class ChunkedEventStreamIterator(StreamIterator):
    pass

class FrameStream(Stream):
    pass

class FrameStreamIterator(StreamIterator):
    pass