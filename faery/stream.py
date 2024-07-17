import dataclasses
import typing

T = typing.TypeVar("T")

class Sink():
    def process(self, data: "Events") -> None:
        raise NotImplementedError()

class Stream[T]:

    def __iter__(self) -> "StreamIterator[T]":
        raise NotImplementedError()

    def output(self, sink: Sink) -> None:
        for data in self:
            sink.process(data)

    # def save(self, path: str) -> None:
    #     if path.endswith(".csv"):
    #         self.output(

class StreamIterator[T]:

    BUFFER_SIZE = 1024

    def __iter__(self) -> typing.Iterator[T]:
        return self

    def __next__(self) -> typing.Any:
        raise NotImplementedError()

@dataclasses.dataclass
class Event:
  t: int
  x: int
  y: int
  p: bool

Events = typing.List[Event]

class EventStream(Stream[Events]):

    pass
