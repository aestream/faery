from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar

from .output import EventOutput, FrameOutput
from .types import Events

T = TypeVar("T")


class Stream(Generic[T]):

    def __iter__(self) -> "StreamIterator":
        raise NotImplementedError()

    # def save(self, path: str) -> None:
    #     if path.endswith(".csv"):
    #         self.output(


class StreamIterator(Generic[T]):

    BUFFER_SIZE: int = 1024 * 64

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Any:
        raise NotImplementedError()


class EndlessStream(Stream[T], Generic[T]):
    pass


class FiniteStream(Stream[T], Generic[T]):

    def rate(self) -> float:
        raise NotImplementedError()
