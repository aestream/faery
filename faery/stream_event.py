from typing import Optional, Union

from .output import EventOutput
from .stdio import StdEventOutput
from .stream import Stream, StreamIterator
import numpy as np
from .types import Event, Events


def _output_from_str(output: str, **kwargs) -> Optional[EventOutput]:
    if output == "stdout":
        return StdEventOutput(**kwargs)
    return None


class EventStream(Stream[Events]):

    def output(self, output: Union[EventOutput, str], **kwargs) -> None:
        if isinstance(output, str):
            output = _output_from_str(output, **kwargs)

        if not isinstance(output, EventOutput):
            raise ValueError(f"Unknown output: {output}")

        for data in self:
            output.apply(data)


class EventStreamIterator(StreamIterator[Events]):
    pass


class ChunkedEventStream(Stream[Events]):
    def __init__(self, parent: StreamIterator, dt: Optional[int] = None, n_events: Optional[int] = None):
        self.parent = parent
        self.dt = dt
        self.n_events = n_events

    def __iter__(self) -> "ChunkedEventStreamIterator":
        return ChunkedEventStreamIterator(parent=self.parent, dt=self.dt, n_events=self.n_events)
    
    def output(self, output: Union[EventOutput, str], **kwargs) -> None:
        if isinstance(output, str):
            output = _output_from_str(output, **kwargs)

        if not isinstance(output, EventOutput):
            raise ValueError(f"Unknown output: {output}")

        for data in self:
            output.apply(data)

class ChunkedEventStreamIterator(StreamIterator[Events]):
    def __init__(self, parent: StreamIterator, dt: Optional[int] = None, n_events: Optional[int] = None) -> None:
        self.parent = parent
        self.dt = dt
        self.n_events = n_events
        self.buffer = []
        self.state = []

    def __next__(self) -> Events:
        if self.n_events is not None:
            if len(self.state) > 0:
                for i in range(len(self.state)):
                    event = self.state[0]
                    self.state = self.state[1:]
                    self.buffer.append(event)
                    if len(self.buffer) >= self.n_events:
                        chunk, self.buffer = self.buffer, []
                        return np.array(chunk, dtype=Events)
            else:
                self.state = next(self.parent)
                for event in self.state:
                    event = self.state[0]
                    self.state = self.state[1:]
                    self.buffer.append(event)
                    if len(self.buffer) >= self.n_events:
                        chunk, self.buffer = self.buffer, []
                        return np.array(chunk, dtype=Events)
