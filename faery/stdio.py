import io
import sys
from typing import Any, List, Tuple

import numpy as np

from faery.output import EventOutput
from faery.stream_types import Event, Events
from faery.events_stream import EventStream, EventStreamIterator


class StdEventOutput(EventOutput):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, data: Events):
        for event in data:
            print(
                f"{event['t']},{event['x']},{event['y']},{int(event['p'])}",
                **self.kwargs,
            )


class StdEventInput(EventStream):

    def __init__(self, file: io.IOBase = sys.stdin, delimiter: Any = ","):
        """
        Initializes an event stream from STDIN.

        Parameters:
        - file (io.IOBase): The file object to be used for the stdio operations. Defaults to sys.stdin.
        - delimiter (bytes): The delimiter to be used for the stdio operations. Defaults to b",".
        """
        self.file = file
        self.delimiter = delimiter

    def __iter__(self):
        return StdEventInputIterator(file=self.file, delimiter=self.delimiter)


class StdEventInputIterator(EventStreamIterator):

    buffer: List[Tuple[int, int, int, bool]]
    index = 0

    def __init__(self, file: io.IOBase, delimiter: Any = ","):
        """
        Initializes an event stream iterator from STDIN.

        Parameters:
        - file (io.IOBase): The file object to be used for the stdio operations.
        """
        self.delimiter = delimiter
        self.file = file
        self.buffer = []

    def close(self):
        if not self.file.closed:
            self.file.close()

    def _vacate_buffer(self):
        array = np.array(self.buffer, dtype=Event)
        self.buffer = []
        return array

    def __next__(self) -> Events:
        while len(self.buffer) <= self.BUFFER_SIZE:
            line = self.file.readline()
            if not line.strip():
                break
            t, x, y, p = line.split(self.delimiter)
            self.buffer.append((int(t), int(x), int(y), bool(int(p))))
        if len(self.buffer) == 0 and not line.strip():
            raise StopIteration()
        else:
            return self._vacate_buffer()
