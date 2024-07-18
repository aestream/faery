import numpy
from atexit import register

from faery.stream_types import Events


class StatefulOutput:

    def __init__(self):
        register(self.close)

    def __enter__(self) -> "StatefulOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        raise NotImplementedError()


class EventOutput(StatefulOutput):
    def apply(self, data: Events) -> None:
        raise NotImplementedError()


class FrameOutput(StatefulOutput):
    def apply(self, data: numpy.ndarray) -> None:
        raise NotImplementedError()
