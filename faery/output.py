from atexit import register
from pathlib import Path
from typing import Tuple, Union

import numpy

from faery import rusty_faery as rusty

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


class DatFileOutput(EventOutput):

    def __init__(
        self,
        path: Union[str, Path],
        dimensions: Tuple[int, int],
        type: str = "cd",
        format: str = "2d",
    ) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.encoder = rusty.dat.Encoder(path, "dat2", format, True, dimensions)

    def apply(self, events: Events) -> None:
        events = events.astype(
            dtype=numpy.dtype(
                [
                    ("t", "<u8"),
                    ("x", "<u2"),
                    ("y", "<u2"),
                    ("payload", "u1"),
                ]
            ),
            casting="unsafe",
            copy=False,
        )
        self.encoder.write(events)


def output_file(name: Union[str, Path], dimensions: tuple[int, int]) -> EventOutput:
    if isinstance(name, str):
        name = Path(name)
    if not isinstance(name, Path):
        raise ValueError(f"Expected a name that is either a Path or str, got {type(name)}")

    if name.suffix == ".csv":
        return CsvEventOutput(name)
    elif name.suffix == ".dat":
        return DatFileOutput(name, dimensions)
    else:
        raise ValueError(f"Unsupported file type: {name}")
