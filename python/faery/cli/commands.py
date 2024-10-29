import dataclasses
from typing import TextIO, Literal

from faery.events_stream import EventsStream
from faery.events_input import (
    events_stream_from_file,
    events_stream_from_stdin,
    events_stream_from_udp,
)


class InputCommand:
    def to_stream(self):
        raise NotImplementedError


@dataclasses.dataclass
class InputFile(InputCommand):
    name: str

    def to_stream(self):
        return events_stream_from_file(self.name)


@dataclasses.dataclass
class InputStdin(InputCommand):
    dimensions: tuple[int, int]

    def to_stream(self):
        return events_stream_from_stdin(dimensions=self.dimensions)


@dataclasses.dataclass
class InputUdp(InputCommand):
    address: str
    dimensions: tuple[int, int]
    port: int
    protocol: Literal["t64_x16_y16_on8", "t32_x16_y15_on1"]

    def to_stream(self):
        return events_stream_from_udp(
            address=(self.address, self.port),
            dimensions=self.dimensions,
            format=self.protocol,
        )


class OutputCommand:
    def to_output(self, stream) -> None:
        raise NotImplementedError


@dataclasses.dataclass
class OutputFile(OutputCommand):
    name: str

    def to_output(self, stream):
        return stream.to_file(self.name)


@dataclasses.dataclass
class OutputStdout(OutputCommand):
    def to_output(self, stream):
        return stream.to_stdout()


@dataclasses.dataclass
class OutputUdp(OutputCommand):
    address: str
    port: int
    protocol: Literal["t64_x16_y16_on8", "t32_x16_y15_on1"]

    def to_output(self, stream):
        return stream.to_udp(address=(self.address, self.port), format=self.protocol)
