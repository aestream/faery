import argparse
import dataclasses
from typing import Callable, Literal

from faery.events_input import (
    events_stream_from_file,
    events_stream_from_stdin,
    events_stream_from_udp,
)


class Command:
    parser: argparse.ArgumentParser


class SubCommand(Command):
    def run(self):
        raise NotImplementedError


class FilterCommand(Command):
    def to_filter(self, stream):
        raise NotImplementedError


class InputCommand(Command):

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


class OutputCommand(Command):

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


class CommandGroup:
    parser: argparse.ArgumentParser
    runner: Callable[[argparse.Namespace], Command]


@dataclasses.dataclass
class SubCommandGroup(CommandGroup):
    parser: argparse.ArgumentParser
    runner: Callable[[argparse.Namespace], Command]


@dataclasses.dataclass
class PipelineCommandGroup(CommandGroup):
    parser: argparse.ArgumentParser
    runner: Callable[[argparse.Namespace], Command]
