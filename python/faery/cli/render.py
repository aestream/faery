import argparse
import dataclasses
from typing import List

from . import commands


@dataclasses.dataclass
class RenderCommand(commands.SubCommand):
    def run(self):
        raise NotImplementedError("render command not implemented")


def render_group():
    def run(args: argparse.Namespace):
        return RenderCommand()

    parser = argparse.ArgumentParser(
        "render", description="generate a video or frames from an event file"
    )
    return commands.SubCommandGroup(parser, run)
