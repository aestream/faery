import json
import typing

import faery

from . import command


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (["faery info <input>"], "prints information about a file")

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return {"info"}

    @typing.override
    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.add_argument("input", help="path of the input file")
        parser.add_argument(
            "--json",
            "-j",
            action="store_true",
            help="output information in JSON format",
        )
        args = parser.parse_args(args=arguments)
        # @DEV TODO
        # faery.events_stream_from_file
