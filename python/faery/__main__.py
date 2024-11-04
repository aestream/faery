import sys
from typing import List

from faery.cli import commands, colormaps, init, parser, render


def print_help():
    import textwrap

    print(
        textwrap.dedent(
            rf"""
             __       __      _____
            /  \     /  \    |  ___|_ _  ___ _ __ _   _
            | ( \___/ ) |    | |_ / _` |/ _ \ '__| | | |
             \__/   \__/     |  _| (_| |  __/ |  | |_| |
               _\___/_       |_|  \__,_|\___|_|   \__, |
              (_/   \_)                           |___/

        Faery converts neuromorphic event-based data between formats.
        It can also generate videos, spectrograms, and event rate curves.

        Usage:
            faery [input <input>]
                  [filter <filter> [filter <filter> ...] ]
                  [output <output>]
        
            faery [colormaps | init | run]
                          
            See faery <command> --help for more information on a specific command.

        Examples:
            # Read from stdin and write to stdout
            faery
            
            # Read from file dvs.es and write to stdout
            faery input file dvs.es

            # Read from UDP and write to file out.aedat4
            faery input udp output file out.aedat4

            # Renders available colormaps to map.png
            faery colormaps map.png
        """
        )
    )


def main():
    arguments = sys.argv[1:]
    command_groups: List[commands.CommandGroup] = [
        parser.input_group(),
        parser.output_group(),
        colormaps.colormaps_group(),
        init.init_group(),
        render.render_group(),
    ]
    groups = {p.parser.prog: p for p in command_groups}
    blocks = parser.parse_blocks_recursive(arguments, keywords=set(groups.keys()))
    if blocks == {}:
        print_help()
        return

    input_command = None
    output_command = None
    filters = []
    for block, arguments in blocks.items():
        command_parser = groups[block]
        namespace = command_parser.parser.parse_args(arguments)
        command = command_parser.runner(namespace)
        if isinstance(command, commands.SubCommand):
            if len(blocks) > 1:
                print(
                    f"Error: {block} must be stand-alone, got {len(blocks)} commands: {list(blocks.keys())}"
                )
                return
            else:
                command.run()
                return
        elif isinstance(command, commands.InputCommand):
            input_command = command
        elif isinstance(command, commands.OutputCommand):
            output_command = command
        elif isinstance(command, commands.FilterCommand):
            # TODO
            # ###
            # ### Alexandre: backport convert code for filters / auto-inspect getattr thing
            # ###

            raise NotImplementedError()

    if input_command is None:
        input_command = commands.InputStdin((1280, 720))
    if output_command is None:
        output_command = commands.OutputStdout()

    output_command.to_output(input_command.to_stream())


if __name__ == "__main__":
    main()
