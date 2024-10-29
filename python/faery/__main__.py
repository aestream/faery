from argparse import ArgumentParser, Namespace
import sys
from typing import Callable, Dict, List, Optional

from faery.cli import commands


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

        Usage: faery [input <input>]
                     [filter <filter> [filter <filter> ...] ]
                     [output <output>]
        
            See faery <command> --help for more information on a specific command.

        Examples:
            # Read from stdin and write to stdout
            faery
            
            # Read from file dvs.es and write to stdout
            faery input file dvs.es

            # Read from stdin and write to file out.aedat4
            faery input file dvs.es output file out.aedat4
        """
        )
    )


def parse_blocks_recursive(
    argv: List[str],
    blocks: Dict[str, List[str]] = {},
    current_block: Optional[str] = None,
    filter_counter: int = 0,
) -> Dict[str, List[str]]:
    if len(argv) == 0:
        return blocks

    # A set of block keywords
    KEYWORDS = {"input", "output", "filter"}

    head = argv[0]
    tail = argv[1:]

    # Match colormaps
    if head == "colormaps":
        if len(tail) > 0:
            print(
                f"Colormaps does not support additional arguments, {len(tail)} given",
                file=sys.stderr,
            )
        return {"colormaps": []}

    # Match 'regular' I/O blocks
    if head in KEYWORDS:
        if head in blocks:
            raise ValueError(f"Repeated keyword {head}")
        if head == "filter":
            current_block = f"{head}{filter_counter}"
            blocks[current_block] = []
            return parse_blocks_recursive(
                tail, blocks, current_block, filter_counter + 1
            )
        else:
            blocks[head] = []
            return parse_blocks_recursive(tail, blocks, head, filter_counter)
    elif current_block is not None:
        blocks[current_block].append(head)
        return parse_blocks_recursive(tail, blocks, current_block, filter_counter)
    elif current_block is None and head == "--help":
        print_help()
        exit(0)
    else:
        raise ValueError(f"Unexpected argument {head}")


def main():
    arguments = sys.argv[1:]
    blocks = parse_blocks_recursive(arguments)

    if "input" in blocks:
        input_parser = ArgumentParser()
        input_type = input_parser.add_subparsers(dest="type")
        input_file_parser = input_type.add_parser("file")
        input_file_parser.add_argument("filename")
        input_stdin_parser = input_type.add_parser("stdin")
        input_stdin_parser.add_argument(
            "--dimensions", type=int, nargs=2, default=[1280, 720]
        )
        input_udp_parser = input_type.add_parser("udp")
        input_udp_parser.add_argument("--address", type=str, default="localhost")
        input_udp_parser.add_argument("--port", type=int, default=5000)
        input_udp_parser.add_argument("--protocol", type=str, default="spiffer")
        input_udp_parser.add_argument(
            "--dimensions", type=int, nargs=2, default=[1280, 720]
        )

        input_ns = input_parser.parse_args(blocks["input"])
        if input_ns.type == "file":
            input_command = commands.InputFile(input_ns.filename)
        elif input_ns.type == "stdin":
            input_command = commands.InputStdin(input_ns.dimensions)
        elif input_ns.type == "udp":
            input_command = commands.InputUdp(
                input_ns.address, input_ns.dimensions, input_ns.port, input_ns.protocol
            )
        elif input_ns.type is None:
            input_command = commands.InputStdin((1280, 720))
        else:
            raise ValueError(f"Unknown input type {input_ns.type}")
    else:
        input_command = commands.InputStdin((1280, 720))

    input_stream = input_command.to_stream()

    ###
    ### Alexandre: backport convert code for filters / auto-inspect getattr thing
    ###

    if "output" in blocks:
        output_parser = ArgumentParser()
        output_type = output_parser.add_subparsers(dest="type")
        output_file_parser = output_type.add_parser("file")
        output_file_parser.add_argument("filename")

        output_udp_parser = output_type.add_parser("udp")
        output_udp_parser.add_argument("--address", type=str, default="localhost")
        output_udp_parser.add_argument("--port", type=int, default=5000)
        output_udp_parser.add_argument(
            "--protocol",
            type=str,
            default="spiffer",
            choices=["spiffer", "t64_x16_y16_on8", "t32_x16_y15_on1"],
        )

        output_ns = output_parser.parse_args(blocks["output"])
        if output_ns.type == "file":
            output_command = commands.OutputFile(output_ns.filename)
        elif output_ns.type == "udp":
            output_command = commands.OutputUdp(
                output_ns.address, output_ns.port, output_ns.protocol
            )
        elif output_ns.type is None:
            output_command = commands.OutputStdout()
        else:
            raise ValueError(f"Unknown output type {output_ns.type}")
    else:
        output_command = commands.OutputStdout()

    output_command.to_output(input_stream)


if __name__ == "__main__":
    main()
