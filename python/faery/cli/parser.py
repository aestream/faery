from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Set

from . import commands


def parse_blocks_recursive(
    argv: List[str],
    keywords: Set[str],
    blocks: Dict[str, List[str]] = {},
    current_block: Optional[str] = None,
    filter_counter: int = 0,
) -> Dict[str, List[str]]:
    if len(argv) == 0:
        return blocks

    head = argv[0]
    tail = argv[1:]

    if head in keywords:
        if head in blocks:
            raise ValueError(f"Repeated keyword {head}")
        if head == "filter":
            current_block = f"{head}{filter_counter}"
            blocks[current_block] = []
            return parse_blocks_recursive(
                tail,
                keywords=keywords,
                blocks=blocks,
                current_block=current_block,
                filter_counter=filter_counter + 1,
            )
        else:
            blocks[head] = []
            return parse_blocks_recursive(
                tail,
                keywords=keywords,
                blocks=blocks,
                current_block=head,
                filter_counter=filter_counter,
            )
    elif current_block is not None:
        blocks[current_block].append(head)
        return parse_blocks_recursive(
            tail,
            keywords=keywords,
            blocks=blocks,
            current_block=current_block,
            filter_counter=filter_counter,
        )
    elif current_block is None and head == "--help":
        return {}
    else:
        raise ValueError(f"Unexpected argument {head}")


def input_group() -> commands.PipelineCommandGroup:
    def parse_input(input_ns: Namespace) -> commands.InputCommand:
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
        return input_command

    parser = ArgumentParser("input", description="Reads input from a source")
    input_type = parser.add_subparsers(dest="type")
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
    return commands.PipelineCommandGroup(parser, parse_input)


def output_group() -> commands.PipelineCommandGroup:
    def parse_output(output_ns: Namespace) -> commands.OutputCommand:
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

        return output_command

    parser = ArgumentParser("output", description="Writes output to a destination")
    output_type = parser.add_subparsers(dest="type")
    output_file_parser = output_type.add_parser("file")
    output_file_parser.add_argument("filename")

    output_udp_parser = output_type.add_parser("udp")
    output_udp_parser.add_argument("--address", type=str, default="localhost")
    output_udp_parser.add_argument("--port", type=int, default=5000)
    output_udp_parser.add_argument(
        "--protocol",
        type=str,
        default="t64_x16_y16_on8",
        choices=["t64_x16_y16_on8", "t32_x16_y15_on1"],
    )
    return commands.PipelineCommandGroup(parser, parse_output)
