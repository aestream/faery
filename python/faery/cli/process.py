import argparse
import functools
import sys
import typing

import faery

from . import command, list_filters

KEYWORDS: set[str] = {"input", "filter", "output"}


def base_parser(keyword: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser(keyword)
    subparser.parse_args = parser.parse_args
    return subparser


def add_csv_properties(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--no-csv-has-header",
        action="store_const",
        const=False,
        default=True,
        dest="csv_has_header",
    )
    parser.add_argument(
        "--csv-separator",
        default=",",
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-t-index",
        type=int,
        default=0,
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-x-index",
        type=int,
        default=1,
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-y-index",
        type=int,
        default=2,
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-on-index",
        type=int,
        default=3,
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-t-scale",
        type=float,
        default=0.0,
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-on-value",
        default="1",
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--csv-off-value",
        default="0",
        help="(default: %(default)s)",
    )
    parser.add_argument("--csv-skip-errors", action="store_true")


@functools.cache
def input_parser() -> argparse.ArgumentParser:
    parser = base_parser("input")
    subparsers = parser.add_subparsers(required=True, dest="input")
    subparser = subparsers.add_parser("stdin")
    subparser.add_argument(
        "--dimensions",
        type=list_filters.parse_dimensions,
        default=(1280, 720),
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--t0",
        type=list_filters.parse_time,
        default=0,
        help="(default: %(default)s)",
    )
    add_csv_properties(subparser)
    subparser = subparsers.add_parser("file")
    subparser.add_argument("path")
    subparser.add_argument(
        "--track-id",
        type=list_filters.parse_optional_int,
        default="none",
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--dimensions-fallback",
        type=list_filters.parse_dimensions,
        default=(1280, 720),
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--version-fallback",
        choices=list(typing.get_args(faery.EventsFileVersion)) + ["none"],
        default="none",
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--t0",
        type=list_filters.parse_time,
        default=0,
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--file-type",
        choices=list(typing.get_args(faery.EventsFileType)) + ["none"],
        default="none",
        help="(default: %(default)s)",
    )
    add_csv_properties(subparser)
    subparser = subparsers.add_parser("udp")
    subparser.add_argument("address", type=list_filters.parse_udp)
    subparser.add_argument(
        "--dimensions",
        type=list_filters.parse_dimensions,
        default=(1280, 720),
        help="(default: %(default)s)",
    )
    subparser.add_argument(
        "--format",
        choices=list(typing.get_args(faery.UdpFormat)),
        default="t64_x16_y16_on8",
        help="(default: %(default)s)",
    )
    return parser


STREAM_CLASSES_SET = set(list_filters.STREAM_CLASSES)


def find_stream_class(
    stream_class: typing.Any,
    classes_found: set[typing.Any],
) -> typing.Any:
    if stream_class in STREAM_CLASSES_SET:
        return stream_class
    for parent_class in stream_class.__mro__:
        if not parent_class in classes_found:
            classes_found.add(parent_class)
            candidate = find_stream_class(parent_class, classes_found)
            if candidate is not None:
                return candidate
    return None


@functools.cache
def filter_parser(stream_class: typing.Any) -> argparse.ArgumentParser:
    parser = base_parser("filter")
    subparsers = parser.add_subparsers(required=True, dest="filter")
    stream_parent_class = find_stream_class(stream_class, {object, stream_class})
    if stream_parent_class is None:
        raise Exception(
            f"{stream_class} is not a stream class (expected one of {STREAM_CLASSES_SET})"
        )
    name_to_filter = list_filters.class_to_name_to_filter()[stream_parent_class]
    for name, filter in name_to_filter.items():
        subparser = subparsers.add_parser(name.replace("_", "-"))
        for parameter in filter.parameters:
            subparser.add_argument(
                parameter.argparse_flag,
                **parameter.options,
            )

    original_parse_args = parser.parse_args

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: None = None,
    ) -> argparse.Namespace:
        result = original_parse_args(args=args, namespace=namespace)
        for parameter in name_to_filter[result.filter].parameters:
            setattr(
                result,
                parameter.parameter_name,
                parameter.transform(getattr(result, parameter.parameter_name)),
            )
        return result

    parser.parse_args = parse_args.__get__(parser, argparse.ArgumentParser)
    return parser


@functools.cache
def output_parser(
    stream_class: typing.Any,
) -> argparse.ArgumentParser:
    parser = base_parser("output")
    subparsers = parser.add_subparsers(required=True, dest="output")
    stream_parent_class = find_stream_class(stream_class, {object, stream_class})
    if stream_parent_class in {
        faery.EventsStream,
        faery.FiniteEventsStream,
        faery.RegularEventsStream,
        faery.FiniteRegularEventsStream,
    }:

        # Output events stdout
        subparser = subparsers.add_parser("stdout")
        subparser.add_argument(
            "--csv-separator",
            default=",",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--no-csv-header",
            action="store_const",
            const=False,
            default=True,
            dest="csv_header",
        )
        subparser.add_argument("--progress", action="store_true")

        # Output events file
        subparser = subparsers.add_parser("file")
        subparser.add_argument("path")
        subparser.add_argument(
            "--version",
            choices=list(typing.get_args(faery.EventsFileVersion)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--no-zero-t0",
            action="store_const",
            const=False,
            default=True,
            dest="zero_t0",
        )
        subparser.add_argument(
            "--compression-type",
            choices=list(typing.get_args(faery.EventsFileCompression)),
            default="lz4",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--compression-level",
            type=int,
            default=1,
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--csv-separator",
            default=",",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--no-csv-header",
            action="store_const",
            const=False,
            default=True,
            dest="csv_header",
        )
        subparser.add_argument(
            "--file-type",
            choices=list(typing.get_args(faery.EventsFileType)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        if stream_parent_class in {
            faery.FiniteEventsStream,
            faery.FiniteRegularEventsStream,
        }:
            subparser.add_argument(
                "--no-progress",
                action="store_const",
                const=False,
                default=True,
                dest="progress",
            )
        else:
            subparser.add_argument("--progress", action="store_true")

        # Output events UDP
        subparser = subparsers.add_parser("udp")
        subparser.add_argument("address", type=list_filters.parse_udp)
        subparser.add_argument(
            "--payload-length",
            type=list_filters.parse_optional_int,
            default="none",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--format",
            choices=list(typing.get_args(faery.UdpFormat)),
            default="t64_x16_y16_on8",
            help="(default: %(default)s)",
        )
        if stream_parent_class in {
            faery.FiniteEventsStream,
            faery.FiniteRegularEventsStream,
        }:
            subparser.add_argument(
                "--no-progress",
                action="store_const",
                const=False,
                default=True,
                dest="progress",
            )
        else:
            subparser.add_argument("--progress", action="store_true")
    elif stream_parent_class in {
        faery.FrameStream,
        faery.FiniteFrameStream,
        faery.RegularFrameStream,
        faery.FiniteRegularFrameStream,
    }:
        # Output frame file (video)
        subparser = subparsers.add_parser("file")
        subparser.add_argument("path")
        subparser.add_argument(
            "--frame-rate",
            type=float,
            default=60.0,
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--crf",
            type=float,
            default=17.0,
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--preset",
            choices=list(typing.get_args(faery.VideoFilePreset)),
            default="medium",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--tune",
            choices=list(typing.get_args(faery.VideoFileTune)),
            default="none",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--profile",
            choices=list(typing.get_args(faery.VideoFileProfile)),
            default="baseline",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--quality-factor",
            type=list_filters.parse_optional_float,
            default="none",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--rewind",
            action="store_true",
        )
        subparser.add_argument(
            "--skip",
            type=int,
            default=0,
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--file-type",
            choices=list(typing.get_args(faery.VideoFileType)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        if stream_parent_class in {
            faery.FiniteFrameStream,
            faery.FiniteRegularFrameStream,
        }:
            subparser.add_argument(
                "--no-progress",
                action="store_const",
                const=False,
                default=True,
                dest="progress",
            )
        else:
            subparser.add_argument("--progress", action="store_true")

        # Output frame files (frame collection)
        subparser = subparsers.add_parser("files")
        subparser.add_argument("path_pattern", metavar="path-pattern")
        subparser.add_argument(
            "--compression-level",
            choices=list(typing.get_args(faery.ImageFileCompressionLevel)),
            default="fast",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--quality-factor",
            type=list_filters.parse_optional_float,
            default="none",
            help="(default: %(default)s)",
        )
        subparser.add_argument(
            "--file-type",
            choices=list(typing.get_args(faery.ImageFileType)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        if stream_parent_class in {
            faery.FiniteFrameStream,
            faery.FiniteRegularFrameStream,
        }:
            subparser.add_argument(
                "--no-progress",
                action="store_const",
                const=False,
                default=True,
                dest="progress",
            )
        else:
            subparser.add_argument("--progress", action="store_true")
    else:
        raise Exception(f"unsupported stream class {stream_parent_class}")
    return parser


class StreamWrapper:
    def __init__(self):
        self.stream: typing.Any = None

    def parent_class(self) -> typing.Any:
        if self.stream is None:
            return None
        return find_stream_class(self.stream.__class__, {object, self.stream.__class__})

    def ensure_input(self):
        if self.stream is None:
            self.stream = faery.events_stream_from_stdin(dimensions=(1280, 720))

    def set_input(self, arguments: list[str]):
        if self.stream is not None:
            sys.stderr.write(
                f'"input" may only appear at the beginning of a pipeline\n'
            )
            sys.exit(1)
        args = input_parser().parse_args(arguments)
        if args.input == "stdin":
            self.stream = faery.events_stream_from_stdin(
                dimensions=args.dimensions,
                t0=args.t0,
                csv_properties=faery.CsvProperties(
                    has_header=args.csv_has_header,
                    separator=args.csv_separator.encode(),
                    t_index=args.csv_t_index,
                    x_index=args.csv_x_index,
                    y_index=args.csv_y_index,
                    on_index=args.csv_on_index,
                    t_scale=args.csv_t_scale,
                    on_value=args.csv_on_value.encode(),
                    off_value=args.csv_off_value.encode(),
                    skip_errors=args.csv_skip_errors,
                ),
            )
        elif args.input == "file":
            self.stream = faery.events_stream_from_file(
                path=args.path,
                track_id=None if args.track_id == "none" else args.track_id,
                dimensions_fallback=args.dimensions_fallback,
                version_fallback=(
                    None if args.version_fallback == "none" else args.version_fallback
                ),
                t0=args.t0,
                file_type=None if args.file_type == "none" else args.file_type,
                csv_properties=faery.CsvProperties(
                    has_header=args.csv_has_header,
                    separator=args.csv_separator.encode(),
                    t_index=args.csv_t_index,
                    x_index=args.csv_x_index,
                    y_index=args.csv_y_index,
                    on_index=args.csv_on_index,
                    t_scale=args.csv_t_scale,
                    on_value=args.csv_on_value.encode(),
                    off_value=args.csv_off_value.encode(),
                    skip_errors=args.csv_skip_errors,
                ),
            )
        elif args.input == "udp":
            self.stream = faery.events_stream_from_udp(
                dimensions=args.dimensions,
                address=args.address,
                format=args.format,
            )
        else:
            raise Exception(f'unknown input type "{args.input}"')

    def add_filter(self, arguments: list[str]):
        self.ensure_input()
        parser = filter_parser(self.stream.__class__)
        args = parser.parse_args(arguments).__dict__
        filter = args["filter"].replace("-", "_")
        del args["filter"]
        self.stream = getattr(self.stream, filter)(**args)

    def to_output(self, arguments: list[str]):
        self.ensure_input()
        parser = output_parser(self.stream.__class__)
        args = parser.parse_args(arguments).__dict__
        output = args["output"]
        del args["output"]
        parent_class = self.parent_class()
        if args["progress"]:
            args["on_progress"] = faery.progress_bar
        else:
            args["on_progress"] = lambda _: None
        del args["progress"]
        if parent_class in {
            faery.EventsStream,
            faery.FiniteEventsStream,
            faery.RegularEventsStream,
            faery.FiniteRegularEventsStream,
        }:
            if output == "stdout":
                args["csv_separator"] = args["csv_separator"].encode()
                self.stream.to_stdout(**args)
            elif output == "file":
                args["version"] = None if args["version"] == "none" else args["version"]
                args["compression"] = (
                    args["compression_type"],
                    args["compression_level"],
                )
                del args["compression_type"]
                del args["compression_level"]
                args["csv_separator"] = args["csv_separator"].encode()
                args["file_type"] = (
                    None if args["file_type"] == "none" else args["file_type"]
                )
                self.stream.to_file(**args)
            elif output == "udp":
                self.stream.to_udp(**args)
            else:
                raise Exception(f'unknown output type "{output}"')
        elif parent_class in {
            faery.FrameStream,
            faery.FiniteFrameStream,
            faery.RegularFrameStream,
            faery.FiniteRegularFrameStream,
        }:
            if output == "file":
                args["file_type"] = (
                    None if args["file_type"] == "none" else args["file_type"]
                )
                self.stream.to_file(**args)
            elif output == "files":
                args["file_type"] = (
                    None if args["file_type"] == "none" else args["file_type"]
                )
                self.stream.to_files(**args)
            else:
                raise Exception(f'unknown output type "{output}"')
        else:
            raise Exception(f'unexpected parent class "{parent_class}"')

    def to_default_output(self):
        self.ensure_input()
        parent_class = self.parent_class()
        if self.parent_class() in {
            faery.EventsStream,
            faery.FiniteEventsStream,
            faery.RegularEventsStream,
            faery.FiniteRegularEventsStream,
        }:
            self.stream.to_stdout()
        elif parent_class in {
            faery.FrameStream,
            faery.FiniteFrameStream,
            faery.RegularFrameStream,
            faery.FiniteRegularFrameStream,
        }:
            raise Exception(
                'default output is not supported by FrameStream, consider adding "output file <path>" to the pipeline'
            )
        else:
            raise Exception(f'unexpected parent class "{parent_class}"')


def split_on_keywords(arguments: list[str]) -> typing.Iterator[list[str]]:
    subcommand_start_index = 0
    for index, argument in enumerate(arguments):
        if index > 0:
            if argument in KEYWORDS:
                yield arguments[subcommand_start_index:index]
                subcommand_start_index = index
    yield arguments[subcommand_start_index:]


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (
            [
                "",
                "faery [input <input>]",
                "    [filter <filter> [filter <filter> ...]]",
                "    [output <output>]",
            ],
            "process data",
        )

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return KEYWORDS

    @typing.override
    def run(self, arguments: list[str]):
        stream_wrapper = StreamWrapper()
        subcommands_arguments = list(split_on_keywords(arguments))
        default_output = True
        for index, subcommand_arguments in enumerate(subcommands_arguments):
            if subcommand_arguments[0] == "input":
                stream_wrapper.set_input(subcommand_arguments)
            elif subcommand_arguments[0] == "filter":
                stream_wrapper.add_filter(subcommand_arguments)
            elif subcommand_arguments[0] == "output":
                if index < len(subcommands_arguments) - 1:
                    sys.stderr.write(
                        f'"output" may only appear at the end of a pipeline\n'
                    )
                    sys.exit(1)
                default_output = False
                stream_wrapper.to_output(subcommand_arguments)
            else:
                raise Exception(f'unknow subcommand "{subcommand_arguments[0]}"')
        if default_output:
            stream_wrapper.to_default_output()
