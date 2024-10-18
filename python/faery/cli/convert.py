import argparse
import sys
import typing

import faery


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "convert", help="convert between event formats"
    )
    parser.add_argument(
        "input", help='Path of the input file (use "stdin" for standard input)'
    )
    parser.add_argument(
        "-f",
        "--input-format",
        choices=typing.get_args(faery.EventsFileType),
        help="input format, required if the input is standard input",
    )
    parser.add_argument(
        "-i",
        "--track-id",
        type=int,
        help="set the track id for edat files (defaults to the first event stream)",
    )
    parser.add_argument(
        "-j",
        "--input-version",
        choices=typing.get_args(faery.EventsFileVersion),
        help="set the version for evt (.raw) or dat files",
    )
    parser.add_argument(
        "-x",
        "--width",
        type=int,
        help="input width, required if the input format is csv",
    )
    parser.add_argument(
        "-y",
        "--height",
        type=int,
        help="input height, required if the input format is csv",
    )
    parser.add_argument(
        "-t",
        "--t0",
        default=0,
        help="set t0 for Event Stream files (defaults to 0)",
    )
    parser.add_argument(
        "output", help='Path of the input file (use "stdout" for standard input)'
    )
    parser.add_argument(
        "-g",
        "--output-format",
        choices=typing.get_args(faery.EventsFileType),
        help="output format, required if the output is standard output",
    )
    parser.add_argument(
        "-k",
        "--output-version",
        choices=typing.get_args(faery.EventsFileVersion),
        help="set the version for evt (.raw) or dat files",
    )
    parser.add_argument(
        "-z",
        "--zero-t0",
        action="store_true",
        help="offset the output's timestamps so that the first one is zero",
    )
    parser.add_argument(
        "-p",
        "--print-t0",
        action="store_true",
        help="print the output's original t0 (before offset if -z/--zero-t0 is set)",
    )
    parser.add_argument(
        "-c",
        "--compression-type",
        choices=typing.get_args(faery.EventsFileCompression),
        default="lz4",
        help='set the compression algorithm for aedat files (defaults to "lz4")',
    )
    parser.add_argument(
        "-d",
        "--compression-level",
        type=int,
        default=1,
        help="set the compression level for aedat files (defaults to 1)",
    )


def run(args: argparse.Namespace):
    if (args.width is None) != (args.height is None):
        sys.stderr.write(
            "only one of width (-x/--width) and height (-y/--height) was provided (either both or neither should be provided)\n"
        )
        sys.exit(1)

    # @TODO add support for udp
    if args.input == "stdin":
        assert args.input_format is not None
        assert args.input_format == "csv"  # only supported stdin format at the moment
        assert args.width is not None
        assert args.height is not None
        stream = faery.events_stream_from_stdin(dimensions=(args.width, args.height))
    else:
        if args.width is not None and args.height is not None:
            stream = faery.events_stream_from_file(
                path=args.input,
                track_id=args.track_id,
                dimensions_fallback=(args.width, args.height),
                version_fallback=args.input_version,
                t0=args.t0,
                file_type=args.input_format,
            )
        else:
            stream = faery.events_stream_from_file(
                path=args.input,
                track_id=args.track_id,
                version_fallback=args.input_version,
                t0=args.t0,
                file_type=args.input_format,
            )

    if args.output == "stdout":
        assert args.output_format == "csv"  # only supported stdout format at the moment
        stream.to_stdout()
    else:
        t0 = stream.to_file(
            path=args.output,
            version=args.output_version,
            zero_t0=args.zero_t0,
            compression=(args.compression_type, args.compression_level),
            file_type=args.output_format,
        )
        if args.print_t0:
            print(t0)
