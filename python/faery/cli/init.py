import argparse
import os
import pathlib


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "init", help="initialize a script to render multiple files"
    )
    parser.add_argument(
        "-c",
        "--configuration",
        default=str(pathlib.Path(os.getcwd()) / "faery_configuration.py"),
        help="path of the configuration file",
    )
    parser.add_argument(
        "-s",
        "--scan",
        default=str(pathlib.Path(os.getcwd()) / "recordings"),
        help="**/*.",
    )
    parser.add_argument(
        "-f",
        "--input-format",
        choices=[
            "aedat",
            "csv",
            "dat",
            "es",
            "evt",
        ],
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
        choices=["dat1", "dat2", "evt2", "evt2.1", "evt3"],
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


def run(args: argparse.Namespace):
    pass
