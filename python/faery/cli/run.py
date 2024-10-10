import argparse
import os
import pathlib
import subprocess
import sys


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "run", help='execute a script generated with "faery init"'
    )
    parser.add_argument(
        "-c",
        "--configuration",
        default=str(pathlib.Path(os.getcwd()) / "faery_configuration.py"),
        help="path of the configuration file",
    )


def run(args: argparse.Namespace):
    subprocess.run([sys.executable, args.configuration])
