import argparse


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "init", help="initialize a script to render multiple files"
    )
    parser.add_argument("--configuration")


def run(args: argparse.Namespace):
    pass
