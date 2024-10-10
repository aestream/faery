import argparse


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "run", help='execute a script generated with "faery init"'
    )


def run(args: argparse.Namespace):
    pass
