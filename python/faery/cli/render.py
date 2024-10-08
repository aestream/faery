import argparse


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "render", help="generate a video or frames from an event file"
    )
