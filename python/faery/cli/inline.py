import argparse

def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser("inline", help="run a Faery pipeline from the command line")
