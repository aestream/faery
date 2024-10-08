import argparse

def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser("convert", help="convert between event formats")
    parser.add_argument("--input")
