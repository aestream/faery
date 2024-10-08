import argparse

parser = argparse.ArgumentParser(prog="faery")
parser.add_argument("--version", "-v", action="store_true", help="Display the library version and exit")
args = parser.parse_args()

if args.version:


    print(
"""Faery is a library that ferries event data and frames from A to B.
By default, we stream from STDIN to STDOUT without modifying the data,
but the behavior can be customized using the following pattern:

\b
    faery [input ...] [filter ...] [output ...]

\b
                    __/___                 _____
                _____/______|               |  ___|_ _  ___ _ __ _   _
_______    _____/_____\\_______\\_____        | |_ / _` |/ _ \\ '__| | | |
\\      \\  |               |      \\   \\      |  _| (_| |  __/ |  | |_| |
~~~~~~~~~~~~~~ ~~~~~ ~~~ ~~~~~~~~ ~~~~~ ~~~  |_|  \\__,_|\\___|_|   \\__, |
    ~~~~  ~~~~~   ~~~   ~~  ~~~~~  ~~~ ~~                        |___/
\n""",
    )
