def main():

    import argparse
    import textwrap

    import faery.cli

    parser = argparse.ArgumentParser(
        prog="faery",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            rf"""
                 __       __      _____
                /  \     /  \    |  ___|_ _  ___ _ __ _   _
                | ( \___/ ) |    | |_ / _` |/ _ \ '__| | | |
                 \__/   \__/     |  _| (_| |  __/ |  | |_| |
                   _\___/_       |_|  \__,_|\___|_|   \__, |
                  (_/   \_)                           |___/

            Faery converts Neuromorphic camera data (events) between formats.
            It can also generate videos, spectrograms, and event rate curves.
            """
        ),
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"Faery {faery.__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")
    faery.cli.convert.add_to_subparsers(subparsers)
    faery.cli.render.add_to_subparsers(subparsers)
    faery.cli.init.add_to_subparsers(subparsers)
    faery.cli.run.add_to_subparsers(subparsers)
    faery.cli.inline.add_to_subparsers(subparsers)
    faery.cli.colormaps.add_to_subparsers(subparsers)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        getattr(faery.cli, args.command).run(args)


if __name__ == "__main__":
    main()
