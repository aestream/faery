import argparse


class Command:
    """
    Base class (interface) for all faery subcommands (faery input, faery colormaps, ...).
    """

    def usage(self) -> tuple[list[str], str]:
        """
        Basic help that describes this command, printed in the common faery help message.

        Must return a list of usage lines and a short description.
        """
        raise NotImplementedError()

    def first_block_keywords(self) -> set[str]:
        """
        Must return all the keywords that trigger this subcommand
        (that is, keywords that can appear right after `faery` on the command line).

        First block keywords must be unique across faery subcommands.
        This is checked at runtime in cli.__init__.py.

        The *first block* keywords `-h`, `--help`, `-v`, and `--version` are reserved and must not appear in this list.
        """
        raise NotImplementedError()

    def run(self, arguments: list[str]):
        """
        Called by __main__.py if the first keyword of a command matched any of the first block keywords.

        `arguments` is identical to `sys.argv[1:]` (all the arguments besides the program name).
        This includes the keyword that matched one of `first_block_keywords`.

        The command is responsible for parsing and validating its arguments, including the presence of `--help`.
        """
        raise NotImplementedError()

    def parser(self) -> argparse.ArgumentParser:
        """
        Utility function that returns an argparse subparser.

        It may only be used if `first_block_keywords` returns one (and only one) keyword.

        Using this parser in `run` is not mandatory (some subcommands need more flexibility, especially if they support more than one keyword),
        but it is encouraged where possible to increase the uniformity of help messages.
        """
        keywords = self.first_block_keywords()
        if len(keywords) != 1:
            raise Exception(
                "Command.parser is only compatible with commands that have a single keyword (manually instantiate a parser for more complex commands)"
            )
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparser = subparsers.add_parser(next(iter(keywords)))
        subparser.parse_args = parser.parse_args
        return subparser
