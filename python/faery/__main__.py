import sys
import textwrap

import faery
import faery.cli as cli


def help() -> str:
    usage: list[str] = []
    longest_line = 0
    for command in cli.commands_list:
        for line in command.usage()[0]:
            longest_line = max(longest_line, len(line))
    for command in cli.commands_list:
        lines, description = command.usage()
        for line in lines[:-1]:
            usage.append(f"    {line}")
        usage.append(
            f"    {lines[-1]}{' ' * (longest_line + 2 - len(lines[-1]))}# {description}"
        )
    return "\n".join(
        (
            textwrap.dedent(
                rf"""
                     __       __      _____
                    /  \     /  \    |  ___|_ _  ___ _ __ _   _
                    | ( \___/ ) |    | |_ / _` |/ _ \ '__| | | |
                     \__/   \__/     |  _| (_| |  __/ |  | |_| |
                       _\___/_       |_|  \__,_|\___|_|   \__, |
                      (_/   \_)                           |___/

                Faery converts neuromorphic event-based data between formats.
                It can also generate videos, spectrograms, and event rate curves.

                Usage:
                """
            )[1:-1],
            *usage,
            textwrap.dedent(
                rf"""
                    See faery <command> --help for more information on a specific command.

                Examples:
                    # Read from file dvs.es and write to stdout
                    faery input file dvs.es

                    # Read from UDP and write to file out.aedat4
                    faery input udp output file out.aedat4

                """,
            ),
        )
    )


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(help())
        sys.exit(0)
    if not sys.argv[1] in cli.first_block_keyword_to_command:
        if sys.argv[1] in {"-h", "--help"}:
            sys.stderr.write(help())
            sys.exit(0)
        if sys.argv[1] in {"-v", "--version"}:
            sys.stdout.write(f"Faery {faery.__version__}\n")
            sys.exit(0)
        commands = sorted(cli.first_block_keyword_to_command.keys())
        commands_string = (
            ", ".join(f'"{command}"' for command in commands[:-1])
            + f', and "{commands[-1]}"'
        )
        sys.stderr.write(
            f'unknown command "{sys.argv[1]}" (the available commands are {commands_string})\n'
        )
        sys.exit(1)
    cli.first_block_keyword_to_command[sys.argv[1]].run(sys.argv[1:])


if __name__ == "__main__":
    main()
