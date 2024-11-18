# See https://github.com/microsoft/pyright/blob/main/docs/typed-libraries.md#library-interface
# for an explanation of the redundant import statements

from . import colormaps, info, init, list_filters, process, run
from .command import Command as Command

commands_list: list[Command] = [
    process.Command(),
    list_filters.Command(),
    init.Command(),
    run.Command(),
    info.Command(),
    colormaps.Command(),
]

first_block_keyword_to_command: dict[str, Command] = {}
for command_object in commands_list:
    keywords = command_object.first_block_keywords()
    if len(keywords) == 0:
        raise Exception(f"{command_object} has no keywords")
    for keyword in keywords:
        if keyword in first_block_keyword_to_command:
            raise Exception(
                f'first block keyword conflict ("{keyword}" is used by {first_block_keyword_to_command[keyword]} and {command_object})'
            )
        first_block_keyword_to_command[keyword] = command_object


__all__ = ["commands_list", "first_block_keyword_to_command"]
