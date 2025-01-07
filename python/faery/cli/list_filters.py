import argparse
import dataclasses
import functools
import inspect
import re
import typing

import faery
from faery import format_bold, format_color

from . import command

STREAM_CLASSES: list[typing.Any] = [
    faery.EventsStream,
    faery.FiniteEventsStream,
    faery.RegularEventsStream,
    faery.FiniteRegularEventsStream,
    faery.FrameStream,
    faery.FiniteFrameStream,
    faery.RegularFrameStream,
    faery.FiniteRegularFrameStream,
]

NONE_KEYWORDS: set[str] = {"None", "none", "Null", "null", "Nil", "nil"}
PUNCTUATION_COLOR = "blue"
LITERAL_COLOR = "cyan"
TYPE_COLOR = "magenta"
FLOAT_COLOR_PATTERN = re.compile(
    r"^\s*\(?\s*(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)\s*[,\s]\s*((?:\d+\.?\d*)\s*[,\s]\s*)?(\d+\.?\d*)\s*\)?\s*$"
)
HEX_COLOR_PATTERN = re.compile(r"^\s*(#?[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?)\s*$")
DIMENSIONS_PATTERN = re.compile(r"^\s*\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?\s*$")
UDP_PATTERN = re.compile(r"^\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5})\s*$")


@dataclasses.dataclass
class Parameter:
    parameter_name: str
    argparse_flag: str
    representation: str
    options: dict[str, typing.Any]
    transform: typing.Callable[[typing.Any], typing.Any]


@dataclasses.dataclass
class Filter:
    parameters: list[Parameter]
    return_annotation: str


def parse_bool(string: str) -> bool:
    if string == "true" or string == "True" or string == "1":
        return True
    if string == "false" or string == "False" or string == "0":
        return False
    raise argparse.ArgumentTypeError(
        f'parsing "{string}" failed (expected "true" or "false")'
    )


def parse_time(string: str) -> faery.Time:
    if all(character.isdigit() for character in string):
        return int(string) * faery.s
    if any(character == ":" for character in string):
        return faery.parse_time(string)
    return float(string) * faery.s


def parse_optional_time(string: str) -> typing.Optional[faery.Time]:
    if string in NONE_KEYWORDS:
        return None
    return parse_time(string)


def parse_color(string: str) -> faery.Color:
    float_color_match = FLOAT_COLOR_PATTERN.match(string)
    if float_color_match is not None:
        if float_color_match[3] is None:
            color = (
                float(float_color_match[1]),
                float(float_color_match[2]),
                float(float_color_match[3]),
            )
        else:
            color = (
                float(float_color_match[1]),
                float(float_color_match[2]),
                float(float_color_match[3].replace(", ", " ").strip()),
                float(float_color_match[4]),
            )
        for component in color:
            if component > 1.0:
                raise argparse.ArgumentTypeError(
                    f"color components must be in the range [0.0, 1.0] (got {color})"
                )
        return color
    hex_color_match = HEX_COLOR_PATTERN.match(string)
    if hex_color_match is not None:
        color = string.strip()
        if color.startswith("#"):
            return color
        return f"#{color}"
    raise argparse.ArgumentTypeError(
        f'parsing "{string}" failed (expected "(r, g, b)", "(r, g, b, a)", "#RRGGBB", or "#RRGGBBAA")'
    )


def parse_dimensions(string: str) -> tuple[int, int]:
    dimensions_match = DIMENSIONS_PATTERN.match(string)
    if dimensions_match is not None:
        return (int(dimensions_match[1]), int(dimensions_match[2]))
    raise argparse.ArgumentTypeError(
        f'parsing "{string}" failed (expected "(width, height)")'
    )


def parse_factor_or_minimum_dimensions(
    string: str,
) -> typing.Union[float, tuple[int, int]]:
    dimensions_match = DIMENSIONS_PATTERN.match(string)
    if dimensions_match is not None:
        return (int(dimensions_match[1]), int(dimensions_match[2]))
    return float(string)


def parse_optional_color(string: str) -> typing.Optional[faery.Color]:
    if string in NONE_KEYWORDS:
        return None
    return parse_color(string)


def parse_optional_float(string: str) -> typing.Optional[float]:
    if string in NONE_KEYWORDS:
        return None
    return float(string)


def parse_optional_int(string: str) -> typing.Optional[int]:
    if string in NONE_KEYWORDS:
        return None
    return int(string)


def parse_udp(
    string: str,
) -> typing.Union[
    tuple[str, int], tuple[str, int, typing.Optional[int], typing.Optional[str]]
]:
    match = UDP_PATTERN.match(string)
    if match is not None:
        return (match[1], int(match[2]))
    raise argparse.ArgumentTypeError(
        f'parsing "{string}" failed (expected "a.b.c.d:port")'
    )


name_to_colormaps = faery.name_to_colormaps()
colormaps_names = list(name_to_colormaps.keys())


@functools.cache
def class_to_name_to_filter() -> dict[typing.Any, dict[str, Filter]]:
    IGNORED_FUNCTIONS: set[str] = {
        "map",
        "mask",
    }
    result: dict[typing.Any, dict[str, Filter]] = {}
    for stream_class in STREAM_CLASSES:
        name_to_filter: dict[str, Filter] = {}
        for function_name, function in inspect.getmembers(
            stream_class, predicate=inspect.isfunction
        ):
            if (
                not hasattr(function, "filter_return_annotation")
                or function_name in IGNORED_FUNCTIONS
            ):
                continue
            signature = inspect.signature(function)
            if len(signature.parameters) == 0:
                raise Exception(
                    f'filter {function_name} of {stream_class} has no parameters (it should have at least "self")'
                )
            filter = Filter(
                parameters=[],
                return_annotation=getattr(function, "filter_return_annotation"),
            )
            parameters_items = iter(signature.parameters.items())
            if next(parameters_items)[0] != "self":
                continue
            for parameter_name, parameter in parameters_items:
                if parameter.annotation in {int, "int", float, "float", str, "str"}:
                    options = {
                        "type": parameter.annotation,
                    }
                    if isinstance(parameter.annotation, str):
                        type_representation = parameter.annotation
                    else:
                        type_representation = parameter.annotation.__name__
                    if parameter.default == inspect._empty:
                        argparse_flag = parameter_name
                        options["metavar"] = parameter_name.replace("_", "-")
                        representation = "{} {}{}".format(
                            format_color(f"<{options['metavar']}", PUNCTUATION_COLOR),
                            format_color(f"({type_representation})", TYPE_COLOR),
                            format_color(">", PUNCTUATION_COLOR),
                        )
                    else:
                        argparse_flag = f'--{parameter_name.replace("_", "-")}'
                        options["default"] = parameter.default
                        options["help"] = "(default: %(default)s)"
                        options["dest"] = parameter_name
                        representation = "{}{} {} {} {}{}".format(
                            format_color("[", PUNCTUATION_COLOR),
                            format_color(argparse_flag, LITERAL_COLOR),
                            format_color(f"({type_representation})", TYPE_COLOR),
                            format_color("=", PUNCTUATION_COLOR),
                            format_color(parameter.default, LITERAL_COLOR),
                            format_color("]", PUNCTUATION_COLOR),
                        )
                    filter.parameters.append(
                        Parameter(
                            parameter_name=parameter_name,
                            argparse_flag=argparse_flag,
                            representation=representation,
                            options=options,
                            transform=lambda value: value,
                        )
                    )
                elif parameter.annotation in {bool, "bool"}:
                    if parameter.default == inspect._empty:
                        filter.parameters.append(
                            Parameter(
                                parameter_name=parameter_name,
                                argparse_flag=parameter_name,
                                representation="{} {}{}".format(
                                    format_color(
                                        f"<{parameter_name}", PUNCTUATION_COLOR
                                    ),
                                    format_color("(bool)", TYPE_COLOR),
                                    format_color(">", PUNCTUATION_COLOR),
                                ),
                                options={
                                    "type": parse_bool,
                                    "metavar": parameter_name.replace("_", "-"),
                                },
                                transform=lambda value: value,
                            )
                        )
                    else:
                        argparse_flag = (
                            f"--{'no-' if parameter.default else ''}{parameter_name}"
                        )
                        filter.parameters.append(
                            Parameter(
                                parameter_name=parameter_name,
                                argparse_flag=argparse_flag,
                                representation="{}{}{}".format(
                                    format_color(f"[", PUNCTUATION_COLOR),
                                    format_color(parameter_name, LITERAL_COLOR),
                                    format_color("]", PUNCTUATION_COLOR),
                                ),
                                options={
                                    "action": "store_const",
                                    "default": parameter.default,
                                    "help": "(default: %(default)s)",
                                    "const": not parameter.default,
                                    "dest": parameter_name,
                                },
                                transform=lambda value: value,
                            )
                        )
                elif parameter.annotation in {
                    faery.Time,
                    str(faery.Time),
                    typing.Optional[faery.Time],
                    str(typing.Optional[faery.Time]),
                    faery.Color,
                    str(faery.Color),
                    typing.Optional[faery.Color],
                    str(typing.Optional[faery.Color]),
                    typing.Union[float, tuple[int, int]],
                    str(typing.Union[float, tuple[int, int]]),
                    typing.Optional[float],
                }:
                    options: dict[str, typing.Any] = {}
                    if parameter.annotation in {faery.Time, str(faery.Time)}:
                        options["type"] = parse_time
                        type_representation = "(int | float | hh:mm:ss.µµµµµµ)"
                    elif parameter.annotation in {
                        typing.Optional[faery.Time],
                        str(typing.Optional[faery.Time]),
                    }:
                        options["type"] = parse_optional_time
                        type_representation = "(int | float | hh:mm:ss.µµµµµµ | none)"
                    elif parameter.annotation in {faery.Color, str(faery.Color)}:
                        options["type"] = parse_color
                        type_representation = (
                            "((r, g, b) | (r, g, b, a) | #RRGGBB | #RRGGBBAA)"
                        )
                    elif parameter.annotation in {
                        typing.Optional[faery.Color],
                        str(typing.Optional[faery.Color]),
                    }:
                        options["type"] = parse_optional_color
                        type_representation = (
                            "((r, g, b) | (r, g, b, a) | #RRGGBB | #RRGGBBAA | none)"
                        )
                    elif parameter.annotation in {
                        typing.Union[float, tuple[int, int]],
                        str(typing.Union[float, tuple[int, int]]),
                    }:
                        options["type"] = parse_factor_or_minimum_dimensions
                        type_representation = "(float | (width, height))"
                    elif parameter.annotation in {
                        typing.Optional[float],
                        str(typing.Optional[float]),
                    }:
                        options["type"] = parse_optional_float
                        type_representation = "(float | none)"
                    else:
                        raise Exception(f"{parameter.annotation} is not implemented")
                    if parameter.default == inspect._empty:
                        argparse_flag = parameter_name
                        options["metavar"] = parameter_name.replace("_", "-")
                        representation = "{} {}{}".format(
                            format_color(f"<{options['metavar']}", PUNCTUATION_COLOR),
                            format_color(type_representation, TYPE_COLOR),
                            format_color(">", PUNCTUATION_COLOR),
                        )
                    else:
                        argparse_flag = f'--{parameter_name.replace("_", "-")}'
                        options["dest"] = parameter_name
                        representation = "{}{} {} {} {}{}".format(
                            format_color("[", PUNCTUATION_COLOR),
                            format_color(argparse_flag, LITERAL_COLOR),
                            format_color(type_representation, TYPE_COLOR),
                            format_color("=", PUNCTUATION_COLOR),
                            format_color(
                                (
                                    "none"
                                    if parameter.default is None
                                    else parameter.default
                                ),
                                LITERAL_COLOR,
                            ),
                            format_color("]", PUNCTUATION_COLOR),
                        )
                        options["default"] = parameter.default
                        options["help"] = "(default: %(default)s)"
                    filter.parameters.append(
                        Parameter(
                            parameter_name=parameter_name,
                            argparse_flag=argparse_flag,
                            representation=representation,
                            options=options,
                            transform=lambda value: value,
                        )
                    )
                elif parameter.annotation in {faery.Colormap, "Colormap"}:
                    options = {"choices": colormaps_names}
                    choices_representation = "{}{}{}".format(
                        format_color("{", PUNCTUATION_COLOR),
                        format_color(", ", PUNCTUATION_COLOR).join(
                            format_color(choice, LITERAL_COLOR)
                            for choice in colormaps_names
                        ),
                        format_color("}", PUNCTUATION_COLOR),
                    )
                    if parameter.default == inspect._empty:
                        argparse_flag = parameter_name
                        options["metavar"] = parameter_name.replace("_", "-")
                        representation = "{} {}{}".format(
                            format_color(f"<{options['metavar']}", PUNCTUATION_COLOR),
                            choices_representation,
                            format_color(">", PUNCTUATION_COLOR),
                        )
                    else:
                        argparse_flag = f'--{parameter_name.replace("_", "-")}'
                        options["default"] = parameter.default
                        options["help"] = "(default: %(default)s)"
                        options["dest"] = parameter_name
                        representation = "{}{} {} {} {}{}".format(
                            format_color("[", PUNCTUATION_COLOR),
                            format_color(argparse_flag, LITERAL_COLOR),
                            choices_representation,
                            format_color("=", PUNCTUATION_COLOR),
                            format_color(
                                (
                                    "none"
                                    if parameter.default is None
                                    else parameter.default
                                ),
                                LITERAL_COLOR,
                            ),
                            format_color("]", PUNCTUATION_COLOR),
                        )
                    filter.parameters.append(
                        Parameter(
                            parameter_name=parameter_name,
                            argparse_flag=argparse_flag,
                            representation=representation,
                            options=options,
                            transform=lambda value: name_to_colormaps[value],
                        )
                    )
                else:
                    found = False
                    for _, member in inspect.getmembers(faery.enums):
                        if typing.get_origin(member) == typing.Literal:
                            if parameter.annotation in {member, str(member)}:
                                choices = list(typing.get_args(member))
                                options = {"choices": choices}
                                choices_representation = "{}{}{}".format(
                                    format_color("{", PUNCTUATION_COLOR),
                                    format_color(", ", PUNCTUATION_COLOR).join(
                                        format_color(choice, LITERAL_COLOR)
                                        for choice in choices
                                    ),
                                    format_color("}", PUNCTUATION_COLOR),
                                )
                                if parameter.default == inspect._empty:
                                    argparse_flag = parameter_name
                                    options["metavar"] = parameter_name.replace(
                                        "_", "-"
                                    )
                                    representation = "{} {}{}".format(
                                        format_color(
                                            f"<{options['metavar']}", PUNCTUATION_COLOR
                                        ),
                                        choices_representation,
                                        format_color(">", PUNCTUATION_COLOR),
                                    )
                                else:
                                    argparse_flag = (
                                        f'--{parameter_name.replace("_", "-")}'
                                    )
                                    options["default"] = parameter.default
                                    options["help"] = "(default: %(default)s)"
                                    options["dest"] = parameter_name
                                    representation = "{}{} {} {} {}{}".format(
                                        format_color("[", PUNCTUATION_COLOR),
                                        format_color(argparse_flag, LITERAL_COLOR),
                                        choices_representation,
                                        format_color("=", PUNCTUATION_COLOR),
                                        format_color(
                                            (
                                                "none"
                                                if parameter.default is None
                                                else parameter.default
                                            ),
                                            LITERAL_COLOR,
                                        ),
                                        format_color("]", PUNCTUATION_COLOR),
                                    )
                                filter.parameters.append(
                                    Parameter(
                                        parameter_name=parameter_name,
                                        argparse_flag=argparse_flag,
                                        representation=representation,
                                        options=options,
                                        transform=lambda value: value,
                                    )
                                )
                                found = True
                                break
                    if not found:
                        raise Exception(
                            f"unsupported parameter type {parameter.annotation} in filter {function_name} of {stream_class}"
                        )
            name_to_filter[function_name.replace("_", "-")] = filter
        result[stream_class] = name_to_filter

    return result


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (["faery list-filters"], "print available filters")

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return {"list-filters"}

    @typing.override
    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.parse_args(args=arguments)
        for stream_class, name_to_filter in class_to_name_to_filter().items():
            title = f"{stream_class.__name__} filters"
            print(format_bold(f"{title}\n{'=' * len(title)}\n"))
            for name, filter in name_to_filter.items():
                print(
                    "{}{}{} {}".format(
                        name,
                        "" if len(filter.parameters) == 0 else " ",
                        " ".join(
                            parameter.representation for parameter in filter.parameters
                        ),
                        format_bold(f"-> {filter.return_annotation}"),
                    )
                )
            print("\n")
