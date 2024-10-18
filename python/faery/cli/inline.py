import argparse
import collections
import dataclasses
import inspect
import os
import sys
import typing

import faery


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "inline", help="run a Faery pipeline from the command line"
    )
    parser.add_argument(
        "pipeline",
        nargs="*",
        help="Path of the input file (defaults to standard input)",
    )
    parser.add_argument(
        "-i", "--input", help="path of the input file (defaults to standard input)"
    )
    parser.add_argument(
        "-f",
        "--input-format",
        choices=[
            "aedat",
            "csv",
            "dat",
            "es",
            "evt",
        ],
        help="input format, required if the input is standard input",
    )
    parser.add_argument(
        "-x",
        "--width",
        type=int,
        help="input width, required if the input format is csv",
    )
    parser.add_argument(
        "-y",
        "--height",
        type=int,
        help="input height, required if the input format is csv",
    )
    parser.add_argument(
        "-o", "--output", help="path of the output file (defaults to standard output)"
    )
    parser.add_argument(
        "-g",
        "--output-format",
        choices=[
            "aedat",
            "csv",
            "dat",
            "es",
            "evt",
        ],
        help="output format, required if the output is standard output",
    )
    parser.add_argument(
        "-l", "--filters", action="store_true", help="print a list of filters and exit"
    )


ANSI_COLORS_ENABLED = os.getenv("ANSI_COLORS_DISABLED") is None
ERROR_LINE_MAXIMUM_LENGTH: int = 80
ERROR_LINE_LEFT_PADDING: int = 4
ERROR_LINE_RIGHT_EXTENT: int = 4
ERROR_LINE_ELLIPSIS: str = "..."
PARAMETER_CHARACTERS: set[str] = {"+", "-", ".", ":"}
IGNORED_FUNCTIONS: set[str] = {
    "map",
    "mask",
}
STREAM_CLASSES: list[typing.Any] = [
    faery.EventsStream,
    faery.FiniteEventsStream,
    faery.RegularEventsStream,
    faery.FiniteRegularEventsStream,
]


@dataclasses.dataclass
class FilterParameter:
    type: str
    parse: typing.Callable[[str], typing.Any]
    default: typing.Optional[str]


@dataclasses.dataclass
class Filter:
    return_annotation: str
    name_to_parameter: collections.OrderedDict[str, FilterParameter]


class_to_name_to_filter: dict[typing.Any, dict[str, Filter]] = {}

stream_classes_names: set[str] = set(
    [stream_class.__name__ for stream_class in STREAM_CLASSES]
)

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
            return_annotation=getattr(function, "filter_return_annotation"),
            name_to_parameter=collections.OrderedDict(),
        )
        parameters_items = iter(signature.parameters.items())
        if next(parameters_items)[0] != "self":
            continue
        for parameter_name, parameter in parameters_items:
            if parameter.annotation == int or parameter.annotation == "int":
                filter.name_to_parameter[parameter_name] = FilterParameter(
                    type="int",
                    parse=int,
                    default=(
                        None
                        if parameter.default == inspect._empty
                        else str(parameter.default)
                    ),
                )
            elif parameter.annotation == bool or parameter.annotation == "bool":

                def parse(string: str):
                    if string == "true" or string == "True" or string == "1":
                        return True
                    if string == "false" or string == "False" or string == "0":
                        return False
                    raise Exception(
                        f'parsing {string} failed (expected "true" or "false")'
                    )

                filter.name_to_parameter[parameter_name] = FilterParameter(
                    type="bool",
                    parse=parse,
                    default=(
                        None
                        if parameter.default == inspect._empty
                        else str(parameter.default)
                    ),
                )
            elif parameter.annotation == faery.Time or parameter.annotation == str(
                faery.Time
            ):
                filter.name_to_parameter[parameter_name] = FilterParameter(
                    type="time",
                    parse=lambda string: (
                        int(string)
                        if all(character.isdigit() for character in string)
                        else (
                            string
                            if any(character == ":" for character in string)
                            else float(string)
                        )
                    ),
                    default=(
                        None
                        if parameter.default == inspect._empty
                        else str(parameter.default)
                    ),
                )
            elif parameter.annotation == typing.Optional[
                faery.Time
            ] or parameter.annotation == str(typing.Optional[faery.Time]):
                filter.name_to_parameter[parameter_name] = FilterParameter(
                    type="optional time",
                    parse=lambda string: (
                        int(string)
                        if all(character.isdigit() for character in string)
                        else (
                            string
                            if any(character == ":" for character in string)
                            else float(string)
                        )
                    ),
                    default=(
                        None
                        if parameter.default == inspect._empty
                        else str(parameter.default)
                    ),
                )
            else:
                found = False
                for name, member in inspect.getmembers(faery.enums):
                    if typing.get_origin(member) == typing.Literal:
                        if (
                            parameter.annotation == member
                            or parameter.annotation == str(member)
                        ):
                            split_name = ""
                            for character in name:
                                if (
                                    character.isupper()
                                    and len(split_name) > 0
                                    and split_name[-1] != " "
                                ):
                                    split_name += f" {character.lower()}"
                                else:
                                    split_name += character.lower()
                            filter.name_to_parameter[parameter_name] = FilterParameter(
                                type=split_name,
                                parse=lambda string: string,
                                default=(
                                    None
                                    if parameter.default == inspect._empty
                                    else str(parameter.default)
                                ),
                            )
                            found = True
                            break
                if not found:
                    raise Exception(
                        f"unsupported parameter type {parameter.annotation} in filter {function_name} of {stream_class}"
                    )
        name_to_filter[function_name] = filter
    class_to_name_to_filter[stream_class] = name_to_filter


def print_filters():
    for stream_class in STREAM_CLASSES:
        if ANSI_COLORS_ENABLED:
            class_name = f"\033[1m{stream_class.__name__}\033[0m"
        else:
            class_name = stream_class.__name__
        sys.stdout.write(f"{class_name}\n")
        for name, filter in class_to_name_to_filter[stream_class].items():
            parameters: list[str] = []
            for parameter_name, parameter in filter.name_to_parameter.items():
                if ANSI_COLORS_ENABLED:
                    type = f"\033[34m({parameter.type})\033[0m"
                else:
                    type = f"({parameter.type})"
                if parameter.default is None:
                    parameters.append(f"{parameter_name} {type}")
                else:
                    parameters.append(f"[{parameter_name} {type}]")
            if ANSI_COLORS_ENABLED:
                name_string = f"\033[1m{name}\033[0m"
            else:
                name_string = name
            if ANSI_COLORS_ENABLED:
                return_annotation = f"\033[2m-> {filter.return_annotation}\033[0m"
            else:
                return_annotation = f"-> {filter.return_annotation}"
            if len(parameters) == 0:
                sys.stdout.write(f"    {name_string} {return_annotation}\n")
            else:
                sys.stdout.write(
                    f"    {name_string} {' '.join(parameters)} {return_annotation}\n"
                )
        if stream_class != STREAM_CLASSES[-1]:
            sys.stdout.write(f"\n")


@dataclasses.dataclass
class StringFilter:
    start: int
    name: str
    args: list[str]
    kwargs: dict[str, str]

    def add_to(self, stream: typing.Any) -> typing.Any:
        stream_class = None
        for parent_class in stream.__class__.__mro__:
            if parent_class in STREAM_CLASSES:
                stream_class = parent_class
                break
        assert stream_class is not None
        name_to_filter = class_to_name_to_filter[stream_class]
        if not self.name in name_to_filter:
            raise Exception(
                f"unknown filter \"{self.name}\" ({stream_class.__name__} supports the following filters: {', '.join(name_to_filter.keys())})"
            )
        filter = name_to_filter[self.name]
        parsed_args: list[typing.Any] = []
        parameters = list(filter.name_to_parameter.values())
        for index, arg in enumerate(self.args):
            if index >= len(parameters):
                raise Exception(
                    f'too many parameters ("{self.name}" expects {len(parameters)} parameters)'
                )
            parsed_args.append(parameters[index].parse(arg))
        parsed_kwargs: dict[str, typing.Any] = {}
        for name, arg in self.kwargs.items():
            if not name in filter.name_to_parameter:
                raise Exception(
                    f"unknwon parameters \"{name}\" (\"{self.name}\" expects the following parameters: {', '.join(filter.name_to_parameter.keys()) })"
                )
            parsed_kwargs[name] = filter.name_to_parameter[name].parse(arg)
        return getattr(stream, self.name)(*parsed_args, **parsed_kwargs)


class ParseError(Exception):
    def __init__(
        self,
        pipeline: str,
        error_start: int,
        error_position: int,
        error_pointer: bool,
        error: str,
    ):
        assert error_start <= error_position
        error_end = error_position + 1 if error_pointer else error_position
        assert error_end <= len(pipeline)

        self.lines: list[str] = []

        pipeline_error = pipeline[error_start:error_end]
        if not all(character.isspace() for character in pipeline_error):
            while len(pipeline_error) > 0 and pipeline_error[0].isspace():
                error_start += 1
                pipeline_error = pipeline[error_start:error_end]
            if not error_pointer:
                while len(pipeline_error) > 0 and pipeline_error[-1].isspace():
                    error_position -= 1
                    error_end -= 1
                    pipeline_error = pipeline[error_start:error_end]
        if len(pipeline) + ERROR_LINE_LEFT_PADDING <= ERROR_LINE_MAXIMUM_LENGTH:
            self.lines.append(f"{' ' * ERROR_LINE_LEFT_PADDING}{pipeline}")
        elif (
            error_end
            + ERROR_LINE_LEFT_PADDING
            + ERROR_LINE_RIGHT_EXTENT
            + len(ERROR_LINE_ELLIPSIS)
            <= ERROR_LINE_MAXIMUM_LENGTH
        ):
            self.lines.append(
                f"{' ' * ERROR_LINE_LEFT_PADDING}{pipeline[0:error_end + ERROR_LINE_RIGHT_EXTENT]}{ERROR_LINE_ELLIPSIS}"
            )
        elif error_end + ERROR_LINE_RIGHT_EXTENT >= len(pipeline):
            trim = len(pipeline) - (ERROR_LINE_MAXIMUM_LENGTH - ERROR_LINE_LEFT_PADDING)
            self.lines.append(
                f"{' ' * ERROR_LINE_LEFT_PADDING}{ERROR_LINE_ELLIPSIS}{pipeline[trim + len(ERROR_LINE_ELLIPSIS):]}"
            )
            error_start -= trim
            if error_start < 0:
                error_start = len(ERROR_LINE_ELLIPSIS)
            error_position -= trim
        else:
            trim = (
                error_end
                + ERROR_LINE_RIGHT_EXTENT
                - (
                    ERROR_LINE_MAXIMUM_LENGTH
                    - ERROR_LINE_LEFT_PADDING
                    - len(ERROR_LINE_ELLIPSIS)
                )
            )
            self.lines.append(
                f"{' ' * ERROR_LINE_LEFT_PADDING}{ERROR_LINE_ELLIPSIS}{pipeline[trim + len(ERROR_LINE_ELLIPSIS):error_end + ERROR_LINE_RIGHT_EXTENT]}{ERROR_LINE_ELLIPSIS}"
            )
            error_start -= trim
            if error_start < 0:
                error_start = len(ERROR_LINE_ELLIPSIS)
            error_position -= trim
        self.lines.append(
            f"{' ' * (ERROR_LINE_LEFT_PADDING + error_start)}{'~' * (error_position - error_start)}{'^' if error_pointer else ''}\nError (position {error_position + 1}): {error}"
        )


def parse_pipeline(pipeline: str, stream: typing.Any):
    string_filter: typing.Optional[StringFilter] = None
    name: typing.Optional[str] = None
    value: typing.Optional[str] = None
    state = 0
    index = 0
    last_exclamation_mark: typing.Optional[int] = None
    last_parameter_start: typing.Optional[int] = None
    for character in pipeline:
        if state == 0:  # expects filter name start
            if character.isspace():
                pass
            elif character.isidentifier():
                name = character
                state = 1
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=index,
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" (expected a name)',
                )
        elif state == 1:  # expects filter name
            assert name is not None
            if character.isspace():
                string_filter = StringFilter(
                    start=index - len(name), name=name, args=[], kwargs={}
                )
                state = 2
            elif character.isidentifier():
                name += character
            elif character == "!":
                last_exclamation_mark = index
                string_filter = StringFilter(
                    start=index - len(name), name=name, args=[], kwargs={}
                )
                try:
                    stream = string_filter.add_to(stream)
                except Exception as exception:
                    raise ParseError(
                        pipeline=pipeline,
                        error_start=string_filter.start,
                        error_position=index,
                        error_pointer=False,
                        error=f"{exception}",
                    )
                state = 0
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=index - len(name),
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" in filter name',
                )
        elif state == 2:  # expects parameter name start
            assert string_filter is not None
            if character.isspace():
                pass
            elif character.isalnum() or character in PARAMETER_CHARACTERS:
                last_parameter_start = index
                name = character
                state = 3
            elif character == "!":
                last_exclamation_mark = index
                try:
                    stream = string_filter.add_to(stream)
                except Exception as exception:
                    raise ParseError(
                        pipeline=pipeline,
                        error_start=string_filter.start,
                        error_position=index,
                        error_pointer=False,
                        error=f"{exception}",
                    )
                state = 0
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=index,
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" (expected a parameter name or value)',
                )
        elif state == 3:  # expects parameter name
            assert name is not None
            assert string_filter is not None
            if character.isspace():
                if len(string_filter.kwargs) > 0:
                    raise ParseError(
                        pipeline=pipeline,
                        error_start=index - len(name),
                        error_position=index,
                        error_pointer=False,
                        error=f"positional argument follows keyword argument",
                    )
                string_filter.args.append(name)
                state = 2
            elif character.isalnum() or character in PARAMETER_CHARACTERS:
                name += character
            elif character == "!":
                last_exclamation_mark = index
                string_filter.args.append(name)
                try:
                    stream = string_filter.add_to(stream)
                except Exception as exception:
                    raise ParseError(
                        pipeline=pipeline,
                        error_start=string_filter.start,
                        error_position=index,
                        error_pointer=False,
                        error=f"{exception}",
                    )
                state = 0
            elif character == "=":
                state = 4
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=index - len(name),
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" in parameter name or value',
                )
        elif state == 4:  # expects parameter value start
            assert last_parameter_start is not None
            if character.isspace():
                pass
            elif character.isalnum() or character in PARAMETER_CHARACTERS:
                value = character
                state = 5
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=last_parameter_start,
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" (expected a parameter value)',
                )
        elif state == 5:  # expects parameter value
            assert name is not None
            assert value is not None
            assert string_filter is not None
            if character.isspace():
                string_filter.kwargs[name] = value
                state = 2
            elif character.isalnum() or character in PARAMETER_CHARACTERS:
                value += character
            elif character == "!":
                last_exclamation_mark = index
                string_filter.kwargs[name] = value
                try:
                    stream = string_filter.add_to(stream)
                except Exception as exception:
                    raise ParseError(
                        pipeline=pipeline,
                        error_start=string_filter.start,
                        error_position=index,
                        error_pointer=False,
                        error=f"{exception}",
                    )
                state = 0
            else:
                raise ParseError(
                    pipeline=pipeline,
                    error_start=index - len(value),
                    error_position=index,
                    error_pointer=True,
                    error=f'unexpected character "{character}" in parameter value',
                )
        else:
            raise Exception(f"unexpected state {state}")
        index += 1
    if state == 0:  # expects filter name start
        if last_exclamation_mark is None:
            raise ParseError(
                pipeline=pipeline,
                error_start=index,
                error_position=index,
                error_pointer=True,
                error="empty pipeline",
            )
        else:
            raise ParseError(
                pipeline=pipeline,
                error_start=last_exclamation_mark,
                error_position=index,
                error_pointer=False,
                error='expected a filter after "!"',
            )
    elif state == 1:  # expects filter name
        assert name is not None
        string_filter = StringFilter(
            start=index - len(name), name=name, args=[], kwargs={}
        )
        try:
            stream = string_filter.add_to(stream)
        except Exception as exception:
            raise ParseError(
                pipeline=pipeline,
                error_start=string_filter.start,
                error_position=index,
                error_pointer=False,
                error=f"{exception}",
            )
    elif state == 2:  # expects parameter name start
        assert string_filter is not None
        try:
            stream = string_filter.add_to(stream)
        except Exception as exception:
            raise ParseError(
                pipeline=pipeline,
                error_start=string_filter.start,
                error_position=index,
                error_pointer=False,
                error=f"{exception}",
            )
    elif state == 3:  # expects parameter name
        assert string_filter is not None
        assert name is not None
        string_filter.args.append(name)
        try:
            stream = string_filter.add_to(stream)
        except Exception as exception:
            raise ParseError(
                pipeline=pipeline,
                error_start=string_filter.start,
                error_position=index,
                error_pointer=False,
                error=f"{exception}",
            )

    elif state == 4:  # expects parameter value start
        assert last_parameter_start is not None
        raise ParseError(
            pipeline=pipeline,
            error_start=last_parameter_start,
            error_position=index,
            error_pointer=True,
            error=f'unexpected character "{character}" (expected a value)',
        )
    elif state == 5:  # expects parameter value
        assert string_filter is not None
        assert name is not None
        assert value is not None
        string_filter.kwargs[name] = value
        try:
            stream = string_filter.add_to(stream)
        except Exception as exception:
            raise ParseError(
                pipeline=pipeline,
                error_start=string_filter.start,
                error_position=index,
                error_pointer=False,
                error=f"{exception}",
            )
    else:
        raise Exception(f"unexpected state {state}")
    return stream


def run(args: argparse.Namespace):
    if args.filters:
        print_filters()
        sys.exit(0)

    if (args.width is None) != (args.height is None):
        sys.stderr.write(
            "only one of width (-x/--width) and height (-y/--height) was provided (either both or neither should be provided)\n"
        )
        sys.exit(1)

    # @TODO add support for udp
    if args.input is None:
        assert args.input_format is not None
        assert args.input_format == "csv"  # only supported stdin format at the moment
        assert args.width is not None
        assert args.height is not None
        # @TODO add support for t0 and CSV properties?
        stream = faery.events_stream_from_stdin(dimensions=(args.width, args.height))
    else:
        # @TODO add support for file properties?
        if args.width is not None and args.height is not None:
            stream = faery.events_stream_from_file(
                path=args.input,
                dimensions_fallback=(args.width, args.height),
            )
        else:
            stream = faery.events_stream_from_file(path=args.input)

    try:
        stream = parse_pipeline(" ".join(args.pipeline), stream)
    except ParseError as parse_error:
        for line in parse_error.lines:
            sys.stderr.write(f"{line}\n")
        sys.exit(1)

    if args.output is None:
        assert args.output_format == "csv"  # only supported stdout format at the moment
        stream.to_stdout()
    else:
        # @TODO add support for file properties?
        stream.to_file(path=args.output)
