import pathlib
import typing

Decay = typing.Literal["exponential", "linear", "window"]

TransposeAction = typing.Literal[
    "flip_left_right",
    "flip_bottom_top",
    "rotate_90_counterclockwise",
    "rotate_180",
    "rotate_270_counterclockwise",
    "flip_up_diagonal",
    "flip_down_diagonal",
]
"""
Spatial transformation that applies to events and frames

- flip_left_right mirrors horizontally
- flip_bottom_top mirrors vertically
- rotate_90_counterclockwise rotates to the left by 90º
- rotate_180 rotates by 180º
- rotate_270_counterclockwise rotates to the right by 90º
- flip_up_diagonal mirrors alongside the diagonal that goes from the bottom left to the top right (also known as transverse)
- flip_down_diagonal mirrors alongside the diagonal that goes from the top left to the bottom right (also known as transpose)
"""


EventsFileType = typing.Literal[
    "aedat",
    "csv",
    "dat",
    "es",
    "evt",
]

EventsFileVersion = typing.Literal["dat1", "dat2", "evt2", "evt2.1", "evt3"]

EventsFileCompression = typing.Literal["lz4", "zstd"]

ImageFileType = typing.Literal["png"]

ImageFileCompressionLevel = typing.Literal["default", "fast", "best"]

VideoFileType = typing.Literal["mp4"]

ColorblindnessType = typing.Literal["protanopia", "deuteranopia", "tritanopia"]

VALIDATORS: dict[str, typing.Any] = {}


def validate_decay(value: Decay) -> Decay:
    return VALIDATORS["validate_decay"](value)


def validate_transpose_action(value: TransposeAction) -> TransposeAction:
    return VALIDATORS["validate_transpose_action"](value)


def validate_events_file_type(value: EventsFileType) -> EventsFileType:
    return VALIDATORS["validate_events_file_type"](value)


def validate_events_file_version(value: EventsFileVersion) -> EventsFileVersion:
    return VALIDATORS["validate_events_file_version"](value)


def validate_events_file_compression(
    value: EventsFileCompression,
) -> EventsFileCompression:
    return VALIDATORS["validate_events_file_compression"](value)


def validate_image_file_type(
    value: ImageFileType,
) -> ImageFileType:
    return VALIDATORS["validate_image_file_type"](value)


def validate_image_file_compression_level(
    value: ImageFileCompressionLevel,
) -> ImageFileCompressionLevel:
    return VALIDATORS["validate_image_file_compression_level"](value)


def validate_video_file_type(
    value: VideoFileType,
) -> VideoFileType:
    return VALIDATORS["validate_video_file_type"](value)


def validate_colorblindness_type(
    value: ColorblindnessType,
) -> ColorblindnessType:
    return VALIDATORS["validate_colorblindness_type"](value)


def bind(name: str, type: typing.Any):
    snake_case_name = ""
    for character in name:
        if (
            character.isupper()
            and len(snake_case_name) > 0
            and snake_case_name[-1] != "_"
        ):
            snake_case_name += f"_{character.lower()}"
        else:
            snake_case_name += character.lower()
    valid_values: tuple[str, ...] = typing.get_args(type)
    if len(valid_values) == 0:
        raise Exception(f"{name} has not args")

    def validate(value: str) -> str:
        if value in valid_values:
            return value
        if len(valid_values) == 1:
            valid_values_as_string = f'"{valid_values[0]}"'
        elif len(valid_values) == 2:
            valid_values_as_string = f'"{valid_values[0]}" or "{valid_values[1]}"'
        else:
            all_but_one = (f'"{value}"' for value in valid_values)
            valid_values_as_string = (
                f"{', '.join(all_but_one)}, or \"{valid_values[-1]}\""
            )
        raise Exception(f'unknown {name} "{value}" (expected {valid_values_as_string})')

    validate.__name__ = f"validate_{snake_case_name}"
    VALIDATORS[validate.__name__] = validate


for name, variable in list(locals().items()):
    if typing.get_origin(variable) == typing.Literal:
        bind(name, variable)


def events_file_type_magic(events_file_type: EventsFileType) -> typing.Optional[bytes]:
    if events_file_type == "aedat":
        return b"#!AER-DAT4.0\r\n"
    if events_file_type == "csv":
        return None
    if events_file_type == "dat":
        return None
    if events_file_type == "es":
        return b"Event Stream"
    if events_file_type == "evt":
        return None
    raise Exception(f"magic is not implemented for {events_file_type}")


def events_file_type_extensions(events_file_type: EventsFileType) -> list[str]:
    if events_file_type == "aedat":
        return [".aedat", ".aedat4"]
    if events_file_type == "csv":
        return [".csv"]
    if events_file_type == "dat":
        return [".dat"]
    if events_file_type == "es":
        return [".es"]
    if events_file_type == "evt":
        return [".evt", ".raw"]
    raise Exception(f"extensions is not implemented for {events_file_type}")


def events_file_type_guess(path: pathlib.Path) -> EventsFileType:
    file_types = typing.get_args(EventsFileType)
    longest_magic = max(
        0 if magic is None else len(magic)
        for magic in (events_file_type_magic(file_type) for file_type in file_types)
    )
    try:
        with open(path, "rb") as file:
            magic = file.read(longest_magic)
        for file_type in file_types:
            if events_file_type_magic(file_type) == magic:
                return file_type
    except FileNotFoundError:
        pass
    extension = path.suffix
    for file_type in file_types:
        if any(
            extension == type_extension
            for type_extension in events_file_type_extensions(file_type)
        ):
            return file_type
    raise Exception(f"unsupported file {path}")


def image_file_type_magic(image_file_type: ImageFileType) -> typing.Optional[bytes]:
    if image_file_type == "png":
        return bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    raise Exception(f"magic is not implemented for {events_file_type}")


def image_file_type_extensions(image_file_type: ImageFileType) -> list[str]:
    if image_file_type == "png":
        return [".png"]
    raise Exception(f"extensions is not implemented for {image_file_type}")


def image_file_type_guess(path: pathlib.Path) -> ImageFileType:
    file_types = typing.get_args(ImageFileType)
    longest_magic = max(
        0 if magic is None else len(magic)
        for magic in (image_file_type_magic(file_type) for file_type in file_types)
    )
    try:
        with open(path, "rb") as file:
            magic = file.read(longest_magic)
        for file_type in file_types:
            if image_file_type_magic(file_type) == magic:
                return file_type
    except FileNotFoundError:
        pass
    extension = path.suffix
    for file_type in file_types:
        if any(
            extension == type_extension
            for type_extension in image_file_type_extensions(file_type)
        ):
            return file_type
    raise Exception(f"unsupported file {path}")
