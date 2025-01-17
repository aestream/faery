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

FilterOrientation = typing.Literal["row", "column"]

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

ImageResizeSamplingFilter = typing.Literal[
    "nearest", "triangle", "catmull_rom", "gaussian", "lanczos3"
]

GraphFileType = typing.Literal["png", "svg"]

VideoFileType = typing.Literal["gif", "mp4"]

VideoFilePreset = typing.Literal[
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
    "none",
]

VideoFileTune = typing.Literal[
    "film",
    "animation",
    "grain",
    "stillimage",
    "psnr",
    "ssim",
    "fastdecode",
    "zerolatency",
    "none",
]

VideoFileProfile = typing.Literal[
    "baseline",
    "main",
    "high",
    "high10",
    "high422",
    "high444",
]


ColormapType = typing.Literal["sequential", "diverging", "cyclic"]

ColorblindnessType = typing.Literal["protanopia", "deuteranopia", "tritanopia"]

UdpFormat = typing.Literal["t64_x16_y16_on8", "t32_x16_y15_on1"]

VALIDATORS: dict[str, typing.Any] = {}


def validate_decay(value: Decay) -> Decay:
    return VALIDATORS["validate_decay"](value)


def validate_transpose_action(value: TransposeAction) -> TransposeAction:
    return VALIDATORS["validate_transpose_action"](value)


def validate_filter_orientation(value: FilterOrientation) -> FilterOrientation:
    return VALIDATORS["validate_filter_orientation"](value)


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


def validate_image_resize_samplin_filter(
    value: ImageResizeSamplingFilter,
) -> ImageResizeSamplingFilter:
    return VALIDATORS["validate_image_resize_sampling_filter"](value)


def validate_graph_file_type(
    value: GraphFileType,
) -> GraphFileType:
    return VALIDATORS["validate_graph_file_type"](value)


def validate_video_file_type(
    value: VideoFileType,
) -> VideoFileType:
    return VALIDATORS["validate_video_file_type"](value)


def validate_video_file_preset(
    value: VideoFilePreset,
) -> VideoFilePreset:
    return VALIDATORS["validate_video_file_preset"](value)


def validate_video_file_tune(
    value: VideoFileTune,
) -> VideoFileTune:
    return VALIDATORS["validate_video_file_tune"](value)


def validate_video_file_profile(
    value: VideoFileProfile,
) -> VideoFileProfile:
    return VALIDATORS["validate_video_file_profile"](value)


def validate_colormap_type(
    value: ColormapType,
) -> ColormapType:
    return VALIDATORS["validate_colormap_type"](value)


def validate_colorblindness_type(
    value: ColorblindnessType,
) -> ColorblindnessType:
    return VALIDATORS["validate_colorblindness_type"](value)


def validate_udp_format(
    value: UdpFormat,
) -> UdpFormat:
    return VALIDATORS["validate_udp_format"](value)


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


def events_file_type_extensions(events_file_type: EventsFileType) -> tuple[str, ...]:
    if events_file_type == "aedat":
        return (".aedat", ".aedat4")
    if events_file_type == "csv":
        return (".csv",)
    if events_file_type == "dat":
        return (".dat",)
    if events_file_type == "es":
        return (".es",)
    if events_file_type == "evt":
        return (".evt", ".raw")
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
    raise Exception(f"unsupported file extension {path}")


def image_file_type_extensions(image_file_type: ImageFileType) -> tuple[str, ...]:
    if image_file_type == "png":
        return (".png",)
    raise Exception(f"extensions is not implemented for {image_file_type}")


def image_file_type_guess(path: pathlib.Path) -> ImageFileType:
    file_types = typing.get_args(ImageFileType)
    extension = path.suffix
    for file_type in file_types:
        if any(
            extension == type_extension
            for type_extension in image_file_type_extensions(file_type)
        ):
            return file_type
    raise Exception(f"unsupported file extension {path}")


def graph_file_type_extensions(graph_file_type: GraphFileType) -> tuple[str, ...]:
    if graph_file_type == "png":
        return (".png",)
    if graph_file_type == "svg":
        return (".svg",)
    raise Exception(f"extensions is not implemented for {graph_file_type}")


def graph_file_type_guess(path: pathlib.Path) -> GraphFileType:
    file_types = typing.get_args(GraphFileType)
    extension = path.suffix
    for file_type in file_types:
        if any(
            extension == type_extension
            for type_extension in graph_file_type_extensions(file_type)
        ):
            return file_type
    raise Exception(f"unsupported file extension {path}")


def video_file_type_extensions(video_file_type: VideoFileType) -> tuple[str, ...]:
    if video_file_type == "gif":
        return (".gif",)
    if video_file_type == "mp4":
        return (".mp4",)
    raise Exception(f"extensions is not implemented for {video_file_type}")


def video_file_type_guess(path: pathlib.Path) -> VideoFileType:
    file_types = typing.get_args(VideoFileType)
    extension = path.suffix
    for file_type in file_types:
        if any(
            extension == type_extension
            for type_extension in video_file_type_extensions(file_type)
        ):
            return file_type
    raise Exception(f"unsupported file extension {path}")
