# isort: skip_file

import importlib.metadata
import inspect
import typing

__version__ = importlib.metadata.version("faery")

from . import colormaps as colormaps
from .colormaps._base import (
    Color as Color,
    Colormap as Colormap,
    gradient as gradient,
    parse_color as parse_color,
)
from .display import (
    progress_bar as progress_bar,
    progress_bar_fold as progress_bar_fold,
    format_bold as format_bold,
    format_color as format_color,
)
from .enums import (
    ColorblindnessType as ColorblindnessType,
    Decay as Decay,
    EventsFileCompression as EventsFileCompression,
    EventsFileType as EventsFileType,
    EventsFileVersion as EventsFileVersion,
    FilterOrientation as FilterOrientation,
    UdpFormat as UdpFormat,
    ImageFileCompressionLevel as ImageFileCompressionLevel,
    ImageFileType as ImageFileType,
    ImageResizeSamplingFilter as ImageResizeSamplingFilter,
    TransposeAction as TransposeAction,
    VideoFilePreset as VideoFilePreset,
    VideoFileProfile as VideoFileProfile,
    VideoFileTune as VideoFileTune,
    VideoFileType as VideoFileType,
)
from .events_input import (
    events_stream_from_array as events_stream_from_array,
    events_stream_from_file as events_stream_from_file,
    events_stream_from_stdin as events_stream_from_stdin,
    events_stream_from_udp as events_stream_from_udp,
)
from .events_stream import (
    EVENTS_DTYPE as EVENTS_DTYPE,
    EventsFilter as EventsFilter,
    EventsStream as EventsStream,
    FiniteEventsFilter as FiniteEventsFilter,
    FiniteEventsStream as FiniteEventsStream,
    FiniteRegularEventsFilter as FiniteRegularEventsFilter,
    FiniteRegularEventsStream as FiniteRegularEventsStream,
    RegularEventsFilter as RegularEventsFilter,
    RegularEventsStream as RegularEventsStream,
)
from .events_stream_state import (
    EventsStreamState as EventsStreamState,
    FiniteEventsStreamState as FiniteEventsStreamState,
    FiniteRegularEventsStreamState as FiniteRegularEventsStreamState,
    RegularEventsStreamState as RegularEventsStreamState,
)
from .frame_stream import (
    Float64Frame as Float64Frame,
    Float64FrameStream as Float64FrameStream,
    FiniteFloat64FrameStream as FiniteFloat64FrameStream,
    FiniteFloat64FrameFilter as FiniteFloat64FrameFilter,
    FiniteRegularFloat64FrameFilter as FiniteRegularFloat64FrameFilter,
    FiniteRegularFloat64FrameStream as FiniteRegularFloat64FrameStream,
    FiniteRegularRgba8888FrameFilter as FiniteRegularRgba8888FrameFilter,
    FiniteRegularRgba8888FrameStream as FiniteRegularRgba8888FrameStream,
    FiniteRgba8888FrameStream as FiniteRgba8888FrameStream,
    FiniteRgba8888FrameFilter as FiniteRgba8888FrameFilter,
    RegularFloat64FrameFilter as RegularFloat64FrameFilter,
    RegularFloat64FrameStream as RegularFloat64FrameStream,
    RegularRgba8888FrameFilter as RegularRgba8888FrameFilter,
    RegularRgba8888FrameStream as RegularRgba8888FrameStream,
    Rgba8888Frame as Rgba8888Frame,
    Rgba8888FrameStream as Rgba8888FrameStream,
)
from .frame_stream_state import (
    FrameStreamState as FrameStreamState,
    FiniteFrameStreamState as FiniteFrameStreamState,
    FiniteRegularFrameStreamState as FiniteRegularFrameStreamState,
    RegularFrameStreamState as RegularFrameStreamState,
)
from .file_decoder import CsvProperties as CsvProperties
from .task import (
    dirname as dirname,
    Task as Task,
    JobManager as JobManager,
    task as task,
)
from .timestamp import (
    Time as Time,
    parse_timestamp as parse_timestamp,
    timestamp_to_seconds as timestamp_to_seconds,
    timestamp_to_timecode as timestamp_to_timecode,
)

if typing.TYPE_CHECKING:
    from .types import (
        aedat,  # type: ignore
        csv,  # type: ignore
        dat,  # type: ignore
        event_stream,  # type: ignore
        evt,  # type: ignore
        image,  # type: ignore
        job_metadata,  # type: ignore
        mp4,  # type: ignore
        mustache,  # type: ignore
    )
else:
    from .extension import (
        aedat,
        csv,
        dat,
        event_stream,
        evt,
        image,
        job_metadata,
        mp4,
        mustache,
    )


def colormaps_list() -> list[Colormap]:
    return [
        colormap
        for _, colormap in inspect.getmembers(
            colormaps, lambda member: isinstance(member, Colormap)
        )
    ]


def name_to_colormaps() -> dict[str, Colormap]:
    return {
        colormap.name: colormap
        for _, colormap in inspect.getmembers(
            colormaps, lambda member: isinstance(member, Colormap)
        )
    }


__all__ = [
    "__version__",
    "colormaps",
    "Color",
    "Colormap",
    "gradient",
    "parse_color",
    "progress_bar",
    "progress_bar_fold",
    "format_bold",
    "format_color",
    "ColorblindnessType",
    "Decay",
    "EventsFileCompression",
    "EventsFileType",
    "EventsFileVersion",
    "FilterOrientation",
    "UdpFormat",
    "ImageFileCompressionLevel",
    "ImageFileType",
    "ImageResizeSamplingFilter",
    "TransposeAction",
    "VideoFilePreset",
    "VideoFileProfile",
    "VideoFileTune",
    "VideoFileType",
    "events_stream_from_array",
    "events_stream_from_file",
    "events_stream_from_stdin",
    "events_stream_from_udp",
    "EVENTS_DTYPE",
    "EventsFilter",
    "EventsStream",
    "FiniteEventsFilter",
    "FiniteEventsStream",
    "FiniteRegularEventsFilter",
    "FiniteRegularEventsStream",
    "RegularEventsFilter",
    "RegularEventsStream",
    "EventsStreamState",
    "FiniteEventsStreamState",
    "FiniteRegularEventsStreamState",
    "RegularEventsStreamState",
    "Float64Frame",
    "Float64FrameStream",
    "FiniteFloat64FrameStream",
    "FiniteFloat64FrameFilter",
    "FiniteRegularFloat64FrameFilter",
    "FiniteRegularFloat64FrameStream",
    "FiniteRegularRgba8888FrameFilter",
    "FiniteRegularRgba8888FrameStream",
    "FiniteRgba8888FrameStream",
    "FiniteRgba8888FrameFilter",
    "RegularFloat64FrameFilter",
    "RegularFloat64FrameStream",
    "RegularRgba8888FrameFilter",
    "RegularRgba8888FrameStream",
    "Rgba8888Frame",
    "Rgba8888FrameStream",
    "FrameStreamState",
    "FiniteFrameStreamState",
    "FiniteRegularFrameStreamState",
    "RegularFrameStreamState",
    "CsvProperties",
    "dirname",
    "Task",
    "UdpFormat",
    "JobManager",
    "task",
    "Time",
    "parse_timestamp",
    "timestamp_to_seconds",
    "timestamp_to_timecode",
    "aedat",
    "csv",
    "dat",
    "event_stream",
    "evt",
    "image",
    "job_metadata",
    "mp4",
    "colormaps_list",
    "name_to_colormaps",
]
