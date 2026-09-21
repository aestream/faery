# isort: skip_file

import importlib.metadata
import inspect
import typing

__version__ = importlib.metadata.version("faery")

from . import colormaps as colormaps
from .color import (
    Color as Color,
    Colormap as Colormap,
    ColorTheme as ColorTheme,
    gradient as gradient,
    color_to_ints as color_to_ints,
    color_to_floats as color_to_floats,
    color_to_hex_string as color_to_hex_string,
    LIGHT_COLOR_THEME as LIGHT_COLOR_THEME,
    DARK_COLOR_THEME as DARK_COLOR_THEME,
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
from .event_rate import EventRate as EventRate
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
from .frame_input import (
    frame_stream_from_list as frame_stream_from_list,
    frame_stream_from_function as frame_stream_from_function,
)
from .frame_stream import (
    FiniteRegularFrameFilter as FiniteRegularFrameFilter,
    FiniteRegularFrameStream as FiniteRegularFrameStream,
    FiniteFrameStream as FiniteFrameStream,
    FiniteFrameFilter as FiniteFrameFilter,
    RegularFrameFilter as RegularFrameFilter,
    RegularFrameStream as RegularFrameStream,
    Frame as Frame,
    FrameStream as FrameStream,
)
from .frame_stream_state import (
    FrameStreamState as FrameStreamState,
    FiniteFrameStreamState as FiniteFrameStreamState,
    FiniteRegularFrameStreamState as FiniteRegularFrameStreamState,
    RegularFrameStreamState as RegularFrameStreamState,
)
from .file_decoder import CsvProperties as CsvProperties
from .kinectograph import Kinectograph as Kinectograph
from .spectrogram import Spectrogram as Spectrogram
from .task import (
    dirname as dirname,
    Task as Task,
    JobManager as JobManager,
    task as task,
)
from .timestamp import (
    Time as Time,
    TimeOrTimecode as TimeOrTimecode,
    parse_time as parse_time,
    us as us,
    ms as ms,
    s as s,
)
from .wiggle import WiggleParameters as WiggleParameters

if typing.TYPE_CHECKING:
    from .types import (
        aedat,
        csv,
        dat,
        es,
        evt,
        gif,
        image,
        job_metadata,
        mp4,
        mustache,
        raster,
    )
else:
    from .extension import (
        aedat,
        csv,
        dat,
        es,
        evt,
        gif,
        image,
        job_metadata,
        mp4,
        mustache,
        raster,
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


from .event_camera_input import events_stream_from_camera

__all__ = [
    "DARK_COLOR_THEME",
    "EVENTS_DTYPE",
    "LIGHT_COLOR_THEME",
    "Color",
    "ColorTheme",
    "ColorblindnessType",
    "Colormap",
    "CsvProperties",
    "Decay",
    "EventRate",
    "EventsFileCompression",
    "EventsFileType",
    "EventsFileVersion",
    "EventsFilter",
    "EventsStream",
    "EventsStreamState",
    "FilterOrientation",
    "FiniteEventsFilter",
    "FiniteEventsStream",
    "FiniteEventsStreamState",
    "FiniteFrameFilter",
    "FiniteFrameStream",
    "FiniteFrameStreamState",
    "FiniteRegularEventsFilter",
    "FiniteRegularEventsStream",
    "FiniteRegularEventsStreamState",
    "FiniteRegularFrameFilter",
    "FiniteRegularFrameStream",
    "FiniteRegularFrameStreamState",
    "Frame",
    "FrameStream",
    "FrameStreamState",
    "ImageFileCompressionLevel",
    "ImageFileType",
    "ImageResizeSamplingFilter",
    "JobManager",
    "Kinectograph",
    "RegularEventsFilter",
    "RegularEventsStream",
    "RegularEventsStreamState",
    "RegularFrameFilter",
    "RegularFrameStream",
    "RegularFrameStreamState",
    "Spectrogram",
    "Task",
    "Time",
    "TimeOrTimecode",
    "TransposeAction",
    "UdpFormat",
    "VideoFilePreset",
    "VideoFileProfile",
    "VideoFileTune",
    "VideoFileType",
    "WiggleParameters",
    "__version__",
    "aedat",
    "color_to_floats",
    "color_to_hex_string",
    "color_to_ints",
    "colormaps",
    "colormaps_list",
    "csv",
    "dat",
    "dirname",
    "es",
    "events_stream_from_array",
    "events_stream_from_camera",
    "events_stream_from_file",
    "events_stream_from_stdin",
    "events_stream_from_udp",
    "evt",
    "format_bold",
    "format_color",
    "frame_stream_from_function",
    "frame_stream_from_list",
    "gif",
    "gradient",
    "image",
    "job_metadata",
    "mp4",
    "ms",
    "mustache",
    "name_to_colormaps",
    "parse_time",
    "progress_bar",
    "progress_bar_fold",
    "raster",
    "s",
    "task",
    "us",
]
