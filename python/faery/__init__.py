import importlib.metadata
import typing

__version__ = importlib.metadata.version("faery")

from . import colormaps as colormaps
from .colormaps._base import Color as Color
from .colormaps._base import Colormap as Colormap
from .colormaps._base import gradient as gradient
from .colormaps._base import parse_color as parse_color
from .display import progress_bar as progress_bar
from .enums import ColorblindnessType as ColorblindnessType
from .enums import Decay as Decay
from .enums import EventsFileCompression as EventsFileCompression
from .enums import EventsFileType as EventsFileType
from .enums import EventsFileVersion as EventsFileVersion
from .enums import ImageFileCompressionLevel as ImageFilesCompressionLevel
from .enums import ImageFileType as ImageFilesType
from .enums import TransposeAction as TransposeAction
from .enums import VideoFileType as VideoFileType
from .events_input import events_stream_from_array as events_stream_from_array
from .events_input import events_stream_from_file as events_stream_from_file
from .events_input import events_stream_from_stdin as events_stream_from_stdin
from .events_input import events_stream_from_udp as events_stream_from_udp
from .events_stream import EVENTS_DTYPE as EVENTS_DTYPE
from .events_stream import EventsStream as EventsStream
from .events_stream import EventsStreamState as EventsStreamState
from .events_stream import FiniteEventsStream as FiniteEventsStream
from .events_stream import FiniteEventsStreamState as FiniteEventsStreamState
from .events_stream import FiniteRegularEventsStream as FiniteRegularEventsStream
from .events_stream import (
    FiniteRegularEventsStreamState as FiniteRegularEventsStreamState,
)
from .events_stream import RegularEventsStream as RegularEventsStream
from .events_stream import RegularEventsStreamState as RegularEventsStreamState
from .file_decoder import CsvProperties as CsvProperties
from .timestamp import Time as Time
from .timestamp import parse_timestamp as parse_timestamp
from .timestamp import timestamp_to_seconds as timestamp_to_seconds
from .timestamp import timestamp_to_timecode as timestamp_to_timecode

if typing.TYPE_CHECKING:
    from .types import aedat, csv, dat, event_stream, evt, image, mp4  # type: ignore
else:
    from .extension import aedat, csv, dat, image, event_stream, evt, mp4

__all__ = [
    "Color",
    "ColorblindnessType",
    "Colormap",
    "CsvProperties",
    "Decay",
    "EVENTS_DTYPE",
    "EventsFileCompression",
    "EventsFileType",
    "EventsFileVersion",
    "EventsStream",
    "EventsStreamState",
    "FiniteEventsStream",
    "FiniteEventsStreamState",
    "FiniteRegularEventsStream",
    "FiniteRegularEventsStreamState",
    "ImageFilesCompressionLevel",
    "ImageFilesType",
    "RegularEventsStream",
    "RegularEventsStreamState",
    "Time",
    "TransposeAction",
    "VideoFileType",
    "__version__",
    "aedat",
    "colormaps",
    "csv",
    "dat",
    "event_stream",
    "events_stream_from_array",
    "events_stream_from_file",
    "events_stream_from_stdin",
    "events_stream_from_udp",
    "evt",
    "gradient",
    "image",
    "mp4",
    "parse_color",
    "parse_timestamp",
    "progress_bar",
    "timestamp_to_seconds",
    "timestamp_to_timecode",
]
