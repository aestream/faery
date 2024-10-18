import importlib.metadata
import typing

__version__ = importlib.metadata.version("faery")

from . import cli as cli
from . import colormaps as colormaps
from .colormaps._base import Color as Color
from .colormaps._base import Colormap as Colormap
from .colormaps._base import gradient as gradient
from .colormaps._base import parse_color as parse_color
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
from .events_stream import FiniteEventsStream as FiniteEventsStream
from .events_stream import FiniteRegularEventsStream as FiniteRegularEventsStream
from .events_stream import RegularEventsStream as RegularEventsStream
from .file_decoder import CsvProperties as CsvProperties
from .timestamp import Time as Time
from .timestamp import parse_timestamp as parse_timestamp
from .timestamp import timestamp_to_seconds as timestamp_to_seconds
from .timestamp import timestamp_to_timecode as timestamp_to_timecode

if typing.TYPE_CHECKING:
    from .types import aedat  # type: ignore
    from .types import csv  # type: ignore
    from .types import dat  # type: ignore
    from .types import event_stream  # type: ignore
    from .types import evt  # type: ignore
    from .types import image  # type: ignore
else:
    from .extension import aedat, csv, dat, event_stream, evt, image

__all__ = [
    "Color",
    "Colormap",
    "CsvProperties",
    "Decay",
    "EVENTS_DTYPE",
    "EventsFileCompression",
    "EventsFileType",
    "EventsFileVersion",
    "EventsStream",
    "FiniteEventsStream",
    "FiniteRegularEventsStream",
    "ImageFilesCompressionLevel",
    "ImageFilesType",
    "RegularEventsStream",
    "Time",
    "TransposeAction",
    "VideoFileType",
    "__version__",
    "aedat",
    "cli",
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
    "parse_color",
    "parse_timestamp",
    "timestamp_to_seconds",
    "timestamp_to_timecode",
]
