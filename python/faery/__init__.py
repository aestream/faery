import importlib.metadata
import typing

__version__ = importlib.metadata.version("faery")

from .file_decoder import CsvProperties as CsvProperties
from .events_filter import EVENTS_DTYPE as EVENTS_DTYPE
from .events_input import events_stream_from_array as events_stream_from_array
from .events_input import events_stream_from_file as events_stream_from_file
from .events_input import events_stream_from_stdin as events_stream_from_stdin
from .events_input import events_stream_from_udp as events_stream_from_udp
from .events_stream import EventsStream as EventsStream
from .events_stream import FiniteEventsStream as FiniteEventsStream
from .events_stream import RegularEventsStream as RegularEventsStream
from .events_stream import FiniteRegularEventsStream as FiniteRegularEventsStream
from .file_type import FileType as FileType
from .stream import TransposeAction as TransposeAction
from .timestamp import Time as Time
from .timestamp import parse_timestamp as parse_timestamp
from .timestamp import timestamp_to_timecode as timestamp_to_timecode
from .timestamp import timestamp_to_seconds as timestamp_to_seconds
from . import cli as cli

if typing.TYPE_CHECKING:
    from .types import aedat  # type: ignore
    from .types import csv  # type: ignore
    from .types import dat  # type: ignore
    from .types import event_stream  # type: ignore
    from .types import evt  # type: ignore
else:
    from .extension import aedat
    from .extension import csv
    from .extension import dat
    from .extension import event_stream
    from .extension import evt

__all__ = [
    "__version__",
    "CsvProperties",
    "EVENTS_DTYPE",
    "events_stream_from_array",
    "events_stream_from_file",
    "events_stream_from_stdin",
    "events_stream_from_udp",
    "EventsStream",
    "FiniteEventsStream",
    "RegularEventsStream",
    "FiniteRegularEventsStream",
    "FileType",
    "TransposeAction",
    "Time",
    "parse_timestamp",
    "timestamp_to_timecode",
    "timestamp_to_seconds",
    "aedat",
    "csv",
    "dat",
    "event_stream",
    "evt",
    "cli",
]
