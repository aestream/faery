import typing

from .file_decoder import CsvProperties as CsvProperties
from .events_filter import EVENTS_DTYPE as EVENTS_DTYPE
from .events_input import events_stream_from_array as events_stream_from_array
from .events_input import events_stream_from_file as events_stream_from_file
from .events_stream import EventsStream as EventsStream
from .events_stream import FiniteEventsStream as FiniteEventsStream
from .events_stream import UniformEventsStream as UniformEventsStream
from .events_stream import FiniteUniformEventsStream as FiniteUniformEventsStream
from .file_type import FileType as FileType
from .timestamp import Time as Time
from .timestamp import parse_timestamp as parse_timestamp
from .timestamp import timestamp_to_timecode as timestamp_to_timecode
from .timestamp import timestamp_to_seconds as timestamp_to_seconds

if typing.TYPE_CHECKING:
    from .extension_types import aedat  # type: ignore
    from .extension_types import csv  # type: ignore
    from .extension_types import dat  # type: ignore
    from .extension_types import event_stream  # type: ignore
    from .extension_types import evt  # type: ignore
else:
    from .extension import aedat
    from .extension import csv
    from .extension import dat
    from .extension import event_stream
    from .extension import evt

__all__ = [
    "CsvProperties",
    "EVENTS_DTYPE",
    "events_stream_from_array",
    "events_stream_from_file",
    "EventsStream",
    "FiniteEventsStream",
    "UniformEventsStream",
    "FiniteUniformEventsStream",
    "FileType",
    "Time",
    "parse_timestamp",
    "timestamp_to_timecode",
    "timestamp_to_seconds",
    "aedat",
    "csv",
    "dat",
    "event_stream",
    "evt",
]
