import typing

from .file_decoder import CsvProperties as CsvProperties
from .events_filter import EVENTS_DTYPE as EVENTS_DTYPE
from .events_input import events_stream_from_array as events_stream_from_array
from .events_input import events_stream_from_file as events_stream_from_file
from .events_stream import EventsStream as EventsStream
from .events_stream import FiniteEventsStream as FiniteEventsStream
from .events_stream import RegularEventsStream as RegularEventsStream
from .events_stream import FiniteRegularEventsStream as FiniteRegularEventsStream
from .file_type import FileType as FileType
from .timestamp import Time as Time
from .timestamp import parse_timestamp as parse_timestamp
from .timestamp import timestamp_to_timecode as timestamp_to_timecode
from .timestamp import timestamp_to_seconds as timestamp_to_seconds

if typing.TYPE_CHECKING:
    from .types import aedat  # type: ignore
    from .types import csv  # type: ignore
    from .types import dat  # type: ignore
    from .types import event_stream  # type: ignore
    from .types import evt  # type: ignore
else:
    from .faery import aedat
    from .faery import csv
    from .faery import dat
    from .faery import event_stream
    from .faery import evt

__all__ = [
    "CsvProperties",
    "EVENTS_DTYPE",
    "events_stream_from_array",
    "events_stream_from_file",
    "EventsStream",
    "FiniteEventsStream",
    "RegularEventsStream",
    "FiniteRegularEventsStream",
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
