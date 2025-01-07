import collections.abc
import dataclasses
import pathlib
import typing

import numpy
import numpy.lib.recfunctions

from . import enums, events_stream, timestamp

if typing.TYPE_CHECKING:
    from .types import aedat, csv, dat, event_stream, evt  # type: ignore
else:
    from .extension import aedat, csv, dat, event_stream, evt


@dataclasses.dataclass
class CsvProperties:
    has_header: bool
    separator: bytes
    t_index: int
    x_index: int
    y_index: int
    on_index: int
    t_scale: float
    on_value: bytes
    off_value: bytes
    skip_errors: bool

    @classmethod
    def default(cls):
        return cls(
            has_header=True,
            separator=b",",
            t_index=0,
            x_index=1,
            y_index=2,
            on_index=3,
            t_scale=0.0,
            on_value=b"1",
            off_value=b"0",
            skip_errors=False,
        )


class TimeRangeCache:
    def __init__(self):
        self.path_to_hash_and_time_range: dict[
            pathlib.Path, tuple[int, tuple[timestamp.Time, timestamp.Time]]
        ] = {}

    def path_hash(self, path: pathlib.Path) -> int:
        stat = path.stat()
        return hash((stat.st_mtime_ns, stat.st_size))

    def get_time_range(
        self, path: pathlib.Path, path_hash: int
    ) -> typing.Optional[tuple[timestamp.Time, timestamp.Time]]:
        if not path in self.path_to_hash_and_time_range:
            return None
        stored_path_hash, time_range = self.path_to_hash_and_time_range[path]
        if path_hash == stored_path_hash:
            return time_range
        del self.path_to_hash_and_time_range[path]
        return None

    def set_time_range(
        self,
        path: pathlib.Path,
        path_hash: int,
        time_range: tuple[timestamp.Time, timestamp.Time],
    ):
        self.path_to_hash_and_time_range[path] = (path_hash, time_range)


TIME_RANGE_CACHE: TimeRangeCache = TimeRangeCache()


class Decoder(events_stream.FiniteEventsStream):
    """
    An event file decoder (supports .aedat4, .es, .raw, and .dat).

    track_id is only used if the type is aedat. It selects a specific stream in the container.
    If left unspecified (None), the first event stream is selected.

    dimensions_fallback is only used if the file type is EVT (.raw), DAT, or CSV and if the file's header
    does not specify the size.

    version_fallback is only used if the file type is EVT (.raw) or DAT and if the file's header
    does not specify the version.

    t0 is only used if the file type is ES or CSV.

    Args:
        path: Path of the input event file or None for stdin (stdin is only compatible with the CSV format).
        track_id: Stream ID, only used with aedat files. Defaults to None.
        dimensions_fallback: Size fallback for EVT (.raw) and DAT files. Defaults to (1280, 720).
        version_fallback: Version fallback for EVT (.raw) and DAT files. Defaults to "dat2" for DAT and "evt3" for EVT.
        t0: Initial time for ES files, in seconds. Defaults to None.
        file_type: Override the type determination algorithm. Defaults to None.
    """

    def __init__(
        self,
        path: typing.Union[pathlib.Path, str, None],
        track_id: typing.Optional[int] = None,
        dimensions_fallback: tuple[int, int] = (1280, 720),
        version_fallback: typing.Optional[enums.EventsFileVersion] = None,
        t0: timestamp.TimeOrTimecode = timestamp.Time(microseconds=0),
        csv_properties: CsvProperties = CsvProperties.default(),
        file_type: typing.Optional[enums.EventsFileType] = None,
        time_range_cache: typing.Optional[TimeRangeCache] = TIME_RANGE_CACHE,
    ):
        super().__init__()
        self.path = None if path is None else pathlib.Path(path)
        self.track_id = track_id
        self.dimensions_fallback = dimensions_fallback
        if version_fallback is None:
            self.version_fallback = None
        else:
            self.version_fallback = enums.validate_events_file_version(version_fallback)
        self.t0 = timestamp.parse_time(t0)
        self.csv_properties = csv_properties
        if self.path is None:
            if file_type != "csv":
                raise Exception(
                    '`file_type` must be "csv" when reading from stdin (path is None)'
                )
            self.file_type = "csv"
        else:
            self.file_type = (
                enums.events_file_type_guess(self.path)
                if file_type is None
                else enums.validate_events_file_type(file_type)
            )
        self.time_range_cache = time_range_cache
        self.inner_dimensions: tuple[int, int]
        self.event_type: typing.Optional[str] = None
        if self.file_type == "aedat":
            assert self.path is not None
            with aedat.Decoder(self.path) as decoder:
                found = False
                for track in decoder.tracks():
                    if self.track_id is None:
                        if track.data_type == "events":
                            self.track_id = track.id
                            assert track.dimensions is not None
                            self.inner_dimensions = track.dimensions
                            found = True
                            break
                    else:
                        if track.id == self.track_id:
                            if track.data_type != "events":
                                raise Exception(
                                    f'track {self.track_id} does not contain events (its type is "{track.data_type}")'
                                )
                            assert track.dimensions is not None
                            self.inner_dimensions = track.dimensions
                            found = True
                            break
                if not found:
                    if self.track_id is None:
                        raise Exception(f"the file contains no event tracks")
                    else:
                        raise Exception(
                            f"track {self.track_id} not found (the available ids are {[track.id for track in decoder.tracks()]})"
                        )
        elif self.file_type == "csv":
            self.inner_dimensions = dimensions_fallback
            assert len(self.csv_properties.separator) == 1
        elif self.file_type == "dat":
            assert self.path is not None
            if self.version_fallback is None:
                self.version_fallback = "dat2"
            with dat.Decoder(
                self.path,
                self.dimensions_fallback,
                self.version_fallback,  # type: ignore
            ) as decoder:
                self.event_type = decoder.event_type
                if self.event_type != "cd":
                    raise Exception(
                        f'the stream "{self.path}" has the unsupported type "{self.event_type}"'
                    )
                assert decoder.dimensions is not None
                self.inner_dimensions = decoder.dimensions
        elif self.file_type == "es":
            assert self.path is not None
            with event_stream.Decoder(
                path=self.path,
                t0=self.t0.to_microseconds(),
            ) as decoder:
                self.event_type = decoder.event_type
                if self.event_type != "dvs" and self.event_type != "atis":
                    raise Exception(
                        f'the stream "{self.path}" has the unsupported type "{self.event_type}"'
                    )
                assert decoder.dimensions is not None
                self.inner_dimensions = decoder.dimensions
        elif self.file_type == "evt":
            if self.version_fallback is None:
                self.version_fallback = "evt3"
            assert self.path is not None
            with evt.Decoder(
                self.path,
                self.dimensions_fallback,
                self.version_fallback,  # type: ignore
            ) as decoder:
                self.inner_dimensions = decoder.dimensions
        else:
            raise Exception(f"file type {self.file_type} not implemented")

    def dimensions(self) -> tuple[int, int]:
        return self.inner_dimensions

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        if self.path is None:
            raise NotImplementedError()
        if self.time_range_cache is not None:
            path_hash = self.time_range_cache.path_hash(path=self.path)
            time_range = self.time_range_cache.get_time_range(
                path=self.path, path_hash=path_hash
            )
            if time_range is not None:
                return time_range
        else:
            path_hash = None
        begin: typing.Optional[int] = None
        end: typing.Optional[int] = None
        for events in self:
            if len(events) > 0:
                if begin is None:
                    begin = events["t"][0]
                end = events["t"][-1]
        if begin is None or end is None:
            time_range = (
                timestamp.Time(microseconds=0),
                timestamp.Time(microseconds=1),
            )
        else:
            time_range = (
                timestamp.Time(microseconds=int(begin)),
                timestamp.Time(microseconds=(int(end) + 1)),
            )
        if path_hash is not None:
            assert self.time_range_cache is not None
            self.time_range_cache.set_time_range(
                path=self.path,
                path_hash=path_hash,
                time_range=time_range,
            )
        return time_range

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        if self.file_type == "aedat":
            assert self.path is not None
            with aedat.Decoder(self.path) as decoder:
                for track, packet in decoder:
                    if (
                        track.id == self.track_id
                        and track.data_type == "events"
                        and len(packet) > 0  # type: ignore
                    ):
                        yield packet  # type: ignore
        elif self.file_type == "csv":
            with csv.Decoder(
                path=self.path,
                dimensions=self.dimensions_fallback,
                has_header=self.csv_properties.has_header,
                separator=self.csv_properties.separator[0],
                t_index=self.csv_properties.t_index,
                x_index=self.csv_properties.x_index,
                y_index=self.csv_properties.y_index,
                on_index=self.csv_properties.on_index,
                t_scale=self.csv_properties.t_scale,
                t0=self.t0.to_microseconds(),
                on_value=self.csv_properties.on_value,
                off_value=self.csv_properties.off_value,
                skip_errors=self.csv_properties.skip_errors,
            ) as decoder:
                for events in decoder:
                    yield events
        elif self.file_type == "dat":
            assert self.path is not None
            with dat.Decoder(
                path=self.path,
                dimensions_fallback=self.dimensions_fallback,
                version_fallback=self.version_fallback,  # type: ignore
            ) as decoder:
                for events in decoder:
                    numpy.clip(events["payload"], 0, 1, events["payload"])
                    yield events.astype(
                        dtype=events_stream.EVENTS_DTYPE,
                        casting="unsafe",
                        copy=False,
                    )
        elif self.file_type == "es":
            assert self.path is not None
            with event_stream.Decoder(
                path=self.path, t0=self.t0.to_microseconds()
            ) as decoder:
                if self.event_type == "atis":
                    for atis_events in decoder:
                        mask = numpy.logical_not(atis_events["exposure"])
                        if len(mask) == 0:
                            continue
                        events = numpy.zeros(
                            numpy.count_nonzero(mask),
                            dtype=events_stream.EVENTS_DTYPE,
                        )
                        events["t"] = atis_events["t"][mask]
                        events["x"] = atis_events["x"][mask]
                        events["y"] = (
                            self.inner_dimensions[1] - 1 - atis_events["y"][mask]
                        )
                        events["on"] = atis_events["polarity"][mask]
                        yield events
                else:
                    for events in decoder:
                        events["y"] = self.inner_dimensions[1] - 1 - events["y"]
                        yield events
        elif self.file_type == "evt":
            with evt.Decoder(self.path, self.dimensions_fallback, self.version_fallback) as decoder:  # type: ignore
                for packet in decoder:
                    if "events" in packet:
                        yield packet["events"]
        else:
            raise Exception(f"file type {self.file_type} not implemented")
