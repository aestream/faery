import collections.abc
import pathlib
import typing
import uuid

import numpy

from . import enums, frame_stream, timestamp

if typing.TYPE_CHECKING:
    from .types import aedat  # type: ignore
    from .types import csv  # type: ignore
    from .types import dat  # type: ignore
    from .types import event_stream  # type: ignore
    from .types import evt  # type: ignore
else:
    from .extension import aedat, csv, dat, event_stream, evt


def events_to_file(
    stream: collections.abc.Iterable[numpy.ndarray],
    path: typing.Union[pathlib.Path, str, None],
    dimensions: tuple[int, int],
    version: typing.Optional[enums.EventsFileVersion] = None,
    zero_t0: bool = True,
    compression: typing.Optional[
        typing.Tuple[enums.EventsFileCompression, int]
    ] = aedat.LZ4_DEFAULT,
    csv_separator: bytes = b",",
    csv_header: bool = True,
    file_type: typing.Optional[enums.EventsFileType] = None,
) -> str:
    """Writes the stream to an event file (supports .aedat4, .es, .raw, and .dat).

    version is only used if the file type is EVT (.raw) or DAT.

    zero_t0 is only used if the file type is ES, EVT (.raw) or DAT.
    The original t0 is stored in the header of EVT and DAT files, and is discarded if the file type is ES.

    compression is only used if the file type is AEDAT.

    Args:
        stream: An iterable of event arrays (structured arrays with dtype faery.EVENTS_DTYPE).
        path: Path of the output event file or None for stdout (stdout is only compatible with the CSV format).
        dimensions: Width and height of the sensor.
        version: Version for EVT (.raw) and DAT files. Defaults to "dat2" for DAT and "evt3" for EVT.
        zero_t0: Whether to normalize timestamps and write the offset in the header for EVT (.raw) and DAT files. Defaults to True.
        compression: Compression for aedat files. Defaults to ("lz4", 1).
        csv_separator: Separator for CSV files, must be a single character. Defaults to b",".
        csv_header: Whether to add a header to the CSV file. Defaults to True.
        file_type: Override the type determination algorithm. Defaults to None.

    Returns:
        The original t0 as a timecode if the file type is ES, EVT (.raw), or DAT, and if `zero_t0` is true. 0 as a timecode otherwise.
        To reconstruct the original timestamps when decoding ES files with Faery, pass the returned value to `faery.stream_from_file`.
        EVT (.raw) and DAT files do not need this (t0 is written in their header), but it is returned here anyway for compatibility
        with software than do not support the t0 header field.
    """
    if path is None:
        if file_type != "csv":
            raise Exception(
                "`file_type` must be CSV (`faery.FileType.CSV`) when writing to stdout (`path` is None)"
            )
    else:
        path = pathlib.Path(path)
        file_type = (
            enums.events_file_type_guess(path)
            if file_type is None
            else enums.validate_events_file_type(file_type)
        )
    if version is not None:
        version = enums.validate_events_file_version(version)
    if compression is not None:
        compression = (
            enums.validate_events_file_compression(compression[0]),
            compression[1],
        )
    if file_type == "aedat":
        assert path is not None
        with aedat.Encoder(
            path,
            description_or_tracks=[
                aedat.Track(id=0, data_type="events", dimensions=dimensions),
            ],
            compression=compression,
        ) as encoder:
            for events in stream:
                encoder.write(0, events)
        t0 = 0
    elif file_type == "csv":
        assert len(csv_separator) == 1
        with csv.Encoder(
            path=path,
            separator=csv_separator[0],
            header=csv_header,
            dimensions=dimensions,
        ) as encoder:

            for events in stream:

                encoder.write(events)
        t0 = 0
    elif file_type == "dat":
        assert path is not None
        with dat.Encoder(
            path,
            version="dat2" if version is None else version,  # type: ignore
            event_type="cd",
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            for events in stream:
                events = events.astype(
                    dtype=numpy.dtype(
                        [
                            ("t", "<u8"),
                            ("x", "<u2"),
                            ("y", "<u2"),
                            ("payload", "u1"),
                        ]
                    ),
                    casting="unsafe",
                    copy=False,
                )
                encoder.write(events)
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
    elif file_type == "es":
        assert path is not None
        with event_stream.Encoder(
            path,
            event_type="dvs",
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            for events in stream:
                events["y"] = dimensions[1] - 1 - events["y"]
                encoder.write(events)
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
    elif file_type == "evt":
        assert path is not None
        with evt.Encoder(
            path,
            version="evt3" if version is None else version,  # type: ignore
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            for events in stream:
                encoder.write({"events": events})
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
    else:
        raise Exception(f"file type {file_type} not implemented")
    return timestamp.timestamp_to_timecode(t0)


def frames_to_files(
    stream: collections.abc.Iterable[frame_stream.Rgba8888Frame],
    path_pattern: typing.Union[pathlib.Path, str],
    compression_level: enums.ImageFileCompressionLevel = "fast",
    file_type: typing.Optional[enums.ImageFileType] = None,
):
    path_pattern = pathlib.Path(path_pattern)
    index_uuid = str(uuid.uuid4())
    timestamp_uuid = str(uuid.uuid4())
    found = False
    try:
        for part in path_pattern.parts:
            formatted_part = part.format_map(
                {
                    "index": index_uuid,
                    "timestamp": timestamp_uuid,
                    "i": index_uuid,
                    "t": timestamp_uuid,
                }
            )
            if index_uuid in formatted_part or timestamp_uuid in formatted_part:
                found = True
                break
    except KeyError as error:
        raise Exception(
            f'unexpected variable "{error.args[0]}" in "{path_pattern}" (Rgba8888FrameStream.to_files calculates paths with Python\'s `format` method and the variables {{i}}/{{index}} and/or {{t}}/{{timestamp}}, see https://docs.python.org/3/library/string.html#formatstrings to escape other variables)'
        )
    if not found:
        raise Exception(
            f'at least one of {{i}}/{{index}} or {{t}}/{{timestamp}} must appear in the path pattern (for example "output/{{i:05}}.png" or "output/{{index:05}}_{{timestamp:010}}.png")'
        )
    index = 0
    for frame in stream:
        path = pathlib.Path(
            *(
                part.format_map(
                    {
                        "index": index,
                        "timestamp": frame.t,
                        "i": index,
                        "t": frame.t,
                    }
                )
                for part in path_pattern.parts
            )
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        frame.to_file(
            path=path,
            compression_level=compression_level,
            file_type=file_type,
        )
        index += 1
