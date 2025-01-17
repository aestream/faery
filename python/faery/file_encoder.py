import collections.abc
import pathlib
import typing
import uuid

import numpy
import numpy.typing

from . import enums, events_stream_state, frame_stream, frame_stream_state, timestamp

if typing.TYPE_CHECKING:
    from .types import aedat, csv, dat, event_stream, evt, gif, mp4  # type: ignore
else:
    from .extension import aedat, csv, dat, event_stream, evt, gif, mp4


def with_write_suffix(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(f"{path.suffix}.write")


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
    use_write_suffix: bool = True,
    on_progress: typing.Callable[
        [events_stream_state.EventsStreamState], None
    ] = lambda _: None,
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
        write_path = None
    else:
        path = pathlib.Path(path)
        if use_write_suffix:
            write_path = with_write_suffix(path=path)
        else:
            write_path = None
        path.parent.mkdir(exist_ok=True, parents=True)
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
    state_manager = events_stream_state.StateManager(
        stream=stream, on_progress=on_progress
    )
    if file_type == "aedat":
        assert path is not None
        with aedat.Encoder(
            path=path if write_path is None else write_path,
            description_or_tracks=[
                aedat.Track(id=0, data_type="events", dimensions=dimensions),
            ],
            compression=compression,
        ) as encoder:
            state_manager.start()
            for events in stream:
                encoder.write(0, events)
                state_manager.commit(events)
        if write_path is not None:
            write_path.replace(path)
        state_manager.end()
        t0 = 0
    elif file_type == "csv":
        assert len(csv_separator) == 1
        with csv.Encoder(
            path=path if write_path is None else write_path,
            separator=csv_separator[0],
            header=csv_header,
            dimensions=dimensions,
        ) as encoder:
            state_manager.start()
            for events in stream:
                encoder.write(events)
                state_manager.commit(events)
        if write_path is not None:
            assert path is not None
            write_path.replace(path)
        state_manager.end()
        t0 = 0
    elif file_type == "dat":
        assert path is not None
        with dat.Encoder(
            path=path if write_path is None else write_path,
            version="dat2" if version is None else version,  # type: ignore
            event_type="cd",
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            state_manager.start()
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
                state_manager.commit(events)
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
        if write_path is not None:
            write_path.replace(path)
        state_manager.end()
    elif file_type == "es":
        assert path is not None
        with event_stream.Encoder(
            path=path if write_path is None else write_path,
            event_type="dvs",
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            state_manager.start()
            for events in stream:
                events["y"] = dimensions[1] - 1 - events["y"]
                encoder.write(events)
                state_manager.commit(events)
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
        if write_path is not None:
            write_path.replace(path)
        state_manager.end()
    elif file_type == "evt":
        assert path is not None
        with evt.Encoder(
            path=path if write_path is None else write_path,
            version="evt3" if version is None else version,  # type: ignore
            zero_t0=zero_t0,
            dimensions=dimensions,
        ) as encoder:
            state_manager.start()
            for events in stream:
                encoder.write({"events": events})
                state_manager.commit(events)
            t0_candidate = encoder.t0()
            if t0_candidate is None:
                t0 = 0
            else:
                t0 = t0_candidate
        if write_path is not None:
            write_path.replace(path)
        state_manager.end()
    else:
        raise Exception(f"file type {file_type} not implemented")
    return timestamp.Time(microseconds=t0).to_timecode()


def frames_to_files(
    stream: collections.abc.Iterable[frame_stream.Frame],
    path_pattern: typing.Union[pathlib.Path, str],
    compression_level: enums.ImageFileCompressionLevel = "fast",
    quality: int = 100,
    file_type: typing.Optional[enums.ImageFileType] = None,
    use_write_suffix: bool = True,
    on_progress: typing.Callable[[frame_stream.OutputState], None] = lambda _: None,
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
            f'unexpected variable "{error.args[0]}" in "{path_pattern}" (FrameStream.to_files calculates paths with Python\'s `format` method and the variables {{i}}/{{index}} and/or {{t}}/{{timestamp}}, see https://docs.python.org/3/library/string.html#formatstrings to escape other variables)'
        )
    if not found:
        raise Exception(
            f'at least one of {{i}}/{{index}} or {{t}}/{{timestamp}} must appear in the path pattern (for example "output/{{i:05}}.png" or "output/{{index:05}}_{{timestamp:010}}.png")'
        )
    state_manager = frame_stream_state.StateManager(
        stream=stream, on_progress=on_progress
    )
    state_manager.start()
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
        frame.to_file(
            path=path,
            compression_level=compression_level,
            file_type=file_type,
            use_write_suffix=use_write_suffix,
        )
        state_manager.commit(frame=frame)
        index += 1
    state_manager.end()


def frames_to_file(
    stream: collections.abc.Iterable[frame_stream.Frame],
    path: typing.Union[pathlib.Path, str],
    dimensions: tuple[int, int],
    frame_rate: float = 60.0,
    crf: float = 17.0,
    preset: enums.VideoFilePreset = "medium",
    tune: enums.VideoFileTune = "none",
    profile: enums.VideoFileProfile = "baseline",
    quality: int = 100,
    rewind: bool = False,
    skip: int = 0,
    file_type: typing.Optional[enums.VideoFileType] = None,
    use_write_suffix: bool = True,
    on_progress: typing.Callable[[frame_stream.OutputState], None] = lambda _: None,
):
    """
    crf: only used if the file type is mp4

    preset: trade-off between size and compression speed for mp4, sets the `fast` flag for gif as follows:
        - "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "none" => fast = True
        - "slow", "slower", "veryslow", "placebo" => fast = False

    tune: only used if the file type is mp4

    profile: only used if the file type is mp4

    quality: only used if the file type is gif, must be a value in the range [1, 100]

    rewind: whether to play the video in reverse after the forward pass.
    This option stores all the frames in memory and can result in high memory usage on long videos.

    skip: remove this many frames from the beginning of the recording.
    The first frames have a different appearance when using a decay that is larger than the frame duration.
    Removing them hides potentially important data but yields more consistent representations, especially when using `rewind`.
    """
    path = pathlib.Path(path)
    if use_write_suffix:
        write_path = with_write_suffix(path=path)
    else:
        write_path = None
    path.parent.mkdir(exist_ok=True, parents=True)
    preset = enums.validate_video_file_preset(preset)
    tune = enums.validate_video_file_tune(tune)
    profile = enums.validate_video_file_profile(profile)
    if file_type is None:
        file_type = enums.video_file_type_guess(path)
    else:
        file_type = enums.validate_video_file_type(file_type)
    state_manager = frame_stream_state.StateManager(
        stream=stream, on_progress=on_progress
    )
    frames: list[frame_stream.Frame] = []
    if file_type == "mp4":
        video_encoder = mp4.Encoder(
            path=path if write_path is None else write_path,
            dimensions=dimensions,
            frame_rate=frame_rate,
            crf=crf,
            preset=preset,
            tune=tune,
            profile=profile,
        )
    elif file_type == "gif":
        video_encoder = gif.Encoder(
            path=path if write_path is None else write_path,
            dimensions=dimensions,
            frame_rate=frame_rate,
            quality=quality,
            fast=preset
            in {
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "none",
            },
        )
    else:
        raise Exception(f"file type {file_type} not implemented")
    with video_encoder as encoder:
        state_manager.start()
        for frame in stream:
            if skip > 0:
                skip -= 1
            else:
                encoder.write(frame=frame.pixels)
                if rewind:
                    frames.append(frame)
            state_manager.commit(frame=frame)
        if rewind and len(frames) >= 3:
            for frame in reversed(frames[1 : len(frames) - 1]):
                encoder.write(frame=frame.pixels)
    if write_path is not None:
        write_path.replace(path)
    state_manager.end()
