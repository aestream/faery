from __future__ import annotations

import numpy
import pytest

import faery

from . import assets


def little_endian_bytes(array: numpy.ndarray) -> bytes:
    return array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()


@pytest.mark.parametrize("file", assets.files)
def test_low_level_decoder(file: assets.File):
    if file.format == "aedat4":
        print(f"faery.aedat.Decoder ({file.path.name})")
        assert file.dimensions is not None
        assert file.tracks is not None
        with faery.aedat.Decoder(path=file.path) as decoder:
            if file.description is not None:
                assert file.description == decoder.description()
            tracks = decoder.tracks()
            assert len(tracks) == len(file.tracks)
            for track, file_track in zip(tracks, file.tracks):
                assert track.id == file_track.id, f"{tracks=}, {file.tracks=}"
                assert track.data_type == file_track.data_type, (
                    f"{tracks=}, {file.tracks=}"
                )
                assert track.dimensions == file_track.dimensions, (
                    f"{tracks=}, {file.tracks=}"
                )
            field_to_hasher = file.field_to_hasher()
            for track, packet in decoder:
                if track.data_type == "events":
                    assert isinstance(packet, numpy.ndarray)
                    field_to_hasher["t"].update(little_endian_bytes(packet["t"]))
                    field_to_hasher["x"].update(little_endian_bytes(packet["x"]))
                    field_to_hasher["y"].update(little_endian_bytes(packet["y"]))
                    field_to_hasher["on"].update(little_endian_bytes(packet["on"]))
                elif track.data_type == "frame":
                    assert isinstance(packet, faery.aedat.Frame)
                    assert packet.pixels.shape == (
                        file.dimensions[1],
                        file.dimensions[0],
                    ), f"{packet.pixels.shape=}, {file.dimensions=}"
                    field_to_hasher["frame"].update(little_endian_bytes(packet.pixels))
                elif track.data_type == "imus":
                    assert isinstance(packet, numpy.ndarray)

                    field_to_hasher["imus"].update(little_endian_bytes(packet))
                elif track.data_type == "triggers":
                    assert isinstance(packet, numpy.ndarray)
                    field_to_hasher["triggers"].update(little_endian_bytes(packet))
                else:
                    raise Exception(f'unexpected data type "{track.data_type}"')
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "csv":
        print(f"faery.csv.Decoder ({file.path.name})")
        assert file.dimensions is not None
        assert file.t0 is not None
        with faery.csv.Decoder(
            path=file.path,
            dimensions=file.dimensions,
            has_header=True,
            separator=b","[0],
            t_index=0,
            x_index=1,
            y_index=2,
            on_index=3,
            t_scale=0.0,
            t0=faery.parse_time(file.t0).to_microseconds(),
            on_value=b"1",
            off_value=b"0",
            skip_errors=False,
        ) as decoder:
            assert decoder.dimensions == file.dimensions
            field_to_hasher = file.field_to_hasher()
            for events in decoder:
                field_to_hasher["t"].update(little_endian_bytes(events["t"]))
                field_to_hasher["x"].update(little_endian_bytes(events["x"]))
                field_to_hasher["y"].update(little_endian_bytes(events["y"]))
                field_to_hasher["on"].update(little_endian_bytes(events["on"]))
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "dat2":
        print(f"faery.dat.Decoder ({file.path.name})")
        with faery.dat.Decoder(
            path=file.path,
            dimensions_fallback=None,
            version_fallback=None,
        ) as decoder:
            assert decoder.version == "dat2"
            assert decoder.event_type == "cd"
            assert decoder.dimensions == file.dimensions
            field_to_hasher = file.field_to_hasher()
            for packet in decoder:
                field_to_hasher["t"].update(little_endian_bytes(packet["t"]))
                field_to_hasher["x"].update(little_endian_bytes(packet["x"]))
                field_to_hasher["y"].update(little_endian_bytes(packet["y"]))
                field_to_hasher["on"].update(little_endian_bytes(packet["payload"]))
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "es-atis":
        print(f"faery.es.Decoder ({file.path.name})")
        assert file.t0 is not None
        with faery.es.Decoder(
            path=file.path,
            t0=faery.parse_time(file.t0).to_microseconds(),
        ) as decoder:
            assert decoder.version == "2.0.0"
            assert decoder.event_type == "atis"
            assert decoder.dimensions == file.dimensions
            assert decoder.dimensions is not None
            field_to_hasher = file.field_to_hasher(
                [
                    "atis_t",
                    "atis_x",
                    "atis_y",
                    "atis_exposure",
                    "atis_polarity",
                    "atis_y_original",
                ]
            )
            for packet in decoder:
                field_to_hasher["atis_t"].update(little_endian_bytes(packet["t"]))
                field_to_hasher["atis_x"].update(little_endian_bytes(packet["x"]))
                field_to_hasher["atis_y"].update(
                    little_endian_bytes(decoder.dimensions[1] - 1 - packet["y"])
                )
                field_to_hasher["atis_exposure"].update(
                    little_endian_bytes(packet["exposure"])
                )
                field_to_hasher["atis_polarity"].update(
                    little_endian_bytes(packet["polarity"])
                )
                field_to_hasher["atis_y_original"].update(
                    little_endian_bytes(packet["y"])
                )
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "es-color":
        print(f"faery.es.Decoder ({file.path.name})")
        assert file.t0 is not None
        with faery.es.Decoder(
            path=file.path,
            t0=faery.parse_time(file.t0).to_microseconds(),
        ) as decoder:
            assert decoder.version == "2.0.0"
            assert decoder.event_type == "color"
            assert decoder.dimensions == file.dimensions
            assert decoder.dimensions is not None
            field_to_hasher = file.field_to_hasher()
            for packet in decoder:
                field_to_hasher["t"].update(little_endian_bytes(packet["t"]))
                field_to_hasher["x"].update(little_endian_bytes(packet["x"]))
                field_to_hasher["y"].update(
                    little_endian_bytes(decoder.dimensions[1] - 1 - packet["y"])
                )
                field_to_hasher["r"].update(little_endian_bytes(packet["r"]))
                field_to_hasher["g"].update(little_endian_bytes(packet["g"]))
                field_to_hasher["b"].update(little_endian_bytes(packet["b"]))
                field_to_hasher["y_original"].update(little_endian_bytes(packet["y"]))
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "es-dvs":
        print(f"faery.es.Decoder ({file.path.name})")
        assert file.t0 is not None
        with faery.es.Decoder(
            path=file.path,
            t0=faery.parse_time(file.t0).to_microseconds(),
        ) as decoder:
            assert decoder.version == "2.0.0"
            assert decoder.event_type == "dvs"
            assert decoder.dimensions == file.dimensions
            assert decoder.dimensions is not None
            field_to_hasher = file.field_to_hasher()
            for packet in decoder:
                field_to_hasher["t"].update(little_endian_bytes(packet["t"]))
                field_to_hasher["x"].update(little_endian_bytes(packet["x"]))
                field_to_hasher["y"].update(
                    little_endian_bytes(decoder.dimensions[1] - 1 - packet["y"])
                )
                field_to_hasher["on"].update(little_endian_bytes(packet["on"]))
                if "y_original" in field_to_hasher:
                    field_to_hasher["y_original"].update(
                        little_endian_bytes(packet["y"])
                    )
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "es-generic":
        print(f"faery.es.Decoder ({file.path.name})")
        assert file.t0 is not None
        with faery.es.Decoder(
            path=file.path,
            t0=faery.parse_time(file.t0).to_microseconds(),
        ) as decoder:
            assert decoder.version == "2.0.0"
            assert decoder.event_type == "generic"
            assert decoder.dimensions == file.dimensions
            assert file.content_lines is not None
            field_to_hasher = file.field_to_hasher()
            index = 0
            first_t: int | None = None
            last_t: int | None = None
            for packet in decoder:
                field_to_hasher["t"].update(little_endian_bytes(packet["t"]))
                for t, bytes in packet:
                    if first_t is None:
                        first_t = t
                    last_t = t
                    assert bytes == file.content_lines[index]
                    index += 1
            assert first_t is not None
            assert last_t is not None
            time_range = (
                (first_t * faery.us).to_timecode(),
                ((last_t + 1) * faery.us).to_timecode(),
            )
            assert time_range == file.time_range, f"{time_range=}, {file.time_range=}"
            assert field_to_hasher["t"].hexdigest() == file.field_to_digest["t"], (
                f'{file=}, field="t"'
            )
    elif file.format == "evt2":
        print(f"faery.evt.Decoder ({file.path.name})")
        assert file.dimensions is not None
        with faery.evt.Decoder(
            path=file.path,
            dimensions_fallback=file.dimensions,
            version_fallback=None,
        ) as decoder:
            assert decoder.version == "evt2"
            assert decoder.dimensions == file.dimensions
            field_to_hasher = file.field_to_hasher()
            for packet in decoder:
                if "events" in packet:
                    field_to_hasher["t"].update(
                        little_endian_bytes(packet["events"]["t"])
                    )
                    field_to_hasher["x"].update(
                        little_endian_bytes(packet["events"]["x"])
                    )
                    field_to_hasher["y"].update(
                        little_endian_bytes(packet["events"]["y"])
                    )
                    field_to_hasher["on"].update(
                        little_endian_bytes(packet["events"]["on"])
                    )
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    elif file.format == "evt3":
        print(f"faery.evt.Decoder ({file.path.name})")
        with faery.evt.Decoder(
            path=file.path,
            dimensions_fallback=None,
            version_fallback=None,
        ) as decoder:
            assert decoder.version == "evt3"
            assert decoder.dimensions == file.dimensions
            field_to_hasher = file.field_to_hasher()
            for packet in decoder:
                if "events" in packet:
                    field_to_hasher["t"].update(
                        little_endian_bytes(packet["events"]["t"])
                    )
                    field_to_hasher["x"].update(
                        little_endian_bytes(packet["events"]["x"])
                    )
                    field_to_hasher["y"].update(
                        little_endian_bytes(packet["events"]["y"])
                    )
                    field_to_hasher["on"].update(
                        little_endian_bytes(packet["events"]["on"])
                    )
            for field, hasher in field_to_hasher.items():
                assert hasher.hexdigest() == file.field_to_digest[field], (
                    f"{file=}, {field=}"
                )
    else:
        raise Exception(f'unknown format "{file.format}"')


@pytest.mark.parametrize(
    "file",
    [file for file in assets.files if file.format in assets.DECODE_DVS_FORMATS],
)
def test_high_level_decoder(file: assets.File):
    assert file.dimensions is not None
    print(f"faery.stream_from_file ({file.path.name})")
    if file.t0 is None:
        stream = faery.events_stream_from_file(
            file.path,
            dimensions_fallback=file.dimensions,
        )
    else:
        stream = faery.events_stream_from_file(
            file.path,
            dimensions_fallback=file.dimensions,
            t0=file.t0,
        )
    time_range = stream.time_range()
    assert (
        time_range[0].to_timecode(),
        time_range[1].to_timecode(),
    ) == file.time_range, f"{stream.time_range()=}, {file.time_range=}"
    assert stream.dimensions() == file.dimensions, (
        f"{stream.dimensions()=}, {file.dimensions=}"
    )
    field_to_hasher = file.field_to_hasher(fields=["t", "x", "y", "on"])
    for events in stream:
        assert events.dtype == faery.EVENTS_DTYPE
        field_to_hasher["t"].update(little_endian_bytes(events["t"]))
        field_to_hasher["x"].update(little_endian_bytes(events["x"]))
        field_to_hasher["y"].update(little_endian_bytes(events["y"]))
        field_to_hasher["on"].update(little_endian_bytes(events["on"]))
    for field, hasher in field_to_hasher.items():
        assert hasher.hexdigest() == file.field_to_digest[field], f"{file=}, {field=}"
