import pathlib

import numpy
import pytest

import faery

data = pathlib.Path(__file__).resolve().parent / "data"


def decode(file_path: pathlib.Path) -> list[faery.aedat.Frame]:
    with faery.aedat.Decoder(path=file_path) as decoder:
        frames = [packet for _, packet in decoder]
    return frames  # type: ignore


def test_decode_gray16():
    frames = decode(data / "test_data_gray16.aedat4")
    assert len(frames) == 1
    assert frames[0].format == "I;16"
    assert frames[0].pixels.dtype == numpy.uint16
    assert frames[0].pixels.tolist() == [[1, 2], [3, 1023]]


def test_unknown_format_does_not_hide_later_frames():
    with pytest.warns(UserWarning, match="OpenCV type code 1"):
        frames = decode(data / "test_data_unknown_then_gray16.aedat4")
    assert [frame.format for frame in frames] == ["I;16"]


def test_description_without_path_attributes():
    with faery.aedat.Decoder(path=data / "test_data_gray16.aedat4") as decoder:
        description = decoder.description()
    assert description[0].name == "outInfo"
    assert description[0].path is None


def test_round_trip_through_the_encoder(tmp_path: pathlib.Path):
    source = data / "test_data_gray16.aedat4"
    output = tmp_path / "round_trip.aedat4"
    with (
        faery.aedat.Decoder(path=source) as decoder,
        faery.aedat.Encoder(
            path=output,
            description=decoder.description(),
            compression=faery.aedat.LZ4_HIGHEST,
        ) as encoder,
    ):
        for track, packet in decoder:
            encoder.write(track.id, packet)
    before, after = decode(source)[0], decode(output)[0]
    assert before.format == after.format
    assert before.pixels.dtype == after.pixels.dtype
    assert numpy.array_equal(before.pixels, after.pixels)
