import pathlib
import shutil
import time

import pytest

import faery

from . import assets, test_decoders

data_generated = pathlib.Path(__file__).resolve().parent / "data_generated"
if data_generated.is_dir():
    shutil.rmtree(data_generated)
data_generated.mkdir()


@pytest.mark.parametrize("file", assets.files)
def test_low_level_decoder_encoder(file: assets.File):
    output = data_generated / file.path.name
    if file.format == "aedat":
        print(f"faery.aedat.Decoder + faery.aedat.Encoder ({file.path.name})")
        with faery.aedat.Decoder(file.path) as decoder:
            with faery.aedat.Encoder(
                path=output,
                description_or_tracks=decoder.description(),
                compression=faery.aedat.LZ4_HIGHEST,
            ) as encoder:
                for track, packet in decoder:
                    encoder.write(track.id, packet)
    elif file.format == "csv":
        print(f"faery.csv.Decoder + faery.csv.Encoder ({file.path.name})")
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
            with faery.csv.Encoder(
                path=output,
                separator=b","[0],
                header=True,
                dimensions=file.dimensions,
            ) as encoder:
                for events in decoder:
                    encoder.write(events)
    elif file.format == "dat2":
        print(f"faery.dat.Decoder + faery.dat.Encoder ({file.path.name})")
        with faery.dat.Decoder(
            path=file.path,
            dimensions_fallback=None,
            version_fallback=None,
        ) as decoder:
            assert decoder.dimensions is not None
            with faery.dat.Encoder(
                path=output,
                version=decoder.version,
                event_type="cd",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "es-atis":
        print(
            f"faery.event_stream.Decoder + faery.event_stream.Encoder ({file.path.name})"
        )
        with faery.event_stream.Decoder(
            path=file.path,
            t0=0,
        ) as decoder:
            assert decoder.dimensions is not None
            with faery.event_stream.Encoder(
                path=output,
                event_type="atis",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "es-color":
        print(
            f"faery.event_stream.Decoder + faery.event_stream.Encoder ({file.path.name})"
        )
        with faery.event_stream.Decoder(
            path=file.path,
            t0=0,
        ) as decoder:
            assert decoder.dimensions is not None
            with faery.event_stream.Encoder(
                path=output,
                event_type="color",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "es-dvs":
        print(
            f"faery.event_stream.Decoder + faery.event_stream.Encoder ({file.path.name})"
        )
        with faery.event_stream.Decoder(
            path=file.path,
            t0=0,
        ) as decoder:
            assert decoder.dimensions is not None
            with faery.event_stream.Encoder(
                path=output,
                event_type="dvs",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "es-generic":
        print(
            f"faery.event_stream.Decoder + faery.event_stream.Encoder ({file.path.name})"
        )
        with faery.event_stream.Decoder(
            path=file.path,
            t0=0,
        ) as decoder:
            with faery.event_stream.Encoder(
                path=output,
                event_type="generic",
                zero_t0=True,
                dimensions=None,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "evt2":
        print(f"faery.evt.Decoder + faery.evt.Encoder ({file.path.name})")
        with faery.evt.Decoder(
            path=file.path,
            dimensions_fallback=file.dimensions,
            version_fallback=None,
        ) as decoder:
            with faery.evt.Encoder(
                path=output,
                version="evt2",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    elif file.format == "evt3":
        print(f"faery.evt.Decoder + faery.evt.Encoder ({file.path.name})")
        with faery.evt.Decoder(
            file.path,
            dimensions_fallback=None,
            version_fallback=None,
        ) as decoder:
            with faery.evt.Encoder(
                path=output,
                version="evt3",
                zero_t0=True,
                dimensions=decoder.dimensions,
            ) as encoder:
                for packet in decoder:
                    encoder.write(packet)
    else:
        raise Exception(f'unknown format "{file.format}"')
    generated_file = file.clone_with(
        path=output,
        format=file.format,
        field_to_digest=file.field_to_digest,
        header_lines=file.header_lines,
        tracks=file.tracks,
        content_lines=file.content_lines,
        t0=file.t0,
    )
    test_decoders.test_low_level_decoder(generated_file)
    if file.format in assets.DECODE_DVS_FORMATS:
        test_decoders.test_high_level_decoder(generated_file)


@pytest.mark.parametrize(
    "file",
    [file for file in assets.files if file.format == "aedat"],
)
def test_aedat_compression(file: assets.File):
    for compression, level in (
        faery.aedat.LZ4_FASTEST,
        faery.aedat.LZ4_DEFAULT,
        faery.aedat.LZ4_HIGHEST,
        faery.aedat.ZSTD_FASTEST,
        faery.aedat.ZSTD_DEFAULT,
        (
            "zstd",
            6,
        ),
        # faery.aedat.ZSTD_HIGHEST = ("zstd", 22) is too slow for tests
    ):
        output = (
            data_generated / f"{file.path.stem}_{compression}_{level}{file.path.suffix}"
        )
        print(
            f"faery.aedat.Decoder + faery.aedat.Encoder, {compression}@{level} ({file.path.name})"
        )
        begin = time.monotonic()
        with faery.aedat.Decoder(file.path) as decoder:
            with faery.aedat.Encoder(
                path=output,
                description_or_tracks=decoder.tracks(),
                compression=(compression, level),  # type: ignore
            ) as encoder:
                for track, packet in decoder:
                    encoder.write(track.id, packet)
        print(f"decoded + encoded in {time.monotonic() - begin:.3f} s")
        generated_file = file.clone_with(
            path=output,
            format=file.format,
            field_to_digest=file.field_to_digest,
            header_lines=None,
            tracks=file.tracks,
            content_lines=file.content_lines,
            t0=file.t0,
        )
        test_decoders.test_low_level_decoder(generated_file)
        if file.format in assets.DECODE_DVS_FORMATS:
            test_decoders.test_high_level_decoder(generated_file)


@pytest.mark.parametrize(
    "file",
    [file for file in assets.files if file.format in assets.DECODE_DVS_FORMATS],
)
def test_high_level_decoder_encoder(file: assets.File):
    assert file.dimensions is not None
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
    for output_format in assets.ENCODE_DVS_FORMATS:
        output = (
            data_generated
            / f"{file.path.name.replace('.', '_')}-as-{output_format}.{assets.format_to_extension(output_format)}"
        )
        print(f"faery.stream_from_file + save ({file.path.name} -> {output.name})")
        if output_format in ["dat2", "evt2", "evt3"]:
            version = output_format
        else:
            version = None
        t0 = stream.to_file(
            output,
            version=version,  # type: ignore
        )
        generated_file = file.clone_with(
            path=output,
            format=output_format,
            field_to_digest={
                "t": file.field_to_digest["t"],
                "x": file.field_to_digest["x"],
                "y": file.field_to_digest["y"],
                "on": file.field_to_digest["on"],
            },
            header_lines=None,
            tracks=(
                [
                    faery.aedat.Track(
                        id=0, data_type="events", dimensions=file.dimensions
                    )
                ]
                if output_format == "aedat"
                else None
            ),
            content_lines=None,
            t0=t0,
        )
        test_decoders.test_low_level_decoder(generated_file)
        test_decoders.test_high_level_decoder(generated_file)
