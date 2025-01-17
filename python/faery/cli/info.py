import hashlib
import json
import pathlib
import typing

import faery

from . import command, list_filters


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (["faery info <input>"], "prints information about a file")

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return {"info"}

    @typing.override
    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.add_argument("path", help="path of the input file")
        parser.add_argument(
            "--track-id",
            type=list_filters.parse_optional_int,
            default="none",
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--dimensions-fallback",
            type=list_filters.parse_dimensions,
            default=(1280, 720),
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--version-fallback",
            choices=list(typing.get_args(faery.EventsFileVersion)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--t0",
            type=list_filters.parse_time,
            default=0,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--file-type",
            choices=list(typing.get_args(faery.EventsFileType)) + ["none"],
            default="none",
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--no-csv-has-header",
            action="store_const",
            const=False,
            default=True,
            dest="csv_has_header",
        )
        parser.add_argument(
            "--csv-separator",
            default=",",
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-t-index",
            type=int,
            default=0,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-x-index",
            type=int,
            default=1,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-y-index",
            type=int,
            default=2,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-on-index",
            type=int,
            default=3,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-t-scale",
            type=float,
            default=0.0,
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-on-value",
            default="1",
            help="(default: %(default)s)",
        )
        parser.add_argument(
            "--csv-off-value",
            default="0",
            help="(default: %(default)s)",
        )
        parser.add_argument("--csv-skip-errors", action="store_true")
        args = parser.parse_args(args=arguments)

        file_type = faery.enums.events_file_type_guess(pathlib.Path(args.path))
        info: dict[str, typing.Any] = {
            "file_type": file_type,
        }
        if file_type == "aedat":
            info["stream_compatible"] = True
            with faery.aedat.Decoder(args.path) as decoder:
                info["metadata"] = {
                    "description": decoder.description(),
                    "tracks": [
                        {
                            "id": track.id,
                            "data_type": track.data_type,
                            "dimensions": track.dimensions,
                        }
                        for track in decoder.tracks()
                    ],
                }
        elif file_type == "csv":
            info["stream_compatible"] = True
        elif file_type == "dat":
            with faery.dat.Decoder(
                args.path,
                args.dimensions_fallback,
                "dat2" if args.version_fallback == "none" else args.version_fallback,  # type: ignore
            ) as decoder:
                event_type = decoder.event_type
                if event_type == "cd":
                    info["stream_compatible"] = True
                else:
                    info["stream_compatible"] = False
                info["metadata"] = {
                    "version": decoder.version,
                    "event_type": event_type,
                }
                if not info["stream_compatible"]:
                    info["metadata"]["dimensions"] = decoder.dimensions  # type: ignore
        elif file_type == "es":
            with faery.event_stream.Decoder(
                path=args.path,
                t0=args.t0,
            ) as decoder:
                event_type = decoder.event_type
                if event_type == "dvs" or event_type == "atis":
                    info["stream_compatible"] = True
                else:
                    info["stream_compatible"] = False
                info["metadata"] = {
                    "version": decoder.version,
                    "event_type": event_type,
                }
                if not info["stream_compatible"]:
                    info["metadata"]["dimensions"] = decoder.dimensions  # type: ignore
        elif file_type == "evt":
            with faery.evt.Decoder(
                args.path,
                args.dimensions_fallback,
                "evt3" if args.version_fallback == "none" else args.version_fallback,  # type: ignore
            ) as decoder:
                info["stream_compatible"] = True
                info["metadata"] = {
                    "version": decoder.version,
                }
        if info["stream_compatible"]:
            stream = faery.events_stream_from_file(
                path=args.path,
                track_id=None if args.track_id == "none" else args.track_id,
                dimensions_fallback=args.dimensions_fallback,
                version_fallback=(
                    None if args.version_fallback == "none" else args.version_fallback
                ),
                t0=args.t0,
                file_type=None if args.file_type == "none" else args.file_type,
                csv_properties=faery.CsvProperties(
                    has_header=args.csv_has_header,
                    separator=args.csv_separator.encode(),
                    t_index=args.csv_t_index,
                    x_index=args.csv_x_index,
                    y_index=args.csv_y_index,
                    on_index=args.csv_on_index,
                    t_scale=args.csv_t_scale,
                    on_value=args.csv_on_value.encode(),
                    off_value=args.csv_off_value.encode(),
                    skip_errors=args.csv_skip_errors,
                ),
            )
            dimensions = stream.dimensions()
            time_range = stream.time_range()
            t_hasher = hashlib.sha3_224()
            x_hasher = hashlib.sha3_224()
            y_hasher = hashlib.sha3_224()
            on_hasher = hashlib.sha3_224()
            for events in stream:
                t_hasher.update(events["t"].tobytes())
                x_hasher.update(events["x"].tobytes())
                y_hasher.update(events["y"].tobytes())
                on_hasher.update(events["on"].tobytes())
            info["data"] = {
                "dimensions": dimensions,
                "time_range": [
                    time_range[0].to_timecode(),
                    time_range[1].to_timecode(),
                ],
                "time_range_us": [
                    time_range[0].to_microseconds(),
                    time_range[1].to_microseconds(),
                ],
                "duration": (time_range[1] - time_range[0]).to_timecode(),
                "t_hash": t_hasher.hexdigest(),
                "x_hash": x_hasher.hexdigest(),
                "y_hash": y_hasher.hexdigest(),
                "on_hash": on_hasher.hexdigest(),
            }
        print(json.dumps(info, indent=4))
