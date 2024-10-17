import argparse
import inspect
import math

import faery
import numpy

PADDING_TOP: int = 20
TITLE_SIZE: int = 20
TITLE_PADDING_BOTTOM: int = 16
LABEL_SIZE: int = 16
LABEL_OFFSET: int = -1
ROW_HEIGHT: int = 20
ROW_GAP: int = 6
ROWS_PADDING: int = 20
PADDING_LEFT: int = 20
PADDING_RIGHT: int = 20
FONT_RATIO: float = 0.6
COLUMN_GAP: int = 20
WIDTH: int = 720


def add_to_subparsers(subparsers: argparse._SubParsersAction):
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "colormaps", help="generate an image of available colormaps"
    )
    parser.add_argument(
        "output",
        help="path of the output PNG file",
    )


def run(args: argparse.Namespace):
    names_and_colormaps: list[tuple[str, faery.Colormap]] = inspect.getmembers(
        faery.colormaps, lambda member: isinstance(member, faery.Colormap)
    )
    type_to_names_and_colormaps: dict[str, list[tuple[str, faery.Colormap]]] = {}
    for name, colormap in names_and_colormaps:
        if not colormap.type in type_to_names_and_colormaps:
            type_to_names_and_colormaps[colormap.type] = []
        type_to_names_and_colormaps[colormap.type].append((name, colormap))
    height = PADDING_TOP
    maximum_label_width = 0
    for names_and_colormaps in type_to_names_and_colormaps.values():
        names_and_colormaps.sort(key=lambda name_and_colormap: name_and_colormap[0])
        height += (
            TITLE_SIZE
            + TITLE_PADDING_BOTTOM
            + ROW_HEIGHT * len(names_and_colormaps)
            + ROW_GAP * (len(names_and_colormaps) - 1)
            + ROWS_PADDING
        )
        maximum_label_width = max(
            maximum_label_width,
            max([len(name) for name, _ in names_and_colormaps]),
        )
    maximum_label_width = int(math.ceil(maximum_label_width * LABEL_SIZE * FONT_RATIO))
    colorbar_width = (
        WIDTH - PADDING_LEFT - PADDING_RIGHT - COLUMN_GAP - maximum_label_width
    )
    colorbar_points = numpy.arange(0, colorbar_width, dtype=numpy.float64) / (
        colorbar_width - 1
    )
    frame = numpy.full((height, WIDTH, 3), 0x19, dtype=numpy.uint8)
    offset = PADDING_TOP
    for type in sorted(type_to_names_and_colormaps.keys()):
        faery.image.annotate(
            frame=frame,
            text=type.capitalize(),
            x_offset=PADDING_LEFT,
            y_offset=offset,
            scale=TITLE_SIZE,
            color=(0xFF, 0xFF, 0xFF, 0xFF),
        )
        offset += TITLE_SIZE + TITLE_PADDING_BOTTOM
        for name, colormap in type_to_names_and_colormaps[type]:
            colormap_points = numpy.arange(
                0, colormap.data.shape[0], dtype=numpy.float64
            ) / (colormap.data.shape[0] - 1)
            colorbar = numpy.tile(
                numpy.column_stack(
                    (
                        numpy.interp(
                            colorbar_points, colormap_points, colormap.data[:, 0]
                        )
                        * 255.0,
                        numpy.interp(
                            colorbar_points, colormap_points, colormap.data[:, 1]
                        )
                        * 255.0,
                        numpy.interp(
                            colorbar_points, colormap_points, colormap.data[:, 2]
                        )
                        * 255.0,
                    )
                ),
                (ROW_HEIGHT, 1, 1),
            )
            colorbar = numpy.round(colorbar).astype(numpy.uint8)
            frame[
                offset : offset + ROW_HEIGHT,
                PADDING_LEFT : PADDING_LEFT + colorbar_width,
            ] = colorbar
            faery.image.annotate(
                frame=frame,
                text=name,
                x_offset=PADDING_LEFT + colorbar_width + COLUMN_GAP,
                y_offset=offset + LABEL_OFFSET,
                scale=LABEL_SIZE,
                color=(0xFF, 0xFF, 0xFF, 0xFF),
            )
            offset += ROW_HEIGHT + ROW_GAP
        offset += ROWS_PADDING - ROW_GAP
    with open(args.output, "wb") as output:
        output.write(faery.image.encode(frame=frame, compression_level="best"))
