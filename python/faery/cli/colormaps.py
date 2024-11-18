import inspect
import math
import typing

import numpy

import faery

from . import command

PADDING_TOP: int = 20
TITLE_SIZE: int = 20
TITLE_PADDING_BOTTOM: int = 16
SUBTITLE_SIZE: int = 16
SUBTITLE_OFFSET: int = 4
LABEL_SIZE: int = 16
LABEL_OFFSET: int = -1
ROW_HEIGHT: int = 20
ROW_GAP: int = 6
ROWS_PADDING: int = 20
PADDING_LEFT: int = 20
PADDING_RIGHT: int = 20
FONT_RATIO: float = 0.6
COLUMN_GAP: int = 20
WIDTH: int = 1440


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (
            ["faery colormaps <output>"],
            "render available colormaps",
        )

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return {"colormaps"}

    @typing.override
    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.add_argument("output", help="path of the output PNG file")
        args = parser.parse_args(args=arguments)

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
        maximum_label_width = int(
            math.ceil(maximum_label_width * LABEL_SIZE * FONT_RATIO)
        )
        colorbar_total = (
            WIDTH - PADDING_LEFT - PADDING_RIGHT - 4 * COLUMN_GAP - maximum_label_width
        )
        colorbar_width = round(colorbar_total / 2)
        colorblind_colorbar_width = round(colorbar_total / 6)
        colorbar_points = numpy.arange(0, colorbar_width, dtype=numpy.float64) / (
            colorbar_width - 1
        )
        colorblind_colorbar_points = numpy.arange(
            0, colorblind_colorbar_width, dtype=numpy.float64
        ) / (colorblind_colorbar_width - 1)
        frame = numpy.full((height, WIDTH, 3), 0x19, dtype=numpy.uint8)
        offset = PADDING_TOP
        for index, colorblindness_type in enumerate(
            typing.get_args(faery.ColorblindnessType)
        ):
            faery.image.annotate(
                frame=frame,
                text=colorblindness_type.capitalize(),
                x=PADDING_LEFT
                + maximum_label_width
                + COLUMN_GAP * 2
                + colorbar_width
                + (COLUMN_GAP + colorblind_colorbar_width) * index
                + int(
                    round(
                        colorblind_colorbar_width / 2
                        - len(colorblindness_type) / 2 * SUBTITLE_SIZE * FONT_RATIO
                    )
                ),
                y=offset + SUBTITLE_OFFSET,
                size=SUBTITLE_SIZE,
                color=(0xFF, 0xFF, 0xFF, 0xFF),
            )

        for type in sorted(
            type_to_names_and_colormaps.keys(),
            key=lambda colormap_type: (
                # show cyclic maps last
                "\U0010FFFD"
                if colormap_type == "cyclic"
                else colormap_type
            ),
        ):
            faery.image.annotate(
                frame=frame,
                text=type.capitalize(),
                x=PADDING_LEFT,
                y=offset,
                size=TITLE_SIZE,
                color=(0xFF, 0xFF, 0xFF, 0xFF),
            )
            offset += TITLE_SIZE + TITLE_PADDING_BOTTOM
            for name, colormap in type_to_names_and_colormaps[type]:
                label_width = int(math.ceil(len(name) * LABEL_SIZE * FONT_RATIO))
                faery.image.annotate(
                    frame=frame,
                    text=name,
                    x=PADDING_LEFT + maximum_label_width - label_width,
                    y=offset + LABEL_OFFSET,
                    size=LABEL_SIZE,
                    color=(0xFF, 0xFF, 0xFF, 0xFF),
                )
                colormap_points = numpy.arange(
                    0, colormap.rgba.shape[0], dtype=numpy.float64
                ) / (colormap.rgba.shape[0] - 1)
                colorbar = numpy.tile(
                    numpy.column_stack(
                        (
                            numpy.interp(
                                colorbar_points, colormap_points, colormap.rgba[:, 0]
                            )
                            * 255.0,
                            numpy.interp(
                                colorbar_points, colormap_points, colormap.rgba[:, 1]
                            )
                            * 255.0,
                            numpy.interp(
                                colorbar_points, colormap_points, colormap.rgba[:, 2]
                            )
                            * 255.0,
                        )
                    ),
                    (ROW_HEIGHT, 1, 1),
                )
                colorbar = numpy.round(colorbar).astype(numpy.uint8)
                left = PADDING_LEFT + maximum_label_width + COLUMN_GAP
                frame[
                    offset : offset + ROW_HEIGHT,
                    left : left + colorbar_width,
                ] = colorbar
                for index, colorblindness_type in enumerate(
                    typing.get_args(faery.ColorblindnessType)
                ):
                    colorblind_colormap = colormap.colorblindness_simulation(
                        colorblindness_type
                    )
                    colorblind_colormap_points = numpy.arange(
                        0, colorblind_colormap.rgba.shape[0], dtype=numpy.float64
                    ) / (colorblind_colormap.rgba.shape[0] - 1)
                    colorblind_colorbar = numpy.tile(
                        numpy.column_stack(
                            (
                                numpy.interp(
                                    colorblind_colorbar_points,
                                    colorblind_colormap_points,
                                    colorblind_colormap.rgba[:, 0],
                                )
                                * 255.0,
                                numpy.interp(
                                    colorblind_colorbar_points,
                                    colorblind_colormap_points,
                                    colorblind_colormap.rgba[:, 1],
                                )
                                * 255.0,
                                numpy.interp(
                                    colorblind_colorbar_points,
                                    colorblind_colormap_points,
                                    colorblind_colormap.rgba[:, 2],
                                )
                                * 255.0,
                            )
                        ),
                        (ROW_HEIGHT, 1, 1),
                    )
                    colorblind_colorbar = numpy.round(colorblind_colorbar).astype(
                        numpy.uint8
                    )
                    left = (
                        PADDING_LEFT
                        + maximum_label_width
                        + COLUMN_GAP * 2
                        + colorbar_width
                        + (COLUMN_GAP + colorblind_colorbar_width) * index
                    )
                    frame[
                        offset : offset + ROW_HEIGHT,
                        left : left + colorblind_colorbar_width,
                    ] = colorblind_colorbar

                offset += ROW_HEIGHT + ROW_GAP
            offset += ROWS_PADDING - ROW_GAP

        with open(args.output, "wb") as output:
            output.write(faery.image.encode(frame=frame, compression_level="default"))
