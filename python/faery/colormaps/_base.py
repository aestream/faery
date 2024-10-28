import dataclasses
import math
import pathlib
import re
import typing

import numpy
import numpy.typing

from .. import enums

if typing.TYPE_CHECKING:
    from ..types import image  # type: ignore
else:
    from ..extension import image

COLOR_PATTERN = re.compile(
    r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?$"
)

TITLE_SIZE: int = 20
TITLE_PADDING_BOTTOM: int = 16
LABEL_SIZE: int = 16
LABEL_OFFSET: int = -1
ROW_HEIGHT: int = 20
ROW_GAP: int = 6
PADDING_TOP: int = 20
PADDING_BOTTOM: int = 20
PADDING_LEFT: int = 20
PADDING_RIGHT: int = 20
FONT_RATIO: float = 0.6
COLUMN_GAP: int = 20
WIDTH: int = 720

Color = typing.Union[tuple[float, float, float], tuple[float, float, float, float], str]


def parse_color(color: Color) -> tuple[float, float, float, float]:
    if isinstance(color, str):
        match = COLOR_PATTERN.match(color)
        if match is None:
            raise Exception(
                f'parsing the color "{color}" failed (expected "#RRGGBB" or "#RRGGBBAA" where R, G, B, and A are hexadecimal digits)'
            )
        return (
            int(f"0x{match[1]}", 16) / 255.0,
            int(f"0x{match[2]}", 16) / 255.0,
            int(f"0x{match[3]}", 16) / 255.0,
            1.0 if match[4] is None else int(f"0x{match[4]}", 16) / 255.0,
        )
    assert color[0] >= 0.0 and color[0] <= 1.0
    assert color[1] >= 0.0 and color[1] <= 1.0
    assert color[2] >= 0.0 and color[2] <= 1.0
    if len(color) == 3:
        return (color[0], color[1], color[2], 1.0)
    assert len(color) == 4
    assert color[3] >= 0.0 and color[3] <= 1.0
    return color


def rgb_to_lab(
    rgb: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    mask = rgb > 0.04045
    rgb[mask] = numpy.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[numpy.logical_not(mask)] /= 12.92
    xyz = rgb @ numpy.array(
        [
            [0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227],
        ]
    )
    xyz /= numpy.array([0.95047, 1.0, 1.08883])
    mask = xyz > 0.008856
    xyz[mask] = numpy.cbrt(xyz[mask])
    xyz[~mask] = 7.787 * xyz[numpy.logical_not(mask)] + 16.0 / 116.0
    lab = numpy.zeros(xyz.shape, dtype=numpy.float64)
    lab[:, 0] = (116.0 * xyz[:, 1]) - 16.0
    lab[:, 1] = 500.0 * (xyz[:, 0] - xyz[:, 1])
    lab[:, 2] = 200.0 * (xyz[:, 1] - xyz[:, 2])
    return lab


def lab_to_rgb(
    lab: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    xyz = numpy.zeros(lab.shape, dtype=numpy.float64)
    xyz[:, 1] = (lab[:, 0] + 16.0) / 116.0
    xyz[:, 0] = (lab[:, 1] / 500.0) + xyz[:, 1]
    xyz[:, 2] = xyz[:, 1] - (lab[:, 2] / 200.0)
    mask = xyz > 0.2068966
    xyz[mask] = numpy.power(xyz[mask], 3.0)
    xyz[numpy.logical_not(mask)] = (xyz[numpy.logical_not(mask)] - 16.0 / 116.0) / 7.787
    xyz *= numpy.array([0.95047, 1.0, 1.08883])
    rgb = xyz @ numpy.array(
        [
            [3.24048134, -0.96925495, 0.05564664],
            [-1.53715152, 1.87599, -0.20404134],
            [-0.49853633, 0.04155593, 1.05731107],
        ]
    )
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * numpy.power(rgb[mask], 1 / 2.4) - 0.055
    rgb[numpy.logical_not(mask)] *= 12.92
    numpy.clip(rgb, 0.0, 1.1, out=rgb)
    return rgb


def rgb_to_lrgb(
    rgb: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    mask = rgb < 0.04045
    rgb[mask] /= 12.92
    rgb[numpy.logical_not(mask)] = numpy.power(
        (rgb[numpy.logical_not(mask)] + 0.055) / 1.055, 2.4
    )
    return numpy.clip(rgb, 0, 1.0)


def lrgb_to_rgb(
    lrgb: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    mask = lrgb < 0.0031308
    lrgb[mask] *= 12.92
    lrgb[numpy.logical_not(mask)] = (
        numpy.power(lrgb[numpy.logical_not(mask)], 1.0 / 2.4) * 1.055 - 0.055
    )
    return numpy.clip(lrgb, 0, 1.0)


BRETTEL_PROTANOPIA_NORMAL: numpy.typing.NDArray[numpy.float64] = numpy.array(
    [0.00048, 0.00393, -0.00441], dtype=numpy.float64
)
BRETTEL_PROTANOPIA: tuple[
    numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]
] = (
    numpy.array(
        [
            [0.1498, 0.10764, 0.00384],
            [1.19548, 0.84864, -0.0054],
            [-0.34528, 0.04372, 1.00156],
        ],
        dtype=numpy.float64,
    ),
    numpy.array(
        [
            [0.1457, 0.10816, 0.00386],
            [1.16172, 0.85291, -0.00524],
            [-0.30742, 0.03892, 1.00139],
        ],
        dtype=numpy.float64,
    ),
)

BRETTEL_DEUTERANOPIA_NORMAL: numpy.typing.NDArray[numpy.float64] = numpy.array(
    [-0.00281, -0.00611, 0.00892], dtype=numpy.float64
)
BRETTEL_DEUTERANOPIA: tuple[
    numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]
] = (
    numpy.array(
        [
            [0.36477, 0.26294, -0.02006],
            [0.86381, 0.64245, 0.02728],
            [-0.22858, 0.09462, 0.99278],
        ],
        dtype=numpy.float64,
    ),
    numpy.array(
        [
            [0.37298, 0.25954, -0.0198],
            [0.88166, 0.63506, 0.02784],
            [-0.25464, 0.1054, 0.99196],
        ],
        dtype=numpy.float64,
    ),
)

BRETTEL_TRITANOPIA_NORMAL: numpy.typing.NDArray[numpy.float64] = numpy.array(
    [0.03901, -0.02788, -0.01113], dtype=numpy.float64
)
BRETTEL_TRITANOPIA: tuple[
    numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]
] = (
    numpy.array(
        [
            [1.01277, -0.01243, 0.07589],
            [0.13548, 0.86812, 0.805],
            [-0.14826, 0.14431, 0.11911],
        ],
        dtype=numpy.float64,
    ),
    numpy.array(
        [
            [0.93678, 0.06154, -0.37562],
            [0.18979, 0.81526, 1.12767],
            [-0.12657, 0.1232, 0.24796],
        ],
        dtype=numpy.float64,
    ),
)


def gradient(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    samples: int,
) -> numpy.typing.NDArray[numpy.float64]:
    lab = rgb_to_lab(
        numpy.array(
            [start[0:3], end[0:3]],
            dtype=numpy.float64,
        )
    )
    parameter = numpy.linspace(0.0, 1.0, num=samples, endpoint=True)
    return numpy.c_[
        lab_to_rgb(
            numpy.outer((1.0 - parameter), lab[0]) + numpy.outer(parameter, lab[1])
        ),
        (1.0 - parameter) * start[3] + parameter * end[3],
    ]


@dataclasses.dataclass
class Colormap:
    type: typing.Literal["sequential", "diverging"]
    rgba: numpy.typing.NDArray[numpy.float64]

    @classmethod
    def from_rgb_table(
        cls,
        type: typing.Literal["sequential", "diverging"],
        rgb: list[tuple[float, float, float]],
    ):
        return cls(
            type=type,
            rgba=numpy.c_[numpy.array(rgb, dtype=numpy.float64), numpy.ones(len(rgb))],
        )

    @classmethod
    def from_rgba_table(
        cls,
        type: typing.Literal["sequential", "diverging"],
        rgba: list[tuple[float, float, float, float]],
    ):
        return cls(type=type, rgba=numpy.array(rgba, dtype=numpy.float64))

    @classmethod
    def sequential_from_pair(
        cls,
        start: Color,
        end: Color,
        samples: int = 256,
    ):
        start = parse_color(color=start)
        end = parse_color(color=end)
        return cls(
            type="sequential",
            rgba=gradient(
                start=start,
                end=end,
                samples=samples,
            ),
        )

    @classmethod
    def diverging_from_triplet(
        cls,
        start: Color,
        middle: Color,
        end: Color,
        half_samples: int = 128,
    ):
        start = parse_color(color=start)
        middle = parse_color(color=middle)
        end = parse_color(color=end)
        return cls(
            type="diverging",
            rgba=numpy.vstack(
                (
                    gradient(
                        start=start,
                        end=middle,
                        samples=half_samples,
                    ),
                    gradient(
                        start=middle,
                        end=end,
                        samples=half_samples,
                    )[1:],
                ),
            ),
        )

    def flipped(self) -> "Colormap":
        return Colormap(
            type=self.type,
            rgba=numpy.flip(self.rgba, axis=0).copy(),
        )

    def colorblindness_simulation(
        self,
        type: enums.ColorblindnessType,
    ) -> "Colormap":
        type = enums.validate_colorblindness_type(type)
        # https://doi.org/10.1002/(SICI)1520-6378(199908)24:4<243::AID-COL5>3.0.CO;2-3
        # https://vision.psychol.cam.ac.uk/jdmollon/papers/colourmaps.pdf
        rgb = self.rgba[:, 0:3].copy()
        lrgb = rgb_to_lrgb(rgb)
        if type == "protanopia":
            normal = BRETTEL_PROTANOPIA_NORMAL
            matrices = BRETTEL_PROTANOPIA
        elif type == "deuteranopia":
            normal = BRETTEL_DEUTERANOPIA_NORMAL
            matrices = BRETTEL_DEUTERANOPIA
        elif type == "tritanopia":
            normal = BRETTEL_TRITANOPIA_NORMAL
            matrices = BRETTEL_TRITANOPIA
        else:
            raise Exception(f"{type} not implemented")
        mask = (lrgb @ normal) < 0.0
        below = lrgb @ matrices[1]
        above = lrgb @ matrices[0]
        lrgb[mask] = below[mask]
        lrgb[numpy.logical_not(mask)] = above[numpy.logical_not(mask)]
        rgb = lrgb_to_rgb(lrgb)
        return Colormap(
            type=self.type,
            rgba=numpy.c_[numpy.array(rgb, dtype=numpy.float64), self.rgba[:, 3]],
        )

    def to_file(
        self,
        path: typing.Union[pathlib.Path, str],
        name: str,
    ):
        frame = numpy.full(
            (
                TITLE_SIZE
                + TITLE_PADDING_BOTTOM
                + PADDING_TOP
                + PADDING_BOTTOM
                + 4 * ROW_HEIGHT
                + 3 * ROW_GAP,
                WIDTH,
                3,
            ),
            0x19,
            dtype=numpy.uint8,
        )
        labels = ("",) + typing.get_args(enums.ColorblindnessType)
        maximum_label_width = int(
            math.ceil(max(len(label) for label in labels) * LABEL_SIZE * FONT_RATIO)
        )
        colorbar_width = (
            WIDTH - PADDING_LEFT - PADDING_RIGHT - COLUMN_GAP - maximum_label_width
        )
        colorbar_points = numpy.arange(0, colorbar_width, dtype=numpy.float64) / (
            colorbar_width - 1
        )
        offset = PADDING_TOP
        image.annotate(
            frame=frame,
            text=name,
            x=PADDING_LEFT,
            y=offset,
            size=TITLE_SIZE,
            color=(0xFF, 0xFF, 0xFF, 0xFF),
        )
        offset += TITLE_SIZE + TITLE_PADDING_BOTTOM
        for label in labels:
            if len(label) > 0:
                colormap = self.colorblindness_simulation(label)  # type: ignore
            else:
                colormap = self
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
            frame[
                offset : offset + ROW_HEIGHT,
                PADDING_LEFT : PADDING_LEFT + colorbar_width,
            ] = colorbar
            if len(label) > 0:
                image.annotate(
                    frame=frame,
                    text=label.capitalize(),
                    x=PADDING_LEFT + colorbar_width + COLUMN_GAP,
                    y=offset + LABEL_OFFSET,
                    size=LABEL_SIZE,
                    color=(0xFF, 0xFF, 0xFF, 0xFF),
                )
            offset += ROW_HEIGHT + ROW_GAP
        with open(path, "wb") as output:
            output.write(image.encode(frame, compression_level="default"))
