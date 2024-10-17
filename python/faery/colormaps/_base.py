import dataclasses
import re
import typing

import numpy
import numpy.typing

COLOR_PATTERN = re.compile(
    r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?$"
)


def parse_color(
    color: typing.Union[
        tuple[float, float, float], tuple[float, float, float, float], str
    ]
) -> tuple[float, float, float, float]:
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
    data: numpy.typing.NDArray[numpy.float64]

    @classmethod
    def from_rgb_table(
        cls,
        type: typing.Literal["sequential", "diverging"],
        data: list[tuple[float, float, float]],
    ):
        return cls(
            type=type,
            data=numpy.c_[
                numpy.array(data, dtype=numpy.float64), numpy.ones(len(data))
            ],
        )

    @classmethod
    def from_rgba_table(
        cls,
        type: typing.Literal["sequential", "diverging"],
        data: list[tuple[float, float, float, float]],
    ):
        return cls(type=type, data=numpy.array(data, dtype=numpy.float64))

    @classmethod
    def sequential_from_pair(
        cls,
        start: typing.Union[
            tuple[float, float, float], tuple[float, float, float, float], str
        ],
        end: typing.Union[
            tuple[float, float, float], tuple[float, float, float, float], str
        ],
        samples: int,
    ):
        start = parse_color(color=start)
        end = parse_color(color=end)
        return cls(
            type="sequential",
            data=gradient(
                start=start,
                end=end,
                samples=samples,
            ),
        )

    @classmethod
    def diverging_from_triplet(
        cls,
        start: typing.Union[
            tuple[float, float, float], tuple[float, float, float, float], str
        ],
        middle: typing.Union[
            tuple[float, float, float], tuple[float, float, float, float], str
        ],
        end: typing.Union[
            tuple[float, float, float], tuple[float, float, float, float], str
        ],
        half_samples: int,
    ):
        start = parse_color(color=start)
        middle = parse_color(color=middle)
        end = parse_color(color=end)
        return cls(
            type="diverging",
            data=numpy.vstack(
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
