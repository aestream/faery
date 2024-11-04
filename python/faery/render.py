import collections.abc
import types
import typing

import numpy

from . import enums, frame_stream, stream, timestamp
from .colormaps._base import Colormap


def restrict(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_render(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(render_class):
        attributes = [
            name
            for name, item in render_class.__dict__.items()
            if isinstance(item, types.FunctionType)
        ]
        for prefix in prefixes:

            class Generated(getattr(frame_stream, f"{prefix}FrameStream")):
                pass

            for attribute in attributes:
                method = getattr(render_class, attribute)
                if not hasattr(method, "_prefixes") or prefix in method._prefixes:
                    setattr(Generated, attribute, getattr(render_class, attribute))
            Generated.__name__ = f"{prefix}{render_class.__name__}"
            Generated.__qualname__ = Generated.__name__
            FILTERS[Generated.__name__] = Generated
        return None

    return decorator


@typed_render({"Float64", "FiniteFloat64", "RegularFloat64", "FiniteRegularFloat64"})
class Envelope(frame_stream.FiniteRegularFloat64FrameStream):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        decay: enums.Decay,
        tau: timestamp.Time,
    ):
        self.parent = parent
        self.decay = enums.validate_decay(decay)
        self.tau = timestamp.parse_timestamp(tau)

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    @restrict({"FiniteFloat64", "FiniteRegularFloat64"})
    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()

    @restrict({"RegularFloat64", "FiniteRegularFloat64"})
    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Float64Frame]:
        if self.decay == "exponential":
            upsilon = -1.0 / self.tau
        elif self.decay == "linear":
            # multiply tau by 2 to get the same overall "illuminance"
            # per event as the other decays
            # we approximate the illuminance as the integral of the decay
            # from 0 (event time) to infinity
            upsilon = -1.0 / (2.0 * self.tau)
        elif self.decay == "window":
            upsilon = self.tau
        else:
            raise Exception(
                f'unknown decay "{self.decay}" (expected "exponential", "linear", or "window")'
            )
        uint64_max = numpy.iinfo(numpy.uint64).max
        dimensions = self.dimensions()
        ts = numpy.full(
            (dimensions[1], dimensions[0]),
            uint64_max,
            dtype=numpy.uint64,
        )
        offs = numpy.full((dimensions[1], dimensions[0]), True, dtype=numpy.bool)
        try:
            period_us = 1e6 / self.parent.frequency_hz()
            try:
                time_range_us = self.parent.time_range_us()
                frame_index = 0
                first_frame_t = time_range_us[0]
            except AttributeError:
                frame_index = 0
                first_frame_t = None
        except AttributeError:
            period_us = None
            frame_index = None
            first_frame_t = None
        for events in self.parent:
            if len(events) == 0:
                if period_us is None:
                    continue
                assert frame_index is not None
                frame_index += 1
                if first_frame_t is None:
                    continue
                frame_t = int(round(first_frame_t + frame_index * period_us))
            else:
                ts[events["y"], events["x"]] = events["t"]
                offs[events["y"], events["x"]] = numpy.logical_not(events["on"])
                if period_us is None:
                    frame_t = int(events["t"][-1]) + 1
                else:
                    assert frame_index is not None
                    if first_frame_t is None:
                        first_frame_t = int(
                            round(int(events["t"][0]) + ((1 - frame_index) * period_us))
                        )
                        if frame_index > 0:
                            for empty_frame_index in range(0, frame_index):
                                empty_frame_t = int(
                                    round(first_frame_t + empty_frame_index * period_us)
                                )
                                assert empty_frame_t >= 0
                                yield frame_stream.Float64Frame(
                                    t=empty_frame_t,
                                    pixels=numpy.zeros(ts.shape, dtype=numpy.float64),
                                )
                    frame_index += 1
                    frame_t = int(round(first_frame_t + frame_index * period_us))

            mask = ts == uint64_max
            if self.decay == "exponential":
                pixels = (frame_t - 1 - ts).astype(numpy.float64)
                numpy.multiply(pixels, upsilon, out=pixels)
                numpy.exp(pixels, out=pixels)
            elif self.decay == "linear":
                pixels = (frame_t - 1 - ts).astype(numpy.float64)
                numpy.multiply(pixels, upsilon, out=pixels)
                numpy.add(pixels, 1.0, out=pixels)
                pixels[pixels < 0.0] = 0.0
            elif self.decay == "window":
                pixels = (frame_t - 1 - ts < upsilon).astype(numpy.float64)
            else:
                raise Exception(
                    f'unknown decay "{self.decay}" (expected "exponential", "linear", or "window")'
                )
            pixels[mask] = 0.0
            pixels[offs] *= -1.0
            yield frame_stream.Float64Frame(t=frame_t, pixels=pixels)


@typed_render(
    {"Rgba8888", "FiniteRgba8888", "RegularRgba8888", "FiniteRegularRgba8888"}
)
class Colorize(frame_stream.FiniteRegularRgba8888FrameStream):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[frame_stream.Float64Frame],
        colormap: Colormap,
    ):
        self.parent = parent
        self.colormap = colormap

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    @restrict({"FiniteRgba8888", "FiniteRegularRgba8888"})
    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()

    @restrict({"RegularRgba8888", "FiniteRegularRgba8888"})
    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Rgba8888Frame]:
        colormap_data = numpy.round(self.colormap.rgba * 255.0).astype(
            dtype=numpy.uint8
        )
        if self.colormap.type == "diverging":
            if colormap_data.shape[0] % 2 == 0:
                upsilon_off = colormap_data.shape[0] / 2.0
                upsilon_on = colormap_data.shape[0] / 2.0 - 1
                phi = colormap_data.shape[0] / 2.0
                for frame in self.parent:
                    mask = frame.pixels < 0.0
                    frame.pixels[mask] *= upsilon_off
                    numpy.logical_not(mask, out=mask)
                    frame.pixels[mask] *= upsilon_on
                    numpy.add(frame.pixels, phi, out=frame.pixels)
                    numpy.round(frame.pixels, out=frame.pixels)
                    yield frame_stream.Rgba8888Frame(
                        t=frame.t,
                        pixels=colormap_data[frame.pixels.astype(numpy.uint32)],
                    )
            else:
                upsilon = (colormap_data.shape[0] - 1) / 2.0
                for frame in self.parent:
                    numpy.add(frame.pixels, 1.0, out=frame.pixels)
                    numpy.multiply(frame.pixels, upsilon, out=frame.pixels)
                    numpy.round(frame.pixels, out=frame.pixels)
                    yield frame_stream.Rgba8888Frame(
                        t=frame.t,
                        pixels=colormap_data[frame.pixels.astype(numpy.uint32)],
                    )
        elif self.colormap.type == "sequential":
            upsilon = colormap_data.shape[0] - 1
            for frame in self.parent:
                numpy.absolute(frame.pixels, out=frame.pixels)
                numpy.multiply(frame.pixels, upsilon, out=frame.pixels)
                numpy.round(frame.pixels, out=frame.pixels)
                yield frame_stream.Rgba8888Frame(
                    t=frame.t,
                    pixels=colormap_data[frame.pixels.astype(numpy.uint32)],
                )
        else:
            raise Exception(
                f'unknown colormap type "{self.colormap.type}" (expected "diverging" or "sequential")'
            )
