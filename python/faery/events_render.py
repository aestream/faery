import collections.abc
import types
import typing

import numpy

from . import color as color_module
from . import enums, events_stream, frame_stream, stream, timestamp

if typing.TYPE_CHECKING:
    from .types import render  # type: ignore
else:
    from .extension import render


def restrict(prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]]):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_render(
    prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]],
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


@typed_render({"", "Finite", "Regular", "FiniteRegular"})
class Render(frame_stream.FiniteRegularFrameStream):
    """
    Converts events to color frames.

    Args:
        parent: Input event stream.
        decay: Function that converts timestamps to bounded values.
            Exponential, linear, and window decays directly generate values in `[0, 1]`, compatible with colormaps.
            Cumulative generates unbounded values that must be tone-mapped to `[0, 1]` before applying the colormap.
            See `minimum_clip`, `maximum_clip`, and `gamma` for details on how to control the tone-mapping process.
        tau: Decay temporal parameter.
            The specific role of this parameter depends on the decay function.
        minimum_clip: Only used if the decay is "cumulative". Must be smaller than `maximum_clip` and in the range `[0, 1]`.
            Selects the tone-map minimum by computing `values[minimum_clip * len(values)]`,
            where `values` is a sorted array of cumulative decays.
            Values only contains non-zero decays.
            Decays below `minimum_clip` are set to zero before applying the colormap.
        maximum_clip: Only used if the decay is "cumulative". Must be larger than `minimum_clip` and in the range `[0, 1]`.
            Identical to `minimum_clip`, but values above `maximum_clip` are set to 1 before applying the colormap.
        gamma: Only used if the decay is "cumulative", can be positive or negative. Controls the gamma ramp shape.
            0 is linear, positive values increase the available range for low values and flattens the range for high values,
            negative does the opposite.
    """

    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        decay: enums.Decay,
        tau: timestamp.TimeOrTimecode,
        colormap: color_module.Colormap,
        minimum_clip: float = 0.0,
        maximum_clip: float = 0.99,
        gamma: float = 0.0,
    ):
        self.parent = parent
        self.decay: enums.Decay = enums.validate_decay(decay)
        self.tau = timestamp.parse_time(tau)
        self.minimum_clip = minimum_clip
        self.maximum_clip = maximum_clip
        self.gamma = gamma
        self.colormap = colormap

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    @restrict({"Finite", "FiniteRegular"})
    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return self.parent.time_range()

    @restrict({"Regular", "FiniteRegular"})
    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
        try:
            period_us = 1e6 / self.parent.frequency_hz()
            try:
                time_range = self.parent.time_range()
                frame_index = 0
                first_frame_t = time_range[0]
            except (AttributeError, NotImplementedError):
                frame_index = 0
                first_frame_t = None
        except (AttributeError, NotImplementedError):
            period_us = None
            frame_index = None
            first_frame_t = None
        with render.Renderer(
            dimensions=self.dimensions(),
            decay=self.decay,
            tau=self.tau.to_microseconds(),
            minimum_clip=self.minimum_clip,
            maximum_clip=self.maximum_clip,
            gamma=self.gamma,
            colormap_type=self.colormap.type,
            colormap_rgba=self.colormap.rgba,
        ) as renderer:
            for events in self.parent:
                if len(events) == 0:
                    if period_us is None:
                        continue
                    assert frame_index is not None
                    frame_index += 1
                    if first_frame_t is None:
                        continue
                    frame_t = timestamp.Time(
                        microseconds=int(
                            round(
                                first_frame_t.to_microseconds()
                                + frame_index * period_us
                            )
                        )
                    )
                else:
                    if period_us is None:
                        frame_t = timestamp.Time(int(events["t"][-1]) + 1)
                    else:
                        assert frame_index is not None
                        if first_frame_t is None:
                            first_frame_t = timestamp.Time(
                                microseconds=int(
                                    round(
                                        int(events["t"][0])
                                        + ((1 - frame_index) * period_us)
                                    )
                                )
                            )
                            if frame_index > 0:
                                for empty_frame_index in range(0, frame_index):
                                    empty_frame_t = timestamp.Time(
                                        microseconds=int(
                                            round(
                                                first_frame_t.to_microseconds()
                                                + empty_frame_index * period_us
                                            )
                                        )
                                    )
                                    assert empty_frame_t >= timestamp.Time(
                                        microseconds=0
                                    )
                                    yield frame_stream.Frame(
                                        t=empty_frame_t,
                                        pixels=renderer.render(
                                            numpy.array(
                                                [],
                                                dtype=events_stream.EVENTS_DTYPE,
                                            ),
                                            render_t=empty_frame_t.to_microseconds(),
                                        ),
                                    )
                        frame_index += 1
                        frame_t = timestamp.Time(
                            microseconds=int(
                                round(
                                    first_frame_t.to_microseconds()
                                    + frame_index * period_us
                                )
                            )
                        )
                pixels = renderer.render(
                    events=events,
                    render_t=frame_t.to_microseconds(),
                )
                yield frame_stream.Frame(t=frame_t, pixels=pixels)
