import collections.abc
import types
import typing

import numpy

from . import events_stream
from . import stream
from . import timestamp

EVENTS_DTYPE: numpy.dtype = numpy.dtype(
    [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), (("p", "on"), "?")]
)

# Empty arrays
# ------------
#
# A stream must yield `numpy.ndarray` objects with dtype `EVENTS_DTYPE`.
# Whilst yielding `None` is not allowed, a stream may yield empty event arrays.
#
# Filters that need to access event data (for instance `events["t"][0]`)
# must check that the array is not empty first (`len(events) > 0`).
#
# Non-uniform filters *may* choose to optimize dispatching by only
# yielding non-empty arrays.
#
# However, uniform filters *must* yield every event packet to preserve uniformity.
# This applies to any class inheriting from `events_stream.FiniteUniformEventsFilter`.


def restrict(prefixes: set[typing.Literal["", "Finite", "Uniform", "FiniteUniform"]]):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_filter(
    prefixes: set[typing.Literal["", "Finite", "Uniform", "FiniteUniform"]]
):
    def decorator(filter_class):
        attributes = [
            name
            for name, item in filter_class.__dict__.items()
            if isinstance(item, types.FunctionType)
        ]
        for prefix in prefixes:

            class Generated(getattr(events_stream, f"{prefix}EventsFilter")):
                pass

            for attribute in attributes:
                method = getattr(filter_class, attribute)
                if not hasattr(method, "_prefixes") or prefix in method._prefixes:
                    setattr(Generated, attribute, getattr(filter_class, attribute))
            Generated.__name__ = f"{prefix}{filter_class.__name__}"
            Generated.__qualname__ = Generated.__name__
            FILTERS[Generated.__name__] = Generated
        return None

    return decorator


@typed_filter({"Uniform", "FiniteUniform"})
class Uniformize(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        period: timestamp.Time,
        start: typing.Optional[timestamp.Time] = None,
    ):
        self.init(parent=parent)
        self._period_us = timestamp.parse_timestamp(period)
        self.start = None if start is None else timestamp.parse_timestamp(start)

    def period_us(self) -> int:
        return self._period_us

    @restrict({"FiniteUniform"})
    def time_range_us(self) -> tuple[int, int]:
        parent_time_range_us = self.parent.time_range_us()
        if self.start is None:
            start = parent_time_range_us[0]
        else:
            start = self.start
        period_us = self.period_us()
        end_index = (
            max(start + 1, parent_time_range_us[1]) - start + period_us - 1
        ) // period_us
        return (start, start + end_index * period_us)

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        try:
            parent_time_range_us = self.parent.time_range_us()
            packet_start = parent_time_range_us[0] if self.start is None else self.start
            packet_end = parent_time_range_us[1]
        except AttributeError:
            packet_start = None
            packet_end = None
        events_buffers: list[numpy.ndarray] = []
        for events in self.parent:
            while len(events) > 0:
                if packet_start is None:
                    packet_start = events["t"][0]
                if events["t"][-1] < packet_start:
                    break
                next_packet_start = packet_start + self.period_us()
                if events["t"][0] >= next_packet_start:
                    yield numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
                    events_buffers = []
                    packet_start = next_packet_start
                    continue
                if events["t"][-1] < next_packet_start:
                    events_buffers.append(events)
                    break
                pivot = numpy.searchsorted(events["t"], next_packet_start)
                if len(events_buffers) == 0:
                    yield events[:pivot]
                else:
                    events_buffers.append(events[:pivot])
                    yield numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
                    events_buffers = []
                events = events[pivot:]
                packet_start = next_packet_start
        if len(events_buffers) > 0:
            yield numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
            events_buffers = []
        if packet_start is not None and packet_end is not None:
            while packet_start < packet_end:
                yield numpy.array([], dtype=EVENTS_DTYPE)
                packet_start += self.period_us()


@typed_filter({"", "Finite"})
class Chunks(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        chunk_length: int,
    ):
        self.init(parent=parent)
        assert chunk_length > 0
        self.chunk_length = chunk_length

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        events_buffers: list[numpy.ndarray] = []
        current_length = 0
        for events in self.parent:
            events_length = len(events)
            while events_length > 0:
                if current_length + events_length < self.chunk_length:
                    events_buffers.append(events)
                    current_length += events_length
                    break
                pivot = self.chunk_length - current_length
                if len(events_buffers) == 0:
                    yield events[:pivot]
                else:
                    events_buffers.append(events[:pivot])
                    yield numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
                    events_buffers = []
                events = events[pivot:]
        if len(events_buffers) > 0:
            yield numpy.concatenate(events_buffers, dtype=EVENTS_DTYPE)
            events_buffers = []


"""
class OffsetT(
    events_stream.FiniteUniformEventsStream,
    events_stream.FiniteUniformEventsFilter,
):
    pass  # @TODO
"""


@typed_filter({"Finite"})
class TimeSlice(events_stream.FiniteEventsFilter):  # type: ignore
    def __init__(
        self,
        parent: stream.FiniteStream[numpy.ndarray],
        start: timestamp.Time,
        end: timestamp.Time,
        zero: bool,
    ):
        self.init(parent=parent)
        self.start = timestamp.parse_timestamp(start)
        self.end = timestamp.parse_timestamp(end)
        assert self.start < self.end, f"{start=} must be strictly smaller than {end=}"
        self.zero = zero

    @restrict({"Finite"})
    def time_range_us(self) -> tuple[int, int]:
        parent_time_range_us = self.parent.time_range_us()
        if self.zero:
            return (
                max(self.start, parent_time_range_us[0]) - self.start,
                min(self.end, parent_time_range_us[1]) - self.start,
            )
        else:
            return (
                max(self.start, parent_time_range_us[0]),
                min(self.end, parent_time_range_us[1]),
            )

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for events in self.parent:
            if len(events) > 0:
                if events["t"][-1] < self.start:
                    continue
                if events["t"][0] >= self.end:
                    return
                events = events[
                    numpy.logical_and(events["t"] >= self.start, events["t"] < self.end)
                ]
                if len(events) > 0:
                    if self.zero:
                        events["t"] -= self.start
                    yield events


@typed_filter({"FiniteUniform"})
class TimeSlice(events_stream.FiniteUniformEventsFilter):  # type: ignore
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        start: timestamp.Time,
        end: timestamp.Time,
        zero: bool,
    ):
        self.init(parent=parent)
        self.start = timestamp.parse_timestamp(start)
        self.end = timestamp.parse_timestamp(end)
        assert self.start < self.end, f"{start=} must be strictly smaller than {end=}"
        self.zero = zero

    @restrict({"FiniteUniform"})
    def time_range_us(self) -> tuple[int, int]:
        period_us = self.period_us()
        assert (
            self.end - self.start
        ) % period_us, f"start={self.start} and={self.end} must be separated by an integer number of periods ({period_us})"
        parent_time_range_us = self.parent.time_range_us()
        start_index = (
            max(self.start, parent_time_range_us[0])
            - parent_time_range_us[0]
            + period_us
            - 1
        ) // period_us
        end_index = (
            max(min(self.end, parent_time_range_us[1]), parent_time_range_us[0])
            - parent_time_range_us[0]
        ) // period_us
        if self.zero:
            return (
                start_index * period_us,
                end_index * period_us,
            )
        else:
            return (
                parent_time_range_us[0] + start_index * period_us,
                parent_time_range_us[0] + end_index * period_us,
            )

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        period_us = self.period_us()
        assert (
            self.end - self.start
        ) % period_us, f"start={self.start} and={self.end} must be separated by an integer number of periods ({period_us})"
        parent_time_range_us = self.parent.time_range_us()
        start_index = (
            max(self.start, parent_time_range_us[0])
            - parent_time_range_us[0]
            + period_us
            - 1
        ) // period_us
        end_index = (
            max(min(self.end, parent_time_range_us[1]), parent_time_range_us[0])
            - parent_time_range_us[0]
            + period_us
            - 1
        ) // period_us
        offset = start_index * period_us
        for index, events in enumerate(self.parent):
            if index < start_index:
                continue
            if index >= end_index:
                break
            if self.zero and len(events) > 0:
                events["t"] -= offset
            yield events


@typed_filter({"Finite"})
class EventSlice(events_stream.FiniteEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteStream[numpy.ndarray],
        start: int,
        end: int,
    ):
        assert start < end, f"{start=} must be strictly smaller than {end=}"
        assert start >= 0
        self.init(parent=parent)
        self.start = start
        self.end = end

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        index = 0
        for events in self.parent:
            length = len(events)
            if length > 0:
                if index + length <= self.start:
                    index += length
                    continue
                if index >= self.end:
                    return
                events = events[
                    max(self.start - index, 0) : min(self.end - index, length)
                ]
                length = len(events)
                if length > 0:
                    yield events
                    index += length


@typed_filter({"Finite"})
class UniformEventSlice(events_stream.UniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        start: int,
        end: int,
    ):
        assert start < end, f"{start=} must be strictly smaller than {end=}"
        assert start >= 0
        self.init(parent=parent)
        self.start = start
        self.end = end

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        index = 0
        for events in self.parent:
            length = len(events)
            if index >= self.start and index + length < self.end:
                yield events
            else:
                yield numpy.ndarray([], dtype=EVENTS_DTYPE)
            index += 1


@typed_filter({"", "Finite", "Uniform", "FiniteUniform"})
class Crop(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        left: int,
        right: int,
        top: int,
        bottom: int,
    ):
        self.init(parent=parent)
        dimensions = parent.dimensions()
        assert left < right, f"{left=} must be strictly smaller than {right=}"
        assert left >= 0
        assert right <= dimensions[0]
        assert top < bottom, f"{top=} must be strictly smaller than {bottom=}"
        assert top >= 0
        assert bottom <= dimensions[1]
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def dimensions(self) -> tuple[int, int]:
        return (self.right - self.left, self.bottom - self.top)

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for events in self.parent:
            events = events[
                numpy.logical_and.reduce(
                    (
                        events["x"] >= self.left,
                        events["x"] < self.right,
                        events["y"] >= self.top,
                        events["y"] < self.bottom,
                    )
                )
            ]
            events["x"] -= self.left
            events["y"] -= self.top
            yield events


@typed_filter({"", "Finite", "Uniform", "FiniteUniform"})
class Mask(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        array: numpy.ndarray,
    ):
        self.init(parent=parent)
        assert array.dtype == numpy.dtype("?")
        assert len(array.shape) == 2
        dimensions = self.dimensions()
        if array.shape[0] != dimensions[1] or array.shape[1] != dimensions[0]:
            raise Exception(
                "array must be {}x{} (got {}x{})",
                dimensions[1],
                dimensions[0],
                array.shape[0],
                array.shape[1],
            )
        self.array = array

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for events in self.parent:
            yield events[self.array[events["y"], events["x"]]]


@typed_filter({"", "Finite", "Uniform", "FiniteUniform"})
class Transpose(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteUniformStream[numpy.ndarray],
        action: typing.Literal[
            "flip_left_right",
            "flip_bottom_top",
            "rotate_90_counterclockwise",
            "rotate_180",
            "rotate_270_counterclockwise",
            "flip_up_diagonal",
            "flip_down_diagonal",
        ],
    ):
        self.init(parent=parent)
        self.action = action

    def dimensions(self) -> tuple[int, int]:
        dimensions = self.parent.dimensions()
        if self.action in ("flip_left_right", "flip_bottom_top", "rotate_180"):
            return dimensions
        if self.action in (
            "rotate_90_counterclockwise",
            "rotate_270_counterclockwise",
            "flip_up_diagonal",
            "flip_down_diagonal",
        ):
            return (dimensions[1], dimensions[0])
        raise Exception(f'unknown action "{self.action}"')

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        dimensions = self.parent.dimensions()
        for events in self.parent:
            if self.action == "flip_left_right":
                events["x"] = dimensions[0] - 1 - events["x"]
            elif self.action == "flip_bottom_top":
                events["y"] = dimensions[1] - 1 - events["y"]
            elif self.action == "rotate_90_counterclockwise":
                x = events["x"].copy()
                events["x"] = dimensions[0] - 1 - events["y"]
                events["y"] = x
            elif self.action == "rotate_180":
                events["x"] = dimensions[0] - 1 - events["x"]
                events["y"] = dimensions[1] - 1 - events["y"]
            elif self.action == "rotate_270_counterclockwise":
                x = events["x"].copy()
                events["x"] = events["y"]
                events["y"] = dimensions[1] - 1 - x
            elif self.action == "flip_up_diagonal":
                x = events["x"].copy()
                events["x"] = events["y"]
                events["y"] = x
            elif self.action == "flip_down_diagonal":
                x = events["x"].copy()
                events["x"] = dimensions[0] - 1 - events["y"]
                events["y"] = dimensions[1] - 1 - x
            else:
                raise Exception(f'unknown action "{self.action}"')
            yield events


@typed_filter({"", "Finite", "Uniform", "FiniteUniform"})
class Map(events_stream.FiniteUniformEventsFilter):
    def __init__(
        self,
        parent: events_stream.FiniteUniformEventsStream,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ):
        self.init(parent=parent)
        self.function = function

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for events in self.parent:
            yield self.function(events)
