import collections.abc
import types
import typing

import numpy

from . import enums, events_stream, stream, timestamp

# Empty arrays
# ------------
#
# A stream must yield `numpy.ndarray` objects with dtype `events_stream.EVENTS_DTYPE`.
# Whilst yielding `None` is not allowed, a stream may yield empty event arrays.
#
# Filters that need to access event data (for instance `events["t"][0]`)
# must check that the array is not empty first (`len(events) > 0`).
#
# Non-regular filters *may* choose to optimize dispatching by only
# yielding non-empty arrays.
#
# However, regular filters *must* yield every event packet to preserve regularity.
# This applies to any class inheriting from `events_stream.FiniteRegularEventsFilter`.


def restrict(prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]]):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_filter(
    prefixes: set[typing.Literal["", "Finite", "Regular", "FiniteRegular"]]
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


@typed_filter({"Regular", "FiniteRegular"})
class Regularize(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        frequency_hz: float,
        start: typing.Optional[timestamp.Time] = None,
    ):
        self.init(parent=parent)
        self._frequency_hz = frequency_hz
        self.start = None if start is None else timestamp.parse_timestamp(start)

    def frequency_hz(self) -> float:
        return self._frequency_hz

    @restrict({"FiniteRegular"})
    def time_range_us(self) -> tuple[int, int]:
        parent_time_range_us = self.parent.time_range_us()
        if self.start is None:
            start = parent_time_range_us[0]
        else:
            start = self.start
        period_us = 1e6 / self.frequency_hz()
        assert period_us > 0
        target_end = max(start + 1, parent_time_range_us[1])
        end_index = 1
        while True:
            end = int(round(start + end_index * period_us))
            if end >= target_end:
                return (start, end)
            end_index += 1

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        try:
            parent_time_range_us = self.parent.time_range_us()
            packet_index = 0
            first_packet_start_t = (
                parent_time_range_us[0] if self.start is None else self.start
            )
            end_t = parent_time_range_us[1]
        except AttributeError:
            packet_index = 0
            first_packet_start_t = self.start
            end_t = None
        events_buffers: list[numpy.ndarray] = []
        period_us = 1e6 / self._frequency_hz
        for events in self.parent:
            while len(events) > 0:
                if first_packet_start_t is None:
                    first_packet_start_t = events["t"][0]
                if events["t"][-1] < first_packet_start_t:
                    break
                next_packet_start_t = int(
                    round(first_packet_start_t + (packet_index + 1) * period_us)
                )
                if events["t"][0] >= next_packet_start_t:
                    yield numpy.concatenate(
                        events_buffers, dtype=events_stream.EVENTS_DTYPE
                    )
                    events_buffers = []
                    packet_index += 1
                    continue
                if events["t"][-1] < next_packet_start_t:
                    events_buffers.append(events)
                    break
                pivot = numpy.searchsorted(events["t"], next_packet_start_t)
                if len(events_buffers) == 0:
                    yield events[:pivot]
                else:
                    events_buffers.append(events[:pivot])
                    yield numpy.concatenate(
                        events_buffers, dtype=events_stream.EVENTS_DTYPE
                    )
                    events_buffers = []
                events = events[pivot:]
                packet_index += 1
        if len(events_buffers) > 0:
            assert first_packet_start_t is not None
            yield numpy.concatenate(events_buffers, dtype=events_stream.EVENTS_DTYPE)
            events_buffers = []
            packet_index += 1
        if first_packet_start_t is not None and end_t is not None:
            while first_packet_start_t + packet_index * period_us < end_t:
                yield numpy.array([], dtype=events_stream.EVENTS_DTYPE)
                packet_index += 1


@typed_filter({"", "Finite"})
class Chunks(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
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
                    yield numpy.concatenate(
                        events_buffers, dtype=events_stream.EVENTS_DTYPE
                    )
                    events_buffers = []
                events = events[pivot:]
        if len(events_buffers) > 0:
            yield numpy.concatenate(events_buffers, dtype=events_stream.EVENTS_DTYPE)
            events_buffers = []


"""
class OffsetT(events_stream.FiniteRegularEventsFilter):
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


@typed_filter({"FiniteRegular"})
class PacketSlice(events_stream.FiniteRegularEventsFilter):  # type: ignore
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        start: int,
        end: int,
        zero: bool = False,
    ):
        self.init(parent=parent)
        self.start = start
        self.end = end
        assert start < end, f"{start=} must be strictly smaller than {end=}"
        self.zero = zero

    def time_range_us(self) -> tuple[int, int]:
        period_us = 1e6 / self.frequency_hz()
        parent_time_range_us = self.parent.time_range_us()
        if self.zero:
            return (
                0,
                round(
                    int(parent_time_range_us[0] + period_us * (self.end - self.start))
                ),
            )
        else:
            return (
                round(int(parent_time_range_us[0] + period_us * self.start)),
                round(int(parent_time_range_us[0] + period_us * self.end)),
            )

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        if self.zero:
            period_us = 1e6 / self.frequency_hz()
            parent_time_range_us = self.parent.time_range_us()
            offset = round(int(parent_time_range_us[0] + period_us * self.start))
        else:
            offset = None
        for index, events in enumerate(self.parent):
            if index < self.start:
                continue
            if index >= self.end:
                break
            if offset is not None:
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
class RegularEventSlice(events_stream.RegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
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
                yield numpy.ndarray([], dtype=events_stream.EVENTS_DTYPE)
            index += 1


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Crop(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
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


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Mask(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
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


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Transpose(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        action: enums.TransposeAction,
    ):
        self.init(parent=parent)
        self.action = enums.validate_transpose_action(action)

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


@typed_filter({"", "Finite", "Regular", "FiniteRegular"})
class Map(events_stream.FiniteRegularEventsFilter):
    def __init__(
        self,
        parent: events_stream.FiniteRegularEventsStream,
        function: collections.abc.Callable[[numpy.ndarray], numpy.ndarray],
    ):
        self.init(parent=parent)
        self.function = function

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for events in self.parent:
            yield self.function(events)
