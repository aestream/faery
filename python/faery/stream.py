import collections.abc
import typing

from . import timestamp

ItemType = typing.TypeVar("ItemType")


class Stream(typing.Generic[ItemType]):
    """
    Generic base class for streams.

    This includes inputs (files, UDP sockets, cameras...) and filters.
    A stream generates packets of a given type (frame or events).
    """

    def __iter__(self) -> collections.abc.Iterator[ItemType]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        """
        Stream dimensions in pixels.

        Returns:
            tuple[int, int]: Width (left-right direction) and height (top-bottom direction) in pixels.
        """
        raise NotImplementedError()


class FiniteStream(Stream[ItemType]):
    """
    A stream whose source has a known begin and a known end.

    A file is a finite stream but a camera is not.
    """

    def time_range_us(self) -> tuple[int, int]:
        """
        Timestamp of the stream's start and end, in microseconds.

        Start is always smaller than or equal to the first event's timestamp.

        End is always strictly larger than the last event's timestamp.

        For instance, if the stream contains 3 events with timestamps `[2, 71, 828]`, the time range may be `(2, 829)`.
        It may also be wider, for instance `(0, 1000)`.

        Returns:
            tuple[int, int]: First and one-past-last timestamps in µs.
        """
        raise NotImplementedError()

    def time_range(self) -> tuple[str, str]:
        """
        Timecodes of the stream's start and end.

        For instance, if the stream contains 3 events with timestamps `[2, 71, 828]`,
        the time range may be `("00:00:00.000002", "00:00:00.000829")`.
        It may also be wider, for instance `("00:00:00.000000", "00:00:00.001000")`.

        Returns:
            tuple[int, int]: First and one-past-last timecodes.
        """
        start, end = self.time_range_us()
        return (
            timestamp.timestamp_to_timecode(start),
            timestamp.timestamp_to_timecode(end),
        )


class RegularStream(Stream[ItemType]):
    """
    A stream whose packets cover a fixed amount of time.

    Frame streams are always regular (under the assumption of constant frame rate)
    whereas event streams can be non-regular (variable packet duration).

    The packets of non-regular event streams may have a variable number of events and / or
    cover a variable amount of time.
    """

    def frequency_hz(self) -> float:
        """
        The stream's frequency (or packet rate), in Hertz.
        """
        raise NotImplementedError()


class FiniteRegularStream(FiniteStream[ItemType], RegularStream[ItemType]):
    """
    A stream that is finite and regular (see FiniteStream and RegularStream).
    """

    pass


class Filter(Stream[ItemType]):
    """
    A filter is a stream that consumes data from another stream.
    """

    def init(self, parent: Stream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()


class FiniteFilter(FiniteStream[ItemType]):
    """
    A finite filter consumes data from a finite stream. It is thus a finite stream.
    """

    def init(self, parent: FiniteStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()


class RegularFilter(RegularStream[ItemType]):
    """
    A regular filter consumes data from a regular stream. It is thus a regular stream.
    """

    def init(self, parent: RegularStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()


class FiniteRegularFilter(FiniteRegularStream[ItemType]):
    """
    A finite regular filter consumes data from a finite regular stream. It is thus a finite regular stream.
    """

    def init(self, parent: FiniteRegularStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()

    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()
