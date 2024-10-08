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


class UniformStream(Stream[ItemType]):
    """
    A stream whose packets cover a fixed amount of time.

    Frame streams are always uniform (under the assumption of constant frame rate)
    whereas event streams can be non-uniform (variable packet duration).

    The packets of non-uniform event streams may have a variable number of events and / or
    cover a variable amount of time.
    """

    def period_us(self) -> int:
        """
        The stream's period (or frame duration), in microseconds.
        """
        raise NotImplementedError()

    def period(self) -> str:
        """
        The stream's period (or frame duration) as a timecode.
        """
        return timestamp.timestamp_to_timecode(self.period_us())


class FiniteUniformStream(FiniteStream[ItemType], UniformStream[ItemType]):
    """
    A stream that is finite and uniform (see FiniteStream and UniformStream).
    """

    pass


class Filter(Stream[ItemType]):
    """
    A filter is a stream that consumes data from another stream.
    """

    def init(self, parent: Stream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()


class FiniteFilter(FiniteStream[ItemType]):
    """
    A finite filter consumes data from a finite stream. It is thus a finite stream.
    """

    def init(self, parent: FiniteStream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()


class UniformFilter(UniformStream[ItemType]):
    """
    A uniform filter consumes data from a uniform stream. It is thus a uniform stream.
    """

    def init(self, parent: UniformStream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def period_us(self) -> int:
        return self.parent.period_us()


class FiniteUniformFilter(FiniteUniformStream[ItemType]):
    """
    A finite uniform filter consumes data from a finite uniform stream. It is thus a finite uniform stream.
    """

    def init(self, parent: FiniteUniformStream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()

    def period_us(self) -> int:
        return self.parent.period_us()
