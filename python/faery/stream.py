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


class RegularFilter(RegularStream[ItemType]):
    """
    A regular filter consumes data from a regular stream. It is thus a regular stream.
    """

    def init(self, parent: RegularStream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def period_us(self) -> int:
        return self.parent.period_us()


class FiniteRegularFilter(FiniteRegularStream[ItemType]):
    """
    A finite regular filter consumes data from a finite regular stream. It is thus a finite regular stream.
    """

    def init(self, parent: FiniteRegularStream[ItemType]):
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range_us(self) -> tuple[int, int]:
        return self.parent.time_range_us()

    def period_us(self) -> int:
        return self.parent.period_us()


TransposeAction = typing.Literal[
    "flip_left_right",
    "flip_bottom_top",
    "rotate_90_counterclockwise",
    "rotate_180",
    "rotate_270_counterclockwise",
    "flip_up_diagonal",
    "flip_down_diagonal",
]
"""
Spatial transformation that applies to events and frames

- flip_left_right mirrors horizontally
- flip_bottom_top mirrors vertically
- rotate_90_counterclockwise rotates to the left by 90º
- rotate_180 rotates by 180º
- rotate_270_counterclockwise rotates to the right by 90º
- flip_up_diagonal mirrors alongside the diagonal that goes from the bottom left to the top right (also known as transverse)
- flip_down_diagonal mirrors alongside the diagonal that goes from the top left to the bottom right (also known as transpose)
"""
