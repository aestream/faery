import dataclasses
import re
import typing

import numpy

FULL_TIMECODE_PATTERN: re.Pattern = re.compile(
    r"^(\d+)[:\-](\d{2})[:\-](\d{2})(\.\d{0,6})?$"
)
MINUTES_TIMECODE_PATTERN: re.Pattern = re.compile(r"^(\d+)[:\-](\d{2})(\.\d{0,6})?$")
SECONDS_TIMECODE_PATTERN: re.Pattern = re.compile(r"^(\d+)(\.\d{0,6})?$")


@dataclasses.dataclass(order=True, frozen=True)
class Time:
    microseconds: int

    def __post_init__(self):
        if isinstance(self.microseconds, numpy.integer):
            object.__setattr__(self, "microseconds", int(self.microseconds))
        elif not isinstance(self.microseconds, int):
            raise TypeError(
                f"the argument of the Time constructor must be an integer (got {self.microseconds})"
            )
        if self.microseconds < 0:
            raise ValueError(
                f"the argument of the Time constructor may not be negative (got {self.microseconds})"
            )

    def __add__(self, other) -> "Time":
        if not isinstance(other, Time):
            raise TypeError(
                f"only Time instances can be added to time objects (got {other})"
            )
        return Time(microseconds=self.microseconds + other.microseconds)

    def __sub__(self, other) -> "Time":
        if not isinstance(other, Time):
            raise TypeError(
                f"only Time instances can be subtracted from time objects (got {other})"
            )
        if self.microseconds < other.microseconds:
            raise TypeError(
                f"Time subtraction yielded a negative number ({self.microseconds} - {other.microseconds})"
            )
        return Time(self.microseconds - other.microseconds)

    def __mul__(self, other) -> "Time":
        if isinstance(other, int):
            assert other >= 0
            return Time(microseconds=self.microseconds * other)
        other = other.__float__()
        assert other >= 0.0
        return Time(microseconds=int(round(self.microseconds * other)))

    def __truediv__(self, other) -> "Time":
        other = other.__float__()
        assert other >= 0.0
        return Time(microseconds=int(round(self.microseconds / other)))

    def __floordiv__(self, other) -> "Time":
        if isinstance(other, int):
            assert other >= 0
            return Time(microseconds=self.microseconds // other)
        other = other.__float__()
        assert other >= 0.0
        return Time(microseconds=int(round(self.microseconds // other)))

    def __mod__(self, other) -> "Time":
        if isinstance(other, int):
            assert other >= 0
            return Time(microseconds=self.microseconds % other)
        other = other.__float__()
        assert other >= 0.0
        return Time(microseconds=int(round(self.microseconds % other)))

    def __divmod__(self, other) -> "tuple[Time, Time]":
        if isinstance(other, int):
            assert other >= 0
            quotient, remainder = divmod(self.microseconds, other)
            return (Time(microseconds=quotient), Time(microseconds=remainder))
        other = other.__float__()
        assert other >= 0.0
        quotient, remainder = divmod(self.microseconds, other)
        return (
            Time(microseconds=int(round(quotient))),
            Time(microseconds=int(round(remainder))),
        )

    def __rmul__(self, other) -> "Time":
        return self.__mul__(other)

    def to_timecode(self) -> str:
        microseconds = self.microseconds
        hours = microseconds // (1000000 * 60 * 60)
        microseconds -= hours * (1000000 * 60 * 60)
        minutes = microseconds // (1000000 * 60)
        microseconds -= minutes * (1000000 * 60)
        seconds = microseconds // 1000000
        microseconds -= seconds * 1000000
        return f"{hours:>02}:{minutes:>02}:{seconds:>02}.{microseconds:>06}"

    def to_timecode_with_dashes(self) -> str:
        microseconds = self.microseconds
        hours = microseconds // (1000000 * 60 * 60)
        microseconds -= hours * (1000000 * 60 * 60)
        minutes = microseconds // (1000000 * 60)
        microseconds -= minutes * (1000000 * 60)
        seconds = microseconds // 1000000
        microseconds -= seconds * 1000000
        return f"{hours:>02}-{minutes:>02}-{seconds:>02}.{microseconds:>06}"

    def to_seconds(self) -> float:
        return self.microseconds / 1e6

    def to_milliseconds(self) -> float:
        return self.microseconds / 1e3

    def to_microseconds(self) -> int:
        return self.microseconds


TimeOrTimecode = typing.Union[Time, str]
"""
A faery.Time object or a timecode (string).

Time objects can be created from integers or floats with the syntax `314 * faery.us` (314 µs) or `3.1 * faery.s` (3.1 s).

A timecode is a string in the form "hh:mm:ss.µµµµµµ" where hh are hours, mm minutes, ss seconds and µµµµµµ microseconds.
Hours, minutes, and microseconds are optional.

See *tests/test_timestamps* for a list of example patterns.
"""

us = Time(microseconds=1)
"""
A faery.Time object that represents one microsecond, can be used with the syntax `3141593 * faery.us` to create arbitrary durations.
"""

ms = Time(microseconds=1000)
"""
A faery.Time object that represents one millisecond, can be used with the syntax `3142.593 * faery.ms` to create arbitrary durations.
"""

s = Time(microseconds=1000000)
"""
A faery.Time object that represents one second, can be used with the syntax `3.142593 * faery.s` to create arbitrary durations.
"""


def parse_time(value: TimeOrTimecode) -> Time:
    """
    Converts a timestamp (timecode or seconds) to an interger number of microseconds.

    Args:
        value: faery.Time object or Timecode (string in the form "hh:mm:ss.µµµµµµ").

    Returns:
        Time: faery.Time object
    """
    if isinstance(value, Time):
        return value
    match = FULL_TIMECODE_PATTERN.match(value)
    if match is not None:
        microseconds = (
            int(match[1]) * (1000000 * 60 * 60)
            + int(match[2]) * (1000000 * 60)
            + int(match[3]) * 1000000
        )
        if match[4] is not None:
            microseconds += int(round(float(f"0{match[4]}") * 1000000.0))
        return Time(microseconds=microseconds)
    match = MINUTES_TIMECODE_PATTERN.match(value)
    if match is not None:
        microseconds = int(match[1]) * (1000000 * 60) + int(match[2]) * 1000000
        if match[3] is not None:
            microseconds += int(round(float(f"0{match[3]}") * 1000000.0))
        return Time(microseconds=microseconds)
    match = SECONDS_TIMECODE_PATTERN.match(value)
    if match is not None:
        microseconds = int(match[1]) * 1000000
        if match[2] is not None:
            microseconds += int(round(float(f"0{match[2]}") * 1000000.0))
        return Time(microseconds=microseconds)
    raise RuntimeError(f'Parsing the timecode "{value}" failed')
