import pytest

import faery


@pytest.mark.parametrize(
    "time,timestamp",
    [
        ("31:41:59.265358", 114119265358),
        ("31:41:59.26535", 114119265350),
        ("31:41:59.2653", 114119265300),
        ("31:41:59.265", 114119265000),
        ("31:41:59.26", 114119260000),
        ("31:41:59.2", 114119200000),
        ("31:41:59", 114119000000),
        ("31:41.592653", 1901592653),
        ("31:41.5", 1901500000),
        ("31:41", 1901000000),
        ("31.415926", 31415926),
        ("31.4", 31400000),
        ("31", 31000000),
        (31.415926, 31415926),
        (31.4, 31400000),
        (31, 31000000),
    ],
)
def test_parse(time: faery.Time, timestamp: int):
    assert faery.parse_timestamp(time) == timestamp
    assert faery.parse_timestamp(faery.timestamp_to_timecode(timestamp)) == timestamp
