import pytest

import faery


@pytest.mark.parametrize(
    "time,microseconds",
    [
        ("31:41:59.265358", 114119265358),
        ("31-41-59.265358", 114119265358),
        ("31:41:59.26535", 114119265350),
        ("31-41-59.26535", 114119265350),
        ("31:41:59.2653", 114119265300),
        ("31-41-59.2653", 114119265300),
        ("31:41:59.265", 114119265000),
        ("31-41-59.265", 114119265000),
        ("31:41:59.26", 114119260000),
        ("31-41-59.26", 114119260000),
        ("31:41:59.2", 114119200000),
        ("31-41-59.2", 114119200000),
        ("31:41:59", 114119000000),
        ("31-41-59", 114119000000),
        ("31:41.592653", 1901592653),
        ("31-41.592653", 1901592653),
        ("31:41.5", 1901500000),
        ("31-41.5", 1901500000),
        ("31:41", 1901000000),
        ("31-41", 1901000000),
        ("31.415926", 31415926),
        ("31.4", 31400000),
        ("31", 31000000),
        (31.415926 * faery.s, 31415926),
        (31.4 * faery.s, 31400000),
        (31 * faery.s, 31000000),
        (31.415926 * faery.ms, 31416),
        (31.4 * faery.ms, 31400),
        (31 * faery.ms, 31000),
        (31.415926 * faery.us, 31),
        (31.4 * faery.us, 31),
        (31 * faery.us, 31),
    ],
)
def test_parse_time(time: faery.Time, microseconds: int):
    assert faery.parse_time(time) == faery.Time(microseconds=microseconds)
    assert (
        faery.parse_time(
            faery.Time(microseconds=microseconds).to_timecode()
        ).to_microseconds()
        == microseconds
    )
    assert (
        faery.parse_time(
            faery.Time(microseconds=microseconds).to_timecode()
        ).to_milliseconds()
        == microseconds / 1e3
    )
    assert (
        faery.parse_time(
            faery.Time(microseconds=microseconds).to_timecode()
        ).to_seconds()
        == microseconds / 1e6
    )


def test_comparison_operators():
    assert 31.415926 * faery.s > 27.182818 * faery.s
    assert 31.415926 * faery.s >= 27.182818 * faery.s
    assert 31.415926 * faery.s >= 31415926 * faery.us
    assert 31.415926 * faery.s == 31415926 * faery.us
    assert 27.182818 * faery.s < 31.415926 * faery.s
    assert 27.182818 * faery.s <= 31.415926 * faery.s
    assert 27.182818 * faery.s <= 27182818 * faery.us
    assert 27.182818 * faery.s == 27182818 * faery.us
    assert min(31.415926 * faery.s, 27.182818 * faery.s) == 27.182818 * faery.s
    assert max(31.415926 * faery.s, 27.182818 * faery.s) == 31.415926 * faery.s


def test_add():
    assert 31.415926 * faery.s + 27.182818 * faery.s == 58598744 * faery.us
    assert 31.415926 * faery.s + 27.182818 * faery.ms == 31443109 * faery.us
    assert 31.415926 * faery.s + 27.182818 * faery.us == 31415953 * faery.us
    with pytest.raises(TypeError):
        _ = 31.415926 * faery.s + 27.182818


def test_subtract():
    assert 31.415926 * faery.s - 27.182818 * faery.s == 4233108 * faery.us
    assert 31.415926 * faery.s - 27.182818 * faery.ms == 31388743 * faery.us
    assert 31.415926 * faery.s - 27.182818 * faery.us == 31415899 * faery.us
    with pytest.raises(TypeError):
        _ = 31.415926 * faery.s - 27.182818


def test_multiply():
    assert 31.415926 * faery.s * 27 == 848230002 * faery.us
    assert 31.415926 * faery.s * 27.182818 == 853973399 * faery.us
    with pytest.raises(AttributeError):
        _ = (31.415926 * faery.s) * (27.182818 * faery.s)


def test_true_division():
    assert 31.415926 * faery.s / 27 == 1163553 * faery.us
    assert 31.415926 * faery.s / 27.182818 == 1155727 * faery.us
    with pytest.raises(AttributeError):
        _ = (31.415926 * faery.s) / (27.182818 * faery.s)


def test_floor_division():
    assert 31.415926 * faery.s // 27 == 1163552 * faery.us
    assert 31.415926 * faery.s // 27.182818 == 1155727 * faery.us
    with pytest.raises(AttributeError):
        _ = (31.415926 * faery.s) // (27.182818 * faery.s)


def test_mod():
    assert 31.415926 * faery.s % 27 == 22 * faery.us
    assert 31.415926 * faery.s % 27.182818 == 9 * faery.us
    with pytest.raises(AttributeError):
        _ = (31.415926 * faery.s) % (27.182818 * faery.s)


def test_divmod():
    assert divmod(31.415926 * faery.s, 27) == (1163552 * faery.us, 22 * faery.us)
    assert divmod(31.415926 * faery.s, 27.182818) == (1155727 * faery.us, 9 * faery.us)
    with pytest.raises(AttributeError):
        _ = divmod(31.415926 * faery.s, 27.182818 * faery.s)


def test_reverse_multiply():
    assert 31.415926 * faery.s == faery.s * 31.415926
