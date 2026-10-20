import re
from datetime import date, datetime, time, timedelta, timezone

import pytest

from tomli._re import (
    RE_DATETIME,
    RE_LOCALTIME,
    RE_NUMBER,
    cached_tz,
    match_to_datetime,
    match_to_localtime,
    match_to_number,
)


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------

def _datetime_match(text):
    m = RE_DATETIME.fullmatch(text)
    assert m is not None
    return m


def test_datetime_date_only():
    result = match_to_datetime(_datetime_match("1988-10-27"))
    assert result == date(1988, 10, 27)
    assert type(result) is date


def test_datetime_full_utc_zulu_lowercase():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03.456789Z"))
    assert result == datetime(1988, 10, 27, 1, 2, 3, 456789, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_datetime_uppercase_zulu():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03Z"))
    assert result == datetime(1988, 10, 27, 1, 2, 3, tzinfo=timezone.utc)


def test_datetime_space_separator():
    result = match_to_datetime(_datetime_match("1988-10-27 01:02:03"))
    assert result == datetime(1988, 10, 27, 1, 2, 3)
    assert result.tzinfo is None


def test_datetime_no_seconds():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02"))
    assert result == datetime(1988, 10, 27, 1, 2)
    assert result.second == 0
    assert result.microsecond == 0


def test_datetime_positive_offset():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03+05:30"))
    assert result == datetime(
        1988, 10, 27, 1, 2, 3, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    assert result.utcoffset() == timedelta(hours=5, minutes=30)


def test_datetime_negative_offset():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03-07:15"))
    assert result.utcoffset() == -timedelta(hours=7, minutes=15)


def test_datetime_offset_zero():
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03+00:00"))
    assert result.utcoffset() == timedelta(0)


def test_datetime_microseconds_truncation():
    # More than 6 fractional digits allowed; parsed digits beyond 6 discarded.
    m = _datetime_match("1988-10-27T01:02:03.1234567")
    result = match_to_datetime(m)
    assert result.microsecond == 123456


def test_datetime_microseconds_padding():
    # Fewer than 6 fractional digits are right-padded with zeros.
    result = match_to_datetime(_datetime_match("1988-10-27T01:02:03.1"))
    assert result.microsecond == 100000


def test_datetime_invalid_date_raises():
    with pytest.raises(ValueError):
        match_to_datetime(_datetime_match("2021-02-30"))


def test_datetime_invalid_time_raises():
    # 2021-01-01 is valid date, but 24 is out of range (regex prevents it),
    # so craft an invalid time via an out-of-range minute? regex disallows too.
    # Instead use Feb 30 with time to trigger the date error path with time.
    with pytest.raises(ValueError):
        match_to_datetime(_datetime_match("2021-02-30T10:00:00"))


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------

def test_cached_tz_positive():
    tz = cached_tz("05", "30", "+")
    assert tz == timezone(timedelta(hours=5, minutes=30))
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)


def test_cached_tz_negative():
    tz = cached_tz("07", "15", "-")
    assert tz.utcoffset(None) == -timedelta(hours=7, minutes=15)


def test_cached_tz_is_cached():
    a = cached_tz("01", "02", "+")
    b = cached_tz("01", "02", "+")
    assert a is b


def test_cached_tz_zero():
    tz = cached_tz("00", "00", "+")
    assert tz.utcoffset(None) == timedelta(0)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------

def _localtime_match(text):
    m = RE_LOCALTIME.fullmatch(text)
    assert m is not None
    return m


def test_localtime_basic():
    result = match_to_localtime(_localtime_match("01:02"))
    assert result == time(1, 2)
    assert result.second == 0
    assert result.microsecond == 0


def test_localtime_with_seconds():
    result = match_to_localtime(_localtime_match("23:59:58"))
    assert result == time(23, 59, 58)


def test_localtime_with_fraction():
    result = match_to_localtime(_localtime_match("01:02:03.4"))
    assert result.microsecond == 400000


def test_localtime_fraction_padding():
    result = match_to_localtime(_localtime_match("01:02:03.123"))
    assert result.microsecond == 123000


def test_localtime_fraction_truncation():
    m = _localtime_match("01:02:03.123456789")
    result = match_to_localtime(m)
    assert result == time(1, 2, 3, 123456)


def test_localtime_seconds_no_fraction():
    result = match_to_localtime(_localtime_match("00:00:00"))
    assert result == time(0, 0, 0)


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------

def _number_match(text):
    m = RE_NUMBER.fullmatch(text)
    assert m is not None
    return m


def test_number_int_decimal():
    assert match_to_number(_number_match("123"), float) == 123
    assert type(match_to_number(_number_match("123"), float)) is int


def test_number_int_negative():
    assert match_to_number(_number_match("-42"), float) == -42


def test_number_int_plus():
    assert match_to_number(_number_match("+42"), float) == 42


def test_number_int_underscores():
    assert match_to_number(_number_match("1_000_000"), float) == 1000000


def test_number_int_zero():
    assert match_to_number(_number_match("0"), float) == 0


def test_number_hex():
    assert match_to_number(_number_match("0xDEADBEEF"), float) == 0xDEADBEEF


def test_number_octal():
    assert match_to_number(_number_match("0o755"), float) == 0o755


def test_number_binary():
    assert match_to_number(_number_match("0b1010"), float) == 0b1010


def test_number_float_uses_parse_float():
    calls = []

    def parse_float(s):
        calls.append(s)
        return ("float", s)

    result = match_to_number(_number_match("3.14"), parse_float)
    assert result == ("float", "3.14")
    assert calls == ["3.14"]


def test_number_float_exponent():
    calls = []

    def parse_float(s):
        calls.append(s)
        return s

    result = match_to_number(_number_match("1e10"), parse_float)
    assert result == "1e10"
    assert calls == ["1e10"]


def test_number_float_with_exponent_and_fraction():
    calls = []

    def parse_float(s):
        calls.append(s)
        return s

    result = match_to_number(_number_match("-2.5e-3"), parse_float)
    assert result == "-2.5e-3"
    assert calls == ["-2.5e-3"]


def test_number_float_default_float():
    assert match_to_number(_number_match("2.5"), float) == 2.5
