# SPDX-License-Identifier: MIT
from __future__ import annotations

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


def _match_datetime(s: str):
    m = RE_DATETIME.fullmatch(s)
    assert m is not None
    return m


def test_match_to_datetime_date_only():
    result = match_to_datetime(_match_datetime("1988-10-27"))
    assert result == date(1988, 10, 27)
    assert type(result) is date


def test_match_to_datetime_leap_day():
    assert match_to_datetime(_match_datetime("2020-02-29")) == date(2020, 2, 29)


@pytest.mark.parametrize("sep", ["T", "t", " "])
def test_match_to_datetime_separator(sep):
    result = match_to_datetime(_match_datetime(f"1979-05-27{sep}07:32:00"))
    assert result == datetime(1979, 5, 27, 7, 32, 0)
    assert type(result) is datetime


def test_match_to_datetime_hour_minute_only():
    result = match_to_datetime(_match_datetime("1979-05-27T07:32"))
    assert result == datetime(1979, 5, 27, 7, 32, 0)
    assert result.second == 0
    assert result.microsecond == 0


def test_match_to_datetime_with_seconds():
    result = match_to_datetime(_match_datetime("1979-05-27T00:32:01"))
    assert result == datetime(1979, 5, 27, 0, 32, 1)


@pytest.mark.parametrize(
    ("frac_str", "expected_micros"),
    [
        ("5", 500000),
        ("50", 500000),
        ("500", 500000),
        ("5000", 500000),
        ("50000", 500000),
        ("500000", 500000),
        ("123456", 123456),
        ("000001", 1),
        ("1", 100000),
    ],
)
def test_match_to_datetime_fractional_seconds(frac_str, expected_micros):
    result = match_to_datetime(
        _match_datetime(f"1979-05-27T00:32:01.{frac_str}")
    )
    assert result.microsecond == expected_micros


def test_match_to_datetime_fraction_truncated_beyond_6_digits():
    # The regex allows more digits but only captures the first 6.
    m = RE_DATETIME.fullmatch("1979-05-27T00:32:00.9999999999")
    assert m is not None
    result = match_to_datetime(m)
    assert result.microsecond == 999999


@pytest.mark.parametrize("z", ["Z", "z"])
def test_match_to_datetime_zulu(z):
    result = match_to_datetime(_match_datetime(f"1979-05-27T07:32:00{z}"))
    assert result == datetime(1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_match_to_datetime_positive_offset():
    result = match_to_datetime(_match_datetime("1979-05-27T07:32:00+01:30"))
    assert result.utcoffset() == timedelta(hours=1, minutes=30)
    assert result == datetime(1979, 5, 27, 7, 32, 0)


def test_match_to_datetime_negative_offset():
    result = match_to_datetime(_match_datetime("1979-05-27T07:32:00-05:00"))
    assert result.utcoffset() == timedelta(hours=-5)


def test_match_to_datetime_negative_offset_with_minutes():
    result = match_to_datetime(_match_datetime("1979-05-27T07:32:00-00:30"))
    assert result.utcoffset() == timedelta(minutes=-30)


def test_match_to_datetime_no_tzinfo_for_local():
    result = match_to_datetime(_match_datetime("1979-05-27T07:32:00"))
    assert result.tzinfo is None


def test_match_to_datetime_full_components():
    result = match_to_datetime(
        _match_datetime("2000-01-02T03:04:05.678+11:45")
    )
    assert result.year == 2000
    assert result.month == 1
    assert result.day == 2
    assert result.hour == 3
    assert result.minute == 4
    assert result.second == 5
    assert result.microsecond == 678000
    assert result.utcoffset() == timedelta(hours=11, minutes=45)


def test_match_to_datetime_invalid_date():
    with pytest.raises(ValueError):
        match_to_datetime(_match_datetime("2019-02-29"))


def test_match_to_datetime_invalid_month_day():
    # April has only 30 days.
    with pytest.raises(ValueError):
        match_to_datetime(_match_datetime("2021-04-31"))


def test_match_to_datetime_invalid_time():
    # Regex restricts hour, but e.g. 24 is disallowed by regex; use day errors.
    with pytest.raises(ValueError):
        match_to_datetime(_match_datetime("2021-13-01")) if False else match_to_datetime(
            _match_datetime("2021-01-00")
        )


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


def test_cached_tz_positive():
    tz = cached_tz("01", "30", "+")
    assert tz == timezone(timedelta(hours=1, minutes=30))
    assert tz.utcoffset(None) == timedelta(hours=1, minutes=30)


def test_cached_tz_negative():
    tz = cached_tz("05", "00", "-")
    assert tz.utcoffset(None) == timedelta(hours=-5)


def test_cached_tz_zero():
    tz = cached_tz("00", "00", "+")
    assert tz.utcoffset(None) == timedelta(0)


def test_cached_tz_is_cached():
    a = cached_tz("10", "15", "+")
    b = cached_tz("10", "15", "+")
    assert a is b
    info = cached_tz.cache_info()
    assert info.maxsize is None
    assert info.hits >= 1


def test_cached_tz_different_signs_not_same():
    pos = cached_tz("02", "00", "+")
    neg = cached_tz("02", "00", "-")
    assert pos is not neg
    assert pos.utcoffset(None) == timedelta(hours=2)
    assert neg.utcoffset(None) == timedelta(hours=-2)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


def _match_localtime(s: str):
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None
    return m


def test_match_to_localtime_hour_minute():
    result = match_to_localtime(_match_localtime("07:32"))
    assert result == time(7, 32)
    assert result.second == 0
    assert result.microsecond == 0


def test_match_to_localtime_with_seconds():
    result = match_to_localtime(_match_localtime("23:59:59"))
    assert result == time(23, 59, 59)


@pytest.mark.parametrize(
    ("frac_str", "expected_micros"),
    [
        ("1", 100000),
        ("12", 120000),
        ("123", 123000),
        ("123456", 123456),
        ("000001", 1),
        ("5", 500000),
    ],
)
def test_match_to_localtime_fractions(frac_str, expected_micros):
    result = match_to_localtime(_match_localtime(f"07:32:00.{frac_str}"))
    assert result.microsecond == expected_micros


def test_match_to_localtime_fraction_truncated_beyond_6_digits():
    m = RE_LOCALTIME.fullmatch("07:32:00.123456789")
    assert m is not None
    result = match_to_localtime(m)
    assert result.microsecond == 123456


def test_match_to_localtime_edge_midnight():
    result = match_to_localtime(_match_localtime("00:00"))
    assert result == time(0, 0)


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


def _parse_float(s: str) -> float:
    # Mimic a simple toml float parse (replace underscores).
    return float(s.replace("_", ""))


def _match_number(s: str):
    m = RE_NUMBER.fullmatch(s)
    assert m is not None
    return m


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("42", 42),
        ("+42", 42),
        ("-42", -42),
        ("1_000", 1000),
        ("1_2_3", 123),
        ("0xDEADBEEF", 0xDEADBEEF),
        ("0xde_ad_be_ef", 0xDEADBEEF),
        ("0x0", 0),
        ("0b1101", 0b1101),
        ("0b1_0_1", 5),
        ("0o755", 0o755),
        ("0o7_5_5", 0o755),
        ("-0", 0),
        ("+0", 0),
    ],
)
def test_match_to_number_integers(s, expected):
    result = match_to_number(_match_number(s), _parse_float)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("1.0", 1.0),
        ("-1.5", -1.5),
        ("+3.14", 3.14),
        ("1e10", 1e10),
        ("1E10", 1e10),
        ("1e+10", 1e10),
        ("1e-10", 1e-10),
        ("1.5e3", 1500.0),
        ("6.626e-34", 6.626e-34),
        ("10.0", 10.0),
        ("1_000.5", 1000.5),
        ("1.5_5", 1.55),
    ],
)
def test_match_to_number_floats(s, expected):
    result = match_to_number(_match_number(s), _parse_float)
    assert result == expected
    assert isinstance(result, float)


def test_match_to_number_uses_provided_parse_float():
    calls = []

    def parse_float(s: str) -> float:
        calls.append(s)
        return 99.0

    result = match_to_number(_match_number("1.5"), parse_float)
    assert result == 99.0
    assert calls == ["1.5"]


def test_match_to_number_parse_float_not_called_for_int():
    def parse_float(s: str) -> float:
        raise AssertionError("should not be called")

    assert match_to_number(_match_number("123"), parse_float) == 123
    assert match_to_number(_match_number("0xFF"), parse_float) == 0xFF


def test_match_to_number_exponent_alone_present_group():
    # "1e5" has a non-empty floatpart group, so parse_float is used.
    m = _match_number("1e5")
    assert m.group("floatpart")
    result = match_to_number(m, _parse_float)
    assert result == 100000.0
    assert isinstance(result, float)


def test_match_to_number_plain_int_floatpart_empty():
    m = _match_number("123")
    assert m.group("floatpart") == ""
    assert isinstance(match_to_number(m, _parse_float), int)
