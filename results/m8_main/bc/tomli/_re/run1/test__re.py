# SPDX-License-Identifier: MIT
"""Tests for tomli._re."""

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
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(s: str) -> float:
    return float(s)


def _fullmatch_datetime(s: str) -> "re.Match[str]":
    m = RE_DATETIME.fullmatch(s)
    assert m is not None, f"RE_DATETIME did not match {s!r}"
    return m


def _fullmatch_localtime(s: str) -> "re.Match[str]":
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None, f"RE_LOCALTIME did not match {s!r}"
    return m


def _fullmatch_number(s: str) -> "re.Match[str]":
    m = RE_NUMBER.fullmatch(s)
    assert m is not None, f"RE_NUMBER did not match {s!r}"
    return m


# ---------------------------------------------------------------------------
# match_to_datetime: date-only
# ---------------------------------------------------------------------------

def test_datetime_date_only():
    result = match_to_datetime(_fullmatch_datetime("2021-05-12"))
    assert result == date(2021, 5, 12)
    assert type(result) is date


def test_datetime_date_only_returns_date_not_datetime():
    result = match_to_datetime(_fullmatch_datetime("2021-05-12"))
    # date-only must be exactly a date, not a datetime
    assert not isinstance(result, datetime)


def test_datetime_date_leap_day():
    result = match_to_datetime(_fullmatch_datetime("2020-02-29"))
    assert result == date(2020, 2, 29)


def test_datetime_date_invalid_day():
    with pytest.raises(ValueError):
        match_to_datetime(_fullmatch_datetime("2021-02-30"))


def test_datetime_date_invalid_month_day_combination():
    # April 31 is not valid even though regex allows day 31
    with pytest.raises(ValueError):
        match_to_datetime(_fullmatch_datetime("2021-04-31"))


# ---------------------------------------------------------------------------
# match_to_datetime: local date-time (no tz)
# ---------------------------------------------------------------------------

def test_datetime_local_full_seconds():
    result = match_to_datetime(
        _fullmatch_datetime("1988-10-27T13:03:04")
    )
    assert result == datetime(1988, 10, 27, 13, 3, 4)
    assert result.tzinfo is None


def test_datetime_accepts_space_separator():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27 13:03:04"))
    assert result == datetime(1988, 10, 27, 13, 3, 4)


def test_datetime_accepts_lowercase_t():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27t13:03:04"))
    assert result == datetime(1988, 10, 27, 13, 3, 4)


def test_datetime_without_seconds():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03"))
    assert result == datetime(1988, 10, 27, 13, 3, 0)


def test_datetime_no_regex_match_when_hour_only():
    # Hours must be followed by minutes, so "13" alone should not match a time
    assert RE_DATETIME.fullmatch("1988-10-27T13") is None


def test_datetime_microseconds_short():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04.1"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, 100000)


def test_datetime_microseconds_full():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04.123456"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, 123456)


def test_datetime_microseconds_truncated_to_six_digits():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04.123456789"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, 123456)


def test_datetime_microseconds_wide_zero_padded():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04.000123"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, 123)


# ---------------------------------------------------------------------------
# match_to_datetime: UTC / zulu
# ---------------------------------------------------------------------------

def test_datetime_zulu_uppercase():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04Z"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_datetime_zulu_lowercase():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04z"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# match_to_datetime: explicit offsets
# ---------------------------------------------------------------------------

def test_datetime_positive_offset():
    result = match_to_datetime(
        _fullmatch_datetime("1988-10-27T13:03:04+05:30")
    )
    assert result == datetime(
        1988, 10, 27, 13, 3, 4, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )


def test_datetime_negative_offset():
    result = match_to_datetime(
        _fullmatch_datetime("1988-10-27T13:03:04-05:30")
    )
    assert result == datetime(
        1988, 10, 27, 13, 3, 4, tzinfo=timezone(-timedelta(hours=5, minutes=30))
    )


def test_datetime_zero_offset():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04+00:00"))
    assert result == datetime(1988, 10, 27, 13, 3, 4, tzinfo=timezone.utc)


def test_datetime_offset_without_seconds():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03+01:00"))
    assert result == datetime(
        1988, 10, 27, 13, 3, 0, tzinfo=timezone(timedelta(hours=1))
    )


def test_datetime_offset_max_hour():
    result = match_to_datetime(_fullmatch_datetime("1988-10-27T13:03:04+23:59"))
    assert result.tzinfo == timezone(timedelta(hours=23, minutes=59))


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------

def test_cached_tz_positive():
    tz = cached_tz("05", "30", "+")
    assert tz == timezone(timedelta(hours=5, minutes=30))


def test_cached_tz_negative():
    tz = cached_tz("05", "30", "-")
    assert tz == timezone(-timedelta(hours=5, minutes=30))


def test_cached_tz_zero():
    tz = cached_tz("00", "00", "+")
    assert tz == timezone.utc


def test_cached_tz_is_cached():
    a = cached_tz("12", "00", "+")
    b = cached_tz("12", "00", "+")
    assert a is b


def test_cached_tz_cache_info():
    cached_tz.cache_clear()
    cached_tz("01", "00", "+")
    cached_tz("01", "00", "+")
    cached_tz("01", "00", "-")
    info = cached_tz.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.currsize == 2
    cached_tz.cache_clear()


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------

def test_localtime_hour_minute():
    result = match_to_localtime(_fullmatch_localtime("13:03"))
    assert result == time(13, 3, 0)
    assert result.tzinfo is None


def test_localtime_with_seconds():
    result = match_to_localtime(_fullmatch_localtime("23:59:58"))
    assert result == time(23, 59, 58)


def test_localtime_with_fraction():
    result = match_to_localtime(_fullmatch_localtime("00:00:00.5"))
    assert result == time(0, 0, 0, 500000)


def test_localtime_with_full_fraction():
    result = match_to_localtime(_fullmatch_localtime("12:34:56.123456"))
    assert result == time(12, 34, 56, 123456)


def test_localtime_fraction_truncated():
    result = match_to_localtime(_fullmatch_localtime("12:34:56.123456789"))
    assert result == time(12, 34, 56, 123456)


def test_localtime_fraction_requires_seconds():
    # Fractional part only allowed after seconds; "12:34.5" should not fully match
    assert RE_LOCALTIME.fullmatch("12:34.5") is None


def test_localtime_bounds():
    assert match_to_localtime(_fullmatch_localtime("00:00")) == time(0, 0)
    assert match_to_localtime(_fullmatch_localtime("23:59:59.999999")) == time(
        23, 59, 59, 999999
    )


def test_localtime_rejects_hour_24():
    assert RE_LOCALTIME.fullmatch("24:00") is None


def test_localtime_rejects_minute_60():
    assert RE_LOCALTIME.fullmatch("12:60") is None


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0),
        ("+0", 0),
        ("-0", 0),
        ("42", 42),
        ("-42", -42),
        ("+42", 42),
        ("1_000_000", 1000000),
        ("1_0", 10),
    ],
)
def test_number_integer_decimal(text, expected):
    m = _fullmatch_number(text)
    assert match_to_number(m, _parse_float) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0xDEADBEEF", 0xDEADBEEF),
        ("0xdeadbeef", 0xDEADBEEF),
        ("0x1_2", 0x12),
        ("0b1010", 0b1010),
        ("0b1_0", 0b10),
        ("0o755", 0o755),
        ("0o7_5_5", 0o755),
    ],
)
def test_number_integer_prefixed(text, expected):
    m = _fullmatch_number(text)
    assert match_to_number(m, _parse_float) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.5", 1.5),
        ("-1.5", -1.5),
        ("+1.5", 1.5),
        ("1.0", 1.0),
        ("1e10", 1e10),
        ("1E10", 1e10),
        ("1e+10", 1e10),
        ("1e-10", 1e-10),
        ("1.5e3", 1500.0),
        ("1_0.5_0e2_0", None),  # placeholder; handled below
    ][:-1],
)
def test_number_float(text, expected):
    m = _fullmatch_number(text)
    assert match_to_number(m, _parse_float) == expected


def test_number_float_calls_parse_float():
    calls = []

    def fake_parse_float(s: str) -> str:
        calls.append(s)
        return "FLOAT"

    m = _fullmatch_number("1.25")
    result = match_to_number(m, fake_parse_float)
    assert result == "FLOAT"
    assert calls == ["1.25"]


def test_number_integer_does_not_call_parse_float():
    def fake_parse_float(s: str) -> str:
        raise AssertionError("parse_float should not be called for integers")

    m = _fullmatch_number("123")
    assert match_to_number(m, fake_parse_float) == 123


def test_number_float_with_underscores_passed_verbatim():
    received = []

    def capturing_parse_float(s: str) -> float:
        received.append(s)
        return float(s.replace("_", ""))

    m = _fullmatch_number("1_0.5_0e2_0")
    # The regex matches the full string; underscores are passed through.
    result = match_to_number(m, capturing_parse_float)
    assert result == 10.5e20
    assert received == ["1_0.5_0e2_0"]
