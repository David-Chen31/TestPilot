# SPDX-License-Identifier: MIT
from __future__ import annotations

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
# Helpers
# ---------------------------------------------------------------------------


def _match_datetime(s: str) -> re.Match[str]:
    m = RE_DATETIME.fullmatch(s)
    assert m is not None, f"RE_DATETIME did not fully match {s!r}"
    return m


def _match_localtime(s: str) -> re.Match[str]:
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None, f"RE_LOCALTIME did not fully match {s!r}"
    return m


def _match_number(s: str) -> re.Match[str]:
    m = RE_NUMBER.fullmatch(s)
    assert m is not None, f"RE_NUMBER did not fully match {s!r}"
    return m


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------


def test_match_to_datetime_date_only():
    assert match_to_datetime(_match_datetime("1988-10-27")) == date(1988, 10, 27)


@pytest.mark.parametrize("sep", ["T", "t", " "])
def test_match_to_datetime_separator_variants(sep):
    result = match_to_datetime(_match_datetime(f"2021-03-04{sep}05:06:07"))
    assert result == datetime(2021, 3, 4, 5, 6, 7)


def test_match_to_datetime_naive_datetime():
    result = match_to_datetime(_match_datetime("2021-03-04T05:06:07"))
    assert isinstance(result, datetime)
    assert result == datetime(2021, 3, 4, 5, 6, 7)
    assert result.tzinfo is None


def test_match_to_datetime_no_seconds():
    result = match_to_datetime(_match_datetime("2021-03-04T05:06"))
    assert result == datetime(2021, 3, 4, 5, 6)
    assert result.tzinfo is None


def test_match_to_datetime_no_seconds_with_offset():
    result = match_to_datetime(_match_datetime("2021-03-04T05:06+02:30"))
    assert result == datetime(2021, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert result.utcoffset() == timedelta(hours=2, minutes=30)


def test_match_to_datetime_microseconds_padding():
    # 1 digit -> 100000 microseconds, ..., 6 digits -> as-is.
    result = match_to_datetime(_match_datetime("2021-03-04T05:06:07.1"))
    assert result.microsecond == 100000

    result = match_to_datetime(_match_datetime("2021-03-04T05:06:07.12"))
    assert result.microsecond == 120000

    result = match_to_datetime(_match_datetime("2021-03-04T05:06:07.123456"))
    assert result.microsecond == 123456


def test_match_to_datetime_microseconds_truncated():
    # More than 6 fractional digits: only the first 6 count.
    result = match_to_datetime(_match_datetime("2021-03-04T05:06:07.1234567899"))
    assert result.microsecond == 123456


@pytest.mark.parametrize("z", ["Z", "z"])
def test_match_to_datetime_zulu(z):
    result = match_to_datetime(_match_datetime(f"2021-03-04T05:06:07{z}"))
    assert result == datetime(2021, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "offset,expected",
    [
        ("+00:00", timedelta(0)),
        ("-00:00", timedelta(0)),
        ("+05:30", timedelta(hours=5, minutes=30)),
        ("-05:30", timedelta(hours=-5, minutes=-30)),
        ("+23:59", timedelta(hours=23, minutes=59)),
        ("-23:59", timedelta(hours=-23, minutes=-59)),
    ],
)
def test_match_to_datetime_offsets(offset, expected):
    result = match_to_datetime(_match_datetime(f"2021-03-04T05:06:07{offset}"))
    assert result.utcoffset() == expected
    assert isinstance(result, datetime)
    assert result.utcoffset() is not None


def test_match_to_datetime_invalid_date_raises():
    # Regex matches day 31 for February, so this passes the regex but the
    # datetime constructor must reject it.
    with pytest.raises(ValueError):
        match_to_datetime(_match_datetime("2021-02-31"))


def test_match_to_datetime_invalid_time_raises_via_no_match():
    # Hour 24 invalid; regex should reject.
    assert RE_DATETIME.fullmatch("2021-03-04T24:00") is None


def test_match_to_datetime_invalid_range_day_zero():
    # Day 00 not allowed by regex.
    assert RE_DATETIME.fullmatch("2021-03-00") is None


def test_match_to_datetime_date_returns_date_type():
    result = match_to_datetime(_match_datetime("0001-01-01"))
    assert isinstance(result, date)
    assert not isinstance(result, datetime)
    assert result == date(1, 1, 1)


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


def test_cached_tz_positive():
    tz = cached_tz("05", "30", "+")
    assert isinstance(tz, timezone)
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)


def test_cached_tz_negative():
    tz = cached_tz("05", "30", "-")
    assert tz.utcoffset(None) == timedelta(hours=-5, minutes=-30)


def test_cached_tz_zero():
    tz = cached_tz("00", "00", "+")
    assert tz.utcoffset(None) == timedelta(0)


def test_cached_tz_none_sign():
    # Anything that isn't "+" is treated as negative.
    tz = cached_tz("01", "00", "x")
    assert tz.utcoffset(None) == timedelta(hours=-1)


def test_cached_tz_is_cached():
    a = cached_tz("02", "15", "+")
    b = cached_tz("02", "15", "+")
    assert a is b


def test_cached_tz_cache_distinguishes_sign():
    a = cached_tz("02", "15", "+")
    b = cached_tz("02", "15", "-")
    assert a is not b
    assert a.utcoffset(None) == -b.utcoffset(None)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


def test_match_to_localtime_basic():
    result = match_to_localtime(_match_localtime("05:06:07"))
    assert result == time(5, 6, 7)
    assert result.microsecond == 0


def test_match_to_localtime_no_seconds():
    result = match_to_localtime(_match_localtime("05:06"))
    assert result == time(5, 6, 0)


def test_match_to_localtime_fraction_one_digit():
    result = match_to_localtime(_match_localtime("05:06:07.1"))
    assert result.microsecond == 100000


def test_match_to_localtime_fraction_six_digits():
    result = match_to_localtime(_match_localtime("05:06:07.123456"))
    assert result.microsecond == 123456


def test_match_to_localtime_fraction_truncation():
    result = match_to_localtime(_match_localtime("05:06:07.123456789"))
    assert result.microsecond == 123456


def test_match_to_localtime_midnight():
    assert match_to_localtime(_match_localtime("00:00")) == time(0, 0)


def test_match_to_localtime_last_second():
    assert match_to_localtime(_match_localtime("23:59:59")) == time(23, 59, 59)


def test_match_to_localtime_rejects_no_fraction_without_seconds():
    # "05:06." should not match (fraction requires seconds per regex).
    assert RE_LOCALTIME.fullmatch("05:06.") is None


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


def _pf_default(s: str) -> float:
    return float(s.replace("_", ""))


def test_match_to_number_integer():
    assert match_to_number(_match_number("42"), _pf_default) == 42
    assert isinstance(match_to_number(_match_number("42"), _pf_default), int)


def test_match_to_number_negative_integer():
    assert match_to_number(_match_number("-17"), _pf_default) == -17


def test_match_to_number_positive_sign():
    assert match_to_number(_match_number("+9"), _pf_default) == 9


def test_match_to_number_integer_underscores():
    assert match_to_number(_match_number("1_000_000"), _pf_default) == 1000000


def test_match_to_number_hex():
    assert match_to_number(_match_number("0xDEAD_beef"), _pf_default) == 0xDEADBEEF


def test_match_to_number_oct():
    assert match_to_number(_match_number("0o7_5_5"), _pf_default) == 0o755


def test_match_to_number_bin():
    assert match_to_number(_match_number("0b1_0_1"), _pf_default) == 0b101


def test_match_to_number_float_uses_parse_float():
    calls = []

    def parse_float(s: str) -> float:
        calls.append(s)
        return 123.456

    result = match_to_number(_match_number("1.5"), parse_float)
    assert result == 123.456
    assert calls == ["1.5"]


def test_match_to_number_float_exponent():
    seen = []

    def parse_float(s: str) -> float:
        seen.append(s)
        return -1.0

    assert match_to_number(_match_number("1e10"), parse_float) == -1.0
    assert seen == ["1e10"]


def test_match_to_number_float_fractional():
    seen = []

    def parse_float(s: str) -> float:
        seen.append(s)
        return 0.5

    assert match_to_number(_match_number("0.25"), parse_float) == 0.5
    assert seen == ["0.25"]


def test_match_to_number_integer_does_not_call_parse_float():
    def parse_float(s: str) -> float:  # pragma: no cover - must not be called
        raise AssertionError("parse_float must not be called for integers")

    assert match_to_number(_match_number("123"), parse_float) == 123


def test_match_to_number_zero():
    assert match_to_number(_match_number("0"), _pf_default) == 0
    assert match_to_number(_match_number("-0"), _pf_default) == 0


def test_match_to_number_underscore_exponent_is_float():
    seen = []

    def parse_float(s: str) -> float:
        seen.append(s)
        return 2.0

    assert match_to_number(_match_number("1e1_0"), parse_float) == 2.0
    assert seen == ["1e1_0"]
