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

def _datetime(pattern: str) -> re.Match[str]:
    match = RE_DATETIME.fullmatch(pattern)
    assert match is not None, pattern
    return match


def _localtime(pattern: str) -> re.Match[str]:
    match = RE_LOCALTIME.fullmatch(pattern)
    assert match is not None, pattern
    return match


def _number(pattern: str) -> re.Match[str]:
    match = RE_NUMBER.fullmatch(pattern)
    assert match is not None, pattern
    return match


import re  # noqa: E402  (placed after helpers for readability)


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------

class TestMatchToDatetime:
    def test_date_only(self) -> None:
        assert match_to_datetime(_datetime("1979-05-27")) == date(1979, 5, 27)

    def test_date_only_returns_date_not_datetime(self) -> None:
        result = match_to_datetime(_datetime("2000-01-01"))
        assert isinstance(result, date)
        assert not isinstance(result, datetime)

    def test_date_boundary_values(self) -> None:
        assert match_to_datetime(_datetime("0001-01-01")) == date(1, 1, 1)
        assert match_to_datetime(_datetime("9999-12-31")) == date(9999, 12, 31)

    def test_datetime_with_t_separator(self) -> None:
        assert match_to_datetime(_datetime("1979-05-27T07:32:00")) == datetime(
            1979, 5, 27, 7, 32, 0
        )

    def test_datetime_with_space_separator(self) -> None:
        assert match_to_datetime(_datetime("1979-05-27 07:32:00")) == datetime(
            1979, 5, 27, 7, 32, 0
        )

    def test_datetime_with_lowercase_t(self) -> None:
        assert match_to_datetime(_datetime("1979-05-27t07:32:00")) == datetime(
            1979, 5, 27, 7, 32, 0
        )

    def test_datetime_omitting_seconds(self) -> None:
        assert match_to_datetime(_datetime("1979-05-27T07:32")) == datetime(
            1979, 5, 27, 7, 32, 0
        )

    def test_datetime_with_seconds(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00"))
        assert result.tzinfo is None
        assert result == datetime(1979, 5, 27, 0, 32, 0)

    def test_datetime_with_fraction(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00.999999"))
        assert result.microsecond == 999999

    def test_datetime_with_short_fraction_padded(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00.5"))
        assert result.microsecond == 500000

    def test_datetime_with_one_digit_fraction(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00.1"))
        assert result.microsecond == 100000

    def test_datetime_with_long_fraction_truncated(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00.123456789"))
        assert result.microsecond == 123456

    def test_datetime_zulu_uppercase(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00Z"))
        assert result.tzinfo == timezone.utc
        assert result == datetime(1979, 5, 27, 0, 32, 0, tzinfo=timezone.utc)

    def test_datetime_zulu_lowercase(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00z"))
        assert result.tzinfo is timezone.utc

    def test_datetime_positive_offset(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00+07:30"))
        expected_tz = timezone(timedelta(hours=7, minutes=30))
        assert result.tzinfo == expected_tz
        assert result == datetime(1979, 5, 27, 0, 32, 0, tzinfo=expected_tz)

    def test_datetime_negative_offset(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:32:00-07:30"))
        expected_tz = timezone(timedelta(hours=-7, minutes=-30))
        assert result.tzinfo == expected_tz
        assert result == datetime(1979, 5, 27, 0, 32, 0, tzinfo=expected_tz)

    def test_datetime_offset_without_seconds(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T07:32+01:00"))
        assert result == datetime(
            1979, 5, 27, 7, 32, tzinfo=timezone(timedelta(hours=1))
        )

    def test_datetime_offset_zero(self) -> None:
        result = match_to_datetime(_datetime("1979-05-27T00:00:00+00:00"))
        assert result == datetime(1979, 5, 27, tzinfo=timezone.utc)
        assert result.tzinfo == timezone.utc

    def test_datetime_seconds_with_offset(self) -> None:
        result = match_to_datetime(_datetime("2000-01-01T12:34:56.123-05:00"))
        expected_tz = timezone(timedelta(hours=-5))
        assert result == datetime(2000, 1, 1, 12, 34, 56, 123000, tzinfo=expected_tz)

    def test_invalid_date_raises_value_error(self) -> None:
        # Feb 30 matches the regex but is not a valid date.
        with pytest.raises(ValueError):
            match_to_datetime(_datetime("2000-02-30"))
        with pytest.raises(ValueError):
            match_to_datetime(_datetime("2000-02-30T00:00:00"))

    def test_invalid_month_day_raises(self) -> None:
        with pytest.raises(ValueError):
            match_to_datetime(_datetime("2000-04-31"))


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------

class TestCachedTz:
    def test_positive(self) -> None:
        assert cached_tz("07", "30", "+") == timezone(
            timedelta(hours=7, minutes=30)
        )

    def test_negative(self) -> None:
        assert cached_tz("07", "30", "-") == timezone(
            timedelta(hours=-7, minutes=-30)
        )

    def test_zero(self) -> None:
        assert cached_tz("00", "00", "+") == timezone.utc
        assert cached_tz("00", "00", "-") == timezone.utc

    def test_full_hours(self) -> None:
        assert cached_tz("23", "00", "+") == timezone(timedelta(hours=23))

    def test_minutes_only(self) -> None:
        assert cached_tz("00", "30", "+") == timezone(timedelta(minutes=30))
        assert cached_tz("00", "30", "-") == timezone(timedelta(minutes=-30))

    def test_returns_timezone_instance(self) -> None:
        assert isinstance(cached_tz("01", "02", "+"), timezone)

    def test_cache_returns_same_result(self) -> None:
        # lru_cache should return the identical object for identical args.
        first = cached_tz("05", "45", "-")
        second = cached_tz("05", "45", "-")
        assert first is second

    def test_cache_distinguishes_sign(self) -> None:
        plus = cached_tz("03", "00", "+")
        minus = cached_tz("03", "00", "-")
        assert plus != minus
        assert plus.utcoffset(None) == timedelta(hours=3)
        assert minus.utcoffset(None) == timedelta(hours=-3)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------

class TestMatchToLocaltime:
    def test_hours_minutes(self) -> None:
        assert match_to_localtime(_localtime("07:32")) == time(7, 32)

    def test_hours_minutes_seconds(self) -> None:
        assert match_to_localtime(_localtime("07:32:00")) == time(7, 32, 0)

    def test_with_fraction(self) -> None:
        assert match_to_localtime(_localtime("07:32:00.999999")) == time(
            7, 32, 0, 999999
        )

    def test_short_fraction_padded(self) -> None:
        assert match_to_localtime(_localtime("07:32:00.5")) == time(
            7, 32, 0, 500000
        )

    def test_long_fraction_truncated(self) -> None:
        assert match_to_localtime(_localtime("07:32:00.123456789")) == time(
            7, 32, 0, 123456
        )

    def test_midnight(self) -> None:
        assert match_to_localtime(_localtime("00:00")) == time(0, 0)

    def test_last_valid_time(self) -> None:
        assert match_to_localtime(_localtime("23:59:59.999999")) == time(
            23, 59, 59, 999999
        )

    def test_no_tzinfo(self) -> None:
        result = match_to_localtime(_localtime("12:00"))
        assert result.tzinfo is None


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------

class TestMatchToNumber:
    def test_integer_decimal(self) -> None:
        assert match_to_number(_number("123"), float) == 123
        assert isinstance(match_to_number(_number("123"), float), int)

    def test_integer_zero(self) -> None:
        assert match_to_number(_number("0"), float) == 0

    def test_signed_integers(self) -> None:
        assert match_to_number(_number("+42"), float) == 42
        assert match_to_number(_number("-42"), float) == -42

    def test_integer_with_underscores(self) -> None:
        assert match_to_number(_number("1_000"), float) == 1000

    def test_negative_with_underscores(self) -> None:
        assert match_to_number(_number("-1_000"), float) == -1000

    def test_hex(self) -> None:
        assert match_to_number(_number("0xDEADBEEF"), float) == 0xDEADBEEF

    def test_hex_with_underscores(self) -> None:
        assert match_to_number(_number("0xdead_beef"), float) == 0xDEADBEEF

    def test_binary(self) -> None:
        assert match_to_number(_number("0b11010110"), float) == 0b11010110

    def test_binary_with_underscores(self) -> None:
        assert match_to_number(_number("0b1_0_1"), float) == 0b101

    def test_octal(self) -> None:
        assert match_to_number(_number("0o755"), float) == 0o755

    def test_octal_with_underscores(self) -> None:
        assert match_to_number(_number("0o7_5_5"), float) == 0o755

    def test_float_uses_parse_float(self) -> None:
        calls = []

        def parse_float(s: str) -> float:
            calls.append(s)
            return 1.5

        result = match_to_number(_number("1.5"), parse_float)
        assert result == 1.5
        assert calls == ["1.5"]

    def test_float_with_fraction_and_exponent(self) -> None:
        assert match_to_number(_number("1.5e10"), float) == 1.5e10

    def test_float_underscore(self) -> None:
        # float() handles underscores; parse_float receives the raw text.
        assert match_to_number(_number("1_000.0"), float) == 1000.0

    def test_float_positive_exponent(self) -> None:
        assert match_to_number(_number("1e+10"), float) == 1e10

    def test_float_negative_exponent(self) -> None:
        assert match_to_number(_number("1e-10"), float) == 1e-10

    def test_float_leading_dot_not_allowed_by_regex(self) -> None:
        # The regex requires a digit after the dot.
        assert RE_NUMBER.fullmatch(".5") is None

    def test_integer_exponent_is_float(self) -> None:
        # "1e0" matches floatpart (exponent), so parse_float is used.
        calls = []

        def parse_float(s: str) -> float:
            calls.append(s)
            return float(s)

        result = match_to_number(_number("1e0"), parse_float)
        assert result == 1.0
        assert calls == ["1e0"]

    def test_no_floatpart_uses_int(self) -> None:
        # If parse_float raises, it means it was (incorrectly) called.
        def parse_float(s: str) -> float:
            raise AssertionError("parse_float should not be called")

        assert match_to_number(_number("7"), parse_float) == 7
