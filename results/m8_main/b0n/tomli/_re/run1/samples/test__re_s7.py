# SPDX-License-Identifier: MIT
"""Tests for tomli._re."""

from __future__ import annotations

from datetime import date, datetime, timedelta, time, timezone
from typing import Any

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


def _datetime_match(s: str):
    m = RE_DATETIME.fullmatch(s)
    assert m is not None, f"{s!r} did not match RE_DATETIME"
    return m


def _localtime_match(s: str):
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None, f"{s!r} did not match RE_LOCALTIME"
    return m


def _number_match(s: str):
    m = RE_NUMBER.fullmatch(s)
    assert m is not None, f"{s!r} did not match RE_NUMBER"
    return m


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------


class TestMatchToDatetime:
    def test_date_only(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27"))
        assert result == date(1988, 10, 27)
        assert type(result) is date

    def test_date_only_leading_zero(self) -> None:
        assert match_to_datetime(_datetime_match("0001-01-01")) == date(1, 1, 1)

    def test_datetime_local_no_offset(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30"))
        assert result == datetime(1988, 10, 27, 10, 20, 30)
        assert type(result) is datetime
        assert result.tzinfo is None

    def test_datetime_space_separator(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27 10:20:30"))
        assert result == datetime(1988, 10, 27, 10, 20, 30)
        assert result.tzinfo is None

    def test_datetime_lowercase_t(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27t10:20:30"))
        assert result == datetime(1988, 10, 27, 10, 20, 30)

    def test_datetime_no_seconds(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20"))
        assert result == datetime(1988, 10, 27, 10, 20, 0)
        assert result.microsecond == 0

    def test_datetime_with_fractional_seconds(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30.123"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, 123000)
        assert result.microsecond == 123000

    def test_datetime_fractional_seconds_padded(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30.5"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, 500000)

    def test_datetime_fractional_seconds_six_digits(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30.123456"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, 123456)

    def test_datetime_fractional_seconds_truncated_to_six(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30.123456789"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, 123456)

    def test_datetime_zulu(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30Z"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_datetime_lowercase_zulu(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30z"))
        assert result == datetime(1988, 10, 27, 10, 20, 30, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_datetime_positive_offset(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30+05:30"))
        expected_tz = timezone(timedelta(hours=5, minutes=30))
        assert result == datetime(
            1988, 10, 27, 10, 20, 30, tzinfo=expected_tz
        )
        assert result.utcoffset() == timedelta(hours=5, minutes=30)

    def test_datetime_negative_offset(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30-03:00"))
        assert result.utcoffset() == timedelta(hours=-3)

    def test_datetime_negative_offset_with_minutes(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30-03:15"))
        assert result.utcoffset() == timedelta(hours=-3, minutes=-15)

    def test_datetime_offset_without_seconds(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20+01:00"))
        assert result == datetime(
            1988, 10, 27, 10, 20, tzinfo=timezone(timedelta(hours=1))
        )

    def test_datetime_offset_max(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30+23:59"))
        assert result.utcoffset() == timedelta(hours=23, minutes=59)

    def test_datetime_offset_zero(self) -> None:
        result = match_to_datetime(_datetime_match("1988-10-27T10:20:30+00:00"))
        assert result.utcoffset() == timedelta(0)

    def test_invalid_date_raises(self) -> None:
        with pytest.raises(ValueError):
            match_to_datetime(_datetime_match("1988-02-30"))

    def test_invalid_date_month_13_not_regex_matched(self) -> None:
        # The regex itself restricts months, so construct a synthetic case
        # via a valid regex match with out-of-range day handled by date().
        with pytest.raises(ValueError):
            match_to_datetime(_datetime_match("2001-01-31") and _datetime_match("2001-02-31").expand(""))  # type: ignore[attr-defined]

    def test_leap_day_valid(self) -> None:
        assert match_to_datetime(_datetime_match("2000-02-29")) == date(2000, 2, 29)

    def test_leap_day_invalid(self) -> None:
        with pytest.raises(ValueError):
            match_to_datetime(_datetime_match("1900-02-29"))


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


class TestCachedTz:
    def test_positive(self) -> None:
        tz = cached_tz("05", "30", "+")
        assert isinstance(tz, timezone)
        assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)

    def test_negative(self) -> None:
        tz = cached_tz("05", "30", "-")
        assert tz.utcoffset(None) == timedelta(hours=-5, minutes=-30)

    def test_plus_zero(self) -> None:
        tz = cached_tz("00", "00", "+")
        assert tz.utcoffset(None) == timedelta(0)

    def test_minus_zero(self) -> None:
        tz = cached_tz("00", "00", "-")
        assert tz.utcoffset(None) == timedelta(0)

    def test_caching(self) -> None:
        a = cached_tz("10", "00", "+")
        b = cached_tz("10", "00", "+")
        assert a is b

    def test_distinct_signs_not_same(self) -> None:
        a = cached_tz("10", "00", "+")
        b = cached_tz("10", "00", "-")
        assert a is not b
        assert a.utcoffset(None) == -b.utcoffset(None)

    def test_sign_other_than_plus_is_negative(self) -> None:
        # Implementation: sign = 1 if sign_str == "+" else -1
        tz = cached_tz("01", "00", "x")
        assert tz.utcoffset(None) == timedelta(hours=-1)

    def test_mixed_minutes(self) -> None:
        tz = cached_tz("01", "45", "-")
        assert tz.utcoffset(None) == timedelta(hours=-1, minutes=-45)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


class TestMatchToLocaltime:
    def test_hh_mm(self) -> None:
        result = match_to_localtime(_localtime_match("10:20"))
        assert result == time(10, 20, 0)
        assert type(result) is time

    def test_hh_mm_ss(self) -> None:
        result = match_to_localtime(_localtime_match("10:20:30"))
        assert result == time(10, 20, 30)

    def test_hh_mm_ss_fraction(self) -> None:
        result = match_to_localtime(_localtime_match("10:20:30.123"))
        assert result == time(10, 20, 30, 123000)

    def test_fraction_padded(self) -> None:
        result = match_to_localtime(_localtime_match("00:00:00.1"))
        assert result == time(0, 0, 0, 100000)

    def test_fraction_truncated(self) -> None:
        result = match_to_localtime(_localtime_match("00:00:00.123456789"))
        assert result == time(0, 0, 0, 123456)

    def test_midnight(self) -> None:
        assert match_to_localtime(_localtime_match("00:00")) == time(0, 0)

    def test_max(self) -> None:
        result = match_to_localtime(_localtime_match("23:59:59.999999"))
        assert result == time(23, 59, 59, 999999)


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


class TestMatchToNumber:
    def test_decimal_integer(self) -> None:
        assert match_to_number(_number_match("42"), float) == 42
        assert isinstance(match_to_number(_number_match("42"), float), int)

    def test_decimal_integer_zero(self) -> None:
        assert match_to_number(_number_match("0"), float) == 0

    def test_signed_integer(self) -> None:
        assert match_to_number(_number_match("-17"), float) == -17
        assert match_to_number(_number_match("+17"), float) == 17

    def test_underscore_integer(self) -> None:
        assert match_to_number(_number_match("1_000_000"), float) == 1000000

    def test_hex(self) -> None:
        assert match_to_number(_number_match("0xDEADbeef"), float) == 0xDEADBEEF

    def test_hex_with_underscores(self) -> None:
        assert match_to_number(_number_match("0xdead_beef"), float) == 0xDEADBEEF

    def test_binary(self) -> None:
        assert match_to_number(_number_match("0b101010"), float) == 0b101010

    def test_binary_with_underscores(self) -> None:
        assert match_to_number(_number_match("0b1010_1010"), float) == 0b10101010

    def test_octal(self) -> None:
        assert match_to_number(_number_match("0o755"), float) == 0o755

    def test_octal_with_underscores(self) -> None:
        assert match_to_number(_number_match("0o7_5_5"), float) == 0o755

    def test_float_uses_parse_float(self) -> None:
        seen: list[str] = []

        def parse_float(s: str) -> Any:
            seen.append(s)
            return float(s.replace("_", ""))

        result = match_to_number(_number_match("1.5"), parse_float)
        assert result == 1.5
        assert isinstance(result, float)
        assert seen == ["1.5"]

    def test_float_exponent(self) -> None:
        seen: list[str] = []

        def parse_float(s: str) -> Any:
            seen.append(s)
            return float(s.replace("_", ""))

        result = match_to_number(_number_match("1e10"), parse_float)
        assert result == 1e10
        assert seen == ["1e10"]

    def test_float_with_underscores(self) -> None:
        seen: list[str] = []

        def parse_float(s: str) -> Any:
            seen.append(s)
            return float(s.replace("_", ""))

        result = match_to_number(_number_match("1_000.5_5"), parse_float)
        assert result == 1000.55
        # Underscores are passed through to parse_float untouched
        assert seen == ["1_000.5_5"]

    def test_float_negative_sign(self) -> None:
        seen: list[str] = []

        def parse_float(s: str) -> Any:
            seen.append(s)
            return float(s)

        assert match_to_number(_number_match("-3.25"), parse_float) == -3.25
        assert seen == ["-3.25"]

    def test_integer_does_not_call_parse_float(self) -> None:
        def parse_float(s: str) -> Any:  # pragma: no cover
            raise AssertionError("parse_float should not be called for integers")

        assert match_to_number(_number_match("123"), parse_float) == 123

    def test_parse_float_return_preserved(self) -> None:
        sentinel = object()

        def parse_float(s: str) -> Any:
            return sentinel

        assert match_to_number(_number_match("0.5"), parse_float) is sentinel
