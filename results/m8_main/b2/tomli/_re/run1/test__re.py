# SPDX-License-Identifier: MIT
"""Unit tests for tomli._re."""

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dt_match(text: str):
    match = RE_DATETIME.fullmatch(text)
    assert match is not None, f"{text!r} did not match RE_DATETIME"
    return match


def _localtime_match(text: str):
    match = RE_LOCALTIME.fullmatch(text)
    assert match is not None, f"{text!r} did not match RE_LOCALTIME"
    return match


def _number_match(text: str):
    match = RE_NUMBER.fullmatch(text)
    assert match is not None, f"{text!r} did not match RE_NUMBER"
    return match


def _parse_float_float(text: str) -> float:
    return float(text.replace("_", ""))


# --------------------------------------------------------------------------- #
# cached_tz
# --------------------------------------------------------------------------- #
class TestCachedTz:
    def test_positive_offset(self):
        tz = cached_tz("05", "30", "+")
        assert isinstance(tz, timezone)
        assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)

    def test_negative_offset(self):
        tz = cached_tz("05", "30", "-")
        assert tz.utcoffset(None) == timedelta(hours=-5, minutes=-30)

    def test_zero_offset(self):
        tz = cached_tz("00", "00", "+")
        assert tz.utcoffset(None) == timedelta(0)
        assert tz is timezone.utc or tz.utcoffset(None) == timedelta(0)

    def test_negative_zero_offset(self):
        tz = cached_tz("00", "00", "-")
        assert tz.utcoffset(None) == timedelta(0)

    def test_max_positive_offset(self):
        tz = cached_tz("23", "59", "+")
        assert tz.utcoffset(None) == timedelta(hours=23, minutes=59)

    def test_max_negative_offset(self):
        tz = cached_tz("23", "59", "-")
        assert tz.utcoffset(None) == timedelta(hours=-23, minutes=-59)

    def test_caching_returns_identical_object(self):
        first = cached_tz("03", "15", "+")
        second = cached_tz("03", "15", "+")
        assert first is second

    def test_cache_exported(self):
        # lru_cache exposes cache_info; sanity-check cache is working.
        assert hasattr(cached_tz, "cache_info")
        before = cached_tz.cache_info()
        cached_tz("11", "11", "-")
        cached_tz("11", "11", "-")
        after = cached_tz.cache_info()
        assert after.hits >= before.hits + 1

    def test_sign_other_than_plus_is_negative(self):
        # Any sign string that is not "+" is treated as negative.
        tz = cached_tz("01", "00", "?")
        assert tz.utcoffset(None) == timedelta(hours=-1)


# --------------------------------------------------------------------------- #
# match_to_datetime
# --------------------------------------------------------------------------- #
class TestMatchToDatetime:
    def test_date_only(self):
        result = match_to_datetime(_dt_match("1988-10-27"))
        assert result == date(1988, 10, 27)
        assert type(result) is date

    def test_datetime_no_offset(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03.123"))
        assert result == datetime(1988, 10, 27, 1, 2, 3, 123000)
        assert result.tzinfo is None

    def test_datetime_space_separator(self):
        result = match_to_datetime(_dt_match("1988-10-27 01:02:03"))
        assert result == datetime(1988, 10, 27, 1, 2, 3)

    def test_datetime_lowercase_t(self):
        result = match_to_datetime(_dt_match("1988-10-27t01:02:03"))
        assert result == datetime(1988, 10, 27, 1, 2, 3)

    def test_time_no_seconds(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02"))
        assert result == datetime(1988, 10, 27, 1, 2, 0)

    def test_time_with_seconds_no_fraction(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03"))
        assert result == datetime(1988, 10, 27, 1, 2, 3)

    @pytest.mark.parametrize(
        "frac, expected_micros",
        [
            ("1", 100000),
            ("12", 120000),
            ("123", 123000),
            ("1234", 123400),
            ("12345", 123450),
            ("123456", 123456),
            ("1234567", 123456),  # extra digits truncated
            ("0000001", 0),
            ("000001", 1),
        ],
    )
    def test_fractional_seconds(self, frac, expected_micros):
        result = match_to_datetime(_dt_match(f"1988-10-27T01:02:03.{frac}"))
        assert result.microsecond == expected_micros

    def test_zulu_uppercase(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03Z"))
        assert result == datetime(1988, 10, 27, 1, 2, 3, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_zulu_lowercase(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03z"))
        assert result == datetime(1988, 10, 27, 1, 2, 3, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_positive_offset(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03+05:30"))
        assert result == datetime(
            1988, 10, 27, 1, 2, 3, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )
        assert result.utcoffset() == timedelta(hours=5, minutes=30)

    def test_negative_offset(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03-05:30"))
        assert result.utcoffset() == timedelta(hours=-5, minutes=-30)

    def test_offset_uses_cache(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03+05:30"))
        expected_tz = cached_tz("05", "30", "+")
        assert result.tzinfo is expected_tz

    def test_zero_offset(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03+00:00"))
        assert result.utcoffset() == timedelta(0)

    def test_offset_without_seconds(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02+01:00"))
        assert result == datetime(
            1988, 10, 27, 1, 2, 0, tzinfo=timezone(timedelta(hours=1))
        )

    def test_offset_with_fraction(self):
        result = match_to_datetime(_dt_match("1988-10-27T01:02:03.5+01:00"))
        assert result.microsecond == 500000
        assert result.utcoffset() == timedelta(hours=1)

    def test_offset_space_separator(self):
        result = match_to_datetime(_dt_match("1988-10-27 01:02:03Z"))
        assert result == datetime(1988, 10, 27, 1, 2, 3, tzinfo=timezone.utc)

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            match_to_datetime(_dt_match("1988-02-30"))

    def test_invalid_date_non_leap(self):
        with pytest.raises(ValueError):
            match_to_datetime(_dt_match("1900-02-29"))

    def test_valid_leap_day(self):
        result = match_to_datetime(_dt_match("2000-02-29"))
        assert result == date(2000, 2, 29)

    def test_invalid_time_hour(self):
        # RE_DATETIME won't allow hour 24, so no match.
        assert RE_DATETIME.fullmatch("1988-10-27T24:00:00") is None

    def test_invalid_month_no_match(self):
        assert RE_DATETIME.fullmatch("1988-13-01") is None

    def test_invalid_day_no_match(self):
        assert RE_DATETIME.fullmatch("1988-01-32") is None

    def test_invalid_minute_no_match(self):
        assert RE_DATETIME.fullmatch("1988-01-01T00:60:00") is None

    def test_invalid_second_no_match(self):
        assert RE_DATETIME.fullmatch("1988-01-01T00:00:60") is None

    def test_offset_hour_out_of_range_no_match(self):
        assert RE_DATETIME.fullmatch("1988-01-01T00:00:00+24:00") is None

    def test_day_zero_no_match(self):
        # The regex requires day to start with 0 followed by 1-9 (or 1x/2x/3x),
        # so "00" and "0" do not match.
        assert RE_DATETIME.fullmatch("1988-01-00") is None

    def test_month_zero_no_match(self):
        # Month pattern is 0[1-9]|1[0-2], so 00 does not match.
        assert RE_DATETIME.fullmatch("1988-00-01") is None

    def test_date_lowercase_boundaries(self):
        # Year can be any 4 digits, including leading zeros.
        result = match_to_datetime(_dt_match("0001-01-01"))
        assert result == date(1, 1, 1)


# --------------------------------------------------------------------------- #
# match_to_localtime
# --------------------------------------------------------------------------- #
class TestMatchToLocaltime:
    def test_hour_minute(self):
        result = match_to_localtime(_localtime_match("01:02"))
        assert result == time(1, 2)
        assert type(result) is time

    def test_with_seconds(self):
        result = match_to_localtime(_localtime_match("01:02:03"))
        assert result == time(1, 2, 3)

    @pytest.mark.parametrize(
        "frac, expected_micros",
        [
            ("1", 100000),
            ("12", 120000),
            ("123", 123000),
            ("123456", 123456),
            ("1".ljust(7, "0"), 100000),
        ],
    )
    def test_fractional_seconds(self, frac, expected_micros):
        result = match_to_localtime(_localtime_match(f"01:02:03.{frac}"))
        assert result.microsecond == expected_micros

    def test_extra_fraction_digits_truncated(self):
        result = match_to_localtime(_localtime_match("01:02:03.1234567"))
        assert result.microsecond == 123456

    def test_midnight(self):
        assert match_to_localtime(_localtime_match("00:00")) == time(0, 0)

    def test_last_minute(self):
        assert match_to_localtime(_localtime_match("23:59")) == time(23, 59)

    def test_no_match_bad_hour(self):
        assert RE_LOCALTIME.fullmatch("24:00") is None

    def test_no_match_bad_minute(self):
        assert RE_LOCALTIME.fullmatch("00:60") is None

    def test_no_match_missing_colon(self):
        assert RE_LOCALTIME.fullmatch("0102") is None

    def test_no_match_bad_second(self):
        assert RE_LOCALTIME.fullmatch("00:00:60") is None


# --------------------------------------------------------------------------- #
# match_to_number
# --------------------------------------------------------------------------- #
class TestMatchToNumber:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0", 0),
            ("1", 1),
            ("42", 42),
            ("+42", 42),
            ("-42", -42),
            ("1_000", 1000),
            ("1_0_0", 100),
            ("123_456_789", 123456789),
        ],
    )
    def test_integers(self, text, expected):
        result = match_to_number(_number_match(text), _parse_float_float)
        assert result == expected
        assert type(result) is int

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0x0", 0),
            ("0xDEADBEEF", 0xDEADBEEF),
            ("0xdeadbeef", 0xDEADBEEF),
            ("0x_1".replace("_", ""), 1),
            ("0xdead_beef", 0xDEADBEEF),
            ("0xAA_BB", 0xAABB),
        ],
    )
    def test_hex(self, text, expected):
        result = match_to_number(_number_match(text), _parse_float_float)
        assert result == expected
        assert type(result) is int

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0b0", 0),
            ("0b1", 1),
            ("0b1010", 0b1010),
            ("0b1010_0101", 0b10100101),
        ],
    )
    def test_binary(self, text, expected):
        result = match_to_number(_number_match(text), _parse_float_float)
        assert result == expected
        assert type(result) is int

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0o0", 0),
            ("0o7", 7),
            ("0o17", 0o17),
            ("0o12_34", 0o1234),
        ],
    )
    def test_octal(self, text, expected):
        result = match_to_number(_number_match(text), _parse_float_float)
        assert result == expected
        assert type(result) is int

    @pytest.mark.parametrize(
        "text",
        ["1.5", "1.0", "-1.5", "+1.5", "1e10", "1E10", "1e+10", "1e-10", "1.5e10"],
    )
    def test_floats_via_parse_float(self, text):
        calls = []

        def parse_float(s: str) -> float:
            calls.append(s)
            return float(s.replace("_", ""))

        result = match_to_number(_number_match(text), parse_float)
        assert calls == [text]
        assert isinstance(result, float)
        assert result == float(text.replace("_", ""))

    def test_float_with_underscores(self):
        result = match_to_number(_number_match("1_0.5"), _parse_float_float)
        assert result == 10.5

    def test_float_part_with_integer_underscores_uses_parse_float(self):
        calls = []

        def parse_float(s: str) -> float:
            calls.append(s)
            return 0.0

        result = match_to_number(_number_match("1_0"), parse_float)
        assert calls == []  # integer part only -> int path
        assert result == 10

    def test_parse_float_not_called_for_int(self):
        def parse_float(s: str):
            raise AssertionError("parse_float should not be called for ints")

        assert match_to_number(_number_match("42"), parse_float) == 42
        assert match_to_number(_number_match("0xFF"), parse_float) == 0xFF

    def test_returns_value_from_parse_float_verbatim(self):
        sentinel = object()

        def parse_float(s: str):
            return sentinel

        result = match_to_number(_number_match("1.5"), parse_float)
        assert result is sentinel

    def test_no_match_empty(self):
        assert RE_NUMBER.fullmatch("") is None

    def test_no_match_bare_leading_zeros(self):
        # Leading zeros for decimal integers are not allowed.
        assert RE_NUMBER.fullmatch("01") is None

    def test_no_match_plus_hex(self):
        assert RE_NUMBER.fullmatch("+0x1") is None

    def test_no_match_double_underscore(self):
        assert RE_NUMBER.fullmatch("1__0") is None

    def test_no_match_trailing_underscore(self):
        assert RE_NUMBER.fullmatch("1_") is None

    def test_no_match_dot_without_digit(self):
        assert RE_NUMBER.fullmatch("1.") is None

    def test_no_match_hex_without_digit(self):
        assert RE_NUMBER.fullmatch("0x") is None
