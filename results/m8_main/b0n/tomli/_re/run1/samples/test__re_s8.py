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
# Helpers
# ---------------------------------------------------------------------------


def _fullmatch_datetime(s: str):
    m = RE_DATETIME.fullmatch(s)
    assert m is not None, f"{s!r} did not fully match RE_DATETIME"
    return m


def _fullmatch_localtime(s: str):
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None, f"{s!r} did not fully match RE_LOCALTIME"
    return m


def _fullmatch_number(s: str):
    m = RE_NUMBER.fullmatch(s)
    assert m is not None, f"{s!r} did not fully match RE_NUMBER"
    return m


def _parse_float_stub(s: str) -> float:
    return float(s.replace("_", ""))


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------


class TestMatchToDatetime:
    def test_date_only(self):
        m = _fullmatch_datetime("1988-10-27")
        result = match_to_datetime(m)
        assert result == date(1988, 10, 27)
        assert type(result) is date

    def test_date_only_single_digits_not_allowed(self):
        # Month/day must be zero-padded
        assert RE_DATETIME.fullmatch("1988-1-27") is None
        assert RE_DATETIME.fullmatch("1988-10-7") is None

    def test_datetime_with_T_separator(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0)
        assert type(result) is datetime
        assert result.tzinfo is None

    def test_datetime_with_lowercase_t_separator(self):
        m = _fullmatch_datetime("1979-05-27t07:32:00")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0)

    def test_datetime_with_space_separator(self):
        m = _fullmatch_datetime("1979-05-27 07:32:00")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0)

    def test_datetime_without_seconds(self):
        m = _fullmatch_datetime("1979-05-27T07:32")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0)

    def test_datetime_with_fractional_seconds(self):
        m = _fullmatch_datetime("1979-05-27T00:32:00.999999")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 0, 32, 0, 999999)

    def test_fractional_seconds_padded(self):
        m = _fullmatch_datetime("1979-05-27T00:32:00.5")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 0, 32, 0, 500000)

    def test_fractional_seconds_truncated_to_six(self):
        # Regex captures first 6 digits, rest ignored
        m = _fullmatch_datetime("1979-05-27T00:32:00.123456789")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 0, 32, 0, 123456)

    def test_fractional_seconds_without_seconds_rejected(self):
        # Fractions may only appear after seconds (per regex)
        assert RE_DATETIME.fullmatch("1979-05-27T07:32.5") is None

    def test_zulu_time(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00Z")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_lowercase_zulu_time(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00z")
        result = match_to_datetime(m)
        assert result == datetime(1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc)

    def test_positive_offset(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00+05:30")
        result = match_to_datetime(m)
        assert result.utcoffset() == timedelta(hours=5, minutes=30)
        assert result == datetime(
            1979, 5, 27, 7, 32, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )

    def test_negative_offset(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00-08:00")
        result = match_to_datetime(m)
        assert result.utcoffset() == timedelta(hours=-8)

    def test_negative_offset_with_minutes(self):
        m = _fullmatch_datetime("1979-05-27T07:32:00-00:30")
        result = match_to_datetime(m)
        assert result.utcoffset() == timedelta(minutes=-30)

    def test_offset_with_fractional_seconds(self):
        m = _fullmatch_datetime("1979-05-27T00:32:00.5+01:00")
        result = match_to_datetime(m)
        assert result == datetime(
            1979,
            5,
            27,
            0,
            32,
            0,
            500000,
            tzinfo=timezone(timedelta(hours=1)),
        )

    def test_offset_without_seconds(self):
        m = _fullmatch_datetime("1979-05-27T07:32+02:00")
        result = match_to_datetime(m)
        assert result.utcoffset() == timedelta(hours=2)
        assert result.second == 0

    def test_invalid_date_raises(self):
        m = _fullmatch_datetime("1979-02-30")
        with pytest.raises(ValueError):
            match_to_datetime(m)

    def test_invalid_month_raises(self):
        # Regex limits month to 01-12; use a month string the regex accepts
        # but is invalid for the day combos, e.g. Feb 31.
        m = _fullmatch_datetime("2020-02-31T00:00:00")
        with pytest.raises(ValueError):
            match_to_datetime(m)

    def test_month_out_of_range_not_matched(self):
        assert RE_DATETIME.fullmatch("1979-13-01") is None
        assert RE_DATETIME.fullmatch("1979-00-01") is None

    def test_day_out_of_range_not_matched(self):
        assert RE_DATETIME.fullmatch("1979-01-32") is None
        assert RE_DATETIME.fullmatch("1979-01-00") is None

    def test_hour_boundaries(self):
        assert RE_DATETIME.fullmatch("1979-05-27T00:00:00") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T23:59:59") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T24:00:00") is None
        assert RE_DATETIME.fullmatch("1979-05-27T-1:00:00") is None

    def test_minute_boundaries(self):
        assert RE_DATETIME.fullmatch("1979-05-27T00:00:00") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T00:59:00") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T00:60:00") is None

    def test_second_boundaries(self):
        assert RE_DATETIME.fullmatch("1979-05-27T00:00:00") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T00:00:59") is not None
        assert RE_DATETIME.fullmatch("1979-05-27T00:00:60") is None

    def test_leap_day_valid(self):
        m = _fullmatch_datetime("2020-02-29")
        assert match_to_datetime(m) == date(2020, 2, 29)

    def test_leap_day_invalid(self):
        m = _fullmatch_datetime("2019-02-29")
        with pytest.raises(ValueError):
            match_to_datetime(m)


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


class TestCachedTz:
    def test_positive_offset(self):
        tz = cached_tz("05", "30", "+")
        assert tz == timezone(timedelta(hours=5, minutes=30))
        assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)

    def test_negative_offset(self):
        tz = cached_tz("08", "00", "-")
        assert tz.utcoffset(None) == timedelta(hours=-8)

    def test_negative_offset_with_minutes(self):
        tz = cached_tz("03", "15", "-")
        assert tz.utcoffset(None) == timedelta(hours=-3, minutes=-15)

    def test_zero_offset(self):
        tz = cached_tz("00", "00", "+")
        assert tz.utcoffset(None) == timedelta(0)
        assert tz == timezone.utc

    def test_negative_zero_matches_utc(self):
        tz = cached_tz("00", "00", "-")
        assert tz.utcoffset(None) == timedelta(0)

    def test_cache_returns_identical_object(self):
        a = cached_tz("04", "45", "+")
        b = cached_tz("04", "45", "+")
        assert a is b

    def test_cache_distinguishes_signs(self):
        pos = cached_tz("01", "00", "+")
        neg = cached_tz("01", "00", "-")
        assert pos is not neg
        assert pos.utcoffset(None) == timedelta(hours=1)
        assert neg.utcoffset(None) == timedelta(hours=-1)

    def test_returns_timezone_instance(self):
        assert isinstance(cached_tz("01", "00", "+"), timezone)


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


class TestMatchToLocaltime:
    def test_hour_and_minute(self):
        m = _fullmatch_localtime("07:32")
        assert match_to_localtime(m) == time(7, 32, 0)

    def test_with_seconds(self):
        m = _fullmatch_localtime("07:32:15")
        assert match_to_localtime(m) == time(7, 32, 15)

    def test_with_fractional_seconds(self):
        m = _fullmatch_localtime("07:32:15.5")
        assert match_to_localtime(m) == time(7, 32, 15, 500000)

    def test_fractional_seconds_padded_to_micros(self):
        m = _fullmatch_localtime("00:00:00.123")
        assert match_to_localtime(m) == time(0, 0, 0, 123000)

    def test_fractional_seconds_extra_digits_truncated(self):
        m = _fullmatch_localtime("00:00:00.123456789")
        assert match_to_localtime(m) == time(0, 0, 0, 123456)

    def test_boundaries(self):
        m = _fullmatch_localtime("23:59:59.999999")
        assert match_to_localtime(m) == time(23, 59, 59, 999999)

    def test_zero(self):
        m = _fullmatch_localtime("00:00")
        assert match_to_localtime(m) == time(0, 0, 0)

    def test_no_timezone(self):
        m = _fullmatch_localtime("12:34:56")
        result = match_to_localtime(m)
        assert result.tzinfo is None

    def test_invalid_hour_not_matched(self):
        assert RE_LOCALTIME.fullmatch("24:00") is None
        assert RE_LOCALTIME.fullmatch("99:00") is None

    def test_invalid_minute_not_matched(self):
        assert RE_LOCALTIME.fullmatch("00:60") is None

    def test_invalid_second_not_matched(self):
        assert RE_LOCALTIME.fullmatch("00:00:60") is None

    def test_requires_hour_and_minute(self):
        assert RE_LOCALTIME.fullmatch("07") is None

    def test_fraction_without_seconds_not_matched(self):
        assert RE_LOCALTIME.fullmatch("07:32.5") is None


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


class TestMatchToNumber:
    @pytest.mark.parametrize(
        "src, expected",
        [
            ("0", 0),
            ("42", 42),
            ("+42", 42),
            ("-42", -42),
            ("1_000", 1000),
            ("+1_0", 10),
            ("-1_0", -10),
            ("0xDEADBEEF", 0xDEADBEEF),
            ("0xdead_beef", 0xDEADBEEF),
            ("0x0", 0),
            ("0o0", 0),
            ("0o755", 0o755),
            ("0o1_0", 0o10),
            ("0b0", 0),
            ("0b1101", 0b1101),
            ("0b1_0", 0b10),
        ],
    )
    def test_integers(self, src, expected):
        m = _fullmatch_number(src)
        assert match_to_number(m, _parse_float_stub) == expected

    @pytest.mark.parametrize(
        "src",
        [
            "1.0",
            "1.5",
            "-0.5",
            "+3.14",
            "1e10",
            "1E10",
            "1e+10",
            "1e-10",
            "-1.5e-3",
            "1_000.000_1",
            "6.626e-34",
            "0.0",
            "0e0",
        ],
    )
    def test_floats_delegate_to_parse_float(self, src):
        calls = []

        def parse_float(s: str) -> float:
            calls.append(s)
            return 42.0

        m = _fullmatch_number(src)
        result = match_to_number(m, parse_float)
        assert result == 42.0
        assert calls == [src]

    def test_float_uses_full_match_string(self):
        def parse_float(s: str) -> str:
            return s

        m = _fullmatch_number("+1_000.000_1")
        assert match_to_number(m, parse_float) == "+1_000.000_1"

    def test_float_with_real_float(self):
        m = _fullmatch_number("3.14")
        result = match_to_number(m, float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_integer_does_not_call_parse_float(self):
        def parse_float(s: str):
            raise AssertionError("parse_float should not be called for integers")

        m = _fullmatch_number("123")
        assert match_to_number(m, parse_float) == 123

    def test_hex_not_treated_as_float(self):
        def parse_float(s: str):
            raise AssertionError("parse_float should not be called for hex")

        m = _fullmatch_number("0xFF")
        assert match_to_number(m, parse_float) == 255

    def test_leading_zero_rejected_by_regex(self):
        assert RE_NUMBER.fullmatch("0123") is None

    def test_double_underscore_rejected(self):
        assert RE_NUMBER.fullmatch("1__0") is None
        assert RE_NUMBER.fullmatch("1_") is None

    def test_empty_fraction_rejected(self):
        assert RE_NUMBER.fullmatch("1.") is None

    def test_empty_exponent_rejected(self):
        assert RE_NUMBER.fullmatch("1e") is None
        assert RE_NUMBER.fullmatch("1e+") is None

    def test_hex_requires_digit(self):
        assert RE_NUMBER.fullmatch("0x") is None

    def test_local_partial_matching(self):
        # fullmatch ensures trailing garbage rejected
        assert RE_NUMBER.fullmatch("1abc") is None
        assert RE_NUMBER.fullmatch("0x1g") is None
