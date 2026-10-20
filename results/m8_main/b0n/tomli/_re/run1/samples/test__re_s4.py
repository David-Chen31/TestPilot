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


def _match_datetime(s: str):
    m = RE_DATETIME.fullmatch(s)
    assert m is not None, f"RE_DATETIME did not fully match {s!r}"
    return m


def _match_localtime(s: str):
    m = RE_LOCALTIME.fullmatch(s)
    assert m is not None, f"RE_LOCALTIME did not fully match {s!r}"
    return m


def _match_number(s: str):
    m = RE_NUMBER.fullmatch(s)
    assert m is not None, f"RE_NUMBER did not fully match {s!r}"
    return m


# ---------------------------------------------------------------------------
# match_to_datetime
# ---------------------------------------------------------------------------


class TestMatchToDatetime:
    def test_date_only(self):
        result = match_to_datetime(_match_datetime("1979-05-27"))
        assert result == date(1979, 5, 27)
        assert type(result) is date

    def test_date_only_leading_zero(self):
        result = match_to_datetime(_match_datetime("0001-01-01"))
        assert result == date(1, 1, 1)
        assert type(result) is date

    @pytest.mark.parametrize("sep", ["T", "t", " "])
    def test_datetime_naive_all_separators(self, sep):
        result = match_to_datetime(_match_datetime(f"1979-05-27{sep}07:32:00"))
        assert result == datetime(1979, 5, 27, 7, 32, 0)
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_datetime_no_seconds(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32"))
        assert result == datetime(1979, 5, 27, 7, 32, 0)
        assert result.tzinfo is None

    def test_datetime_seconds_no_fraction(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32:59"))
        assert result == datetime(1979, 5, 27, 7, 32, 59)

    @pytest.mark.parametrize(
        "frac,expected_micro",
        [
            ("1", 100000),
            ("12", 120000),
            ("123", 123000),
            ("1234", 123400),
            ("12345", 123450),
            ("123456", 123456),
        ],
    )
    def test_datetime_fraction_padding(self, frac, expected_micro):
        result = match_to_datetime(_match_datetime(f"1979-05-27T07:32:00.{frac}"))
        assert result == datetime(1979, 5, 27, 7, 32, 0, expected_micro)

    def test_datetime_fraction_truncation(self):
        # More than 6 digits are matched but truncated to microsecond precision.
        result = match_to_datetime(_match_datetime("1979-05-27T07:32:00.123456789"))
        assert result == datetime(1979, 5, 27, 7, 32, 0, 123456)

    @pytest.mark.parametrize("z", ["Z", "z"])
    def test_datetime_zulu(self, z):
        result = match_to_datetime(_match_datetime(f"1979-05-27T07:32:00{z}"))
        assert result == datetime(1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc)
        assert result.tzinfo is timezone.utc

    def test_datetime_offset_positive(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32:00+07:30"))
        assert result == datetime(
            1979, 5, 27, 7, 32, 0, tzinfo=timezone(timedelta(hours=7, minutes=30))
        )
        assert result.utcoffset() == timedelta(hours=7, minutes=30)

    def test_datetime_offset_negative(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32:00-08:00"))
        assert result == datetime(
            1979, 5, 27, 7, 32, 0, tzinfo=timezone(timedelta(hours=-8))
        )
        assert result.utcoffset() == timedelta(hours=-8)

    def test_datetime_offset_zero(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32:00+00:00"))
        assert result.utcoffset() == timedelta(0)

    def test_offset_without_seconds(self):
        result = match_to_datetime(_match_datetime("1979-05-27T07:32+01:00"))
        assert result == datetime(
            1979, 5, 27, 7, 32, 0, tzinfo=timezone(timedelta(hours=1))
        )

    def test_invalid_date_raises(self):
        # Feb 30 matches the regex but is not a valid date.
        with pytest.raises(ValueError):
            match_to_datetime(_match_datetime("1979-02-30"))

    def test_invalid_time_raises(self):
        # Regex ensures hour/minute ranges, but day=31 in April would fail.
        with pytest.raises(ValueError):
            match_to_datetime(_match_datetime("1979-04-31T00:00:00"))

    def test_returns_date_type_for_date_only(self):
        result = match_to_datetime(_match_datetime("2020-12-31"))
        assert type(result) is date


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


class TestCachedTz:
    def test_positive(self):
        tz = cached_tz("05", "30", "+")
        assert tz == timezone(timedelta(hours=5, minutes=30))

    def test_negative(self):
        tz = cached_tz("05", "30", "-")
        assert tz == timezone(-timedelta(hours=5, minutes=30))

    def test_zero(self):
        tz = cached_tz("00", "00", "+")
        assert tz == timezone(timedelta(0))

    def test_negative_zero(self):
        tz = cached_tz("00", "00", "-")
        assert tz.utcoffset(None) == timedelta(0)

    def test_boundary_hours(self):
        assert cached_tz("23", "59", "+").utcoffset(None) == timedelta(
            hours=23, minutes=59
        )
        assert cached_tz("23", "59", "-").utcoffset(None) == -timedelta(
            hours=23, minutes=59
        )

    def test_lru_cache_identity(self):
        tz1 = cached_tz("10", "15", "+")
        tz2 = cached_tz("10", "15", "+")
        assert tz1 is tz2

    def test_lru_cache_distinct_signs(self):
        pos = cached_tz("01", "00", "+")
        neg = cached_tz("01", "00", "-")
        assert pos is not neg
        assert pos.utcoffset(None) == -neg.utcoffset(None)

    def test_cache_info_exists(self):
        # lru_cache wrapper should be present.
        assert hasattr(cached_tz, "cache_info")


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


class TestMatchToLocaltime:
    def test_hour_minute_only(self):
        result = match_to_localtime(_match_localtime("07:32"))
        assert result == time(7, 32, 0, 0)
        assert result.tzinfo is None

    def test_with_seconds(self):
        result = match_to_localtime(_match_localtime("23:59:59"))
        assert result == time(23, 59, 59)

    def test_seconds_zero(self):
        result = match_to_localtime(_match_localtime("00:00:00"))
        assert result == time(0, 0, 0)

    @pytest.mark.parametrize(
        "frac,expected_micro",
        [
            ("1", 100000),
            ("12", 120000),
            ("123", 123000),
            ("1234", 123400),
            ("12345", 123450),
            ("123456", 123456),
        ],
    )
    def test_fraction_padding(self, frac, expected_micro):
        result = match_to_localtime(_match_localtime(f"07:32:00.{frac}"))
        assert result == time(7, 32, 0, expected_micro)

    def test_fraction_truncation(self):
        result = match_to_localtime(_match_localtime("07:32:00.987654321"))
        assert result == time(7, 32, 0, 987654)

    def test_returns_time_type(self):
        result = match_to_localtime(_match_localtime("12:34:56.789"))
        assert type(result) is time
        assert result == time(12, 34, 56, 789000)


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


class _RecordingParseFloat:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, s: str) -> float:
        self.calls.append(s)
        return float(s)


class TestMatchToNumber:
    def test_integer_decimal(self):
        result = match_to_number(_match_number("42"), float)
        assert result == 42
        assert type(result) is int

    def test_integer_signed(self):
        assert match_to_number(_match_number("+42"), float) == 42
        assert match_to_number(_match_number("-42"), float) == -42

    def test_integer_zero(self):
        assert match_to_number(_match_number("0"), float) == 0
        assert match_to_number(_match_number("-0"), float) == 0

    def test_integer_underscores(self):
        assert match_to_number(_match_number("1_000_000"), float) == 1000000

    def test_hex(self):
        assert match_to_number(_match_number("0xDEADBEEF"), float) == 0xDEADBEEF
        assert match_to_number(_match_number("0xff"), float) == 255

    def test_bin(self):
        assert match_to_number(_match_number("0b1010"), float) == 0b1010

    def test_oct(self):
        assert match_to_number(_match_number("0o755"), float) == 0o755

    def test_hex_underscores(self):
        assert match_to_number(_match_number("0xdead_beef"), float) == 0xDEADBEEF

    def test_float_no_parse_float_call_for_int(self):
        # parse_float must not be called for integer input.
        parse_float = _RecordingParseFloat()
        result = match_to_number(_match_number("123"), parse_float)
        assert result == 123
        assert parse_float.calls == []

    def test_float_calls_parse_float(self):
        parse_float = _RecordingParseFloat()
        result = match_to_number(_match_number("1.5"), parse_float)
        assert result == 1.5
        assert parse_float.calls == ["1.5"]

    def test_float_underscores_passed_raw(self):
        parse_float = _RecordingParseFloat()
        # The raw matched string (with underscores) is passed to parse_float.
        result = match_to_number(_match_number("1_000.5"), parse_float)
        assert result == 1000.5
        assert parse_float.calls == ["1_000.5"]

    def test_exponent_only_triggers_parse_float(self):
        parse_float = _RecordingParseFloat()
        result = match_to_number(_match_number("1e10"), parse_float)
        assert result == 1e10
        assert parse_float.calls == ["1e10"]

    def test_float_with_exponent(self):
        parse_float = _RecordingParseFloat()
        result = match_to_number(_match_number("-2.5E-3"), parse_float)
        assert result == -2.5e-3
        assert parse_float.calls == ["-2.5E-3"]

    def test_custom_parse_float_return_value(self):
        marker = object()

        def parse_float(s: str):
            return marker

        result = match_to_number(_match_number("3.14"), parse_float)
        assert result is marker

    def test_float_leading_dot_required_digit(self):
        # A bare "." after integer part is not a float part per regex; verify
        # that "1." does not match fully.
        assert RE_NUMBER.fullmatch("1.") is None

    def test_float_sign_and_fraction(self):
        parse_float = _RecordingParseFloat()
        result = match_to_number(_match_number("+0.0"), parse_float)
        assert result == 0.0
        assert parse_float.calls == ["+0.0"]
