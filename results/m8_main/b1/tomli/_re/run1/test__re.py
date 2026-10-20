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


# ---------------------------------------------------------------------------
# match_to_datetime via RE_DATETIME
# ---------------------------------------------------------------------------


def _dt(s: str) -> date | datetime:
    match = RE_DATETIME.fullmatch(s)
    assert match is not None, f"{s!r} did not match RE_DATETIME"
    return match_to_datetime(match)


def test_datetime_local_date_only() -> None:
    assert _dt("1988-10-27") == date(1988, 10, 27)


def test_datetime_local_date_only_leap_year() -> None:
    assert _dt("2000-02-29") == date(2000, 2, 29)


def test_datetime_invalid_date_raises() -> None:
    match = RE_DATETIME.fullmatch("2021-02-30")
    assert match is not None
    with pytest.raises(ValueError):
        match_to_datetime(match)


def test_datetime_local_naive() -> None:
    result = _dt("1979-05-27T07:32:00")
    assert result == datetime(1979, 5, 27, 7, 32, 0)
    assert isinstance(result, datetime)
    assert result.tzinfo is None


def test_datetime_space_separator() -> None:
    assert _dt("1979-05-27 07:32:00") == datetime(1979, 5, 27, 7, 32, 0)


def test_datetime_lowercase_t_separator() -> None:
    assert _dt("1979-05-27t07:32:00") == datetime(1979, 5, 27, 7, 32, 0)


def test_datetime_no_seconds() -> None:
    result = _dt("1979-05-27T07:32")
    assert result == datetime(1979, 5, 27, 7, 32, 0)


def test_datetime_with_seconds() -> None:
    assert _dt("1979-05-27T07:32:12") == datetime(1979, 5, 27, 7, 32, 12)


def test_datetime_fractional_seconds_padding() -> None:
    # Fractions are left-padded (via ljust) to microseconds.
    assert _dt("1979-05-27T00:32:00.1") == datetime(1979, 5, 27, 0, 32, 0, 100000)
    assert _dt("1979-05-27T00:32:00.12") == datetime(1979, 5, 27, 0, 32, 0, 120000)
    assert _dt("1979-05-27T00:32:00.999999") == datetime(
        1979, 5, 27, 0, 32, 0, 999999
    )


def test_datetime_extra_fraction_digits_truncated() -> None:
    # More than 6 fractional digits: extra digits are matched but not captured.
    assert _dt("1979-05-27T00:32:00.123456789") == datetime(
        1979, 5, 27, 0, 32, 0, 123456
    )


def test_datetime_zulu() -> None:
    result = _dt("1979-05-27T07:32:00Z")
    assert result == datetime(1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


def test_datetime_zulu_lowercase() -> None:
    assert _dt("1979-05-27T07:32:00z") == datetime(
        1979, 5, 27, 7, 32, 0, tzinfo=timezone.utc
    )


def test_datetime_positive_offset() -> None:
    result = _dt("1979-05-27T07:32:00+07:00")
    assert result.utcoffset() == timedelta(hours=7)
    assert result == datetime(
        1979, 5, 27, 7, 32, 0, tzinfo=timezone(timedelta(hours=7))
    )


def test_datetime_negative_offset() -> None:
    result = _dt("1979-05-27T07:32:00-07:30")
    assert result.utcoffset() == timedelta(hours=-7, minutes=-30)


def test_datetime_offset_positive_zero() -> None:
    result = _dt("1979-05-27T07:32:00+00:00")
    assert result.utcoffset() == timedelta(0)


def test_datetime_negative_zero_offset() -> None:
    # A "-00:00" is a valid offset sign with zero magnitude.
    result = _dt("1979-05-27T07:32:00-00:00")
    assert result.utcoffset() == timedelta(0)


def test_datetime_offset_with_fractional_seconds() -> None:
    result = _dt("1979-05-27T07:32:00.5+01:00")
    assert result.microsecond == 500000
    assert result.utcoffset() == timedelta(hours=1)


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


def test_cached_tz_positive() -> None:
    tz = cached_tz("07", "30", "+")
    assert isinstance(tz, timezone)
    assert tz.utcoffset(None) == timedelta(hours=7, minutes=30)


def test_cached_tz_negative() -> None:
    tz = cached_tz("07", "30", "-")
    assert tz.utcoffset(None) == timedelta(hours=-7, minutes=-30)


def test_cached_tz_zero() -> None:
    tz = cached_tz("00", "00", "+")
    assert tz.utcoffset(None) == timedelta(0)
    assert cached_tz("00", "00", "-").utcoffset(None) == timedelta(0)


def test_cached_tz_is_cached() -> None:
    first = cached_tz("05", "15", "+")
    second = cached_tz("05", "15", "+")
    assert first is second


def test_cached_tz_different_signs_not_equal() -> None:
    assert cached_tz("05", "00", "+") != cached_tz("05", "00", "-")


def test_cached_tz_sign_treated_as_not_supported_other() -> None:
    # Any sign other than "+" is treated as negative.
    tz = cached_tz("03", "00", "x")
    assert tz.utcoffset(None) == timedelta(hours=-3)


# ---------------------------------------------------------------------------
# match_to_localtime via RE_LOCALTIME
# ---------------------------------------------------------------------------


def _lt(s: str) -> time:
    match = RE_LOCALTIME.fullmatch(s)
    assert match is not None, f"{s!r} did not match RE_LOCALTIME"
    return match_to_localtime(match)


def test_localtime_hour_minute() -> None:
    result = _lt("07:32")
    assert result == time(7, 32)
    assert result.hour == 7
    assert result.minute == 32
    assert result.second == 0
    assert result.microsecond == 0
    assert result.tzinfo is None


def test_localtime_with_seconds() -> None:
    assert _lt("23:59:59") == time(23, 59, 59)


def test_localtime_midnight() -> None:
    assert _lt("00:00:00") == time(0, 0, 0)


def test_localtime_fractional_seconds_padding() -> None:
    assert _lt("00:00:00.1") == time(0, 0, 0, 100000)
    assert _lt("00:00:00.12") == time(0, 0, 0, 120000)
    assert _lt("00:00:00.123456") == time(0, 0, 0, 123456)


def test_localtime_extra_fraction_digits_truncated() -> None:
    assert _lt("00:00:00.987654321") == time(0, 0, 0, 987654)


def test_localtime_invalid_hour_does_not_match() -> None:
    assert RE_LOCALTIME.fullmatch("24:00") is None


def test_localtime_invalid_minute_does_not_match() -> None:
    assert RE_LOCALTIME.fullmatch("12:60") is None


# ---------------------------------------------------------------------------
# match_to_number via RE_NUMBER
# ---------------------------------------------------------------------------


def _num(s: str, parse_float=float):
    match = RE_NUMBER.fullmatch(s)
    assert match is not None, f"{s!r} did not match RE_NUMBER"
    return match_to_number(match, parse_float)


def test_number_decimal_integer() -> None:
    assert _num("42") == 42
    assert isinstance(_num("42"), int)


def test_number_zero() -> None:
    assert _num("0") == 0
    assert type(_num("0")) is int


def test_number_negative_integer() -> None:
    assert _num("-17") == -17


def test_number_positive_integer() -> None:
    assert _num("+17") == 17


def test_number_underscore_in_integer() -> None:
    assert _num("1_000") == 1000


def test_number_hex() -> None:
    assert _num("0xDEAD_BEEF") == 0xDEADBEEF


def test_number_binary() -> None:
    assert _num("0b1010_1010") == 0b10101010


def test_number_octal() -> None:
    assert _num("0o755") == 0o755


def test_number_leading_zero_invalid() -> None:
    # "01" must not match as a decimal integer (only "0" alone is allowed).
    assert RE_NUMBER.fullmatch("01") is None


def test_number_float_uses_parse_float() -> None:
    calls: list[str] = []

    def parse_float(text: str) -> str:
        calls.append(text)
        return "FLOAT"

    assert _num("1.5", parse_float) == "FLOAT"
    assert calls == ["1.5"]


def test_number_exponent_uses_parse_float() -> None:
    calls: list[str] = []

    def parse_float(text: str) -> str:
        calls.append(text)
        return "FLOAT"

    assert _num("1e10", parse_float) == "FLOAT"
    assert calls == ["1e10"]


def test_number_float_with_exponent_calls_parse_float_once() -> None:
    calls: list[str] = []

    def parse_float(text: str) -> str:
        calls.append(text)
        return "FLOAT"

    assert _num("6.67e-11", parse_float) == "FLOAT"
    assert calls == ["6.67e-11"]


def test_number_integer_does_not_call_parse_float() -> None:
    def parse_float(text: str) -> str:  # pragma: no cover - should not run
        raise AssertionError("parse_float should not be called for integers")

    assert _num("123", parse_float) == 123


def test_number_parse_float_receives_full_match_text() -> None:
    received: list[str] = []

    def parse_float(text: str) -> int:
        received.append(text)
        return -1

    _num("-3.14", parse_float)
    _num("+2e5", parse_float)
    assert received == ["-3.14", "+2e5"]


def test_number_float_zero_is_float_not_int() -> None:
    # 0.0 has a fractional part, so parse_float is used.
    assert _num("0.0") == 0.0
    assert isinstance(_num("0.0"), float)


def test_number_invalid_multiple_dots_not_full_match() -> None:
    assert RE_NUMBER.fullmatch("1.2.3") is None


def test_number_trailing_underscore_not_full_match() -> None:
    assert RE_NUMBER.fullmatch("1_") is None


def test_number_hex_uppercase_prefix_not_matched_by_number() -> None:
    assert RE_NUMBER.fullmatch("0XFF") is None
