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


def _dt(s: str) -> datetime | date:
    match = RE_DATETIME.fullmatch(s)
    assert match is not None
    return match_to_datetime(match)


def test_datetime_date_only():
    assert _dt("1988-10-27") == date(1988, 10, 27)
    # Returned object is a date, not a datetime.
    result = _dt("1988-10-27")
    assert type(result) is date


@pytest.mark.parametrize(
    "s,expected",
    [
        ("1988-10-27T01:02:03", datetime(1988, 10, 27, 1, 2, 3)),
        ("1988-10-27t01:02:03", datetime(1988, 10, 27, 1, 2, 3)),
        ("1988-10-27 01:02:03", datetime(1988, 10, 27, 1, 2, 3)),
        ("1988-10-27T01:02", datetime(1988, 10, 27, 1, 2)),
        ("1988-10-27T23:59:59", datetime(1988, 10, 27, 23, 59, 59)),
        ("0000-01-01T00:00:00", datetime(0, 1, 1, 0, 0, 0)),
    ],
)
def test_datetime_naive(s, expected):
    result = _dt(s)
    assert result == expected
    assert result.tzinfo is None


def test_datetime_has_datetime_type():
    result = _dt("1988-10-27T01:02:03")
    assert type(result) is datetime


@pytest.mark.parametrize(
    "s,expected_micros",
    [
        ("1988-10-27T01:02:03.1", 100000),
        ("1988-10-27T01:02:03.12", 120000),
        ("1988-10-27T01:02:03.123", 123000),
        ("1988-10-27T01:02:03.1234", 123400),
        ("1988-10-27T01:02:03.12345", 123450),
        ("1988-10-27T01:02:03.123456", 123456),
        # More than 6 digits: extra digits are truncated.
        ("1988-10-27T01:02:03.1234567", 123456),
        ("1988-10-27T01:02:03.1" + "0" * 20, 100000),
    ],
)
def test_datetime_microseconds(s, expected_micros):
    assert _dt(s) == datetime(1988, 10, 27, 1, 2, 3, expected_micros)


def test_datetime_fraction_requires_seconds():
    # Fraction is only allowed when seconds are present.
    assert RE_DATETIME.fullmatch("1988-10-27T01:02.5") is None


@pytest.mark.parametrize(
    "s",
    [
        "1988-10-27T01:02:03Z",
        "1988-10-27T01:02:03z",
    ],
)
def test_datetime_zulu(s):
    result = _dt(s)
    assert result == datetime(1988, 10, 27, 1, 2, 3, tzinfo=timezone.utc)
    assert result.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "s,offset",
    [
        ("1988-10-27T01:02:03+00:00", timedelta(0)),
        ("1988-10-27T01:02:03+01:00", timedelta(hours=1)),
        ("1988-10-27T01:02:03-01:00", timedelta(hours=-1)),
        ("1988-10-27T01:02:03+01:30", timedelta(hours=1, minutes=30)),
        ("1988-10-27T01:02:03-01:30", timedelta(hours=-1, minutes=-30)),
        ("1988-10-27T01:02:03+23:59", timedelta(hours=23, minutes=59)),
        ("1988-10-27T01:02:03-23:59", timedelta(hours=-23, minutes=-59)),
        ("1988-10-27T01:02:03+00:30", timedelta(minutes=30)),
    ],
)
def test_datetime_offset(s, offset):
    result = _dt(s)
    assert isinstance(result, datetime)
    assert result.utcoffset() == offset
    # The naive components are unchanged; the offset is stored separately.
    assert result.replace(tzinfo=None) == datetime(1988, 10, 27, 1, 2, 3)


def test_datetime_offset_with_micros():
    result = _dt("1988-10-27T01:02:03.5+02:30")
    assert result == datetime(
        1988, 10, 27, 1, 2, 3, 500000, tzinfo=timezone(timedelta(hours=2, minutes=30))
    )


@pytest.mark.parametrize(
    "s",
    [
        "1988-13-27",  # bad month
        "1988-00-27",
        "1988-10-32",  # bad day
        "1988-10-00",
        "2021-02-30",  # not a real date
        "0000-01-01T24:00:00",  # bad hour
        "1988-10-27T25:00:00",
        "1988-10-27T01:60:00",  # bad minute
        "1988-10-27T01:00:60",  # bad second
        "1988-10-27T01:02:03+24:00",  # bad offset
        "1988-10-27T01:02:03+01:60",  # bad offset minute
    ],
)
def test_datetime_invalid_raises_value_error(s):
    match = RE_DATETIME.fullmatch(s)
    # For the offset cases the pattern matches but datetime construction fails;
    # for the others the pattern itself fails. Guard with a fullmatch check.
    if match is None:
        assert RE_DATETIME.fullmatch(s) is None
    else:
        with pytest.raises(ValueError):
            match_to_datetime(match)


def test_datetime_pattern_rejects_invalid_shapes():
    for s in [
        "1988-10-27T",
        "1988-10-27T1:02:03",
        "1988-1-27",
        "88-10-27",
        "1988-10-27T01:02:03+0100",
        "1988-10-27T01:02:03Z+01:00",
    ]:
        assert RE_DATETIME.fullmatch(s) is None, s


# ---------------------------------------------------------------------------
# cached_tz
# ---------------------------------------------------------------------------


def test_cached_tz_plus():
    tz = cached_tz("01", "30", "+")
    assert tz.utcoffset(None) == timedelta(hours=1, minutes=30)


def test_cached_tz_minus():
    tz = cached_tz("01", "30", "-")
    assert tz.utcoffset(None) == timedelta(hours=-1, minutes=-30)


def test_cached_tz_positive_sign_other_than_plus():
    # Anything not exactly "+" is treated as negative.
    tz = cached_tz("02", "00", "anything")
    assert tz.utcoffset(None) == timedelta(hours=-2)


def test_cached_tz_zero():
    tz = cached_tz("00", "00", "+")
    assert tz.utcoffset(None) == timedelta(0)
    assert tz == timezone.utc


def test_cached_tz_max():
    tz = cached_tz("23", "59", "+")
    assert tz.utcoffset(None) == timedelta(hours=23, minutes=59)


def test_cached_tz_cached():
    a = cached_tz("05", "15", "+")
    b = cached_tz("05", "15", "+")
    assert a is b


def test_cached_tz_cache_distinct_keys():
    a = cached_tz("05", "15", "+")
    b = cached_tz("05", "15", "-")
    c = cached_tz("05", "16", "+")
    assert a is not b
    assert a is not c
    assert b is not c


def test_cached_tz_cache_info():
    cached_tz.cache_clear()
    cached_tz("07", "00", "+")
    cached_tz("07", "00", "+")
    info = cached_tz.cache_info()
    assert info.misses == 1
    assert info.hits == 1
    assert info.maxsize is None


# ---------------------------------------------------------------------------
# match_to_localtime
# ---------------------------------------------------------------------------


def _lt(s: str) -> time:
    match = RE_LOCALTIME.fullmatch(s)
    assert match is not None
    return match_to_localtime(match)


@pytest.mark.parametrize(
    "s,expected",
    [
        ("00:00", time(0, 0)),
        ("23:59", time(23, 59)),
        ("01:02:03", time(1, 2, 3)),
        ("01:02:00", time(1, 2, 0)),
        ("00:00:00", time(0, 0, 0)),
    ],
)
def test_localtime_basic(s, expected):
    assert _lt(s) == expected


@pytest.mark.parametrize(
    "s,micros",
    [
        ("01:02:03.1", 100000),
        ("01:02:03.12", 120000),
        ("01:02:03.123", 123000),
        ("01:02:03.1234", 123400),
        ("01:02:03.12345", 123450),
        ("01:02:03.123456", 123456),
        ("01:02:03.1234567", 123456),  # truncated
        ("01:02:03.0", 0),
    ],
)
def test_localtime_microseconds(s, micros):
    assert _lt(s) == time(1, 2, 3, micros)


def test_localtime_fraction_requires_seconds():
    assert RE_LOCALTIME.fullmatch("01:02.5") is None


def test_localtime_type():
    result = _lt("01:02:03")
    assert type(result) is time
    assert result.tzinfo is None


@pytest.mark.parametrize(
    "s",
    [
        "24:00",
        "23:60",
        "1:02",
        "01:2",
        "01:02:60",
        "01:02:0",
        "01:02:03.",
        "01:02:03.12345678x",
        "0",
    ],
)
def test_localtime_invalid_shape(s):
    assert RE_LOCALTIME.fullmatch(s) is None


# ---------------------------------------------------------------------------
# match_to_number
# ---------------------------------------------------------------------------


def _num(s: str, parse_float=float):
    match = RE_NUMBER.fullmatch(s)
    assert match is not None
    return match_to_number(match, parse_float)


@pytest.mark.parametrize(
    "s,expected",
    [
        ("0", 0),
        ("1", 1),
        ("+1", 1),
        ("-1", -1),
        ("12345", 12345),
        ("1_000_000", 1000000),
        ("-1_000", -1000),
        ("+42", 42),
        ("0x0", 0),
        ("0x10", 16),
        ("0xDEADBEEF", 0xDEADBEEF),
        ("0xff", 0xFF),
        ("0x_ff", None),  # invalid - handled below separately
        ("0xDE_AD_BE_EF", 0xDEADBEEF),
        ("0b0", 0),
        ("0b1010", 0b1010),
        ("0b10_10", 0b1010),
        ("0o0", 0),
        ("0o755", 0o755),
        ("0o7_5_5", 0o755),
    ],
)
def test_number_integers(s, expected):
    if expected is None:
        # Placeholder invalid entries verified elsewhere.
        pytest.skip("invalid")
    result = _num(s)
    assert result == expected
    assert type(result) is int


@pytest.mark.parametrize(
    "s",
    [
        "0x_ff",  # underscore right after prefix not allowed
        "0x",
        "0b",
        "0o",
        "0xG",
        "00",
        "1__0",
        "1_",
        "_1",
    ],
)
def test_number_invalid_shape(s):
    assert RE_NUMBER.fullmatch(s) is None


def test_number_float_calls_parse_float():
    calls = []

    def parse_float(text):
        calls.append(text)
        return ("FLOAT", text)

    result = _num("1.5", parse_float)
    assert result == ("FLOAT", "1.5")
    assert calls == ["1.5"]


def test_number_float_int_part_only_no_float():
    # "1" has an empty (but matched) floatpart group -> int.
    result = _num("1", parse_float=lambda t: ("FLOAT", t))
    assert result == 1
    assert type(result) is int


@pytest.mark.parametrize(
    "s",
    [
        "1.5",
        "1e10",
        "1E10",
        "1e+10",
        "1e-10",
        "-1.5",
        "+1.5",
        "0.0",
        "1_0.5",
        "1.5_5",
        "1.5e1_0",
        "3.14159",
        "0e0",
    ],
)
def test_number_float_regex_matches(s):
    match = RE_NUMBER.fullmatch(s)
    assert match is not None
    # The floatpart group should be non-empty for these.
    assert match.group("floatpart")


def test_number_float_returns_parse_float_value():
    result = match_to_number(RE_NUMBER.fullmatch("2.5"), lambda t: t.upper())
    assert result == "2.5"


def test_number_float_with_dot_only_via_group():
    # Leading "0." without following digit should not match.
    assert RE_NUMBER.fullmatch("0.") is None
    assert RE_NUMBER.fullmatch(".5") is None


def test_number_float_exponent_no_digits():
    assert RE_NUMBER.fullmatch("1e") is None
    assert RE_NUMBER.fullmatch("1e+") is None


def test_number_no_floatpart_means_int():
    # A plain decimal integer has an empty floatpart group.
    match = RE_NUMBER.fullmatch("123")
    assert match is not None
    assert match.group("floatpart") == ""
    assert match_to_number(match, float) == 123


def test_number_invalid():
    for s in ["", "abc", "0xg", "1.2.3", "1e", "--1", "1_000_", "0x1.0"]:
        assert RE_NUMBER.fullmatch(s) is None, s
