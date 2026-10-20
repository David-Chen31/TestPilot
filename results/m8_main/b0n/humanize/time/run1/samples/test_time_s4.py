"""Tests for humanize.time."""

from __future__ import annotations

import datetime as dt

import pytest

from humanize.time import (
    Unit,
    naturaldate,
    naturalday,
    naturaldelta,
    naturaltime,
    precisedelta,
)


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_unit_ordering_and_values():
    assert Unit.MICROSECONDS.value == 0
    assert Unit.MILLISECONDS.value == 1
    assert Unit.SECONDS.value == 2
    assert Unit.MINUTES.value == 3
    assert Unit.HOURS.value == 4
    assert Unit.DAYS.value == 5
    assert Unit.MONTHS.value == 6
    assert Unit.YEARS.value == 7

    assert Unit.SECONDS < Unit.MINUTES
    assert Unit.DAYS > Unit.HOURS
    assert Unit.MINUTES <= Unit.MINUTES
    assert Unit.MINUTES >= Unit.MINUTES
    assert sorted([Unit.YEARS, Unit.SECONDS, Unit.DAYS]) == [
        Unit.SECONDS,
        Unit.DAYS,
        Unit.YEARS,
    ]


def test_unit_comparison_with_non_unit_is_not_implemented():
    with pytest.raises(TypeError):
        Unit.SECONDS < 5
    with pytest.raises(TypeError):
        Unit.SECONDS > "x"


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (dt.timedelta(microseconds=1), "a moment"),
        (dt.timedelta(microseconds=500), "a moment"),
        (dt.timedelta(milliseconds=1), "a moment"),
        (dt.timedelta(seconds=0), "a moment"),
        (dt.timedelta(seconds=1), "a second"),
        (dt.timedelta(seconds=2), "2 seconds"),
        (dt.timedelta(seconds=30), "30 seconds"),
        (dt.timedelta(seconds=60), "a minute"),
        (dt.timedelta(seconds=120), "2 minutes"),
        (dt.timedelta(minutes=30), "30 minutes"),
        (dt.timedelta(hours=1), "an hour"),
        (dt.timedelta(hours=2), "2 hours"),
        (dt.timedelta(hours=23), "23 hours"),
        (dt.timedelta(hours=24), "a day"),
        (dt.timedelta(days=1), "a day"),
        (dt.timedelta(days=2), "2 days"),
        (dt.timedelta(days=29), "29 days"),
        (dt.timedelta(days=31), "a month"),
        (dt.timedelta(days=61), "2 months"),
        (dt.timedelta(days=365), "a year"),
        (dt.timedelta(days=730), "2 years"),
    ],
)
def test_naturaldelta_basic(value, expected):
    assert naturaldelta(value) == expected


def test_naturaldelta_minimum_unit_microseconds():
    assert naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds") == (
        "500 microseconds"
    )
    assert naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds") == (
        "1 microsecond"
    )


def test_naturaldelta_minimum_unit_milliseconds():
    assert (
        naturaldelta(dt.timedelta(milliseconds=500), minimum_unit="milliseconds")
        == "500 milliseconds"
    )
    assert (
        naturaldelta(dt.timedelta(microseconds=500), minimum_unit="milliseconds")
        == "a moment"
    )


def test_naturaldelta_invalid_minimum_unit():
    with pytest.raises(ValueError, match="not supported"):
        naturaldelta(dt.timedelta(seconds=1), minimum_unit="minutes")


def test_naturaldelta_non_numeric_returns_str():
    assert naturaldelta("foo") == "foo"
    assert naturaldelta(float("nan")) == "nan"


def test_naturaldelta_infinite_returns_str():
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("-inf")) == "-inf"


def test_naturaldelta_too_large_raises_overflow():
    with pytest.raises(OverflowError):
        naturaldelta(1e300)


def test_naturaldelta_from_seconds():
    assert naturaldelta(30) == "30 seconds"
    assert naturaldelta(60) == "a minute"
    assert naturaldelta(3600) == "an hour"
    assert naturaldelta(86400) == "a day"
    assert naturaldelta(1.5) == "2 seconds"


def test_naturaldelta_negative_timedelta_abs():
    assert naturaldelta(dt.timedelta(seconds=-30)) == "30 seconds"
    assert naturaldelta(dt.timedelta(days=-2)) == "2 days"


def test_naturaldelta_years_with_months():
    # 366 days == 1 year + 1 day
    assert naturaldelta(dt.timedelta(days=366)) == "1 year, 1 day"
    # 365 + 31 = 396 days -> 1 year 1 month
    assert naturaldelta(dt.timedelta(days=396)) == "1 year, 1 month"
    # 365 + 61 = 426 days -> 1 year 2 months
    assert naturaldelta(dt.timedelta(days=426)) == "1 year, 2 months"


def test_naturaldelta_years_without_months():
    assert naturaldelta(dt.timedelta(days=396), months=False) == "1 year, 31 days"


def test_naturaldelta_large_years_uses_commas():
    result = naturaldelta(dt.timedelta(days=365 * 3000))
    assert result == "3,000 years"


def test_naturaldelta_month_boundary_to_year():
    # ~11 months rounds to 12 months -> "a year" for years == 0 branch
    delta = dt.timedelta(days=round(30.5 * 12) - 1)
    assert naturaldelta(delta) == "a year"


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


def test_naturaltime_seconds_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now - dt.timedelta(seconds=30)
    assert naturaltime(value, when=now) == "30 seconds ago"


def test_naturaltime_seconds_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now + dt.timedelta(seconds=30)
    assert naturaltime(value, when=now) == "30 seconds from now"


def test_naturaltime_now():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now, when=now) == "now"


def test_naturaltime_timedelta_uses_tense():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(minutes=5), when=now) == "5 minutes from now"
    assert naturaltime(dt.timedelta(minutes=5), when=now, future=True) == (
        "5 minutes from now"
    )


def test_naturaltime_number_default_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(30, when=now) == "30 seconds ago"
    assert naturaltime(30, future=True, when=now) == "30 seconds from now"


def test_naturaltime_non_convertible_returns_str():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime("not-a-number", when=now) == "not-a-number"


def test_naturaltime_minimum_unit():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now - dt.timedelta(microseconds=500)
    assert naturaltime(value, when=now, minimum_unit="microseconds") == (
        "500 microseconds ago"
    )


def test_naturaltime_aware_datetime():
    now_naive = dt.datetime(2020, 1, 1, 12, 0, 0)
    tz = dt.timezone.utc
    aware_now = now_naive.replace(tzinfo=tz)
    value = aware_now - dt.timedelta(minutes=5)
    # Aware is converted to naive local time; result should still be "5 minutes ago"
    assert naturaltime(value, when=aware_now) == "5 minutes ago"


def test_naturaltime_months_false():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now - dt.timedelta(days=400)
    assert "year" in naturaltime(value, when=now, months=False)


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


def test_naturalday_today_tomorrow_yesterday():
    today = dt.date.today()
    assert naturalday(today) == "today"
    assert naturalday(today + dt.timedelta(days=1)) == "tomorrow"
    assert naturalday(today - dt.timedelta(days=1)) == "yesterday"


def test_naturalday_other_days_uses_format():
    today = dt.date.today()
    future = today + dt.timedelta(days=10)
    assert naturalday(future) == future.strftime("%b %d")


def test_naturalday_custom_format():
    today = dt.date.today()
    future = today + dt.timedelta(days=10)
    assert naturalday(future, format="%Y-%m-%d") == future.strftime("%Y-%m-%d")


def test_naturalday_with_datetime():
    today = dt.datetime.now()
    assert naturalday(today) == "today"


def test_naturalday_non_date_returns_str():
    assert naturalday("hello") == "hello"
    assert naturalday(123) == "123"


def test_naturalday_aware_datetime():
    tz = dt.timezone.utc
    aware_today = dt.datetime.now(tz)
    assert naturalday(aware_today) == "today"


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


def test_naturaldate_recent_uses_no_year():
    today = dt.date.today()
    future = today + dt.timedelta(days=10)
    assert naturaldate(future) == naturalday(future)


def test_naturaldate_far_future_appends_year():
    today = dt.date.today()
    future = today + dt.timedelta(days=365)
    assert naturaldate(future) == future.strftime("%b %d %Y")


def test_naturaldate_far_past_appends_year():
    today = dt.date.today()
    past = today - dt.timedelta(days=365)
    assert naturaldate(past) == past.strftime("%b %d %Y")


def test_naturaldate_today():
    assert naturaldate(dt.date.today()) == "today"


def test_naturaldate_non_date_returns_str():
    assert naturaldate("hello") == "hello"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


def test_precisedelta_documented_example():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_custom_format():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"


def test_precisedelta_microseconds_unit():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, minimum_unit="microseconds") == (
        "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )


def test_precisedelta_suppress_days():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_suppress_seconds_below():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert (
        precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
        == "1.50 minutes"
    )


def test_precisedelta_small_delta_minutes():
    assert precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == "0.02 minutes"
    assert precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == "0 minutes"


def test_precisedelta_single_unit():
    assert precisedelta(dt.timedelta(seconds=5)) == "5.00 seconds"
    assert precisedelta(dt.timedelta(seconds=1)) == "1.00 seconds"


def test_precisedelta_zero():
    assert precisedelta(dt.timedelta(0)) == "0.00 seconds"


def test_precisedelta_from_number():
    assert precisedelta(90) == "1.50 minutes"


def test_precisedelta_negative_timedelta():
    assert precisedelta(dt.timedelta(seconds=-90)) == "1.50 minutes"


def test_precisedelta_non_convertible_returns_str():
    assert precisedelta("foo") == "foo"
    assert precisedelta(None) == "None"


def test_precisedelta_year_promotion():
    # 12 months exactly -> should promote to a year
    delta = dt.timedelta(days=365)
    result = precisedelta(delta, minimum_unit="days")
    assert result == "1.00 years"


def test_precisedelta_rounding_promotion():
    # 59.9 minutes should promote to one hour
    delta = dt.timedelta(minutes=59, seconds=54)
    result = precisedelta(delta, minimum_unit="minutes")
    assert result == "1.00 hours"


def test_precisedelta_large_years_uses_commas():
    delta = dt.timedelta(days=365 * 3000)
    result = precisedelta(delta, minimum_unit="years")
    assert result == "3,000.00 years"


def test_precisedelta_suppress_single_unit():
    delta = dt.timedelta(hours=1, minutes=30)
    assert precisedelta(delta, suppress=["minutes"]) == "1.50 hours"


def test_precisedelta_invalid_minimum_unit():
    with pytest.raises(KeyError):
        precisedelta(dt.timedelta(seconds=1), minimum_unit="centuries")


def test_precisedelta_invalid_suppress():
    with pytest.raises(KeyError):
        precisedelta(dt.timedelta(seconds=1), suppress=["centuries"])


def test_precisedelta_all_suppressed_below_minimum():
    # Suppressing the minimum unit forces a suitable replacement
    result = precisedelta(dt.timedelta(seconds=1), minimum_unit="seconds", suppress=["seconds"])
    assert "minute" in result
