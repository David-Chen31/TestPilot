"""Tests for humanize.time."""

from __future__ import annotations

import datetime as dt
import math

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


def test_unit_ordering():
    assert Unit.MICROSECONDS < Unit.SECONDS
    assert Unit.SECONDS < Unit.YEARS
    assert Unit.YEARS > Unit.HOURS
    assert Unit.DAYS >= Unit.DAYS
    assert Unit.DAYS <= Unit.DAYS


def test_unit_values():
    assert Unit.MICROSECONDS.value == 0
    assert Unit.MILLISECONDS.value == 1
    assert Unit.SECONDS.value == 2
    assert Unit.MINUTES.value == 3
    assert Unit.HOURS.value == 4
    assert Unit.DAYS.value == 5
    assert Unit.MONTHS.value == 6
    assert Unit.YEARS.value == 7


def test_unit_lt_with_non_unit_returns_not_implemented():
    with pytest.raises(TypeError):
        Unit.SECONDS < 5


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


def test_naturaldelta_seconds():
    assert naturaldelta(dt.timedelta(seconds=5)) == "5 seconds"


def test_naturaldelta_one_second():
    assert naturaldelta(dt.timedelta(seconds=1)) == "a second"


def test_naturaldelta_moment():
    assert naturaldelta(dt.timedelta(0)) == "a moment"


def test_naturaldelta_minutes():
    assert naturaldelta(dt.timedelta(minutes=1)) == "a minute"
    assert naturaldelta(dt.timedelta(minutes=30)) == "30 minutes"


def test_naturaldelta_minutes_round_to_hour():
    assert naturaldelta(dt.timedelta(seconds=59 * 60 + 30)) == "an hour"


def test_naturaldelta_hours():
    assert naturaldelta(dt.timedelta(hours=1)) == "an hour"
    assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"


def test_naturaldelta_hours_round_to_day():
    assert naturaldelta(dt.timedelta(seconds=round(23.6 * 3600))) == "a day"


def test_naturaldelta_days():
    assert naturaldelta(dt.timedelta(days=1)) == "a day"
    assert naturaldelta(dt.timedelta(days=5)) == "5 days"


def test_naturaldelta_days_no_months():
    assert naturaldelta(dt.timedelta(days=40), months=False) == "40 days"
    assert naturaldelta(dt.timedelta(days=2), months=False) == "2 days"


def test_naturaldelta_months():
    assert naturaldelta(dt.timedelta(days=31)) == "a month"
    assert naturaldelta(dt.timedelta(days=61)) == "2 months"


def test_naturaldelta_rounds_to_year():
    # 360 days -> round(360/30.5)=12 months -> "a year"
    assert naturaldelta(dt.timedelta(days=360)) == "a year"


def test_naturaldelta_days_rounded_zero_months():
    # 15 days -> round(15/30.5) = 0 -> falls back to days
    assert naturaldelta(dt.timedelta(days=15)) == "15 days"


def test_naturaldelta_one_year_exact():
    assert naturaldelta(dt.timedelta(days=365)) == "a year"


def test_naturaldelta_one_year_and_days_no_months():
    assert (
        naturaldelta(dt.timedelta(days=365 + 3), months=False) == "1 year, 3 days"
    )


def test_naturaldelta_one_year_and_one_day():
    # 366 days: years=1, days=1 => num_months=0, days=1
    assert naturaldelta(dt.timedelta(days=366)) == "1 year, 1 day"


def test_naturaldelta_one_year_and_one_month():
    # years=1, days~31 -> num_months=1
    assert naturaldelta(dt.timedelta(days=365 + 31)) == "1 year, 1 month"


def test_naturaldelta_one_year_and_months():
    assert naturaldelta(dt.timedelta(days=365 + 62)) == "1 year, 2 months"


def test_naturaldelta_one_year_twelve_months_promoted():
    # years=1, days=365ish would not reach 12 months; craft 1 year + 360 days
    # days % 365 = 360 -> num_months = round(360/30.5) = 12
    assert naturaldelta(dt.timedelta(days=365 + 360)) == "2 years"


def test_naturaldelta_multiple_years():
    assert naturaldelta(dt.timedelta(days=730)) == "2 years"


def test_naturaldelta_years_with_comma():
    assert naturaldelta(dt.timedelta(days=365 * 1234)) == "1,234 years"


def test_naturaldelta_from_number():
    assert naturaldelta(30) == "30 seconds"


def test_naturaldelta_from_float():
    assert naturaldelta(30.0) == "30 seconds"


def test_naturaldelta_negative_timedelta():
    assert naturaldelta(dt.timedelta(seconds=-5)) == "5 seconds"


def test_naturaldelta_nan_returns_string():
    assert naturaldelta(float("nan")) == "nan"


def test_naturaldelta_inf_returns_string():
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("-inf")) == "-inf"


def test_naturaldelta_non_numeric_string_returns_string():
    assert naturaldelta("abc") == "abc"


def test_naturaldelta_overflow_raises():
    with pytest.raises(OverflowError):
        naturaldelta(1e20)


def test_naturaldelta_invalid_minimum_unit():
    with pytest.raises(ValueError):
        naturaldelta(dt.timedelta(seconds=1), minimum_unit="days")


def test_naturaldelta_milliseconds():
    assert (
        naturaldelta(dt.timedelta(milliseconds=500), minimum_unit="milliseconds")
        == "500 milliseconds"
    )


def test_naturaldelta_microseconds():
    assert (
        naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
        == "500 microseconds"
    )


def test_naturaldelta_microseconds_promotes_to_milliseconds():
    assert (
        naturaldelta(dt.timedelta(microseconds=5000), minimum_unit="microseconds")
        == "5 milliseconds"
    )


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


def test_naturaltime_now():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now, when=now) == "now"


def test_naturaltime_past_datetime():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now - dt.timedelta(days=5)
    assert naturaltime(value, when=now) == "5 days ago"


def test_naturaltime_future_datetime():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    value = now + dt.timedelta(days=5)
    assert naturaltime(value, when=now) == "5 days from now"


def test_naturaltime_timedelta_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(hours=3), when=now) == "3 hours ago"


def test_naturaltime_timedelta_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(hours=-3), when=now) == "3 hours ago"


def test_naturaltime_number_past_default():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(30, when=now) == "30 seconds ago"


def test_naturaltime_number_future_flag():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(30, future=True, when=now) == "30 seconds from now"


def test_naturaltime_invalid_value_returns_string():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime("abc", when=now) == "abc"


def test_naturaltime_aware_datetime():
    tz = dt.timezone.utc
    now = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
    value = now - dt.timedelta(hours=2)
    assert naturaltime(value, when=now) == "2 hours ago"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


def _today():
    return dt.date.today()


def test_naturalday_today():
    assert naturalday(_today()) == "today"


def test_naturalday_tomorrow():
    assert naturalday(_today() + dt.timedelta(days=1)) == "tomorrow"


def test_naturalday_yesterday():
    assert naturalday(_today() - dt.timedelta(days=1)) == "yesterday"


def test_naturalday_other_date_uses_format():
    # Use a date far in the past so it isn't today/tomorrow/yesterday.
    d = dt.date(2000, 5, 17)
    assert naturalday(d) == "May 17"
    assert naturalday(d, "%Y-%m-%d") == "2000-05-17"


def test_naturalday_accepts_datetime():
    d = dt.datetime(2000, 5, 17, 10, 30)
    assert naturalday(d) == "May 17"


def test_naturalday_non_date_returns_string():
    assert naturalday("not a date") == "not a date"


def test_naturalday_out_of_range_returns_string():
    # Year 10000 raises ValueError when constructing dt.date
    class Fake:
        year = 10000
        month = 1
        day = 1

        def __str__(self):
            return "fake"

    assert naturalday(Fake()) == "10000-1-1" or naturalday(Fake()) == "fake"


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


def test_naturaldate_recent_uses_naturalday():
    d = _today()
    assert naturaldate(d) == "today"


def test_naturaldate_far_in_past_includes_year():
    d = dt.date(2000, 5, 17)
    assert naturaldate(d) == "May 17 2000"


def test_naturaldate_far_in_future_includes_year():
    d = _today() + dt.timedelta(days=400)
    result = naturaldate(d)
    assert str(d.year) in result


def test_naturaldate_non_date_returns_string():
    assert naturaldate("not a date") == "not a date"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


def test_precisedelta_docstring_example():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_custom_format():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"


def test_precisedelta_microseconds_minimum():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert (
        precisedelta(delta, minimum_unit="microseconds")
        == "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )


def test_precisedelta_suppress_days():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_suppress_low_units():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert (
        precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
        == "1.50 minutes"
    )


def test_precisedelta_minimum_minutes_rounding():
    assert precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == "0.02 minutes"


def test_precisedelta_minimum_minutes_zero():
    assert precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == "0 minutes"


def test_precisedelta_zero():
    assert precisedelta(dt.timedelta(0)) == "0 seconds"


def test_precisedelta_number_input():
    assert precisedelta(90) == "1 minute and 30 seconds"


def test_precisedelta_invalid_value_returns_string():
    assert precisedelta("abc") == "abc"


def test_precisedelta_none_returns_string():
    assert precisedelta(None) == "None"


def test_precisedelta_years():
    delta = dt.timedelta(days=365 + 30)
    assert precisedelta(delta) == "1 year and 1 month"


def test_precisedelta_plural_years():
    delta = dt.timedelta(days=730)
    assert precisedelta(delta) == "2 years"


def test_precisedelta_years_with_comma():
    delta = dt.timedelta(days=365 * 2000)
    assert precisedelta(delta) == "2,000 years"


def test_precisedelta_single_unit():
    assert precisedelta(dt.timedelta(seconds=5)) == "5 seconds"


def test_precisedelta_suppress_raises_when_no_replacement():
    with pytest.raises(ValueError):
        precisedelta(
            dt.timedelta(days=1),
            suppress=["years", "months", "days", "hours", "minutes", "seconds"],
        )
