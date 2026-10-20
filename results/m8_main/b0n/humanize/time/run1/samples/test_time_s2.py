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


def test_unit_values_and_ordering():
    assert Unit.MICROSECONDS.value == 0
    assert Unit.YEARS.value == 7
    assert Unit.SECONDS < Unit.MINUTES
    assert Unit.DAYS < Unit.YEARS
    assert not (Unit.YEARS < Unit.DAYS)
    assert Unit.SECONDS <= Unit.SECONDS
    assert Unit.MINUTES >= Unit.SECONDS
    assert Unit.MONTHS < Unit.YEARS


def test_unit_ordering_other_type_returns_not_implemented():
    assert Unit.SECONDS.__lt__(5) is NotImplemented


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta,expected",
    [
        (dt.timedelta(seconds=0), "a moment"),
        (dt.timedelta(seconds=1), "a second"),
        (dt.timedelta(seconds=2), "2 seconds"),
        (dt.timedelta(seconds=30), "30 seconds"),
        (dt.timedelta(seconds=60), "a minute"),
        (dt.timedelta(minutes=2), "2 minutes"),
        (dt.timedelta(minutes=59), "an hour"),
        (dt.timedelta(hours=1), "an hour"),
        (dt.timedelta(hours=2), "2 hours"),
        (dt.timedelta(hours=23, minutes=59), "a day"),
        (dt.timedelta(days=1), "a day"),
        (dt.timedelta(days=2), "2 days"),
    ],
)
def test_naturaldelta_small_units(delta, expected):
    assert naturaldelta(delta) == expected


def test_naturaldelta_seconds_float_rounding():
    assert naturaldelta(1.4) == "a second"
    assert naturaldelta(1.5) == "2 seconds"


def test_naturaldelta_minutes_rounding():
    # 90 seconds -> round(90/60) = 2 minutes
    assert naturaldelta(dt.timedelta(seconds=90)) == "2 minutes"


def test_naturaldelta_hours_rounding():
    assert naturaldelta(dt.timedelta(seconds=90 * 60)) == "an hour"
    assert naturaldelta(dt.timedelta(seconds=150 * 60)) == "2 hours"


def test_naturaldelta_months():
    # 35 days -> 1 month per 30.5 rounding
    assert naturaldelta(dt.timedelta(days=35)) == "a month"
    assert naturaldelta(dt.timedelta(days=70)) == "2 months"


def test_naturaldelta_months_disabled():
    assert naturaldelta(dt.timedelta(days=70), months=False) == "70 days"


def test_naturaldelta_months_disabled_year():
    result = naturaldelta(dt.timedelta(days=400), months=False)
    assert result == "1 year, 35 days"


def test_naturaldelta_years():
    assert naturaldelta(dt.timedelta(days=365)) == "a year"
    assert naturaldelta(dt.timedelta(days=730)) == "2 years"


def test_naturaldelta_years_with_months():
    assert naturaldelta(dt.timedelta(days=365 + 30)) == "1 year, 1 month"
    assert naturaldelta(dt.timedelta(days=365 + 60)) == "1 year, 2 months"


def test_naturaldelta_years_with_days_no_months():
    assert naturaldelta(dt.timedelta(days=400), months=False) == "1 year, 35 days"


def test_naturaldelta_months_round_to_year():
    # 1 year + 365-ish days where remaining days round to 12 months
    delta = dt.timedelta(days=365 + 360)  # 725 days -> years=1, days=360 -> months=12
    assert naturaldelta(delta) == "2 years"


def test_naturaldelta_large_years_uses_intcomma():
    # > ~10000 years triggers intcomma formatting
    delta = dt.timedelta(days=365 * 12345)
    assert naturaldelta(delta) == "12,345 years"


def test_naturaldelta_minimum_unit_milliseconds():
    assert naturaldelta(dt.timedelta(microseconds=1500), minimum_unit="milliseconds") == "1.5 milliseconds"


def test_naturaldelta_minimum_unit_microseconds():
    assert naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds") == "500 microseconds"
    assert naturaldelta(dt.timedelta(microseconds=1500), minimum_unit="microseconds") == "1.5 milliseconds"


def test_naturaldelta_minimum_unit_microseconds_small():
    assert naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds") == "1 microsecond"


def test_naturaldelta_minimum_unit_invalid():
    with pytest.raises(ValueError, match="not supported"):
        naturaldelta(dt.timedelta(seconds=5), minimum_unit="days")


def test_naturaldelta_string_input_returns_str():
    assert naturaldelta("nan") == "nan"
    assert naturaldelta("inf") == "inf"


def test_naturaldelta_float_inf_returns_str():
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("-inf")) == "-inf"


def test_naturaldelta_float_nan_returns_str():
    assert naturaldelta(float("nan")) == "nan"


def test_naturaldelta_overflow_raises():
    with pytest.raises(OverflowError):
        naturaldelta(1e300)


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


def test_naturaltime_moment():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now, when=now) == "now"


def test_naturaltime_datetime_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now - dt.timedelta(hours=1), when=now) == "an hour ago"


def test_naturaltime_datetime_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now + dt.timedelta(days=2), when=now) == "2 days from now"


def test_naturaltime_timedelta_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(hours=3), when=now) == "3 hours ago"


def test_naturaltime_float_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(3600, when=now) == "an hour ago"


def test_naturaltime_float_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(3600, future=True, when=now) == "an hour from now"


def test_naturaltime_invalid_input_returns_str():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime("garbage", when=now) == "garbage"


def test_naturaltime_aware_datetime():
    tz = dt.timezone(dt.timedelta(hours=2))
    when = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
    value = when - dt.timedelta(minutes=5)
    result = naturaltime(value, when=when)
    assert result == "5 minutes ago"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


def test_naturalday_today():
    today = dt.date.today()
    assert naturalday(today) == "today"


def test_naturalday_tomorrow():
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    assert naturalday(tomorrow) == "tomorrow"


def test_naturalday_yesterday():
    yesterday = dt.date.today() - dt.timedelta(days=1)
    assert naturalday(yesterday) == "yesterday"


def test_naturalday_other_format():
    other = dt.date.today() + dt.timedelta(days=10)
    assert naturalday(other) == other.strftime("%b %d")


def test_naturalday_custom_format():
    other = dt.date.today() + dt.timedelta(days=10)
    assert naturalday(other, format="%Y-%m-%d") == other.strftime("%Y-%m-%d")


def test_naturalday_datetime():
    today = dt.date.today()
    value = dt.datetime(today.year, today.month, today.day, 5, 0, 0)
    assert naturalday(value) == "today"


def test_naturalday_non_date_returns_str():
    assert naturalday("not a date") == "not a date"


def test_naturalday_out_of_range_returns_str():
    assert naturalday(dt.date(1, 1, 1)) == "0001-01-01"


def test_naturalday_aware_datetime():
    tz = dt.timezone.utc
    value = dt.datetime.now(tz)
    assert naturalday(value) == "today"


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


def test_naturaldate_recent():
    today = dt.date.today()
    assert naturaldate(today) == "today"


def test_naturaldate_yesterday():
    yesterday = dt.date.today() - dt.timedelta(days=1)
    assert naturaldate(yesterday) == "yesterday"


def test_naturaldate_far_includes_year():
    today = dt.date.today()
    far = today + dt.timedelta(days=400)
    result = naturaldate(far)
    assert result == far.strftime("%b %d %Y")


def test_naturaldate_far_past_includes_year():
    today = dt.date.today()
    far = today - dt.timedelta(days=400)
    result = naturaldate(far)
    assert result == far.strftime("%b %d %Y")


def test_naturaldate_non_date_returns_str():
    assert naturaldate(123) == "123"


def test_naturaldate_out_of_range_returns_str():
    assert naturaldate(dt.date(1, 1, 1)) == "0001-01-01"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


def test_precisedelta_basic():
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


def test_precisedelta_suppress_subseconds():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert (
        precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
        == "1.50 minutes"
    )


def test_precisedelta_min_minutes():
    assert precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == "0.02 minutes"
    assert precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == "0 minutes"


def test_precisedelta_none_returns_str():
    assert precisedelta(None) == "None"


def test_precisedelta_string_returns_str():
    assert precisedelta("garbage") == "garbage"


def test_precisedelta_zero():
    assert precisedelta(dt.timedelta(0)) == "0 seconds"


def test_precisedelta_single_unit():
    assert precisedelta(dt.timedelta(hours=1)) == "1 hour"


def test_precisedelta_years_included():
    delta = dt.timedelta(days=365)
    assert precisedelta(delta) == "12 months, 30 days" or precisedelta(delta) == "1 year"


def test_precisedelta_years_with_min_years():
    delta = dt.timedelta(days=400)
    result = precisedelta(delta, minimum_unit="years")
    assert result == "1.10 years"


def test_precisedelta_years_uses_intcomma():
    delta = dt.timedelta(days=365 * 12345)
    result = precisedelta(delta)
    assert "12,345" in result


def test_precisedelta_suppress_all_raises():
    with pytest.raises(ValueError, match="suppressed"):
        precisedelta(
            dt.timedelta(seconds=1),
            suppress=[u.name.lower() for u in Unit],
        )


def test_precisedelta_suppress_promotion_minutes_to_hour():
    # 59.9 minutes should promote to 1 hour
    delta = dt.timedelta(minutes=59, seconds=59)
    result = precisedelta(delta, suppress=["seconds"])
    assert result == "1 hour"


def test_precisedelta_suppress_promotion_hours_to_day():
    delta = dt.timedelta(hours=23, minutes=59, seconds=59)
    result = precisedelta(delta, suppress=["seconds", "minutes"])
    assert result == "1 day"


def test_precisedelta_suppress_promotion_months_to_year():
    delta = dt.timedelta(days=364)
    result = precisedelta(delta)
    # days < 365 so no year, should be months/days
    assert "year" not in result


def test_precisedelta_days_promote_to_month():
    delta = dt.timedelta(days=31)
    result = precisedelta(delta)
    assert result == "1 month, 1 day" or result.startswith("1 month")


def test_precisedelta_negative_timedelta():
    delta = dt.timedelta(seconds=-10)
    result = precisedelta(delta)
    assert "second" in result
