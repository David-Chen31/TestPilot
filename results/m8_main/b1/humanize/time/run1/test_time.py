"""Tests for humanize.time."""

from __future__ import annotations

import datetime as dt

import pytest

from humanize.time import (
    Unit,
    _abs_timedelta,
    _date_and_delta,
    _quotient_and_remainder,
    _rounding_by_fmt,
    _suitable_minimum_unit,
    _suppress_lower_units,
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
    assert Unit.SECONDS < Unit.MINUTES
    assert Unit.YEARS > Unit.DAYS
    assert not (Unit.HOURS < Unit.HOURS)


def test_unit_equality():
    assert Unit.DAYS == Unit.DAYS
    assert Unit.DAYS != Unit.HOURS


def test_unit_compare_with_non_unit_returns_notimplemented():
    with pytest.raises(TypeError):
        Unit.DAYS < 5  # type: ignore[operator]


def test_unit_values():
    assert Unit.MICROSECONDS.value == 0
    assert Unit.YEARS.value == 7


# ---------------------------------------------------------------------------
# _abs_timedelta
# ---------------------------------------------------------------------------


def test_abs_timedelta_positive():
    delta = dt.timedelta(days=3)
    assert _abs_timedelta(delta) == delta


def test_abs_timedelta_negative():
    delta = dt.timedelta(days=-2, seconds=-5)
    result = _abs_timedelta(delta)
    assert result == dt.timedelta(days=2, seconds=5)


# ---------------------------------------------------------------------------
# _date_and_delta
# ---------------------------------------------------------------------------


def test_date_and_delta_datetime():
    now = dt.datetime(2020, 1, 10)
    date, delta = _date_and_delta(dt.datetime(2020, 1, 5), now=now)
    assert date == dt.datetime(2020, 1, 5)
    assert delta == dt.timedelta(days=5)


def test_date_and_delta_timedelta():
    now = dt.datetime(2020, 1, 10)
    date, delta = _date_and_delta(dt.timedelta(days=2), now=now)
    assert date == dt.datetime(2020, 1, 8)
    assert delta == dt.timedelta(days=2)


def test_date_and_delta_number():
    now = dt.datetime(2020, 1, 10)
    date, delta = _date_and_delta(60, now=now)
    assert date == dt.datetime(2020, 1, 10) - dt.timedelta(seconds=60)
    assert delta == dt.timedelta(seconds=60)


def test_date_and_delta_precise_no_rounding():
    now = dt.datetime(2020, 1, 10)
    _, delta = _date_and_delta(1.4, now=now, precise=True)
    assert delta == dt.timedelta(seconds=1.4)


def test_date_and_delta_invalid_value():
    now = dt.datetime(2020, 1, 10)
    date, value = _date_and_delta("bad", now=now)
    assert date is None
    assert value == "bad"


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


def test_naturaldelta_moment():
    assert naturaldelta(dt.timedelta(0)) == "a moment"


def test_naturaldelta_one_second():
    assert naturaldelta(dt.timedelta(seconds=1)) == "a second"


def test_naturaldelta_seconds():
    assert naturaldelta(dt.timedelta(seconds=30)) == "30 seconds"


def test_naturaldelta_one_minute():
    assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"


def test_naturaldelta_minutes():
    assert naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"


def test_naturaldelta_sixty_minutes_is_an_hour():
    assert naturaldelta(dt.timedelta(minutes=60)) == "an hour"


def test_naturaldelta_one_hour():
    assert naturaldelta(dt.timedelta(hours=1)) == "an hour"


def test_naturaldelta_hours():
    assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"


def test_naturaldelta_24_hours_is_a_day():
    assert naturaldelta(dt.timedelta(hours=24)) == "a day"


def test_naturaldelta_one_day():
    assert naturaldelta(dt.timedelta(days=1)) == "a day"


def test_naturaldelta_days():
    assert naturaldelta(dt.timedelta(days=3)) == "3 days"


def test_naturaldelta_days_no_months():
    assert naturaldelta(dt.timedelta(days=45), months=False) == "45 days"


def test_naturaldelta_month():
    assert naturaldelta(dt.timedelta(days=31)) == "a month"


def test_naturaldelta_months():
    assert naturaldelta(dt.timedelta(days=92)) == "3 months"


def test_naturaldelta_one_year():
    assert naturaldelta(dt.timedelta(days=365)) == "a year"


def test_naturaldelta_year_one_month():
    assert naturaldelta(dt.timedelta(days=365 + 30)) == "1 year, 1 month"


def test_naturaldelta_year_months():
    assert naturaldelta(dt.timedelta(days=365 + 61)) == "1 year, 2 months"


def test_naturaldelta_one_year_days_only():
    assert naturaldelta(dt.timedelta(days=365 + 5), months=False) == "1 year, 5 days"


def test_naturaldelta_years():
    assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"


def test_naturaldelta_years_with_comma():
    # years >= 2 goes through intcomma branch
    result = naturaldelta(dt.timedelta(days=365 * 2000))
    assert result == "2,000 years"


def test_naturaldelta_float_input():
    assert naturaldelta(90) == "2 minutes"


def test_naturaldelta_nan_returns_str():
    assert naturaldelta(float("nan")) == "nan"


def test_naturaldelta_inf_returns_str():
    assert naturaldelta(float("inf")) == "inf"


def test_naturaldelta_too_large_raises():
    with pytest.raises(OverflowError):
        naturaldelta(1e20)


def test_naturaldelta_bad_type_returns_str():
    assert naturaldelta("bad") == "bad"  # type: ignore[arg-type]


def test_naturaldelta_minimum_unit_microseconds():
    result = naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
    assert result == "500 microseconds"


def test_naturaldelta_minimum_unit_milliseconds():
    # The milliseconds branch formats via int(), truncating the fractional part.
    result = naturaldelta(dt.timedelta(microseconds=1500), minimum_unit="milliseconds")
    assert result == "1 millisecond"


def test_naturaldelta_minimum_unit_invalid():
    with pytest.raises(ValueError):
        naturaldelta(dt.timedelta(seconds=5), minimum_unit="minutes")


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


def test_naturaltime_now():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    assert naturaltime(when, when=when) == "now"


def test_naturaltime_past_datetime():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    value = when - dt.timedelta(minutes=5)
    assert naturaltime(value, when=when) == "5 minutes ago"


def test_naturaltime_future_datetime():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    value = when + dt.timedelta(hours=2)
    assert naturaltime(value, when=when) == "2 hours from now"


def test_naturaltime_seconds_future_flag():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    assert naturaltime(120, future=True, when=when) == "2 minutes from now"


def test_naturaltime_seconds_past():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    assert naturaltime(120, when=when) == "2 minutes ago"


def test_naturaltime_timedelta_tense():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    # A timedelta is interpreted as "value ago": date = now - value, so past
    assert naturaltime(dt.timedelta(days=1), when=when) == "a day ago"


def test_naturaltime_invalid_value_returns_str():
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    assert naturaltime("bad", when=when) == "bad"  # type: ignore[arg-type]


def test_naturaltime_aware_datetime():
    tz = dt.timezone(dt.timedelta(hours=2))
    when = dt.datetime(2020, 1, 10, 12, 0, 0)
    value = dt.datetime(2020, 1, 10, 10, 0, 0, tzinfo=tz)
    # aware datetime is converted; no exception, reasonable tense
    result = naturaltime(value, when=when)
    assert result.endswith("ago") or result.endswith("from now") or result == "now"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


def test_naturalday_today():
    today = dt.date.today()
    assert naturalday(today) == "today"


def test_naturalday_tomorrow():
    today = dt.date.today()
    assert naturalday(today + dt.timedelta(days=1)) == "tomorrow"


def test_naturalday_yesterday():
    today = dt.date.today()
    assert naturalday(today - dt.timedelta(days=1)) == "yesterday"


def test_naturalday_other_date():
    today = dt.date.today()
    other = today + dt.timedelta(days=10)
    assert naturalday(other) == other.strftime("%b %d")


def test_naturalday_custom_format():
    today = dt.date.today()
    other = today + dt.timedelta(days=10)
    assert naturalday(other, format="%Y-%m-%d") == other.strftime("%Y-%m-%d")


def test_naturalday_datetime_value():
    today = dt.date.today()
    value = dt.datetime.combine(today, dt.time(15, 30))
    assert naturalday(value) == "today"


def test_naturalday_non_date():
    assert naturalday("not a date") == "not a date"  # type: ignore[arg-type]


def test_naturalday_out_of_range():
    class Weird:
        year = 999999
        month = 1
        day = 1

    assert naturalday(Weird()) == str(Weird()) or isinstance(naturalday(Weird()), str)


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


def test_naturaldate_today():
    today = dt.date.today()
    assert naturaldate(today) == "today"


def test_naturaldate_tomorrow():
    today = dt.date.today()
    assert naturaldate(today + dt.timedelta(days=1)) == "tomorrow"


def test_naturaldate_near_date_uses_default_format():
    today = dt.date.today()
    other = today + dt.timedelta(days=10)
    assert naturaldate(other) == other.strftime("%b %d")


def test_naturaldate_far_date_appends_year():
    today = dt.date.today()
    other = today + dt.timedelta(days=365)
    assert naturaldate(other) == other.strftime("%b %d %Y")


def test_naturaldate_non_date():
    assert naturaldate("bad") == "bad"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


def test_precisedelta_example():
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


def test_precisedelta_suppress_seconds_and_below():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert (
        precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
        == "1.50 minutes"
    )


def test_precisedelta_minimum_unit_minutes():
    delta = dt.timedelta(seconds=1)
    assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"


def test_precisedelta_below_minimum_unit_zero():
    delta = dt.timedelta(seconds=0.1)
    assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"


def test_precisedelta_invalid_value_returns_str():
    assert precisedelta("bad") == "bad"  # type: ignore[arg-type]


def test_precisedelta_none_returns_str():
    assert precisedelta(None) == "None"


def test_precisedelta_years():
    delta = dt.timedelta(days=365 * 2)
    result = precisedelta(delta)
    assert result == "2 years"


def test_precisedelta_large_years_comma():
    delta = dt.timedelta(days=365 * 2000)
    assert precisedelta(delta) == "2,000 years"


def test_precisedelta_single_unit():
    assert precisedelta(dt.timedelta(seconds=30)) == "30 seconds"


def test_precisedelta_seconds_one():
    assert precisedelta(dt.timedelta(seconds=1)) == "1 second"


def test_precisedelta_seconds_and_minutes():
    # 3599.99 seconds = 59 minutes and 59.99 seconds; no promotion happens here.
    delta = dt.timedelta(seconds=3599.99)
    result = precisedelta(delta)
    assert result == "59 minutes and 59.99 seconds"


def test_precisedelta_invalid_minimum_unit():
    with pytest.raises(KeyError):
        precisedelta(dt.timedelta(seconds=5), minimum_unit="bogus")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_quotient_and_remainder_minimum_unit():
    q, r = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f")
    assert q == 1.5
    assert r == 0


def test_quotient_and_remainder_suppressed():
    q, r = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f")
    assert q == 0
    assert r == 36


def test_quotient_and_remainder_divmod():
    q, r = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f")
    assert q == 1
    assert r == 12


def test_suitable_minimum_unit_not_suppressed():
    assert _suitable_minimum_unit(Unit.HOURS, []) == Unit.HOURS


def test_suitable_minimum_unit_suppressed():
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]) == Unit.DAYS


def test_suitable_minimum_unit_suppressed_multiple():
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS]) == Unit.MONTHS


def test_suitable_minimum_unit_all_suppressed_raises():
    with pytest.raises(ValueError):
        _suitable_minimum_unit(Unit.HOURS, list(Unit))


def test_suppress_lower_units():
    result = _suppress_lower_units(Unit.SECONDS, [Unit.DAYS])
    assert result == {Unit.MICROSECONDS, Unit.MILLISECONDS, Unit.DAYS}


def test_rounding_by_fmt_float():
    assert _rounding_by_fmt("%0.2f", 1.234) == 1.23


def test_rounding_by_fmt_int():
    assert _rounding_by_fmt("%d", 1.9) == 1
