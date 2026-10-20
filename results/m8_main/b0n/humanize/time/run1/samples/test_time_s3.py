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
# Unit enum
# ---------------------------------------------------------------------------


def test_unit_ordering():
    assert Unit.SECONDS < Unit.MINUTES < Unit.HOURS < Unit.DAYS
    assert not (Unit.DAYS < Unit.HOURS)
    assert Unit.YEARS.value == 7
    assert Unit.MICROSECONDS.value == 0


def test_unit_sorting():
    ordered = sorted(
        [Unit.YEARS, Unit.SECONDS, Unit.MICROSECONDS, Unit.DAYS]
    )
    assert ordered == [
        Unit.MICROSECONDS,
        Unit.SECONDS,
        Unit.DAYS,
        Unit.YEARS,
    ]


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


def test_naturaldelta_none_datetime_returns_str_of_value():
    # naturaltime handles date None; here test naturaldelta with a datetime
    assert naturaldelta(dt.timedelta(seconds=0)) == "a moment"


def test_naturaldelta_invalid_returns_str():
    assert naturaldelta("NaN") == "NaN"
    assert naturaldelta("inf") == "inf"
    assert naturaldelta("-inf") == "-inf"


def test_naturaldelta_non_finite_float():
    assert naturaldelta(float("nan")) == "nan"
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("-inf")) == "-inf"


def test_naturaldelta_overflow():
    with pytest.raises(OverflowError):
        naturaldelta(1e300)


def test_naturaldelta_minimum_unit_validation():
    with pytest.raises(ValueError):
        naturaldelta(1, minimum_unit="minutes")


def test_naturaldelta_moment():
    assert naturaldelta(dt.timedelta(milliseconds=1)) == "a moment"


def test_naturaldelta_seconds():
    assert naturaldelta(dt.timedelta(seconds=1)) == "a second"
    assert naturaldelta(dt.timedelta(seconds=2)) == "2 seconds"
    assert naturaldelta(dt.timedelta(seconds=45)) == "45 seconds"


def test_naturaldelta_minutes():
    assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"
    assert naturaldelta(dt.timedelta(minutes=2)) == "2 minutes"
    assert naturaldelta(dt.timedelta(seconds=90)) == "2 minutes"


def test_naturaldelta_minutes_rounds_to_hour():
    assert naturaldelta(dt.timedelta(minutes=59, seconds=40)) == "an hour"


def test_naturaldelta_hours():
    assert naturaldelta(dt.timedelta(hours=1)) == "an hour"
    assert naturaldelta(dt.timedelta(hours=2)) == "2 hours"
    assert naturaldelta(dt.timedelta(hours=23)) == "23 hours"


def test_naturaldelta_hours_rounds_to_day():
    assert naturaldelta(dt.timedelta(hours=23, minutes=30)) == "a day"


def test_naturaldelta_days():
    assert naturaldelta(dt.timedelta(days=1)) == "a day"
    assert naturaldelta(dt.timedelta(days=2)) == "2 days"
    assert naturaldelta(dt.timedelta(days=20)) == "20 days"


def test_naturaldelta_days_no_months():
    # 45 days rounds to ~1 month by default, but with months=False just days
    assert naturaldelta(dt.timedelta(days=45), months=False) == "45 days"


def test_naturaldelta_months():
    assert naturaldelta(dt.timedelta(days=31)) == "a month"
    assert naturaldelta(dt.timedelta(days=62)) == "2 months"
    # 12 months -> a year
    assert naturaldelta(dt.timedelta(days=365 - 20)) == "a year"


def test_naturaldelta_one_year():
    assert naturaldelta(dt.timedelta(days=365)) == "a year"
    assert naturaldelta(dt.timedelta(days=366)) == "1 year, 1 day"


def test_naturaldelta_one_year_months():
    assert naturaldelta(dt.timedelta(days=365 + 31)) == "1 year, 1 month"
    assert naturaldelta(dt.timedelta(days=365 + 62)) == "1 year, 2 months"


def test_naturaldelta_one_year_days_no_months():
    assert (
        naturaldelta(dt.timedelta(days=365 + 10), months=False)
        == "1 year, 10 days"
    )


def test_naturaldelta_years():
    assert naturaldelta(dt.timedelta(days=365 * 2)) == "2 years"
    assert naturaldelta(dt.timedelta(days=365 * 10)) == "10 years"


def test_naturaldelta_milliseconds():
    assert (
        naturaldelta(dt.timedelta(milliseconds=500), minimum_unit="milliseconds")
        == "500 milliseconds"
    )
    assert (
        naturaldelta(dt.timedelta(milliseconds=1), minimum_unit="milliseconds")
        == "1 millisecond"
    )


def test_naturaldelta_microseconds():
    assert (
        naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
        == "500 microseconds"
    )
    assert (
        naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds")
        == "1 microsecond"
    )


def test_naturaldelta_microseconds_promoted_to_milliseconds():
    assert (
        naturaldelta(dt.timedelta(microseconds=2000), minimum_unit="microseconds")
        == "2 milliseconds"
    )


def test_naturaldelta_from_number():
    assert naturaldelta(60) == "a minute"
    assert naturaldelta(120.0) == "2 minutes"


def test_naturaldelta_negative_timedelta():
    assert naturaldelta(dt.timedelta(seconds=-60)) == "a minute"


def test_naturaldelta_years_with_commas():
    assert ",000" in naturaldelta(dt.timedelta(days=365 * 1000))


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


def test_naturaltime_returns_now_for_moment():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(now, when=now) == "now"


def test_naturaltime_past_datetime():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    when = now - dt.timedelta(minutes=5)
    assert naturaltime(when, when=now) == "5 minutes ago"


def test_naturaltime_future_datetime():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    when = now + dt.timedelta(minutes=5)
    assert naturaltime(when, when=now) == "5 minutes from now"


def test_naturaltime_timedelta_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    # A positive timedelta is interpreted as "in the past" (now - delta)
    assert naturaltime(dt.timedelta(minutes=5), when=now) == "5 minutes ago"


def test_naturaltime_timedelta_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(minutes=-5), when=now) == "5 minutes from now"


def test_naturaltime_number_past():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(60, when=now) == "a minute ago"


def test_naturaltime_number_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(60, future=True, when=now) == "a minute from now"


def test_naturaltime_invalid_returns_str():
    assert naturaltime("NaN") == "NaN"


def test_naturaltime_aware_datetime():
    import datetime as dt

    now = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    when = now - dt.timedelta(hours=2)
    assert naturaltime(when, when=now) == "2 hours ago"


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
    other = dt.date.today() + dt.timedelta(days=30)
    assert naturalday(other, format="%Y-%m-%d") == other.strftime("%Y-%m-%d")


def test_naturalday_non_date_returns_str():
    assert naturalday("not a date") == "not a date"


def test_naturalday_out_of_range():
    # dt.date with out-of-range isn't possible normally, but simulate via attribute
    class FakeDate:
        year = 10000
        month = 1
        day = 1

    result = naturalday(FakeDate())
    # Catches OverflowError / ValueError -> str
    assert isinstance(result, str)


def test_naturalday_datetime():
    now = dt.datetime.now()
    assert naturalday(now) == "today"


def test_naturalday_aware_datetime_today():
    aware = dt.datetime.now(dt.timezone.utc)
    assert naturalday(aware) == "today"


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


def test_naturaldate_today():
    assert naturaldate(dt.date.today()) == "today"


def test_naturaldate_recent_uses_naturalday():
    recent = dt.date.today() - dt.timedelta(days=30)
    assert naturaldate(recent) == naturalday(recent)


def test_naturaldate_far_uses_year():
    far = dt.date.today() + dt.timedelta(days=365)
    expected = naturalday(far, "%b %d %Y")
    assert naturaldate(far) == expected
    assert str(far.year) in naturaldate(far)


def test_naturaldate_non_date_returns_str():
    assert naturaldate("hello") == "hello"


def test_naturaldate_datetime():
    now = dt.datetime.now()
    assert naturaldate(now) == "today"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


def test_precisedelta_basic():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_custom_format():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, format="%0.4f") == (
        "2 days, 1 hour and 33.1230 seconds"
    )


def test_precisedelta_microseconds_min():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert (
        precisedelta(delta, minimum_unit="microseconds")
        == "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )


def test_precisedelta_suppress_days():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_suppress_seconds():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert (
        precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
        == "1.50 minutes"
    )


def test_precisedelta_seconds_minimum_minutes():
    delta = dt.timedelta(seconds=1)
    assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"


def test_precisedelta_too_small():
    delta = dt.timedelta(seconds=0.1)
    assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"


def test_precisedelta_invalid_value():
    assert precisedelta("NaN") == "NaN"


def test_precisedelta_single_unit():
    assert precisedelta(dt.timedelta(seconds=5)) == "5.00 seconds"


def test_precisedelta_only_min_unit_zero():
    assert precisedelta(dt.timedelta(0)) == "0.00 seconds"


def test_precisedelta_from_number():
    assert precisedelta(90.5) == "1.51 minutes"


def test_precisedelta_number_no_precision_round():
    # precise=True means the value is not rounded to integer before conversion
    assert precisedelta(1.5) == "1.50 seconds"


def test_precisedelta_singular_second_format():
    assert precisedelta(dt.timedelta(seconds=1)) == "1.00 seconds"


def test_precisedelta_format_with_int():
    assert precisedelta(dt.timedelta(seconds=5), format="%d") == "5 seconds"


def test_precisedelta_years():
    delta = dt.timedelta(days=365 * 2)
    assert precisedelta(delta) == "2 years"


def test_precisedelta_years_with_commas():
    delta = dt.timedelta(days=365 * 1000)
    assert precisedelta(delta) == "1,000 years"


def test_precisedelta_suppressed_minimum_unit_promoted():
    # suppress seconds and below, min is seconds -> promoted to minutes
    result = precisedelta(
        dt.timedelta(seconds=90),
        suppress=["seconds", "milliseconds", "microseconds"],
    )
    assert result == "1.50 minutes"


def test_precisedelta_suppressed_all_units_raises():
    with pytest.raises(ValueError):
        precisedelta(
            dt.timedelta(seconds=1),
            minimum_unit="years",
            suppress=[
                "years",
                "months",
                "days",
                "hours",
                "minutes",
                "seconds",
                "milliseconds",
                "microseconds",
            ],
        )


def test_precisedelta_mixed_units():
    delta = dt.timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert precisedelta(delta) == "1 day, 2 hours, 3 minutes and 4.00 seconds"


def test_precisedelta_plural_singular():
    assert precisedelta(dt.timedelta(days=1)) == "1.00 days"
    assert precisedelta(dt.timedelta(days=2)) == "2 days"
