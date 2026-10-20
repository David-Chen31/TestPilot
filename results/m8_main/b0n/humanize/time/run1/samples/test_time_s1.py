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
import humanize.time as time_module


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


class TestUnit:
    def test_ordering(self):
        assert Unit.MICROSECONDS < Unit.MILLISECONDS
        assert Unit.SECONDS < Unit.MINUTES
        assert Unit.DAYS < Unit.MONTHS
        assert Unit.YEARS > Unit.HOURS

    def test_ordering_equal_not_less(self):
        assert not (Unit.DAYS < Unit.DAYS)
        assert Unit.DAYS <= Unit.DAYS

    def test_ordering_against_other_type_returns_notimplemented(self):
        assert Unit.DAYS.__lt__(5) is NotImplemented

    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.YEARS.value == 7

    def test_names(self):
        assert Unit["DAYS"] is Unit.DAYS


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


class TestNaturaldelta:
    def test_moment(self):
        assert naturaldelta(dt.timedelta(seconds=0)) == "a moment"

    def test_one_second(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_seconds(self):
        assert naturaldelta(dt.timedelta(seconds=5)) == "5 seconds"

    def test_seconds_59(self):
        assert naturaldelta(dt.timedelta(seconds=59)) == "59 seconds"

    def test_one_minute(self):
        assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"

    def test_minutes(self):
        assert naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"

    def test_rounding_to_hour(self):
        # 59 minutes 40 seconds rounds to 60 minutes -> an hour
        assert naturaldelta(dt.timedelta(minutes=59, seconds=40)) == "an hour"

    def test_one_hour(self):
        assert naturaldelta(dt.timedelta(hours=1)) == "an hour"

    def test_hours(self):
        assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"

    def test_rounding_hours_to_day(self):
        # 23 hours 45 minutes rounds to 24 hours -> a day
        assert naturaldelta(dt.timedelta(hours=23, minutes=45)) == "a day"

    def test_one_day(self):
        assert naturaldelta(dt.timedelta(days=1)) == "a day"

    def test_days_without_months(self):
        assert naturaldelta(dt.timedelta(days=5), months=False) == "5 days"

    def test_small_days_months_zero(self):
        # 10 days -> num_months round(10/30.5) == 0 -> days
        assert naturaldelta(dt.timedelta(days=10)) == "10 days"

    def test_one_month(self):
        # 30 or 31 days -> round(30/30.5) == 1
        assert naturaldelta(dt.timedelta(days=30)) == "a month"

    def test_months(self):
        assert naturaldelta(dt.timedelta(days=90)) == "3 months"

    def test_rounding_days_to_year(self):
        # 365-ish days but under year threshold counts get num_months==12 -> a year
        # days = 364 -> years=0, days=364, num_months = round(364/30.5)=12
        assert naturaldelta(dt.timedelta(days=364)) == "a year"

    def test_one_year(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_one_year_and_days(self):
        assert naturaldelta(dt.timedelta(days=365 + 5)) == "1 year, 5 days"

    def test_one_year_one_month(self):
        # 365 + 30 days -> 1 year, 1 month
        assert naturaldelta(dt.timedelta(days=395)) == "1 year, 1 month"

    def test_one_year_months(self):
        # 365 + 90 days
        assert naturaldelta(dt.timedelta(days=365 + 90)) == "1 year, 3 months"

    def test_one_year_no_months(self):
        # months=False falls to days
        result = naturaldelta(dt.timedelta(days=365 + 30), months=False)
        assert result == "1 year, 30 days"

    def test_one_year_twelve_months_promotes(self):
        # years == 1 and num_months == 12 -> years += 1
        # days = 365 -> years=1, days=0 -> already handled; need days resulting in 12 months
        # days=360 -> round(360/30.5)=12
        assert naturaldelta(dt.timedelta(days=365 + 360)) == "2 years"

    def test_many_years(self):
        assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"

    def test_many_years_rounding(self):
        # 365*2 + 200 -> round((2*365+200)/365) == round(2.54) == 3
        assert naturaldelta(dt.timedelta(days=365 * 2 + 200)) == "3 years"

    def test_thousands_years_intcomma(self):
        assert naturaldelta(dt.timedelta(days=365 * 1000)) == "1,000 years"

    def test_float_seconds(self):
        assert naturaldelta(30) == "30 seconds"
        assert naturaldelta(30.0) == "30 seconds"

    def test_negative_timedelta(self):
        assert naturaldelta(dt.timedelta(seconds=-30)) == "30 seconds"

    def test_nan_returns_str(self):
        assert naturaldelta(float("nan")) == "nan"

    def test_inf_returns_str(self):
        assert naturaldelta(float("inf")) == "inf"
        assert naturaldelta(float("-inf")) == "-inf"

    def test_overflow_raises(self):
        with pytest.raises(OverflowError):
            naturaldelta(float(1e30))

    def test_string_returns_str(self):
        assert naturaldelta("foo") == "foo"

    def test_minimum_unit_invalid_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="days")

    def test_minimum_unit_microseconds(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
            == "500 microseconds"
        )

    def test_minimum_unit_microseconds_one(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds")
            == "1 microsecond"
        )

    def test_minimum_unit_milliseconds(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=1500), minimum_unit="milliseconds")
            == "1.5 milliseconds"
        )

    def test_minimum_unit_microseconds_shows_milliseconds(self):
        # microseconds minimum but >= 1000 us and < 1e6 us -> milliseconds
        assert (
            naturaldelta(dt.timedelta(microseconds=500000), minimum_unit="microseconds")
            == "500.0 milliseconds"
        )

    def test_moment_with_seconds_min_unit(self):
        # sub-second with seconds min_unit -> a moment
        assert naturaldelta(dt.timedelta(microseconds=500)) == "a moment"


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


class TestNaturaltime:
    def test_seconds_ago(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(30, when=now) == "30 seconds ago"

    def test_seconds_future(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(30, future=True, when=now) == "30 seconds from now"

    def test_now(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(0, when=now) == "now"

    def test_datetime_past(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = now - dt.timedelta(minutes=5)
        assert naturaltime(value, when=now) == "5 minutes ago"

    def test_datetime_future(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = now + dt.timedelta(minutes=5)
        assert naturaltime(value, when=now) == "5 minutes from now"

    def test_timedelta_past(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        # timedelta is interpreted as value seconds, date = now - value -> past
        assert naturaltime(dt.timedelta(hours=2), when=now) == "2 hours ago"

    def test_string_returns_str(self):
        assert naturaltime("foo") == "foo"

    def test_aware_datetime_converted(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        now = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
        value = now - dt.timedelta(minutes=5)
        assert naturaltime(value, when=now) == "5 minutes ago"

    def test_months_false(self):
        now = dt.datetime(2020, 1, 1)
        value = now - dt.timedelta(days=90)
        assert naturaltime(value, months=False, when=now) == "90 days ago"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


class TestNaturalday:
    def test_today(self):
        today = dt.date.today()
        assert naturalday(today) == "today"

    def test_tomorrow(self):
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        assert naturalday(tomorrow) == "tomorrow"

    def test_yesterday(self):
        yesterday = dt.date.today() - dt.timedelta(days=1)
        assert naturalday(yesterday) == "yesterday"

    def test_other_date_formatted(self):
        target = dt.date.today() + dt.timedelta(days=10)
        assert naturalday(target) == target.strftime("%b %d")

    def test_custom_format(self):
        target = dt.date.today() + dt.timedelta(days=10)
        assert naturalday(target, format="%Y-%m-%d") == target.strftime("%Y-%m-%d")

    def test_datetime_input(self):
        today = dt.date.today()
        value = dt.datetime(today.year, today.month, today.day, 5, 0, 0)
        assert naturalday(value) == "today"

    def test_non_dateish_returns_str(self):
        assert naturalday("foo") == "foo"
        assert naturalday(123) == "123"


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


class TestNaturaldate:
    def test_near_date(self):
        target = dt.date.today() + dt.timedelta(days=1)
        assert naturaldate(target) == "tomorrow"

    def test_today(self):
        assert naturaldate(dt.date.today()) == "today"

    def test_far_future_appends_year(self):
        target = dt.date.today() + dt.timedelta(days=365)
        assert naturaldate(target) == target.strftime("%b %d %Y")

    def test_far_past_appends_year(self):
        target = dt.date.today() - dt.timedelta(days=365)
        assert naturaldate(target) == target.strftime("%b %d %Y")

    def test_near_other_format(self):
        # ~30 days away is < 5*365/12 (~152), so no year
        target = dt.date.today() + dt.timedelta(days=30)
        expected = target.strftime("%b %d")
        assert naturaldate(target) == expected

    def test_non_dateish_returns_str(self):
        assert naturaldate("foo") == "foo"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


class TestPrecisedelta:
    def test_default(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"

    def test_custom_format(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, format="%0.4f") == (
            "2 days, 1 hour and 33.1230 seconds"
        )

    def test_minimum_unit_microseconds(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, minimum_unit="microseconds") == (
            "2 days, 1 hour, 33 seconds and 123 milliseconds"
        )

    def test_suppress_days(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, suppress=["days"]) == (
            "49 hours and 33.12 seconds"
        )

    def test_suppress_seconds_and_below(self):
        delta = dt.timedelta(seconds=90, microseconds=100)
        assert (
            precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
            == "1.50 minutes"
        )

    def test_minimum_unit_minutes(self):
        delta = dt.timedelta(seconds=1)
        assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"

    def test_tiny_minimum_unit_minutes(self):
        delta = dt.timedelta(seconds=0.1)
        assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"

    def test_zero(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_one_second(self):
        assert precisedelta(dt.timedelta(seconds=1)) == "1 second"

    def test_one_year(self):
        assert precisedelta(dt.timedelta(days=365)) == "1 year"

    def test_many_years_intcomma(self):
        assert precisedelta(dt.timedelta(days=365 * 1000)) == "1,000 years"

    def test_float_input(self):
        assert precisedelta(3633.123) == "1 hour and 33.12 seconds"

    def test_bad_value_returns_str(self):
        assert precisedelta("foo") == "foo"

    def test_none_returns_str(self):
        assert precisedelta(None) == "None"

    def test_invalid_suppress_raises(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=1), suppress=["bogus"])

    def test_invalid_minimum_unit_raises(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=1), minimum_unit="bogus")

    def test_all_suppressed_raises(self):
        with pytest.raises(ValueError, match="Minimum unit is suppressed"):
            precisedelta(
                dt.timedelta(seconds=1),
                minimum_unit="years",
                suppress=[u.name.lower() for u in Unit],
            )

    def test_two_units_joined_with_and(self):
        delta = dt.timedelta(hours=1, minutes=30)
        assert precisedelta(delta) == "1 hour and 30 minutes"

    def test_three_units(self):
        delta = dt.timedelta(days=1, hours=2, minutes=3)
        assert precisedelta(delta) == "1 day, 2 hours and 3 minutes"

    def test_milliseconds(self):
        delta = dt.timedelta(milliseconds=500)
        assert precisedelta(delta, minimum_unit="milliseconds") == "500 milliseconds"

    def test_microseconds(self):
        delta = dt.timedelta(microseconds=500)
        assert (
            precisedelta(delta, minimum_unit="microseconds") == "500 microseconds"
        )

    def test_promotion_of_seconds(self):
        # 59.9 seconds rounding to 60 promotes to minute
        delta = dt.timedelta(seconds=59.9)
        # seconds is minimum_unit -> precision preserved
        assert precisedelta(delta) == "59.90 seconds"

    def test_rounding_promotion(self):
        # seconds is not min_unit -> should promote to minute when suppressed to integer
        delta = dt.timedelta(seconds=59, microseconds=999999)
        # seconds is minimum unit by default so precision preserved
        assert precisedelta(delta) == "60.00 seconds"

    def test_suppress_promotion(self):
        delta = dt.timedelta(days=30, hours=23, minutes=59, seconds=59)
        # Fine-grained suppression causing promotions
        result = precisedelta(
            delta, minimum_unit="hours", suppress=["seconds", "minutes"]
        )
        assert result == "743 hours"

    def test_days_promote_to_month(self):
        # 31 days promotes to 1 month when months not suppressed
        delta = dt.timedelta(days=31)
        assert precisedelta(delta) == "1 month"

    def test_months_promote_to_year(self):
        delta = dt.timedelta(days=365)
        # 365 days = 1 year exactly
        assert precisedelta(delta) == "1 year"


class TestPrecisedeltaHelpers:
    def test_rounding_by_fmt_int(self):
        assert time_module._rounding_by_fmt("%d", 3.7) == 3
        assert isinstance(time_module._rounding_by_fmt("%d", 3.7), int)

    def test_rounding_by_fmt_float(self):
        assert time_module._rounding_by_fmt("%.2f", 3.14159) == 3.14

    def test_quotient_and_remainder_min_unit(self):
        q, r = time_module._quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f"
        )
        assert (q, r) == (1.5, 0)

    def test_quotient_and_remainder_suppressed(self):
        q, r = time_module._quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f"
        )
        assert (q, r) == (0, 36)

    def test_quotient_and_remainder_normal(self):
        q, r = time_module._quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f"
        )
        assert (q, r) == (1, 12)

    def test_suitable_minimum_unit_unsuppressed(self):
        assert time_module._suitable_minimum_unit(Unit.HOURS, []) is Unit.HOURS

    def test_suitable_minimum_unit_suppressed(self):
        assert (
            time_module._suitable_minimum_unit(Unit.HOURS, [Unit.HOURS])
            is Unit.DAYS
        )

    def test_suitable_minimum_unit_suppressed_multiple(self):
        assert (
            time_module._suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS])
            is Unit.MONTHS
        )

    def test_suppress_lower_units(self):
        result = time_module._suppress_lower_units(Unit.SECONDS, [Unit.DAYS])
        assert result == {Unit.MICROSECONDS, Unit.MILLISECONDS, Unit.DAYS}

    def test_abs_timedelta_positive(self):
        delta = dt.timedelta(days=1)
        assert time_module._abs_timedelta(delta) == delta

    def test_abs_timedelta_negative(self):
        delta = dt.timedelta(days=-1)
        assert time_module._abs_timedelta(delta) == dt.timedelta(days=1)
