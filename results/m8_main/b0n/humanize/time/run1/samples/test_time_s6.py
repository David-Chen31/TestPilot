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
class TestUnit:
    def test_ordering(self):
        assert Unit.MICROSECONDS < Unit.SECONDS
        assert Unit.SECONDS < Unit.YEARS
        assert Unit.YEARS > Unit.DAYS
        assert Unit.MINUTES >= Unit.MINUTES
        assert Unit.HOURS <= Unit.HOURS

    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.MILLISECONDS.value == 1
        assert Unit.SECONDS.value == 2
        assert Unit.MINUTES.value == 3
        assert Unit.HOURS.value == 4
        assert Unit.DAYS.value == 5
        assert Unit.MONTHS.value == 6
        assert Unit.YEARS.value == 7

    def test_compare_with_other_type_returns_notimplemented(self):
        assert Unit.SECONDS.__lt__(5) is NotImplemented
        assert Unit.SECONDS.__lt__("x") is NotImplemented
        with pytest.raises(TypeError):
            Unit.SECONDS < 5


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------
class TestNaturaldelta:
    def test_moment(self):
        assert naturaldelta(dt.timedelta(seconds=0)) == "a moment"

    def test_tiny_seconds(self):
        assert naturaldelta(dt.timedelta(milliseconds=400)) == "a moment"

    def test_one_second(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_seconds(self):
        assert naturaldelta(dt.timedelta(seconds=30)) == "30 seconds"

    def test_one_minute(self):
        assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"

    def test_minutes(self):
        assert naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"

    def test_minutes_rounds_to_hour(self):
        # 59.6 minutes rounds to 60 minutes -> "an hour"
        assert naturaldelta(dt.timedelta(seconds=59 * 60 + 40)) == "an hour"

    def test_one_hour(self):
        assert naturaldelta(dt.timedelta(hours=1)) == "an hour"

    def test_hours(self):
        assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"

    def test_hours_round_to_day(self):
        assert naturaldelta(dt.timedelta(hours=23, minutes=40)) == "a day"

    def test_one_day(self):
        assert naturaldelta(dt.timedelta(days=1)) == "a day"

    def test_days_plain(self):
        assert naturaldelta(dt.timedelta(days=5)) == "5 days"

    def test_days_with_months(self):
        # 40 days -> 1.31 months -> round to 1 -> "a month"
        assert naturaldelta(dt.timedelta(days=40)) == "a month"

    def test_days_no_months(self):
        assert naturaldelta(dt.timedelta(days=40), months=False) == "40 days"

    def test_months(self):
        assert naturaldelta(dt.timedelta(days=100)) == "3 months"

    def test_year_exact(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_year_with_days(self):
        assert naturaldelta(dt.timedelta(days=365 + 1)) == "1 year, 1 day"

    def test_year_with_months(self):
        # 1 year + 40 days -> 40 days ~ 1 month
        assert naturaldelta(dt.timedelta(days=405)) == "1 year, 1 month"

    def test_year_with_months_plural(self):
        # 1 year + 90 days -> ~3 months
        assert naturaldelta(dt.timedelta(days=455)) == "1 year, 3 months"

    def test_years_plural(self):
        assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"

    def test_years_intcomma(self):
        value = dt.timedelta(days=365 * 1234)
        assert naturaldelta(value) == "1,234 years"

    def test_negative_timedelta(self):
        assert naturaldelta(dt.timedelta(seconds=-30)) == "30 seconds"

    def test_numeric_input(self):
        assert naturaldelta(65) == "a minute"

    def test_numeric_float_input(self):
        assert naturaldelta(0.5) == "a moment"

    def test_numeric_string_input_returns_string(self):
        result = naturaldelta("not a number")
        assert result == "not a number"

    def test_inf_input_returns_string(self):
        assert naturaldelta(float("inf")) == "inf"

    def test_nan_input_returns_string(self):
        assert naturaldelta(float("nan")) == "nan"

    def test_overflow_raises(self):
        with pytest.raises(OverflowError):
            naturaldelta(10**18)

    def test_minimum_unit_microseconds(self):
        # 400 microseconds, min unit microseconds
        result = naturaldelta(dt.timedelta(microseconds=400), minimum_unit="microseconds")
        assert result == "400 microseconds"

    def test_minimum_unit_microseconds_one(self):
        result = naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds")
        assert result == "1 microsecond"

    def test_minimum_unit_microseconds_milliseconds(self):
        result = naturaldelta(
            dt.timedelta(milliseconds=5), minimum_unit="microseconds"
        )
        assert result == "5 milliseconds"

    def test_minimum_unit_milliseconds(self):
        result = naturaldelta(dt.timedelta(microseconds=500), minimum_unit="milliseconds")
        assert result == "0 milliseconds"

    def test_minimum_unit_invalid(self):
        with pytest.raises(ValueError):
            naturaldelta(dt.timedelta(seconds=10), minimum_unit="minutes")

    def test_minimum_unit_unknown_name(self):
        with pytest.raises(KeyError):
            naturaldelta(dt.timedelta(seconds=10), minimum_unit="bogus")


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------
class TestNaturaltime:
    NOW = dt.datetime(2020, 6, 15, 12, 0, 0)

    def test_now(self):
        assert naturaltime(self.NOW, when=self.NOW) == "now"

    def test_past_seconds(self):
        value = self.NOW - dt.timedelta(seconds=30)
        assert naturaltime(value, when=self.NOW) == "30 seconds ago"

    def test_future_seconds(self):
        value = self.NOW + dt.timedelta(seconds=30)
        assert naturaltime(value, when=self.NOW) == "30 seconds from now"

    def test_past_minute(self):
        value = self.NOW - dt.timedelta(minutes=1)
        assert naturaltime(value, when=self.NOW) == "a minute ago"

    def test_past_hour(self):
        value = self.NOW - dt.timedelta(hours=1)
        assert naturaltime(value, when=self.NOW) == "an hour ago"

    def test_timedelta_past(self):
        assert naturaltime(dt.timedelta(seconds=10), when=self.NOW) == "10 seconds ago"

    def test_numeric_past(self):
        assert naturaltime(10, when=self.NOW) == "10 seconds ago"

    def test_numeric_future(self):
        assert naturaltime(10, future=True, when=self.NOW) == "10 seconds from now"

    def test_numeric_negative(self):
        assert naturaltime(-30, when=self.NOW) == "30 seconds ago"

    def test_string_returns_unchanged(self):
        assert naturaltime("bogus", when=self.NOW) == "bogus"

    def test_inf_returns_string(self):
        assert naturaltime(float("inf"), when=self.NOW) == "inf"

    def test_nan_returns_string(self):
        assert naturaltime(float("nan"), when=self.NOW) == "nan"

    def test_aware_datetime_converted(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        value = dt.datetime(2020, 6, 15, 14, 0, 0, tzinfo=tz)
        # Local naive interpretation of that instant, compared against naive NOW
        # depends on machine TZ so instead compare against the converted value.
        result = naturaltime(value)
        assert isinstance(result, str)
        assert result.endswith(" ago") or result.endswith(" from now") or result == "now"

    def test_a_moment_is_now(self):
        value = self.NOW - dt.timedelta(milliseconds=100)
        assert naturaltime(value, when=self.NOW) == "now"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------
class TestNaturalDay:
    def test_today(self):
        assert naturalday(dt.date.today()) == "today"

    def test_tomorrow(self):
        assert naturalday(dt.date.today() + dt.timedelta(days=1)) == "tomorrow"

    def test_yesterday(self):
        assert naturalday(dt.date.today() - dt.timedelta(days=1)) == "yesterday"

    def test_other_date_uses_format(self):
        today = dt.date.today()
        far = today + dt.timedelta(days=30)
        assert naturalday(far) == far.strftime("%b %d")

    def test_custom_format(self):
        today = dt.date.today()
        far = today + dt.timedelta(days=30)
        assert naturalday(far, format="%Y-%m-%d") == far.strftime("%Y-%m-%d")

    def test_datetime_input(self):
        today = dt.date.today()
        assert naturalday(dt.datetime.combine(today, dt.time(12, 0))) == "today"

    def test_non_date_returns_str(self):
        assert naturalday("hello") == "hello"

    def test_out_of_range_returns_str(self):
        # Year 10000 overflows strftime in some cases; dt.date rejects it first.
        bad = dt.date(9999, 12, 31)
        # 9999 is valid; use an object whose .year raises OverflowError via date range.
        class FakeDate:
            year = 10000
            month = 1
            day = 1
        assert naturalday(FakeDate()) == str(FakeDate())


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------
class TestNaturalDate:
    def test_today(self):
        assert naturaldate(dt.date.today()) == "today"

    def test_tomorrow(self):
        assert naturaldate(dt.date.today() + dt.timedelta(days=1)) == "tomorrow"

    def test_yesterday(self):
        assert naturaldate(dt.date.today() - dt.timedelta(days=1)) == "yesterday"

    def test_near_future_no_year(self):
        today = dt.date.today()
        near = today + dt.timedelta(days=30)
        assert naturaldate(near) == near.strftime("%b %d")

    def test_far_future_with_year(self):
        today = dt.date.today()
        far = today + dt.timedelta(days=365)
        assert naturaldate(far) == far.strftime("%b %d %Y")

    def test_far_past_with_year(self):
        today = dt.date.today()
        far = today - dt.timedelta(days=365)
        assert naturaldate(far) == far.strftime("%b %d %Y")

    def test_datetime_input(self):
        today = dt.date.today()
        assert naturaldate(dt.datetime.combine(today, dt.time(9, 0))) == "today"

    def test_non_date_returns_str(self):
        assert naturaldate(123) == "123"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------
class TestPrecisedelta:
    def test_documented_example(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"

    def test_custom_format(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"

    def test_minimum_unit_microseconds(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert (
            precisedelta(delta, minimum_unit="microseconds")
            == "2 days, 1 hour, 33 seconds and 123 milliseconds"
        )

    def test_suppress_days(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"

    def test_suppress_seconds_millis_micros(self):
        delta = dt.timedelta(seconds=90, microseconds=100)
        assert (
            precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
            == "1.50 minutes"
        )

    def test_minimum_unit_minutes_small(self):
        assert precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == "0.02 minutes"

    def test_minimum_unit_minutes_rounds_to_zero(self):
        assert precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == "0 minutes"

    def test_zero(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_single_second(self):
        assert precisedelta(dt.timedelta(seconds=1)) == "1 second"

    def test_single_minute(self):
        assert precisedelta(dt.timedelta(minutes=1)) == "1 minute"

    def test_hours_minutes(self):
        assert precisedelta(dt.timedelta(hours=1, minutes=30)) == "1 hour and 30 minutes"

    def test_years(self):
        assert precisedelta(dt.timedelta(days=365)) == "1 year"

    def test_years_intcomma(self):
        delta = dt.timedelta(days=365 * 1234)
        assert precisedelta(delta) == "1,234 years"

    def test_months(self):
        # 30.5 days rounds to a month
        assert precisedelta(dt.timedelta(days=31)) == "1 month"

    def test_negative_timedelta(self):
        assert precisedelta(dt.timedelta(seconds=-10)) == "10 seconds"

    def test_numeric_input(self):
        assert precisedelta(90) == "1 minute and 30 seconds"

    def test_numeric_float_input(self):
        assert precisedelta(1.5) == "1.50 seconds"

    def test_invalid_input_returns_str(self):
        assert precisedelta("bad") == "bad"

    def test_minimum_unit_suppressed_raises(self):
        with pytest.raises(ValueError):
            precisedelta(
                dt.timedelta(seconds=10),
                minimum_unit="seconds",
                suppress=[u.name.lower() for u in Unit],
            )

    def test_suppress_invalid_unit_raises(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=10), suppress=["bogus"])

    def test_milliseconds(self):
        assert precisedelta(dt.timedelta(milliseconds=123)) == "0.12 seconds"

    def test_microseconds_min(self):
        assert precisedelta(dt.timedelta(microseconds=500), minimum_unit="microseconds") == "500 microseconds"

    def test_promotion_minutes_to_hour(self):
        # 59 minutes 59 seconds -> 59 minutes 59.00 seconds
        delta = dt.timedelta(minutes=59, seconds=59)
        assert precisedelta(delta) == "59 minutes and 59.00 seconds"
