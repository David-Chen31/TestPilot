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


class TestUnit:
    def test_ordering(self):
        assert Unit.MICROSECONDS < Unit.MILLISECONDS
        assert Unit.SECONDS < Unit.MINUTES
        assert Unit.DAYS < Unit.MONTHS
        assert Unit.MONTHS < Unit.YEARS

    def test_comparison_with_non_unit(self):
        with pytest.raises(TypeError):
            Unit.SECONDS < "seconds"  # noqa: B015

    def test_equality(self):
        assert Unit.HOURS == Unit.HOURS
        assert Unit.HOURS != Unit.DAYS

    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.YEARS.value == 7


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------


class TestNaturaldelta:
    def test_a_moment(self):
        assert naturaldelta(dt.timedelta(0)) == "a moment"

    def test_one_second(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_seconds(self):
        assert naturaldelta(dt.timedelta(seconds=5)) == "5 seconds"

    def test_seconds_round(self):
        assert naturaldelta(dt.timedelta(seconds=59)) == "59 seconds"

    def test_a_minute(self):
        assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"

    def test_minutes(self):
        assert naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"

    def test_minutes_round_to_hour(self):
        assert naturaldelta(dt.timedelta(seconds=3599)) == "an hour"

    def test_an_hour(self):
        assert naturaldelta(dt.timedelta(hours=1)) == "an hour"

    def test_hours(self):
        assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"

    def test_hours_round_to_day(self):
        assert naturaldelta(dt.timedelta(hours=23, minutes=59)) == "a day"

    def test_a_day(self):
        assert naturaldelta(dt.timedelta(days=1)) == "a day"

    def test_days(self):
        assert naturaldelta(dt.timedelta(days=3)) == "3 days"

    def test_days_a_month(self):
        # ~31 days -> 1 month
        assert naturaldelta(dt.timedelta(days=31)) == "a month"

    def test_months(self):
        assert naturaldelta(dt.timedelta(days=90)) == "3 months"

    def test_days_months_disabled(self):
        assert naturaldelta(dt.timedelta(days=31), months=False) == "31 days"

    def test_a_year(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_a_year_exact_with_months_off(self):
        assert (
            naturaldelta(dt.timedelta(days=365), months=False) == "a year"
        )

    def test_year_plus_day(self):
        assert naturaldelta(dt.timedelta(days=366)) == "1 year, 1 day"

    def test_year_plus_days(self):
        assert naturaldelta(dt.timedelta(days=370)) == "1 year, 5 days"

    def test_year_plus_month(self):
        # 365 + 31 days -> 1 year 1 month
        assert naturaldelta(dt.timedelta(days=396)) == "1 year, 1 month"

    def test_years(self):
        assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"

    def test_years_with_commas(self):
        # large number of years should use intcomma
        assert naturaldelta(dt.timedelta(days=365 * 1234)) == "1,234 years"

    def test_negative_timedelta(self):
        assert naturaldelta(dt.timedelta(days=-1)) == "a day"

    def test_float_seconds(self):
        assert naturaldelta(60) == "a minute"

    def test_float_rounds(self):
        assert naturaldelta(60.4) == "a minute"

    def test_string_returns_unchanged(self):
        assert naturaldelta("not a number") == "not a number"

    def test_nan_returns_str(self):
        assert naturaldelta(float("nan")) == "nan"

    def test_inf_returns_str(self):
        assert naturaldelta(float("inf")) == "inf"

    def test_minus_inf_returns_str(self):
        assert naturaldelta(float("-inf")) == "-inf"

    def test_too_large_raises(self):
        with pytest.raises(OverflowError):
            naturaldelta(1e30)

    def test_microseconds(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
            == "500 microseconds"
        )

    def test_one_microsecond(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds")
            == "1 microsecond"
        )

    def test_milliseconds(self):
        assert (
            naturaldelta(dt.timedelta(milliseconds=5), minimum_unit="milliseconds")
            == "5 milliseconds"
        )

    def test_millisecond_plural_zero(self):
        # zero with milliseconds minimum unit
        result = naturaldelta(
            dt.timedelta(microseconds=0), minimum_unit="milliseconds"
        )
        assert result == "0 milliseconds"

    def test_milliseconds_from_microseconds(self):
        assert (
            naturaldelta(dt.timedelta(milliseconds=5), minimum_unit="microseconds")
            == "5 milliseconds"
        )

    def test_a_moment_with_milliseconds_min(self):
        # seconds==0 & microseconds < 1000 with millisecond min -> 0 milliseconds
        assert (
            naturaldelta(dt.timedelta(microseconds=500), minimum_unit="milliseconds")
            == "0 milliseconds"
        )

    def test_invalid_minimum_unit(self):
        with pytest.raises(ValueError):
            naturaldelta(60, minimum_unit="days")

    def test_unknown_minimum_unit(self):
        with pytest.raises(KeyError):
            naturaldelta(60, minimum_unit="bananas")


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------


class TestNaturaltime:
    def test_a_moment_becomes_now(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(now, when=now) == "now"

    def test_past_datetime(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when - dt.timedelta(minutes=5)
        assert naturaltime(value, when=when) == "5 minutes ago"

    def test_future_datetime(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when + dt.timedelta(minutes=5)
        assert naturaltime(value, when=when) == "5 minutes from now"

    def test_timedelta_positive_is_future(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        # timedelta -> date = now - value (past); so tense = past
        assert naturaltime(dt.timedelta(minutes=5), when=when) == "5 minutes ago"

    def test_numeric_past(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(300, when=when) == "5 minutes ago"

    def test_numeric_future(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(300, future=True, when=when) == "5 minutes from now"

    def test_string_returns_unchanged(self):
        assert naturaltime("nope") == "nope"

    def test_months_false(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when - dt.timedelta(days=31)
        assert naturaltime(value, months=False, when=when) == "31 days ago"

    def test_aware_datetime(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when - dt.timedelta(minutes=5)
        assert naturaltime(value, when=when) == "5 minutes ago"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------


class TestNaturalday:
    def test_today(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(fixed) == "today"

    def test_tomorrow(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(dt.date(2020, 6, 16)) == "tomorrow"

    def test_yesterday(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(dt.date(2020, 6, 14)) == "yesterday"

    def test_other_day(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(dt.date(2020, 6, 10)) == "Jun 10"

    def test_custom_format(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(dt.date(2020, 6, 10), "%Y-%m-%d") == "2020-06-10"

    def test_datetime_input(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturalday(dt.datetime(2020, 6, 15, 9, 0)) == "today"

    def test_non_date_returns_str(self):
        assert naturalday("not a date") == "not a date"

    def test_out_of_range_returns_str(self):
        # Year 10000 out of range should not raise
        class Weird:
            year = 10000
            month = 1
            day = 1

        assert naturalday(Weird()) == str(Weird())


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------


class TestNaturaldate:
    def test_close_date(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturaldate(dt.date(2020, 6, 10)) == "Jun 10"

    def test_today(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturaldate(fixed) == "today"

    def test_far_future_includes_year(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturaldate(dt.date(2021, 6, 15)) == "Jun 15 2021"

    def test_far_past_includes_year(self, monkeypatch):
        fixed = dt.date(2020, 6, 15)

        class FakeDate(dt.date):
            @classmethod
            def today(cls):
                return fixed

        monkeypatch.setattr(dt, "date", FakeDate)
        assert naturaldate(dt.date(2019, 6, 15)) == "Jun 15 2019"

    def test_non_date_returns_str(self):
        assert naturaldate(42) == "42"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------


class TestPrecisedelta:
    def test_basic(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"

    def test_custom_format(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"

    def test_microseconds_minimum(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert (
            precisedelta(delta, minimum_unit="microseconds")
            == "2 days, 1 hour, 33 seconds and 123 milliseconds"
        )

    def test_suppress_days(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"

    def test_suppress_seconds_and_below(self):
        delta = dt.timedelta(seconds=90, microseconds=100)
        assert (
            precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
            == "1.50 minutes"
        )

    def test_minimum_minutes(self):
        delta = dt.timedelta(seconds=1)
        assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"

    def test_minimum_minutes_small(self):
        delta = dt.timedelta(seconds=0.1)
        assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"

    def test_zero(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_numeric_seconds(self):
        assert precisedelta(90) == "1 minute and 30.00 seconds"

    def test_string_returns_unchanged(self):
        assert precisedelta("nope") == "nope"

    def test_years(self):
        delta = dt.timedelta(days=365 * 2, hours=3)
        assert precisedelta(delta) == "2 years and 3.00 hours"

    def test_years_with_comma(self):
        delta = dt.timedelta(days=365 * 1000)
        result = precisedelta(delta)
        assert result == "1,000 years"

    def test_singular_year(self):
        delta = dt.timedelta(days=365)
        assert precisedelta(delta) == "1 year"

    def test_months(self):
        delta = dt.timedelta(days=60)
        assert precisedelta(delta) == "1 month and 30.00 days"

    def test_suppress_years(self):
        delta = dt.timedelta(days=365 * 2)
        assert precisedelta(delta, suppress=["years"]) == "730.00 days"

    def test_minimum_unit_years(self):
        delta = dt.timedelta(days=365 * 2)
        assert precisedelta(delta, minimum_unit="years") == "2.00 years"

    def test_minimum_unit_months(self):
        delta = dt.timedelta(days=730)
        assert precisedelta(delta, minimum_unit="months") == "23.93 months"

    def test_invalid_minimum_unit(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=10), minimum_unit="bananas")

    def test_invalid_suppress(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=10), suppress=["bananas"])

    def test_all_units_suppressed_raises(self):
        all_units = [u.name.lower() for u in Unit]
        with pytest.raises(ValueError):
            precisedelta(dt.timedelta(seconds=10), suppress=all_units)

    def test_suppress_lower_than_minimum(self):
        # Suppressing a unit lower than minimum should not raise.
        delta = dt.timedelta(minutes=5)
        assert precisedelta(delta, minimum_unit="hours", suppress=["seconds"]) == "0.08 hours"

    def test_single_unit_output(self):
        delta = dt.timedelta(days=1)
        assert precisedelta(delta) == "1 day"

    def test_rounding_promotes_to_hour(self):
        # 59.9 minutes should round up and promote to 1 hour
        delta = dt.timedelta(minutes=59, seconds=54)  # 59.9 minutes
        result = precisedelta(delta)
        assert result == "59.90 minutes"

    def test_format_integer(self):
        delta = dt.timedelta(seconds=90)
        # Using %d format -> integer truncation
        assert precisedelta(delta, format="%d") == "1 minute and 30 seconds"
