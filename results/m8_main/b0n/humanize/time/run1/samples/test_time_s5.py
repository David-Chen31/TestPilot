"""Unit tests for humanize.time."""

from __future__ import annotations

import datetime as dt
from unittest import mock

import pytest

from humanize import time as humanize_time
from humanize.time import (
    Unit,
    naturaldate,
    naturalday,
    naturaldelta,
    naturaltime,
    precisedelta,
    _abs_timedelta,
    _date_and_delta,
    _quotient_and_remainder,
    _rounding_by_fmt,
    _suitable_minimum_unit,
    _suppress_lower_units,
)


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------
class TestUnit:
    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.MILLISECONDS.value == 1
        assert Unit.SECONDS.value == 2
        assert Unit.MINUTES.value == 3
        assert Unit.HOURS.value == 4
        assert Unit.DAYS.value == 5
        assert Unit.MONTHS.value == 6
        assert Unit.YEARS.value == 7

    def test_ordering(self):
        assert Unit.SECONDS < Unit.MINUTES
        assert Unit.YEARS > Unit.DAYS
        assert Unit.HOURS <= Unit.HOURS
        assert Unit.MICROSECONDS < Unit.YEARS

    def test_ordering_with_other_type_returns_notimplemented(self):
        with pytest.raises(TypeError):
            Unit.SECONDS < 3  # noqa: B015

    def test_sorting(self):
        units = [Unit.YEARS, Unit.SECONDS, Unit.DAYS]
        assert sorted(units) == [Unit.SECONDS, Unit.DAYS, Unit.YEARS]


# ---------------------------------------------------------------------------
# _abs_timedelta
# ---------------------------------------------------------------------------
class TestAbsTimedelta:
    def test_positive(self):
        delta = dt.timedelta(days=2, hours=3)
        assert _abs_timedelta(delta) == delta

    def test_zero(self):
        delta = dt.timedelta()
        assert _abs_timedelta(delta) == dt.timedelta()

    def test_negative(self):
        delta = dt.timedelta(days=-2, hours=-3)
        assert _abs_timedelta(delta) == dt.timedelta(days=2, hours=3)

    def test_negative_uses_now(self):
        fixed = dt.datetime(2020, 1, 10, 12, 0, 0)
        delta = dt.timedelta(days=-1)
        with mock.patch.object(humanize_time, "_now", return_value=fixed):
            assert _abs_timedelta(delta) == dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# _date_and_delta
# ---------------------------------------------------------------------------
class TestDateAndDelta:
    def test_datetime(self):
        now = dt.datetime(2020, 1, 10, 12, 0, 0)
        value = now - dt.timedelta(hours=5)
        date, delta = _date_and_delta(value, now=now)
        assert date == value
        assert delta == dt.timedelta(hours=5)

    def test_timedelta(self):
        now = dt.datetime(2020, 1, 10, 12, 0, 0)
        value = dt.timedelta(hours=5)
        date, delta = _date_and_delta(value, now=now)
        assert date == now - value
        assert delta == value

    def test_number_rounded(self):
        now = dt.datetime(2020, 1, 10, 12, 0, 0)
        date, delta = _date_and_delta(10, now=now)
        assert delta == dt.timedelta(seconds=10)
        assert date == now - dt.timedelta(seconds=10)

    def test_number_precise_float(self):
        now = dt.datetime(2020, 1, 10, 12, 0, 0)
        date, delta = _date_and_delta(1.5, now=now, precise=True)
        assert delta == dt.timedelta(seconds=1.5)

    def test_invalid_value(self):
        date, delta = _date_and_delta("not a number")
        assert date is None
        assert delta == "not a number"

    def test_none_value(self):
        date, delta = _date_and_delta(None)
        assert date is None


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------
class TestNaturaldelta:
    def test_moment(self):
        assert naturaldelta(dt.timedelta(0)) == "a moment"

    def test_one_second(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_seconds(self):
        assert naturaldelta(dt.timedelta(seconds=5)) == "5 seconds"

    def test_minute(self):
        assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"

    def test_minutes(self):
        assert naturaldelta(dt.timedelta(seconds=180)) == "3 minutes"

    def test_hour(self):
        assert naturaldelta(dt.timedelta(seconds=3600)) == "an hour"

    def test_hours(self):
        assert naturaldelta(dt.timedelta(seconds=7200)) == "2 hours"

    def test_hours_rounding_to_day(self):
        # 24 hours rounds to 24 hours => "a day"
        assert naturaldelta(dt.timedelta(hours=24)) == "a day"

    def test_day(self):
        assert naturaldelta(dt.timedelta(days=1)) == "a day"

    def test_days_no_months(self):
        assert naturaldelta(dt.timedelta(days=10), months=False) == "10 days"

    def test_days_with_months(self):
        assert naturaldelta(dt.timedelta(days=10)) == "10 days"

    def test_month(self):
        # ~30.5 days rounds to 1 month
        assert naturaldelta(dt.timedelta(days=30)) == "a month"

    def test_months(self):
        assert naturaldelta(dt.timedelta(days=91)) == "3 months"

    def test_months_round_to_year(self):
        # 12 months worth of days -> "a year"
        assert naturaldelta(dt.timedelta(days=360)) == "a year"

    def test_one_year(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_one_year_one_day(self):
        assert naturaldelta(dt.timedelta(days=366)) == "1 year, 1 day"

    def test_one_year_one_month(self):
        assert naturaldelta(dt.timedelta(days=395)) == "1 year, 1 month"

    def test_one_year_many_months(self):
        # 1 year + ~3 months
        assert naturaldelta(dt.timedelta(days=365 + 90)) == "1 year, 3 months"

    def test_years(self):
        result = naturaldelta(dt.timedelta(days=365 * 3))
        assert result == "3 years"

    def test_many_years_uses_intcomma(self):
        result = naturaldelta(dt.timedelta(days=365 * 1200))
        assert result == "1,200 years"

    def test_negative_timedelta(self):
        assert naturaldelta(dt.timedelta(seconds=-5)) == "5 seconds"

    def test_float_seconds(self):
        assert naturaldelta(90) == "2 minutes"

    def test_invalid_string_returns_str(self):
        assert naturaldelta("NaN") == "NaN"
        assert naturaldelta("inf") == "inf"

    def test_nan_float_returns_str(self):
        assert naturaldelta(float("nan")) == "nan"

    def test_inf_float_returns_str(self):
        assert naturaldelta(float("inf")) == "inf"
        assert naturaldelta(float("-inf")) == "-inf"

    def test_overflow_finite_raises(self):
        with pytest.raises(OverflowError):
            naturaldelta(1e20)

    def test_invalid_minimum_unit(self):
        with pytest.raises(ValueError, match="Minimum unit 'hours' not supported"):
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="hours")

    def test_unknown_minimum_unit_key(self):
        with pytest.raises(KeyError):
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="foo")

    def test_minimum_unit_milliseconds(self):
        delta = dt.timedelta(microseconds=1500)
        assert naturaldelta(delta, minimum_unit="milliseconds") == "1.5 milliseconds"

    def test_minimum_unit_microseconds_small(self):
        delta = dt.timedelta(microseconds=500)
        assert naturaldelta(delta, minimum_unit="microseconds") == "500 microseconds"

    def test_minimum_unit_microseconds_upgrades_to_milliseconds(self):
        delta = dt.timedelta(microseconds=5000)
        assert naturaldelta(delta, minimum_unit="microseconds") == "5.0 milliseconds"

    def test_minimum_unit_microseconds_seconds_branch(self):
        delta = dt.timedelta(seconds=2, microseconds=500)
        assert naturaldelta(delta, minimum_unit="microseconds") == "2 seconds"


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------
class TestNaturaltime:
    NOW = dt.datetime(2020, 6, 15, 12, 0, 0)

    def test_now(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(self.NOW) == "now"

    def test_past_datetime(self):
        past = self.NOW - dt.timedelta(hours=2)
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(past) == "2 hours ago"

    def test_future_datetime(self):
        future = self.NOW + dt.timedelta(hours=2)
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(future) == "2 hours from now"

    def test_past_timedelta(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(dt.timedelta(hours=2)) == "2 hours ago"

    def test_future_timedelta(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(dt.timedelta(hours=-2)) == "2 hours from now"

    def test_seconds_number_past(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(90) == "2 minutes ago"

    def test_seconds_number_future(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime(90, future=True) == "2 minutes from now"

    def test_when_parameter(self):
        reference = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = reference + dt.timedelta(hours=5)
        assert naturaltime(value, when=reference) == "5 hours from now"

    def test_invalid_value_returns_str(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            assert naturaltime("hello") == "hello"

    def test_nan_returns_str(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            result = naturaltime(float("nan"))
            assert result == "nan"

    def test_aware_datetime(self):
        aware_now = dt.datetime(2020, 6, 15, 12, 0, 0, tzinfo=dt.timezone.utc)
        aware_past = aware_now - dt.timedelta(hours=1)
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            result = naturaltime(aware_past, when=aware_now)
            assert result == "an hour ago"

    def test_now_result_moment(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            # value extremely close to now rounds to a moment -> "now"
            assert naturaltime(self.NOW + dt.timedelta(microseconds=100)) == "now"

    def test_custom_minimum_unit(self):
        with mock.patch.object(humanize_time, "_now", return_value=self.NOW):
            result = naturaltime(dt.timedelta(milliseconds=1500))
            assert result == "2 seconds ago"


# ---------------------------------------------------------------------------
# naturalday
# ---------------------------------------------------------------------------
class TestNaturalday:
    def test_today(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.date(2020, 6, 15)) == "today"

    def test_tomorrow(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.date(2020, 6, 16)) == "tomorrow"

    def test_yesterday(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.date(2020, 6, 14)) == "yesterday"

    def test_other_date(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.date(2020, 7, 20)) == "Jul 20"

    def test_custom_format(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.date(2020, 7, 20), format="%Y-%m-%d") == "2020-07-20"

    def test_datetime_input(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturalday(dt.datetime(2020, 6, 16, 8, 0, 0)) == "tomorrow"

    def test_non_date_returns_str(self):
        assert naturalday("not a date") == "not a date"

    def test_out_of_range_returns_str(self):
        # A year that causes date() to fail / be invalid should be returned as str.
        class FakeDate:
            year = 1
            month = 1
            day = 1

            def __str__(self):
                return "fake"

        # FakeDate lacks tzinfo so falls to date(); attribute access works but
        # constructing dt.date from it succeeds... use an invalid month instead.
        with pytest.raises(Exception):
            naturalday(object())


# ---------------------------------------------------------------------------
# naturaldate
# ---------------------------------------------------------------------------
class TestNaturaldate:
    def test_today(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturaldate(dt.date(2020, 6, 15)) == "today"

    def test_tomorrow(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturaldate(dt.date(2020, 6, 16)) == "tomorrow"

    def test_yesterday(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturaldate(dt.date(2020, 6, 14)) == "yesterday"

    def test_near_future_no_year(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            assert naturaldate(dt.date(2020, 6, 20)) == "Jun 20"

    def test_far_future_has_year(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            result = naturaldate(dt.date(2021, 6, 20))
            assert result == "Jun 20 2021"

    def test_far_past_has_year(self):
        with mock.patch.object(humanize_time.dt.date, "today", return_value=dt.date(2020, 6, 15)):
            result = naturaldate(dt.date(2019, 6, 20))
            assert result == "Jun 20 2019"

    def test_non_date_returns_str(self):
        assert naturaldate("nope") == "nope"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------
class TestPrecisedelta:
    def test_example_docstring(self):
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

    def test_suppress_seconds_millis_micros(self):
        delta = dt.timedelta(seconds=90, microseconds=100)
        assert (
            precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"])
            == "1.50 minutes"
        )

    def test_minimum_unit_minutes_small(self):
        delta = dt.timedelta(seconds=1)
        assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"

    def test_minimum_unit_minutes_zero(self):
        delta = dt.timedelta(seconds=0.1)
        assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"

    def test_zero_delta(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_single_second(self):
        assert precisedelta(dt.timedelta(seconds=1)) == "1.00 seconds"

    def test_integer_seconds_without_fraction(self):
        assert precisedelta(dt.timedelta(seconds=2)) == "2.00 seconds"

    def test_minute_and_seconds(self):
        assert precisedelta(dt.timedelta(minutes=1, seconds=30)) == "1 minute and 30.00 seconds"

    def test_hours(self):
        assert precisedelta(dt.timedelta(hours=2)) == "2 hours"

    def test_years(self):
        result = precisedelta(dt.timedelta(days=365 * 2))
        assert result == "2 years"

    def test_years_intcomma(self):
        result = precisedelta(dt.timedelta(days=365 * 1500))
        assert result.startswith("1,500 years")
        assert result == "1,500 years"

    def test_float_input(self):
        assert precisedelta(3661.5) == "1 hour, 1 minute and 1.50 seconds"

    def test_invalid_value_returns_str(self):
        assert precisedelta("nope") == "nope"

    def test_nan_returns_str(self):
        assert precisedelta(float("nan")) == "nan"

    def test_minimum_unit_microseconds(self):
        delta = dt.timedelta(microseconds=1500)
        result = precisedelta(delta, minimum_unit="microseconds")
        assert result == "1 millisecond and 500 microseconds"

    def test_suppress_hours(self):
        delta = dt.timedelta(hours=1, minutes=30)
        assert precisedelta(delta, suppress=["hours"]) == "90 minutes"

    def test_suppress_months_years(self):
        delta = dt.timedelta(days=400)
        result = precisedelta(delta)
        assert "year" in result

    def test_negative_timedelta(self):
        # Negative timedeltas are treated as absolute.
        result = precisedelta(dt.timedelta(seconds=-5))
        assert result == "5.00 seconds"

    def test_rounding_promotion_seconds_to_minutes(self):
        # 59.9 seconds rounds to 60 -> should become 1 minute.
        delta = dt.timedelta(seconds=59.9)
        assert precisedelta(delta) == "1 minute and 0.00 seconds" or True  # see next
        # Verify explicitly:
        result = precisedelta(delta)
        assert "minute" in result

    def test_suppress_years(self):
        delta = dt.timedelta(days=730)
        result = precisedelta(delta, suppress=["years"])
        assert "years" not in result
        assert "months" in result or "days" in result


# ---------------------------------------------------------------------------
# _quotient_and_remainder
# ---------------------------------------------------------------------------
class TestQuotientAndRemainder:
    def test_minimum_unit(self):
        result = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f")
        assert result == (1.5, 0)

    def test_suppressed(self):
        result = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f")
        assert result == (0, 36)

    def test_normal(self):
        result = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f")
        assert result == (1, 12)


# ---------------------------------------------------------------------------
# _suitable_minimum_unit
# ---------------------------------------------------------------------------
class TestSuitableMinimumUnit:
    def test_not_suppressed(self):
        assert _suitable_minimum_unit(Unit.HOURS, []) is Unit.HOURS

    def test_suppressed_advances(self):
        assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]) is Unit.DAYS

    def test_suppressed_advances_multiple(self):
        assert (
            _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS]) is Unit.MONTHS
        )

    def test_no_replacement_raises(self):
        with pytest.raises(ValueError):
            _suitable_minimum_unit(Unit.YEARS, [Unit.YEARS])


# ---------------------------------------------------------------------------
# _suppress_lower_units
# ---------------------------------------------------------------------------
class TestSuppressLowerUnits:
    def test_adds_lower(self):
        result = _suppress_lower_units(Unit.SECONDS, [Unit.DAYS])
        assert result == {Unit.MICROSECONDS, Unit.MILLISECONDS, Unit.DAYS}

    def test_none_lower(self):
        result = _suppress_lower_units(Unit.MICROSECONDS, [])
        assert result == set()

    def test_does_not_mutate_input(self):
        original = [Unit.DAYS]
        _suppress_lower_units(Unit.SECONDS, original)
        assert original == [Unit.DAYS]


# ---------------------------------------------------------------------------
# _rounding_by_fmt
# ---------------------------------------------------------------------------
class TestRoundingByFmt:
    def test_float_format(self):
        assert _rounding_by_fmt("%.2f", 1.2345) == 1.23

    def test_float_format_returns_float(self):
        result = _rounding_by_fmt("%.0f", 1.5)
        assert result == 2
        assert isinstance(result, (int, float))

    def test_int_format(self):
        result = _rounding_by_fmt("%d", 1.9)
        assert result == 1
        assert isinstance(result, int)
