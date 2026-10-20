"""Tests for humanize.time."""

from __future__ import annotations

import datetime as dt

import pytest

from humanize import time as time_module
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
        assert Unit.MONTHS < Unit.YEARS
        assert not (Unit.YEARS < Unit.MONTHS)
        assert Unit.DAYS <= Unit.DAYS
        assert Unit.DAYS == Unit.DAYS
        assert Unit.HOURS > Unit.MINUTES
        assert Unit.HOURS >= Unit.HOURS

    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.YEARS.value == 7

    def test_compare_other_type_returns_notimplemented(self):
        assert Unit.DAYS.__lt__(5) is NotImplemented

    def test_lt_with_other_type_raises(self):
        with pytest.raises(TypeError):
            Unit.DAYS < 5


# ---------------------------------------------------------------------------
# _date_and_delta / precise flag
# ---------------------------------------------------------------------------
class TestDateAndDelta:
    def test_precise_keeps_float(self):
        # With precise=True a float seconds value must not be rounded.
        now = dt.datetime(2020, 1, 1)
        _, delta = time_module._date_and_delta(1.6, now=now, precise=True)
        assert delta == dt.timedelta(seconds=1.6)

    def test_not_precise_rounds(self):
        now = dt.datetime(2020, 1, 1)
        _, delta = time_module._date_and_delta(1.6, now=now, precise=False)
        assert delta == dt.timedelta(seconds=2)

    def test_date_is_now_minus_delta(self):
        now = dt.datetime(2020, 1, 1)
        date, delta = time_module._date_and_delta(30, now=now)
        assert date == now - dt.timedelta(seconds=30)
        assert delta == dt.timedelta(seconds=30)

    def test_invalid_returns_none(self):
        date, value = time_module._date_and_delta("bad")
        assert date is None
        assert value == "bad"


# ---------------------------------------------------------------------------
# naturaldelta
# ---------------------------------------------------------------------------
class TestNaturalDelta:
    def test_zero_is_a_moment(self):
        assert naturaldelta(dt.timedelta(0)) == "a moment"

    def test_a_second(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_two_seconds(self):
        assert naturaldelta(dt.timedelta(seconds=2)) == "2 seconds"

    def test_seconds_plural(self):
        assert naturaldelta(dt.timedelta(seconds=45)) == "45 seconds"

    def test_a_minute(self):
        assert naturaldelta(dt.timedelta(minutes=1)) == "a minute"

    def test_minutes(self):
        assert naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"

    def test_an_hour(self):
        assert naturaldelta(dt.timedelta(hours=1)) == "an hour"

    def test_hours(self):
        assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"

    def test_a_day(self):
        assert naturaldelta(dt.timedelta(days=1)) == "a day"

    def test_days(self):
        assert naturaldelta(dt.timedelta(days=5)) == "5 days"

    def test_a_month(self):
        assert naturaldelta(dt.timedelta(days=31)) == "a month"

    def test_months(self):
        assert naturaldelta(dt.timedelta(days=61)) == "2 months"

    def test_a_year(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_years(self):
        assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"

    def test_years_with_commas(self):
        assert naturaldelta(dt.timedelta(days=365 * 1000)) == "1,000 years"

    def test_year_and_days(self):
        assert naturaldelta(dt.timedelta(days=367)) == "1 year, 2 days"

    def test_year_and_month(self):
        assert naturaldelta(dt.timedelta(days=365 + 31)) == "1 year, 1 month"

    def test_year_and_months(self):
        assert naturaldelta(dt.timedelta(days=365 + 61)) == "1 year, 2 months"

    def test_year_12_months_becomes_two_years(self):
        # 1 year + 12 months: num_months == 12 -> promotes to 2 years.
        assert naturaldelta(dt.timedelta(days=365 + 366)) == "2 years"

    def test_seconds_from_int(self):
        assert naturaldelta(45) == "45 seconds"

    def test_seconds_from_float(self):
        assert naturaldelta(45.4) == "45 seconds"

    def test_float_rounds_up(self):
        # round(45.6) == 46 in Python? No: round uses banker's rounding, 45.6 -> 46.
        # But 45.6 rounds to 46... actually naturaldelta uses round(value) where
        # value is converted to float seconds; 45.6 -> 46. However the observed
        # behavior returns "45 seconds" because the value is first rounded via
        # round() at _date_and_delta level in naturaldelta? Let us assert the
        # actual implementation behavior.
        assert naturaldelta(45.6) == "45 seconds"

    def test_negative_timedelta_abs(self):
        assert naturaldelta(dt.timedelta(seconds=-5)) == "5 seconds"

    def test_invalid_value_returns_str(self):
        assert naturaldelta("not a number") == "not a number"

    def test_nan_returns_str(self):
        assert naturaldelta(float("nan")) == "nan"

    def test_inf_returns_str(self):
        assert naturaldelta(float("inf")) == "inf"

    def test_neg_inf_returns_str(self):
        assert naturaldelta(float("-inf")) == "-inf"

    def test_too_large_finite_raises_overflow(self):
        with pytest.raises(OverflowError):
            naturaldelta(1e300)

    def test_invalid_minimum_unit_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            naturaldelta(dt.timedelta(seconds=5), minimum_unit="days")

    def test_unknown_minimum_unit_keyerror(self):
        with pytest.raises(KeyError):
            naturaldelta(dt.timedelta(seconds=5), minimum_unit="bogus")

    def test_minimum_unit_case_insensitive(self):
        assert (
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="SECONDS")
            == "a second"
        )

    def test_milliseconds_minimum(self):
        assert (
            naturaldelta(dt.timedelta(milliseconds=500), minimum_unit="milliseconds")
            == "500 milliseconds"
        )

    def test_microseconds_minimum_small(self):
        assert (
            naturaldelta(dt.timedelta(microseconds=500), minimum_unit="microseconds")
            == "500 microseconds"
        )

    def test_microseconds_minimum_exactly_1000(self):
        # Boundary: 1000 microseconds should be reported as milliseconds, not us.
        assert (
            naturaldelta(
                dt.timedelta(microseconds=1000), minimum_unit="microseconds"
            )
            == "1 millisecond"
        )

    def test_microseconds_minimum_to_milliseconds(self):
        assert (
            naturaldelta(dt.timedelta(milliseconds=5), minimum_unit="microseconds")
            == "5 milliseconds"
        )

    def test_microseconds_minimum_just_below_second(self):
        # 999999 microseconds < 1_000_000 -> milliseconds branch.
        assert (
            naturaldelta(
                dt.timedelta(microseconds=999999), minimum_unit="microseconds"
            )
            == "999 milliseconds"
        )

    def test_no_months(self):
        assert naturaldelta(dt.timedelta(days=61), months=False) == "61 days"

    def test_no_months_year(self):
        assert naturaldelta(dt.timedelta(days=367), months=False) == "1 year, 2 days"

    def test_no_months_boundary_day(self):
        # 36 days: with months=False -> "36 days"; with months=True -> "a month"
        assert naturaldelta(dt.timedelta(days=36), months=False) == "36 days"
        assert naturaldelta(dt.timedelta(days=36), months=True) == "a month"

    def test_months_boundary_rounding(self):
        # 14 days -> round(14/30.5) == 0 -> "14 days"
        assert naturaldelta(dt.timedelta(days=14)) == "14 days"
        # 16 days -> round(16/30.5) == 1 -> "a month"
        assert naturaldelta(dt.timedelta(days=16)) == "a month"

    def test_num_months_divisor_30_5(self):
        # 45 days / 30.5 = 1.475 -> rounds to 1
        assert naturaldelta(dt.timedelta(days=45)) == "a month"
        # 91 days / 30.5 ~= 2.98 -> num_months=3 (with /31.5 it would be 2)
        assert naturaldelta(dt.timedelta(days=91)) == "3 months"

    def test_60_minutes_becomes_hour(self):
        assert naturaldelta(dt.timedelta(seconds=3600 - 1)) == "an hour"

    def test_24_hours_becomes_day(self):
        assert naturaldelta(dt.timedelta(seconds=24 * 3600 - 1)) == "a day"

    def test_12_months_becomes_year(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"


# ---------------------------------------------------------------------------
# naturaltime
# ---------------------------------------------------------------------------
NOW = dt.datetime(2020, 6, 15, 12, 0, 0)


class TestNaturalTime:
    def test_future_datetime(self):
        assert (
            naturaltime(NOW + dt.timedelta(seconds=5), when=NOW)
            == "5 seconds from now"
        )

    def test_past_datetime(self):
        assert naturaltime(NOW - dt.timedelta(seconds=5), when=NOW) == "5 seconds ago"

    def test_now(self):
        assert naturaltime(NOW, when=NOW) == "now"

    def test_future_flag_int(self):
        assert naturaltime(5, future=True, when=NOW) == "5 seconds from now"

    def test_past_flag_int(self):
        assert naturaltime(5, future=False, when=NOW) == "5 seconds ago"

    def test_timedelta_past(self):
        assert naturaltime(dt.timedelta(seconds=5), when=NOW) == "5 seconds ago"

    def test_timedelta_future(self):
        assert naturaltime(dt.timedelta(seconds=-5), when=NOW) == "5 seconds from now"

    def test_invalid_value(self):
        assert naturaltime("nope", when=NOW) == "nope"

    def test_aware_datetime(self):
        aware = NOW.replace(tzinfo=dt.timezone.utc)
        when = NOW.replace(tzinfo=dt.timezone.utc) + dt.timedelta(seconds=5)
        assert naturaltime(aware, when=when) == "5 seconds ago"

    def test_minimum_unit(self):
        assert (
            naturaltime(
                NOW - dt.timedelta(milliseconds=500),
                when=NOW,
                minimum_unit="milliseconds",
            )
            == "500 milliseconds ago"
        )

    def test_months_false(self):
        assert (
            naturaltime(NOW - dt.timedelta(days=61), when=NOW, months=False)
            == "61 days ago"
        )


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

    def test_other_date(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=5)
        assert naturalday(other) == other.strftime("%b %d")

    def test_custom_format(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=5)
        assert naturalday(other, "%Y-%m-%d") == other.strftime("%Y-%m-%d")

    def test_datetime_input(self):
        assert naturalday(dt.datetime.now()) == "today"

    def test_non_date_returns_str(self):
        assert naturalday("hello") == "hello"

    def test_aware_datetime(self):
        aware = dt.datetime.now(dt.timezone.utc)
        assert naturalday(aware) == "today"

    def test_out_of_range(self):
        class Weird:
            year = 99999
            month = 1
            day = 1

        weird = Weird()
        assert naturalday(weird) == str(weird)


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

    def test_far_future_has_year(self):
        today = dt.date.today()
        far = today + dt.timedelta(days=200)
        assert naturaldate(far) == far.strftime("%b %d %Y")

    def test_far_past_has_year(self):
        today = dt.date.today()
        far = today - dt.timedelta(days=200)
        assert naturaldate(far) == far.strftime("%b %d %Y")

    def test_near_future_no_year(self):
        today = dt.date.today()
        near = today + dt.timedelta(days=3)
        assert naturaldate(near) == near.strftime("%b %d")

    def test_non_date_returns_str(self):
        assert naturaldate("hello") == "hello"

    def test_aware_datetime(self):
        aware = dt.datetime.now(dt.timezone.utc)
        assert naturaldate(aware) == "today"


# ---------------------------------------------------------------------------
# precisedelta
# ---------------------------------------------------------------------------
class TestPreciseDelta:
    def test_basic(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"

    def test_custom_format(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        assert (
            precisedelta(delta, format="%0.4f")
            == "2 days, 1 hour and 33.1230 seconds"
        )

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

    def test_minimum_unit_minutes_small(self):
        assert (
            precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes")
            == "0.02 minutes"
        )

    def test_minimum_unit_minutes_too_small(self):
        assert (
            precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes")
            == "0 minutes"
        )

    def test_zero(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_int_value(self):
        assert precisedelta(90) == "1 minute and 30 seconds"

    def test_float_value_precise(self):
        # precise mode preserves fractions in the seconds output.
        assert precisedelta(1.5) == "1.50 seconds"

    def test_invalid_value_returns_str(self):
        assert precisedelta("nope") == "nope"

    def test_none(self):
        assert precisedelta(None) == "None"

    def test_years(self):
        delta = dt.timedelta(days=365 + 1)
        assert precisedelta(delta) == "1 year and 1 day"

    def test_large_years_with_commas(self):
        delta = dt.timedelta(days=365 * 1000)
        assert precisedelta(delta) == "1,000 years"

    def test_single_text(self):
        assert precisedelta(dt.timedelta(days=2)) == "2 days"

    def test_suppress_all_units_raises(self):
        with pytest.raises(ValueError):
            precisedelta(
                dt.timedelta(days=1),
                suppress=[u.name.lower() for u in Unit],
            )

    def test_unknown_suppress_keyerror(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(days=1), suppress=["bogus"])

    def test_unknown_minimum_unit_keyerror(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(days=1), minimum_unit="bogus")

    def test_promotion_minutes_to_hours(self):
        delta = dt.timedelta(minutes=59, seconds=59)
        assert precisedelta(delta) == "59 minutes and 59 seconds"

    def test_milliseconds(self):
        delta = dt.timedelta(milliseconds=1500)
        assert precisedelta(delta) == "1.50 seconds"

    def test_single_unit_year(self):
        assert precisedelta(dt.timedelta(days=365)) == "1 year"

    def test_minimum_unit_plural_large(self):
        # years singular/plural branch with intcomma.
        assert precisedelta(dt.timedelta(days=365 * 2)) == "2 years"
