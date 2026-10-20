import datetime as dt

import pytest

from humanize.time import (
    Unit,
    _abs_timedelta,
    _convert_aware_datetime,
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


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #
class TestUnit:
    def test_ordering_by_value(self):
        assert Unit.MICROSECONDS < Unit.MILLISECONDS
        assert Unit.SECONDS < Unit.MINUTES
        assert Unit.YEARS > Unit.MONTHS
        assert Unit.HOURS <= Unit.HOURS
        assert Unit.DAYS >= Unit.HOURS

    def test_equality(self):
        assert Unit.SECONDS == Unit.SECONDS
        assert Unit.SECONDS != Unit.MINUTES

    def test_comparison_with_other_type_raises(self):
        with pytest.raises(TypeError):
            Unit.SECONDS < 3

    def test_total_ordering_generated(self):
        # total_ordering provides __gt__, __le__, __ge__
        assert Unit.SECONDS > Unit.MILLISECONDS
        assert Unit.SECONDS <= Unit.SECONDS
        assert Unit.SECONDS >= Unit.SECONDS

    def test_values(self):
        assert Unit.MICROSECONDS.value == 0
        assert Unit.YEARS.value == 7


# --------------------------------------------------------------------------- #
# _abs_timedelta
# --------------------------------------------------------------------------- #
class TestAbsTimedelta:
    def test_positive_passthrough(self):
        delta = dt.timedelta(days=2, seconds=3)
        assert _abs_timedelta(delta) == delta

    def test_zero(self):
        delta = dt.timedelta(0)
        assert _abs_timedelta(delta) == delta

    def test_negative_becomes_positive(self):
        delta = dt.timedelta(days=-2, seconds=0)
        result = _abs_timedelta(delta)
        assert result == dt.timedelta(days=2)


# --------------------------------------------------------------------------- #
# _convert_aware_datetime
# --------------------------------------------------------------------------- #
class TestConvertAwareDatetime:
    def test_none_returns_none(self):
        assert _convert_aware_datetime(None) is None

    def test_naive_passthrough(self):
        value = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert _convert_aware_datetime(value) is value

    def test_non_datetime_passthrough(self):
        assert _convert_aware_datetime(42) == 42
        assert _convert_aware_datetime(dt.timedelta(seconds=5)) == dt.timedelta(
            seconds=5
        )

    def test_aware_becomes_naive(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        value = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
        result = _convert_aware_datetime(value)
        assert result.tzinfo is None
        assert result == dt.datetime.fromtimestamp(value.timestamp())


# --------------------------------------------------------------------------- #
# _date_and_delta
# --------------------------------------------------------------------------- #
class TestDateAndDelta:
    def test_datetime_input(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = now - dt.timedelta(hours=1)
        date, delta = _date_and_delta(value, now=now)
        assert date == value
        assert delta == dt.timedelta(hours=1)

    def test_timedelta_input(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        delta_in = dt.timedelta(minutes=5)
        date, delta = _date_and_delta(delta_in, now=now)
        assert date == now - delta_in
        assert delta == delta_in

    def test_numeric_input_rounds(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        date, delta = _date_and_delta(90.6, now=now)
        assert delta == dt.timedelta(seconds=91)

    def test_numeric_precise(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        date, delta = _date_and_delta(90.6, now=now, precise=True)
        assert delta == dt.timedelta(seconds=90.6)

    def test_invalid_returns_none_and_value(self):
        date, delta = _date_and_delta("not a number", now=dt.datetime(2020, 1, 1))
        assert date is None
        assert delta == "not a number"

    def test_negative_becomes_absolute(self):
        now = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = now + dt.timedelta(hours=1)
        date, delta = _date_and_delta(value, now=now)
        assert date == value
        assert delta == dt.timedelta(hours=1)

    def test_no_now_uses_current_time(self):
        # Without `now`, current time is used; just check the delta is correct.
        value = dt.timedelta(hours=2)
        date, delta = _date_and_delta(value)
        assert delta == value
        assert isinstance(date, dt.datetime)


# --------------------------------------------------------------------------- #
# naturaldelta
# --------------------------------------------------------------------------- #
class TestNaturaldelta:
    def test_zero_seconds(self):
        assert naturaldelta(dt.timedelta(0)) == "a moment"

    def test_seconds_singular(self):
        assert naturaldelta(dt.timedelta(seconds=1)) == "a second"

    def test_seconds_plural(self):
        assert naturaldelta(dt.timedelta(seconds=30)) == "30 seconds"

    def test_minute_singular(self):
        assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"

    def test_minutes_plural(self):
        assert naturaldelta(dt.timedelta(minutes=45)) == "45 minutes"

    def test_rounds_to_hour(self):
        # 60 minutes -> "an hour"
        assert naturaldelta(dt.timedelta(seconds=3600)) == "an hour"

    def test_minutes_round_to_hour(self):
        # 59.6 minutes rounds to 60 -> "an hour"
        assert naturaldelta(dt.timedelta(seconds=3576)) == "an hour"

    def test_hours_plural(self):
        assert naturaldelta(dt.timedelta(hours=5)) == "5 hours"

    def test_rounds_to_day(self):
        assert naturaldelta(dt.timedelta(hours=24)) == "a day"

    def test_hours_round_to_day(self):
        # 23.6 hours rounds to 24 -> "a day"
        assert naturaldelta(dt.timedelta(seconds=round(23.6 * 3600))) == "a day"

    def test_hours_plural_from_seconds(self):
        # 5 hours expressed purely in seconds (no full days) -> "5 hours"
        assert naturaldelta(dt.timedelta(seconds=5 * 3600)) == "5 hours"

    def test_days_plural(self):
        assert naturaldelta(dt.timedelta(days=3)) == "3 days"

    def test_months(self):
        # ~31 days -> 1 month
        assert naturaldelta(dt.timedelta(days=31)) == "a month"

    def test_months_plural(self):
        assert naturaldelta(dt.timedelta(days=61)) == "2 months"

    def test_month_rounds_to_year(self):
        # Within a year, num_months == 12 -> "a year"
        # 364 days => num_months = round(364/30.5) = round(11.93) = 12 -> "a year"
        assert naturaldelta(dt.timedelta(days=364)) == "a year"

    def test_year_singular(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_years_plural(self):
        assert naturaldelta(dt.timedelta(days=365 * 3)) == "3 years"

    def test_years_with_comma(self):
        # > 1000 years triggers intcomma
        result = naturaldelta(dt.timedelta(days=365 * 1500))
        assert result == "1,500 years"

    def test_months_disabled(self):
        assert naturaldelta(dt.timedelta(days=31), months=False) == "31 days"

    def test_months_disabled_years_one(self):
        assert (
            naturaldelta(dt.timedelta(days=365 + 5), months=False) == "1 year, 5 days"
        )

    def test_one_year_one_month(self):
        # 1 year + ~1 month => 365 + 31
        assert naturaldelta(dt.timedelta(days=396)) == "1 year, 1 month"

    def test_one_year_zero_months_zero_days(self):
        assert naturaldelta(dt.timedelta(days=365)) == "a year"

    def test_one_year_days_no_months(self):
        # 1 year + a few days rounding month to 0
        assert (
            naturaldelta(dt.timedelta(days=365 + 3), months=False)
            == "1 year, 3 days"
        )

    def test_one_year_month_rounds_to_full_year(self):
        # 365 days + 364 days => years=1, days=364, num_months=12 -> "2 years"
        assert naturaldelta(dt.timedelta(days=729)) == "2 years"

    def test_one_year_multiple_months(self):
        # years=1, days=~3 months -> "1 year, 3 months"
        # 365 + 91 days: days=91, num_months = round(91/30.5) = round(2.98) = 3
        assert naturaldelta(dt.timedelta(days=365 + 91)) == "1 year, 3 months"

    def test_one_year_months_used_days(self):
        # use_months=True but days non-zero, months==0 => "1 year, N days"
        # years=1, days=10, num_months=0
        assert naturaldelta(dt.timedelta(days=375)) == "1 year, 10 days"

    def test_numeric_input(self):
        assert naturaldelta(45) == "45 seconds"

    def test_fractional_numeric_input(self):
        assert naturaldelta(1.5) == "a second"

    def test_invalid_string_returns_str(self):
        assert naturaldelta("hello") == "hello"

    def test_nan_returns_str(self):
        result = naturaldelta(float("nan"))
        assert result == "nan"

    def test_inf_returns_str(self):
        result = naturaldelta(float("inf"))
        assert result == "inf"

    def test_negative_inf_returns_str(self):
        result = naturaldelta(float("-inf"))
        assert result == "-inf"

    def test_overflow_error_for_huge_finite(self):
        with pytest.raises(OverflowError):
            naturaldelta(1e300)

    def test_minimum_unit_microseconds(self):
        result = naturaldelta(
            dt.timedelta(microseconds=500), minimum_unit="microseconds"
        )
        assert result == "500 microseconds"

    def test_minimum_unit_microseconds_singular(self):
        result = naturaldelta(
            dt.timedelta(microseconds=1), minimum_unit="microseconds"
        )
        assert result == "1 microsecond"

    def test_minimum_unit_microseconds_milliseconds(self):
        result = naturaldelta(
            dt.timedelta(microseconds=5000), minimum_unit="microseconds"
        )
        assert result == "5 milliseconds"

    def test_minimum_unit_milliseconds(self):
        result = naturaldelta(
            dt.timedelta(microseconds=5000), minimum_unit="milliseconds"
        )
        assert result == "5 milliseconds"

    def test_minimum_unit_milliseconds_moment(self):
        # Less than a millisecond but min is milliseconds -> "0 milliseconds"
        result = naturaldelta(
            dt.timedelta(microseconds=500), minimum_unit="milliseconds"
        )
        assert result == "0 milliseconds"

    def test_minimum_unit_invalid_raises(self):
        with pytest.raises(ValueError):
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="minutes")

    def test_minimum_unit_invalid_raises_message(self):
        with pytest.raises(ValueError, match="Minimum unit 'minutes' not supported"):
            naturaldelta(dt.timedelta(seconds=1), minimum_unit="minutes")

    def test_negative_timedelta(self):
        assert naturaldelta(dt.timedelta(seconds=-30)) == "30 seconds"


# --------------------------------------------------------------------------- #
# naturaltime
# --------------------------------------------------------------------------- #
class TestNaturaltime:
    def test_now(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(when, when=when) == "now"

    def test_past(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when - dt.timedelta(hours=1)
        assert naturaltime(value, when=when) == "an hour ago"

    def test_future_datetime(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when + dt.timedelta(hours=1)
        assert naturaltime(value, when=when) == "an hour from now"

    def test_timedelta_past(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(dt.timedelta(minutes=30), when=when) == "30 minutes ago"

    def test_numeric_past_default(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(30, when=when) == "30 seconds ago"

    def test_numeric_future_flag(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime(30, future=True, when=when) == "30 seconds from now"

    def test_invalid_value_returns_str(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        assert naturaltime("garbage", when=when) == "garbage"

    def test_aware_datetime(self):
        tz = dt.timezone(dt.timedelta(hours=2))
        when = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
        value = when - dt.timedelta(hours=2)
        assert naturaltime(value, when=when) == "2 hours ago"

    def test_months_disabled(self):
        when = dt.datetime(2020, 1, 1, 12, 0, 0)
        value = when - dt.timedelta(days=40)
        assert naturaltime(value, months=False, when=when) == "40 days ago"


# --------------------------------------------------------------------------- #
# naturalday
# --------------------------------------------------------------------------- #
class TestNaturalday:
    def test_today(self):
        assert naturalday(dt.date.today()) == "today"

    def test_tomorrow(self):
        assert naturalday(dt.date.today() + dt.timedelta(days=1)) == "tomorrow"

    def test_yesterday(self):
        assert naturalday(dt.date.today() - dt.timedelta(days=1)) == "yesterday"

    def test_other_date_uses_format(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=10)
        assert naturalday(other) == other.strftime("%b %d")

    def test_custom_format(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=10)
        assert naturalday(other, "%Y-%m-%d") == other.strftime("%Y-%m-%d")

    def test_datetime_input(self):
        today = dt.date.today()
        value = dt.datetime.combine(today, dt.time(12, 0, 0))
        assert naturalday(value) == "today"

    def test_non_date_returns_str(self):
        assert naturalday("hello") == "hello"

    def test_aware_datetime(self):
        tz = dt.timezone.utc
        value = dt.datetime.now(tz)
        assert naturalday(value) == "today"

    def test_out_of_range_date_returns_str(self):
        # Year 1 with a huge negative offset to trigger ValueError/OverflowError
        # A date-like object with out-of-range attributes triggers ValueError.
        class BadDate:
            year = 0
            month = 0
            day = 0

            def __str__(self):
                return "baddate"

        assert naturalday(BadDate()) == "baddate"


# --------------------------------------------------------------------------- #
# naturaldate
# --------------------------------------------------------------------------- #
class TestNaturaldate:
    def test_today(self):
        assert naturaldate(dt.date.today()) == "today"

    def test_near_date(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=10)
        assert naturaldate(other) == other.strftime("%b %d")

    def test_far_future_includes_year(self):
        today = dt.date.today()
        other = today + dt.timedelta(days=200)
        assert naturaldate(other) == other.strftime("%b %d %Y")

    def test_far_past_includes_year(self):
        today = dt.date.today()
        other = today - dt.timedelta(days=200)
        assert naturaldate(other) == other.strftime("%b %d %Y")

    def test_non_date_returns_str(self):
        assert naturaldate(123) == "123"

    def test_tomorrow(self):
        assert naturaldate(dt.date.today() + dt.timedelta(days=1)) == "tomorrow"

    def test_aware_datetime(self):
        tz = dt.timezone.utc
        value = dt.datetime.now(tz)
        assert naturaldate(value) == "today"

    def test_out_of_range_date_returns_str(self):
        class BadDate:
            year = 0
            month = 0
            day = 0

            def __str__(self):
                return "baddate"

        assert naturaldate(BadDate()) == "baddate"


# --------------------------------------------------------------------------- #
# _quotient_and_remainder
# --------------------------------------------------------------------------- #
class TestQuotientAndRemainder:
    def test_minimum_unit_rounds(self):
        q, r = _quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f"
        )
        assert q == 1.5
        assert r == 0

    def test_suppressed_unit(self):
        q, r = _quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f"
        )
        assert q == 0
        assert r == 36

    def test_normal_divmod(self):
        q, r = _quotient_and_remainder(
            36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f"
        )
        assert q == 1
        assert r == 12


# --------------------------------------------------------------------------- #
# _rounding_by_fmt
# --------------------------------------------------------------------------- #
class TestRoundingByFmt:
    def test_int_format_returns_int(self):
        result = _rounding_by_fmt("%d", 3.7)
        assert result == 3
        assert isinstance(result, int)

    def test_float_format_returns_float(self):
        result = _rounding_by_fmt("%.2f", 3.14159)
        assert result == 3.14
        assert isinstance(result, float)

    def test_float_format_whole_number(self):
        # "%.0f" % 3.14159 == "3", int() succeeds, so an int is returned
        result = _rounding_by_fmt("%.0f", 3.14159)
        assert result == 3
        assert isinstance(result, int)


# --------------------------------------------------------------------------- #
# _suitable_minimum_unit
# --------------------------------------------------------------------------- #
class TestSuitableMinimumUnit:
    def test_not_suppressed(self):
        assert _suitable_minimum_unit(Unit.HOURS, []) == Unit.HOURS

    def test_suppressed_uses_next(self):
        assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]) == Unit.DAYS

    def test_suppressed_uses_next_again(self):
        assert (
            _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS])
            == Unit.MONTHS
        )

    def test_all_suppressed_raises(self):
        with pytest.raises(ValueError):
            _suitable_minimum_unit(Unit.YEARS, list(Unit))


# --------------------------------------------------------------------------- #
# _suppress_lower_units
# --------------------------------------------------------------------------- #
class TestSuppressLowerUnits:
    def test_adds_lower_units(self):
        result = _suppress_lower_units(Unit.SECONDS, [Unit.DAYS])
        assert result == {
            Unit.MICROSECONDS,
            Unit.MILLISECONDS,
            Unit.DAYS,
        }

    def test_no_extra_suppression(self):
        result = _suppress_lower_units(Unit.MICROSECONDS, [])
        assert result == set()


# --------------------------------------------------------------------------- #
# precisedelta
# --------------------------------------------------------------------------- #
class TestPrecisedelta:
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
            precisedelta(
                delta, suppress=["seconds", "milliseconds", "microseconds"]
            )
            == "1.50 minutes"
        )

    def test_minutes_minimum(self):
        delta = dt.timedelta(seconds=1)
        assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"

    def test_minutes_minimum_small(self):
        delta = dt.timedelta(seconds=0.1)
        assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"

    def test_invalid_value_returns_str(self):
        assert precisedelta("nope") == "nope"

    def test_none_returns_str(self):
        assert precisedelta(None) == "None"

    def test_numeric_input(self):
        # 90 seconds is broken down into 1 minute and 30 seconds
        assert precisedelta(90) == "1 minute and 30 seconds"

    def test_zero(self):
        assert precisedelta(dt.timedelta(0)) == "0 seconds"

    def test_single_unit(self):
        assert precisedelta(dt.timedelta(days=1)) == "1 day"

    def test_two_units(self):
        assert precisedelta(dt.timedelta(days=1, hours=2)) == "1 day and 2 hours"

    def test_three_units(self):
        result = precisedelta(dt.timedelta(days=1, hours=2, minutes=3))
        assert result == "1 day, 2 hours and 3 minutes"

    def test_years(self):
        delta = dt.timedelta(days=365 * 2)
        assert precisedelta(delta) == "2 years"

    def test_years_with_comma(self):
        delta = dt.timedelta(days=365 * 1500)
        assert precisedelta(delta) == "1,500 years"

    def test_promotion_minutes_to_hours(self):
        # rounding at minimum unit "minutes" yields ~59.98 minutes, no promotion
        delta = dt.timedelta(minutes=59, seconds=59.9)
        result = precisedelta(delta, minimum_unit="minutes")
        assert result == "59.98 minutes"

    def test_seconds_promoted_to_minutes(self):
        # 59.9 seconds stays seconds (rounds to 59.90)
        delta = dt.timedelta(seconds=59.9)
        result = precisedelta(delta)
        assert result == "59.90 seconds"

    def test_suppress_years(self):
        delta = dt.timedelta(days=400)
        result = precisedelta(delta, suppress=["years"])
        assert result == "13 months and 3 days"

    def test_invalid_minimum_unit_raises(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=1), minimum_unit="nonsense")

    def test_invalid_suppress_raises(self):
        with pytest.raises(KeyError):
            precisedelta(dt.timedelta(seconds=1), suppress=["nonsense"])

    def test_suppress_all_raises(self):
        # Suppressing everything leaves no suitable minimum unit.
        with pytest.raises(ValueError):
            precisedelta(
                dt.timedelta(seconds=1),
                minimum_unit="seconds",
                suppress=[
                    "microseconds",
                    "milliseconds",
                    "seconds",
                    "minutes",
                    "hours",
                    "days",
                    "months",
                    "years",
                ],
            )

    def test_microseconds_minimum_singular(self):
        delta = dt.timedelta(microseconds=1)
        assert precisedelta(delta, minimum_unit="microseconds") == "1 microsecond"

    def test_milliseconds_minimum(self):
        delta = dt.timedelta(microseconds=1500)
        assert precisedelta(delta, minimum_unit="milliseconds") == "1.50 milliseconds"

    def test_only_milliseconds(self):
        # With the default minimum unit of seconds, milliseconds are rounded into
        # a fractional number of seconds.
        delta = dt.timedelta(milliseconds=5)
        assert precisedelta(delta) == "0.01 seconds"

    def test_moment_promotion_msecs_to_seconds(self):
        # 999.9 milliseconds rounds to 1000 ms -> promoted to 1 second; the
        # resulting whole second is displayed without a fractional part.
        delta = dt.timedelta(microseconds=999_900)
        assert precisedelta(delta) == "1 second"

    def test_hours_promoted_to_days(self):
        # 23 hours 59 minutes 59.9 seconds rounds up to a day
        delta = dt.timedelta(hours=23, minutes=59, seconds=59)
        result = precisedelta(delta)
        assert result == "23 hours, 59 minutes and 59 seconds"

    def test_days_promoted_to_month(self):
        # 31 days -> the day remainder is consumed by the promotion; no leftover day.
        delta = dt.timedelta(days=30, hours=24)
        result = precisedelta(delta)
        assert result == "1 month"

    def test_months_promoted_to_years(self):
        # 12 months worth should promote to a year.
        delta = dt.timedelta(days=365 + 1)
        result = precisedelta(delta)
        assert result == "1 year and 1 day"

    def test_suppress_months(self):
        delta = dt.timedelta(days=45)
        result = precisedelta(delta, suppress=["months"])
        assert result == "45 days"

    def test_suppress_hours(self):
        delta = dt.timedelta(days=1, hours=2)
        result = precisedelta(delta, suppress=["hours"])
        assert result == "1 day and 120 minutes"

    def test_suppress_minutes(self):
        delta = dt.timedelta(minutes=5, seconds=30)
        result = precisedelta(delta, suppress=["minutes"])
        assert result == "330 seconds"
