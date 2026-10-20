import time
import bisect
import operator
from datetime import tzinfo, timedelta, date, datetime, timezone

import pytest

from boltons.timeutils import (
    dt_to_timestamp,
    isoparse,
    parse_timedelta,
    parse_td,
    decimal_relative_time,
    relative_time,
    strpdate,
    daterange,
    ConstantTZInfo,
    LocalTZInfo,
    USTimeZone,
    UTC,
    LocalTZ,
    EPOCH_AWARE,
    Eastern,
    Central,
    Mountain,
    Pacific,
    ZERO,
    HOUR,
    _first_sunday_on_or_after,
    _cardinalize_time_unit,
    total_seconds,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

class TestDtToTimestamp:
    def test_aware_utc_epoch(self):
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert dt_to_timestamp(dt) == 0.0

    def test_naive_epoch(self):
        assert dt_to_timestamp(datetime(1970, 1, 1)) == 0.0

    def test_aware_known_value(self):
        dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert dt_to_timestamp(dt) == 946684800.0

    def test_naive_known_value(self):
        # naive treated as UTC
        dt = datetime(2000, 1, 1)
        assert dt_to_timestamp(dt) == 946684800.0

    def test_offset_aware(self):
        tz = timezone(timedelta(hours=2))
        dt = datetime(1970, 1, 1, 2, 0, 0, tzinfo=tz)
        assert dt_to_timestamp(dt) == 0.0

    def test_microseconds(self):
        dt = datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
        assert dt_to_timestamp(dt) == 0.5

    def test_roundtrip_aware(self):
        dt = datetime(2020, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        ts = dt_to_timestamp(dt)
        back = datetime.fromtimestamp(ts, timezone.utc)
        assert back == dt

    def test_utc_constant(self):
        # UTC's offset is zero
        dt = datetime(1970, 1, 1, tzinfo=UTC)
        assert dt_to_timestamp(dt) == 0.0

    def test_negative_pre_epoch(self):
        dt = datetime(1969, 12, 31, tzinfo=timezone.utc)
        assert dt_to_timestamp(dt) == -86400.0


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

class TestIsoParse:
    def test_epoch_basic(self):
        assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)

    def test_fractional(self):
        assert isoparse('1970-01-01T00:00:00.851') == datetime(
            1970, 1, 1, 0, 0, 0, 851000)

    def test_fractional_padded(self):
        assert isoparse('1970-01-01T00:00:00.5') == datetime(
            1970, 1, 1, 0, 0, 0, 500000)

    def test_fractional_truncated_nanoseconds(self):
        assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
            1970, 1, 1, 0, 0, 0, 123456)

    def test_full(self):
        result = isoparse('2015-06-15T12:34:56')
        assert result == datetime(2015, 6, 15, 12, 34, 56)

    def test_roundtrip(self):
        dt = datetime(2018, 3, 4, 5, 6, 7, 89012)
        assert isoparse(dt.isoformat()) == dt

    def test_separator_variants(self):
        assert isoparse('2016-02-14 10:20:30') == datetime(2016, 2, 14, 10, 20, 30)


# ---------------------------------------------------------------------------
# parse_timedelta / parse_td
# ---------------------------------------------------------------------------

class TestParseTimedelta:
    def test_basic(self):
        assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)

    def test_full_words(self):
        assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)

    def test_negative(self):
        assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(
            days=-11, seconds=43400)

    def test_seconds(self):
        assert parse_timedelta('30s') == timedelta(seconds=30)

    def test_decimal_seconds(self):
        assert parse_timedelta('1.5s') == timedelta(seconds=1.5)

    def test_empty(self):
        assert parse_timedelta('') == timedelta(0)

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            parse_timedelta('1x')

    def test_legacy_alias(self):
        assert parse_td is parse_timedelta

    def test_whitespace_between(self):
        assert parse_timedelta('1 day 2 hours') == timedelta(days=1, hours=2)


# ---------------------------------------------------------------------------
# _cardinalize_time_unit
# ---------------------------------------------------------------------------

class TestCardinalize:
    def test_singular(self):
        assert _cardinalize_time_unit('day', 1) == 'day'

    def test_plural(self):
        assert _cardinalize_time_unit('day', 2) == 'days'

    def test_zero(self):
        assert _cardinalize_time_unit('day', 0) == 'days'


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

class TestDecimalRelativeTime:
    def test_days(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(days=1, hours=1)
        assert decimal_relative_time(d, now) == (1.0, 'day')

    def test_seconds_decimal(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(seconds=0.002)
        assert decimal_relative_time(d, now, ndigits=5) == (0.002, 'seconds')

    def test_negative_days(self):
        now = datetime(2020, 1, 10, 12)
        other = now - timedelta(days=900)
        assert decimal_relative_time(now, other, ndigits=1) == (-2.5, 'years')

    def test_no_cardinalize(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(days=2)
        assert decimal_relative_time(d, now, cardinalize=False) == (2.0, 'day')

    def test_minutes(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(minutes=5)
        assert decimal_relative_time(d, now) == (5.0, 'minutes')

    def test_hours(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(hours=3)
        assert decimal_relative_time(d, now) == (3.0, 'hours')

    def test_weeks(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(weeks=2)
        assert decimal_relative_time(d, now) == (2.0, 'weeks')

    def test_zero(self):
        now = datetime(2020, 1, 10, 12)
        assert decimal_relative_time(now, now) == (0.0, 'seconds')

    def test_other_default_naive(self):
        # cannot control now, just assert structure
        d = datetime(2000, 1, 1)
        val, unit = decimal_relative_time(d)
        assert unit in ('seconds', 'minutes', 'hours', 'days',
                        'weeks', 'months', 'years')
        assert isinstance(val, float)

    def test_other_default_aware(self):
        d = datetime(2000, 1, 1, tzinfo=timezone.utc)
        val, unit = decimal_relative_time(d)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

class TestRelativeTime:
    def test_seconds_ago(self):
        # Provide an explicit `other` so behavior is deterministic.
        now = datetime(2020, 1, 10, 12)
        assert relative_time(now, now, ndigits=1) == '0 seconds ago'

    def test_days_ago(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(days=1, seconds=36000)
        assert relative_time(d, now, ndigits=1) == '1.4 days ago'

    def test_week_from_now(self):
        now = datetime(2020, 1, 10, 12)
        d = now + timedelta(days=7)
        assert relative_time(d, now, ndigits=1) == '1 week from now'

    def test_hours_ago(self):
        now = datetime(2020, 1, 10, 12)
        d = now - timedelta(hours=2)
        assert relative_time(d, now) == '2 hours ago'

    def test_from_now(self):
        now = datetime(2020, 1, 10, 12)
        d = now + timedelta(days=1)
        assert relative_time(d, now) == '1 day from now'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

class TestStrpdate:
    def test_simple(self):
        assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)

    def test_with_parens(self):
        assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)

    def test_time_ignored(self):
        assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)

    def test_microseconds_ignored(self):
        assert strpdate('20160101 00:00:00.001',
                        '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)

    def test_bad_format(self):
        with pytest.raises(ValueError):
            strpdate('not a date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

class TestDaterange:
    def test_simple_range(self):
        start = date(2015, 12, 25)
        stop = date(2016, 1, 1)
        result = list(daterange(start, stop))
        assert result == [
            date(2015, 12, 26) - timedelta(days=1) + timedelta(days=i)
            for i in range(7)
        ]
        assert result[0] == date(2015, 12, 25)
        assert result[-1] == date(2015, 12, 31)

    def test_stop_exclusive(self):
        result = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
        assert result == [date(2015, 12, 25)]

    def test_inclusive(self):
        result = list(daterange(date(2015, 12, 25), date(2015, 12, 28),
                                inclusive=True))
        assert result == [date(2015, 12, 25), date(2015, 12, 26),
                          date(2015, 12, 27), date(2015, 12, 28)]

    def test_inclusive_not_even(self):
        result = list(daterange(date(2015, 12, 25), date(2015, 12, 27),
                                step=2, inclusive=True))
        assert result == [date(2015, 12, 25), date(2015, 12, 27)]

    def test_step_int(self):
        result = list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=3))
        assert result == [date(2020, 1, 1), date(2020, 1, 4),
                          date(2020, 1, 7)]

    def test_step_timedelta(self):
        result = list(daterange(date(2020, 1, 1), date(2020, 1, 5),
                                step=timedelta(days=2)))
        assert result == [date(2020, 1, 1), date(2020, 1, 3)]

    def test_step_tuple_month(self):
        result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                                step=(0, 1, 0), inclusive=True))
        assert result == [date(2017, 5, 1), date(2017, 6, 1),
                          date(2017, 7, 1), date(2017, 8, 1)]

    def test_step_tuple_year(self):
        result = list(daterange(date(2010, 1, 1), date(2013, 1, 1),
                                step=(1, 0, 0)))
        assert result == [date(2010, 1, 1), date(2011, 1, 1),
                          date(2012, 1, 1)]

    def test_negative_step(self):
        result = list(daterange(date(2020, 1, 10), date(2020, 1, 1), step=-3))
        assert result == [date(2020, 1, 10), date(2020, 1, 7),
                          date(2020, 1, 4)]

    def test_negative_step_inclusive(self):
        result = list(daterange(date(2020, 1, 10), date(2020, 1, 4), step=-2,
                                inclusive=True))
        assert result == [date(2020, 1, 10), date(2020, 1, 8),
                          date(2020, 1, 6), date(2020, 1, 4)]

    def test_wrong_direction_yields_nothing(self):
        result = list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=-1))
        assert result == []

    def test_start_not_date(self):
        with pytest.raises(TypeError):
            list(daterange('2020-01-01', date(2020, 1, 10)))

    def test_stop_not_date(self):
        with pytest.raises(TypeError):
            list(daterange(date(2020, 1, 1), '2020-01-10'))

    def test_stop_none_infinite(self):
        gen = daterange(date(2020, 1, 1), None)
        assert next(gen) == date(2020, 1, 1)
        assert next(gen) == date(2020, 1, 2)
        assert next(gen) == date(2020, 1, 3)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError):
            list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=0))

    def test_self_cancelling_tuple_raises(self):
        with pytest.raises(ValueError):
            list(daterange(date(2020, 1, 1), date(2020, 1, 5),
                           step=(0, 1, -31)))

    def test_bad_step_type(self):
        with pytest.raises(ValueError):
            list(daterange(date(2020, 1, 1), date(2020, 1, 5), step='a'))

    def test_start_equals_stop(self):
        # start == stop: nothing yielded since start is not < stop
        assert list(daterange(date(2020, 1, 1), date(2020, 1, 1))) == []

    def test_start_equals_stop_inclusive(self):
        assert list(daterange(date(2020, 1, 1), date(2020, 1, 1),
                              inclusive=True)) == [date(2020, 1, 1)]

    def test_generator_type(self):
        import types
        assert isinstance(daterange(date(2020, 1, 1), date(2020, 1, 5)),
                          types.GeneratorType)

    def test_month_step_over_year_boundary(self):
        result = list(daterange(date(2019, 11, 15), date(2020, 2, 15),
                                step=(0, 1, 0)))
        assert result == [date(2019, 11, 15), date(2019, 12, 15),
                          date(2020, 1, 15)]


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

class TestConstantTZInfo:
    def test_defaults(self):
        tz = ConstantTZInfo()
        assert tz.name == 'ConstantTZ'
        assert tz.offset == ZERO
        assert tz.utcoffset(None) == ZERO
        assert tz.tzname(None) == 'ConstantTZ'
        assert tz.dst(None) == ZERO

    def test_offset_hours(self):
        tz = ConstantTZInfo('X', timedelta(hours=3))
        assert tz.utcoffset_hours == 3.0

    def test_custom(self):
        off = timedelta(hours=-5)
        tz = ConstantTZInfo('EST5', off)
        assert tz.name == 'EST5'
        assert tz.offset == off
        assert tz.utcoffset(None) == off
        assert tz.tzname(None) == 'EST5'

    def test_repr(self):
        tz = ConstantTZInfo('ABC', timedelta(hours=1))
        assert repr(tz) == "ConstantTZInfo(name='ABC', offset=datetime.timedelta(days=..., seconds=3600))" \
            or 'ABC' in repr(tz) and 'ConstantTZInfo' in repr(tz)

    def test_utc_constant(self):
        assert UTC.name == 'UTC'
        assert UTC.offset == ZERO
        assert UTC.utcoffset(None) == ZERO
        assert UTC.tzname(None) == 'UTC'
        assert UTC.utcoffset_hours == 0.0

    def test_epoch_aware(self):
        assert EPOCH_AWARE == datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

class TestLocalTZInfo:
    def test_repr(self):
        assert repr(LocalTZ) == 'LocalTZInfo()'

    def test_is_tzinfo(self):
        assert isinstance(LocalTZ, tzinfo)

    def test_utcoffset_daylight(self):
        # is_dst can be forced via a dt; check utcoffset returns an offset
        dt = datetime(2020, 1, 1, 12)
        off = LocalTZ.utcoffset(dt)
        assert isinstance(off, timedelta)

    def test_tzname(self):
        dt = datetime(2020, 1, 1, 12)
        name = LocalTZ.tzname(dt)
        assert isinstance(name, str)


# ---------------------------------------------------------------------------
# _first_sunday_on_or_after
# ---------------------------------------------------------------------------

class TestFirstSunday:
    def test_already_sunday(self):
        # 2015-03-08 is a Sunday
        assert _first_sunday_on_or_after(datetime(2015, 3, 8)) == \
            datetime(2015, 3, 8)

    def test_mid_week(self):
        # 2015-03-11 is Wednesday
        result = _first_sunday_on_or_after(datetime(2015, 3, 11))
        assert result == datetime(2015, 3, 15)

    def test_monday(self):
        # 2015-03-09 is Monday
        assert _first_sunday_on_or_after(datetime(2015, 3, 9)) == \
            datetime(2015, 3, 15)


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

class TestUSTimeZone:
    def test_repr(self):
        assert repr(Eastern) == 'Eastern'
        assert repr(Central) == 'Central'
        assert repr(Mountain) == 'Mountain'
        assert repr(Pacific) == 'Pacific'

    def test_standard_names(self):
        assert Eastern.stdname == 'EST'
        assert Eastern.dstname == 'EDT'
        assert Central.stdname == 'CST'
        assert Mountain.stdname == 'MST'
        assert Pacific.stdname == 'PST'

    def test_std_offset(self):
        assert Eastern.stdoffset == timedelta(hours=-5)
        assert Central.stdoffset == timedelta(hours=-6)
        assert Mountain.stdoffset == timedelta(hours=-7)
        assert Pacific.stdoffset == timedelta(hours=-8)

    def test_dst_none(self):
        assert Eastern.dst(None) == ZERO
        assert Eastern.utcoffset(None) == Eastern.stdoffset

    def test_dst_naive(self):
        dt = datetime(2020, 7, 1)
        assert Eastern.dst(dt) == ZERO

    def test_dst_summer(self):
        dt = datetime(2020, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.dst(dt) == HOUR

    def test_dst_winter(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=Eastern)
        assert Eastern.dst(dt) == ZERO

    def test_tzname_summer(self):
        dt = datetime(2020, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.tzname(dt) == 'EDT'

    def test_tzname_winter(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=Eastern)
        assert Eastern.tzname(dt) == 'EST'

    def test_utcoffset_summer(self):
        dt = datetime(2020, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.utcoffset(dt) == timedelta(hours=-4)

    def test_utcoffset_winter(self):
        dt = datetime(2020, 1, 1, 12, tzinfo=Eastern)
        assert Eastern.utcoffset(dt) == timedelta(hours=-5)

    def test_dst_before_1967(self):
        dt = datetime(1960, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.dst(dt) == ZERO

    def test_dst_1987_2006(self):
        # 1990: DST starts first Sunday in April
        dt = datetime(1990, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.dst(dt) == HOUR

    def test_dst_1967_1986(self):
        # 1970: DST starts last Sunday in April (on or after Apr 24)
        dt = datetime(1970, 7, 1, 12, tzinfo=Eastern)
        assert Eastern.dst(dt) == HOUR

    def test_fromutc_wrong_tz(self):
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            Eastern.fromutc(dt)

    def test_fromutc_summer(self):
        # 2020-07-01 12:00 UTC -> 08:00 EDT
        dt = datetime(2020, 7, 1, 12, 0, tzinfo=Eastern)
        result = Eastern.fromutc(dt.replace(tzinfo=Eastern))
        assert result.tzinfo is Eastern
        assert result.hour == 8

    def test_fromutc_no_dst_range(self):
        # Year before 1967 -> no DST, just add stdoffset
        dt = datetime(1960, 7, 1, 12, tzinfo=Eastern)
        result = Eastern.fromutc(dt)
        assert result == datetime(1960, 7, 1, 7, tzinfo=Eastern)

    def test_dst_boundary_gap(self):
        # Spring forward: 2020-03-08 02:00 local standard -> 03:00 EDT.
        # 02:30 does not exist; fold=0 should be DST (per implementation)
        dt = datetime(2020, 3, 8, 2, 30, tzinfo=Eastern)
        assert Eastern.dst(dt) == ZERO

    def test_dst_boundary_gap_fold1(self):
        dt = datetime(2020, 3, 8, 2, 30, tzinfo=Eastern, fold=1)
        assert Eastern.dst(dt) == HOUR

    def test_dst_fall_back_ambiguous(self):
        # Fall back: 2020-11-01 02:00 EDT -> 01:00 EST. 01:30 ambiguous.
        dt = datetime(2020, 11, 1, 1, 30, tzinfo=Eastern)
        assert Eastern.dst(dt) == HOUR

    def test_dst_fall_back_ambiguous_fold1(self):
        dt = datetime(2020, 11, 1, 1, 30, tzinfo=Eastern, fold=1)
        assert Eastern.dst(dt) == ZERO

    def test_fromutc_fall_back(self):
        # 2020-11-01 06:00 UTC == 01:00 EST (fold=1)
        dt = datetime(2020, 11, 1, 6, 0, tzinfo=Eastern)
        result = Eastern.fromutc(dt)
        assert result.hour == 1
        assert result.fold == 1

    def test_dst_2006_year(self):
        # 2006 is in the 1987-2006 range
        start, end = Eastern._dst_range(2006)
        assert start is not None
        assert end is not None

    def test_dst_2007_year(self):
        start, end = Eastern._dst_range(2007)
        assert start is not None
        assert end is not None
        # second Sunday in March 2007 is the 11th
        assert start.day == 11
        assert start.month == 3

    def test_dst_range_pre_1967(self):
        start, end = Eastern._dst_range(1960)
        assert start is None
        assert end is None


# ---------------------------------------------------------------------------
# total_seconds alias
# ---------------------------------------------------------------------------

def test_total_seconds_alias():
    assert total_seconds is timedelta.total_seconds
    assert total_seconds(timedelta(days=1)) == 86400.0
