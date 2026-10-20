# test_timeutils.py
import time
import bisect
from datetime import (tzinfo, timedelta, date, datetime, timezone)

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
    EPOCH_AWARE,
    Eastern,
    Central,
    Mountain,
    Pacific,
    total_seconds,
    ZERO,
    HOUR,
    LocalTZ,
    _first_sunday_on_or_after,
    DSTSTART_2007,
    DSTEND_2007,
    DSTSTART_1987_2006,
    DSTEND_1987_2006,
    DSTSTART_1967_1986,
    DSTEND_1967_1986,
)


# ---------------------------------------------------------------------------
# total_seconds legacy alias
# ---------------------------------------------------------------------------

def test_total_seconds_alias():
    assert total_seconds(timedelta(seconds=1.5)) == 1.5
    assert total_seconds(timedelta(days=1)) == 86400.0


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_aware_utc():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_aware_constant_tz():
    dt = datetime(1970, 1, 1, tzinfo=UTC)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_naive_treated_as_utc():
    dt = datetime(1970, 1, 1)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_known_value():
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 946684800.0


def test_dt_to_timestamp_with_offset():
    offset = timezone(timedelta(hours=2))
    dt = datetime(1970, 1, 1, 2, 0, 0, tzinfo=offset)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_fractional():
    dt = datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 1.5


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_epoch_naive():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0)


def test_isoparse_fractional():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fractional_short():
    assert isoparse('1970-01-01T00:00:00.5') == datetime(1970, 1, 1, 0, 0, 0, 500000)


def test_isoparse_truncates_nanoseconds():
    # 9 fractional digits: only first 6 used
    result = isoparse('1970-01-01T00:00:00.123456789')
    assert result == datetime(1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_full_datetime():
    assert isoparse('2016-02-14T12:34:56') == datetime(2016, 2, 14, 12, 34, 56)


def test_isoparse_roundtrip_isoformat():
    dt = datetime(2020, 6, 15, 10, 20, 30)
    assert isoparse(dt.isoformat()) == dt


# ---------------------------------------------------------------------------
# parse_timedelta / parse_td
# ---------------------------------------------------------------------------

def test_parse_td_single_unit():
    assert parse_timedelta('1d') == timedelta(days=1)


def test_parse_td_docstring():
    assert parse_td('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_td_words():
    assert parse_td('2 weeks 1 day') == timedelta(days=15)


def test_parse_td_negative():
    assert parse_td('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)


def test_parse_td_full_word_unit():
    assert parse_td('3 hours') == timedelta(hours=3)


def test_parse_td_seconds_unit():
    assert parse_td('90 seconds') == timedelta(seconds=90)


def test_parse_td_minutes():
    assert parse_td('5 minutes') == timedelta(minutes=5)


def test_parse_td_empty():
    assert parse_td('') == timedelta(0)


def test_parse_td_invalid_unit():
    with pytest.raises(ValueError):
        parse_td('5x')


def test_parse_td_alias_is_same():
    assert parse_td is parse_timedelta


def test_parse_td_decimal_value():
    result = parse_td('1.5h')
    assert result == timedelta(hours=1.5)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_with_other_day():
    now = datetime(2020, 1, 2)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_seconds_small():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(seconds=0.002)
    assert decimal_relative_time(d, now, ndigits=5) == (0.002, 'seconds')


def test_decimal_relative_time_negative_future():
    now = datetime(2020, 1, 1)
    other = now - timedelta(days=900)
    # diff = other - d = past - now(-ish) -> -900 days ; with ndigits=1 gives -2.5 years
    result = decimal_relative_time(now, other, ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_cardinalize_false():
    now = datetime(2020, 1, 2)
    d = now - timedelta(days=1)
    val, unit = decimal_relative_time(d, now, cardinalize=False)
    assert val == 1.0
    assert unit == 'day'


def test_decimal_relative_time_singular_cardinalize():
    now = datetime(2020, 1, 2)
    d = now - timedelta(days=1)
    val, unit = decimal_relative_time(d, now)
    assert unit == 'day'


def test_decimal_relative_time_plural_cardinalize():
    now = datetime(2020, 1, 3)
    d = now - timedelta(days=2)
    val, unit = decimal_relative_time(d, now)
    assert val == 2.0
    assert unit == 'days'


def test_decimal_relative_time_minutes():
    now = datetime(2020, 1, 1, 0, 2, 0)
    d = now - timedelta(seconds=90)
    val, unit = decimal_relative_time(d, now)
    assert unit == 'minutes'
    assert val == 1.0


def test_decimal_relative_time_weeks():
    now = datetime(2020, 1, 15)
    d = now - timedelta(days=14)
    val, unit = decimal_relative_time(d, now)
    assert val == 2.0
    assert unit == 'weeks'


def test_decimal_relative_time_hours():
    now = datetime(2020, 1, 1, 3, 0, 0)
    d = now - timedelta(hours=3)
    val, unit = decimal_relative_time(d, now)
    assert val == 3.0
    assert unit == 'hours'


def test_decimal_relative_time_months():
    now = datetime(2020, 3, 1)
    d = now - timedelta(days=60)
    val, unit = decimal_relative_time(d, now)
    assert unit == 'months'
    assert val == 2.0


def test_decimal_relative_time_zero():
    now = datetime(2020, 1, 1)
    val, unit = decimal_relative_time(now, now)
    assert val == 0.0
    assert unit == 'seconds'


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_same():
    now = datetime(2020, 1, 1)
    assert relative_time(now, now, ndigits=1) == '0 seconds ago'


def test_relative_time_past():
    now = datetime(2020, 1, 1, 12, 0, 0)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_future():
    now = datetime(2020, 1, 1)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_singular():
    now = datetime(2020, 1, 2)
    d = now - timedelta(seconds=1)
    # 1 second difference -> "1 second ago"
    assert relative_time(d, now) == '1 second ago'


def test_relative_time_future_seconds():
    now = datetime(2020, 1, 1)
    d = now + timedelta(seconds=5)
    assert relative_time(d, now) == '5 seconds from now'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_simple():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_with_literal():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_with_time():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_with_microseconds():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not-a-date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    start = date(2015, 12, 25)
    stop = date(2016, 1, 1)
    result = list(daterange(start, stop))
    assert result == [date(2015, 12, 25) + timedelta(days=i) for i in range(7)]


def test_daterange_single():
    start = date(2015, 12, 25)
    stop = date(2015, 12, 26)
    assert list(daterange(start, stop)) == [date(2015, 12, 25)]


def test_daterange_inclusive_even():
    start = date(2017, 5, 1)
    stop = date(2017, 8, 1)
    result = list(daterange(start, stop, step=(0, 1, 0), inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_exclusive_tuple():
    start = date(2017, 5, 1)
    stop = date(2017, 8, 1)
    result = list(daterange(start, stop, step=(0, 1, 0)))
    assert result == [date(2017, 5, 1), date(2017, 6, 1), date(2017, 7, 1)]


def test_daterange_timedelta_step():
    start = date(2020, 1, 1)
    stop = date(2020, 1, 10)
    result = list(daterange(start, stop, step=timedelta(days=3)))
    assert result == [date(2020, 1, 1), date(2020, 1, 4),
                      date(2020, 1, 7)]


def test_daterange_negative_step():
    start = date(2020, 1, 10)
    stop = date(2020, 1, 1)
    result = list(daterange(start, stop, step=-3))
    assert result == [date(2020, 1, 10), date(2020, 1, 7),
                      date(2020, 1, 4)]


def test_daterange_negative_timedelta():
    start = date(2020, 1, 10)
    stop = date(2020, 1, 1)
    result = list(daterange(start, stop, step=timedelta(days=-3)))
    assert result == [date(2020, 1, 10), date(2020, 1, 7),
                      date(2020, 1, 4)]


def test_daterange_empty_same_start_stop():
    d = date(2020, 1, 1)
    assert list(daterange(d, d)) == []


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1), step=0))


def test_daterange_zero_timedelta_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1),
                       step=timedelta(0)))


def test_daterange_self_cancelling_tuple_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 5, 1), step=(0, 1, -31)))


def test_daterange_wrong_direction_yields_nothing():
    # start < stop but negative step
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=-1))
    assert result == []


def test_daterange_wrong_direction_positive_yields_nothing():
    result = list(daterange(date(2020, 1, 10), date(2020, 1, 1), step=1))
    assert result == []


def test_daterange_start_type_error():
    with pytest.raises(TypeError):
        list(daterange('2020-01-01', date(2020, 2, 1)))


def test_daterange_stop_type_error():
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), '2020-02-01'))


def test_daterange_invalid_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1), step='abc'))


def test_daterange_stop_none_infinite():
    gen = daterange(date(2020, 1, 1), None)
    first = next(gen)
    second = next(gen)
    assert first == date(2020, 1, 1)
    assert second == date(2020, 1, 2)


def test_daterange_year_step():
    start = date(2015, 1, 1)
    stop = date(2020, 1, 1)
    result = list(daterange(start, stop, step=(1, 0, 0)))
    assert result == [date(2015, 1, 1), date(2016, 1, 1),
                      date(2017, 1, 1), date(2018, 1, 1),
                      date(2019, 1, 1)]


def test_daterange_inclusive_overshoot_not_returned():
    start = date(2020, 1, 1)
    stop = date(2020, 1, 10)
    result = list(daterange(start, stop, step=3, inclusive=True))
    assert result == [date(2020, 1, 1), date(2020, 1, 4),
                      date(2020, 1, 7)]


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tz_defaults():
    tz = ConstantTZInfo()
    assert tz.name == "ConstantTZ"
    assert tz.offset == ZERO


def test_constant_tz_attributes():
    off = timedelta(hours=2)
    tz = ConstantTZInfo('MyTZ', off)
    assert tz.name == 'MyTZ'
    assert tz.offset == off


def test_constant_tz_methods():
    off = timedelta(hours=3)
    tz = ConstantTZInfo('Foo', off)
    assert tz.utcoffset(None) == off
    assert tz.tzname(None) == 'Foo'
    assert tz.dst(None) == ZERO


def test_constant_tz_offset_hours():
    tz = ConstantTZInfo('X', timedelta(hours=3, minutes=30))
    assert tz.utcoffset_hours == 3.5


def test_constant_tz_repr():
    tz = ConstantTZInfo('X', timedelta(hours=1))
    r = repr(tz)
    assert r.startswith('ConstantTZInfo(')
    assert "'X'" in r


def test_utc_instance():
    assert isinstance(UTC, ConstantTZInfo)
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


def test_epoch_aware():
    assert EPOCH_AWARE == datetime(1970, 1, 1, tzinfo=UTC)
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tz_is_tzinfo():
    assert isinstance(LocalTZ, LocalTZInfo)


def test_local_tz_has_offsets():
    tz = LocalTZInfo()
    assert isinstance(tz._std_offset, timedelta)
    assert isinstance(tz._dst_offset, timedelta)


def test_local_tz_repr():
    tz = LocalTZInfo()
    assert repr(tz) == 'LocalTZInfo()'


def test_local_tz_methods_callable():
    tz = LocalTZInfo()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    # Should not raise; values valid
    off = tz.utcoffset(dt)
    dst = tz.dst(dt)
    name = tz.tzname(dt)
    assert isinstance(off, timedelta)
    assert isinstance(dst, timedelta)
    assert isinstance(name, str)


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_repr():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_ustimezone_std_offset():
    assert Eastern.stdoffset == timedelta(hours=-5)
    assert Central.stdoffset == timedelta(hours=-6)
    assert Mountain.stdoffset == timedelta(hours=-7)
    assert Pacific.stdoffset == timedelta(hours=-8)


def test_ustimezone_dst_none_returns_zero():
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive_returns_zero():
    dt = datetime(2020, 6, 15, 12, 0, 0)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_dst_outside_dst():
    # January is standard time
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_dst_inside_dst():
    # July is DST
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == HOUR


def test_ustimezone_utcoffset_std():
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)


def test_ustimezone_utcoffset_dst():
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)


def test_ustimezone_tzname_std():
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EST'


def test_ustimezone_tzname_dst():
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EDT'


def test_ustimezone_dst_before_1967():
    # Year 1960: no DST in this impl
    dt = datetime(1960, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_dst_range_none_old_years():
    start, end = Eastern._dst_range(1960)
    assert start is None
    assert end is None


def test_ustimezone_dst_range_2007():
    start, end = Eastern._dst_range(2020)
    assert start == DSTSTART_2007.replace(year=2020).replace(
        year=2020)
    # second sunday of march 2020 is March 8
    assert start == datetime(2020, 3, 8, 2, 0, 0)
    # first sunday of November 2020 is Nov 1
    assert end == datetime(2020, 11, 1, 2, 0, 0)


def test_ustimezone_dst_range_1987_2006():
    start, end = Eastern._dst_range(2000)
    # first sunday on or after April 1, 2000 is April 2
    assert start == datetime(2000, 4, 2, 2, 0, 0)
    # first sunday on or after Oct 25, 2000 is Oct 29
    assert end == datetime(2000, 10, 29, 2, 0, 0)


def test_ustimezone_dst_range_1967_1986():
    start, end = Eastern._dst_range(1980)
    # first sunday on or after April 24, 1980 is April 27
    assert start == datetime(1980, 4, 27, 2, 0, 0)
    assert end == datetime(1980, 10, 26, 2, 0, 0)


def test_ustimezone_fromutc_wrong_tz():
    dt = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_ustimezone_fromutc_old_year():
    dt = datetime(1960, 6, 1, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result == dt + Eastern.stdoffset
    assert result.tzinfo is Eastern


def test_ustimezone_fromutc_winter():
    # UTC winter time -> EST
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 7
    assert result.utcoffset() == timedelta(hours=-5)


def test_ustimezone_fromutc_summer():
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 8
    assert result.utcoffset() == timedelta(hours=-4)


def test_ustimezone_dst_assert_wrong_tz():
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(AssertionError):
        Eastern.dst(dt)


# ---------------------------------------------------------------------------
# _first_sunday_on_or_after
# ---------------------------------------------------------------------------

def test_first_sunday_on_or_after_is_sunday():
    dt = datetime(2020, 3, 1)
    result = _first_sunday_on_or_after(dt)
    assert result.weekday() == 6
    assert result >= dt


def test_first_sunday_on_or_after_same_day():
    sunday = datetime(2020, 3, 8)
    assert sunday.weekday() == 6
    assert _first_sunday_on_or_after(sunday) == sunday


# ---------------------------------------------------------------------------
# Integration: to/from UTC round-trips
# ---------------------------------------------------------------------------

def test_roundtrip_fromutc_for_summer():
    dt_utc = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    local = Eastern.fromutc(dt_utc)
    back = local.astimezone(timezone.utc)
    assert back.hour == 16


def test_ustimezone_zero_constant():
    assert ZERO == timedelta(0)
    assert HOUR == timedelta(hours=1) == timedelta(minutes=60)
