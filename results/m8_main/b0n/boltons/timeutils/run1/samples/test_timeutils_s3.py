import re
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from boltons.timeutils import (
    ConstantTZInfo,
    LocalTZInfo,
    USTimeZone,
    UTC,
    EPOCH_AWARE,
    ZERO,
    HOUR,
    total_seconds,
    dt_to_timestamp,
    isoparse,
    parse_timedelta,
    parse_td,
    decimal_relative_time,
    relative_time,
    strpdate,
    daterange,
    _first_sunday_on_or_after,
    Eastern,
    Central,
    Mountain,
    Pacific,
)


# ---------------------------------------------------------------------------
# total_seconds / dt_to_timestamp
# ---------------------------------------------------------------------------

def test_total_seconds_alias():
    assert total_seconds(timedelta(seconds=90)) == 90.0
    assert total_seconds(timedelta(days=1)) == 86400.0


def test_dt_to_timestamp_aware():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0
    dt2 = datetime(1970, 1, 2, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt2) == 86400.0


def test_dt_to_timestamp_aware_non_utc():
    # 1970-01-01 01:00 in +01:00 equals epoch
    tz = timezone(timedelta(hours=1))
    dt = datetime(1970, 1, 1, 1, 0, tzinfo=tz)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_naive_assumed_utc():
    dt = datetime(1970, 1, 1)
    assert dt_to_timestamp(dt) == 0.0
    dt2 = datetime(1970, 1, 1, 0, 0, 1)
    assert dt_to_timestamp(dt2) == 1.0


def test_dt_to_timestamp_fractional():
    dt = datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.5


def test_epoch_aware():
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0
    assert EPOCH_AWARE.year == 1970


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0)


def test_isoparse_fractional():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fractional_padding():
    assert isoparse('1970-01-01T00:00:00.1') == datetime(1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_truncate_beyond_microseconds():
    # extra digits past microsecond precision are truncated
    assert isoparse('1970-01-01T00:00:00.851234567') == datetime(
        1970, 1, 1, 0, 0, 0, 851234
    )


def test_isoparse_no_fraction():
    result = isoparse('2016-02-14T13:30:45')
    assert result == datetime(2016, 2, 14, 13, 30, 45)


def test_isoparse_roundtrip_isoformat():
    dt = datetime(2016, 2, 14, 13, 30, 45, 123456)
    assert isoparse(dt.isoformat()) == dt


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_td_alias():
    assert parse_td is parse_timedelta


def test_parse_timedelta_basic():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_timedelta_full_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_single_unit():
    assert parse_timedelta('5 hours') == timedelta(hours=5)
    assert parse_timedelta('2 minutes') == timedelta(minutes=2)
    assert parse_timedelta('3s') == timedelta(seconds=3)


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta('3 fortnights')


def test_parse_timedelta_float_values():
    assert parse_timedelta('1.5 days') == timedelta(days=1, hours=12)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_seconds_fraction():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(seconds=0.002)
    result, unit = decimal_relative_time(d, now, ndigits=5)
    assert result == 0.002
    assert unit == 'seconds'


def test_decimal_relative_time_years_future():
    now = datetime(2020, 1, 1, 0, 0, 0)
    other = now - timedelta(days=900)
    result = decimal_relative_time(now, other, ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_cardinalize_false():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(days=2)
    result, unit = decimal_relative_time(d, now, cardinalize=False)
    assert result == 2.0
    assert unit == 'day'


def test_decimal_relative_time_singular():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(hours=1)
    result, unit = decimal_relative_time(d, now)
    assert result == 1.0
    assert unit == 'hour'


def test_decimal_relative_time_other_none_naive():
    # other defaults to "now"; use a far-future d so result is negative
    d = datetime(2000, 1, 1)
    result, unit = decimal_relative_time(d)
    assert isinstance(result, float)
    # date is in the past -> positive diff -> "ago"
    assert result > 0


def test_decimal_relative_time_zero_diff():
    now = datetime(2020, 1, 1, 0, 0, 0)
    result, unit = decimal_relative_time(now, now)
    assert result == 0.0
    assert unit == 'seconds'


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_ago():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_from_now():
    now = datetime(2020, 1, 1, 0, 0, 0)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_zero():
    now = datetime(2020, 1, 1, 0, 0, 0)
    assert relative_time(now, now, ndigits=1) == '0 seconds ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_embedded():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_ignores_time():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_microseconds():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not-a-date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_default_step():
    christmas = date(2015, 12, 25)
    new_year = date(2016, 1, 1)
    result = list(daterange(christmas, new_year))
    assert result == [
        date(2015, 12, 25),
        date(2015, 12, 26),
        date(2015, 12, 27),
        date(2015, 12, 28),
        date(2015, 12, 29),
        date(2015, 12, 30),
        date(2015, 12, 31),
    ]


def test_daterange_single_element():
    christmas = date(2015, 12, 25)
    boxing_day = date(2015, 12, 26)
    assert list(daterange(christmas, boxing_day)) == [date(2015, 12, 25)]


def test_daterange_inclusive_step_tuple():
    result = list(
        daterange(
            date(2017, 5, 1), date(2017, 8, 1), step=(0, 1, 0), inclusive=True
        )
    )
    assert result == [
        date(2017, 5, 1),
        date(2017, 6, 1),
        date(2017, 7, 1),
        date(2017, 8, 1),
    ]


def test_daterange_step_int():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=3))
    assert result == [date(2020, 1, 1), date(2020, 1, 4), date(2020, 1, 7)]


def test_daterange_step_timedelta():
    result = list(
        daterange(date(2020, 1, 1), date(2020, 1, 4), step=timedelta(days=1))
    )
    assert result == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_daterange_inclusive_exact():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 3), inclusive=True))
    assert result == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_daterange_negative_step():
    result = list(daterange(date(2020, 1, 10), date(2020, 1, 1), step=-3))
    assert result == [date(2020, 1, 10), date(2020, 1, 7), date(2020, 1, 4)]


def test_daterange_negative_inclusive():
    result = list(
        daterange(date(2020, 1, 10), date(2020, 1, 1), step=-3, inclusive=True)
    )
    assert result == [
        date(2020, 1, 10),
        date(2020, 1, 7),
        date(2020, 1, 4),
        date(2020, 1, 1),
    ]


def test_daterange_wrong_direction_yields_nothing():
    # step pointed away from stop -> empty, like range(1, 5, -1)
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=-1)) == []


def test_daterange_start_equals_stop():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1))) == []


def test_daterange_start_equals_stop_inclusive():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1), inclusive=True)) == []


def test_daterange_invalid_start_type():
    with pytest.raises(TypeError):
        list(daterange('2020-01-01', date(2020, 1, 10)))


def test_daterange_invalid_stop_type():
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), '2020-01-10'))


def test_daterange_invalid_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=1.5))


def test_daterange_zero_step():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=0))


def test_daterange_canceling_tuple_step():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 6, 1), step=(0, 1, -31)))


def test_daterange_stop_none_infinite():
    gen = daterange(date(2020, 1, 1), None)
    first = next(gen)
    second = next(gen)
    third = next(gen)
    assert first == date(2020, 1, 1)
    assert second == date(2020, 1, 2)
    assert third == date(2020, 1, 3)
    gen.close()


# ---------------------------------------------------------------------------
# ConstantTZInfo / UTC
# ---------------------------------------------------------------------------

def test_constant_tzinfo_defaults():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == ZERO
    assert tz.utcoffset(None) == ZERO
    assert tz.dst(None) == ZERO
    assert tz.tzname(None) == 'ConstantTZ'


def test_constant_tzinfo_custom():
    tz = ConstantTZInfo('Test', timedelta(hours=3))
    assert tz.name == 'Test'
    assert tz.offset == timedelta(hours=3)
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert tz.utcoffset_hours == 3.0
    assert tz.dst(None) == ZERO
    assert tz.tzname(None) == 'Test'


def test_constant_tzinfo_negative_offset_hours():
    tz = ConstantTZInfo('Neg', timedelta(hours=-5, minutes=-30))
    assert tz.utcoffset_hours == -5.5


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('Foo', timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='Foo', offset=datetime.timedelta(seconds=3600))"


def test_utc_instance():
    assert isinstance(UTC, ConstantTZInfo)
    assert UTC.name == 'UTC'
    assert UTC.offset == ZERO
    assert UTC.utcoffset(None) == ZERO


def test_constant_tz_with_datetime():
    dt = datetime(2020, 1, 1, tzinfo=ConstantTZInfo('X', timedelta(hours=2)))
    assert dt.utcoffset() == timedelta(hours=2)


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_type():
    tz = LocalTZInfo()
    assert isinstance(tz, LocalTZInfo)


def test_local_tzinfo_repr():
    assert repr(LocalTZInfo()) == 'LocalTZInfo()'


def test_local_tzinfo_utcoffset_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    offset = tz.utcoffset(dt)
    assert isinstance(offset, timedelta)


def test_local_tzinfo_dst_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    dst = tz.dst(dt)
    assert isinstance(dst, timedelta)


def test_local_tzinfo_tzname_returns_string():
    tz = LocalTZInfo()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    name = tz.tzname(dt)
    assert isinstance(name, str)
    assert name in (time.tzname[0], time.tzname[1])


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_repr():
    tz = USTimeZone(-5, "Eastern", "EST", "EDT")
    assert repr(tz) == "Eastern"
    assert Eastern.reprname == "Eastern"


def test_ustimezone_winter_no_dst():
    # January, standard time
    dt = datetime(2020, 1, 15, 12, 0, tzinfo=Eastern)
    assert dt.dst() == ZERO
    assert dt.utcoffset() == timedelta(hours=-5)
    assert dt.tzname() == 'EST'


def test_ustimezone_summer_dst():
    # July, daylight saving time
    dt = datetime(2020, 7, 15, 12, 0, tzinfo=Eastern)
    assert dt.dst() == HOUR
    assert dt.utcoffset() == timedelta(hours=-4)
    assert dt.tzname() == 'EDT'


def test_ustimezone_dst_none():
    # dt None returns ZERO
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive():
    dt = datetime(2020, 7, 15, 12, 0)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_before_1967_no_dst():
    dt = datetime(1950, 7, 15, 12, 0, tzinfo=Eastern)
    assert dt.dst() == ZERO
    assert dt.utcoffset() == timedelta(hours=-5)


def test_ustimezone_dst_range_years():
    tz = Eastern
    # 2007+ rules
    start, end = tz._dst_range(2020)
    assert start is not None
    assert start.month == 3
    # 1987-2006
    start, end = tz._dst_range(2000)
    assert start is not None
    assert start.month == 4
    # 1967-1986
    start, end = tz._dst_range(1975)
    assert start is not None
    assert start.month == 4
    # before 1967
    start, end = tz._dst_range(1950)
    assert start is None
    assert end is None


def test_ustimezone_instances():
    for tz, hours, std, dst in [
        (Eastern, -5, 'EST', 'EDT'),
        (Central, -6, 'CST', 'CDT'),
        (Mountain, -7, 'MST', 'MDT'),
        (Pacific, -8, 'PST', 'PDT'),
    ]:
        assert tz.stdoffset == timedelta(hours=hours)
        assert tz.stdname == std
        assert tz.dstname == dst


def test_ustimezone_fromutc_wrong_tz():
    with pytest.raises(ValueError):
        Eastern.fromutc(datetime(2020, 1, 1, tzinfo=UTC))


def test_ustimezone_fromutc_winter():
    dt = datetime(2020, 1, 15, 17, 0, tzinfo=UTC)  # 12:00 EST
    local = Eastern.fromutc(dt.replace(tzinfo=Eastern))
    # fromutc expects dt.tzinfo is self; call with Eastern-marked UTC wallclock
    dt2 = datetime(2020, 1, 15, 17, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt2)
    assert result == datetime(2020, 1, 15, 12, 0, tzinfo=Eastern)


def test_ustimezone_fromutc_summer():
    dt = datetime(2020, 7, 15, 16, 0, tzinfo=Eastern)  # UTC 16:00 -> 12:00 EDT
    result = Eastern.fromutc(dt)
    assert result.hour == 12
    assert result.utcoffset() == timedelta(hours=-4)


def test_ustimezone_fromutc_before_1967():
    dt = datetime(1950, 7, 15, 17, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result == dt + timedelta(hours=-5)


def test_ustimezone_roundtrip_astimezone():
    dt = datetime(2020, 7, 15, 12, 0, tzinfo=timezone.utc)
    local = dt.astimezone(Eastern)
    assert local.hour == 8
    assert local.utcoffset() == timedelta(hours=-4)


def test_ustimezone_dst_boundary_spring_forward():
    # 2020 DST starts March 8 at 2am standard time.
    # Just before: 1:30 am EST; just after: 3:00 am EDT.
    before = datetime(2020, 3, 8, 1, 30, tzinfo=Eastern)
    assert before.dst() == ZERO
    after = datetime(2020, 3, 8, 3, 0, tzinfo=Eastern)
    assert after.dst() == HOUR


def test_ustimezone_dst_boundary_fall_back():
    # 2020 DST ends November 1 at 2am daylight time.
    before = datetime(2020, 11, 1, 0, 30, tzinfo=Eastern)
    assert before.dst() == HOUR
    after = datetime(2020, 11, 1, 3, 0, tzinfo=Eastern)
    assert after.dst() == ZERO


# ---------------------------------------------------------------------------
# _first_sunday_on_or_after helper
# ---------------------------------------------------------------------------

def test_first_sunday_on_or_after_sunday():
    sunday = datetime(2020, 3, 8)  # a Sunday
    assert _first_sunday_on_or_after(sunday) == sunday


def test_first_sunday_on_or_after_monday():
    monday = datetime(2020, 3, 9)
    result = _first_sunday_on_or_after(monday)
    assert result.weekday() == 6
    assert result == datetime(2020, 3, 15)


def test_first_sunday_on_or_after_saturday():
    saturday = datetime(2020, 3, 7)
    result = _first_sunday_on_or_after(saturday)
    assert result == datetime(2020, 3, 8)
