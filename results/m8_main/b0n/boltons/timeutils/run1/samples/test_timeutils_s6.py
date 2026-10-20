import sys
import time
from datetime import datetime, date, timedelta, timezone, tzinfo

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
    LocalTZ,
    USTimeZone,
    UTC,
    Eastern,
    Central,
    Mountain,
    Pacific,
    EPOCH_AWARE,
    total_seconds,
    ZERO,
    HOUR,
    _first_sunday_on_or_after,
    ConstantTZInfo,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_epoch_aware():
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0


def test_dt_to_timestamp_naive_treated_as_utc():
    # naive datetime is assumed UTC
    naive = datetime(1970, 1, 1, 0, 0, 0)
    assert dt_to_timestamp(naive) == 0.0


def test_dt_to_timestamp_known_value():
    dt = datetime(2011, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 1293840000.0


def test_dt_to_timestamp_naive_known_value():
    dt = datetime(2011, 1, 1, 0, 0, 0)
    assert dt_to_timestamp(dt) == 1293840000.0


def test_dt_to_timestamp_roundtrip():
    ts = 1234567890
    dt = datetime.fromtimestamp(ts, timezone.utc)
    assert dt_to_timestamp(dt) == float(ts)


def test_dt_to_timestamp_with_fractional():
    dt = datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.5


def test_total_seconds_alias():
    assert total_seconds(timedelta(seconds=5)) == 5.0


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0, 0)


def test_isoparse_with_fraction():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(
        1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_truncates_nanoseconds():
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_short_fraction_pads():
    assert isoparse('1970-01-01T00:00:00.1') == datetime(
        1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_roundtrip():
    dt = datetime(2016, 2, 14, 12, 30, 45, 123456)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_full_date():
    assert isoparse('2016-02-14') == datetime(2016, 2, 14)


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_timedelta_doctest():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_timedelta_weeks_days():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(
        days=-11, seconds=43400)


def test_parse_timedelta_alias():
    assert parse_td is parse_timedelta


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_seconds():
    assert parse_timedelta('30 seconds') == timedelta(seconds=30)


def test_parse_timedelta_decimal():
    assert parse_timedelta('1.5 hours') == timedelta(hours=1.5)


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta('5 fortnights')


def test_parse_timedelta_invalid_value():
    # no valid matches => empty timedelta, but invalid unit should raise
    with pytest.raises(ValueError):
        parse_timedelta('1x')


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

NOW = datetime(2016, 2, 14, 12, 0, 0)


def test_decimal_relative_time_days():
    result = decimal_relative_time(NOW - timedelta(days=1, seconds=3600), NOW)
    assert result == (1.0, 'day')


def test_decimal_relative_time_small():
    result = decimal_relative_time(
        NOW - timedelta(seconds=0.002), NOW, ndigits=5)
    assert result == (0.002, 'seconds')


def test_decimal_relative_time_future():
    result = decimal_relative_time(NOW, NOW - timedelta(days=900), ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_zero():
    result = decimal_relative_time(NOW, NOW)
    assert result == (0.0, 'seconds')


def test_decimal_relative_time_singular():
    result = decimal_relative_time(NOW - timedelta(seconds=1), NOW)
    assert result == (1.0, 'second')


def test_decimal_relative_time_no_cardinalize():
    result = decimal_relative_time(
        NOW - timedelta(seconds=2), NOW, cardinalize=False)
    assert result == (2.0, 'second')


def test_decimal_relative_time_minute():
    result = decimal_relative_time(NOW - timedelta(minutes=2), NOW)
    assert result == (2.0, 'minutes')


def test_decimal_relative_time_hour():
    result = decimal_relative_time(NOW - timedelta(hours=3), NOW)
    assert result == (3.0, 'hours')


def test_decimal_relative_time_week():
    result = decimal_relative_time(NOW - timedelta(weeks=2), NOW)
    assert result == (2.0, 'weeks')


def test_decimal_relative_time_month():
    result = decimal_relative_time(NOW - timedelta(days=60), NOW)
    assert result == (2.0, 'months')


def test_decimal_relative_time_year():
    result = decimal_relative_time(NOW - timedelta(days=730), NOW)
    assert result == (2.0, 'years')


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_now():
    assert relative_time(NOW, NOW, ndigits=1) == '0 seconds ago'


def test_relative_time_days():
    result = relative_time(
        NOW - timedelta(days=1, seconds=36000), NOW, ndigits=1)
    assert result == '1.4 days ago'


def test_relative_time_future():
    result = relative_time(NOW + timedelta(days=7), NOW, ndigits=1)
    assert result == '1 week from now'


def test_relative_time_singular():
    result = relative_time(NOW - timedelta(seconds=1), NOW, ndigits=1)
    assert result == '1 second ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_custom_format():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_with_time():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(
        2015, 12, 31)


def test_strpdate_with_microseconds():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(
        2016, 1, 1)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    start = date(2015, 12, 25)
    stop = date(2016, 1, 1)
    result = list(daterange(start, stop))
    assert result == [date(2015, 12, 25), date(2015, 12, 26),
                      date(2015, 12, 27), date(2015, 12, 28),
                      date(2015, 12, 29), date(2015, 12, 30),
                      date(2015, 12, 31)]


def test_daterange_single():
    result = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
    assert result == [date(2015, 12, 25)]


def test_daterange_month_step_inclusive():
    result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0), inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_month_step_exclusive():
    result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0)))
    assert result == [date(2017, 5, 1), date(2017, 6, 1), date(2017, 7, 1)]


def test_daterange_step_int():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10), step=3))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_step_timedelta():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10),
                            step=timedelta(days=3)))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_negative_step():
    result = list(daterange(date(2016, 1, 10), date(2016, 1, 1), step=-3))
    assert result == [date(2016, 1, 10), date(2016, 1, 7),
                      date(2016, 1, 4)]


def test_daterange_inclusive_even():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 7), step=3,
                            inclusive=True))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_inclusive_odd():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 8), step=3,
                            inclusive=True))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 10), step=0))


def test_daterange_self_cancelling_tuple_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 31), date(2016, 5, 1),
                       step=(0, 1, -31)))


def test_daterange_wrong_direction_is_empty():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10), step=-1))
    assert result == []


def test_daterange_year_step():
    result = list(daterange(date(2010, 1, 1), date(2014, 1, 1),
                            step=(1, 0, 0)))
    assert result == [date(2010, 1, 1), date(2011, 1, 1),
                      date(2012, 1, 1), date(2013, 1, 1)]


def test_daterange_start_type_error():
    with pytest.raises(TypeError):
        list(daterange('2016-01-01', date(2016, 1, 10)))


def test_daterange_stop_type_error():
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), '2016-01-10'))


def test_daterange_invalid_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 10), step='x'))


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tzinfo_defaults():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == ZERO
    assert tz.utcoffset(None) == ZERO
    assert tz.tzname(None) == 'ConstantTZ'
    assert tz.dst(None) == ZERO


def test_constant_tzinfo_custom():
    tz = ConstantTZInfo('Test', timedelta(hours=5))
    assert tz.utcoffset(None) == timedelta(hours=5)
    assert tz.tzname(None) == 'Test'
    assert tz.dst(None) == ZERO


def test_constant_tzinfo_utcoffset_hours():
    tz = ConstantTZInfo('X', timedelta(hours=3, minutes=30))
    assert tz.utcoffset_hours == 3.5


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('Foo', timedelta(hours=2))
    r = repr(tz)
    assert 'ConstantTZInfo' in r
    assert 'Foo' in r


def test_utc_constant():
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


def test_epoch_aware():
    assert EPOCH_AWARE.year == 1970
    assert EPOCH_AWARE.month == 1
    assert EPOCH_AWARE.day == 1


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_localtzinfo_repr():
    assert repr(LocalTZ) == 'LocalTZInfo()'


def test_localtzinfo_std_offset():
    assert LocalTZ._std_offset == timedelta(seconds=-time.timezone)


def test_localtzinfo_offsets_are_timedeltas():
    # utcoffset should return a timedelta for an arbitrary datetime
    dt = datetime(2016, 1, 1, 12, 0, 0)
    off = LocalTZ.utcoffset(dt)
    assert isinstance(off, timedelta)


def test_localtzinfo_tzname():
    dt = datetime(2016, 1, 1, 12, 0, 0)
    name = LocalTZ.tzname(dt)
    assert isinstance(name, str)


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def _aware(year, month, day, hour, tz):
    return datetime(year, month, day, hour, 0, 0, tzinfo=tz)


def test_ustimezone_repr_names():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_ustimezone_std_offset_summer():
    # July -> DST active
    dt = _aware(2016, 7, 1, 12, Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.tzname(dt) == 'EDT'


def test_ustimezone_std_offset_winter():
    dt = _aware(2016, 1, 1, 12, Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.tzname(dt) == 'EST'


def test_ustimezone_dst_none():
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive():
    assert Eastern.dst(datetime(2016, 7, 1, 12)) == ZERO


def test_ustimezone_dst_pre_1967():
    dt = _aware(1900, 7, 1, 12, Eastern)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_central_offsets():
    summer = _aware(2016, 7, 1, 12, Central)
    winter = _aware(2016, 1, 1, 12, Central)
    assert Central.utcoffset(summer) == timedelta(hours=-5)
    assert Central.utcoffset(winter) == timedelta(hours=-6)


def test_ustimezone_mountain_offsets():
    summer = _aware(2016, 7, 1, 12, Mountain)
    winter = _aware(2016, 1, 1, 12, Mountain)
    assert Mountain.utcoffset(summer) == timedelta(hours=-6)
    assert Mountain.utcoffset(winter) == timedelta(hours=-7)


def test_ustimezone_pacific_offsets():
    summer = _aware(2016, 7, 1, 12, Pacific)
    winter = _aware(2016, 1, 1, 12, Pacific)
    assert Pacific.utcoffset(summer) == timedelta(hours=-7)
    assert Pacific.utcoffset(winter) == timedelta(hours=-8)


def test_ustimezone_fromutc_wrong_tz():
    dt = _aware(2016, 7, 1, 12, Central)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_ustimezone_fromutc_winter():
    # 2016-01-01 12:00 UTC
    utc_dt = datetime(2016, 1, 1, 12, 0, 0, tzinfo=Eastern)
    local = Eastern.fromutc(utc_dt)
    # Standard time offset -5
    assert local == datetime(2016, 1, 1, 7, 0, 0, tzinfo=Eastern)


def test_ustimezone_fromutc_summer():
    utc_dt = datetime(2016, 7, 1, 12, 0, 0, tzinfo=Eastern)
    local = Eastern.fromutc(utc_dt)
    # DST offset -4
    assert local == datetime(2016, 7, 1, 8, 0, 0, tzinfo=Eastern)


def test_ustimezone_fromutc_pre_1967():
    utc_dt = datetime(1900, 7, 1, 12, 0, 0, tzinfo=Eastern)
    local = Eastern.fromutc(utc_dt)
    assert local == datetime(1900, 7, 1, 7, 0, 0, tzinfo=Eastern)


def test_ustimezone_dst_boundaries():
    # 2016: DST started March 13, 2016 2am
    # Just before start (standard time), 2016-03-13 01:59 is standard
    before = _aware(2016, 3, 13, 1, Eastern)
    # DST transition at 2am; between 2am and 3am is ambiguous, use fold
    assert Eastern.dst(_ware(2016, 3, 12, 12, Eastern)) == ZERO
    assert Eastern.dst(_ware(2016, 3, 14, 12, Eastern)) == HOUR


def _ware(*args):
    return _aware(*args)


def test_first_sunday_on_or_after():
    # 2016-03-08 is a Tuesday; first Sunday on or after is 2016-03-13
    result = _first_sunday_on_or_after(datetime(2016, 3, 8))
    assert result == datetime(2016, 3, 13)


def test_first_sunday_on_or_after_already_sunday():
    # 2016-03-13 is a Sunday
    result = _first_sunday_on_or_after(datetime(2016, 3, 13))
    assert result == datetime(2016, 3, 13)


def test_dst_range_branches():
    # year > 2006
    start, end = Eastern._dst_range(2016)
    assert start == datetime(2016, 3, 13, 2)
    assert end == datetime(2016, 11, 6, 2)
    # 1987-2006
    start, end = Eastern._dst_range(2000)
    assert start == datetime(2000, 4, 2, 2)
    assert end == datetime(2000, 10, 29, 2)
    # 1967-1986
    start, end = Eastern._dst_range(1980)
    assert start == datetime(1980, 4, 27, 2)
    assert end == datetime(1980, 10, 26, 2)
    # pre 1967
    assert Eastern._dst_range(1950) == (None, None)
