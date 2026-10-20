import time
import bisect
from datetime import datetime, date, timedelta, timezone

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
    Eastern,
    Central,
    Mountain,
    Pacific,
    UTC,
    EPOCH_AWARE,
    total_seconds,
    ZERO,
    HOUR,
)


# ---------------------------------------------------------------------------
# total_seconds / dt_to_timestamp
# ---------------------------------------------------------------------------

def test_total_seconds_alias():
    assert total_seconds(timedelta(seconds=5)) == 5.0


def test_dt_to_timestamp_aware_utc():
    dt = datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 1.0


def test_dt_to_timestamp_naive_assumed_utc():
    dt = datetime(1970, 1, 1, 0, 0, 10)
    assert dt_to_timestamp(dt) == 10.0


def test_dt_to_timestamp_epoch_aware():
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0


def test_dt_to_timestamp_roundtrip():
    ts = 1234567890
    utc_dt = datetime.fromtimestamp(ts, timezone.utc)
    assert dt_to_timestamp(utc_dt) == float(ts)


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0)


def test_isoparse_fractional_scaled():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fractional_padding():
    assert isoparse('1970-01-01T00:00:00.1') == datetime(1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_fractional_truncated():
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_roundtrip():
    dt = datetime(2015, 6, 30, 12, 34, 56, 789000)
    assert isoparse(dt.isoformat()) == dt


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_td_docstring():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_td_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_td_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)


def test_parse_td_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_td_single_second():
    assert parse_timedelta('5s') == timedelta(seconds=5)


def test_parse_td_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta('5 parsecs')


def test_parse_td_alias():
    assert parse_td is parse_timedelta


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now - timedelta(days=1, seconds=3600), now)
    assert res == (1.0, 'day')


def test_decimal_relative_time_subsecond():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now - timedelta(seconds=0.002), now, ndigits=5)
    assert res == (0.002, 'seconds')


def test_decimal_relative_time_years():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now, now - timedelta(days=900), ndigits=1)
    assert res == (-2.5, 'years')


def test_decimal_relative_time_singular():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now - timedelta(weeks=1), now)
    assert res == (1.0, 'week')


def test_decimal_relative_time_no_cardinalize():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now - timedelta(weeks=2), now, cardinalize=False)
    assert res == (2.0, 'week')


def test_decimal_relative_time_zero():
    now = datetime(2020, 1, 10)
    res = decimal_relative_time(now, now)
    assert res == (0.0, 'seconds')


def test_decimal_relative_time_aware():
    tz = ConstantTZInfo('X', timedelta(hours=2))
    now = datetime(2020, 1, 10, 12, 0, 0, tzinfo=tz)
    res = decimal_relative_time(now - timedelta(days=1), now)
    assert res == (1.0, 'day')


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_zero():
    now = datetime(2020, 1, 10)
    assert relative_time(now, now) == '0 seconds ago'


def test_relative_time_past():
    now = datetime(2020, 1, 10)
    assert relative_time(now - timedelta(days=1, seconds=36000), now, ndigits=1) == '1.4 days ago'


def test_relative_time_future():
    now = datetime(2020, 1, 10)
    assert relative_time(now + timedelta(days=7), now, ndigits=1) == '1 week from now'


def test_relative_time_singular():
    now = datetime(2020, 1, 10)
    assert relative_time(now - timedelta(hours=1), now) == '1 hour ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_weird():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_ignores_time():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_microseconds_ignored():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not-a-date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_daily():
    res = list(daterange(date(2015, 12, 25), date(2016, 1, 1)))
    assert res == [
        date(2015, 12, 25),
        date(2015, 12, 26),
        date(2015, 12, 27),
        date(2015, 12, 28),
        date(2015, 12, 29),
        date(2015, 12, 30),
        date(2015, 12, 31),
    ]


def test_daterange_stop_less_than_one_step():
    res = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
    assert res == [date(2015, 12, 25)]


def test_daterange_monthly_inclusive():
    res = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                        step=(0, 1, 0), inclusive=True))
    assert res == [
        date(2017, 5, 1),
        date(2017, 6, 1),
        date(2017, 7, 1),
        date(2017, 8, 1),
    ]


def test_daterange_monthly_exclusive():
    res = list(daterange(date(2017, 5, 1), date(2017, 8, 1), step=(0, 1, 0)))
    assert res == [date(2017, 5, 1), date(2017, 6, 1), date(2017, 7, 1)]


def test_daterange_timedelta_step():
    res = list(daterange(date(2017, 1, 1), date(2017, 1, 5),
                        step=timedelta(days=2)))
    assert res == [date(2017, 1, 1), date(2017, 1, 3)]


def test_daterange_negative_step():
    res = list(daterange(date(2017, 1, 5), date(2017, 1, 1), step=-1))
    assert res == [
        date(2017, 1, 5),
        date(2017, 1, 4),
        date(2017, 1, 3),
        date(2017, 1, 2),
    ]


def test_daterange_negative_step_inclusive():
    res = list(daterange(date(2017, 1, 5), date(2017, 1, 1),
                        step=-1, inclusive=True))
    assert res == [
        date(2017, 1, 5),
        date(2017, 1, 4),
        date(2017, 1, 3),
        date(2017, 1, 2),
        date(2017, 1, 1),
    ]


def test_daterange_start_not_date():
    with pytest.raises(TypeError):
        list(daterange('2017-01-01', date(2017, 1, 5)))


def test_daterange_stop_not_date():
    with pytest.raises(TypeError):
        list(daterange(date(2017, 1, 1), 'not-a-date'))


def test_daterange_stop_none_infinite():
    gen = daterange(date(2017, 1, 1), None)
    first = next(gen)
    second = next(gen)
    third = next(gen)
    assert first == date(2017, 1, 1)
    assert second == date(2017, 1, 2)
    assert third == date(2017, 1, 3)
    gen.close()


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2017, 1, 1), date(2017, 1, 10), step=0))


def test_daterange_cancelling_tuple_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2017, 1, 1), date(2017, 3, 1), step=(0, 1, -31)))


def test_daterange_step_away_from_stop_yields_nothing():
    res = list(daterange(date(2017, 1, 1), date(2017, 1, 5), step=-1))
    assert res == []


def test_daterange_invalid_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2017, 1, 1), date(2017, 1, 5), step='x'))


def test_daterange_year_step():
    res = list(daterange(date(2015, 1, 1), date(2018, 1, 1), step=(1, 0, 0)))
    assert res == [date(2015, 1, 1), date(2016, 1, 1), date(2017, 1, 1)]


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tz_attrs():
    tz = ConstantTZInfo('Test', timedelta(hours=3))
    assert tz.name == 'Test'
    assert tz.offset == timedelta(hours=3)
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert tz.tzname(None) == 'Test'
    assert tz.dst(None) == ZERO


def test_constant_tz_defaults():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == ZERO


def test_constant_tz_utcoffset_hours():
    tz = ConstantTZInfo('X', timedelta(hours=2, minutes=30))
    assert tz.utcoffset_hours == 2.5


def test_constant_tz_repr():
    tz = ConstantTZInfo('Test', timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='Test', offset=datetime.timedelta(seconds=3600))"


def test_utc_instance():
    assert UTC.name == 'UTC'
    assert UTC.offset == ZERO
    assert repr(EPOCH_AWARE) == 'datetime.datetime(1970, 1, 1, 0, 0, tzinfo=ConstantTZInfo(name=\'UTC\', offset=datetime.timedelta(0)))'


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tz_singleton_type():
    assert isinstance(LocalTZ, LocalTZInfo)
    assert repr(LocalTZ) == 'LocalTZInfo()'


def test_local_tz_offsets():
    dt = datetime(2020, 1, 1)
    tz = LocalTZ
    off = tz.utcoffset(dt)
    assert isinstance(off, timedelta)
    dst = tz.dst(dt)
    assert isinstance(dst, timedelta)
    name = tz.tzname(dt)
    assert isinstance(name, str)


def test_local_tz_explicit():
    tz = LocalTZInfo()
    dt = datetime(2020, 6, 1, 12, 0)
    assert isinstance(tz.utcoffset(dt), timedelta)
    assert isinstance(tz.dst(dt), timedelta)
    assert tz.tzname(dt) in time.tzname


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_us_timezone_reprs():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_us_timezone_names():
    winter = datetime(2020, 1, 15, 12, 0, tzinfo=Eastern)
    summer = datetime(2020, 7, 15, 12, 0, tzinfo=Eastern)
    assert Eastern.tzname(winter) == 'EST'
    assert Eastern.tzname(summer) == 'EDT'


def test_us_timezone_utcoffset():
    winter = datetime(2020, 1, 15, 12, 0, tzinfo=Eastern)
    summer = datetime(2020, 7, 15, 12, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(winter) == timedelta(hours=-5)
    assert Eastern.utcoffset(summer) == timedelta(hours=-4)


def test_us_timezone_dst_none():
    assert Eastern.dst(None) == ZERO
    assert Eastern.dst(datetime(2020, 7, 15)) == ZERO


def test_us_timezone_dst_pre_1967():
    dt = datetime(1950, 7, 15, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_us_timezone_dst_ranges():
    # 1967-1986
    assert Eastern.dst(datetime(1980, 7, 1, 12, tzinfo=Eastern)) == HOUR
    # 1987-2006
    assert Eastern.dst(datetime(2000, 7, 1, 12, tzinfo=Eastern)) == HOUR
    # 2007+
    assert Eastern.dst(datetime(2020, 7, 1, 12, tzinfo=Eastern)) == HOUR


def test_us_timezone_dst_boundaries():
    # Start boundary for 2007+: first Sunday on/after Mar 8, 2am.
    start = datetime(2020, 3, 8, 2, 0, tzinfo=Eastern)
    assert Eastern.dst(start.replace(hour=1, minute=59)) == ZERO
    assert Eastern.dst(start.replace(hour=3, minute=0)) == HOUR


def test_us_timezone_fromutc_wrong_tz():
    dt = datetime(2020, 7, 15, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_us_timezone_fromutc_basic():
    # Winter: standard time UTC -> EST
    utc_dt = datetime(2020, 1, 15, 17, 0, tzinfo=Eastern)
    result = Eastern.fromutc(utc_dt)
    assert result.hour == 12
    assert result.tzinfo is Eastern
    assert result.utcoffset() == timedelta(hours=-5)


def test_us_timezone_fromutc_summer():
    utc_dt = datetime(2020, 7, 15, 16, 0, tzinfo=Eastern)
    result = Eastern.fromutc(utc_dt)
    assert result.hour == 12
    assert result.tzinfo is Eastern
    assert result.utcoffset() == timedelta(hours=-4)


def test_us_timezone_fromutc_pre_1967():
    utc_dt = datetime(1950, 7, 15, 16, 0, tzinfo=Eastern)
    result = Eastern.fromutc(utc_dt)
    assert result.tzinfo is Eastern
    assert result.hour == 11  # -5 offset, no DST


def test_us_timezone_offsets_all():
    for tz, hrs in [(Eastern, -5), (Central, -6), (Mountain, -7), (Pacific, -8)]:
        winter = datetime(2020, 1, 15, 12, 0, tzinfo=tz)
        assert tz.utcoffset(winter) == timedelta(hours=hrs)
