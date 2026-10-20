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
    Eastern,
    Central,
    Mountain,
    Pacific,
    dt_to_timestamp,
    isoparse,
    parse_timedelta,
    parse_td,
    decimal_relative_time,
    relative_time,
    strpdate,
    daterange,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_aware_utc():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_naive_treated_as_utc():
    dt = datetime(1970, 1, 1)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_known_epoch():
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 946684800.0


def test_dt_to_timestamp_with_utc_constant_tz():
    dt = datetime(1970, 1, 1, tzinfo=UTC)
    assert dt_to_timestamp(dt) == 0.0


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_epoch():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)


def test_isoparse_roundtrip_naive():
    dt = datetime(2020, 6, 15, 12, 34, 56)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_fractional_seconds_scaled():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_single_digit_fraction():
    assert isoparse('1970-01-01T00:00:00.5') == datetime(1970, 1, 1, 0, 0, 0, 500000)


def test_isoparse_truncates_nanoseconds():
    # 9 fractional digits truncated to microsecond precision
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_full_with_microseconds():
    result = isoparse('2016-02-14T13:14:15.123456')
    assert result == datetime(2016, 2, 14, 13, 14, 15, 123456)


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_timedelta_basic():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_timedelta_full_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)


def test_parse_timedelta_empty_returns_zero():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_alias_is_same():
    assert parse_td is parse_timedelta


def test_parse_timedelta_invalid_unit_raises():
    with pytest.raises(ValueError):
        parse_timedelta('5 fortnights')


def test_parse_timedelta_seconds_alias():
    assert parse_timedelta('30s') == timedelta(seconds=30)


def test_parse_timedelta_minutes():
    assert parse_timedelta('2 minutes') == timedelta(minutes=2)


def test_parse_timedelta_hours():
    assert parse_timedelta('1.5 hours') == timedelta(hours=1.5)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_one_day():
    now = datetime(2020, 1, 2)
    d = datetime(2020, 1, 1)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_sub_second():
    now = datetime(2020, 1, 1)
    d = now - timedelta(seconds=0.002)
    value, unit = decimal_relative_time(d, now, ndigits=5)
    assert value == 0.002
    assert unit == 'seconds'


def test_decimal_relative_time_years_negative():
    now = datetime(2020, 1, 1)
    d = now + timedelta(days=900)
    assert decimal_relative_time(d, now, ndigits=1) == (-2.5, 'years')


def test_decimal_relative_time_singular():
    now = datetime(2020, 1, 1, 0, 0, 1)
    d = datetime(2020, 1, 1, 0, 0, 0)
    assert decimal_relative_time(d, now) == (1.0, 'second')


def test_decimal_relative_time_no_cardinalize():
    now = datetime(2020, 1, 2)
    d = datetime(2020, 1, 1)
    assert decimal_relative_time(d, now, cardinalize=False) == (1.0, 'day')


def test_decimal_relative_time_plural_units():
    now = datetime(2020, 1, 1)
    d = now - timedelta(days=2)
    assert decimal_relative_time(d, now) == (2.0, 'days')


def test_decimal_relative_time_weeks():
    now = datetime(2020, 1, 15)
    d = datetime(2020, 1, 1)
    assert decimal_relative_time(d, now) == (2.0, 'weeks')


def test_decimal_relative_time_hours():
    now = datetime(2020, 1, 1, 3)
    d = datetime(2020, 1, 1)
    assert decimal_relative_time(d, now) == (3.0, 'hours')


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_past():
    now = datetime(2020, 1, 8)
    d = datetime(2020, 1, 1)
    assert relative_time(d, now) == '1 week ago'


def test_relative_time_future():
    now = datetime(2020, 1, 1)
    d = datetime(2020, 1, 8)
    assert relative_time(d, now) == '1 week from now'


def test_relative_time_with_ndigits():
    now = datetime(2020, 1, 2, 0, 0, 0)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_zero():
    now = datetime(2020, 1, 1)
    assert relative_time(now, now) == '0 seconds ago'


def test_relative_time_seconds():
    now = datetime(2020, 1, 1, 0, 0, 30)
    d = datetime(2020, 1, 1, 0, 0, 0)
    assert relative_time(d, now) == '30 seconds ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_simple():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_with_parentheses():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_ignores_time():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_ignores_microseconds():
    assert strpdate('20160101 00:00:00.001',
                    '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_returns_date_type():
    result = strpdate('2016-02-14', '%Y-%m-%d')
    assert type(result) is date


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    start = date(2015, 12, 25)
    stop = date(2016, 1, 1)
    result = list(daterange(start, stop))
    assert result == [date(2015, 12, 25) + timedelta(days=i) for i in range(7)]


def test_daterange_single_day():
    assert list(daterange(date(2015, 12, 25), date(2015, 12, 26))) == [
        date(2015, 12, 25)]


def test_daterange_excludes_stop_by_default():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 4)))
    assert result == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_daterange_inclusive():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 3),
                            inclusive=True))
    assert result == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_daterange_step_timedelta():
    start = date(2020, 1, 1)
    stop = date(2020, 1, 8)
    result = list(daterange(start, stop, step=timedelta(days=2)))
    assert result == [date(2020, 1, 1), date(2020, 1, 3),
                      date(2020, 1, 5), date(2020, 1, 7)]


def test_daterange_monthly_tuple_step_inclusive():
    result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0), inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_negative_step():
    result = list(daterange(date(2020, 1, 5), date(2020, 1, 1), step=-1))
    assert result == [date(2020, 1, 5), date(2020, 1, 4),
                      date(2020, 1, 3), date(2020, 1, 2)]


def test_daterange_negative_inclusive():
    result = list(daterange(date(2020, 1, 3), date(2020, 1, 1),
                            step=-1, inclusive=True))
    assert result == [date(2020, 1, 3), date(2020, 1, 2), date(2020, 1, 1)]


def test_daterange_wrong_direction_yields_nothing():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=-1))
    assert result == []


def test_daterange_start_equals_stop():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1))) == []


def test_daterange_start_equals_stop_inclusive():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1),
                          inclusive=True)) == [date(2020, 1, 1)]


def test_daterange_infinite_yields_lazily():
    gen = daterange(date(2020, 1, 1), None)
    first_three = [next(gen) for _ in range(3)]
    assert first_three == [date(2020, 1, 1), date(2020, 1, 2),
                           date(2020, 1, 3)]


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1), step=0))


def test_daterange_cancelling_tuple_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1),
                       step=(0, 1, -31)))


def test_daterange_bad_start_type_raises():
    with pytest.raises(TypeError):
        list(daterange('2020-01-01', date(2020, 2, 1)))


def test_daterange_bad_stop_type_raises():
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), '2020-02-01'))


def test_daterange_bad_step_type_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 2, 1), step='1'))


def test_daterange_yearly_tuple_step():
    result = list(daterange(date(2015, 1, 1), date(2018, 1, 1),
                            step=(1, 0, 0)))
    assert result == [date(2015, 1, 1), date(2016, 1, 1),
                      date(2017, 1, 1)]


def test_daterange_works_with_datetime():
    # datetime is a subclass of date
    start = datetime(2020, 1, 1)
    stop = datetime(2020, 1, 3)
    result = list(daterange(start, stop))
    assert result == [datetime(2020, 1, 1), datetime(2020, 1, 2)]


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
    off = timedelta(hours=5, minutes=30)
    tz = ConstantTZInfo('IST', off)
    assert tz.name == 'IST'
    assert tz.offset == off
    assert tz.utcoffset(None) == off
    assert tz.tzname(None) == 'IST'


def test_constant_tzinfo_utcoffset_hours():
    tz = ConstantTZInfo('X', timedelta(hours=3))
    assert tz.utcoffset_hours == 3.0


def test_constant_tzinfo_utcoffset_hours_fractional():
    tz = ConstantTZInfo('X', timedelta(minutes=-90))
    assert tz.utcoffset_hours == -1.5


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('MyTZ', timedelta(hours=1))
    r = repr(tz)
    assert 'ConstantTZInfo' in r
    assert 'MyTZ' in r


def test_utc_instance():
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


def test_epoch_aware():
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_instantiable():
    tz = LocalTZInfo()
    assert isinstance(tz, LocalTZInfo)


def test_local_tzinfo_repr():
    tz = LocalTZInfo()
    assert repr(tz) == 'LocalTZInfo()'


def test_local_tzinfo_utcoffset_standard():
    tz = LocalTZInfo()
    # Pick a winter datetime; should get some timedelta
    dt = datetime(2020, 1, 15, 12, 0, 0)
    result = tz.utcoffset(dt)
    assert isinstance(result, timedelta)


def test_local_tzinfo_dst():
    tz = LocalTZInfo()
    dt = datetime(2020, 1, 15, 12, 0, 0)
    result = tz.dst(dt)
    assert isinstance(result, timedelta)


def test_local_tzinfo_tzname():
    tz = LocalTZInfo()
    dt = datetime(2020, 1, 15, 12, 0, 0)
    name = tz.tzname(dt)
    assert isinstance(name, str)


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_repr():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_ustimezone_standard_offset():
    # Winter time - no DST
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.tzname(dt) == 'EST'
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_daylight_offset():
    # Summer time - DST
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.tzname(dt) == 'EDT'
    assert Eastern.dst(dt) == HOUR


def test_ustimezone_dst_none_returns_zero():
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive_returns_zero():
    assert Eastern.dst(datetime(2020, 7, 15, 12, 0, 0)) == ZERO


def test_ustimezone_pre_1967_no_dst():
    dt = datetime(1950, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)


def test_ustimezone_1987_2006_rules():
    # 2000: DST starts first Sunday April, ends last Sunday October
    summer = datetime(2000, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(summer) == HOUR


def test_ustimezone_1967_1986_rules():
    summer = datetime(1975, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(summer) == HOUR


def test_ustimezone_fromutc_wrong_tz():
    dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_ustimezone_fromutc_winter():
    dt = datetime(2020, 1, 15, 17, 0, 0, tzinfo=Eastern)
    # 17:00 UTC -> 12:00 EST
    converted = Eastern.fromutc(dt)
    assert converted.replace(tzinfo=None) == datetime(2020, 1, 15, 12, 0, 0)
    assert converted.tzinfo is Eastern


def test_ustimezone_fromutc_summer():
    dt = datetime(2020, 7, 15, 16, 0, 0, tzinfo=Eastern)
    # 16:00 UTC -> 12:00 EDT
    converted = Eastern.fromutc(dt)
    assert converted.replace(tzinfo=None) == datetime(2020, 7, 15, 12, 0, 0)


def test_ustimezone_fromutc_pre_1967():
    dt = datetime(1950, 1, 15, 17, 0, 0, tzinfo=Eastern)
    converted = Eastern.fromutc(dt)
    assert converted.replace(tzinfo=None) == datetime(1950, 1, 15, 12, 0, 0)


def test_ustimezone_utc_roundtrip():
    aware = datetime(2020, 6, 15, 12, 0, 0, tzinfo=Eastern)
    utc_dt = aware.astimezone(timezone.utc)
    assert utc_dt.hour == 16


def test_ustimezone_constructor_custom():
    tz = USTimeZone(-5, 'Custom', 'CSTD', 'CDST')
    assert repr(tz) == 'Custom'
    assert tz.stdoffset == timedelta(hours=-5)
    assert tz.stdname == 'CSTD'
    assert tz.dstname == 'CDST'


def test_ustimezone_offsets():
    assert Central.utcoffset(datetime(2020, 1, 15, 12, tzinfo=Central)) == \
        timedelta(hours=-6)
    assert Mountain.utcoffset(datetime(2020, 1, 15, 12, tzinfo=Mountain)) == \
        timedelta(hours=-7)
    assert Pacific.utcoffset(datetime(2020, 1, 15, 12, tzinfo=Pacific)) == \
        timedelta(hours=-8)
