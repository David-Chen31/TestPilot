import time
from datetime import date, datetime, timedelta, timezone

import pytest

from boltons.timeutils import (
    ConstantTZInfo,
    Eastern,
    Central,
    Mountain,
    Pacific,
    LocalTZInfo,
    USTimeZone,
    UTC,
    EPOCH_AWARE,
    ZERO,
    HOUR,
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


def test_dt_to_timestamp_aware_known():
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 946684800.0


def test_dt_to_timestamp_naive_treated_as_utc():
    naive = datetime(2000, 1, 1)
    aware = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(naive) == dt_to_timestamp(aware)


def test_dt_to_timestamp_constant_tz():
    tz = ConstantTZInfo('plus2', timedelta(hours=2))
    dt = datetime(2000, 1, 1, 2, tzinfo=tz)
    # 2000-01-01 02:00 +02:00 == 2000-01-01 00:00 UTC
    assert dt_to_timestamp(dt) == 946684800.0


def test_dt_to_timestamp_microseconds():
    dt = datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.5


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_epoch():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)


def test_isoparse_with_fraction():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fraction_scaling():
    # .1 means 100000us
    assert isoparse('1970-01-01T00:00:00.1') == datetime(1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_fraction_truncation():
    # nanosecond precision truncated to microsecond
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456
    )


def test_isoparse_full_date():
    assert isoparse('2016-02-14T13:14:15') == datetime(2016, 2, 14, 13, 14, 15)


def test_isoparse_roundtrip():
    dt = datetime(2016, 2, 14, 13, 14, 15, 123456)
    assert isoparse(dt.isoformat()) == dt


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_timedelta_basic():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_timedelta_full_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_legacy_alias():
    assert parse_td is parse_timedelta


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta('5 lightyears')


def test_parse_timedelta_value():
    assert parse_timedelta('2.5 hours') == timedelta(hours=2, minutes=30)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_seconds_small():
    now = datetime(2016, 1, 1, 0, 0, 0)
    d = now - timedelta(seconds=0.002)
    assert decimal_relative_time(d, now, ndigits=5) == (0.002, 'seconds')


def test_decimal_relative_time_future():
    now = datetime(2016, 1, 1)
    other = now - timedelta(days=900)
    result = decimal_relative_time(now, other, ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_no_cardinalize():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now, cardinalize=False) == (1.0, 'day')


def test_decimal_relative_time_cardinalize_singular():
    # exactly one day -> 'day' not 'days'
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_other_none_naive():
    d = datetime.utcnow() - timedelta(days=2)
    val, unit = decimal_relative_time(d, ndigits=0)
    assert unit in ('day', 'days')
    assert val >= 1.9


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_ago():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_from_now():
    now = datetime(2016, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_zero():
    now = datetime(2016, 1, 1)
    assert relative_time(now, now, ndigits=1) == '0 seconds ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_with_parens():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_time_ignored():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_microseconds_ignored():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_returns_date_type():
    result = strpdate('2016-02-14', '%Y-%m-%d')
    assert type(result) is date


def test_strpdate_bad_format():
    with pytest.raises(ValueError):
        strpdate('not-a-date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    christmas = date(2015, 12, 25)
    new_year = date(2016, 1, 1)
    result = list(daterange(christmas, new_year))
    assert result == [date(2015, 12, 25) + timedelta(days=i) for i in range(7)]


def test_daterange_stop_not_included():
    result = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
    assert result == [date(2015, 12, 25)]


def test_daterange_inclusive_month_step():
    result = list(
        daterange(date(2017, 5, 1), date(2017, 8, 1), step=(0, 1, 0), inclusive=True)
    )
    assert result == [
        date(2017, 5, 1),
        date(2017, 6, 1),
        date(2017, 7, 1),
        date(2017, 8, 1),
    ]


def test_daterange_negative_step():
    result = list(daterange(date(2016, 1, 5), date(2016, 1, 1), step=-1))
    assert result == [
        date(2016, 1, 5),
        date(2016, 1, 4),
        date(2016, 1, 3),
        date(2016, 1, 2),
    ]


def test_daterange_negative_step_inclusive():
    result = list(daterange(date(2016, 1, 3), date(2016, 1, 1), step=-1, inclusive=True))
    assert result == [date(2016, 1, 3), date(2016, 1, 2), date(2016, 1, 1)]


def test_daterange_step_wrong_direction():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 5), step=-1))
    assert result == []


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 5), step=0))


def test_daterange_start_type_error():
    with pytest.raises(TypeError):
        list(daterange('2016-01-01', date(2016, 1, 5)))


def test_daterange_stop_type_error():
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), '2016-01-05'))


def test_daterange_bad_step():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 5), step='x'))


def test_daterange_timedelta_step():
    result = list(
        daterange(date(2016, 1, 1), date(2016, 1, 10), step=timedelta(days=3))
    )
    assert result == [date(2016, 1, 1), date(2016, 1, 4), date(2016, 1, 7)]


def test_daterange_yearly_step():
    result = list(daterange(date(2010, 1, 1), date(2016, 1, 1), step=(1, 0, 0)))
    assert result == [date(y, 1, 1) for y in range(2010, 2016)]


def test_daterange_stop_none_is_generator():
    gen = daterange(date(2016, 1, 1), None)
    first = next(gen)
    assert first == date(2016, 1, 1)
    second = next(gen)
    assert second == date(2016, 1, 2)


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tzinfo_attrs():
    tz = ConstantTZInfo('test', timedelta(hours=3))
    assert tz.name == 'test'
    assert tz.offset == timedelta(hours=3)
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert tz.dst(None) == ZERO
    assert tz.tzname(None) == 'test'


def test_constant_tzinfo_default():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == ZERO


def test_constant_tzinfo_utcoffset_hours():
    tz = ConstantTZInfo('x', timedelta(hours=2, minutes=30))
    assert tz.utcoffset_hours == 2.5


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('test', timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='test', offset=datetime.timedelta(seconds=3600))"


def test_utc_instance():
    assert isinstance(UTC, ConstantTZInfo)
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


def test_epoch_aware():
    assert EPOCH_AWARE.year == 1970
    assert EPOCH_AWARE.month == 1
    assert EPOCH_AWARE.day == 1


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_repr():
    tz = LocalTZInfo()
    assert repr(tz) == 'LocalTZInfo()'


def test_local_tzinfo_dst_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2016, 7, 1, 12, 0, 0)
    result = tz.dst(dt)
    assert isinstance(result, timedelta)
    assert result in (ZERO, tz._dst_offset - tz._std_offset)


def test_local_tzinfo_utcoffset():
    tz = LocalTZInfo()
    dt = datetime(2016, 1, 1, 12, 0, 0)
    assert tz.utcoffset(dt) in (tz._std_offset, tz._dst_offset)


def test_local_tzinfo_tzname():
    tz = LocalTZInfo()
    dt = datetime(2016, 1, 1, 12, 0, 0)
    assert tz.tzname(dt) in time.tzname


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_reprs():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_ustimezone_std_offset():
    assert Eastern.stdoffset == timedelta(hours=-5)
    assert Pacific.stdoffset == timedelta(hours=-8)


def test_ustimezone_standard_time():
    # January - standard time
    dt = datetime(2016, 1, 15, 12, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.tzname(dt) == 'EST'


def test_ustimezone_dst_time():
    # July - daylight saving time
    dt = datetime(2016, 7, 15, 12, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == HOUR
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.tzname(dt) == 'EDT'


def test_ustimezone_dst_none():
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive():
    assert Eastern.dst(datetime(2016, 7, 15)) == ZERO


def test_ustimezone_pre_1967_no_dst():
    dt = datetime(1960, 7, 15, 12, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_dst_range_old():
    # 1987-2006: April first Sunday
    start, end = Eastern._dst_range(1990)
    assert start == datetime(1990, 4, 1, 2)
    assert end == datetime(1990, 10, 28, 2)


def test_ustimezone_dst_range_mid():
    # 1967-1986: April 24 -> last Sunday of April
    start, end = Eastern._dst_range(1980)
    assert start == datetime(1980, 4, 27, 2)
    assert end == datetime(1980, 10, 26, 2)


def test_ustimezone_dst_range_none():
    start, end = Eastern._dst_range(1900)
    assert start is None
    assert end is None


def test_ustimezone_fromutc():
    # Standard time UTC -> Eastern
    dt = datetime(2016, 1, 15, 12, 0, tzinfo=timezone.utc)
    result = Eastern.fromutc(dt)
    assert result.hour == 7


def test_ustimezone_fromutc_dst():
    dt = datetime(2016, 7, 15, 12, 0, tzinfo=timezone.utc)
    result = Eastern.fromutc(dt)
    assert result.hour == 8


def test_ustimezone_fromutc_wrong_tz():
    dt = datetime(2016, 1, 15, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_ustimezone_fromutc_pre_1967():
    dt = datetime(1960, 7, 15, 12, 0, tzinfo=Eastern.replace(tzinfo=Eastern) if False else timezone.utc)
    # For pre-1967, fromutc just adds stdoffset
    result = Eastern.fromutc(dt)
    assert result.hour == 7
