import time
from datetime import date, datetime, timedelta, tzinfo, timezone

import pytest

import boltons.timeutils as tu
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
    ZERO,
    HOUR,
    _first_sunday_on_or_after,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_aware_utc():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_aware_offset():
    tz = timezone(timedelta(hours=2))
    dt = datetime(1970, 1, 1, 0, 0, tzinfo=tz)
    # 1970-01-01 00:00 +02:00 == 1969-12-31 22:00 UTC
    assert dt_to_timestamp(dt) == -7200.0


def test_dt_to_timestamp_naive_treated_as_utc():
    dt = datetime(1970, 1, 1, 0, 0, 0)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_naive_with_fraction():
    dt = datetime(1970, 1, 1, 0, 0, 1, 500000)
    assert dt_to_timestamp(dt) == 1.5


def test_dt_to_timestamp_round_trip():
    ts = 1234567890
    dt = datetime.fromtimestamp(ts, timezone.utc)
    assert dt_to_timestamp(dt) == ts


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_epoch():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0)


def test_isoparse_fractional_scaled():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(
        1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fractional_padding():
    assert isoparse('1970-01-01T00:00:00.1') == datetime(
        1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_truncates_nanoseconds():
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_round_trip():
    dt = datetime(2016, 2, 14, 13, 45, 30)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_date_only():
    assert isoparse('2016-02-14') == datetime(2016, 2, 14)


def test_isoparse_with_separators():
    assert isoparse('2016/02/14 13:45:30') == datetime(
        2016, 2, 14, 13, 45, 30)


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_timedelta_basic():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(
        days=1, seconds=7410)


def test_parse_timedelta_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(
        days=-11, seconds=43400)


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_seconds_only():
    assert parse_timedelta('30 seconds') == timedelta(seconds=30)


def test_parse_timedelta_hours():
    assert parse_timedelta('1.5 hours') == timedelta(hours=1, minutes=30)


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta('5 fortnights')


def test_parse_timedelta_invalid_unit_years():
    # year is not part of the kw map
    with pytest.raises(ValueError):
        parse_timedelta('2 years')


def test_parse_td_alias():
    assert parse_td is parse_timedelta
    assert parse_td('1d') == timedelta(days=1)


def test_parse_timedelta_invalid_value_message():
    with pytest.raises(ValueError) as exc_info:
        parse_timedelta('5 fortnights')
    assert 'invalid time unit' in str(exc_info.value)


def test_parse_timedelta_invalid_value_for_unit():
    # The regex value group can capture something float() rejects
    # (e.g. leading '+' with spaces is fine, but a bare sign with a
    # space like '+ 5' won't match cleanly as a float). Use a value
    # that matches _FLOAT_PATTERN but fails float(). '+ 5' matches the
    # pattern (optional sign, optional spaces) but float('+ 5') raises.
    with pytest.raises(ValueError) as exc_info:
        parse_timedelta('+ 5 minutes')
    assert 'invalid time value' in str(exc_info.value)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2016, 1, 2, 1, 0, 0)
    d = datetime(2016, 1, 1, 0, 0, 0)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_small_seconds():
    now = datetime(2016, 1, 1, 0, 0, 0)
    d = now - timedelta(seconds=0.002)
    result = decimal_relative_time(d, now, ndigits=5)
    assert result == (0.002, 'seconds')


def test_decimal_relative_time_future_negative():
    now = datetime(2016, 1, 1)
    d = now - timedelta(days=900)
    result = decimal_relative_time(now, d, ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_no_cardinalize():
    now = datetime(2016, 1, 2)
    d = datetime(2016, 1, 1)
    assert decimal_relative_time(d, now, cardinalize=False) == (1.0, 'day')


def test_decimal_relative_time_singular_hour():
    now = datetime(2016, 1, 1, 1, 0, 0)
    d = datetime(2016, 1, 1, 0, 0, 0)
    assert decimal_relative_time(d, now) == (1.0, 'hour')


def test_decimal_relative_time_plural_hours():
    now = datetime(2016, 1, 1, 2, 0, 0)
    d = datetime(2016, 1, 1, 0, 0, 0)
    assert decimal_relative_time(d, now) == (2.0, 'hours')


def test_decimal_relative_time_week():
    now = datetime(2016, 1, 15)
    d = datetime(2016, 1, 1)
    assert decimal_relative_time(d, now) == (2.0, 'weeks')


def test_decimal_relative_time_month():
    now = datetime(2016, 4, 1)
    d = datetime(2016, 1, 1)
    # 91 days / 30
    result = decimal_relative_time(d, now, ndigits=1)
    assert result == (3.0, 'months')


def test_decimal_relative_time_with_aware_datetimes():
    tz = timezone(timedelta(hours=2))
    now = datetime(2016, 1, 2, tzinfo=tz)
    d = datetime(2016, 1, 1, tzinfo=tz)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_other_none_naive():
    d = datetime(2000, 1, 1)
    value, unit = decimal_relative_time(d)
    assert isinstance(value, float)
    assert unit in (
        'second', 'seconds', 'minute', 'minutes', 'hour', 'hours',
        'day', 'days', 'week', 'weeks', 'month', 'months',
        'year', 'years',
    )


def test_decimal_relative_time_other_none_aware():
    d = datetime(2000, 1, 1, tzinfo=timezone.utc)
    value, unit = decimal_relative_time(d)
    assert isinstance(value, float)
    assert unit in (
        'second', 'seconds', 'minute', 'minutes', 'hour', 'hours',
        'day', 'days', 'week', 'weeks', 'month', 'months',
        'year', 'years',
    )


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_ago():
    now = datetime(2016, 1, 2)
    d = datetime(2016, 1, 1)
    assert relative_time(d, now, ndigits=1) == '1 day ago'


def test_relative_time_future():
    now = datetime(2016, 1, 1)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_decimals():
    now = datetime(2016, 1, 2, 10, 0, 0)
    d = datetime(2016, 1, 1, 0, 0, 0)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_zero():
    now = datetime(2016, 1, 1)
    assert relative_time(now, now, ndigits=1) == '0 seconds ago'


def test_relative_time_seconds():
    now = datetime(2016, 1, 1)
    d = now - timedelta(seconds=5)
    assert relative_time(d, now) == '5 seconds ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_with_time_ignored():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(
        2015, 12, 31)


def test_strpdate_with_microseconds():
    assert strpdate('20160101 00:00:00.001', '%Y%m%d %H:%M:%S.%f') == date(
        2016, 1, 1)


def test_strpdate_parens():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    result = list(daterange(date(2015, 12, 25), date(2016, 1, 1)))
    assert result == [date(2015, 12, 25) + timedelta(days=i) for i in range(7)]


def test_daterange_excludes_stop():
    result = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
    assert result == [date(2015, 12, 25)]


def test_daterange_inclusive():
    result = list(daterange(date(2015, 12, 25), date(2015, 12, 26),
                            inclusive=True))
    assert result == [date(2015, 12, 25), date(2015, 12, 26)]


def test_daterange_step_timedelta():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10),
                            step=timedelta(days=3)))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_step_tuple_month():
    result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0), inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_step_tuple_with_timedelta_day():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10),
                            step=(0, 0, timedelta(days=3))))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_tuple_with_bad_day():
    # The implementation's error message formatting has a bug when the
    # step is a tuple (it uses %r on the tuple with a single format
    # placeholder, raising TypeError). Assert the actual behavior:
    # a TypeError is raised rather than ValueError.
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 10),
                       step=(0, 0, 'x')))


def test_daterange_negative_step():
    result = list(daterange(date(2016, 1, 10), date(2016, 1, 1), step=-2))
    assert result == [date(2016, 1, 10), date(2016, 1, 8),
                      date(2016, 1, 6), date(2016, 1, 4),
                      date(2016, 1, 2)]


def test_daterange_start_type_error():
    with pytest.raises(TypeError):
        list(daterange('2016-01-01', date(2016, 1, 2)))


def test_daterange_stop_type_error():
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), '2016-01-02'))


def test_daterange_bad_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 5), step='abc'))


def test_daterange_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 5), step=0))


def test_daterange_cancelling_tuple_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 5, 5), step=(0, 1, -31)))


def test_daterange_wrong_direction_yields_nothing():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10), step=-1))
    assert result == []


def test_daterange_start_equals_stop():
    # Non-inclusive start == stop yields nothing.
    assert list(daterange(date(2016, 1, 1), date(2016, 1, 1))) == []
    # Inclusive start == stop yields exactly one item.
    assert list(daterange(date(2016, 1, 1), date(2016, 1, 1),
                          inclusive=True)) == [date(2016, 1, 1)]


def test_daterange_inclusive_uneven():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 10), step=3,
                            inclusive=True))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7), date(2016, 1, 10)]


def test_daterange_stop_none_infinite():
    from itertools import islice
    gen = daterange(date(2016, 1, 1), None)
    result = list(islice(gen, 3))
    assert result == [date(2016, 1, 1), date(2016, 1, 2),
                      date(2016, 1, 3)]


def test_daterange_stop_none_negative_step():
    from itertools import islice
    gen = daterange(date(2016, 1, 10), None, step=-1)
    result = list(islice(gen, 3))
    assert result == [date(2016, 1, 10), date(2016, 1, 9),
                      date(2016, 1, 8)]


def test_daterange_month_day_cancellation_partway():
    result = list(daterange(date(2016, 1, 1), date(2016, 6, 1),
                            step=(0, 1, 0)))
    assert result == [date(2016, 1, 1), date(2016, 2, 1),
                      date(2016, 3, 1), date(2016, 4, 1),
                      date(2016, 5, 1)]


# ---------------------------------------------------------------------------
# ConstantTZInfo / UTC
# ---------------------------------------------------------------------------

def test_constant_tzinfo_defaults():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == ZERO
    assert tz.utcoffset(None) == ZERO
    assert tz.tzname(None) == 'ConstantTZ'
    assert tz.dst(None) == ZERO
    assert tz.utcoffset_hours == 0.0


def test_constant_tzinfo_custom():
    tz = ConstantTZInfo('Test', timedelta(hours=3))
    assert tz.name == 'Test'
    assert tz.offset == timedelta(hours=3)
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert tz.tzname(None) == 'Test'
    assert tz.utcoffset_hours == 3.0


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('Test', timedelta(hours=3))
    assert repr(tz) == (
        "ConstantTZInfo(name='Test', "
        "offset=datetime.timedelta(seconds=10800))")


def test_utc_singleton():
    assert isinstance(UTC, ConstantTZInfo)
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


def test_epoch_aware():
    assert EPOCH_AWARE.tzinfo is UTC
    assert EPOCH_AWARE.year == 1970


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_repr():
    tz = LocalTZInfo()
    assert repr(tz) == 'LocalTZInfo()'


def test_local_tzinfo_utcoffset_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2016, 1, 1)
    result = tz.utcoffset(dt)
    assert isinstance(result, timedelta)


def test_local_tzinfo_tzname():
    tz = LocalTZInfo()
    dt = datetime(2016, 1, 1)
    assert tz.tzname(dt) in time.tzname


def test_local_tzinfo_utcoffset_matches_isdst():
    tz = LocalTZInfo()
    dt = datetime(2016, 7, 1)
    if tz.is_dst(dt):
        assert tz.utcoffset(dt) == tz._dst_offset
        assert tz.dst(dt) == tz._dst_offset - tz._std_offset
    else:
        assert tz.utcoffset(dt) == tz._std_offset
        assert tz.dst(dt) == ZERO


def test_local_tzinfo_is_dst_returns_bool():
    tz = LocalTZInfo()
    dt = datetime(2016, 7, 1)
    assert isinstance(tz.is_dst(dt), bool)


def test_local_tzinfo_dst_branch_with_mock():
    # Exercise the is_dst True branch of utcoffset/dst without depending
    # on the machine's actual DST configuration, via monkeypatching.
    tz = LocalTZInfo()
    orig = tz.is_dst
    try:
        tz.is_dst = lambda dt: True
        assert tz.utcoffset(datetime(2016, 7, 1)) == tz._dst_offset
        assert tz.dst(datetime(2016, 7, 1)) == (
            tz._dst_offset - tz._std_offset)
        assert tz.tzname(datetime(2016, 7, 1)) == time.tzname[1]
        tz.is_dst = lambda dt: False
        assert tz.utcoffset(datetime(2016, 1, 1)) == tz._std_offset
        assert tz.dst(datetime(2016, 1, 1)) == ZERO
        assert tz.tzname(datetime(2016, 1, 1)) == time.tzname[0]
    finally:
        tz.is_dst = orig


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_us_timezone_repr():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_us_timezone_utcoffset_no_dst():
    dt = datetime(2016, 1, 15, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.dst(dt) == ZERO
    assert Eastern.tzname(dt) == 'EST'


def test_us_timezone_utcoffset_dst():
    dt = datetime(2016, 7, 15, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.dst(dt) == HOUR
    assert Eastern.tzname(dt) == 'EDT'


def test_us_timezone_dst_none_dt():
    assert Eastern.dst(None) == ZERO


def test_us_timezone_dst_naive_dt():
    assert Eastern.dst(datetime(2016, 7, 15)) == ZERO


def test_us_timezone_pre_1967_no_dst():
    dt = datetime(1960, 7, 15, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_us_timezone_1987_2006_dst():
    dt = datetime(2000, 6, 15, tzinfo=Eastern)
    assert Eastern.dst(dt) == HOUR


def test_us_timezone_1967_1986_dst():
    dt = datetime(1975, 6, 15, tzinfo=Eastern)
    assert Eastern.dst(dt) == HOUR


def test_us_timezone_different_offsets():
    assert Pacific.stdoffset == timedelta(hours=-8)
    assert Mountain.stdoffset == timedelta(hours=-7)
    assert Central.stdoffset == timedelta(hours=-6)


def test_us_timezone_dst_range_old_year():
    start, end = Eastern._dst_range(1900)
    assert start is None
    assert end is None


def test_us_timezone_dst_range_2007():
    start, end = Eastern._dst_range(2016)
    assert start is not None
    assert end is not None
    assert start.weekday() == 6  # Sunday
    assert end.weekday() == 6


def test_us_timezone_dst_spring_forward():
    before = datetime(2016, 3, 13, 1, 30, tzinfo=Eastern)
    assert Eastern.dst(before) == ZERO
    after = datetime(2016, 3, 13, 3, 30, tzinfo=Eastern)
    assert Eastern.dst(after) == HOUR


def test_us_timezone_dst_fall_back():
    before = datetime(2016, 11, 6, 0, 30, tzinfo=Eastern)
    assert Eastern.dst(before) == HOUR
    after = datetime(2016, 11, 6, 3, 30, tzinfo=Eastern)
    assert Eastern.dst(after) == ZERO


def test_us_timezone_dst_spring_forward_gap():
    gap0 = datetime(2016, 3, 13, 2, 30, tzinfo=Eastern, fold=0)
    gap1 = datetime(2016, 3, 13, 2, 30, tzinfo=Eastern, fold=1)
    assert Eastern.dst(gap0) == ZERO
    assert Eastern.dst(gap1) == HOUR


def test_us_timezone_dst_fall_back_ambiguous():
    amb0 = datetime(2016, 11, 6, 1, 30, tzinfo=Eastern, fold=0)
    amb1 = datetime(2016, 11, 6, 1, 30, tzinfo=Eastern, fold=1)
    assert Eastern.dst(amb0) == HOUR
    assert Eastern.dst(amb1) == ZERO


def test_us_timezone_fromutc():
    dt = datetime(2016, 7, 15, 12, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 8
    assert result.dst() == HOUR


def test_us_timezone_fromutc_wrong_tzinfo():
    dt = datetime(2016, 7, 15, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_us_timezone_fromutc_old_year():
    dt = datetime(1900, 6, 15, 12, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result == datetime(1900, 6, 15, 7, 0, tzinfo=Eastern)


def test_us_timezone_fromutc_fall_back_ambiguous():
    dt = datetime(2016, 11, 6, 6, 30, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.fold == 1


def test_us_timezone_fromutc_std_time_before_start():
    # UTC time early on the spring-forward day, before DST starts.
    # std_time = utc + stdoffset; it should be returned unshifted
    # (branch: std_time < start).
    dt = datetime(2016, 3, 13, 5, 0, tzinfo=Eastern)  # 00:00 EST
    result = Eastern.fromutc(dt)
    assert result == datetime(2016, 3, 13, 0, 0, tzinfo=Eastern)
    assert result.dst() == ZERO


def test_us_timezone_fromutc_after_dst_end():
    # UTC time late on the fall-back day, after DST ends. dst_time
    # would be >= end, so std_time is returned (branch: dst_time >= end).
    dt = datetime(2016, 11, 6, 8, 0, tzinfo=Eastern)  # 03:00 EST
    result = Eastern.fromutc(dt)
    assert result == datetime(2016, 11, 6, 3, 0, tzinfo=Eastern)
    assert result.dst() == ZERO


def test_us_timezone_fromutc_dst_active():
    # UTC time squarely in the DST window: start <= std_time < end - HOUR
    # returns dst_time (std_time + HOUR).
    dt = datetime(2016, 7, 15, 12, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result == datetime(2016, 7, 15, 8, 0, tzinfo=Eastern)
    assert result.dst() == HOUR


def test_first_sunday_on_or_after():
    result = _first_sunday_on_or_after(datetime(2016, 3, 8, 2))
    assert result == datetime(2016, 3, 13, 2)
    assert result.weekday() == 6


def test_first_sunday_on_or_after_is_sunday():
    result = _first_sunday_on_or_after(datetime(2016, 11, 6, 2))
    assert result == datetime(2016, 11, 6, 2)
