import time
from datetime import date, datetime, timedelta, timezone

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
    Eastern,
    Central,
    Mountain,
    Pacific,
    EPOCH_AWARE,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_aware_utc_epoch():
    assert dt_to_timestamp(datetime(1970, 1, 1, tzinfo=timezone.utc)) == 0.0


def test_dt_to_timestamp_aware_nonzero():
    dt = datetime(1970, 1, 2, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 86400.0


def test_dt_to_timestamp_naive_assumed_utc():
    naive = datetime(1970, 1, 2)
    assert dt_to_timestamp(naive) == 86400.0


def test_dt_to_timestamp_aware_offset():
    tz = ConstantTZInfo('plus2', timedelta(hours=2))
    dt = datetime(1970, 1, 1, 2, tzinfo=tz)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_fractional():
    dt = datetime(1970, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.5


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1, 0, 0)


def test_isoparse_with_microseconds():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(
        1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_truncates_beyond_microseconds():
    # 9-digit nanosecond fraction should be truncated to 6 digits
    assert isoparse('1970-01-01T00:00:00.123456789') == datetime(
        1970, 1, 1, 0, 0, 0, 123456)


def test_isoparse_short_fraction_padded():
    assert isoparse('1970-01-01T00:00:00.1') == datetime(
        1970, 1, 1, 0, 0, 0, 100000)


def test_isoparse_round_trip_naive():
    dt = datetime(2020, 5, 6, 7, 8, 9, 123456)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_date_only():
    assert isoparse('2020-05-06') == datetime(2020, 5, 6)


# ---------------------------------------------------------------------------
# parse_timedelta / parse_td
# ---------------------------------------------------------------------------

def test_parse_timedelta_short_units():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)


def test_parse_timedelta_full_words():
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(
        days=-11, seconds=43400)


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_invalid_unit_raises():
    with pytest.raises(ValueError):
        parse_timedelta('5 fortnights')


def test_parse_td_alias():
    assert parse_td is parse_timedelta


def test_parse_timedelta_single_second():
    assert parse_timedelta('1 second') == timedelta(seconds=1)


def test_parse_timedelta_minutes():
    assert parse_timedelta('2 minutes') == timedelta(minutes=2)


def test_parse_timedelta_hours():
    assert parse_timedelta('3 hours') == timedelta(hours=3)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_small_seconds_with_ndigits():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(seconds=0.002)
    result = decimal_relative_time(d, now, ndigits=5)
    assert result == (0.002, 'seconds')


def test_decimal_relative_time_future_years_negative():
    now = datetime(2020, 1, 10, 0, 0, 0)
    other = now - timedelta(days=900)
    result = decimal_relative_time(now, other, ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_cardinalize_false():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=2)
    assert decimal_relative_time(d, now, cardinalize=False) == (2.0, 'day')


def test_decimal_relative_time_singular_one():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=1)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_weeks():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=14)
    assert decimal_relative_time(d, now) == (2.0, 'weeks')


def test_decimal_relative_time_zero():
    now = datetime(2020, 1, 10, 0, 0, 0)
    result = decimal_relative_time(now, now)
    assert result == (0.0, 'seconds')


def test_decimal_relative_time_aware_other_none():
    d = datetime.now(timezone.utc)
    result = decimal_relative_time(d)
    assert isinstance(result, tuple)
    assert isinstance(result[0], float)
    assert result[1] == 'seconds'


def test_decimal_relative_time_naive_other_none():
    d = datetime.utcnow()
    result = decimal_relative_time(d)
    assert isinstance(result, tuple)
    assert result[1] == 'seconds'


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_now():
    now = datetime(2020, 1, 10, 0, 0, 0)
    assert relative_time(now, now, ndigits=1) == '0 seconds ago'


def test_relative_time_ago():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_from_now():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_singular():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=1)
    assert relative_time(d, now) == '1 day ago'


def test_relative_time_plural():
    now = datetime(2020, 1, 10, 0, 0, 0)
    d = now - timedelta(days=3)
    assert relative_time(d, now) == '3 days ago'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_embedded_full_year():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_with_time_ignored():
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_with_microseconds_ignored():
    assert strpdate('20160101 00:00:00.001',
                    '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_invalid_raises():
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_exclusive_stop():
    start = date(2015, 12, 25)
    stop = date(2016, 1, 1)
    result = list(daterange(start, stop))
    assert result == [
        date(2015, 12, 25),
        date(2015, 12, 26),
        date(2015, 12, 27),
        date(2015, 12, 28),
        date(2015, 12, 29),
        date(2015, 12, 30),
        date(2015, 12, 31),
    ]


def test_daterange_adjacent():
    start = date(2015, 12, 25)
    stop = date(2015, 12, 26)
    assert list(daterange(start, stop)) == [date(2015, 12, 25)]


def test_daterange_month_step_inclusive():
    start = date(2017, 5, 1)
    stop = date(2017, 8, 1)
    result = list(daterange(start, stop, step=(0, 1, 0), inclusive=True))
    assert result == [
        date(2017, 5, 1),
        date(2017, 6, 1),
        date(2017, 7, 1),
        date(2017, 8, 1),
    ]


def test_daterange_inclusive_stop_not_evenly_hit():
    start = date(2017, 5, 1)
    stop = date(2017, 5, 10)
    result = list(daterange(start, stop, step=3, inclusive=True))
    assert result == [
        date(2017, 5, 1),
        date(2017, 5, 4),
        date(2017, 5, 7),
    ]


def test_daterange_timedelta_step():
    start = date(2020, 1, 1)
    stop = date(2020, 1, 6)
    result = list(daterange(start, stop, step=timedelta(days=2)))
    assert result == [
        date(2020, 1, 1),
        date(2020, 1, 3),
        date(2020, 1, 5),
    ]


def test_daterange_negative_step():
    start = date(2020, 1, 5)
    stop = date(2020, 1, 1)
    result = list(daterange(start, stop))
    assert result == [
        date(2020, 1, 5),
        date(2020, 1, 4),
        date(2020, 1, 3),
        date(2020, 1, 2),
    ]


def test_daterange_negative_step_inclusive():
    start = date(2020, 1, 5)
    stop = date(2020, 1, 1)
    result = list(daterange(start, stop, inclusive=True))
    assert result == [
        date(2020, 1, 5),
        date(2020, 1, 4),
        date(2020, 1, 3),
        date(2020, 1, 2),
        date(2020, 1, 1),
    ]


def test_daterange_step_zero_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=0))


def test_daterange_stationary_tuple_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=(0, 1, -31)))


def test_daterange_wrong_direction_yields_nothing():
    result = list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=-1))
    assert result == []


def test_daterange_start_not_date_raises():
    with pytest.raises(TypeError):
        list(daterange(123, date(2020, 1, 5)))


def test_daterange_stop_not_date_raises():
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), 'not a date'))


def test_daterange_invalid_step_type_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step='abc'))


def test_daterange_start_equals_stop():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1))) == []


def test_daterange_start_equals_stop_inclusive():
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 1),
                          inclusive=True)) == []


def test_daterange_stop_none_infinite():
    gen = daterange(date(2020, 1, 1), None)
    first = next(gen)
    second = next(gen)
    third = next(gen)
    assert first == date(2020, 1, 1)
    assert second == date(2020, 1, 2)
    assert third == date(2020, 1, 3)


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tzinfo_defaults():
    tz = ConstantTZInfo()
    assert tz.name == 'ConstantTZ'
    assert tz.offset == timedelta(0)
    assert tz.utcoffset(None) == timedelta(0)
    assert tz.tzname(None) == 'ConstantTZ'
    assert tz.dst(None) == timedelta(0)


def test_constant_tzinfo_custom():
    offset = timedelta(hours=5, minutes=30)
    tz = ConstantTZInfo('IST', offset)
    assert tz.utcoffset(None) == offset
    assert tz.tzname(None) == 'IST'
    assert tz.utcoffset_hours == 5.5


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('X', timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='X', offset=datetime.timedelta(seconds=3600))"


def test_utc_constant():
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == timedelta(0)
    assert EPOCH_AWARE == datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_offsets():
    std = LocalTZInfo._std_offset
    dst = LocalTZInfo._dst_offset
    if time.daylight:
        assert dst == timedelta(seconds=-time.altzone)
    else:
        assert dst == std


def test_local_tzinfo_is_dst_returns_bool():
    tz = LocalTZ()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    assert isinstance(tz.is_dst(dt), bool)


def test_local_tzinfo_repr():
    assert repr(LocalTZ()) == 'LocalTZInfo()'


def test_local_tzinfo_tzname():
    tz = LocalTZ()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    assert tz.tzname(dt) in time.tzname


def test_local_tzinfo_dst():
    tz = LocalTZ()
    dt = datetime(2020, 6, 15, 12, 0, 0)
    result = tz.dst(dt)
    if tz.is_dst(dt):
        assert result == tz._dst_offset - tz._std_offset
    else:
        assert result == timedelta(0)


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_repr():
    assert repr(Eastern) == 'Eastern'
    assert repr(Central) == 'Central'
    assert repr(Mountain) == 'Mountain'
    assert repr(Pacific) == 'Pacific'


def test_ustimezone_standard_offset():
    # January -> standard time
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)


def test_ustimezone_summer_dst():
    # July -> daylight time
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.dst(dt) == timedelta(hours=1)


def test_ustimezone_tzname_standard():
    dt = datetime(2020, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EST'


def test_ustimezone_tzname_dst():
    dt = datetime(2020, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EDT'


def test_ustimezone_dst_none():
    assert Eastern.dst(None) == timedelta(0)


def test_ustimezone_dst_naive():
    assert Eastern.dst(datetime(2020, 7, 15, 12, 0, 0)) == timedelta(0)


def test_ustimezone_before_1967_no_dst():
    dt = datetime(1960, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(0)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)


def test_ustimezone_dst_1967_1986():
    # mid-summer 1975 should be DST
    dt = datetime(1975, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(hours=1)


def test_ustimezone_dst_1987_2006():
    dt = datetime(1995, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(hours=1)


def test_ustimezone_dst_2007_onward():
    dt = datetime(2015, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(hours=1)


def test_ustimezone_fromutc_wrong_tz_raises():
    dt = datetime(2020, 1, 1, tzinfo=Central)
    with pytest.raises(ValueError):
        Eastern.fromutc(dt)


def test_ustimezone_fromutc_standard_offset():
    # UTC January 1 12:00 -> 07:00 EST
    dt = datetime(2020, 1, 1, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 7
    assert result.utcoffset() == timedelta(hours=-5)


def test_ustimezone_fromutc_summer():
    # UTC July 1 12:00 -> 08:00 EDT
    dt = datetime(2020, 7, 1, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 8
    assert result.utcoffset() == timedelta(hours=-4)


def test_ustimezone_fromutc_before_1967():
    dt = datetime(1960, 7, 1, 12, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result == datetime(1960, 7, 1, 7, 0, 0, tzinfo=Eastern)


def test_ustimezone_aware_datetime_astimezone_utc():
    # sanity check that converting via astimezone works with fromutc
    dt = datetime(2020, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    eastern = dt.astimezone(Eastern)
    assert eastern.hour == 8
    assert eastern.utcoffset() == timedelta(hours=-4)


def test_ustimezone_fold_handling():
    # ambiguous time in the fall
    dt = datetime(2020, 11, 1, 1, 30, 0, tzinfo=Eastern, fold=0)
    dt_fold = datetime(2020, 11, 1, 1, 30, 0, tzinfo=Eastern, fold=1)
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)
    assert Eastern.utcoffset(dt_fold) == timedelta(hours=-5)


def test_first_sunday_on_or_after_helper():
    from boltons.timeutils import _first_sunday_on_or_after
    # 2020-03-08 is a Sunday
    sunday = date(2020, 3, 8)
    assert _first_sunday_on_or_after(sunday) == sunday
    # 2020-03-07 is Saturday -> next day Sunday
    saturday = date(2020, 3, 7)
    assert _first_sunday_on_or_after(saturday) == sunday
    # 2020-03-09 Monday -> following Sunday 2020-03-15
    monday = date(2020, 3, 9)
    assert _first_sunday_on_or_after(monday) == date(2020, 3, 15)
