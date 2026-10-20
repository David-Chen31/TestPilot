import math
from datetime import date, datetime, timedelta, timezone

import pytest

from boltons.timeutils import (
    ConstantTZInfo,
    LocalTZ,
    LocalTZInfo,
    USTimeZone,
    UTC,
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
    EPOCH_AWARE,
    ZERO,
    HOUR,
)
import boltons.timeutils as tu


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_epoch_aware():
    assert dt_to_timestamp(EPOCH_AWARE) == 0.0


def test_dt_to_timestamp_naive_epoch():
    naive_epoch = datetime(1970, 1, 1)
    assert dt_to_timestamp(naive_epoch) == 0.0


def test_dt_to_timestamp_aware_offset():
    dt = datetime(1970, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 3600.0


def test_dt_to_timestamp_known_date():
    # 2000-03-01 00:00 UTC
    dt = datetime(2000, 3, 1, tzinfo=timezone.utc)
    expected = datetime(2000, 3, 1) - datetime(1970, 1, 1)
    assert dt_to_timestamp(dt) == expected.total_seconds()


def test_dt_to_timestamp_naive_treated_as_utc():
    naive = datetime(2000, 3, 1)
    aware = datetime(2000, 3, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(naive) == dt_to_timestamp(aware)


def test_dt_to_timestamp_with_non_utc_tz():
    tz = ConstantTZInfo('Plus5', timedelta(hours=5))
    dt = datetime(1970, 1, 1, 5, 0, 0, tzinfo=tz)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_fractional():
    dt = EPOCH_AWARE + timedelta(seconds=1.5)
    assert dt_to_timestamp(dt) == 1.5


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)


def test_isoparse_with_fraction():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(
        1970, 1, 1, 0, 0, 0, 851000)


def test_isoparse_fraction_truncated():
    # more than 6 digits: truncate beyond microsecond
    assert isoparse('2016-01-01T00:00:00.123456789') == datetime(
        2016, 1, 1, 0, 0, 0, 123456)


def test_isoparse_short_fraction_padded():
    assert isoparse('2016-01-01T00:00:00.1') == datetime(
        2016, 1, 1, 0, 0, 0, 100000)


def test_isoparse_full_datetime():
    iso = '2016-12-31T23:59:58'
    assert isoparse(iso) == datetime(2016, 12, 31, 23, 59, 58)


def test_isoparse_roundtrip():
    dt = datetime(2015, 6, 15, 12, 34, 56, 789012)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_date_only():
    # Just four parts (date) should still work
    assert isoparse('2016-02-14') == datetime(2016, 2, 14)


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


def test_parse_timedelta_alias():
    assert parse_td is parse_timedelta
    assert parse_td('1h') == timedelta(hours=1)


def test_parse_timedelta_single_units():
    assert parse_timedelta('1s') == timedelta(seconds=1)
    assert parse_timedelta('1m') == timedelta(minutes=1)
    assert parse_timedelta('1h') == timedelta(hours=1)
    assert parse_timedelta('1d') == timedelta(days=1)
    assert parse_timedelta('1w') == timedelta(weeks=1)


def test_parse_timedelta_decimal():
    assert parse_timedelta('1.5h') == timedelta(hours=1, minutes=30)


def test_parse_timedelta_empty():
    assert parse_timedelta('') == timedelta(0)


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError) as excinfo:
        parse_timedelta('1x')
    assert 'invalid time unit' in str(excinfo.value)


def test_parse_timedelta_exponent():
    assert parse_timedelta('1e2s') == timedelta(seconds=100)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_basic():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_seconds():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(seconds=0.002)
    result = decimal_relative_time(d, now, ndigits=5)
    assert result == (0.002, 'seconds')


def test_decimal_relative_time_future():
    now = datetime(2016, 1, 1)
    result = decimal_relative_time(now, now - timedelta(days=900),
                                   ndigits=1)
    assert result == (-2.5, 'years')


def test_decimal_relative_time_cardinalize_false():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(seconds=30)
    assert decimal_relative_time(d, now, cardinalize=False) == (30.0, 'second')


def test_decimal_relative_time_singular():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(days=1)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_minutes():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(minutes=3)
    assert decimal_relative_time(d, now) == (3.0, 'minutes')


def test_decimal_relative_time_hours():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(hours=5)
    assert decimal_relative_time(d, now) == (5.0, 'hours')


def test_decimal_relative_time_weeks():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(days=14)
    assert decimal_relative_time(d, now) == (2.0, 'weeks')


def test_decimal_relative_time_months():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(days=90)
    assert decimal_relative_time(d, now) == (3.0, 'months')


def test_decimal_relative_time_aware_no_other():
    # Naive "now" defaults; just test it produces something sensible
    d = datetime(2000, 1, 1)
    result = decimal_relative_time(d)
    assert result[1] in ('years', 'year')
    assert result[0] > 0


def test_decimal_relative_time_aware_other():
    tz = timezone.utc
    now = datetime(2016, 1, 1, tzinfo=tz)
    d = now - timedelta(days=1)
    assert decimal_relative_time(d, now) == (1.0, 'day')


def test_decimal_relative_time_zero():
    now = datetime(2016, 1, 1, 12, 0, 0)
    assert decimal_relative_time(now, now) == (0.0, 'seconds')


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_zero():
    now = datetime(2016, 1, 1, 12, 0, 0)
    assert relative_time(now, ndigits=1, other=now) == '0 seconds ago'


def test_relative_time_ago():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == '1.4 days ago'


def test_relative_time_from_now():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == '1 week from now'


def test_relative_time_singular():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now - timedelta(hours=1)
    assert relative_time(d, now) == '1 hour ago'


def test_relative_time_future_seconds():
    now = datetime(2016, 1, 1, 12, 0, 0)
    d = now + timedelta(seconds=30)
    assert relative_time(d, now) == '30 seconds from now'


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)


def test_strpdate_custom_format():
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)


def test_strpdate_with_time_ignored():
    assert strpdate('20151231 23:59:59',
                    '%Y%m%d %H:%M:%S') == date(2015, 12, 31)


def test_strpdate_with_microseconds():
    assert strpdate('20160101 00:00:00.001',
                    '%Y%m%d %H:%M:%S.%f') == date(2016, 1, 1)


def test_strpdate_invalid():
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
    result = list(daterange(date(2015, 12, 25), date(2016, 1, 1)))
    assert result == [date(2015, 12, 25), date(2015, 12, 26),
                      date(2015, 12, 27), date(2015, 12, 28),
                      date(2015, 12, 29), date(2015, 12, 30),
                      date(2015, 12, 31)]


def test_daterange_single():
    result = list(daterange(date(2015, 12, 25), date(2015, 12, 26)))
    assert result == [date(2015, 12, 25)]


def test_daterange_inclusive_step_month():
    result = list(daterange(date(2017, 5, 1), date(2017, 8, 1),
                            step=(0, 1, 0), inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 6, 1),
                      date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_inclusive_not_even():
    # stop not included when step doesn't land on it
    result = list(daterange(date(2017, 5, 1), date(2017, 6, 15),
                            step=7, inclusive=True))
    assert result == [date(2017, 5, 1), date(2017, 5, 8),
                      date(2017, 5, 15), date(2017, 5, 22),
                      date(2017, 5, 29), date(2017, 6, 5),
                      date(2017, 6, 12)]


def test_daterange_timedelta_step():
    start = date(2016, 1, 1)
    result = list(daterange(start, date(2016, 1, 10),
                            step=timedelta(days=3)))
    assert result == [date(2016, 1, 1), date(2016, 1, 4),
                      date(2016, 1, 7)]


def test_daterange_negative_step():
    result = list(daterange(date(2016, 1, 5), date(2016, 1, 1), step=-1))
    assert result == [date(2016, 1, 5), date(2016, 1, 4),
                      date(2016, 1, 3), date(2016, 1, 2)]


def test_daterange_step_away_yields_nothing():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 5), step=-1))
    assert result == []


def test_daterange_start_equals_stop():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 1)))
    assert result == []


def test_daterange_start_equals_stop_inclusive():
    result = list(daterange(date(2016, 1, 1), date(2016, 1, 1),
                            inclusive=True))
    assert result == [date(2016, 1, 1)]


def test_daterange_stop_none_infinite():
    gen = daterange(date(2016, 1, 1), None)
    values = [next(gen) for _ in range(3)]
    assert values == [date(2016, 1, 1), date(2016, 1, 2),
                      date(2016, 1, 3)]


def test_daterange_invalid_start_type():
    with pytest.raises(TypeError):
        list(daterange('not a date', date(2016, 1, 1)))


def test_daterange_invalid_stop_type():
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), 'not a date'))


def test_daterange_invalid_step_type():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 2, 1), step='a'))


def test_daterange_zero_step():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 2, 1), step=0))


def test_daterange_month_step():
    result = list(daterange(date(2016, 1, 31), date(2016, 5, 1),
                            step=(0, 1, 0)))
    # Note behavior: month replacement at day 31 wraps/rejects; assert actual
    assert result[0] == date(2016, 1, 31)
    assert all(isinstance(x, date) for x in result)


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
    tz = ConstantTZInfo('Plus2', timedelta(hours=2))
    assert tz.name == 'Plus2'
    assert tz.offset == timedelta(hours=2)
    assert tz.utcoffset(None) == timedelta(hours=2)
    assert tz.tzname(None) == 'Plus2'
    assert tz.dst(None) == ZERO


def test_constant_tzinfo_utcoffset_hours():
    tz = ConstantTZInfo('Plus2', timedelta(hours=2))
    assert tz.utcoffset_hours == 2.0
    tz2 = ConstantTZInfo('Minus', timedelta(hours=-3))
    assert tz2.utcoffset_hours == -3.0


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo('Foo', timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='Foo', offset=datetime.timedelta(days=1, seconds=3600))"


def test_utc_singleton():
    assert UTC.name == 'UTC'
    assert UTC.utcoffset(None) == ZERO


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_localtz_repr():
    tz = LocalTZInfo()
    assert repr(tz) == 'LocalTZInfo()'


def test_localtz_utcoffset():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    offset = tz.utcoffset(dt)
    assert isinstance(offset, timedelta)


def test_localtz_dst_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    assert isinstance(tz.dst(dt), timedelta)


def test_localtz_tzname():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    assert isinstance(tz.tzname(dt), str)


def test_localtz_is_dst_returns_bool():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    assert tz.is_dst(dt) in (True, False)


def test_localtz_singleton_type():
    assert isinstance(LocalTZ, LocalTZInfo)


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


def test_ustimezone_names_std():
    # Winter, standard time
    dt = datetime(2016, 1, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EST'
    assert Eastern.dst(dt) == ZERO
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)


def test_ustimezone_names_dst():
    # Summer, daylight time
    dt = datetime(2016, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.tzname(dt) == 'EDT'
    assert Eastern.dst(dt) == HOUR
    assert Eastern.utcoffset(dt) == timedelta(hours=-4)


def test_ustimezone_dst_none():
    assert Eastern.dst(None) == ZERO


def test_ustimezone_dst_naive():
    assert Eastern.dst(datetime(2016, 7, 15, 12, 0, 0)) == ZERO


def test_ustimezone_dst_before_1967():
    dt = datetime(1960, 7, 15, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == ZERO


def test_ustimezone_dst_range_2007():
    start, end = Eastern._dst_range(2016)
    # Second Sunday of March 2016 == March 13
    assert start == datetime(2016, 3, 13, 2, 0)
    # First Sunday of November 2016 == November 6
    assert end == datetime(2016, 11, 6, 2, 0)


def test_ustimezone_dst_range_1987_2006():
    start, end = Eastern._dst_range(2000)
    # First Sunday of April 2000 == April 2
    assert start == datetime(2000, 4, 2, 2, 0)
    # Last Sunday of October 2000 == October 29
    assert end == datetime(2000, 10, 29, 2, 0)


def test_ustimezone_dst_range_1967_1986():
    start, end = Eastern._dst_range(1975)
    assert start.month == 4
    assert end.month == 10


def test_ustimezone_dst_range_old():
    start, end = Eastern._dst_range(1950)
    assert start is None
    assert end is None


def test_ustimezone_dst_boundaries():
    # Just before DST start (2016-03-13 02:00)
    before = datetime(2016, 3, 13, 1, 59, 59, tzinfo=Eastern)
    assert Eastern.dst(before) == ZERO
    # Within ambiguous hour at start
    within = datetime(2016, 3, 13, 2, 30, 0, tzinfo=Eastern)
    assert Eastern.dst(within) == ZERO
    # After the transition
    after = datetime(2016, 3, 13, 3, 30, 0, tzinfo=Eastern)
    assert Eastern.dst(after) == HOUR


def test_ustimezone_fromutc_wrong_tz():
    with pytest.raises(ValueError):
        Eastern.fromutc(datetime(2016, 7, 1, tzinfo=UTC))


def test_ustimezone_fromutc_summer():
    dt = datetime(2016, 7, 15, 16, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 12
    assert result.tzinfo is Eastern


def test_ustimezone_fromutc_winter():
    dt = datetime(2016, 1, 15, 17, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 12
    assert result.tzinfo is Eastern


def test_ustimezone_fromutc_pre_1967():
    dt = datetime(1950, 7, 15, 17, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.hour == 12
    assert result.tzinfo is Eastern


def test_ustimezone_fromutc_end_transition_fold():
    # 2016-11-06 06:00 UTC = DST end transition start (2am EDT -> 1am EST)
    dt = datetime(2016, 11, 6, 6, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.tzinfo is Eastern
    assert result.fold == 1


def test_ustimezone_dst_end_ambiguous():
    # 2016-11-06 01:30 is ambiguous; fold matters
    dt = datetime(2016, 11, 6, 1, 30, 0, tzinfo=Eastern, fold=0)
    assert Eastern.dst(dt) == HOUR
    dt_fold = datetime(2016, 11, 6, 1, 30, 0, tzinfo=Eastern, fold=1)
    assert Eastern.dst(dt_fold) == ZERO


def test_ustimezone_dst_start_ambiguous():
    # 2016-03-13 02:30 is ambiguous; fold matters
    dt = datetime(2016, 3, 13, 2, 30, 0, tzinfo=Eastern, fold=0)
    assert Eastern.dst(dt) == ZERO
    dt_fold = datetime(2016, 3, 13, 2, 30, 0, tzinfo=Eastern, fold=1)
    assert Eastern.dst(dt_fold) == HOUR
