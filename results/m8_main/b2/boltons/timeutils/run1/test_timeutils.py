import time
from datetime import date, datetime, timedelta, timezone

import pytest

from boltons.timeutils import (
    ConstantTZInfo,
    LocalTZInfo,
    USTimeZone,
    UTC,
    Central,
    Eastern,
    Mountain,
    Pacific,
    decimal_relative_time,
    daterange,
    dt_to_timestamp,
    isoparse,
    parse_td,
    parse_timedelta,
    relative_time,
    strpdate,
)


# ---------------------------------------------------------------------------
# dt_to_timestamp
# ---------------------------------------------------------------------------

def test_dt_to_timestamp_epoch_aware():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0


def test_dt_to_timestamp_known_value_aware():
    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 946684800.0


def test_dt_to_timestamp_naive_treated_as_utc():
    naive = datetime(2000, 1, 1)
    aware = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(naive) == dt_to_timestamp(aware)


def test_dt_to_timestamp_fractional():
    dt = datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 1.5


def test_dt_to_timestamp_with_constant_tz():
    tz = ConstantTZInfo("X", timedelta(hours=2))
    dt = datetime(1970, 1, 1, 2, tzinfo=tz)
    assert dt_to_timestamp(dt) == 0.0


# ---------------------------------------------------------------------------
# isoparse
# ---------------------------------------------------------------------------

def test_isoparse_basic():
    assert isoparse("1970-01-01T00:00:00") == datetime(1970, 1, 1, 0, 0)


def test_isoparse_with_fraction():
    assert isoparse("1970-01-01T00:00:00.851") == datetime(
        1970, 1, 1, 0, 0, 0, 851000
    )


def test_isoparse_truncates_beyond_microseconds():
    # nanosecond precision should be truncated to microseconds
    assert isoparse("1970-01-01T00:00:00.123456789") == datetime(
        1970, 1, 1, 0, 0, 0, 123456
    )


def test_isoparse_roundtrip():
    dt = datetime(2016, 2, 14, 12, 34, 56, 789012)
    assert isoparse(dt.isoformat()) == dt


def test_isoparse_no_time():
    assert isoparse("2015-05-06") == datetime(2015, 5, 6)


# ---------------------------------------------------------------------------
# parse_timedelta
# ---------------------------------------------------------------------------

def test_parse_timedelta_basic():
    assert parse_timedelta("1d 2h 3.5m 0s") == timedelta(days=1, seconds=7410)


def test_parse_timedelta_words():
    assert parse_timedelta("2 weeks 1 day") == timedelta(days=15)


def test_parse_timedelta_negative():
    assert parse_timedelta("-1.5 weeks 3m 20s") == timedelta(
        days=-11, seconds=43400
    )


def test_parse_td_alias_is_same():
    assert parse_td is parse_timedelta


def test_parse_timedelta_full_word_units():
    assert parse_timedelta("3 seconds 2 minutes") == timedelta(
        minutes=2, seconds=3
    )


def test_parse_timedelta_empty_string():
    assert parse_timedelta("") == timedelta(0)


def test_parse_timedelta_invalid_unit():
    with pytest.raises(ValueError):
        parse_timedelta("5 fortnights")


def test_parse_timedelta_invalid_value_regex_no_match():
    # Non-matching input yields empty timedelta rather than error
    assert parse_timedelta("not a duration") == timedelta(0)


# ---------------------------------------------------------------------------
# decimal_relative_time
# ---------------------------------------------------------------------------

def test_decimal_relative_time_days():
    now = datetime(2016, 1, 2, 1, 0, 0)
    d = now - timedelta(days=1, seconds=3600)
    assert decimal_relative_time(d, now) == (1.0, "day")


def test_decimal_relative_time_small_seconds():
    now = datetime(2016, 1, 2)
    d = now - timedelta(seconds=0.002)
    assert decimal_relative_time(d, now, ndigits=5) == (0.002, "seconds")


def test_decimal_relative_time_future_years():
    now = datetime(2016, 1, 2)
    d = now + timedelta(days=900)
    assert decimal_relative_time(d, now, ndigits=1) == (-2.5, "years")


def test_decimal_relative_time_cardinalize_false():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=2)
    val, unit = decimal_relative_time(d, now, cardinalize=False)
    assert unit == "day"
    assert val == 2.0


def test_decimal_relative_time_singular():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1)
    assert decimal_relative_time(d, now) == (1.0, "day")


def test_decimal_relative_time_plural():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=3)
    assert decimal_relative_time(d, now) == (3.0, "days")


def test_decimal_relative_time_zero_seconds():
    now = datetime(2016, 1, 2)
    val, unit = decimal_relative_time(now, now)
    assert val == 0
    assert unit == "seconds"


# ---------------------------------------------------------------------------
# relative_time
# ---------------------------------------------------------------------------

def test_relative_time_ago():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1, seconds=36000)
    assert relative_time(d, now, ndigits=1) == "1.4 days ago"


def test_relative_time_from_now():
    now = datetime(2016, 1, 2)
    d = now + timedelta(days=7)
    assert relative_time(d, now, ndigits=1) == "1 week from now"


def test_relative_time_zero():
    now = datetime(2016, 1, 2)
    assert relative_time(now, now, ndigits=1) == "0 seconds ago"


def test_relative_time_singular_unit():
    now = datetime(2016, 1, 2)
    d = now - timedelta(days=1)
    assert relative_time(d, now) == "1 day ago"


# ---------------------------------------------------------------------------
# strpdate
# ---------------------------------------------------------------------------

def test_strpdate_basic():
    assert strpdate("2016-02-14", "%Y-%m-%d") == date(2016, 2, 14)


def test_strpdate_mixed_literal():
    assert strpdate("26/12 (2015)", "%d/%m (%Y)") == date(2015, 12, 26)


def test_strpdate_with_time_ignored():
    assert strpdate("20151231 23:59:59", "%Y%m%d %H:%M:%S") == date(2015, 12, 31)


def test_strpdate_with_microseconds_ignored():
    assert strpdate(
        "20160101 00:00:00.001", "%Y%m%d %H:%M:%S.%f"
    ) == date(2016, 1, 1)


def test_strpdate_invalid_raises():
    with pytest.raises(ValueError):
        strpdate("not-a-date", "%Y-%m-%d")


# ---------------------------------------------------------------------------
# daterange
# ---------------------------------------------------------------------------

def test_daterange_basic():
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


def test_daterange_stop_equal_start():
    start = date(2015, 12, 25)
    assert list(daterange(start, start)) == []


def test_daterange_inclusive():
    start = date(2016, 1, 1)
    stop = date(2016, 1, 3)
    assert list(daterange(start, stop, inclusive=True)) == [
        date(2016, 1, 1),
        date(2016, 1, 2),
        date(2016, 1, 3),
    ]


def test_daterange_inclusive_not_even():
    start = date(2016, 1, 1)
    stop = date(2016, 1, 4)
    assert list(daterange(start, stop, step=2, inclusive=True)) == [
        date(2016, 1, 1),
        date(2016, 1, 3),
    ]


def test_daterange_timedelta_step():
    start = date(2016, 1, 1)
    stop = date(2016, 1, 11)
    assert list(daterange(start, stop, step=timedelta(days=5))) == [
        date(2016, 1, 1),
        date(2016, 1, 6),
    ]


def test_daterange_month_step():
    start = date(2017, 5, 1)
    stop = date(2017, 8, 1)
    assert list(daterange(start, stop, step=(0, 1, 0), inclusive=True)) == [
        date(2017, 5, 1),
        date(2017, 6, 1),
        date(2017, 7, 1),
        date(2017, 8, 1),
    ]


def test_daterange_negative_step():
    # The default step is +1 day, which points away from the stop (which is
    # in the past). This yields nothing, like range(1, 5, -1).
    start = date(2016, 1, 5)
    stop = date(2016, 1, 1)
    assert list(daterange(start, stop)) == []


def test_daterange_negative_step_inclusive():
    # Same as above but inclusive, still empty since the step points away.
    start = date(2016, 1, 5)
    stop = date(2016, 1, 1)
    assert list(daterange(start, stop, inclusive=True)) == []


def test_daterange_negative_step_explicit():
    # An explicit negative step from a later start to an earlier stop works.
    start = date(2016, 1, 5)
    stop = date(2016, 1, 1)
    assert list(daterange(start, stop, step=-1)) == [
        date(2016, 1, 5),
        date(2016, 1, 4),
        date(2016, 1, 3),
        date(2016, 1, 2),
    ]


def test_daterange_negative_step_explicit_inclusive():
    start = date(2016, 1, 5)
    stop = date(2016, 1, 1)
    assert list(daterange(start, stop, step=-1, inclusive=True)) == [
        date(2016, 1, 5),
        date(2016, 1, 4),
        date(2016, 1, 3),
        date(2016, 1, 2),
        date(2016, 1, 1),
    ]


def test_daterange_step_wrong_direction_yields_nothing():
    start = date(2016, 1, 1)
    stop = date(2016, 1, 5)
    assert list(daterange(start, stop, step=-1)) == []


def test_daterange_zero_step_raises():
    start = date(2016, 1, 1)
    stop = date(2016, 1, 5)
    with pytest.raises(ValueError):
        list(daterange(start, stop, step=0))


def test_daterange_start_not_date_raises():
    with pytest.raises(TypeError):
        list(daterange("2016-01-01", date(2016, 1, 5)))


def test_daterange_stop_not_date_raises():
    with pytest.raises(TypeError):
        list(daterange(date(2016, 1, 1), "2016-01-05"))


def test_daterange_bad_step_type_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), date(2016, 1, 5), step="nope"))


def test_daterange_stop_none_infinite():
    start = date(2016, 1, 1)
    gen = daterange(start, None)
    assert next(gen) == date(2016, 1, 1)
    assert next(gen) == date(2016, 1, 2)
    assert next(gen) == date(2016, 1, 3)


def test_daterange_stop_none_zero_step_raises():
    with pytest.raises(ValueError):
        list(daterange(date(2016, 1, 1), None, step=0))


# ---------------------------------------------------------------------------
# ConstantTZInfo
# ---------------------------------------------------------------------------

def test_constant_tzinfo_defaults():
    tz = ConstantTZInfo()
    assert tz.name == "ConstantTZ"
    assert tz.offset == timedelta(0)
    assert tz.utcoffset(None) == timedelta(0)
    assert tz.tzname(None) == "ConstantTZ"
    assert tz.dst(None) == timedelta(0)


def test_constant_tzinfo_custom():
    tz = ConstantTZInfo("Custom", timedelta(hours=3))
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert tz.tzname(None) == "Custom"
    assert tz.utcoffset_hours == 3.0


def test_constant_tzinfo_repr():
    tz = ConstantTZInfo("Name", timedelta(hours=1))
    assert repr(tz) == "ConstantTZInfo(name='Name', offset=datetime.timedelta(seconds=3600))"


def test_utc_is_constant_tzinfo():
    assert isinstance(UTC, ConstantTZInfo)
    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == "UTC"


# ---------------------------------------------------------------------------
# LocalTZInfo
# ---------------------------------------------------------------------------

def test_local_tzinfo_repr():
    assert repr(LocalTZInfo()) == "LocalTZInfo()"


def test_local_tzinfo_utcoffset_returns_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    off = tz.utcoffset(dt)
    assert isinstance(off, timedelta)


def test_local_tzinfo_dst_is_zero_or_timedelta():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    dst = tz.dst(dt)
    assert isinstance(dst, timedelta)


def test_local_tzinfo_tzname():
    tz = LocalTZInfo()
    dt = datetime(2016, 6, 15, 12, 0, 0)
    name = tz.tzname(dt)
    assert name in time.tzname


# ---------------------------------------------------------------------------
# USTimeZone
# ---------------------------------------------------------------------------

def test_ustimezone_names():
    assert repr(Eastern) == "Eastern"
    assert repr(Central) == "Central"
    assert repr(Mountain) == "Mountain"
    assert repr(Pacific) == "Pacific"


def test_eastern_std_offset():
    dt = datetime(2016, 1, 1, 12, 0, 0)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.tzname(dt) == "EST"


def test_eastern_dst_offset():
    # A naive datetime is treated as non-DST by dst(), so utcoffset is the
    # standard offset for naive datetimes.
    dt = datetime(2016, 7, 1, 12, 0, 0)
    assert Eastern.utcoffset(dt) == timedelta(hours=-5)
    assert Eastern.tzname(dt) == "EST"

    # An aware datetime with tzinfo=Eastern gets the DST offset in summer.
    aware = datetime(2016, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert aware.utcoffset() == timedelta(hours=-4)
    assert aware.tzname() == "EDT"


def test_ustimezone_dst_none():
    assert Eastern.dst(None) == timedelta(0)


def test_ustimezone_dst_naive_returns_zero():
    assert Eastern.dst(datetime(2016, 7, 1, 12, 0, 0)) == timedelta(0)


def test_ustimezone_dst_aware_requires_tz_self():
    aware = datetime(2016, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert aware.tzinfo is Eastern


def test_ustimezone_dst_wrong_tz_raises():
    other = ConstantTZInfo("Other")
    dt = datetime(2016, 7, 1, 12, 0, 0, tzinfo=other)
    with pytest.raises(AssertionError):
        Eastern.dst(dt)


def test_ustimezone_dst_pre_1967():
    dt = datetime(1960, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(0)


def test_ustimezone_dst_1967_1986_range():
    # DST in effect mid-summer
    dt = datetime(1980, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt) == timedelta(hours=1)
    # not in effect in winter
    dt_w = datetime(1980, 1, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(dt_w) == timedelta(0)


def test_ustimezone_dst_1987_2006_range():
    summer = datetime(2000, 7, 1, 12, 0, 0, tzinfo=Eastern)
    winter = datetime(2000, 1, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(summer) == timedelta(hours=1)
    assert Eastern.dst(winter) == timedelta(0)


def test_ustimezone_dst_2007_range():
    summer = datetime(2016, 7, 1, 12, 0, 0, tzinfo=Eastern)
    winter = datetime(2016, 1, 1, 12, 0, 0, tzinfo=Eastern)
    assert Eastern.dst(summer) == timedelta(hours=1)
    assert Eastern.dst(winter) == timedelta(0)


def test_central_pacific_offsets():
    dt = datetime(2016, 1, 1, 12, 0, 0)
    assert Central.utcoffset(dt) == timedelta(hours=-6)
    assert Pacific.utcoffset(dt) == timedelta(hours=-8)


def test_ustimezone_fromutc_wrong_tz():
    with pytest.raises(ValueError):
        Eastern.fromutc(datetime(2016, 1, 1, 12, 0, 0, tzinfo=Central))


def test_ustimezone_fromutc_winter():
    dt = datetime(2016, 1, 1, 17, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.replace(tzinfo=None) == datetime(2016, 1, 1, 12, 0, 0)
    assert result.tzinfo is Eastern


def test_ustimezone_fromutc_summer():
    dt = datetime(2016, 7, 1, 16, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.replace(tzinfo=None) == datetime(2016, 7, 1, 12, 0, 0)


def test_ustimezone_fromutc_pre_1967():
    dt = datetime(1960, 7, 1, 17, 0, 0, tzinfo=Eastern)
    result = Eastern.fromutc(dt)
    assert result.replace(tzinfo=None) == datetime(1960, 7, 1, 12, 0, 0)


def test_ustimezone_roundtrip_utcoffset():
    # A known EDT time round-trips through utcoffset
    dt = datetime(2016, 7, 1, 12, 0, 0, tzinfo=Eastern)
    assert dt.utcoffset() == timedelta(hours=-4)
    assert dt.tzname() == "EDT"
