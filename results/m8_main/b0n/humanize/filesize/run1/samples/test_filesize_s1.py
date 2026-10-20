"""Tests for humanize.filesize."""

import math

import pytest

from humanize.filesize import naturalsize, suffixes


# ---------------------------------------------------------------------------
# suffixes mapping
# ---------------------------------------------------------------------------

def test_suffixes_decimal():
    assert suffixes["decimal"] == (
        "kB",
        "MB",
        "GB",
        "TB",
        "PB",
        "EB",
        "ZB",
        "YB",
        "RB",
        "QB",
    )


def test_suffixes_binary():
    assert suffixes["binary"] == (
        "KiB",
        "MiB",
        "GiB",
        "TiB",
        "PiB",
        "EiB",
        "ZiB",
        "YiB",
        "RiB",
        "QiB",
    )


def test_suffixes_gnu():
    assert suffixes["gnu"] == "KMGTPEZYRQ"


# ---------------------------------------------------------------------------
# Decimal (default) mode
# ---------------------------------------------------------------------------

def test_decimal_example():
    assert naturalsize(3000000) == "3.0 MB"


def test_decimal_bytes():
    assert naturalsize(0) == "0 Bytes"
    assert naturalsize(1) == "1 Byte"
    assert naturalsize(2) == "2 Bytes"
    assert naturalsize(999) == "999 Bytes"
    assert naturalsize(1000) == "1.0 kB"


def test_decimal_negative_single_byte():
    assert naturalsize(-1) == "-1 Byte"


def test_decimal_negative_bytes():
    assert naturalsize(-2) == "-2 Bytes"
    assert naturalsize(-999) == "-999 Bytes"


def test_decimal_negative_kilobytes():
    assert naturalsize(-4096) == "-4.1 kB"


def test_decimal_base_thousand():
    assert naturalsize(1000) == "1.0 kB"
    assert naturalsize(1500) == "1.5 kB"
    assert naturalsize(10**6) == "1.0 MB"
    assert naturalsize(10**9) == "1.0 GB"
    assert naturalsize(10**12) == "1.0 TB"
    assert naturalsize(10**15) == "1.0 PB"
    assert naturalsize(10**18) == "1.0 EB"
    assert naturalsize(10**21) == "1.0 ZB"
    assert naturalsize(10**24) == "1.0 YB"
    assert naturalsize(10**27) == "1.0 RB"


def test_decimal_high_suffixes():
    assert naturalsize(10**28) == "10.0 RB"
    assert naturalsize(10**30) == "1000.0 QB"


def test_decimal_max_qb_clamped():
    # Beyond QB the suffix list length clamps exponent.
    assert naturalsize(10**34 * 3) == "30000.0 QB"


def test_decimal_rounding_steps_up_suffix():
    # 999999 is 999.999 kB which rounds up to 1000.0 -> step up to MB.
    assert naturalsize(999999) == "1.0 MB"
    assert naturalsize(999999999) == "1.0 GB"


def test_decimal_float_input():
    assert naturalsize(3000000.5) == "3.0 MB"


def test_decimal_string_input():
    assert naturalsize("3000000") == "3.0 MB"


# ---------------------------------------------------------------------------
# Binary mode
# ---------------------------------------------------------------------------

def test_binary_example():
    assert naturalsize(3000, True) == "2.9 KiB"


def test_binary_bytes():
    assert naturalsize(0, True) == "0 Bytes"
    assert naturalsize(1, True) == "1 Byte"
    assert naturalsize(1023, True) == "1023 Bytes"
    assert naturalsize(1024, True) == "1.0 KiB"


def test_binary_negative():
    assert naturalsize(-4096, True) == "-4.0 KiB"


def test_binary_suffixes():
    assert naturalsize(2**10, True) == "1.0 KiB"
    assert naturalsize(2**20, True) == "1.0 MiB"
    assert naturalsize(2**30, True) == "1.0 GiB"
    assert naturalsize(2**40, True) == "1.0 TiB"
    assert naturalsize(2**50, True) == "1.0 PiB"
    assert naturalsize(2**60, True) == "1.0 EiB"
    assert naturalsize(2**70, True) == "1.0 ZiB"
    assert naturalsize(2**80, True) == "1.0 YiB"
    assert naturalsize(2**90, True) == "1.0 RiB"
    assert naturalsize(2**100, True) == "1.0 QiB"


def test_binary_rounding_step_up():
    # 1024**2 - 1 rounded to 1024.0 KiB -> 1.0 MiB
    assert naturalsize(1024**2 - 1, True) == "1.0 MiB"


# ---------------------------------------------------------------------------
# GNU mode
# ---------------------------------------------------------------------------

def test_gnu_example_bytes():
    assert naturalsize(300, False, True) == "300B"


def test_gnu_example_kilo():
    assert naturalsize(3000, False, True) == "2.9K"


def test_gnu_custom_format():
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_gnu_zero_and_small():
    assert naturalsize(0, False, True) == "0B"
    assert naturalsize(1, False, True) == "1B"
    assert naturalsize(-1, False, True) == "-1B"
    assert naturalsize(-300, False, True) == "-300B"


def test_gnu_ignores_binary():
    assert naturalsize(1024, binary=True, gnu=True) == "1.0K"
    assert naturalsize(3000, binary=True, gnu=True) == "2.9K"


def test_gnu_suffixes():
    assert naturalsize(1024, False, True) == "1.0K"
    assert naturalsize(1024**2, False, True) == "1.0M"
    assert naturalsize(1024**3, False, True) == "1.0G"
    assert naturalsize(1024**4, False, True) == "1.0T"
    assert naturalsize(1024**5, False, True) == "1.0P"
    assert naturalsize(1024**6, False, True) == "1.0E"
    assert naturalsize(1024**7, False, True) == "1.0Z"
    assert naturalsize(1024**8, False, True) == "1.0Y"
    assert naturalsize(1024**9, False, True) == "1.0R"
    assert naturalsize(1024**10, False, True) == "1.0Q"


def test_gnu_no_space():
    result = naturalsize(1024, False, True)
    assert " " not in result


def test_gnu_negative():
    assert naturalsize(-3000, False, True) == "-2.9K"


def test_gnu_rounding_step_up():
    # 1024**2 - 1 formatted rounds to 1024.0K -> 1.0M
    assert naturalsize(1024**2 - 1, False, True) == "1.0M"


# ---------------------------------------------------------------------------
# Custom format
# ---------------------------------------------------------------------------

def test_custom_format_decimal():
    assert naturalsize(1500, format="%.2f") == "1.50 kB"


def test_custom_format_binary():
    assert naturalsize(1536, True, format="%.3f") == "1.500 KiB"


def test_custom_format_integer():
    assert naturalsize(1500, format="%d") == "1 kB"


# ---------------------------------------------------------------------------
# Rounding step-up boundary conditions
# ---------------------------------------------------------------------------

def test_rounding_no_step_at_exact_top():
    # When already at highest suffix, no step-up possible, stays clamped.
    assert naturalsize(10**33) == "1000.0 QB"


def test_value_just_below_byte_boundary():
    assert naturalsize(999.9) == "999 Bytes"


def test_value_just_below_byte_boundary_binary():
    assert naturalsize(1023.9, True) == "1023 Bytes"


def test_value_just_below_byte_boundary_gnu():
    assert naturalsize(1023.9, False, True) == "1023B"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_string_raises():
    with pytest.raises(ValueError):
        naturalsize("not a number")


def test_none_raises():
    with pytest.raises(TypeError):
        naturalsize(None)


# ---------------------------------------------------------------------------
# Consistency checks using math to derive expectations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (1000, "1.0 kB"),
        (1_000_000, "1.0 MB"),
        (1_000_000_000, "1.0 GB"),
        (1_500_000, "1.5 MB"),
        (2_500_000_000, "2.5 GB"),
    ],
)
def test_decimal_parametrized(value, expected):
    assert naturalsize(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1024, "1.0 KiB"),
        (1024 * 1024, "1.0 MiB"),
        (1024 * 1024 * 1024, "1.0 GiB"),
        (1536, "1.5 KiB"),
    ],
)
def test_binary_parametrized(value, expected):
    assert naturalsize(value, True) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0B"),
        (512, "512B"),
        (1024, "1.0K"),
        (2048, "2.0K"),
        (1024 * 1024, "1.0M"),
    ],
)
def test_gnu_parametrized(value, expected):
    assert naturalsize(value, False, True) == expected
