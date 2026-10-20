"""Tests for humanize.filesize.naturalsize."""

import pytest

from humanize.filesize import naturalsize, suffixes


# ---------------------------------------------------------------------------
# Basic decimal behavior
# ---------------------------------------------------------------------------

def test_decimal_byte():
    assert naturalsize(0) == "0 Bytes"


def test_single_byte_is_singular():
    assert naturalsize(1) == "1 Byte"


def test_single_byte_negative_is_singular():
    assert naturalsize(-1) == "-1 Byte"


def test_single_byte_binary_is_singular():
    assert naturalsize(1, binary=True) == "1 Byte"


def test_two_bytes_plural():
    assert naturalsize(2) == "2 Bytes"


def test_under_base_returns_bytes():
    assert naturalsize(999) == "999 Bytes"


def test_at_base_decimal():
    assert naturalsize(1000) == "1.0 kB"


def test_decimal_megabytes():
    assert naturalsize(3000000) == "3.0 MB"


def test_decimal_gigabytes():
    assert naturalsize(10**9) == "1.0 GB"


def test_decimal_truncates_bytes():
    # int() truncation on fractional inputs below base
    assert naturalsize(999.9) == "999 Bytes"


def test_decimal_negative_megabytes():
    assert naturalsize(-3000000) == "-3.0 MB"


# ---------------------------------------------------------------------------
# Binary behavior
# ---------------------------------------------------------------------------

def test_binary_kib():
    assert naturalsize(3000, binary=True) == "2.9 KiB"


def test_binary_under_base():
    assert naturalsize(1023, binary=True) == "1023 Bytes"


def test_binary_at_base():
    assert naturalsize(1024, binary=True) == "1.0 KiB"


def test_binary_negative():
    assert naturalsize(-4096, binary=True) == "-4.0 KiB"


def test_binary_large():
    assert naturalsize(1024**2, binary=True) == "1.0 MiB"


# ---------------------------------------------------------------------------
# GNU behavior
# ---------------------------------------------------------------------------

def test_gnu_bytes_under_base():
    assert naturalsize(300, False, True) == "300B"


def test_gnu_bytes_zero():
    assert naturalsize(0, gnu=True) == "0B"


def test_gnu_single_byte_not_singular():
    # gnu mode skips the "1 Byte" special-case
    assert naturalsize(1, gnu=True) == "1B"


def test_gnu_kilo():
    assert naturalsize(3000, False, True) == "2.9K"


def test_gnu_custom_format():
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_gnu_ignores_binary_flag():
    assert naturalsize(3000, binary=True, gnu=True) == "2.9K"


def test_gnu_no_space():
    assert " " not in naturalsize(1024 * 1024, gnu=True)


def test_gnu_negative():
    assert naturalsize(-3000, gnu=True) == "-2.9K"


# ---------------------------------------------------------------------------
# Suffix rollover
# ---------------------------------------------------------------------------

def test_rollover_decimal():
    # 999999 -> 999.999 kB rounds to 1000.0 -> step up to MB
    assert naturalsize(999999) == "1.0 MB"


def test_rollover_binary():
    # just below 1 MiB in binary
    assert naturalsize(1024 * 1024 - 1, binary=True) == "1.0 MiB"


def test_rollover_gnu():
    assert naturalsize(1024 * 1024 - 1, gnu=True) == "1.0M"


def test_rollover_does_not_exceed_suffix():
    # When already at the largest suffix, no rollover index used
    val = 10**34 * 3
    result = naturalsize(val)
    assert result == "30000.0 QB"


# ---------------------------------------------------------------------------
# Large values / max suffixes
# ---------------------------------------------------------------------------

def test_decimal_ronnabyte():
    assert naturalsize(10**28) == "10.0 RB"


def test_decimal_quettabyte():
    assert naturalsize(10**34 * 3) == "30000.0 QB"


def test_binary_max_suffix():
    # QiB is the last binary suffix
    assert naturalsize(1024**10, binary=True) == "1.0 QiB"


def test_gnu_max_suffix():
    # 'Q' is the last GNU suffix
    assert naturalsize(1024**10, gnu=True) == "1.0Q"


# ---------------------------------------------------------------------------
# Custom format
# ---------------------------------------------------------------------------

def test_custom_format_decimal():
    assert naturalsize(3000000, format="%.3f") == "3.000 MB"


def test_custom_format_integer():
    assert naturalsize(3000000, format="%d") == "3 MB"


# ---------------------------------------------------------------------------
# String input (value accepts str)
# ---------------------------------------------------------------------------

def test_string_input_decimal():
    assert naturalsize("3000") == "3.0 kB"


def test_string_input_binary():
    assert naturalsize("1024", binary=True) == "1.0 KiB"


def test_invalid_string_raises():
    with pytest.raises(ValueError):
        naturalsize("not a number")


# ---------------------------------------------------------------------------
# Suffix table sanity checks
# ---------------------------------------------------------------------------

def test_suffixes_structure():
    assert set(suffixes.keys()) == {"decimal", "binary", "gnu"}


def test_decimal_suffixes_content():
    assert suffixes["decimal"] == (
        "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB",
    )


def test_binary_suffixes_content():
    assert suffixes["binary"] == (
        "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB",
    )


def test_gnu_suffixes_content():
    assert suffixes["gnu"] == "KMGTPEZYRQ"
