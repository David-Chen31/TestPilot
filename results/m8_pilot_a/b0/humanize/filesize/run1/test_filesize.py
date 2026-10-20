"""Tests for humanize.filesize.naturalsize."""

from __future__ import annotations

import pytest

from humanize.filesize import naturalsize


# ---------------------------------------------------------------------------
# Basic / decimal (default) behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0 Bytes"),
        (1, "1 Byte"),
        (-1, "-1 Byte"),
        (300, "300 Bytes"),
        (999, "999 Bytes"),
        (-999, "-999 Bytes"),
        (1000, "1.0 kB"),
        (3000000, "3.0 MB"),
        (10**9, "1.0 GB"),
        (10**12, "1.0 TB"),
        (10**15, "1.0 PB"),
        (10**18, "1.0 EB"),
        (10**21, "1.0 ZB"),
        (10**24, "1.0 YB"),
        (10**27, "1.0 RB"),
        (10**30, "1.0 QB"),
    ],
)
def test_naturalsize_decimal(value, expected):
    assert naturalsize(value) == expected


def test_naturalsize_docstring_examples():
    assert naturalsize(3000000) == "3.0 MB"
    assert naturalsize(10**28) == "10.0 RB"
    assert naturalsize(10**34 * 3) == "30000.0 QB"


def test_naturalsize_negative_decimal():
    assert naturalsize(-1000) == "-1.0 kB"
    assert naturalsize(-3000000) == "-3.0 MB"


def test_naturalsize_float_value():
    assert naturalsize(1500.0) == "1.5 kB"
    assert naturalsize(999.9) == "999 Bytes"


def test_naturalsize_string_value():
    assert naturalsize("3000000") == "3.0 MB"
    assert naturalsize("1000") == "1.0 kB"
    assert naturalsize("0") == "0 Bytes"


# ---------------------------------------------------------------------------
# Binary behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0 Bytes"),
        (1, "1 Byte"),
        (-1, "-1 Byte"),
        (1023, "1023 Bytes"),
        (1024, "1.0 KiB"),
        (1024 * 1024, "1.0 MiB"),
        (1024**3, "1.0 GiB"),
        (1024**4, "1.0 TiB"),
        (1024**5, "1.0 PiB"),
        (1024**6, "1.0 EiB"),
        (1024**7, "1.0 ZiB"),
        (1024**8, "1.0 YiB"),
        (1024**9, "1.0 RiB"),
        (1024**10, "1.0 QiB"),
    ],
)
def test_naturalsize_binary(value, expected):
    assert naturalsize(value, binary=True) == expected


def test_naturalsize_binary_negative():
    assert naturalsize(-4096, True) == "-4.0 KiB"


def test_naturalsize_binary_example():
    assert naturalsize(3000, True) == "2.9 KiB"


# ---------------------------------------------------------------------------
# GNU behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0B"),
        (1, "1B"),
        (-1, "-1B"),
        (300, "300B"),
        (1023, "1023B"),
        (1024, "1.0K"),
        (1024 * 1024, "1.0M"),
        (1024**3, "1.0G"),
        (1024**4, "1.0T"),
        (1024**5, "1.0P"),
        (1024**6, "1.0E"),
        (1024**7, "1.0Z"),
        (1024**8, "1.0Y"),
        (1024**9, "1.0R"),
        (1024**10, "1.0Q"),
    ],
)
def test_naturalsize_gnu(value, expected):
    assert naturalsize(value, False, True) == expected


def test_naturalsize_gnu_docstring_examples():
    assert naturalsize(300, False, True) == "300B"
    assert naturalsize(3000, False, True) == "2.9K"
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_naturalsize_gnu_ignores_binary():
    assert naturalsize(1024, binary=True, gnu=True) == "1.0K"
    assert naturalsize(1024, True, True) == "1.0K"


def test_naturalsize_gnu_no_space():
    # GNU mode never inserts a space between number and suffix.
    assert " " not in naturalsize(1024 * 1024, False, True)
    # Non-GNU modes do.
    assert " " in naturalsize(1024 * 1024)


# ---------------------------------------------------------------------------
# Custom format strings
# ---------------------------------------------------------------------------

def test_naturalsize_custom_format():
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"
    assert naturalsize(3000000, format="%.0f") == "3 MB"
    assert naturalsize(3000000, format="%.2f") == "3.00 MB"


def test_naturalsize_custom_format_rounding_promotes_suffix():
    # 999999 is 999.999 kB -> rounds to "1000.0 kB", which should step up.
    assert naturalsize(999999) == "1.0 MB"
    assert naturalsize(999999, format="%.0f") == "1 MB"


def test_naturalsize_custom_format_no_promotion_when_below_base():
    # 999000 formats to "999.0 kB" - stays.
    assert naturalsize(999000) == "999.0 kB"


# ---------------------------------------------------------------------------
# Rounding promotion at the top of the suffix ladder
# ---------------------------------------------------------------------------

def test_naturalsize_suffix_promotion_at_max_suffix():
    # At the largest suffix (QB), there is no "next" suffix; exp stays.
    # 10**34 * 3 -> "30000.0 QB"
    result = naturalsize(10**34 * 3)
    assert result == "30000.0 QB"


def test_naturalsize_exp_clamped_to_suffix_length():
    # Very large values: exp is capped at len(suffix).
    huge = 10**100
    result = naturalsize(huge)
    assert result.endswith(" QB")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_naturalsize_invalid_string_raises_value_error():
    with pytest.raises(ValueError):
        naturalsize("not a number")


def test_naturalsize_none_raises_type_error():
    with pytest.raises(TypeError):
        naturalsize(None)  # type: ignore[arg-type]


def test_naturalsize_non_numeric_object_raises():
    with pytest.raises((TypeError, ValueError)):
        naturalsize([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Byte singular / plural localization hooks
# ---------------------------------------------------------------------------

def test_naturalsize_one_byte_not_gnu():
    # Single byte (non-GNU) uses singular form.
    assert naturalsize(1) == "1 Byte"
    assert naturalsize(-1) == "-1 Byte"
    assert naturalsize(1.0) == "1 Byte"


def test_naturalsize_multiple_bytes():
    assert naturalsize(2) == "2 Bytes"
    assert naturalsize(-2) == "-2 Bytes"
