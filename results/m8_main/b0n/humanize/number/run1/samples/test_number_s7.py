"""Tests for humanize.number."""

import math

import pytest

from humanize.number import (
    apnumber,
    clamp,
    fractional,
    intcomma,
    intword,
    metric,
    ordinal,
    scientific,
)


# ---------------------------------------------------------------------------
# ordinal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (101, "101st"),
        (102, "102nd"),
        (103, "103rd"),
        (104, "104th"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
        (1002, "1002nd"),
        (0, "0th"),
        (-1, "-1st"),
        (-2, "-2nd"),
    ],
)
def test_ordinal_numbers(value, expected):
    assert ordinal(value) == expected


def test_ordinal_string_numeric():
    assert ordinal("3") == "3rd"


def test_ordinal_float_truncates():
    assert ordinal(2.9) == "2nd"


def test_ordinal_invalid_returns_str():
    assert ordinal("something else") == "something else"
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert ordinal(None) == "None"


def test_ordinal_infinite_and_nan():
    assert ordinal(float("inf")) == "+Inf"
    assert ordinal(float("-inf")) == "-Inf"
    assert ordinal(float("nan")) == "NaN"


def test_ordinal_gender_default_and_female():
    assert ordinal(1) == "1st"
    assert ordinal(1, gender="female") == "1st"
    assert ordinal(2, gender="female") == "2nd"
    # Anything other than "male" is treated as female; same suffixes here.
    assert ordinal(1, gender="other") == "1st"


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (100, "100"),
        ("1000", "1,000"),
        (1_000_000, "1,000,000"),
        (1_234_567, "1,234,567"),
        (0, "0"),
        (-1000, "-1,000"),
        (1234567890, "1,234,567,890"),
    ],
)
def test_intcomma_integers(value, expected):
    assert intcomma(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1_234_567.25, "1,234,567.25"),
        (1000.0, "1,000.0"),
        (14308.40, "14,308.4"),
    ],
)
def test_intcomma_floats(value, expected):
    assert intcomma(value) == expected


def test_intcomma_ndigits():
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"
    assert intcomma("14308.40", 1) == "14,308.4"
    assert intcomma(1000, 2) == "1,000.00"


def test_intcomma_string_int():
    assert intcomma("1000") == "1,000"
    assert intcomma("1234567") == "1,234,567"


def test_intcomma_string_with_commas():
    assert intcomma("1,234,567") == "1,234,567"


def test_intcomma_none():
    assert intcomma(None) == "None"


def test_intcomma_invalid_string():
    assert intcomma("foo") == "foo"


def test_intcomma_infinite():
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"
    assert intcomma(float("nan")) == "NaN"
    assert intcomma("inf") == "+Inf"
    assert intcomma("-inf") == "-Inf"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("100", "100"),
        (100, "100"),
        (0, "0"),
        ("12400", "12.4 thousand"),
        (12400, "12.4 thousand"),
        ("1000000", "1.0 million"),
        (1_000_000, "1.0 million"),
        (1_200_000, "1.2 million"),
        (1_200_000_000, "1.2 billion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (1_000, "1.0 thousand"),
    ],
)
def test_intword_values(value, expected):
    assert intword(value) == expected


def test_intword_negative():
    assert intword(-1_200_000) == "-1.2 million"
    assert intword(-100) == "-100"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"


def test_intword_invalid():
    assert intword(None) == "None"
    assert intword("foo") == "foo"


def test_intword_infinite():
    assert intword(float("inf")) == "+Inf"
    assert intword(float("-inf")) == "-Inf"
    assert intword(float("nan")) == "NaN"


def test_intword_googol():
    result = intword(10**100)
    assert result == "1.0 googol"


def test_intword_rounding_carry():
    # 10**24 - 1 should carry round to 1.0 septillion, not 1000.0 sextillion
    result = intword(10**24 - 1)
    assert result == "1.0 septillion"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "zero"),
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
        (7, "seven"),
        (8, "eight"),
        (9, "nine"),
    ],
)
def test_apnumber_small(value, expected):
    assert apnumber(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (10, "10"),
        (11, "11"),
        (100, "100"),
        (-1, "-1"),
    ],
)
def test_apnumber_large(value, expected):
    assert apnumber(value) == expected


def test_apnumber_string():
    assert apnumber("7") == "seven"
    assert apnumber("10") == "10"


def test_apnumber_invalid():
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"


def test_apnumber_infinite():
    assert apnumber(float("inf")) == "+Inf"
    assert apnumber(float("-inf")) == "-Inf"
    assert apnumber(float("nan")) == "NaN"


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (1 / 3, "1/3"),
        (1, "1"),
        (0, "0"),
        (2, "2"),
        (1.5, "1 1/2"),
        (0.5, "1/2"),
        (0.25, "1/4"),
        (2.25, "2 1/4"),
    ],
)
def test_fractional_values(value, expected):
    assert fractional(value) == expected


def test_fractional_negative():
    assert fractional(-1.3) == "-1 3/10"
    assert fractional(-0.5) == "-1/2"
    assert fractional(-2) == "-2"


def test_fractional_string_input():
    assert fractional("0.5") == "1/2"
    assert fractional("1.5") == "1 1/2"


def test_fractional_invalid():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"


def test_fractional_infinite():
    assert fractional(float("inf")) == "+Inf"
    assert fractional(float("-inf")) == "-Inf"
    assert fractional(float("nan")) == "NaN"


def test_fractional_rounding_up_to_whole():
    # fractional part rounds to 1/1, folds into integer
    assert fractional(2.9999999) == "3"


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.3, "3.00 x 10⁻¹"),
        (500, "5.00 x 10²"),
        (-1000, "-1.00 x 10³"),
        (1, "1.00 x 10⁰"),
        (0, "0.00 x 10⁰"),
    ],
)
def test_scientific_default(value, expected):
    assert scientific(value) == expected


def test_scientific_precision():
    assert scientific(1000, 1) == "1.0 x 10³"
    assert scientific(1000, 3) == "1.000 x 10³"
    assert scientific(1000, 0) == "1 x 10³"


def test_scientific_string():
    assert scientific("99") == "9.90 x 10¹"


def test_scientific_invalid():
    assert scientific("foo") == "foo"
    assert scientific(None) == "None"


def test_scientific_infinite():
    assert scientific(float("inf")) == "+Inf"
    assert scientific(float("-inf")) == "-Inf"
    assert scientific(float("nan")) == "NaN"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
def test_clamp_passthrough():
    assert clamp(123.456) == "123.456"


def test_clamp_floor():
    assert clamp(0.0001, floor=0.01) == "<0.01"


def test_clamp_ceil():
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_exact_ceil():
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"


def test_clamp_exact_floor():
    assert clamp(0.01, floor=0.01) == "0.01"


def test_clamp_callable_format():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
        "under 1.0 million"
    )


def test_clamp_custom_tokens():
    assert clamp(5, floor=10, floor_token="min ") == "min 10"
    assert clamp(5, ceil=1, ceil_token="max ") == "max 1"


def test_clamp_none():
    assert clamp(None) is None


def test_clamp_infinite():
    assert clamp(float("inf")) == "+Inf"
    assert clamp(float("-inf")) == "-Inf"
    assert clamp(float("nan")) == "NaN"


def test_clamp_invalid_format():
    with pytest.raises(ValueError):
        clamp(1, format=123)


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (1500, "V", "1.50 kV"),
        (2e8, "W", "200 MW"),
        (220e-6, "F", "220 μF"),
        (1000, "V", "1.00 kV"),
        (0, "", "0.00 "),
    ],
)
def test_metric_values(value, unit, expected):
    assert metric(value, unit) == expected


def test_metric_unitless():
    assert metric(1500) == "1.50 k"


def test_metric_precision():
    assert metric(1e-14, precision=4) == "10.00 f"


def test_metric_scientific_fallback():
    assert metric(1e40) == "1.00 x 10⁴⁰"


def test_metric_infinite():
    assert metric(float("inf")) == "+Inf"
    assert metric(float("-inf")) == "-Inf"
    assert metric(float("nan")) == "NaN"


def test_metric_degree_unit_no_space():
    assert metric(1500, "°") == "1.50 k°"


def test_metric_rounding_carry():
    # 9.999e3 should carry into the next bucket
    result = metric(9999, "V")
    assert result == "10.0 kV"


def test_metric_negative():
    assert metric(-1500, "V") == "-1.50 kV"
