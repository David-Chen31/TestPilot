"""Tests for humanize.number."""

from __future__ import annotations

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
    ("value", "expected"),
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
        (100, "100th"),
        (101, "101st"),
        (103, "103rd"),
        (111, "111th"),
        (1002, "1002nd"),
        (0, "0th"),
        (-1, "-1th"),
        (-3, "-3th"),
        (-11, "-11th"),
        (-22, "-22th"),
    ],
)
def test_ordinal(value, expected):
    assert ordinal(value) == expected


def test_ordinal_strings_and_floats():
    assert ordinal("1") == "1st"
    assert ordinal("3") == "3rd"
    assert ordinal(3.0) == "3rd"
    assert ordinal("something else") == "something else"
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"


def test_ordinal_gender():
    # Default (English) both genders yield the same suffixes here
    assert ordinal(1, gender="male") == "1st"
    assert ordinal(1, gender="female") == "1st"
    # Anything not "male" is treated as "female"
    assert ordinal(2, gender="unknown") == "2nd"


def test_ordinal_not_finite():
    assert ordinal(float("nan")) == "NaN"
    assert ordinal(float("inf")) == "+Inf"
    assert ordinal(float("-inf")) == "-Inf"


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
def test_intcomma_defaults():
    assert intcomma(100) == "100"
    assert intcomma("1000") == "1,000"
    assert intcomma(1_000_000) == "1,000,000"
    assert intcomma(1_234_567.25) == "1,234,567.25"
    assert intcomma(0) == "0"
    assert intcomma(-1234) == "-1,234"


def test_intcomma_ndigits():
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"
    assert intcomma("14308.40", 1) == "14,308.4"
    assert intcomma(1_000_000, 2) == "1,000,000.00"


def test_intcomma_string_with_commas():
    assert intcomma("1,000") == "1,000"
    assert intcomma("1,234,567") == "1,234,567"


def test_intcomma_unparseable():
    assert intcomma(None) == "None"
    assert intcomma("foo") == "foo"
    # Existing thousands separators are stripped before re-parsing;
    # the resulting non-numeric string is returned as-is (commas removed).
    assert intcomma("abc,def") == "abcdef"


def test_intcomma_not_finite():
    assert intcomma(float("nan")) == "NaN"
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"
    assert intcomma("inf") == "+Inf"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
def test_intword_basic():
    assert intword("100") == "100"
    assert intword("12400") == "12.4 thousand"
    assert intword("1000000") == "1.0 million"
    assert intword(1_200_000_000) == "1.2 billion"
    assert intword(8100000000000000000000000000000000) == "8.1 decillion"


def test_intword_negative():
    assert intword(-1_000_000) == "-1.0 million"
    assert intword(-100) == "-100"


def test_intword_format():
    assert intword("1234000", "%0.3f") == "1.234 million"
    assert intword(1_500, "%d") == "1 thousand"


def test_intword_unparseable():
    assert intword(None) == "None"
    assert intword("foo") == "foo"


def test_intword_not_finite():
    assert intword(float("nan")) == "NaN"
    assert intword(float("inf")) == "+Inf"
    assert intword(float("-inf")) == "-Inf"


def test_intword_carry_to_next_power():
    # Rounding should carry up to the next unit rather than showing 1000.0
    assert intword(10**24 - 1) == "1.0 septillion"


def test_intword_googol():
    assert intword(10**100) == "1.0 googol"


def test_intword_just_below_thousand():
    assert intword(999) == "999"
    assert intword(1000) == "1.0 thousand"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
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
        (10, "10"),
        (11, "11"),
        (100, "100"),
    ],
)
def test_apnumber(value, expected):
    assert apnumber(value) == expected


def test_apnumber_strings():
    assert apnumber("7") == "seven"
    assert apnumber("0") == "zero"


def test_apnumber_unparseable():
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"


def test_apnumber_not_finite():
    assert apnumber(float("nan")) == "NaN"
    assert apnumber(float("inf")) == "+Inf"
    assert apnumber(float("-inf")) == "-Inf"


def test_apnumber_negative():
    assert apnumber(-1) == "-1"


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
def test_fractional_basic():
    assert fractional(0.3) == "3/10"
    assert fractional(1.3) == "1 3/10"
    assert fractional(float(1 / 3)) == "1/3"
    assert fractional(1) == "1"
    assert fractional(0) == "0"


def test_fractional_unparseable():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"


def test_fractional_not_finite():
    assert fractional(float("nan")) == "NaN"
    assert fractional(float("inf")) == "+Inf"
    assert fractional(float("-inf")) == "-Inf"


def test_fractional_negative():
    assert fractional(-1.3) == "-1 3/10"
    assert fractional(-0.3) == "-3/10"


def test_fractional_rounds_to_whole():
    assert fractional(2.9999999) == "3"
    assert fractional(1.0000001) == "1"


def test_fractional_denominator_limit():
    # limit_denominator(1000) approximates 0.3333 as 1/3
    assert fractional(0.3333) == "1/3"


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
def test_scientific_basic():
    assert scientific(float(0.3)) == "3.00 x 10⁻¹"
    assert scientific(int(500)) == "5.00 x 10²"
    assert scientific(-1000) == "-1.00 x 10³"
    assert scientific(1000, 1) == "1.0 x 10³"
    assert scientific(1000, 3) == "1.000 x 10³"


def test_scientific_strings():
    assert scientific("99") == "9.90 x 10¹"


def test_scientific_unparseable():
    assert scientific("foo") == "foo"
    assert scientific(None) == "None"


def test_scientific_not_finite():
    assert scientific(float("nan")) == "NaN"
    assert scientific(float("inf")) == "+Inf"
    assert scientific(float("-inf")) == "-Inf"


def test_scientific_zero():
    assert scientific(0) == "0.00 x 10⁰"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
def test_clamp_no_limits():
    assert clamp(123.456) == "123.456"


def test_clamp_floor():
    assert clamp(0.0001, floor=0.01) == "<0.01"


def test_clamp_ceil_format():
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_callable_format():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"


def test_clamp_none():
    assert clamp(None) is None


def test_clamp_not_finite():
    assert clamp(float("nan")) == "NaN"
    assert clamp(float("inf")) == "+Inf"
    assert clamp(float("-inf")) == "-Inf"


def test_clamp_custom_tokens():
    assert clamp(0.0001, floor=0.01, floor_token="min ") == "min 0.01"
    assert clamp(2, ceil=1, ceil_token="max ") == "max 1"


def test_clamp_invalid_format():
    with pytest.raises(ValueError):
        clamp(1, format=123)


def test_clamp_inside_bounds():
    assert clamp(5, floor=1, ceil=10) == "5"


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def test_metric_basic():
    assert metric(1500, "V") == "1.50 kV"
    assert metric(2e8, "W") == "200 MW"
    assert metric(220e-6, "F") == "220 μF"
    assert metric(1e-14, precision=4) == "10.00 f"


def test_metric_no_unit():
    assert metric(1500) == "1.50 k"
    assert metric(1000) == "1.00 k"


def test_metric_scientific_fallback():
    assert metric(1e40) == "1.00 x 10⁴⁰"
    assert metric(1e-40) == "1.00 x 10⁻⁴⁰"


def test_metric_not_finite():
    assert metric(float("nan")) == "NaN"
    assert metric(float("inf")) == "+Inf"
    assert metric(float("-inf")) == "-Inf"


def test_metric_zero():
    assert metric(0) == "0.00"


def test_metric_degree_units():
    assert metric(1000, "°") == "1.00k°"
    assert metric(1000, "′") == "1.00k′"
    assert metric(1000, "″") == "1.00k″"


def test_metric_small_numbers():
    assert metric(1e-3, "s") == "1.00 ms"
    assert metric(1e-6, "s") == "1.00 μs"
    assert metric(1e-9, "s") == "1.00 ns"


def test_metric_large_numbers():
    assert metric(1e12, "B") == "1.00 TB"
    assert metric(1e15, "B") == "1.00 PB"


def test_metric_rounding_carry():
    # 9.999 k rounds up to 10.0 -> should become 10.0 k (not 10000.0)
    assert metric(9999) == "10.0 k"


def test_metric_precision():
    assert metric(1234, precision=2) == "1.2 k"
    assert metric(1234, precision=4) == "1.234 k"
