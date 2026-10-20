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
        (5, "5th"),
        (9, "9th"),
        (10, "10th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (24, "24th"),
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
        (-3, "-3rd"),
        (-4, "-4th"),
        (-11, "-11th"),
    ],
)
def test_ordinal(value, expected):
    assert ordinal(value) == expected


def test_ordinal_string_number():
    assert ordinal("7") == "7th"
    assert ordinal("21") == "21st"


def test_ordinal_float():
    assert ordinal(1.9) == "1st"
    assert ordinal(-2.9) == "-2nd"


def test_ordinal_invalid_string():
    assert ordinal("something else") == "something else"
    assert ordinal("foo") == "foo"


def test_ordinal_list():
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"


def test_ordinal_none():
    assert ordinal(None) == "None"


def test_ordinal_inf_nan():
    assert ordinal(float("inf")) == "+Inf"
    assert ordinal(float("-inf")) == "-Inf"
    assert ordinal(float("nan")) == "NaN"


def test_ordinal_female_gender():
    assert ordinal(1, gender="female") == "1st"
    assert ordinal(2, gender="female") == "2nd"
    assert ordinal(3, gender="female") == "3rd"
    assert ordinal(4, gender="female") == "4th"


def test_ordinal_invalid_gender_defaults_to_female():
    assert ordinal(1, gender="other") == "1st"
    assert ordinal(2, gender="") == "2nd"


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100, "100"),
        (1000, "1,000"),
        (1_000_000, "1,000,000"),
        (-1000, "-1,000"),
        (0, "0"),
    ],
)
def test_intcomma_int(value, expected):
    assert intcomma(value) == expected


def test_intcomma_string():
    assert intcomma("1000") == "1,000"
    assert intcomma("1_000_000") == "1_000,000".replace("_", "")  # "1000000"
    # avoid confusion: check directly
    assert intcomma("1000000") == "1,000,000"


def test_intcomma_float():
    assert intcomma(1_234_567.25) == "1,234,567.25"
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"


def test_intcomma_string_float_ndigits():
    assert intcomma("14308.40", 1) == "14,308.4"


def test_intcomma_none():
    assert intcomma(None) == "None"


def test_intcomma_invalid_string():
    assert intcomma("not a number") == "not a number"


def test_intcomma_ndigits():
    assert intcomma(1234, 2) == "1,234.00"
    assert intcomma(1234, 0) == "1,234"


def test_intcomma_inf_nan():
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"
    assert intcomma(float("nan")) == "NaN"
    assert intcomma("inf") == "+Inf"
    assert intcomma("-inf") == "-Inf"
    assert intcomma("nan") == "NaN"


def test_intcomma_negative_float():
    assert intcomma(-1234567.5, 1) == "-1,234,567.5"


def test_intcomma_string_with_commas():
    assert intcomma("1,234") == "1,234"


def test_intcomma_list():
    assert intcomma([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("100", "100"),
        ("12400", "12.4 thousand"),
        ("1000000", "1.0 million"),
        (1_200_000_000, "1.2 billion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (1_000, "1.0 thousand"),
        (999, "999"),
        (0, "0"),
    ],
)
def test_intword(value, expected):
    assert intword(value) == expected


def test_intword_negative():
    assert intword(-1000000) == "-1.0 million"
    assert intword(-12400) == "-12.4 thousand"
    assert intword(-100) == "-100"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"


def test_intword_none():
    assert intword(None) == "None"


def test_intword_invalid():
    assert intword("foo") == "foo"
    assert intword([1, 2]) == "[1, 2]"


def test_intword_inf_nan():
    assert intword(float("inf")) == "+Inf"
    assert intword(float("-inf")) == "-Inf"
    assert intword(float("nan")) == "NaN"


def test_intword_googol():
    assert intword(10**100) == "10.0 googol"


def test_intword_rounding_carry():
    # 10**24 - 1 previously rendered as "1000.0 sextillion"; should carry over.
    assert intword(10**24 - 1) == "1.0 septillion"


def test_intword_rounding_boundary():
    # Just below 1000x a unit, rounds up to next power.
    assert intword(999_999) == "1.0 million"


def test_intword_float_input():
    assert intword(1_500_000.0) == "1.5 million"


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
        (-1, "-1"),
        (100, "100"),
    ],
)
def test_apnumber(value, expected):
    assert apnumber(value) == expected


def test_apnumber_string():
    assert apnumber("7") == "seven"
    assert apnumber("10") == "10"


def test_apnumber_invalid():
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"


def test_apnumber_float():
    assert apnumber(5.9) == "five"
    assert apnumber(10.9) == "10"


def test_apnumber_inf_nan():
    assert apnumber(float("inf")) == "+Inf"
    assert apnumber(float("-inf")) == "-Inf"
    assert apnumber(float("nan")) == "NaN"


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (1 / 3, "1/3"),
        (1, "1"),
        (0.5, "1/2"),
        (0.25, "1/4"),
        (0.75, "3/4"),
        (0.0, "0"),
        (2.5, "2 1/2"),
        (-0.3, "-3/10"),
        (-1.3, "-1 3/10"),
        (-2.5, "-2 1/2"),
    ],
)
def test_fractional(value, expected):
    assert fractional(value) == expected


def test_fractional_string():
    assert fractional("0.3") == "3/10"
    assert fractional("1.5") == "1 1/2"


def test_fractional_invalid():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"


def test_fractional_inf_nan():
    assert fractional(float("inf")) == "+Inf"
    assert fractional(float("-inf")) == "-Inf"
    assert fractional(float("nan")) == "NaN"


def test_fractional_rounds_up_to_whole():
    # 2.9999999 -> whole 2 + frac 1/1 -> "3"
    assert fractional(2.9999999) == "3"


def test_fractional_large_denominator():
    # Fraction reduced with limit_denominator(1000)
    result = fractional(0.123456789)
    assert "/" in result or result.isdigit()


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float(0.3), "3.00 x 10⁻¹"),
        (int(500), "5.00 x 10²"),
        (-1000, "-1.00 x 10³"),
        (1, "1.00 x 10⁰"),
        (0, "0.00 x 10⁰"),
        (0.0, "0.00 x 10⁰"),
    ],
)
def test_scientific(value, expected):
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


def test_scientific_inf_nan():
    assert scientific(float("inf")) == "+Inf"
    assert scientific(float("-inf")) == "-Inf"
    assert scientific(float("nan")) == "NaN"


def test_scientific_superscript_negative_exponent():
    assert scientific(1e-5) == "1.00 x 10⁻⁵"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_default():
    assert clamp(123.456) == "123.456"
    assert clamp(0) == "0"
    assert clamp(-5) == "-5"


def test_clamp_none():
    assert clamp(None) is None


def test_clamp_inf_nan():
    assert clamp(float("inf")) == "+Inf"
    assert clamp(float("-inf")) == "-Inf"
    assert clamp(float("nan")) == "NaN"


def test_clamp_floor():
    assert clamp(0.0001, floor=0.01) == "<0.01"
    assert clamp(0.5, floor=1) == "<1"
    assert clamp(1, floor=1) == "1"


def test_clamp_ceil():
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_custom_tokens():
    assert clamp(0.0001, floor=0.01, floor_token="under ") == "under 0.01"
    assert clamp(10, ceil=5, ceil_token="over ") == "over 5"


def test_clamp_callable_format():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
        "under 1.0 million"
    )
    assert clamp(2_000_000, format=intword) == "2.0 million"


def test_clamp_floor_wins_over_ceil():
    # floor and ceil both triggered - floor branch takes precedence
    assert clamp(-100, floor=0, ceil=5) == "<0"


def test_clamp_ceil_branch():
    assert clamp(100, ceil=5) == ">5"


def test_clamp_invalid_format():
    with pytest.raises(ValueError):
        clamp(1, format=123)


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1500, "V", "1.50 kV"),
        (2e8, "W", "200 MW"),
        (220e-6, "F", "220 μF"),
        (1e-14, "", "10.00 f"),
        (0, "", "0.000"),
        (1, "", "1.00"),
        (1e3, "", "1.00 k"),
        (1e6, "", "1.00 M"),
        (1e9, "", "1.00 G"),
        (-1500, "V", "-1.50 kV"),
    ],
)
def test_metric(value, unit, expected):
    assert metric(value, unit) == expected


def test_metric_precision():
    assert metric(1e-14, precision=4) == "10.00 f"
    assert metric(1500, "V", precision=2) == "1.5 kV"
    assert metric(1500, "V", precision=4) == "1.500 kV"


def test_metric_scientific_fallback_huge():
    assert metric(1e40) == "1.00 x 10⁴⁰"


def test_metric_scientific_fallback_tiny():
    assert metric(1e-35) == "1.00 x 10⁻³⁵"


def test_metric_inf_nan():
    assert metric(float("inf")) == "+Inf"
    assert metric(float("-inf")) == "-Inf"
    assert metric(float("nan")) == "NaN"


def test_metric_degree_no_space():
    assert metric(1, "°") == "1.00°"
    assert metric(1e3, "°") == "1.00k°"


def test_metric_prime_no_space():
    assert metric(1, "′") == "1.00′"
    assert metric(1, "″") == "1.00″"


def test_metric_rounding_carry():
    # 9.999 -> should carry up to 10.0 in the next bucket
    result = metric(999.9)
    assert result == "1.00 k"


def test_metric_no_unit_respects_space():
    # With no unit but a prefix, space should be present.
    assert metric(1e3) == "1.00 k"
