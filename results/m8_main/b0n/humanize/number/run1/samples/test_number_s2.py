"""Unit tests for humanize.number."""

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


class TestOrdinal:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (5, "5th"),
            (10, "10th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (101, "101st"),
            (103, "103rd"),
            (111, "111th"),
            (1002, "1002nd"),
            (0, "0th"),
        ],
    )
    def test_basic(self, value, expected):
        assert ordinal(value) == expected

    def test_string_input(self):
        assert ordinal("1") == "1st"
        assert ordinal("1002") == "1002nd"
        assert ordinal("103") == "103rd"

    def test_float_input(self):
        assert ordinal(1.9) == "1st"
        assert ordinal(4.0) == "4th"

    def test_non_numeric(self):
        assert ordinal("something else") == "something else"
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"
        assert ordinal(None) == "None"

    def test_non_finite(self):
        assert ordinal(float("nan")) == "NaN"
        assert ordinal(float("inf")) == "+Inf"
        assert ordinal(float("-inf")) == "-Inf"
        assert ordinal(math.nan) == "NaN"

    def test_gender(self):
        assert ordinal(1, gender="male") == "1st"
        assert ordinal(1, gender="female") == "1st"
        assert ordinal(2, gender="female") == "2nd"
        # Invalid gender falls back to female
        assert ordinal(3, gender="unknown") == "3rd"


class TestIntcomma:
    def test_basic(self):
        assert intcomma(100) == "100"
        assert intcomma("1000") == "1,000"
        assert intcomma(1_000_000) == "1,000,000"
        assert intcomma(3000) == "3,000"
        assert intcomma(45000) == "45,000"

    def test_float(self):
        assert intcomma(1_234_567.25) == "1,234,567.25"
        assert intcomma(1234.5454545, 2) == "1,234.55"
        assert intcomma(14308.40, 1) == "14,308.4"

    def test_string_float_ndigits(self):
        assert intcomma("14308.40", 1) == "14,308.4"

    def test_negative(self):
        assert intcomma(-1000) == "-1,000"
        assert intcomma(-1234.5) == "-1,234.5"

    def test_ndigits(self):
        assert intcomma(1234.5678, 0) == "1,235"
        assert intcomma(1234.5678, 3) == "1,234.568"

    def test_none(self):
        assert intcomma(None) == "None"

    def test_non_numeric(self):
        assert intcomma("foo") == "foo"
        assert intcomma("something else") == "something else"

    def test_non_finite(self):
        assert intcomma(float("nan")) == "NaN"
        assert intcomma(float("inf")) == "+Inf"
        assert intcomma(float("-inf")) == "-Inf"
        assert intcomma("inf") == "+Inf"
        assert intcomma("nan") == "NaN"

    def test_string_with_existing_commas(self):
        assert intcomma("1,000") == "1,000"


class TestIntword:
    def test_small(self):
        assert intword("100") == "100"
        assert intword(100) == "100"
        assert intword(0) == "0"
        assert intword(999) == "999"

    def test_thousands(self):
        assert intword("12400") == "12.4 thousand"
        assert intword(1200) == "1.2 thousand"

    def test_millions(self):
        assert intword("1000000") == "1.0 million"
        assert intword(1_200_000_000) == "1.2 billion"

    def test_decillion(self):
        assert intword(8100000000000000000000000000000000) == "8.1 decillion"

    def test_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_negative(self):
        assert intword(-1000) == "-1.0 thousand"
        assert intword(-1_200_000) == "-1.2 million"

    def test_none(self):
        assert intword(None) == "None"

    def test_non_numeric(self):
        assert intword("foo") == "foo"

    def test_non_finite(self):
        assert intword(float("nan")) == "NaN"
        assert intword(float("inf")) == "+Inf"
        assert intword(float("-inf")) == "-Inf"

    def test_float_input(self):
        assert intword(1500.0) == "1.5 thousand"

    def test_rounding_carry(self):
        # 999999 rounds up to the next power
        assert intword(999_999) == "1.0 million"
        assert intword(999_900_000) == "999.9 million"

    def test_googol(self):
        assert intword(10**100) == "10.0 googol"


class TestApnumber:
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
    def test_small(self, value, expected):
        assert apnumber(value) == expected

    def test_boundary(self):
        assert apnumber(10) == "10"
        assert apnumber(-1) == "-1"
        assert apnumber(100) == "100"

    def test_string_input(self):
        assert apnumber("7") == "seven"
        assert apnumber("10") == "10"

    def test_non_numeric(self):
        assert apnumber("foo") == "foo"
        assert apnumber(None) == "None"

    def test_non_finite(self):
        assert apnumber(float("nan")) == "NaN"
        assert apnumber(float("inf")) == "+Inf"
        assert apnumber(float("-inf")) == "-Inf"


class TestFractional:
    def test_basic(self):
        assert fractional(0.3) == "3/10"
        assert fractional(1.3) == "1 3/10"
        assert fractional(float(1 / 3)) == "1/3"

    def test_integer(self):
        assert fractional(1) == "1"
        assert fractional(5) == "5"
        assert fractional(1.0) == "1"

    def test_zero(self):
        assert fractional(0) == "0"

    def test_negative(self):
        assert fractional(-1.3) == "-1 3/10"

    def test_non_numeric(self):
        assert fractional("ten") == "ten"
        assert fractional(None) == "None"

    def test_non_finite(self):
        assert fractional(float("nan")) == "NaN"
        assert fractional(float("inf")) == "+Inf"
        assert fractional(float("-inf")) == "-Inf"

    def test_string_number(self):
        assert fractional("0.5") == "1/2"

    def test_fractions(self):
        assert fractional(0.5) == "1/2"
        assert fractional(0.25) == "1/4"
        assert fractional(2.5) == "2 1/2"


class TestScientific:
    def test_basic(self):
        assert scientific(float(0.3)) == "3.00 x 10⁻¹"
        assert scientific(int(500)) == "5.00 x 10²"

    def test_negative(self):
        assert scientific(-1000) == "-1.00 x 10³"

    def test_precision(self):
        assert scientific(1000, 1) == "1.0 x 10³"
        assert scientific(1000, 3) == "1.000 x 10³"

    def test_string(self):
        assert scientific("99") == "9.90 x 10¹"

    def test_non_numeric(self):
        assert scientific("foo") == "foo"
        assert scientific(None) == "None"

    def test_non_finite(self):
        assert scientific(float("nan")) == "NaN"
        assert scientific(float("inf")) == "+Inf"
        assert scientific(float("-inf")) == "-Inf"

    def test_zero(self):
        assert scientific(0) == "0.00 x 10⁰"

    def test_positive_exponent(self):
        assert scientific(1000) == "1.00 x 10³"

    def test_large(self):
        assert scientific(1e40) == "1.00 x 10⁴⁰"


class TestClamp:
    def test_no_clamp(self):
        assert clamp(123.456) == "123.456"

    def test_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_ceil(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_callable_format(self):
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"

    def test_none(self):
        assert clamp(None) is None

    def test_non_finite(self):
        assert clamp(float("nan")) == "NaN"
        assert clamp(float("inf")) == "+Inf"
        assert clamp(float("-inf")) == "-Inf"

    def test_custom_tokens(self):
        assert clamp(0, floor=1, floor_token="min ") == "min 1"
        assert clamp(10, ceil=5, ceil_token="max ") == "max 5"

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            clamp(1, format=123)

    def test_within_bounds(self):
        assert clamp(5, floor=0, ceil=10) == "5"


class TestMetric:
    def test_basic(self):
        assert metric(1500, "V") == "1.50 kV"
        assert metric(2e8, "W") == "200 MW"
        assert metric(220e-6, "F") == "220 μF"

    def test_precision(self):
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_fallback_to_scientific(self):
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_zero(self):
        assert metric(0) == "0"

    def test_no_unit(self):
        assert metric(1000) == "1.00 k"
        assert metric(1e6) == "1.00 M"

    def test_degree_units(self):
        assert metric(1000, "°") == "1.00k°"

    def test_non_finite(self):
        assert metric(float("nan")) == "NaN"
        assert metric(float("inf")) == "+Inf"
        assert metric(float("-inf")) == "-Inf"

    def test_small_values(self):
        assert metric(1e-3, "s") == "1.00 ms"
        assert metric(1e-6, "s") == "1.00 μs"

    def test_carry(self):
        # 999.999 should carry to next bucket
        assert metric(999999.0) == "1.00 M"
