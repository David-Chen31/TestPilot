"""Tests for humanize.number."""

from __future__ import annotations

import math

import pytest

from humanize.number import (
    _format_not_finite,
    apnumber,
    clamp,
    fractional,
    intcomma,
    intword,
    metric,
    ordinal,
    scientific,
)


class TestFormatNotFinite:
    def test_nan(self):
        assert _format_not_finite(float("nan")) == "NaN"

    def test_negative_infinity(self):
        assert _format_not_finite(float("-inf")) == "-Inf"

    def test_positive_infinity(self):
        assert _format_not_finite(float("inf")) == "+Inf"

    def test_finite(self):
        assert _format_not_finite(1.0) == ""


class TestOrdinal:
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
            (111, "111th"),
            (1002, "1002nd"),
            (103, "103rd"),
            (0, "0th"),
        ],
    )
    def test_suffixes(self, value, expected):
        assert ordinal(value) == expected

    def test_negative(self):
        assert ordinal(-1) == "-1st"
        assert ordinal(-11) == "-11th"

    def test_string_number(self):
        assert ordinal("3") == "3rd"

    def test_float_truncated(self):
        assert ordinal(3.9) == "3rd"

    def test_non_numeric_string(self):
        assert ordinal("something else") == "something else"

    def test_list(self):
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"

    def test_none(self):
        assert ordinal(None) == "None"

    def test_nan(self):
        assert ordinal(float("nan")) == "NaN"

    def test_inf(self):
        assert ordinal(float("inf")) == "+Inf"
        assert ordinal(float("-inf")) == "-Inf"

    def test_gender_female(self):
        assert ordinal(1, gender="female") == "1st"
        assert ordinal(2, gender="female") == "2nd"

    def test_gender_other_defaults_female(self):
        # Any non-"male" gender becomes "female", which currently has same suffixes.
        assert ordinal(3, gender="unknown") == "3rd"


class TestIntcomma:
    def test_int(self):
        assert intcomma(100) == "100"
        assert intcomma(1000) == "1,000"
        assert intcomma(1_000_000) == "1,000,000"

    def test_string(self):
        assert intcomma("1000") == "1,000"
        assert intcomma("1000000") == "1,000,000"

    def test_float(self):
        assert intcomma(1_234_567.25) == "1,234,567.25"

    def test_negative(self):
        assert intcomma(-1000) == "-1,000"

    def test_ndigits(self):
        assert intcomma(1234.5454545, 2) == "1,234.55"
        assert intcomma(14308.40, 1) == "14,308.4"
        assert intcomma("14308.40", 1) == "14,308.4"

    def test_ndigits_zero(self):
        assert intcomma(1234.5, 0) == "1,234"

    def test_none(self):
        assert intcomma(None) == "None"

    def test_non_numeric_string(self):
        assert intcomma("foo") == "foo"

    def test_nan(self):
        assert intcomma(float("nan")) == "NaN"

    def test_inf(self):
        assert intcomma(float("inf")) == "+Inf"
        assert intcomma(float("-inf")) == "-Inf"

    def test_inf_as_string(self):
        assert intcomma("inf") == "+Inf"

    def test_bool(self):
        # bool is a subclass of int
        assert intcomma(True) == "1"
        assert intcomma(False) == "0"


class TestIntword:
    def test_small(self):
        assert intword("100") == "100"
        assert intword(999) == "999"

    def test_thousand(self):
        assert intword("12400") == "12.4 thousand"

    def test_million(self):
        assert intword("1000000") == "1.0 million"

    def test_billion(self):
        assert intword(1_200_000_000) == "1.2 billion"

    def test_decillion(self):
        assert intword(8100000000000000000000000000000000) == "8.1 decillion"

    def test_none(self):
        assert intword(None) == "None"

    def test_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_negative(self):
        assert intword(-1_200_000) == "-1.2 million"

    def test_negative_small(self):
        assert intword(-100) == "-100"

    def test_float_truncated(self):
        assert intword(1_200_000.9) == "1.2 million"

    def test_nan(self):
        assert intword(float("nan")) == "NaN"

    def test_inf(self):
        assert intword(float("inf")) == "+Inf"
        assert intword(float("-inf")) == "-Inf"

    def test_non_numeric_string(self):
        assert intword("foo") == "foo"

    def test_carry_rounding(self):
        # 999999 rounds to 1.0 million
        assert intword(999999) == "1.0 million"

    def test_googol(self):
        # 10**100 should become googol
        assert intword(10**100) == "1.0 googol"

    def test_carry_at_large_power(self):
        # Rounding carry at high powers should bump to the next unit
        result = intword(10**24 - 1)
        assert "septillion" in result


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
    def test_spelled(self, value, expected):
        assert apnumber(value) == expected

    def test_out_of_range(self):
        assert apnumber(10) == "10"
        assert apnumber(100) == "100"

    def test_negative(self):
        assert apnumber(-1) == "-1"

    def test_string_number(self):
        assert apnumber("7") == "seven"

    def test_non_numeric_string(self):
        assert apnumber("foo") == "foo"

    def test_none(self):
        assert apnumber(None) == "None"

    def test_nan(self):
        assert apnumber(float("nan")) == "NaN"

    def test_inf(self):
        assert apnumber(float("inf")) == "+Inf"
        assert apnumber(float("-inf")) == "-Inf"

    def test_float_truncated(self):
        assert apnumber(3.9) == "three"


class TestFractional:
    def test_simple_fraction(self):
        assert fractional(0.3) == "3/10"

    def test_mixed_fraction(self):
        assert fractional(1.3) == "1 3/10"

    def test_one_third(self):
        assert fractional(float(1 / 3)) == "1/3"

    def test_whole_number(self):
        assert fractional(1) == "1"

    def test_zero(self):
        assert fractional(0) == "0"

    def test_non_numeric_string(self):
        assert fractional("ten") == "ten"

    def test_none(self):
        assert fractional(None) == "None"

    def test_nan(self):
        assert fractional(float("nan")) == "NaN"

    def test_inf(self):
        assert fractional(float("inf")) == "+Inf"
        assert fractional(float("-inf")) == "-Inf"

    def test_negative_whole_number(self):
        assert fractional(-1) == "-1"

    def test_negative_mixed(self):
        assert fractional(-1.3) == "-1 3/10"

    def test_negative_simple_fraction(self):
        assert fractional(-0.5) == "-1/2"

    def test_rounds_up_to_whole(self):
        # 2.9999999 rounds up to a whole; should be 3
        assert fractional(2.9999999) == "3"

    def test_string_input(self):
        assert fractional("0.5") == "1/2"


class TestScientific:
    def test_small(self):
        assert scientific(float(0.3)) == "3.00 x 10⁻¹"

    def test_int(self):
        assert scientific(int(500)) == "5.00 x 10²"

    def test_negative(self):
        assert scientific(-1000) == "-1.00 x 10³"

    def test_precision_1(self):
        assert scientific(1000, 1) == "1.0 x 10³"

    def test_precision_3(self):
        assert scientific(1000, 3) == "1.000 x 10³"

    def test_string_number(self):
        assert scientific("99") == "9.90 x 10¹"

    def test_non_numeric_string(self):
        assert scientific("foo") == "foo"

    def test_none(self):
        assert scientific(None) == "None"

    def test_nan(self):
        assert scientific(float("nan")) == "NaN"

    def test_inf(self):
        assert scientific(float("inf")) == "+Inf"
        assert scientific(float("-inf")) == "-Inf"

    def test_zero(self):
        assert scientific(0) == "0.00 x 10⁰"

    def test_precision_zero(self):
        assert scientific(500, 0) == "5 x 10²"


class TestClamp:
    def test_no_clamp(self):
        assert clamp(123.456) == "123.456"

    def test_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_ceil_no_clamp_format(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"

    def test_ceil_clamp(self):
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_callable_format(self):
        assert (
            clamp(1, format=intword, floor=1e6, floor_token="under ")
            == "under 1.0 million"
        )

    def test_none_returns_none(self):
        assert clamp(None) is None

    def test_nan(self):
        assert clamp(float("nan")) == "NaN"

    def test_inf(self):
        assert clamp(float("inf")) == "+Inf"
        assert clamp(float("-inf")) == "-Inf"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            clamp(1, format=123)

    def test_custom_tokens(self):
        assert clamp(0, floor=10, floor_token="low: ") == "low: 10"
        assert clamp(20, ceil=10, ceil_token="high: ") == "high: 10"

    def test_int_input(self):
        assert clamp(5) == "5"


class TestMetric:
    def test_kilo(self):
        assert metric(1500, "V") == "1.50 kV"

    def test_mega(self):
        assert metric(2e8, "W") == "200 MW"

    def test_micro(self):
        assert metric(220e-6, "F") == "220 μF"

    def test_femto(self):
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_scientific_fallback_large(self):
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_scientific_fallback_small(self):
        result = metric(1e-40)
        assert "x 10" in result

    def test_nan(self):
        assert metric(float("nan")) == "NaN"

    def test_inf(self):
        assert metric(float("inf")) == "+Inf"
        assert metric(float("-inf")) == "-Inf"

    def test_zero(self):
        assert metric(0) == "0.00"

    def test_zero_with_unit(self):
        assert metric(0, "V") == "0.00 V"

    def test_no_unit_no_prefix(self):
        assert metric(1) == "1.00"

    def test_no_space_for_degree(self):
        assert metric(1500, "°") == "1.50k°"

    def test_no_space_for_prime(self):
        assert metric(1500, "′") == "1.50k′"

    def test_no_space_for_double_prime(self):
        assert metric(1500, "″") == "1.50k″"

    def test_negative_value(self):
        assert metric(-1500, "V") == "-1.50 kV"

    def test_carry_rounding(self):
        # 9.999k should round up to 10.0k, not "10.00k" with wrong prefix
        result = metric(9999.9)
        assert result.endswith("k")

    def test_giga(self):
        assert metric(1.5e9, "W") == "1.50 GW"
