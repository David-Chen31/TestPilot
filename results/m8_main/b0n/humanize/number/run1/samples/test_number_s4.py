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
class TestOrdinal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (5, "5th"),
            (9, "9th"),
            (0, "0th"),
            (10, "10th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (14, "14th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (24, "24th"),
            (100, "100th"),
            (101, "101st"),
            (102, "102nd"),
            (103, "103rd"),
            (104, "104th"),
            (111, "111th"),
            (112, "112th"),
            (113, "113th"),
            (121, "121st"),
            (1002, "1002nd"),
        ],
    )
    def test_basic(self, value, expected):
        assert ordinal(value) == expected

    def test_string_intable(self):
        assert ordinal("3") == "3rd"
        assert ordinal("42") == "42nd"

    def test_float_intable(self):
        assert ordinal(3.9) == "3rd"
        assert ordinal(-1.5) == "-1st"

    def test_negative(self):
        assert ordinal(-1) == "-1st"
        assert ordinal(-11) == "-11th"

    def test_non_intable_returns_str(self):
        assert ordinal("something else") == "something else"
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"
        assert ordinal(None) == "None"
        assert ordinal({"a": 1}) == "{'a': 1}"

    def test_gender_male_female(self):
        assert ordinal(1, "male") == "1st"
        assert ordinal(1, "female") == "1st"
        assert ordinal(2, "female") == "2nd"

    def test_gender_unknown_defaults_female(self):
        # Any non-"male" string maps to "female".
        assert ordinal(3, "other") == "3rd"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
            ("nan", "NaN"),
            ("inf", "+Inf"),
            ("-inf", "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert ordinal(value) == expected


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
class TestIntcomma:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (100, "100"),
            (1000, "1,000"),
            ("1000", "1,000"),
            (1_000_000, "1,000,000"),
            (1_234_567.25, "1,234,567.25"),
            (0, "0"),
            (-1234, "-1,234"),
            (-1234567, "-1,234,567"),
            (123, "123"),
        ],
    )
    def test_basic(self, value, expected):
        assert intcomma(value) == expected

    @pytest.mark.parametrize(
        ("value", "ndigits", "expected"),
        [
            (1234.5454545, 2, "1,234.55"),
            (14308.40, 1, "14,308.4"),
            ("14308.40", 1, "14,308.4"),
            (1000, 2, "1,000.00"),
            (1.005, 2, "1.00"),
            (1.015, 2, "1.01"),
        ],
    )
    def test_ndigits(self, value, ndigits, expected):
        assert intcomma(value, ndigits) == expected

    def test_string_with_commas(self):
        assert intcomma("1,000") == "1,000"
        assert intcomma("1,234,567") == "1,234,567"

    def test_none(self):
        assert intcomma(None) == "None"

    def test_non_numeric_string(self):
        assert intcomma("foo") == "foo"

    def test_non_numeric_object(self):
        assert intcomma([1, 2]) == "[1, 2]"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
            ("nan", "NaN"),
            ("inf", "+Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert intcomma(value) == expected

    def test_bool_is_int(self):
        # bool is a subclass of int.
        assert intcomma(True) == "1"
        assert intcomma(False) == "0"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
class TestIntword:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("100", "100"),
            (0, "0"),
            (999, "999"),
            (1000, "1.0 thousand"),
            ("12400", "12.4 thousand"),
            (45_000, "45.0 thousand"),
            ("1000000", "1.0 million"),
            (1_200_000, "1.2 million"),
            (1_200_000_000, "1.2 billion"),
            (1_000_000_000_000, "1.0 trillion"),
            (2_000_000_000_000_000, "2.0 quadrillion"),
            (8100000000000000000000000000000000, "8.1 decillion"),
        ],
    )
    def test_basic(self, value, expected):
        assert intword(value) == expected

    def test_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"
        assert intword(1_500_000, "%.0f") == "2 million"

    def test_negative(self):
        assert intword(-1000) == "-1.0 thousand"
        assert intword(-1_200_000) == "-1.2 million"
        assert intword(-100) == "-100"

    def test_none(self):
        assert intword(None) == "None"

    def test_non_numeric(self):
        assert intword("foo") == "foo"
        assert intword([1]) == "[1]"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert intword(value) == expected

    def test_googol(self):
        assert intword(10**100) == "1.0 googol"
        assert intword(10**101) == "10.0 googol"

    def test_rounding_carry_to_next_power(self):
        # 10**24 - 1 should carry up to 1.0 septillion, not "1000.0 sextillion".
        assert intword(10**24 - 1) == "1.0 septillion"

    def test_largest_ordinal_no_carry(self):
        # Beyond googol there is no next power, so no carry logic is applied.
        result = intword(10**101)
        assert result == "10.0 googol"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
class TestApnumber:
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
            ("7", "seven"),
            ("10", "10"),
            (9.9, "nine"),
        ],
    )
    def test_basic(self, value, expected):
        assert apnumber(value) == expected

    def test_negative(self):
        assert apnumber(-1) == "-1"
        assert apnumber(-5) == "-5"

    def test_non_intable(self):
        assert apnumber("foo") == "foo"
        assert apnumber(None) == "None"
        assert apnumber([1]) == "[1]"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert apnumber(value) == expected


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
class TestFractional:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.3, "3/10"),
            (1.3, "1 3/10"),
            (float(1 / 3), "1/3"),
            (1, "1"),
            (2, "2"),
            (0, "0"),
            (0.5, "1/2"),
            (0.25, "1/4"),
            (1.5, "1 1/2"),
            (2.25, "2 1/4"),
            ("0.75", "3/4"),
            (1.0000, "1"),
        ],
    )
    def test_basic(self, value, expected):
        assert fractional(value) == expected

    def test_negative(self):
        assert fractional(-0.3) == "-3/10"
        assert fractional(-1.3) == "-1 3/10"
        assert fractional(-1.5) == "-1 1/2"

    def test_fraction_rounds_up_to_whole(self):
        assert fractional(2.9999999) == "3"

    def test_non_numeric(self):
        assert fractional("ten") == "ten"
        assert fractional(None) == "None"
        assert fractional([1]) == "[1]"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert fractional(value) == expected

    def test_float_denominator_limit(self):
        # limit_denominator(1000) is applied.
        result = fractional(math.pi)
        assert result == "3 16/113"


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
class TestScientific:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float(0.3), "3.00 x 10⁻¹"),
            (int(500), "5.00 x 10²"),
            (-1000, "-1.00 x 10³"),
            (1000, "1.00 x 10³"),
            ("99", "9.90 x 10¹"),
            (1000, "1.00 x 10³"),
            (0, "0.00 x 10⁰"),
            (1, "1.00 x 10⁰"),
            (10, "1.00 x 10¹"),
            (0.001, "1.00 x 10⁻³"),
            (123456, "1.23 x 10⁵"),
        ],
    )
    def test_basic(self, value, expected):
        assert scientific(value) == expected

    @pytest.mark.parametrize(
        ("value", "precision", "expected"),
        [
            (1000, 1, "1.0 x 10³"),
            (1000, 3, "1.000 x 10³"),
            (0.3, 0, "3 x 10⁻¹"),
            (500, 4, "5.0000 x 10²"),
        ],
    )
    def test_precision(self, value, precision, expected):
        assert scientific(value, precision) == expected

    def test_non_numeric(self):
        assert scientific("foo") == "foo"
        assert scientific(None) == "None"
        assert scientific([1]) == "[1]"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert scientific(value) == expected


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_basic(self):
        assert clamp(123.456) == "123.456"
        assert clamp(0) == "0"

    def test_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"
        assert clamp(0.5, floor=0.1) == "0.5"
        assert clamp(0.1, floor=0.1) == "0.1"

    def test_ceil(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_custom_tokens(self):
        assert clamp(0.0001, floor=0.01, floor_token="low: ") == "low: 0.01"
        assert clamp(100, ceil=10, ceil_token="high: ") == "high: 10"

    def test_callable_format(self):
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"
        assert clamp(1_500_000, format=intword) == "1.5 million"

    def test_none_returns_none(self):
        assert clamp(None) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert clamp(value) == expected

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            clamp(1, format=123)

    def test_invalid_format_message(self):
        with pytest.raises(ValueError) as excinfo:
            clamp(1, format=object())
        assert "Invalid format" in str(excinfo.value)


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
class TestMetric:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (1500, "V", "1.50 kV"),
            (2e8, "W", "200 MW"),
            (220e-6, "F", "220 μF"),
            (1e-14, "", "10.00 f"),
            (1000, "V", "1.00 kV"),
            (1, "V", "1.00 V"),
            (0, "V", "0.00 V"),
            (100, "V", "100 V"),
            (123_000, "", "123 k"),
            (1_230_000, "", "1.23 M"),
            (1e-6, "F", "1.00 μF"),
            (1e-9, "F", "1.00 nF"),
            (1e-12, "F", "1.00 pF"),
            (1e-15, "F", "1.00 fF"),
            (1e-18, "F", "1.00 aF"),
            (1e6, "", "1.00 M"),
            (1e9, "", "1.00 G"),
            (1e12, "", "1.00 T"),
            (1e15, "", "1.00 P"),
            (1e18, "", "1.00 E"),
            (1e21, "", "1.00 Z"),
            (1e24, "", "1.00 Y"),
            (1e27, "", "1.00 R"),
            (1e30, "", "1.00 Q"),
        ],
    )
    def test_basic(self, value, unit, expected):
        assert metric(value, unit) == expected

    def test_negative(self):
        assert metric(-1500, "V") == "-1.50 kV"
        assert metric(-1e-6, "F") == "-1.00 μF"

    @pytest.mark.parametrize(
        ("value", "precision", "expected"),
        [
            (1e-14, 4, "10.00 f"),
            (1500, 2, "1.5 kV"),
            (1500, 4, "1.500 kV"),
            (123_000, 3, "123 k"),
            (1_230_000, 3, "1.23 M"),
        ],
    )
    def test_precision(self, value, precision, expected):
        assert metric(value, "", precision) == expected

    def test_degree_units_no_space(self):
        assert metric(1500, "°") == "1.50k°"
        assert metric(1500, "′") == "1.50k′"
        assert metric(1500, "″") == "1.50k″"

    def test_unit_with_no_ordinal(self):
        # No prefix and a unit: single space.
        assert metric(1, "m") == "1.00 m"
        # No unit and no prefix: no space.
        assert metric(1, "") == "1.00"

    def test_too_large_falls_back_to_scientific(self):
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_too_small_falls_back_to_scientific(self):
        assert metric(1e-40) == "1.00 x 10⁻⁴⁰"

    def test_fallback_includes_unit(self):
        assert metric(1e40, "V") == "1.00 x 10⁴⁰V"

    def test_rounding_carry(self):
        # 9.999 -> 10.0 should bump the exponent.
        assert metric(999_950, "", 3) == "1.00 M"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_not_finite(self, value, expected):
        assert metric(value) == expected
