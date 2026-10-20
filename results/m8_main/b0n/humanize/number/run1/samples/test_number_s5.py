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


class TestOrdinal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (5, "5th"),
            (6, "6th"),
            (7, "7th"),
            (8, "8th"),
            (9, "9th"),
            (10, "10th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (14, "14th"),
            (20, "20th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (24, "24th"),
            (100, "100th"),
            (101, "101st"),
            (102, "102nd"),
            (103, "103rd"),
            (111, "111th"),
            (112, "112th"),
            (113, "113th"),
            (1002, "1002nd"),
        ],
    )
    def test_ordinal_values(self, value: int, expected: str) -> None:
        assert ordinal(value) == expected

    def test_ordinal_string_number(self) -> None:
        assert ordinal("7") == "7th"

    def test_ordinal_float(self) -> None:
        assert ordinal(1.9) == "1st"

    def test_ordinal_not_convertible(self) -> None:
        assert ordinal("something else") == "something else"

    def test_ordinal_list(self) -> None:
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"

    def test_ordinal_none(self) -> None:
        assert ordinal(None) == "None"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_ordinal_not_finite(self, value: float, expected: str) -> None:
        assert ordinal(value) == expected

    def test_ordinal_gender(self) -> None:
        # English suffixes are the same for both genders.
        assert ordinal(1, gender="female") == "1st"
        assert ordinal(2, gender="female") == "2nd"

    def test_ordinal_unknown_gender_defaults_to_female(self) -> None:
        assert ordinal(3, gender="unknown") == "3rd"

    def test_ordinal_negative(self) -> None:
        assert ordinal(-1) == "-1st"
        assert ordinal(-11) == "-11th"


class TestIntcomma:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (100, "100"),
            ("1000", "1,000"),
            (1_000_000, "1,000,000"),
            (1_234_567.25, "1,234,567.25"),
            (-1000, "-1,000"),
            (999, "999"),
            (1000, "1,000"),
        ],
    )
    def test_intcomma_values(self, value, expected: str) -> None:
        assert intcomma(value) == expected

    def test_intcomma_ndigits(self) -> None:
        assert intcomma(1234.5454545, 2) == "1,234.55"

    def test_intcomma_ndigits_trailing_zero(self) -> None:
        assert intcomma(14308.40, 1) == "14,308.4"

    def test_intcomma_string_ndigits(self) -> None:
        assert intcomma("14308.40", 1) == "14,308.4"

    def test_intcomma_string_with_commas(self) -> None:
        assert intcomma("1,000,000") == "1,000,000"

    def test_intcomma_none(self) -> None:
        assert intcomma(None) == "None"

    def test_intcomma_invalid_string(self) -> None:
        assert intcomma("foo") == "foo"

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
    def test_intcomma_not_finite(self, value, expected: str) -> None:
        assert intcomma(value) == expected

    def test_intcomma_bool(self) -> None:
        # bool is a subclass of int
        assert intcomma(True) == "1"
        assert intcomma(False) == "0"


class TestIntword:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("100", "100"),
            (100, "100"),
            ("12400", "12.4 thousand"),
            (12400, "12.4 thousand"),
            ("1000000", "1.0 million"),
            (1_200_000, "1.2 million"),
            (1_200_000_000, "1.2 billion"),
            (8100000000000000000000000000000000, "8.1 decillion"),
        ],
    )
    def test_intword_values(self, value, expected: str) -> None:
        assert intword(value) == expected

    def test_intword_none(self) -> None:
        assert intword(None) == "None"

    def test_intword_invalid(self) -> None:
        assert intword("foo") == "foo"

    def test_intword_custom_format(self) -> None:
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_intword_negative(self) -> None:
        assert intword(-1_200_000) == "-1.2 million"
        assert intword(-100) == "-100"

    def test_intword_below_thousand(self) -> None:
        assert intword(999) == "999"

    def test_intword_zero(self) -> None:
        assert intword(0) == "0"

    def test_intword_googol(self) -> None:
        assert intword(10**100) == "1.0 googol"

    def test_intword_rounding_carry(self) -> None:
        # 999,999 rounds up to 1000.0 thousand -> 1.0 million
        assert intword(999_999) == "1.0 million"

    def test_intword_rounding_carry_large(self) -> None:
        # Previously rendered as "1000.0 sextillion"
        assert intword(10**24 - 1) == "1.0 septillion"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_intword_not_finite(self, value: float, expected: str) -> None:
        assert intword(value) == expected


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
            (-1, "-1"),
            (100, "100"),
        ],
    )
    def test_apnumber_values(self, value, expected: str) -> None:
        assert apnumber(value) == expected

    def test_apnumber_string_number(self) -> None:
        assert apnumber("7") == "seven"

    def test_apnumber_string_non_number(self) -> None:
        assert apnumber("foo") == "foo"

    def test_apnumber_none(self) -> None:
        assert apnumber(None) == "None"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_apnumber_not_finite(self, value: float, expected: str) -> None:
        assert apnumber(value) == expected


class TestFractional:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.3, "3/10"),
            (1.3, "1 3/10"),
            (1 / 3, "1/3"),
            (1, "1"),
            (2, "2"),
            (0, "0"),
        ],
    )
    def test_fractional_values(self, value, expected: str) -> None:
        assert fractional(value) == expected

    def test_fractional_string_number(self) -> None:
        assert fractional("0.3") == "3/10"

    def test_fractional_string_non_number(self) -> None:
        assert fractional("ten") == "ten"

    def test_fractional_none(self) -> None:
        assert fractional(None) == "None"

    def test_fractional_negative(self) -> None:
        assert fractional(-1.3) == "-1 3/10"

    def test_fractional_negative_pure_fraction(self) -> None:
        assert fractional(-0.3) == "-3/10"

    def test_fractional_rounds_up_to_whole(self) -> None:
        # Fractional part rounds up to 1/1 -> folded into integer?
        # 2.9999999 -> frac(0.9999999) limit_denominator(1000) -> 1/1
        assert fractional(2.9999999) == "2"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_fractional_not_finite(self, value: float, expected: str) -> None:
        assert fractional(value) == expected


class TestScientific:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.3, "3.00 x 10⁻¹"),
            (500, "5.00 x 10²"),
            (-1000, "-1.00 x 10³"),
            (1000, "1.00 x 10³"),
            (99, "9.90 x 10¹"),
            (1, "1.00 x 10⁰"),
        ],
    )
    def test_scientific_values(self, value, expected: str) -> None:
        assert scientific(value) == expected

    def test_scientific_precision_1(self) -> None:
        assert scientific(1000, 1) == "1.0 x 10³"

    def test_scientific_precision_3(self) -> None:
        assert scientific(1000, 3) == "1.000 x 10³"

    def test_scientific_string_number(self) -> None:
        assert scientific("99") == "9.90 x 10¹"

    def test_scientific_string_non_number(self) -> None:
        assert scientific("foo") == "foo"

    def test_scientific_none(self) -> None:
        assert scientific(None) == "None"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_scientific_not_finite(self, value: float, expected: str) -> None:
        assert scientific(value) == expected


class TestClamp:
    def test_clamp_no_bounds(self) -> None:
        assert clamp(123.456) == "123.456"

    def test_clamp_floor(self) -> None:
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_clamp_floor_not_triggered(self) -> None:
        assert clamp(1, floor=0.01) == "1"

    def test_clamp_ceil_format(self) -> None:
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"

    def test_clamp_ceil_triggered(self) -> None:
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_clamp_callable_format(self) -> None:
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
            "under 1.0 million"
        )

    def test_clamp_none(self) -> None:
        assert clamp(None) is None

    def test_clamp_custom_tokens(self) -> None:
        assert clamp(0, floor=1, floor_token="min ") == "min 1"
        assert clamp(10, ceil=5, ceil_token="max ") == "max 5"

    def test_clamp_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid format"):
            clamp(1, format=123)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_clamp_not_finite(self, value: float, expected: str) -> None:
        assert clamp(value) == expected

    def test_clamp_floor_and_ceil(self) -> None:
        assert clamp(0.5, floor=0, ceil=1) == "0.5"
        assert clamp(-1, floor=0, ceil=1) == "<0"
        assert clamp(2, floor=0, ceil=1) == ">1"


class TestMetric:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (1500, "V", "1.50 kV"),
            (2e8, "W", "200 MW"),
            (220e-6, "F", "220 μF"),
            (0, "", "0.00"),
            (1, "", "1.00"),
            (1000, "", "1.00 k"),
            (1_000_000, "", "1.00 M"),
        ],
    )
    def test_metric_values(self, value: float, unit: str, expected: str) -> None:
        assert metric(value, unit) == expected

    def test_metric_default_unit(self) -> None:
        assert metric(1500) == "1.50 k"

    def test_metric_precision(self) -> None:
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_metric_precision_2(self) -> None:
        assert metric(1230, precision=2) == "1.2 k"

    def test_metric_falls_back_to_scientific(self) -> None:
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_metric_falls_back_to_scientific_tiny(self) -> None:
        assert metric(1e-40) == "1.00 x 10⁻⁴⁰"

    def test_metric_negative(self) -> None:
        assert metric(-1500, "V") == "-1.50 kV"

    def test_metric_rounding_carry(self) -> None:
        # 999.999 -> 1000 -> should bump to k
        assert metric(999.999) == "1.00 k"

    def test_metric_degree_unit_no_space(self) -> None:
        assert metric(1500, "°") == "1.50k°"

    def test_metric_prime_unit_no_space(self) -> None:
        assert metric(1500, "′") == "1.50k′"

    def test_metric_no_unit_no_prefix(self) -> None:
        # 0.5 has exponent -1 -> bucket -3, ordinal "m"
        assert metric(0.5) == "500.00 m"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (float("nan"), "NaN"),
            (float("inf"), "+Inf"),
            (float("-inf"), "-Inf"),
        ],
    )
    def test_metric_not_finite(self, value: float, expected: str) -> None:
        assert metric(value) == expected

    def test_metric_milli(self) -> None:
        assert metric(0.005) == "5.00 m"

    def test_metric_micro(self) -> None:
        assert metric(0.000005) == "5.00 μ"

    def test_metric_giga(self) -> None:
        assert metric(1e9, "Hz") == "1.00 GHz"


class TestFormatNotFiniteIndirect:
    """Ensure _format_not_finite is exercised through several entry points."""

    def test_all_functions_handle_nan(self) -> None:
        assert ordinal(float("nan")) == "NaN"
        assert intcomma(float("nan")) == "NaN"
        assert intword(float("nan")) == "NaN"
        assert apnumber(float("nan")) == "NaN"
        assert fractional(float("nan")) == "NaN"
        assert scientific(float("nan")) == "NaN"
        assert clamp(float("nan")) == "NaN"
        assert metric(float("nan")) == "NaN"

    def test_all_functions_handle_pos_inf(self) -> None:
        assert ordinal(float("inf")) == "+Inf"
        assert intcomma(float("inf")) == "+Inf"
        assert intword(float("inf")) == "+Inf"
        assert apnumber(float("inf")) == "+Inf"
        assert fractional(float("inf")) == "+Inf"
        assert scientific(float("inf")) == "+Inf"
        assert clamp(float("inf")) == "+Inf"
        assert metric(float("inf")) == "+Inf"

    def test_all_functions_handle_neg_inf(self) -> None:
        assert ordinal(float("-inf")) == "-Inf"
        assert intcomma(float("-inf")) == "-Inf"
        assert intword(float("-inf")) == "-Inf"
        assert apnumber(float("-inf")) == "-Inf"
        assert fractional(float("-inf")) == "-Inf"
        assert scientific(float("-inf")) == "-Inf"
        assert clamp(float("-inf")) == "-Inf"
        assert metric(float("-inf")) == "-Inf"


class TestMathConsistency:
    def test_is_nan_sanity(self) -> None:
        assert math.isnan(float("nan"))
