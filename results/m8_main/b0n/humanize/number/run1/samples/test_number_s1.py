"""Tests for humanize.number."""

from __future__ import annotations

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
            (111, "111th"),
            (112, "112th"),
            (113, "113th"),
            (1002, "1002nd"),
            (0, "0th"),
            (-1, "-1th" if False else "-1st"),
        ],
    )
    def test_ordinal_basic(self, value, expected):
        assert ordinal(value) == expected

    def test_ordinal_string_intable(self):
        assert ordinal("7") == "7th"

    def test_ordinal_string_not_intable(self):
        assert ordinal("something else") == "something else"

    def test_ordinal_list(self):
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"

    def test_ordinal_float_rounding_toward_zero(self):
        assert ordinal(1.9) == "1st"

    def test_ordinal_gender_default_male(self):
        assert ordinal(1) == "1st"

    def test_ordinal_gender_female(self):
        assert ordinal(1, gender="female") == "1st"

    def test_ordinal_invalid_gender_treated_as_female(self):
        assert ordinal(2, gender="other") == "2nd"

    def test_ordinal_inf(self):
        assert ordinal(float("inf")) == "+Inf"

    def test_ordinal_neg_inf(self):
        assert ordinal(float("-inf")) == "-Inf"

    def test_ordinal_nan(self):
        assert ordinal(float("nan")) == "NaN"

    def test_ordinal_none(self):
        assert ordinal(None) == "None"

    def test_ordinal_bool(self):
        assert ordinal(True) == "1st"
        assert ordinal(False) == "0th"


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
class TestIntcomma:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (100, "100"),
            ("1000", "1,000"),
            (1_000_000, "1,000,000"),
            (1_234_567.25, "1,234,567.25"),
            (0, "0"),
            (-1000, "-1,000"),
        ],
    )
    def test_intcomma_basic(self, value, expected):
        assert intcomma(value) == expected

    @pytest.mark.parametrize(
        "value,ndigits,expected",
        [
            (1234.5454545, 2, "1,234.55"),
            (14308.40, 1, "14,308.4"),
            ("14308.40", 1, "14,308.4"),
            (1234.5, 0, "1,234"),
        ],
    )
    def test_intcomma_ndigits(self, value, ndigits, expected):
        assert intcomma(value, ndigits) == expected

    def test_intcomma_none(self):
        assert intcomma(None) == "None"

    def test_intcomma_non_numeric_str(self):
        assert intcomma("foo") == "foo"

    def test_intcomma_list(self):
        assert intcomma([1, 2, 3]) == "[1, 2, 3]"

    def test_intcomma_inf(self):
        assert intcomma(float("inf")) == "+Inf"

    def test_intcomma_neg_inf(self):
        assert intcomma(float("-inf")) == "-Inf"

    def test_intcomma_nan(self):
        assert intcomma(float("nan")) == "NaN"

    def test_intcomma_string_inf(self):
        assert intcomma("inf") == "+Inf"

    def test_intcomma_string_with_existing_commas(self):
        assert intcomma("1,234,567") == "1,234,567"

    def test_intcomma_bool(self):
        assert intcomma(True) == "1"
        assert intcomma(False) == "0"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
class TestIntword:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("100", "100"),
            ("12400", "12.4 thousand"),
            ("1000000", "1.0 million"),
            (1_200_000_000, "1.2 billion"),
            (8100000000000000000000000000000000, "8.1 decillion"),
            (999, "999"),
            (1000, "1.0 thousand"),
            (-12400, "-12.4 thousand"),
            (0, "0"),
        ],
    )
    def test_intword_basic(self, value, expected):
        assert intword(value) == expected

    def test_intword_none(self):
        assert intword(None) == "None"

    def test_intword_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_intword_inf(self):
        assert intword(float("inf")) == "+Inf"

    def test_intword_neg_inf(self):
        assert intword(float("-inf")) == "-Inf"

    def test_intword_nan(self):
        assert intword(float("nan")) == "NaN"

    def test_intword_non_numeric(self):
        assert intword("foo") == "foo"

    def test_intword_googol(self):
        assert intword(10**100) == "1.0 googol"

    def test_intword_float_truncated(self):
        assert intword(1500.9) == "1.5 thousand"

    def test_intword_rounding_carry(self):
        # 999_999 rounds up to "1000.0" -> must carry to million
        assert intword(999_999) == "1.0 million"

    def test_intword_large_carry(self):
        # 10**24 - 1 should carry to septillion
        result = intword(10**24 - 1)
        assert result == "1.0 septillion"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
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
            (10, "10"),
            (100, "100"),
            (-1, "-1"),
            ("7", "seven"),
        ],
    )
    def test_apnumber_basic(self, value, expected):
        assert apnumber(value) == expected

    def test_apnumber_non_numeric(self):
        assert apnumber("foo") == "foo"

    def test_apnumber_none(self):
        assert apnumber(None) == "None"

    def test_apnumber_inf(self):
        assert apnumber(float("inf")) == "+Inf"

    def test_apnumber_neg_inf(self):
        assert apnumber(float("-inf")) == "-Inf"

    def test_apnumber_nan(self):
        assert apnumber(float("nan")) == "NaN"

    def test_apnumber_float_truncated(self):
        assert apnumber(9.9) == "nine"


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
class TestFractional:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.3, "3/10"),
            (1.3, "1 3/10"),
            (1 / 3, "1/3"),
            (1, "1"),
            (0, "0"),
            (-1.3, "-1 3/10"),
            (0.5, "1/2"),
            (2.9999999, "3"),
        ],
    )
    def test_fractional_basic(self, value, expected):
        assert fractional(value) == expected

    def test_fractional_string(self):
        assert fractional("0.25") == "1/4"

    def test_fractional_non_numeric(self):
        assert fractional("ten") == "ten"

    def test_fractional_none(self):
        assert fractional(None) == "None"

    def test_fractional_inf(self):
        assert fractional(float("inf")) == "+Inf"

    def test_fractional_neg_inf(self):
        assert fractional(float("-inf")) == "-Inf"

    def test_fractional_nan(self):
        assert fractional(float("nan")) == "NaN"

    def test_fractional_list(self):
        assert fractional([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
class TestScientific:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.3, "3.00 x 10⁻¹"),
            (500, "5.00 x 10²"),
            (-1000, "-1.00 x 10³"),
            (1000, "1.00 x 10³"),
            ("99", "9.90 x 10¹"),
            (1, "1.00 x 10⁰"),
            (0, "0.00 x 10⁰"),
        ],
    )
    def test_scientific_default_precision(self, value, expected):
        assert scientific(value) == expected

    def test_scientific_precision_1(self):
        assert scientific(1000, 1) == "1.0 x 10³"

    def test_scientific_precision_3(self):
        assert scientific(1000, 3) == "1.000 x 10³"

    def test_scientific_precision_0(self):
        assert scientific(1000, 0) == "1 x 10³"

    def test_scientific_non_numeric(self):
        assert scientific("foo") == "foo"

    def test_scientific_none(self):
        assert scientific(None) == "None"

    def test_scientific_inf(self):
        assert scientific(float("inf")) == "+Inf"

    def test_scientific_neg_inf(self):
        assert scientific(float("-inf")) == "-Inf"

    def test_scientific_nan(self):
        assert scientific(float("nan")) == "NaN"

    def test_scientific_large_exponent(self):
        assert scientific(1e100) == "1.00 x 10¹⁰⁰"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_clamp_plain(self):
        assert clamp(123.456) == "123.456"

    def test_clamp_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_clamp_floor_not_triggered(self):
        assert clamp(0.5, floor=0.01) == "0.5"

    def test_clamp_ceil_exact(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"

    def test_clamp_ceil_exceeded(self):
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_clamp_callable_format(self):
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"

    def test_clamp_none(self):
        assert clamp(None) is None

    def test_clamp_inf(self):
        assert clamp(float("inf")) == "+Inf"

    def test_clamp_neg_inf(self):
        assert clamp(float("-inf")) == "-Inf"

    def test_clamp_nan(self):
        assert clamp(float("nan")) == "NaN"

    def test_clamp_custom_tokens(self):
        assert clamp(0, floor=1, floor_token="less than ") == "less than 1"
        assert clamp(10, ceil=5, ceil_token="more than ") == "more than 5"

    def test_clamp_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid format"):
            clamp(1, format=123)

    def test_clamp_int_input(self):
        assert clamp(5) == "5"


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
class TestMetric:
    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            (1500, "V", "1.50 kV"),
            (2e8, "W", "200 MW"),
            (220e-6, "F", "220 μF"),
            (123_000, "", "123 k"),
            (1_230_000, "", "1.23 M"),
            (0, "", "0.00 "),
            (1, "", "1.00 "),
            (-1500, "V", "-1.50 kV"),
        ],
    )
    def test_metric_basic(self, value, unit, expected):
        assert metric(value, unit) == expected

    def test_metric_precision_4_tiny(self):
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_metric_fallback_scientific_huge(self):
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_metric_fallback_scientific_tiny(self):
        result = metric(1e-32)
        assert "x 10" in result

    def test_metric_degree_no_space(self):
        assert metric(1500, "°") == "1.50 k°"

    def test_metric_prime_no_space(self):
        assert metric(1500, "′") == "1.50 k′"

    def test_metric_double_prime_no_space(self):
        assert metric(1500, "″") == "1.50 k″"

    def test_metric_inf(self):
        assert metric(float("inf")) == "+Inf"

    def test_metric_neg_inf(self):
        assert metric(float("-inf")) == "-Inf"

    def test_metric_nan(self):
        assert metric(float("nan")) == "NaN"

    def test_metric_rounding_carry(self):
        # 999.9k should carry to 1.00 M
        result = metric(999_900)
        assert result == "1.00 M"

    def test_metric_no_unit_no_prefix(self):
        assert metric(0.5) == "500 m"
