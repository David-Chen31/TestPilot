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
class TestOrdinal:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
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
            (-1, "-1st"),
        ],
    )
    def test_basic(self, value, expected):
        assert ordinal(value) == expected

    def test_string_numeric(self):
        assert ordinal("7") == "7th"

    def test_non_numeric_string(self):
        assert ordinal("something else") == "something else"

    def test_list(self):
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"

    def test_none(self):
        assert ordinal(None) == "None"

    def test_float_truncates(self):
        # int() truncates toward zero
        assert ordinal(2.9) == "2nd"

    def test_gender_female(self):
        assert ordinal(1, "female") == "1st"
        assert ordinal(2, "female") == "2nd"
        assert ordinal(3, "female") == "3rd"

    def test_gender_unknown_defaults_to_female(self):
        # any non-"male" is treated as female
        assert ordinal(1, "x") == "1st"

    def test_nan(self):
        assert ordinal(float("nan")) == "NaN"

    def test_positive_inf(self):
        assert ordinal(float("inf")) == "+Inf"

    def test_negative_inf(self):
        assert ordinal(float("-inf")) == "-Inf"


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
class TestIntcomma:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (100, "100"),
            (1000, "1,000"),
            (1_000_000, "1,000,000"),
            ("1000", "1,000"),
            (1_234_567.25, "1,234,567.25"),
            (-1234, "-1,234"),
        ],
    )
    def test_basic(self, value, expected):
        assert intcomma(value) == expected

    def test_ndigits(self):
        assert intcomma(1234.5454545, 2) == "1,234.55"

    def test_ndigits_float_string(self):
        assert intcomma("14308.40", 1) == "14,308.4"

    def test_ndigits_float(self):
        assert intcomma(14308.40, 1) == "14,308.4"

    def test_none(self):
        assert intcomma(None) == "None"

    def test_non_numeric_string(self):
        assert intcomma("foo") == "foo"

    def test_nan(self):
        assert intcomma(float("nan")) == "NaN"

    def test_inf(self):
        assert intcomma(float("inf")) == "+Inf"
        assert intcomma(float("-inf")) == "-Inf"

    def test_string_nan(self):
        assert intcomma("nan") == "NaN"

    def test_string_inf(self):
        assert intcomma("-inf") == "-Inf"

    def test_strips_thousands_sep_in_string(self):
        assert intcomma("1,000,000") == "1,000,000"

    def test_zero(self):
        assert intcomma(0) == "0"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
class TestIntword:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("100", "100"),
            ("12400", "12.4 thousand"),
            ("1000000", "1.0 million"),
            (1_200_000_000, "1.2 billion"),
            (8100000000000000000000000000000000, "8.1 decillion"),
            (1_000, "1.0 thousand"),
            (999, "999"),
        ],
    )
    def test_basic(self, value, expected):
        assert intword(value) == expected

    def test_negative(self):
        assert intword(-1_200_000) == "-1.2 million"

    def test_negative_small(self):
        assert intword(-100) == "-100"

    def test_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_none(self):
        assert intword(None) == "None"

    def test_non_numeric(self):
        assert intword("foo") == "foo"

    def test_nan(self):
        assert intword(float("nan")) == "NaN"

    def test_inf(self):
        assert intword(float("inf")) == "+Inf"
        assert intword(float("-inf")) == "-Inf"

    def test_float_truncates(self):
        # values are truncated to int before processing
        assert intword("999.9") == "999"

    def test_googol(self):
        result = intword(10**100)
        assert result == "1.0 googol"

    def test_rounding_carry(self):
        # Rounding near a power boundary should bump to next unit.
        assert intword(999_999) == "1.0 million"

    def test_large_rounding_carry(self):
        # Regression: 10**24 - 1 should render as 1.0 septillion, not 1000.0 sextillion
        assert intword(10**24 - 1) == "1.0 septillion"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
class TestApnumber:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "zero"),
            (1, "one"),
            (5, "five"),
            (9, "nine"),
            (10, "10"),
            (-1, "-1"),
            ("7", "seven"),
        ],
    )
    def test_basic(self, value, expected):
        assert apnumber(value) == expected

    def test_non_numeric(self):
        assert apnumber("foo") == "foo"

    def test_none(self):
        assert apnumber(None) == "None"

    def test_float_truncates(self):
        assert apnumber(5.9) == "five"

    def test_nan(self):
        assert apnumber(float("nan")) == "NaN"

    def test_inf(self):
        assert apnumber(float("inf")) == "+Inf"
        assert apnumber(float("-inf")) == "-Inf"


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
class TestFractional:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.3, "3/10"),
            (1.3, "1 3/10"),
            (1 / 3, "1/3"),
            (1, "1"),
            (2, "2"),
            (0, "0"),
        ],
    )
    def test_basic(self, value, expected):
        assert fractional(value) == expected

    def test_negative_mixed(self):
        assert fractional(-1.3) == "-1 3/10"

    def test_negative_fraction(self):
        assert fractional(-0.3) == "-3/10"

    def test_non_numeric(self):
        assert fractional("ten") == "ten"

    def test_none(self):
        assert fractional(None) == "None"

    def test_nan(self):
        assert fractional(float("nan")) == "NaN"

    def test_inf(self):
        assert fractional(float("inf")) == "+Inf"
        assert fractional(float("-inf")) == "-Inf"

    def test_rounds_up_to_whole(self):
        # 2.9999999 -> fractional part rounds to 1/1 -> fold into integer
        assert fractional(2.9999999) == "3"

    def test_float_integral(self):
        assert fractional(1.0000) == "1"

    def test_string_number(self):
        assert fractional("0.5") == "1/2"


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
class TestScientific:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.3, "3.00 x 10⁻¹"),
            (500, "5.00 x 10²"),
            (-1000, "-1.00 x 10³"),
            ("99", "9.90 x 10¹"),
            (1, "1.00 x 10⁰"),
        ],
    )
    def test_basic(self, value, expected):
        assert scientific(value) == expected

    def test_precision(self):
        assert scientific(1000, 1) == "1.0 x 10³"
        assert scientific(1000, 3) == "1.000 x 10³"
        assert scientific(1000, 0) == "1 x 10³"

    def test_non_numeric(self):
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

    def test_large_exponent(self):
        assert scientific(1e40) == "1.00 x 10⁴⁰"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_no_clamp(self):
        assert clamp(123.456) == "123.456"

    def test_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_ceil_exact_no_token(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"

    def test_ceil_exceeded(self):
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_callable_format(self):
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
            "under 1.0 million"
        )

    def test_custom_tokens(self):
        assert clamp(0.5, floor=1, floor_token="min: ") == "min: 1"
        assert clamp(2, ceil=1, ceil_token="max: ") == "max: 1"

    def test_none_returns_none(self):
        assert clamp(None) is None

    def test_nan(self):
        assert clamp(float("nan")) == "NaN"

    def test_inf(self):
        assert clamp(float("inf")) == "+Inf"
        assert clamp(float("-inf")) == "-Inf"

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            clamp(1, format=123)

    def test_default_format_preserves_value(self):
        assert clamp(1.5) == "1.5"

    def test_int_format(self):
        assert clamp(5, format="{:d}") == "5"


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
class TestMetric:
    @pytest.mark.parametrize(
        "value, unit, expected",
        [
            (1500, "V", "1.50 kV"),
            (2e8, "W", "200 MW"),
            (220e-6, "F", "220 μF"),
            (1e-14, "", "10.00 f"),
            (123_000, "", "123 k"),
            (1_230_000, "", "1.23 M"),
            (1e40, "", "1.00 x 10⁴⁰"),
        ],
    )
    def test_basic(self, value, unit, expected):
        assert metric(value, unit) == expected

    def test_precision(self):
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_zero(self):
        assert metric(0) == "0"

    def test_no_unit(self):
        assert metric(1500) == "1.50 k"

    def test_degree_no_space(self):
        assert metric(1500, "°") == "1.50k°"

    def test_prime_no_space(self):
        assert metric(1500, "′") == "1.50k′"

    def test_double_prime_no_space(self):
        assert metric(1500, "″") == "1.50k″"

    def test_negative(self):
        assert metric(-1500, "V") == "-1.50 kV"

    def test_nan(self):
        assert metric(float("nan")) == "NaN"

    def test_inf(self):
        assert metric(float("inf")) == "+Inf"
        assert metric(float("-inf")) == "-Inf"

    def test_metric_prefixes_small(self):
        assert metric(1e-3) == "1.00 m"
        assert metric(1e-6) == "1.00 μ"
        assert metric(1e-9) == "1.00 n"

    def test_metric_prefixes_large(self):
        assert metric(1e3) == "1.00 k"
        assert metric(1e6) == "1.00 M"
        assert metric(1e9) == "1.00 G"
        assert metric(1e12) == "1.00 T"

    def test_rounding_carry(self):
        # 999.999 -> should absorb the carry into the next prefix
        result = metric(999_999)
        assert result == "1.00 M"

    def test_fallback_to_scientific_tiny(self):
        assert metric(1e-35) == "1.00 x 10⁻³⁵"
