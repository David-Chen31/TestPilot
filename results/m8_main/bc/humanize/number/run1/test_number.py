"""Tests for humanize.number."""

from __future__ import annotations

import math

import pytest

from humanize import i18n
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
            (5, "5th"),
            (9, "9th"),
            (10, "10th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (14, "14th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (100, "100th"),
            (101, "101st"),
            (102, "102nd"),
            (103, "103rd"),
            (111, "111th"),
            (112, "112th"),
            (113, "113th"),
            (1002, "1002nd"),
            (0, "0th"),
        ],
    )
    def test_basic(self, value, expected):
        assert ordinal(value) == expected

    def test_string_numeric(self):
        assert ordinal("1") == "1st"
        assert ordinal("113") == "113th"

    def test_float_truncates(self):
        assert ordinal(3.9) == "3rd"

    def test_non_numeric_string(self):
        assert ordinal("something else") == "something else"

    def test_list(self):
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"

    def test_nan(self):
        assert ordinal(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert ordinal(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert ordinal(float("-inf")) == "-Inf"

    def test_gender_female(self):
        assert ordinal(1, gender="female") == "1st"

    def test_gender_unknown_defaults_to_female(self):
        # Anything other than "male" is treated as "female".
        assert ordinal(2, gender="unknown") == "2nd"

    def test_negative(self):
        # For negative values, ``value % 10`` yields the positive remainder in
        # Python (e.g. -1 % 10 == 9), so the suffix is derived from digit 9.
        assert ordinal(-1) == "-1th"
        assert ordinal(-11) == "-11th"


# ---------------------------------------------------------------------------
# _format_not_finite
# ---------------------------------------------------------------------------
class TestFormatNotFinite:
    def test_nan(self):
        assert _format_not_finite(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert _format_not_finite(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert _format_not_finite(float("-inf")) == "-Inf"

    def test_finite_falls_through(self):
        # Finite values hit none of the branches and return the empty string.
        assert _format_not_finite(0.0) == ""
        assert _format_not_finite(42.0) == ""
        assert _format_not_finite(-1.5) == ""


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
class TestIntComma:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (100, "100"),
            ("1000", "1,000"),
            (1_000_000, "1,000,000"),
            (1_234_567.25, "1,234,567.25"),
            (0, "0"),
            (-1000, "-1,000"),
            (1234.5454545, "1,234.5454545"),
        ],
    )
    def test_basic(self, value, expected):
        assert intcomma(value) == expected

    def test_ndigits(self):
        assert intcomma(1234.5454545, 2) == "1,234.55"

    def test_ndigits_one(self):
        assert intcomma(14308.40, 1) == "14,308.4"

    def test_string_ndigits(self):
        assert intcomma("14308.40", 1) == "14,308.4"

    def test_none(self):
        assert intcomma(None) == "None"

    def test_nan(self):
        assert intcomma(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert intcomma(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert intcomma(float("-inf")) == "-Inf"

    def test_string_nan(self):
        assert intcomma("nan") == "NaN"

    def test_non_numeric_string(self):
        assert intcomma("foo") == "foo"

    def test_string_with_existing_commas(self):
        assert intcomma("1,000,000") == "1,000,000"

    def test_string_negative(self):
        assert intcomma("-1000") == "-1,000"

    def test_ndigits_with_string(self):
        assert intcomma("1234.5454545", 2) == "1,234.55"


# ---------------------------------------------------------------------------
# intcomma with a non-default locale (exercises separator translation)
#
# We drive the locale via the internal separator tables that
# ``thousands_separator()`` / ``decimal_separator()`` consult, rather than via
# ``i18n.activate`` (which requires compiled .mo files that may not be present).
# ---------------------------------------------------------------------------
class TestIntCommaLocale:
    @pytest.fixture
    def de_locale(self):
        # Pretend the active locale is German: thousands "." / decimal ",".
        sentinel = "_test_de_DE_"
        old_locale = getattr(i18n._CURRENT, "locale", None)
        i18n._CURRENT.locale = sentinel
        # Register separators for the sentinel locale, saving any prior values.
        old_thousands = i18n._THOUSANDS_SEPARATOR.get(sentinel, "<missing>")
        old_decimal = i18n._DECIMAL_SEPARATOR.get(sentinel, "<missing>")
        i18n._THOUSANDS_SEPARATOR[sentinel] = "."
        i18n._DECIMAL_SEPARATOR[sentinel] = ","
        try:
            yield
        finally:
            if old_thousands == "<missing>":
                i18n._THOUSANDS_SEPARATOR.pop(sentinel, None)
            else:
                i18n._THOUSANDS_SEPARATOR[sentinel] = old_thousands
            if old_decimal == "<missing>":
                i18n._DECIMAL_SEPARATOR.pop(sentinel, None)
            else:
                i18n._DECIMAL_SEPARATOR[sentinel] = old_decimal
            if old_locale is None:
                if hasattr(i18n._CURRENT, "locale"):
                    del i18n._CURRENT.locale
            else:
                i18n._CURRENT.locale = old_locale

    def test_separators_active(self, de_locale):
        # Sanity-check that the fixture really changes what the module sees.
        assert i18n.thousands_separator() == "."
        assert i18n.decimal_separator() == ","

    def test_int_with_locale(self, de_locale):
        # The "," -> thousands_sep and "." -> decimal_sep translation branch.
        assert intcomma(1_000_000) == "1.000.000"

    def test_negative_int_with_locale(self, de_locale):
        assert intcomma(-1000) == "-1.000"

    def test_float_with_locale(self, de_locale):
        assert intcomma(1234.5) == "1.234,5"

    def test_ndigits_with_locale(self, de_locale):
        assert intcomma(1234.5454, 2) == "1.234,55"

    def test_string_with_locale(self, de_locale):
        # A string with the local thousands separator "." should be normalized.
        assert intcomma("1.000.000") == "1.000.000"

    def test_string_decimal_with_locale(self, de_locale):
        # A string with a local decimal separator "," is parsed and re-formatted.
        assert intcomma("1234,5") == "1.234,5"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
class TestIntWord:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("100", "100"),
            ("12400", "12.4 thousand"),
            ("1000000", "1.0 million"),
            (1_200_000_000, "1.2 billion"),
            (8100000000000000000000000000000000, "8.1 decillion"),
            (0, "0"),
            (999, "999"),
            (1000, "1.0 thousand"),
            (-1000, "-1.0 thousand"),
            (-1_200_000, "-1.2 million"),
        ],
    )
    def test_basic(self, value, expected):
        assert intword(value) == expected

    def test_none(self):
        assert intword(None) == "None"

    def test_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_nan(self):
        assert intword(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert intword(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert intword(float("-inf")) == "-Inf"

    def test_non_numeric_string(self):
        assert intword("foo") == "foo"

    def test_googol(self):
        assert intword(10**100) == "1.0 googol"

    def test_rounding_carry_to_next_power(self):
        # 999_999 rounds up to 1000.0 thousand -> 1.0 million.
        assert intword(999_999) == "1.0 million"

    def test_rounding_carry_between_large_powers(self):
        # Rounding must carry to the next power even for very large numbers.
        assert intword(10**24 - 1) == "1.0 septillion"

    def test_trillion(self):
        assert intword(1_000_000_000_000) == "1.0 trillion"

    def test_float_input(self):
        assert intword(1_500_000.0) == "1.5 million"


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
class TestApNumber:
    @pytest.mark.parametrize(
        "value, expected",
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
    def test_basic(self, value, expected):
        assert apnumber(value) == expected

    def test_string_number(self):
        assert apnumber("7") == "seven"

    def test_non_numeric_string(self):
        assert apnumber("foo") == "foo"

    def test_none(self):
        assert apnumber(None) == "None"

    def test_nan(self):
        assert apnumber(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert apnumber(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert apnumber(float("-inf")) == "-Inf"

    def test_float_truncates(self):
        assert apnumber(5.9) == "five"


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
            (0, "0"),
            (2, "2"),
            (0.5, "1/2"),
            (0.25, "1/4"),
            (-1.3, "-1 3/10"),
            (-0.3, "-3/10"),
        ],
    )
    def test_basic(self, value, expected):
        assert fractional(value) == expected

    def test_non_numeric_string(self):
        assert fractional("ten") == "ten"

    def test_none(self):
        assert fractional(None) == "None"

    def test_nan(self):
        assert fractional(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert fractional(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert fractional(float("-inf")) == "-Inf"

    def test_near_integer_rounds_up(self):
        # Fractional part rounds up to 1/1 -> folded into integer.
        assert fractional(2.9999999) == "3"

    def test_integer_float(self):
        assert fractional(1.0) == "1"

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
            (0, "0.00 x 10⁰"),
            (1, "1.00 x 10⁰"),
        ],
    )
    def test_basic(self, value, expected):
        assert scientific(value) == expected

    def test_precision_1(self):
        assert scientific(1000, 1) == "1.0 x 10³"

    def test_precision_3(self):
        assert scientific(1000, 3) == "1.000 x 10³"

    def test_precision_0(self):
        assert scientific(1000, 0) == "1 x 10³"

    def test_non_numeric_string(self):
        assert scientific("foo") == "foo"

    def test_none(self):
        assert scientific(None) == "None"

    def test_nan(self):
        assert scientific(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert scientific(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert scientific(float("-inf")) == "-Inf"

    def test_large_positive_exponent_no_plus(self):
        assert scientific(1e10) == "1.00 x 10¹⁰"

    def test_small_negative_exponent(self):
        assert scientific(1e-10) == "1.00 x 10⁻¹⁰"


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
class TestClamp:
    def test_no_clamping(self):
        assert clamp(123.456) == "123.456"

    def test_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_floor_token_custom(self):
        assert clamp(0.0001, floor=0.01, floor_token="<") == "<0.01"

    def test_ceil_at_boundary(self):
        assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"

    def test_ceil_above(self):
        assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"

    def test_ceil_token_custom(self):
        assert clamp(1.5, ceil=1.0, ceil_token="over ") == "over 1.0"

    def test_none(self):
        assert clamp(None) is None

    def test_nan(self):
        assert clamp(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert clamp(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert clamp(float("-inf")) == "-Inf"

    def test_callable_format(self):
        assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"

    def test_callable_format_no_clamp(self):
        assert clamp(2_000_000, format=intword) == "2.0 million"

    def test_int_format(self):
        assert clamp(123.456, format="{:.0f}") == "123"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            clamp(1.0, format=123)

    def test_floor_and_ceil_prefers_floor(self):
        # Value below floor; floor branch taken first.
        assert clamp(-5, floor=0, ceil=10) == "<0"

    def test_ceil_only(self):
        assert clamp(15, ceil=10) == ">10"


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
        ],
    )
    def test_basic(self, value, unit, expected):
        assert metric(value, unit) == expected

    def test_precision_4(self):
        assert metric(1e-14, precision=4) == "10.00 f"

    def test_no_unit(self):
        assert metric(1500) == "1.50 k"

    def test_zero(self):
        assert metric(0) == "0.00"

    def test_negative(self):
        assert metric(-1500, "V") == "-1.50 kV"

    def test_falls_back_to_scientific_huge(self):
        assert metric(1e40) == "1.00 x 10⁴⁰"

    def test_falls_back_to_scientific_tiny(self):
        assert metric(1e-40) == "1.00 x 10⁻⁴⁰"

    def test_nan(self):
        assert metric(float("nan")) == "NaN"

    def test_pos_inf(self):
        assert metric(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert metric(float("-inf")) == "-Inf"

    def test_degree_no_space(self):
        assert metric(1500, "°") == "1.50k°"

    def test_prime_no_space(self):
        assert metric(1500, "′") == "1.50k′"

    def test_double_prime_no_space(self):
        assert metric(1500, "″") == "1.50k″"

    def test_milli(self):
        assert metric(0.001, "s") == "1.00 ms"

    def test_micro(self):
        assert metric(0.000001, "s") == "1.00 μs"

    def test_kilo_rounding_carry(self):
        # 999.9 k rounds up to 1.00 M
        result = metric(999_900, precision=3)
        assert result == "1.00 M"

    def test_base_unit(self):
        assert metric(1, "s") == "1.00 s"

    def test_deca_range(self):
        assert metric(12, "m") == "12.0 m"
