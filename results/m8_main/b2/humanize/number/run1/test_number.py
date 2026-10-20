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
        (102, "102nd"),
        (103, "103rd"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
        (1002, "1002nd"),
        (0, "0th"),
        # For negatives, value % 10 in Python is 9 (e.g. -1 % 10 == 9), which
        # yields "th".
        (-1, "-1th"),
        (-21, "-21th"),
        (-22, "-22th"),
        (-23, "-23th"),
    ],
)
def test_ordinal(value, expected):
    assert ordinal(value) == expected


def test_ordinal_string_numbers():
    assert ordinal("1") == "1st"
    assert ordinal("22") == "22nd"
    # int("3.0") raises ValueError, so the original string is returned.
    assert ordinal("3.0") == "3.0"
    assert ordinal(3.9) == "3rd"


def test_ordinal_non_numeric_returns_str():
    assert ordinal("something else") == "something else"
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert ordinal(None) == "None"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_ordinal_not_finite(value, expected):
    assert ordinal(value) == expected


def test_ordinal_gender_defaults_female_for_unknown():
    # Unknown gender values fall back to "female"; with the default (untranslated)
    # catalogs both genders render the same suffix.
    assert ordinal(1, gender="female") == "1st"
    assert ordinal(1, gender="male") == "1st"
    assert ordinal(2, gender="weird") == "2nd"


def test_ordinal_default_gender_is_male():
    # The default must be exactly "male"; any other default would be coerced
    # to "female". With untranslated catalogs this is indistinguishable, so we
    # verify the exact suffix mapping for each gender category directly.
    from humanize.number import _ORDINAL_SUFFIXES

    assert len(_ORDINAL_SUFFIXES["male"]) == 10
    assert len(_ORDINAL_SUFFIXES["female"]) == 10
    for i in range(10):
        assert _ORDINAL_SUFFIXES["male"][i][0] == f"{i} (male)"
        assert _ORDINAL_SUFFIXES["female"][i][0] == f"{i} (female)"


def test_ordinal_gender_selection_logic():
    # Verify gender normalization: only the literal "male" maps to the male
    # context; everything else (including "MALE", "Male", "") maps to female.
    # We do this by patching the pgettext used at call time via module reference.
    import humanize.number as num_mod

    calls = []

    def fake_pgettext(msgctxt, message):
        calls.append((msgctxt, message))
        return message

    original = num_mod.P_
    num_mod.P_ = fake_pgettext
    try:
        ordinal(1, gender="male")
        ordinal(1, gender="MALE")
        ordinal(1, gender="female")
        ordinal(1, gender="")
        ordinal(1)
    finally:
        num_mod.P_ = original

    contexts = [c[0] for c in calls]
    assert contexts == [
        "1 (male)",
        "1 (female)",
        "1 (female)",
        "1 (female)",
        "1 (male)",
    ]


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100, "100"),
        ("1000", "1,000"),
        (1_000_000, "1,000,000"),
        (1_234_567.25, "1,234,567.25"),
        (1234, "1,234"),
        (-1234567, "-1,234,567"),
        (0, "0"),
        ("-1000", "-1,000"),
    ],
)
def test_intcomma(value, expected):
    assert intcomma(value) == expected


def test_intcomma_ndigits():
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"
    assert intcomma("14308.40", 1) == "14,308.4"
    assert intcomma(1000, 2) == "1,000.00"


def test_intcomma_string_with_commas():
    assert intcomma("1,000,000") == "1,000,000"
    assert intcomma("1,234.5", 1) == "1,234.5"


def test_intcomma_none():
    assert intcomma(None) == "None"


def test_intcomma_invalid_string():
    assert intcomma("foo") == "foo"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
        ("inf", "+Inf"),
        ("-inf", "-Inf"),
        ("nan", "NaN"),
    ],
)
def test_intcomma_not_finite(value, expected):
    assert intcomma(value) == expected


def test_intcomma_float_no_ndigits():
    assert intcomma(1234.0) == "1,234.0"


def test_intcomma_default_separators_untouched():
    # With default separators ("," and "."), the result must pass through
    # the translation branch untouched.
    assert intcomma(1234567.25) == "1,234,567.25"
    assert intcomma(1000) == "1,000"


def test_intcomma_locale_separators(monkeypatch):
    # Force a locale with thousands "." and decimal "," to exercise the
    # translate branch and verify both sides of the "or" condition.
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "thousands_separator", lambda: ".")
    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ",")
    assert intcomma(1234567.5) == "1.234.567,5"
    assert intcomma("1.234.567,5") == "1.234.567,5"


def test_intcomma_only_thousands_changed(monkeypatch):
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "thousands_separator", lambda: " ")
    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ".")
    assert intcomma(1234567.25) == "1 234 567.25"


def test_intcomma_only_decimal_changed(monkeypatch):
    # With thousands_sep == decimal_sep == ",", a string like "1,234,5" has
    # every comma stripped (thousands) and then no decimal comma remains, so
    # it becomes int 12345 and formats as "12,345".
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "thousands_separator", lambda: ",")
    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ",")
    assert intcomma("1,234,5") == "12,345"


def test_intcomma_only_decimal_changed_real_decimal(monkeypatch):
    # Only the decimal separator differs from the default; the result must
    # still have its "." translated to the locale decimal separator.
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "thousands_separator", lambda: ",")
    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ",")
    assert intcomma(1234.5) == "1,234,5"


def test_intcomma_only_thousands_changed_int(monkeypatch):
    # Only thousands differs from the default; the result must be translated.
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "thousands_separator", lambda: "_")
    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ".")
    assert intcomma(1234567) == "1_234_567"


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("100", "100"),
        (100, "100"),
        (-100, "-100"),
        ("12400", "12.4 thousand"),
        ("1000000", "1.0 million"),
        (1_200_000, "1.2 million"),
        (1_200_000_000, "1.2 billion"),
        (1_000_000_000_000, "1.0 trillion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (10**3, "1.0 thousand"),
        (-12400, "-12.4 thousand"),
        (999, "999"),
        (1000, "1.0 thousand"),
    ],
)
def test_intword(value, expected):
    assert intword(value) == expected


def test_intword_zero_no_negative_prefix():
    # 0 is not negative; ensures "< 0" rather than "<= 0" / "< 1".
    assert intword(0) == "0"


def test_intword_one_no_negative_prefix():
    assert intword(1) == "1"


def test_intword_negative_below_thousand():
    assert intword(-1) == "-1"
    assert intword(-999) == "-999"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"
    assert intword(1_500_000, "%.0f") == "2 million"


def test_intword_default_format_string():
    # Default format is "%.1f"; check its exact behaviour.
    assert intword(1_000_000) == "1.0 million"
    assert intword(1_234_567) == "1.2 million"
    assert intword(1_250_000) == "1.2 million"  # rounds down (banker's? no: 1.25 -> 1.2)
    assert intword(1_350_000) == "1.4 million"


def test_intword_none():
    assert intword(None) == "None"


def test_intword_invalid():
    assert intword("foo") == "foo"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_intword_not_finite(value, expected):
    assert intword(value) == expected


def test_intword_googol():
    assert intword(10**100) == "1.0 googol"


def test_intword_rounding_carry_to_next_power():
    # 999_999 rounds up within "thousand" bucket to the next power.
    assert intword(999_999) == "1.0 million"
    assert intword(999_999_999) == "1.0 billion"


def test_intword_large_carry_no_float_loss():
    # Values above ~10**22 previously failed to carry due to float precision.
    assert intword(10**24 - 1) == "1.0 septillion"


def test_intword_singular_plural_ngettext_argument():
    # ngettext is called with math.ceil(rounded_value); with untranslated
    # catalogs both branches yield the same word, but we verify the arg
    # passed by intercepting.
    import humanize.number as num_mod

    captured = []
    original = num_mod._ngettext

    def fake_ngettext(singular, plural, num):
        captured.append(num)
        return original(singular, plural, num)

    num_mod._ngettext = fake_ngettext
    try:
        intword(1_000_000)  # 1.0 million -> ceil 1
        intword(2_500_000)  # 2.5 million -> ceil 3
        intword(1_100_000)  # 1.1 million -> ceil 2
    finally:
        num_mod._ngettext = original

    assert captured == [1, 3, 2]


def test_intword_decimal_separator(monkeypatch):
    import humanize.number as num_mod

    monkeypatch.setattr(num_mod, "decimal_separator", lambda: ",")
    assert intword(1_200_000) == "1,2 million"
    assert intword(-1_200_000) == "-1,2 million"


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
        (100, "100"),
        (-1, "-1"),
        ("7", "seven"),
        (9.9, "nine"),
        ("0", "zero"),
    ],
)
def test_apnumber(value, expected):
    assert apnumber(value) == expected


def test_apnumber_invalid():
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_apnumber_not_finite(value, expected):
    assert apnumber(value) == expected


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
        (2, "2"),
        (0, "0"),
        (0.5, "1/2"),
        (0.25, "1/4"),
        (-0.3, "-3/10"),
        (-1.3, "-1 3/10"),
        (1.0000, "1"),
        (2.9999999, "3"),
        (1.5, "1 1/2"),
        ("0.3", "3/10"),
    ],
)
def test_fractional(value, expected):
    assert fractional(value) == expected


def test_fractional_limit_denominator_1000():
    # Fraction 1/1001 has denominator 1001 > 1000, so limit_denominator(1000)
    # must choose a fraction with a denominator <= 1000, not 1001.
    # 1/1001 ~= 0.000999000999; the best approximation with denominator <= 1000
    # is 1/1000? Let's check: limit_denominator(1000) of Fraction(1,1001).
    # Actually 1/1001 is already in lowest terms with denom 1001, so it reduces.
    # Use a value whose limit_denominator(1000) vs (1001) differ.
    from fractions import Fraction

    # 1/999 with limit_denominator(1000) -> 1/999 ; with (1001) -> 1/999 too.
    # Use e.g. 1/1500: limit_denominator(1000) -> 1/1499 ; limit_denominator(1001) -> 1/1499.
    # Need a case where denominators differ by exactly one boundary.
    # Fraction(1, 997) has denom 997 which is <= both.
    # Pick x such that best lower-denominator approx is 1/1001 vs 1/1000.
    # 1/1001 itself: limit_denominator(1000) of 1/1001.
    f = Fraction(1, 1001).limit_denominator(1000)
    assert f.denominator <= 1000
    # Verify fractional() honours limit 1000: use 1000/1001 ~= 0.999001.
    # limit_denominator(1000) of 1000/1001 = 999/1000? or 1000/1001? denom 1001>1000.
    val = 1000 / 1001
    result = fractional(val)
    # Must not be "1000/1001" (denominator 1001 disallowed).
    assert "1001" not in result


def test_fractional_boundary_denominator():
    # Direct: choose input whose best fraction has denominator 1001 exactly.
    # Fraction(1, 1001) -> limit_denominator(1000) gives denominator != 1001.
    from fractions import Fraction

    approx = Fraction(1, 1001).limit_denominator(1000)
    assert approx.denominator <= 1000


def test_fractional_invalid():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_fractional_not_finite(value, expected):
    assert fractional(value) == expected


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.3, "3.00 x 10⁻¹"),
        (500, "5.00 x 10²"),
        (-1000, "-1.00 x 10³"),
        (1000, "1.00 x 10³"),
        ("99", "9.90 x 10¹"),
        (1, "1.00 x 10⁰"),
        (0.0, "0.00 x 10⁰"),
    ],
)
def test_scientific(value, expected):
    assert scientific(value) == expected


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        (1000, 1, "1.0 x 10³"),
        (1000, 3, "1.000 x 10³"),
        (1000, 0, "1 x 10³"),
    ],
)
def test_scientific_precision(value, precision, expected):
    assert scientific(value, precision) == expected


def test_scientific_invalid():
    assert scientific("foo") == "foo"
    assert scientific(None) == "None"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_scientific_not_finite(value, expected):
    assert scientific(value) == expected


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_no_limits():
    assert clamp(123.456) == "123.456"
    assert clamp(0) == "0"


def test_clamp_floor():
    assert clamp(0.0001, floor=0.01) == "<0.01"
    assert clamp(0.02, floor=0.01) == "0.02"


def test_clamp_ceil():
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"


def test_clamp_custom_tokens():
    assert clamp(0, floor=10, floor_token="under ") == "under 10"
    assert clamp(100, ceil=10, ceil_token="over ") == "over 10"


def test_clamp_callable_format():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
        "under 1.0 million"
    )
    assert clamp(1_000_000, format=intword) == "1.0 million"


def test_clamp_none():
    assert clamp(None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_clamp_not_finite(value, expected):
    assert clamp(value) == expected


def test_clamp_invalid_format_message():
    with pytest.raises(ValueError) as exc_info:
        clamp(1, format=123)
    assert str(exc_info.value) == (
        "Invalid format. Must be either a valid formatting string, or a function "
        "that accepts value and returns a string."
    )


def test_clamp_invalid_format_callable_check():
    # A non-string, non-callable object triggers the error; None format too.
    with pytest.raises(ValueError):
        clamp(1, format=None)
    with pytest.raises(ValueError):
        clamp(1, format=[1, 2])


def test_clamp_floor_and_ceil_boundaries():
    # Exactly at floor/ceil is not clamped.
    assert clamp(1, floor=1, ceil=10) == "1"
    assert clamp(10, floor=1, ceil=10) == "10"


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1500, "V", "1.50 kV"),
        (2e8, "W", "200 MW"),
        (220e-6, "F", "220 μF"),
        (1000, "", "1.00 k"),
        # No unit and no ordinal (exponent 0) -> no trailing space.
        (1, "", "1.00"),
        (0, "", "0.00"),
        (123000, "", "123 k"),
        (1_230_000, "", "1.23 M"),
        # Degree/prime symbols don't get a separating space.
        (1500, "°", "1.50k°"),
        (1500, "′", "1.50k′"),
        (1500, "″", "1.50k″"),
    ],
)
def test_metric(value, unit, expected):
    assert metric(value, unit) == expected


def test_metric_precision():
    assert metric(1e-14, precision=4) == "10.00 f"
    assert metric(1500, "V", 2) == "1.5 kV"
    assert metric(1500, "V", 4) == "1.500 kV"


def test_metric_scientific_fallback_large():
    assert metric(1e40) == "1.00 x 10⁴⁰"


def test_metric_scientific_fallback_small():
    assert metric(1e-40) == "1.00 x 10⁻⁴⁰"


def test_metric_negative():
    assert metric(-1500, "V") == "-1.50 kV"


def test_metric_rounding_carry():
    # 999.9 with k bucket rounds to 1.00 M.
    assert metric(999_900, "V") == "1.00 MV"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_metric_not_finite(value, expected):
    assert metric(value) == expected


def test_metric_unit_only_no_prefix():
    # exponent 0, unit present -> space and no ordinal
    assert metric(1, "s") == "1.00 s"
    assert metric(0, "s") == "0.00 s"


def test_metric_precision_exact_digits():
    # Check trailing-zero trimming behavior by varying precision.
    assert metric(1500, "V", 1) == "2 kV"
    assert metric(1500, "V", 5) == "1.5000 kV"
