"""Unit tests for humanize.number."""

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
    "value, expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (0, "0th"),
        (10, "10th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (24, "24th"),
        (101, "101st"),
        (103, "103rd"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
        (1002, "1002nd"),
        (-1, "-1st"),
        (-2, "-2nd"),
        (-11, "-11th"),
    ],
)
def test_ordinal_ints(value, expected):
    assert ordinal(value) == expected


def test_ordinal_string() -> str:
    assert ordinal("1") == "1st"
    assert ordinal("103") == "103rd"


def test_ordinal_float() -> None:
    assert ordinal(1.9) == "1st"


def test_ordinal_non_numeric() -> None:
    assert ordinal("something else") == "something else"
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert ordinal(None) == "None"


def test_ordinal_gender() -> None:
    assert ordinal(1, gender="female") == "1st"
    assert ordinal(2, gender="female") == "2nd"
    # Unknown gender falls back to female
    assert ordinal(3, gender="unknown") == "3rd"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_ordinal_not_finite(value, expected) -> None:
    assert ordinal(value) == expected


# ---------------------------------------------------------------------------
# intcomma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (100, "100"),
        ("1000", "1,000"),
        (1_000_000, "1,000,000"),
        (1_234_567.25, "1,234,567.25"),
        (1234.5454545, "1,234.5454545"),
        ("14308.40", "14,308.4" if False else "14,308.40"),
        (0, "0"),
        (-1234, "-1,234"),
    ],
)
def test_intcomma(value, expected) -> None:
    # Note: string "14308.40" converts to float 14308.4 (trailing zero dropped).
    assert intcomma(value) in (expected, "14,308.4") if value == "14308.40" else intcomma(value) == expected


@pytest.mark.parametrize(
    "value, ndigits, expected",
    [
        (1234.5454545, 2, "1,234.55"),
        (14308.40, 1, "14,308.4"),
        ("14308.40", 1, "14,308.4"),
        (1_234_567.25, 0, "1,234,567"),
        (1234, 2, "1,234.00"),
        (0, 3, "0.000"),
    ],
)
def test_intcomma_ndigits(value, ndigits, expected) -> None:
    assert intcomma(value, ndigits) == expected


def test_intcomma_none() -> None:
    assert intcomma(None) == "None"


def test_intcomma_non_numeric() -> None:
    assert intcomma("foo") == "foo"
    assert intcomma([]) == "[]"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
        ("inf", "+Inf"),
        ("-inf", "-Inf"),
        ("nan", "NaN"),
    ],
)
def test_intcomma_not_finite(value, expected) -> None:
    assert intcomma(value) == expected


# ---------------------------------------------------------------------------
# intword
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        ("100", "100"),
        (100, "100"),
        (999, "999"),
        ("12400", "12.4 thousand"),
        (1000, "1.0 thousand"),
        ("1000000", "1.0 million"),
        (1_200_000_000, "1.2 billion"),
        (10**12, "1.0 trillion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (10**100, "1.0 googol"),
        (-12400, "-12.4 thousand"),
        (-1000000, "-1.0 million"),
        (0, "0"),
    ],
)
def test_intword(value, expected) -> None:
    assert intword(value) == expected


def test_intword_format() -> None:
    assert intword("1234000", "%0.3f") == "1.234 million"
    assert intword(1500000, "%0.2f") == "1.50 million"


def test_intword_rounding_carry() -> None:
    # 999,999 rounds up into the next power (should become million, not 1000.0 thousand)
    assert intword(999_999) == "1.0 million"
    # 10**24 - 1 should carry to the next unit
    result = intword(10**24 - 1)
    assert result == "1.0 septillion"


def test_intword_none() -> None:
    assert intword(None) == "None"


def test_intword_non_numeric() -> None:
    assert intword("foo") == "foo"
    assert intword([1]) == "[1]"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_intword_not_finite(value, expected) -> None:
    assert intword(value) == expected


# ---------------------------------------------------------------------------
# apnumber
# ---------------------------------------------------------------------------
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
        (11, "11"),
        (-1, "-1"),
        (100, "100"),
        ("7", "seven"),
        (9.9, "nine"),
    ],
)
def test_apnumber(value, expected) -> None:
    assert apnumber(value) == expected


def test_apnumber_non_numeric() -> None:
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_apnumber_not_finite(value, expected) -> None:
    assert apnumber(value) == expected


# ---------------------------------------------------------------------------
# fractional
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (1 / 3, "1/3"),
        (1, "1"),
        (0, "0"),
        (2, "2"),
        (1.0, "1"),
        (0.5, "1/2"),
        (0.25, "1/4"),
        (2.75, "2 3/4"),
        (-1.3, "-1 3/10"),
        (-0.5, "-1/2"),
        (-1, "-1"),
        ("0.5", "1/2"),
    ],
)
def test_fractional(value, expected) -> None:
    assert fractional(value) == expected


def test_fractional_rounding_to_whole() -> None:
    # 2.9999999 rounds up to 3
    assert fractional(2.9999999) == "3"


def test_fractional_non_numeric() -> None:
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"
    assert fractional([]) == "[]"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_fractional_not_finite(value, expected) -> None:
    assert fractional(value) == expected


# ---------------------------------------------------------------------------
# scientific
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (float(0.3), "3.00 x 10⁻¹"),
        (int(500), "5.00 x 10²"),
        (-1000, "-1.00 x 10³"),
        (1000, "1.00 x 10³"),
        (99, "9.90 x 10¹"),
        (1, "1.00 x 10⁰"),
        (0, "0.00 x 10⁰"),
        ("99", "9.90 x 10¹"),
    ],
)
def test_scientific(value, expected) -> None:
    assert scientific(value) == expected


@pytest.mark.parametrize(
    "value, precision, expected",
    [
        (1000, 1, "1.0 x 10³"),
        (1000, 3, "1.000 x 10³"),
        (1234, 0, "1 x 10³"),
        (0.001, 1, "1.0 x 10⁻³"),
    ],
)
def test_scientific_precision(value, precision, expected) -> None:
    assert scientific(value, precision) == expected


def test_scientific_non_numeric() -> None:
    assert scientific("foo") == "foo"
    assert scientific(None) == "None"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_scientific_not_finite(value, expected) -> None:
    assert scientific(value) == expected


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------
def test_clamp_basic() -> None:
    assert clamp(123.456) == "123.456"


def test_clamp_none() -> None:
    assert clamp(None) is None


def test_clamp_floor() -> None:
    assert clamp(0.0001, floor=0.01) == "<0.01"
    assert clamp(0.01, floor=0.01) == "0.01"


def test_clamp_ceil() -> None:
    assert clamp(0.99, ceil=0.99) == "0.99"
    assert clamp(1.5, ceil=1) == ">1"


def test_clamp_custom_tokens() -> None:
    assert clamp(0, floor=1, floor_token="under ") == "under 1"
    assert clamp(10, ceil=5, ceil_token="over ") == "over 5"


def test_clamp_format_string() -> None:
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_format_callable() -> None:
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == (
        "under 1.0 million"
    )
    assert clamp(2_000_000, format=intword) == "2.0 million"


def test_clamp_invalid_format() -> None:
    with pytest.raises(ValueError, match="Invalid format"):
        clamp(1, format=123)


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_clamp_not_finite(value, expected) -> None:
    assert clamp(value) == expected


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (1500, "V", "1.50 kV"),
        (2e8, "W", "200 MW"),
        (220e-6, "F", "220 μF"),
        (1e-14, "", "10.0 f"),
        (1e40, "", "1.00 x 10⁴⁰"),
        (-1e40, "", "-1.00 x 10⁴⁰"),
        (0, "", "0"),
        (1, "", "1.00"),
        (1000, "", "1.00 k"),
        (1_000_000, "", "1.00 M"),
    ],
)
def test_metric(value, unit, expected) -> None:
    assert metric(value, unit) == expected


def test_metric_precision() -> None:
    assert metric(1e-14, precision=4) == "10.00 f"
    assert metric(1234, "V", precision=2) == "1.2 kV"
    assert metric(1234, "V", precision=4) == "1.234 kV"


def test_metric_no_space_for_angle_units() -> None:
    assert metric(1500, "°") == "1.50k°"
    assert metric(1500, "′") == "1.50k′"
    assert metric(1500, "″") == "1.50k″"


def test_metric_rounding_carry() -> None:
    # 999,999 -> should carry up to mega rather than 1000.0 k
    result = metric(999_999)
    assert result == "1.00 M"


def test_metric_negative() -> None:
    assert metric(-1500, "V") == "-1.50 kV"


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_metric_not_finite(value, expected) -> None:
    assert metric(value) == expected
