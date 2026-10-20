"""Unit tests for llmground.normalize."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import pytest

from llmground.normalize import (
    fold_unicode,
    iter_dates,
    iter_numbers,
    levenshtein,
    normalize_text,
    numbers_equal,
    parse_date,
    parse_number,
    similarity,
    strip_punctuation,
    token_set_ratio,
)


# ---------------------------------------------------------------------------
# fold_unicode
# ---------------------------------------------------------------------------

class TestFoldUnicode:
    def test_smart_quotes(self):
        assert fold_unicode("\u2018a\u2019") == "'a'"
        assert fold_unicode("\u201cb\u201d") == '"b"'

    def test_dashes(self):
        assert fold_unicode("a\u2013b\u2014c") == "a-b-c"
        assert fold_unicode("\u22122") == "-2"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b") == "a b"
        assert fold_unicode("a\u202fb") == "a b"

    def test_zero_width(self):
        assert fold_unicode("a\u200bb\ufeffc") == "abc"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        # U+FB01 LATIN SMALL LIGATURE FI
        assert fold_unicode("\ufb01n") == "fin"

    def test_nfkc_fullwidth(self):
        # Full-width digits fold to ASCII digits.
        assert fold_unicode("\uff11\uff12") == "12"

    def test_plain_unchanged(self):
        assert fold_unicode("hello world") == "hello world"

    def test_empty(self):
        assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("  a   b\t\nc  ") == "a b c"

    def test_casefold_default(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_casefold_false(self):
        assert normalize_text("Hello WORLD", casefold=False) == "Hello WORLD"

    def test_unicode_folding_applied(self):
        assert normalize_text("\u201cHi\u201d") == '"hi"'

    def test_nbsp_collapsed(self):
        assert normalize_text("a\u00a0\u00a0b") == "a b"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_german_sharp_s_casefold(self):
        assert normalize_text("STRASSE") == "strasse"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_basic(self):
        assert strip_punctuation("Hello, world!") == "Hello world"

    def test_removes_all_punct(self):
        assert strip_punctuation("a.b,c;d:e") == "abcdef"

    def test_keeps_underscore_and_digits(self):
        assert strip_punctuation("a_b 12") == "a_b 12"

    def test_empty(self):
        assert strip_punctuation("") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_integer(self):
        assert parse_number("1234") == Decimal("1234")

    def test_currency(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")

    def test_currency_space(self):
        assert parse_number("$ 4,500") == Decimal("4500")

    def test_european_decimal(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_us_decimal(self):
        assert parse_number("1,234.56") == Decimal("1234.56")

    def test_single_comma_thousands(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_single_comma_decimal(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_negative_sign(self):
        assert parse_number("-42") == Decimal("-42")

    def test_plus_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_accounting_negative_no_symbol(self):
        assert parse_number("(42)") == Decimal("-42")

    def test_currency_code(self):
        assert parse_number("USD 1000") == Decimal("1000")

    def test_currency_code_case_insensitive(self):
        assert parse_number("eur500") == Decimal("500")

    def test_euro_symbol(self):
        assert parse_number("\u20ac99") == Decimal("99")

    def test_none_returns_none(self):
        assert parse_number(None) is None

    def test_empty_returns_none(self):
        assert parse_number("") is None
        assert parse_number("   ") is None

    def test_no_digit_returns_none(self):
        assert parse_number("abc") is None

    def test_only_currency_returns_none(self):
        assert parse_number("$") is None

    def test_decimal_only(self):
        assert parse_number(".5") == Decimal("0.5")

    def test_invalid_fullmatch_returns_none(self):
        # Contains something that fails the full-match guard.
        assert parse_number("1.2.3") is None

    def test_value_equality(self):
        assert parse_number("$4,500.00") == parse_number("4500.0")

    def test_dot_only_returns_none(self):
        assert parse_number(".") is None


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_exact_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.5")) is True

    def test_exact_not_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6"), tolerance=Decimal("0.2")) is True

    def test_tolerance_exact_boundary(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6"), tolerance=Decimal("0.1")) is True

    def test_tolerance_outside(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.7"), tolerance=Decimal("0.1")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_ymd_dash(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)

    def test_iso_ymd_slash(self):
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)

    def test_ymd_dot(self):
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_dmy_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_dmy_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_dmy_ambiguous_returns_none(self):
        assert parse_date("03/04/2026") is None

    def test_dmy_disambiguated_by_big_first(self):
        # 13 is only valid as a day.
        assert parse_date("13/04/2026") == _dt.date(2026, 4, 13)

    def test_dmy_disambiguated_by_big_second(self):
        assert parse_date("04/13/2026") == _dt.date(2026, 4, 13)

    def test_dmonthy(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_ordinal(self):
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_abbrev(self):
        assert parse_date("4 Mar 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_abbrev_dot(self):
        assert parse_date("4 Mar. 2026") == _dt.date(2026, 3, 4)

    def test_monthdy(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)

    def test_monthdy_ordinal(self):
        assert parse_date("March 4th 2026") == _dt.date(2026, 3, 4)

    def test_monthdy_abbrev(self):
        assert parse_date("Mar 4 2026") == _dt.date(2026, 3, 4)

    def test_sept_abbrev(self):
        assert parse_date("Sept 5, 2026") == _dt.date(2026, 9, 5)

    def test_invalid_day_returns_none(self):
        assert parse_date("31 February 2026") is None

    def test_invalid_month_returns_none(self):
        assert parse_date("2026-13-01") is None

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_empty_returns_none(self):
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_no_match_returns_none(self):
        assert parse_date("not a date") is None

    def test_unicode_dash_in_date(self):
        # En-dash folded to ascii dash and parsed.
        assert parse_date("2026\u201303\u201304") == _dt.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("Ahmed Khan", "Ahmed Khan") == 1.0

    def test_reordered(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial_overlap(self):
        # {a, b} vs {a} -> intersection 1, union 2
        assert token_set_ratio("a b", "a") == 0.5

    def test_disjoint(self):
        assert token_set_ratio("red", "blue") == 0.0

    def test_empty_both(self):
        assert token_set_ratio("", "") == 0.0

    def test_empty_one(self):
        assert token_set_ratio("a", "") == 0.0

    def test_punctuation_ignored(self):
        assert token_set_ratio("hello,", "hello") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("Hello", "hello") == 1.0

    def test_duplicates_collapsed(self):
        # Duplicates collapse in set tokens.
        assert token_set_ratio("a a a b", "a b") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_equal(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_left(self):
        assert levenshtein("", "abc") == 3

    def test_empty_right(self):
        assert levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_substitution(self):
        assert levenshtein("kitten", "sitten") == 1

    def test_classic(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_insertion(self):
        assert levenshtein("abc", "abcd") == 1


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("hello", "") == 0.0
        assert similarity("", "hello") == 0.0

    def test_one_char_diff(self):
        # "kitten"(6) vs "sitten"(6): distance 1 -> 5/6
        assert similarity("kitten", "sitten") == pytest.approx(5 / 6)

    def test_classic_similarity(self):
        # "kitten" vs "sitting": distance 3, longest 7 -> 4/7
        assert similarity("kitten", "sitting") == pytest.approx(4 / 7)

    def test_case_and_space_normalized(self):
        assert similarity("Hello", "hello ") == 1.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single_number(self):
        results = list(iter_numbers("Total: $4,500.00 dollars"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "$4,500.00"
        assert val == Decimal("4500.00")
        text = "Total: $4,500.00 dollars"
        assert text[start:end] == "$4,500.00"

    def test_multiple_numbers(self):
        results = list(iter_numbers("a 1 b 2 c 3"))
        values = [r[3] for r in results]
        assert values == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_spans_exclude_leading_space(self):
        text = "value 42"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "42"
        assert text[start:end] == "42"

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_empty(self):
        assert list(iter_numbers("")) == []

    def test_unicode_digits_folded(self):
        # Full-width digit folds to ascii and is found.
        results = list(iter_numbers("\uff14\uff12"))
        assert len(results) == 1
        assert results[0][3] == Decimal("42")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_date(self):
        text = "Due on 2026-03-04 please"
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "2026-03-04"
        assert val == _dt.date(2026, 3, 4)
        assert text[start:end] == "2026-03-04"

    def test_multiple_dates(self):
        text = "from 2026-01-01 to 2026-12-31"
        results = list(iter_dates(text))
        values = [r[3] for r in results]
        assert _dt.date(2026, 1, 1) in values
        assert _dt.date(2026, 12, 31) in values

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_day_first_param(self):
        text = "on 03/04/2026"
        results = list(iter_dates(text, day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_ambiguous_excluded(self):
        # Ambiguous date parsed with day_first=None returns None and is skipped.
        assert list(iter_dates("on 03/04/2026")) == []

    def test_spans_are_actual_text(self):
        text = "date: 4 March 2026."
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert text[start:end] == raw
        assert val == _dt.date(2026, 3, 4)
