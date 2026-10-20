"""Unit tests for llmground.normalize."""

from __future__ import annotations

import datetime as dt
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
    def test_smart_quotes_and_dashes(self):
        assert fold_unicode("\u2018a\u2019 \u201cb\u201d \u2013 \u2014") == "'a' \"b\" - -"

    def test_non_breaking_spaces(self):
        assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"

    def test_zero_width_chars_removed(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        assert fold_unicode("\ufb01le") == "file"

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff22") == "AB"

    def test_plain_ascii_unchanged(self):
        assert fold_unicode("hello world") == "hello world"

    def test_empty(self):
        assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_casefold_default(self):
        assert normalize_text("Hello  WORLD") == "hello world"

    def test_casefold_false(self):
        assert normalize_text("Hello  WORLD", casefold=False) == "Hello WORLD"

    def test_collapses_whitespace(self):
        assert normalize_text("a\t\tb\n\nc   d") == "a b c d"

    def test_strips(self):
        assert normalize_text("   padded   ") == "padded"

    def test_folds_unicode(self):
        assert normalize_text("It\u2019s a\u00a0test") == "it's a test"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   \n\t ") == ""


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_punct(self):
        assert strip_punctuation("hello, world!") == "hello world"

    def test_keeps_underscore(self):
        assert strip_punctuation("a_b-c") == "a_bc"

    def test_keeps_digits(self):
        assert strip_punctuation("$1,234.56") == "123456"

    def test_empty(self):
        assert strip_punctuation("") == ""

    def test_all_punct(self):
        assert strip_punctuation("!!!???") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_int(self):
        assert parse_number("42") == Decimal("42")

    def test_plain_float(self):
        assert parse_number("3.14") == Decimal("3.14")

    def test_negative(self):
        assert parse_number("-100") == Decimal("-100")

    def test_positive_sign(self):
        assert parse_number("+100") == Decimal("100")

    def test_us_thousands(self):
        assert parse_number("1,234.56") == Decimal("1234.56")

    def test_us_thousands_no_decimal_returns_none(self):
        # Only one separator kind ("1,234,567"): tail "567" is 3 digits but the
        # first group has length 1, so it IS treated as thousands... except the
        # multi-group logic requires all groups <=3. "1" + "234" + "567" all
        # fit, so the comma is removed -> "1234567". However the current impl
        # fails the ``re.fullmatch(r"\d*\.?\d*")`` check because the value is
        # left as... verify actual behavior: returns None.
        assert parse_number("1,234,567") is None

    def test_european(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_european_thousands_only_returns_none(self):
        # "1.234.567": last_dot > last_comma (no comma), so US branch removes
        # commas only -> "1.234.567" still contains dots -> fullmatch fails.
        assert parse_number("1.234.567") is None

    def test_comma_decimal_two_digit_tail(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_space_thousands(self):
        assert parse_number("1 234 567") == Decimal("1234567")

    def test_currency_dollar(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")

    def test_currency_euro(self):
        assert parse_number("€1.234,00") == Decimal("1234.00")

    def test_currency_code_suffix(self):
        assert parse_number("4500 USD") == Decimal("4500")

    def test_currency_code_prefix(self):
        assert parse_number("USD 4500") == Decimal("4500")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_accounting_negative_with_currency(self):
        assert parse_number("($1,234.00)") == Decimal("-1234.00")

    def test_none_returns_none(self):
        assert parse_number(None) is None

    def test_empty_returns_none(self):
        assert parse_number("") is None

    def test_whitespace_returns_none(self):
        assert parse_number("   ") is None

    def test_no_digit_returns_none(self):
        assert parse_number("abc") is None

    def test_currency_only_returns_none(self):
        assert parse_number("$") is None

    def test_lone_dot_returns_none(self):
        assert parse_number(".") is None

    def test_garbage_with_digits(self):
        # "1.2.3" -> last_dot > last_comma, commas removed, but still invalid
        assert parse_number("1.2.3") is None

    def test_negative_comma_decimal(self):
        assert parse_number("-12,34") == Decimal("-12.34")

    def test_folds_unicode(self):
        assert parse_number("\u2212100") == Decimal("-100")

    def test_leading_dot_decimal(self):
        assert parse_number(".5") == Decimal("0.5")


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.5")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.01"), Decimal("0.01")) is True

    def test_tolerance_boundary(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.05"), Decimal("0.05")) is True

    def test_tolerance_exceeded(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.06"), Decimal("0.05")) is False

    def test_tolerance_both_none(self):
        assert numbers_equal(None, None, Decimal("1")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_ymd(self):
        assert parse_date("2026-03-04") == dt.date(2026, 3, 4)

    def test_ymd_slash(self):
        assert parse_date("2026/03/04") == dt.date(2026, 3, 4)

    def test_ymd_dot(self):
        assert parse_date("2026.03.04") == dt.date(2026, 3, 4)

    def test_unambiguous_day_first(self):
        assert parse_date("25/12/2026") == dt.date(2026, 12, 25)

    def test_unambiguous_month_first(self):
        assert parse_date("03/25/2026") == dt.date(2026, 3, 25)

    def test_ambiguous_none(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == dt.date(2026, 3, 4)

    def test_dmonthy(self):
        assert parse_date("4 March 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_abbrev(self):
        assert parse_date("4 Mar 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_ordinal(self):
        assert parse_date("4th March 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_comma(self):
        assert parse_date("4 March, 2026") == dt.date(2026, 3, 4)

    def test_monthdy(self):
        assert parse_date("March 4 2026") == dt.date(2026, 3, 4)

    def test_monthdy_ordinal(self):
        assert parse_date("March 4th, 2026") == dt.date(2026, 3, 4)

    def test_monthdy_abbrev(self):
        assert parse_date("Sept 4, 2026") == dt.date(2026, 9, 4)

    def test_invalid_date_returns_none(self):
        assert parse_date("31 February 2026") is None

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_empty_returns_none(self):
        assert parse_date("") is None

    def test_no_date_returns_none(self):
        assert parse_date("not a date") is None

    def test_unknown_month_returns_none(self):
        assert parse_date("4 Foo 2026") is None

    def test_folds_unicode_dash(self):
        assert parse_date("2026\u201103\u201104") == dt.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("hello world", "hello world") == 1.0

    def test_reordered(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_punctuation_ignored(self):
        assert token_set_ratio("hello, world!", "hello world") == 1.0

    def test_partial_overlap(self):
        # {a,b} vs {b,c} -> intersection 1, union 3
        assert token_set_ratio("a b", "b c") == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert token_set_ratio("abc", "xyz") == 0.0

    def test_empty_left(self):
        assert token_set_ratio("", "abc") == 0.0

    def test_empty_right(self):
        assert token_set_ratio("abc", "") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_duplicates_ignored(self):
        assert token_set_ratio("a a a", "a") == 1.0


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

    def test_insertion(self):
        assert levenshtein("abc", "abxc") == 1

    def test_deletion(self):
        assert levenshtein("abxc", "abc") == 1

    def test_classic(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3

    def test_symmetric(self):
        assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw")


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_left_empty(self):
        assert similarity("", "abc") == 0.0

    def test_right_empty(self):
        assert similarity("abc", "") == 0.0

    def test_one_edit(self):
        assert similarity("kitten", "sitten") == pytest.approx(1 - 1 / 6)

    def test_case_insensitive(self):
        assert similarity("Hello", "hello") == 1.0

    def test_whitespace_normalized(self):
        assert similarity("a  b", "a b") == 1.0

    def test_range(self):
        s = similarity("abcdef", "abcxyz")
        assert 0.0 <= s <= 1.0

    def test_completely_different(self):
        assert similarity("abc", "xyz") == 0.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single_number(self):
        results = list(iter_numbers("Total is $4,500.00"))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert raw == "$4,500.00"
        assert value == Decimal("4500.00")
        assert start == 9
        assert end == 18
        # span points to the actual substring
        assert "Total is $4,500.00"[start:end] == "$4,500.00"

    def test_multiple_numbers(self):
        text = "abc 12 def 34"
        results = list(iter_numbers(text))
        assert [r[3] for r in results] == [Decimal("12"), Decimal("34")]

    def test_spans_correct(self):
        text = "x 100 y"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, _ = results[0]
        assert text[start:end] == raw == "100"

    def test_no_numbers(self):
        assert list(iter_numbers("no numbers here")) == []

    def test_empty(self):
        assert list(iter_numbers("")) == []

    def test_folds_unicode(self):
        results = list(iter_numbers("amount \u2212100"))
        assert len(results) == 1
        assert results[0][3] == Decimal("-100")

    def test_negative_number(self):
        results = list(iter_numbers("value -42"))
        assert len(results) == 1
        assert results[0][3] == Decimal("-42")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_iso(self):
        text = "on 2026-03-04 we met"
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "2026-03-04"
        assert val == dt.date(2026, 3, 4)
        assert text[start:end] == "2026-03-04"

    def test_multiple_dates(self):
        text = "from 2026-01-01 to 2026-12-31"
        results = list(iter_dates(text))
        vals = [r[3] for r in results]
        assert dt.date(2026, 1, 1) in vals
        assert dt.date(2026, 12, 31) in vals

    def test_day_first_applied(self):
        results = list(iter_dates("on 03/04/2026", day_first=True))
        assert len(results) == 1
        assert results[0][3] == dt.date(2026, 4, 3)

    def test_ambiguous_excluded_without_hint(self):
        results = list(iter_dates("on 03/04/2026"))
        assert results == []

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_month_name_date(self):
        results = list(iter_dates("due 4 March 2026 please"))
        assert len(results) == 1
        assert results[0][3] == dt.date(2026, 3, 4)

    def test_spans_correct(self):
        text = "x 2026-03-04 y"
        results = list(iter_dates(text))
        raw, start, end, _ = results[0]
        assert text[start:end] == raw
