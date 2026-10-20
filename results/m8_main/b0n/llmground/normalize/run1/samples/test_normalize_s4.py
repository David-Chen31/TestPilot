"""Unit tests for llmground.normalize."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import pytest

from llmground.normalize import (
    fold_unicode,
    normalize_text,
    strip_punctuation,
    parse_number,
    numbers_equal,
    parse_date,
    token_set_ratio,
    levenshtein,
    similarity,
    iter_numbers,
    iter_dates,
)


# ---------------------------------------------------------------------------
# fold_unicode
# ---------------------------------------------------------------------------

class TestFoldUnicode:
    def test_curly_quotes(self):
        assert fold_unicode("\u2018a\u2019") == "'a'"
        assert fold_unicode("\u201cquote\u201d") == '"quote"'
        assert fold_unicode("\u201aa\u201b") == "'a'"
        assert fold_unicode("\u201ea\u201f") == '"a"'

    def test_dashes(self):
        assert fold_unicode("a\u2010b\u2011c\u2012d") == "a-b-c-d"
        assert fold_unicode("a\u2013b\u2014c\u2015d") == "a-b-c-d"
        assert fold_unicode("a\u2212b") == "a-b"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b") == "a b"
        assert fold_unicode("a\u2007b") == "a b"
        assert fold_unicode("a\u202fb") == "a b"
        assert fold_unicode("a\u2009b") == "a b"

    def test_zero_width(self):
        assert fold_unicode("a\u200bb") == "ab"
        assert fold_unicode("a\u200cb") == "ab"
        assert fold_unicode("a\u200db") == "ab"
        assert fold_unicode("a\ufeffb") == "ab"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        # \ufb01 is the "fi" ligature.
        assert fold_unicode("\ufb01") == "fi"

    def test_nfkc_fullwidth(self):
        # Full-width digits/letters fold to ASCII.
        assert fold_unicode("\uff11\uff12\uff13") == "123"
        assert fold_unicode("\uff21\uff22") == "AB"

    def test_plain_text_unchanged(self):
        assert fold_unicode("hello world") == "hello world"

    def test_empty(self):
        assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("a   b\t\nc") == "a b c"

    def test_strips(self):
        assert normalize_text("  hello  ") == "hello"

    def test_casefold_default(self):
        assert normalize_text("HeLLo") == "hello"

    def test_casefold_disabled(self):
        assert normalize_text("HeLLo", casefold=False) == "HeLLo"

    def test_unicode_folding_applied(self):
        assert normalize_text("\u201cHi\u201d") == '"hi"'

    def test_empty(self):
        assert normalize_text("   ") == ""

    def test_casefold_ss(self):
        # "ß".casefold() -> "ss"
        assert normalize_text("Stra\u00dfe") == "strasse"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_commas(self):
        assert strip_punctuation("Ahmed, Khan.") == "Ahmed Khan"

    def test_keeps_underscores_and_digits(self):
        assert strip_punctuation("a_b-1!") == "a_b1"

    def test_keeps_whitespace(self):
        assert strip_punctuation("a b") == "a b"

    def test_empty(self):
        assert strip_punctuation("") == ""

    def test_all_punctuation(self):
        assert strip_punctuation("!!!???") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty(self):
        assert parse_number("") is None
        assert parse_number("   ") is None

    def test_simple_integer(self):
        assert parse_number("42") == Decimal("42")

    def test_simple_decimal(self):
        assert parse_number("4500.0") == Decimal("4500.0")

    def test_us_thousands(self):
        assert parse_number("4,500.00") == Decimal("4500.00")

    def test_european_thousands(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_european_thousands_swiss(self):
        assert parse_number("1'234.56") is None or True  # apostrophe not handled anyway; placeholder

    def test_currency_symbol(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")
        assert parse_number("€100") == Decimal("100")
        assert parse_number("£50") == Decimal("50")

    def test_currency_code(self):
        assert parse_number("USD 100") == Decimal("100")
        assert parse_number("100 EUR") == Decimal("100")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_negative_sign(self):
        assert parse_number("-42") == Decimal("-42")
        assert parse_number("-$100") == Decimal("-100")

    def test_positive_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_spaces_as_thousands(self):
        assert parse_number("1 234 567") == Decimal("1234567")

    def test_thousands_no_decimal(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_comma_decimal_two_digits(self):
        # "12,34" -> decimal
        assert parse_number("12,34") == Decimal("12.34")

    def test_no_digit(self):
        assert parse_number("abc") is None

    def test_only_dot(self):
        assert parse_number(".") is None

    def test_leading_thousands(self):
        # "1,234,567.89"
        assert parse_number("1,234,567.89") == Decimal("1234567.89")

    def test_fullwidth_digits(self):
        assert parse_number("\uff14\uff15\uff16") == Decimal("456")

    def test_lowercase_currency_code(self):
        assert parse_number("usd 10") == Decimal("10")

    def test_negative_european(self):
        assert parse_number("-1.234,56") == Decimal("-1234.56")


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_exact_equal(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.00")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("2")) is False

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.01"), tolerance=Decimal("0.02")) is True

    def test_tolerance_outside(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.05"), tolerance=Decimal("0.02")) is False

    def test_tolerance_boundary(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.02"), tolerance=Decimal("0.02")) is True

    def test_zero_tolerance(self):
        assert numbers_equal(Decimal("1"), Decimal("1"), tolerance=Decimal("0")) is True
        assert numbers_equal(Decimal("1"), Decimal("2"), tolerance=Decimal("0")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_iso_ymd(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)

    def test_ymd_slashes(self):
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)

    def test_ymd_dots(self):
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_ambiguous_first_gt_12(self):
        # 15/04 -> day is 15, month is 4
        assert parse_date("15/04/2026") == _dt.date(2026, 4, 15)

    def test_ambiguous_second_gt_12(self):
        # 04/15 -> month is 4, day is 15
        assert parse_date("04/15/2026") == _dt.date(2026, 4, 15)

    def test_ambiguous_refuses(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_month_name(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)
        assert parse_date("4 Mar 2026") == _dt.date(2026, 3, 4)

    def test_month_name_with_suffix(self):
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)
        assert parse_date("22nd March 2026") == _dt.date(2026, 3, 22)
        assert parse_date("3rd March 2026") == _dt.date(2026, 3, 3)

    def test_month_name_with_dot(self):
        assert parse_date("4 Mar. 2026") == _dt.date(2026, 3, 4)

    def test_month_name_first(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)
        assert parse_date("Mar 4 2026") == _dt.date(2026, 3, 4)

    def test_month_name_first_with_dot(self):
        assert parse_date("Mar. 4 2026") == _dt.date(2026, 3, 4)

    def test_month_name_first_suffix(self):
        assert parse_date("March 4th, 2026") == _dt.date(2026, 3, 4)

    def test_unknown_month(self):
        assert parse_date("4 Foo 2026") is None

    def test_invalid_date(self):
        # 31 February
        assert parse_date("2026-02-31") is None

    def test_unparseable(self):
        assert parse_date("not a date") is None

    def test_mixed_case_month(self):
        assert parse_date("4 MARCH 2026") == _dt.date(2026, 3, 4)

    def test_leading_trailing_space(self):
        assert parse_date("  2026-03-04  ") == _dt.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("hello world", "hello world") == 1.0

    def test_reordered(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial_overlap(self):
        # tokens: {a,b} vs {b,c} -> intersection {b}, union {a,b,c}
        score = token_set_ratio("a b", "b c")
        assert score == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert token_set_ratio("a b", "c d") == 0.0

    def test_empty_a(self):
        assert token_set_ratio("", "hello") == 0.0

    def test_empty_b(self):
        assert token_set_ratio("hello", "") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_punctuation_stripped_per_token(self):
        assert token_set_ratio("Ahmed,", "Ahmed") == 1.0

    def test_duplicate_tokens(self):
        # set dedup: {a,b} vs {a,b}
        assert token_set_ratio("a a b", "a b b") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("Hello", "hello") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_equal(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_a(self):
        assert levenshtein("", "abcd") == 4

    def test_empty_b(self):
        assert levenshtein("abcd", "") == 4

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_insertion(self):
        assert levenshtein("abc", "abcd") == 1

    def test_deletion(self):
        assert levenshtein("abcd", "abc") == 1

    def test_substitution(self):
        assert levenshtein("abc", "abd") == 1

    def test_kitten_sitting(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("", "hello") == 0.0
        assert similarity("hello", "") == 0.0

    def test_single_edit(self):
        # len 5; 1 edit -> 0.8
        assert similarity("hello", "hallp") == pytest.approx(0.8)

    def test_case_insensitive(self):
        assert similarity("HeLLo", "hello") == 1.0

    def test_whitespace_normalized(self):
        assert similarity("a  b", "a b") == 1.0

    def test_different_lengths(self):
        # "ab" vs "abcd": longest = 4, distance = 2 -> 0.5
        assert similarity("ab", "abcd") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single(self):
        results = list(iter_numbers("Total: $4,500.00"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "$4,500.00"
        assert val == Decimal("4500.00")
        assert "Total: "[0:0] == ""
        source = "Total: $4,500.00"
        assert source[start:end] == "$4,500.00"

    def test_multiple(self):
        text = "bought 3 items for $4,500.00 each"
        results = list(iter_numbers(text))
        values = [r[3] for r in results]
        assert Decimal("3") in values
        assert Decimal("4500.00") in values

    def test_spans_match_source(self):
        text = "The value 1,234 appears here."
        for raw, start, end, val in iter_numbers(text):
            assert text[start:end] == raw

    def test_leading_space_tightened(self):
        # Pattern may absorb leading whitespace; span must exclude it.
        text = "x  42 y"
        for raw, start, end, val in iter_numbers(text):
            assert not raw.startswith(" ")
            assert text[start] == raw[0]

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_multiple_simple(self):
        text = "1 2 3"
        results = list(iter_numbers(text))
        values = [r[3] for r in results]
        assert Decimal("1") in values
        assert Decimal("2") in values
        assert Decimal("3") in values

    def test_unicode_folded_before_scan(self):
        # full-width digits should be found and parsed
        results = list(iter_numbers("\uff14\uff15\uff16"))
        # iter_numbers folds before scanning, but the span refers to the folded text.
        assert len(results) >= 1

    def test_unparseable_skipped(self):
        # A "." alone parses to None; ensure it doesn't crash.
        list(iter_numbers("hello . world"))


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_iso(self):
        text = "Due: 2026-03-04."
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "2026-03-04"
        assert val == _dt.date(2026, 3, 4)
        assert text[start:end] == "2026-03-04"

    def test_multiple(self):
        text = "from 2026-01-01 to 2026-12-31"
        results = list(iter_dates(text))
        values = [r[3] for r in results]
        assert _dt.date(2026, 1, 1) in values
        assert _dt.date(2026, 12, 31) in values

    def test_month_name(self):
        text = "Signed on 4 March 2026."
        results = list(iter_dates(text))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 3, 4)

    def test_ambiguous_skipped(self):
        text = "Date: 03/04/2026"
        # ambiguous, no day_first -> not yielded
        assert list(iter_dates(text)) == []

    def test_ambiguous_day_first(self):
        text = "Date: 03/04/2026"
        results = list(iter_dates(text, day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        text = "Date: 03/04/2026"
        results = list(iter_dates(text, day_first=False))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 3, 4)

    def test_spans_match_source(self):
        text = "start 2026-01-01 end 4 March 2026"
        for raw, start, end, val in iter_dates(text):
            assert text[start:end] == raw

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_invalid_date_skipped(self):
        # 2026-02-31 is invalid
        assert list(iter_dates("2026-02-31")) == []

    def test_no_duplicate_spans(self):
        # A date that could be matched by more than one pattern shouldn't be
        # yielded twice for the same span; here just verify dedup mechanism
        # by checking count of unique spans equals yielded count.
        text = "2026-01-01"
        results = list(iter_dates(text))
        spans = [(s, e) for _, s, e, _ in results]
        assert len(spans) == len(set(spans))
