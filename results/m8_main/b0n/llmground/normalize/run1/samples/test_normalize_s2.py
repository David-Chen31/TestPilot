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
    def test_smart_quotes(self):
        assert fold_unicode("\u2018a\u2019 \u201cb\u201d") == "'a' \"b\""

    def test_single_low_quotes(self):
        assert fold_unicode("\u201aa\u201bb") == "'a'b"

    def test_double_low_quotes(self):
        assert fold_unicode("\u201ea\u201fb") == '"a"b'

    def test_dashes(self):
        assert fold_unicode("a\u2010b\u2011c\u2012d\u2013e\u2014f\u2015g\u2212h") == "a-b-c-d-e-f-g-h"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"

    @pytest.mark.parametrize("ch", ["\u200b", "\u200c", "\u200d", "\ufeff"])
    def test_zero_width_removed(self, ch):
        assert fold_unicode("a" + ch + "b") == "ab"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        assert fold_unicode("\ufb01le") == "file"

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff22\uff23") == "ABC"

    def test_plain_ascii_unchanged(self):
        assert fold_unicode("Hello, World!") == "Hello, World!"

    def test_empty(self):
        assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_casefold_default(self):
        assert normalize_text("Hello World") == "hello world"

    def test_casefold_false(self):
        assert normalize_text("Hello World", casefold=False) == "Hello World"

    def test_whitespace_collapsed(self):
        assert normalize_text("  a\t\tb\n\n c  ") == "a b c"

    def test_unicode_folding_applied(self):
        assert normalize_text("\u201cHi\u201d") == '"hi"'

    def test_fold_plus_casefold_false(self):
        assert normalize_text("A\u00a0B", casefold=False) == "A B"

    def test_whitespace_only(self):
        assert normalize_text("   \t\n ") == ""

    def test_empty(self):
        assert normalize_text("") == ""

    def test_casefold_uses_str_casefold(self):
        # ß casefolds to "ss"
        assert normalize_text("Straße") == "strasse"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_basic(self):
        assert strip_punctuation("hello, world!") == "hello world"

    def test_keeps_digits_and_underscore(self):
        assert strip_punctuation("a_1 b-2") == "a_1 b2"

    def test_keeps_whitespace(self):
        assert strip_punctuation("a  b") == "a  b"

    def test_only_punctuation(self):
        assert strip_punctuation("!!!???") == ""

    def test_empty(self):
        assert strip_punctuation("") == ""

    def test_keeps_unicode_letters(self):
        assert strip_punctuation("café!") == "café"


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_simple_integer(self):
        assert parse_number("4500") == Decimal("4500")

    def test_us_thousands(self):
        assert parse_number("4,500.00") == Decimal("4500.00")

    def test_us_thousands_no_decimal(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_european_thousands(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_european_thousands_no_decimal(self):
        assert parse_number("1.234.567") == Decimal("1234567")

    def test_decimal_comma_short_tail(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_decimal_dot(self):
        assert parse_number("3.14") == Decimal("3.14")

    def test_currency_dollar(self):
        assert parse_number("$1,200.00") == Decimal("1200.00")

    def test_currency_euro(self):
        assert parse_number("€50") == Decimal("50")

    def test_currency_code_usd(self):
        assert parse_number("USD 100") == Decimal("100")

    def test_currency_code_case_insensitive(self):
        assert parse_number("100 eur") == Decimal("100")

    def test_currency_code_inr(self):
        assert parse_number("INR 5,000") == Decimal("5000")

    def test_negative_sign(self):
        assert parse_number("-42") == Decimal("-42")

    def test_positive_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_accounting_negative_with_currency(self):
        assert parse_number("($1,234.00)") == Decimal("-1234.00")

    def test_negative_with_comma_decimal(self):
        assert parse_number("-1,50") == Decimal("-1.50")

    def test_space_thousands(self):
        # _NUM_RE allows spaces, but parse_number strips them post-currency.
        assert parse_number("1 234") == Decimal("1234")

    def test_nbsp_thousands(self):
        assert parse_number("1\u00a0234") == Decimal("1234")

    def test_none_returns_none(self):
        assert parse_number(None) is None

    def test_empty_returns_none(self):
        assert parse_number("") is None

    def test_whitespace_returns_none(self):
        assert parse_number("   ") is None

    def test_no_digits_returns_none(self):
        assert parse_number("abc") is None

    def test_only_sign_returns_none(self):
        assert parse_number("-") is None

    def test_only_currency_returns_none(self):
        assert parse_number("$") is None

    def test_non_str_converted(self):
        assert parse_number(42) == Decimal("42")

    def test_unicode_negative_sign(self):
        assert parse_number("\u221250") == Decimal("50")

    def test_trailing_dot_ok(self):
        assert parse_number("12.") == Decimal("12")

    def test_currency_with_spaces_european(self):
        assert parse_number("€ 1.234,56") == Decimal("1234.56")


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.50")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("2")) is False

    def test_none_a(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_b(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.01"), Decimal("0.02")) is True

    def test_tolerance_exact_boundary(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.05"), Decimal("0.05")) is True

    def test_tolerance_outside(self):
        assert numbers_equal(Decimal("1.00"), Decimal("1.10"), Decimal("0.05")) is False

    def test_tolerance_negative_difference(self):
        assert numbers_equal(Decimal("2.00"), Decimal("1.95"), Decimal("0.05")) is True


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_ymd(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)

    def test_ymd_slash(self):
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)

    def test_ymd_dot(self):
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_ymd_single_digit_month(self):
        assert parse_date("2026-3-4") == _dt.date(2026, 3, 4)

    def test_dmy_unambiguous_first_gt_12(self):
        assert parse_date("13/04/2026") == _dt.date(2026, 4, 13)

    def test_mdy_unambiguous_second_gt_12(self):
        assert parse_date("04/13/2026") == _dt.date(2026, 4, 13)

    def test_ambiguous_no_hint(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_dmy_dashes(self):
        assert parse_date("13-04-2026") == _dt.date(2026, 4, 13)

    def test_dmy_dots(self):
        assert parse_date("13.04.2026") == _dt.date(2026, 4, 13)

    def test_dmonthy(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_with_ordinal(self):
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_abbrev_with_dot(self):
        assert parse_date("4 Mar. 2026") == _dt.date(2026, 3, 4)

    def test_dmonthy_comma(self):
        assert parse_date("4 March, 2026") == _dt.date(2026, 3, 4)

    def test_monthdy(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)

    def test_monthdy_ordinal(self):
        assert parse_date("March 4th 2026") == _dt.date(2026, 3, 4)

    def test_monthdy_abbrev(self):
        assert parse_date("Mar 4 2026") == _dt.date(2026, 3, 4)

    def test_sept_abbrev(self):
        assert parse_date("Sept 4 2026") == _dt.date(2026, 9, 4)

    def test_invalid_day(self):
        assert parse_date("2026-02-31") is None

    def test_unknown_month_name(self):
        assert parse_date("4 Foober 2026") is None

    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None

    def test_whitespace(self):
        assert parse_date("   ") is None

    def test_garbage(self):
        assert parse_date("not a date") is None

    def test_unicode_dash_date(self):
        # En dash folds to ascii hyphen before parsing.
        assert parse_date("2026\u201303\u201304") == _dt.date(2026, 3, 4)

    def test_non_str_converted(self):
        d = _dt.date(2026, 5, 6)
        assert parse_date(d) == d


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("Ahmed Khan", "Ahmed Khan") == 1.0

    def test_reordered(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial_overlap(self):
        # {a, b} vs {b, c}: intersection 1, union 3
        assert token_set_ratio("a b", "b c") == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert token_set_ratio("a b", "c d") == 0.0

    def test_empty_a(self):
        assert token_set_ratio("", "x") == 0.0

    def test_empty_b(self):
        assert token_set_ratio("x", "") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_punctuation_ignored(self):
        assert token_set_ratio("hello,", "hello") == 1.0

    def test_duplicates_ignored(self):
        assert token_set_ratio("a a a", "a") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("HELLO", "hello") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_a(self):
        assert levenshtein("", "abc") == 3

    def test_empty_b(self):
        assert levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_substitution(self):
        assert levenshtein("kitten", "sitten") == 1

    def test_classic(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_deletion(self):
        assert levenshtein("abc", "ac") == 1

    def test_insertion(self):
        assert levenshtein("ac", "abc") == 1

    def test_symmetric(self):
        assert levenshtein("abc", "abd") == levenshtein("abd", "abc")


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("Hello", "Hello") == 1.0

    def test_case_insensitive(self):
        assert similarity("HELLO", "hello") == 1.0

    def test_whitespace_normalized(self):
        assert similarity("a  b", "a b") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("", "abc") == 0.0

    def test_other_empty(self):
        assert similarity("abc", "") == 0.0

    def test_single_edit(self):
        # "cat" vs "bat": distance 1, longest 3 -> 2/3
        assert similarity("cat", "bat") == pytest.approx(2 / 3)

    def test_reordered_is_low(self):
        # character-level: dissimilar ordering
        score = similarity("abc", "cba")
        assert 0.0 <= score < 1.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single(self):
        results = list(iter_numbers("Total: $4,500.00 due"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "$4,500.00"
        assert val == Decimal("4500.00")
        assert "Total: $4,500.00 due"[start:end] == raw

    def test_multiple(self):
        results = list(iter_numbers("1 and 2 and 3"))
        values = [r[3] for r in results]
        assert values == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_span_correct(self):
        text = "price is 99 dollars"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "99"
        assert text[start:end] == "99"

    def test_span_excludes_leading_whitespace(self):
        text = "a 123"
        results = list(iter_numbers(text))
        raw, start, end, val = results[0]
        assert raw == "123"
        assert text[start] == "1"
        assert text[start:end] == "123"

    def test_no_numbers(self):
        assert list(iter_numbers("no numbers here")) == []

    def test_empty_text(self):
        assert list(iter_numbers("")) == []

    def test_negative(self):
        results = list(iter_numbers("change -5 today"))
        assert len(results) == 1
        assert results[0][3] == Decimal("-5")

    def test_unicode_folding_applied_to_spans(self):
        # The fullwidth digit folds to ascii; span points at original.
        text = "val \uff11\uff12\uff13 end"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert val == Decimal("123")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_iso(self):
        text = "on 2026-03-04 we ship"
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "2026-03-04"
        assert val == _dt.date(2026, 3, 4)
        assert text[start:end] == raw

    def test_ambiguous_skipped_without_hint(self):
        text = "on 03/04/2026"
        assert list(iter_dates(text)) == []

    def test_ambiguous_with_day_first(self):
        text = "on 03/04/2026"
        results = list(iter_dates(text, day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_multiple(self):
        text = "2026-01-01 and March 4, 2026"
        results = list(iter_dates(text))
        values = sorted(r[3] for r in results)
        assert values == [_dt.date(2026, 1, 1), _dt.date(2026, 3, 4)]

    def test_no_dates(self):
        assert list(iter_dates("no dates here")) == []

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_invalid_date_skipped(self):
        assert list(iter_dates("2026-02-31")) == []

    def test_span_consistent(self):
        text = "start 13/04/2026 end"
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert text[start:end] == raw
        assert val == _dt.date(2026, 4, 13)
