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
    def test_ascii_passthrough(self):
        assert fold_unicode("hello world") == "hello world"

    def test_empty(self):
        assert fold_unicode("") == ""

    def test_smart_quotes(self):
        assert fold_unicode("\u2018a\u2019") == "'a'"
        assert fold_unicode("\u201c\u201d\u201e\u201f") == '""""'

    def test_dashes(self):
        assert fold_unicode("a\u2010b") == "a-b"
        assert fold_unicode("\u2013\u2014\u2015") == "---"
        assert fold_unicode("1\u22122") == "1-2"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b") == "a b"
        assert fold_unicode("a\u2007b\u202fc\u2009d") == "a b c d"

    def test_zero_width_removed(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeff") == "abcd"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        # U+FB01 LATIN SMALL LIGATURE FI
        assert fold_unicode("\ufb01le") == "file"

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff22") == "AB"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("  a   b\t\tc\n") == "a b c"

    def test_casefold_default(self):
        assert normalize_text("HeLLo World") == "hello world"

    def test_casefold_false(self):
        assert normalize_text("HeLLo   World", casefold=False) == "HeLLo World"

    def test_folds_unicode(self):
        assert normalize_text("\u201cHi\u201d") == '"hi"'

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   \t\n") == ""

    def test_combining_whitespace_forms(self):
        assert normalize_text("a\u00a0b") == "a b"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_punctuation(self):
        assert strip_punctuation("Hello, World!") == "Hello World"

    def test_keeps_underscore_and_digits(self):
        assert strip_punctuation("a_b 1-2") == "a_b 12"

    def test_empty(self):
        assert strip_punctuation("") == ""

    def test_only_punctuation(self):
        assert strip_punctuation("!@#$%^&*()") == ""

    def test_unicode_punctuation(self):
        # em dash and smart quotes are non-word chars
        assert strip_punctuation("\u2014a\u201c") == "a"


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty(self):
        assert parse_number("") is None
        assert parse_number("   ") is None

    def test_plain_int(self):
        assert parse_number("4500") == Decimal("4500")

    def test_plain_decimal(self):
        assert parse_number("4500.50") == Decimal("4500.50")
        assert parse_number("4500.5") == Decimal("4500.5")
        assert parse_number("0.0") == Decimal("0.0")
        assert parse_number("1") == Decimal("1.0")

    def test_negative_sign(self):
        assert parse_number("-42") == Decimal("-42")
        assert parse_number("-42.10") == Decimal("-42.10")

    def test_positive_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_accounting_negative(self):
        assert parse_number("(1,234.56)") == Decimal("-1234.56")
        assert parse_number("($1,234.56)") == Decimal("-1234.56")

    def test_currency_symbol(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")
        assert parse_number("\u20ac1.234,56") == Decimal("1234.56")
        assert parse_number("\u00a34,500.00") == Decimal("4500.00")

    def test_currency_code(self):
        assert parse_number("USD 4,500.00") == Decimal("4500.00")
        assert parse_number("EUR 1000") == Decimal("1000")

    def test_us_thousands(self):
        assert parse_number("1,234,567.89") == Decimal("1234567.89")
        assert parse_number("1,234") == Decimal("1234")

    def test_european_thousands(self):
        assert parse_number("1.234.567,89") == Decimal("1234567.89")
        assert parse_number("1.234") == Decimal("1234")

    def test_single_separator_2_digit_decimal(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_space_thousands(self):
        assert parse_number("1 234 567") == Decimal("1234567")

    def test_no_digits(self):
        assert parse_number("abc") is None
        assert parse_number("$") is None

    def test_stray_text(self):
        # letters interspersed -> not a plain number
        assert parse_number("1a2") is None

    def test_not_fully_numeric(self):
        assert parse_number("--5") is None

    def test_leading_dot(self):
        assert parse_number(".5") == Decimal(".5")

    def test_only_dot(self):
        assert parse_number(".") is None

    def test_whitespace_stripped(self):
        assert parse_number("   $ 1,000  ") == Decimal("1000")

    def test_int_input(self):
        assert parse_number(42) == Decimal("42")

    def test_float_input(self):
        val = parse_number(4500.0)
        assert val == Decimal("4500")

    def test_fullwidth_digits(self):
        # NFKC turns full-width into ASCII
        assert parse_number("\uff11\uff12\uff13") == Decimal("123")


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("1.0")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("2")) is False

    def test_none_returns_false(self):
        assert numbers_equal(None, Decimal("1")) is False
        assert numbers_equal(Decimal("1"), None) is False
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("100.0"), Decimal("100.5"), Decimal("1")) is True

    def test_tolerance_boundary(self):
        assert numbers_equal(Decimal("100.0"), Decimal("101.0"), Decimal("1")) is True

    def test_tolerance_exceeded(self):
        assert numbers_equal(Decimal("100.0"), Decimal("101.1"), Decimal("1")) is False

    def test_tolerance_negative_diff(self):
        assert numbers_equal(Decimal("100.0"), Decimal("99.5"), Decimal("1")) is True

    def test_zero_tolerance(self):
        assert numbers_equal(Decimal("1"), Decimal("1"), Decimal("0")) is True
        assert numbers_equal(Decimal("1"), Decimal("1.1"), Decimal("0")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_ymd(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_dmy_mdy_day_gt_12(self):
        assert parse_date("13/04/2026") == _dt.date(2026, 4, 13)

    def test_dmy_mdy_second_gt_12(self):
        assert parse_date("04/13/2026") == _dt.date(2026, 4, 13)

    def test_ambiguous_no_hint(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_d_mon_yyyy(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)
        assert parse_date("04 Mar. 2026") == _dt.date(2026, 3, 4)
        assert parse_date("4 March, 2026") == _dt.date(2026, 3, 4)

    def test_month_d_yyyy(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)
        assert parse_date("Mar 4th 2026") == _dt.date(2026, 3, 4)
        assert parse_date("Sept. 4, 2026") == _dt.date(2026, 9, 4)

    def test_invalid_date_value_error(self):
        assert parse_date("2026-02-31") is None
        assert parse_date("31 February 2026") is None

    def test_unparseable(self):
        assert parse_date("not a date") is None
        assert parse_date("hello 123") is None

    def test_unknown_month(self):
        assert parse_date("4 Foo 2026") is None

    def test_embedded_in_text(self):
        assert parse_date("On 2026-03-04 we met") == _dt.date(2026, 3, 4)

    def test_day_first_with_unambiguous(self):
        # first > 12 forces the reading regardless of hint
        assert parse_date("25/03/2026", day_first=False) == _dt.date(2026, 3, 25)


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
        assert levenshtein("kitten", "sitting") == 3

    def test_insertion(self):
        assert levenshtein("abc", "abcd") == 1

    def test_deletion(self):
        assert levenshtein("abcd", "abc") == 1

    def test_case_sensitive_no_normalize(self):
        assert levenshtein("Abc", "abc") == 1


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

    def test_case_and_whitespace_ignored(self):
        assert similarity("Hello  World", "hello world") == 1.0

    def test_partial(self):
        # 4 chars, one edit -> 0.75
        assert similarity("abcd", "abce") == pytest.approx(0.75)

    def test_leven_and_ratio(self):
        # "abc" vs "ab" -> distance 1, longest 3 -> 2/3
        assert similarity("abc", "ab") == pytest.approx(2.0 / 3.0)

    def test_range(self):
        for a, b in [("a", "b"), ("xyz", "xy"), ("", "a"), ("abc", "abc")]:
            s = similarity(a, b)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("hello world", "hello world") == 1.0

    def test_order_insensitive(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("Ahmed KHAN", "ahmed khan") == 1.0

    def test_partial_overlap(self):
        # {a, b} & {b, c} = {b}; union {a,b,c} -> 1/3
        assert token_set_ratio("a b", "b c") == pytest.approx(1.0 / 3.0)

    def test_disjoint(self):
        assert token_set_ratio("a b", "c d") == 0.0

    def test_empty_a(self):
        assert token_set_ratio("", "x y") == 0.0

    def test_empty_b(self):
        assert token_set_ratio("x y", "") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_duplicates_ignored(self):
        # sets collapse duplicates
        assert token_set_ratio("a a a", "a") == 1.0

    def test_punctuation_stripped_per_token(self):
        assert token_set_ratio("hello,", "hello") == 1.0

    def test_only_punctuation(self):
        # tokens all empty after strip -> 0
        assert token_set_ratio("!!!", "???") == 0.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single(self):
        results = list(iter_numbers("The total is $4,500.00 today"))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert raw == "$4,500.00"
        assert value == Decimal("4500.00")
        assert start == 13
        assert end == start + len(raw)

    def test_multiple(self):
        results = list(iter_numbers("1 dollar and 2 dollars"))
        values = [v for _, _, _, v in results]
        assert values == [Decimal("1"), Decimal("2")]

    def test_empty(self):
        assert list(iter_numbers("")) == []

    def test_no_numbers(self):
        assert list(iter_numbers("no numbers here")) == []

    def test_span_tightened(self):
        text = "value:   42"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert raw == "42"
        assert text[start:end] == "42"
        assert value == Decimal("42")

    def test_folded_unicode(self):
        # Full-width currency/full-width digits should be folded
        results = list(iter_numbers("\uff041,234"))  # full-width dollar + 1,234
        assert len(results) == 1
        _, _, _, value = results[0]
        assert value == Decimal("1234")

    def test_union_spans_each_number(self):
        text = "abc 100 def 200"
        spans = [(raw, s, e) for raw, s, e, _ in iter_numbers(text)]
        for raw, s, e in spans:
            assert text[s:e] == raw

    def test_negative(self):
        results = list(iter_numbers("balance: -$50"))
        assert len(results) == 1
        assert results[0][3] == Decimal("-50")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_iso(self):
        results = list(iter_dates("On 2026-03-04 we meet"))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert value == _dt.date(2026, 3, 4)
        assert raw == "2026-03-04"

    def test_multiple_formats(self):
        text = "2026-03-04 and March 5, 2026 and 6 March 2026"
        values = [v for _, _, _, v in iter_dates(text)]
        assert values == [_dt.date(2026, 3, 4), _dt.date(2026, 3, 5), _dt.date(2026, 3, 6)]

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_ambiguous_skipped_without_hint(self):
        assert list(iter_dates("03/04/2026")) == []

    def test_ambiguous_with_hint(self):
        results = list(iter_dates("03/04/2026", day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_invalid_date_skipped(self):
        assert list(iter_dates("2026-02-31")) == []

    def test_spans_match_text(self):
        text = "Some date 2026-03-04 here"
        for raw, start, end, _ in iter_dates(text):
            assert text[start:end] == raw

    def test_no_duplicate_spans(self):
        # A single unambiguous date should appear only once
        results = list(iter_dates("2026-03-04"))
        assert len(results) == 1
