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

    def test_zero_width_removed(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        assert fold_unicode("\ufb01") == "fi"

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff11") == "A1"

    def test_plain_ascii_unchanged(self):
        assert fold_unicode("hello world") == "hello world"

    def test_empty(self):
        assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_whitespace_collapse_and_strip(self):
        assert normalize_text("  a   b\t\n c  ") == "a b c"

    def test_default_casefold(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_casefold_disabled(self):
        assert normalize_text("Hello WORLD", casefold=False) == "Hello WORLD"

    def test_unicode_folding_applied(self):
        assert normalize_text("\u201cHi\u201d\u2014there") == '"hi"-there'

    def test_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   \t\n ") == ""

    def test_nbsp_collapsed_to_single_space(self):
        assert normalize_text("a\u00a0\u00a0b") == "a b"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_punctuation(self):
        assert strip_punctuation("Hello, world!") == "Hello world"

    def test_keeps_underscore_and_digits(self):
        assert strip_punctuation("a_1-b#2") == "a_1b2"

    def test_keeps_whitespace(self):
        assert strip_punctuation("a  b") == "a  b"

    def test_all_punctuation(self):
        assert strip_punctuation("!!!???") == ""

    def test_empty(self):
        assert strip_punctuation("") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_integer(self):
        assert parse_number("42") == Decimal("42")

    def test_float(self):
        assert parse_number("3.14") == Decimal("3.14")

    def test_negative(self):
        assert parse_number("-7") == Decimal("-7")

    def test_explicit_positive(self):
        assert parse_number("+7") == Decimal("7")

    def test_currency_symbol(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")

    def test_euro_symbol(self):
        assert parse_number("€1234") == Decimal("1234")

    def test_us_thousands(self):
        assert parse_number("1,234,567") == Decimal("1234567")

    def test_us_decimal_with_thousands(self):
        assert parse_number("1,234.56") == Decimal("1234.56")

    def test_european_decimal(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_european_thousands(self):
        assert parse_number("1.234.567") == Decimal("1234567")

    def test_ambiguous_single_comma_two_digit_tail_decimal(self):
        # "12,34" -> decimal
        assert parse_number("12,34") == Decimal("12.34")

    def test_single_comma_three_digit_tail_thousands(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_accounting_negative_with_spaces(self):
        assert parse_number("(  1,234.00 )") == Decimal("-1234.00")

    def test_currency_code_removed(self):
        assert parse_number("USD 1,000") == Decimal("1000")

    def test_currency_code_lowercase(self):
        assert parse_number("1000 usd") == Decimal("1000")

    def test_spaces_removed(self):
        assert parse_number("1 234 567") == Decimal("1234567")

    def test_none_returns_none(self):
        assert parse_number(None) is None

    def test_empty_string(self):
        assert parse_number("") is None

    def test_whitespace_only(self):
        assert parse_number("   ") is None

    def test_no_digits(self):
        assert parse_number("abc") is None

    def test_currency_only(self):
        assert parse_number("$") is None

    def test_lone_dot(self):
        assert parse_number(".") is None

    def test_two_dots_unrecognised(self):
        assert parse_number("1.2.3") is None

    def test_negative_currency(self):
        assert parse_number("-$500") == Decimal("-500")

    def test_returns_decimal_type(self):
        result = parse_number("10")
        assert isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_equal_exact(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.5")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_within_tolerance(self):
        assert numbers_equal(Decimal("1.005"), Decimal("1.0"), tolerance=Decimal("0.01")) is True

    def test_exactly_at_tolerance(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.0"), tolerance=Decimal("0.5")) is True

    def test_outside_tolerance(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.0"), tolerance=Decimal("0.4")) is False

    def test_exact_different_with_zero_tolerance(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.1"), tolerance=Decimal("0")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_ymd(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)

    def test_ymd_slashes(self):
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)

    def test_ymd_dots(self):
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_dmy_unambiguous_first_gt_12(self):
        assert parse_date("13/04/2026") == _dt.date(2026, 4, 13)

    def test_mdy_unambiguous_second_gt_12(self):
        assert parse_date("04/13/2026") == _dt.date(2026, 4, 13)

    def test_ambiguous_default_none(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_day_month_year(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)

    def test_day_month_year_with_suffix(self):
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)

    def test_day_month_year_with_comma(self):
        assert parse_date("4 Mar, 2026") == _dt.date(2026, 3, 4)

    def test_month_day_year(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)

    def test_month_day_year_suffix(self):
        assert parse_date("March 4th 2026") == _dt.date(2026, 3, 4)

    def test_month_abbrev_with_dot(self):
        assert parse_date("Mar. 4 2026") == _dt.date(2026, 3, 4)

    def test_sept_abbrev(self):
        assert parse_date("Sept 5 2026") == _dt.date(2026, 9, 5)

    def test_invalid_day_in_month(self):
        assert parse_date("2026-02-31") is None

    def test_none_input(self):
        assert parse_date(None) is None

    def test_empty_input(self):
        assert parse_date("") is None

    def test_no_date(self):
        assert parse_date("hello world") is None

    def test_unknown_month_name(self):
        assert parse_date("4 Foo 2026") is None

    def test_returns_date_type(self):
        assert isinstance(parse_date("2026-03-04"), _dt.date)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("Ahmed Khan", "Ahmed Khan") == 1.0

    def test_order_insensitive(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial_overlap(self):
        # {ahmed, khan} vs {ahmed, ali}: intersection 1, union 3
        assert token_set_ratio("Ahmed Khan", "Ahmed Ali") == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert token_set_ratio("abc", "xyz") == 0.0

    def test_empty_a(self):
        assert token_set_ratio("", "abc") == 0.0

    def test_empty_b(self):
        assert token_set_ratio("abc", "") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_punctuation_stripped_per_token(self):
        assert token_set_ratio("Ahmed, Khan", "Khan Ahmed") == 1.0

    def test_duplicates_ignored(self):
        assert token_set_ratio("a a a b", "b a") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("HELLO world", "hello WORLD") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_both(self):
        assert levenshtein("", "") == 0

    def test_empty_a(self):
        assert levenshtein("", "abc") == 3

    def test_empty_b(self):
        assert levenshtein("abc", "") == 3

    def test_single_substitution(self):
        assert levenshtein("cat", "bat") == 1

    def test_insertion(self):
        assert levenshtein("cat", "cart") == 1

    def test_deletion(self):
        assert levenshtein("cart", "cat") == 1

    def test_kitten_sitting(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_symmetric(self):
        assert levenshtein("abc", "abd") == levenshtein("abd", "abc")


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

    def test_other_empty(self):
        assert similarity("", "hello") == 0.0

    def test_single_edit(self):
        # "cat" vs "bat": distance 1, longest 3 -> 1 - 1/3
        assert similarity("cat", "bat") == pytest.approx(2 / 3)

    def test_case_and_whitespace_normalized(self):
        assert similarity("Hello  World", "hello world") == 1.0

    def test_whitespace_only_strings(self):
        # both normalise to empty
        assert similarity("   ", "\t\n") == 1.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single_number_in_text(self):
        results = list(iter_numbers("The total is $4,500.00 today."))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert value == Decimal("4500.00")
        assert "The total is ".find  # sanity no-op guard
        text = "The total is $4,500.00 today."
        assert text[start:end] == raw
        assert start == text.index("$")

    def test_multiple_numbers(self):
        text = "from 10 to 20"
        results = list(iter_numbers(text))
        assert [r[3] for r in results] == [Decimal("10"), Decimal("20")]

    def test_spans_exclude_leading_whitespace(self):
        text = "x 123"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert raw == "123"
        assert text[start:end] == "123"
        assert value == Decimal("123")

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_empty_string(self):
        assert list(iter_numbers("")) == []

    def test_unicode_folding_applied(self):
        # ellipsis etc should not break; ensure a number after unicode is found
        results = list(iter_numbers("a\u2014b 5"))
        assert [r[3] for r in results] == [Decimal("5")]

    def test_yields_four_tuple(self):
        results = list(iter_numbers("42"))
        assert len(results) == 1
        item = results[0]
        assert isinstance(item, tuple)
        assert len(item) == 4
        assert item[0] == "42"
        assert item[1] == 0
        assert item[2] == 2
        assert item[3] == Decimal("42")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_date(self):
        text = "Due on 2026-03-04 please."
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert value == _dt.date(2026, 3, 4)
        assert text[start:end] == raw

    def test_multiple_dates(self):
        text = "From 2026-03-04 to 2026-04-05"
        results = list(iter_dates(text))
        values = [r[3] for r in results]
        assert _dt.date(2026, 3, 4) in values
        assert _dt.date(2026, 4, 5) in values

    def test_text_date(self):
        results = list(iter_dates("March 4, 2026"))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 3, 4)

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_day_first_forwarded(self):
        results = list(iter_dates("03/04/2026", day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_ambiguous_no_day_first_yields_nothing(self):
        assert list(iter_dates("03/04/2026")) == []

    def test_spans_match_slice(self):
        text = "date 2026-12-25 end"
        raw, start, end, value = list(iter_dates(text))[0]
        assert text[start:end] == raw
        assert value == _dt.date(2026, 12, 25)
