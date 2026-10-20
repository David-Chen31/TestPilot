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
    def test_curly_quotes(self):
        assert fold_unicode("\u2018a\u2019 \u201cb\u201d") == "'a' \"b\""

    def test_single_low_quotes(self):
        assert fold_unicode("\u201a\u201b") == "''"

    def test_double_low_quotes(self):
        assert fold_unicode("\u201e\u201f") == '""'

    def test_dashes(self):
        assert fold_unicode("a\u2010b\u2011c\u2012d\u2013e\u2014f\u2015g\u2212h") == "a-b-c-d-e-f-g-h"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"

    def test_zero_width_removed(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        # ﬁ ligature is folded to "fi" by NFKC
        assert fold_unicode("\ufb01n") == "fin"

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff22\uff23") == "ABC"

    def test_plain_ascii_unchanged(self):
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

    def test_casefold_false_preserves_case(self):
        assert normalize_text("Hello WORLD", casefold=False) == "Hello WORLD"

    def test_unicode_fold_applied(self):
        assert normalize_text("\u201cHi\u201d") == '"hi"'

    def test_nbsp_collapsed(self):
        assert normalize_text("a\u00a0\u00a0b") == "a b"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   \t\n ") == ""


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_basic(self):
        assert strip_punctuation("Hello, World!") == "Hello World"

    def test_numbers_kept(self):
        assert strip_punctuation("$1,234.56") == "123456"

    def test_underscore_kept(self):
        # \w includes underscore
        assert strip_punctuation("a_b") == "a_b"

    def test_all_punctuation(self):
        assert strip_punctuation("!!!???") == ""

    def test_preserves_whitespace(self):
        assert strip_punctuation("a  b") == "a  b"


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty(self):
        assert parse_number("") is None

    def test_whitespace(self):
        assert parse_number("   ") is None

    def test_simple_int(self):
        assert parse_number("42") == Decimal("42")

    def test_simple_float(self):
        assert parse_number("3.14") == Decimal("3.14")

    def test_us_thousands(self):
        assert parse_number("1,234.56") == Decimal("1234.56")

    def test_us_multiple_thousands(self):
        assert parse_number("1,234,567.89") == Decimal("1234567.89")

    def test_european_thousands_decimal(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_european_multi(self):
        assert parse_number("1.234.567,89") == Decimal("1234567.89")

    def test_comma_two_digit_decimal(self):
        # "12,34" -> decimal, tail not 3 digits
        assert parse_number("12,34") == Decimal("12.34")

    def test_comma_three_digit_thousands(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_currency_dollar(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")

    def test_currency_euro(self):
        assert parse_number("€100") == Decimal("100")

    def test_currency_code(self):
        assert parse_number("USD 100") == Decimal("100")

    def test_currency_code_case_insensitive(self):
        assert parse_number("usd 100") == Decimal("100")

    def test_currency_code_no_word_boundary_issue(self):
        assert parse_number("100 EUR") == Decimal("100")

    def test_negative(self):
        assert parse_number("-5") == Decimal("-5")

    def test_plus_sign(self):
        assert parse_number("+5") == Decimal("5")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_accounting_with_currency(self):
        assert parse_number("($1,234.00)") == Decimal("-1234.00")

    def test_accounting_with_sign(self):
        # Parens then leading '-' should remain negative
        assert parse_number("(-5)") == Decimal("-5")

    def test_no_digits(self):
        assert parse_number("abc") is None

    def test_only_currency(self):
        assert parse_number("$") is None

    def test_multiple_dots_no_digit_between(self):
        # "1.2.3" -> last_dot > last_comma branch removes commas only, then
        # "1.2.3" fails fullmatch -> None
        assert parse_number("1.2.3") is None

    def test_fullwidth_digits_nfkc(self):
        assert parse_number("\uff11\uff12\uff13") == Decimal("123")

    def test_already_decimal_input(self):
        # str(Decimal) path
        assert parse_number(Decimal("10.5")) == Decimal("10.5")

    def test_tuple_parens_only_negative_whitespace(self):
        assert parse_number("( 42 )") == Decimal("-42")

    def test_leading_currency_space(self):
        assert parse_number("$ 1,000") == Decimal("1000")

    def test_dot_only(self):
        assert parse_number(".") is None

    def test_comma_only(self):
        assert parse_number(",") is None

    def test_return_type(self):
        assert isinstance(parse_number("1"), Decimal)


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_exact_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.5")) is True

    def test_exact_not_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False

    def test_trailing_zeros_equal(self):
        assert numbers_equal(Decimal("1.50"), Decimal("1.5")) is True

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6"), Decimal("0.1")) is True

    def test_tolerance_boundary_inclusive(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.1"), Decimal("0.1")) is True

    def test_tolerance_exceeded(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.2"), Decimal("0.1")) is False

    def test_tolerance_negative_diff(self):
        assert numbers_equal(Decimal("2"), Decimal("1"), Decimal("1")) is True


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None

    def test_whitespace(self):
        assert parse_date("   ") is None

    def test_iso_ymd(self):
        assert parse_date("2026-03-04") == _dt.date(2026, 3, 4)

    def test_ymd_slash(self):
        assert parse_date("2026/03/04") == _dt.date(2026, 3, 4)

    def test_ymd_dot(self):
        assert parse_date("2026.03.04") == _dt.date(2026, 3, 4)

    def test_unambiguous_mdy(self):
        # second > 12 and first <= 12
        assert parse_date("05/13/2026") == _dt.date(2026, 5, 13)

    def test_unambiguous_dmy(self):
        # first > 12 and second <= 12
        assert parse_date("13/05/2026") == _dt.date(2026, 5, 13)

    def test_ambiguous_returns_none(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == _dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == _dt.date(2026, 3, 4)

    def test_day_month_year(self):
        assert parse_date("4 March 2026") == _dt.date(2026, 3, 4)

    def test_day_month_year_ordinal(self):
        assert parse_date("4th March 2026") == _dt.date(2026, 3, 4)

    def test_day_month_abbrev(self):
        assert parse_date("4 Mar 2026") == _dt.date(2026, 3, 4)

    def test_day_month_with_comma(self):
        assert parse_date("4 March, 2026") == _dt.date(2026, 3, 4)

    def test_month_day_year(self):
        assert parse_date("March 4, 2026") == _dt.date(2026, 3, 4)

    def test_month_day_abbrev(self):
        assert parse_date("Mar 4 2026") == _dt.date(2026, 3, 4)

    def test_sept_abbrev(self):
        assert parse_date("Sept 4 2026") == _dt.date(2026, 9, 4)

    def test_invalid_day_in_month(self):
        assert parse_date("31 February 2026") is None

    def test_invalid_month(self):
        assert parse_date("2026-13-01") is None

    def test_no_match(self):
        assert parse_date("not a date") is None

    def test_leading_trailing_text(self):
        assert parse_date("Due on 2026-03-04 please") == _dt.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("hello world", "hello world") == 1.0

    def test_order_insensitive(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial(self):
        # {a,b} vs {a,b,c} -> 2/3
        assert token_set_ratio("a b", "a b c") == pytest.approx(2 / 3)

    def test_disjoint(self):
        assert token_set_ratio("a b", "c d") == 0.0

    def test_both_empty(self):
        assert token_set_ratio("", "") == 0.0

    def test_one_empty(self):
        assert token_set_ratio("a", "") == 0.0

    def test_punctuation_stripped_per_token(self):
        assert token_set_ratio("hello,", "hello") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("Hello", "hello") == 1.0

    def test_duplicates_collapse(self):
        assert token_set_ratio("a a b", "a b") == 1.0

    def test_range(self):
        v = token_set_ratio("a b c", "a c")
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_first_empty(self):
        assert levenshtein("", "abc") == 3

    def test_second_empty(self):
        assert levenshtein("abc", "") == 3

    def test_insertion(self):
        assert levenshtein("ab", "abc") == 1

    def test_deletion(self):
        assert levenshtein("abc", "ab") == 1

    def test_substitution(self):
        assert levenshtein("abc", "abd") == 1

    def test_kitten_sitting(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_classic_flaw(self):
        assert levenshtein("flaw", "lawn") == 2

    def test_symmetric(self):
        assert levenshtein("abc", "xyz") == levenshtein("xyz", "abc")


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("abc", "") == 0.0

    def test_other_empty(self):
        assert similarity("", "abc") == 0.0

    def test_one_char_diff(self):
        # "cat" -> "bat": distance 1 / max len 3
        assert similarity("cat", "bat") == pytest.approx(2 / 3)

    def test_case_and_whitespace_normalized(self):
        assert similarity("  Hello  ", "hello") == 1.0

    def test_punctuation_ignored_via_normalize(self):
        # only unicode folding + whitespace are normalized, not punctuation, so
        # "hello," vs "hello" differs by length 6 -> distance 1
        assert similarity("hello,", "hello") == pytest.approx(5 / 6)


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_basic_single(self):
        results = list(iter_numbers("total is 42"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "42"
        assert start == 9
        assert end == 11
        assert val == Decimal("42")

    def test_multiple(self):
        results = list(iter_numbers("a 1 b 2 c 3"))
        values = [r[3] for r in results]
        assert values == [Decimal("1"), Decimal("2"), Decimal("3")]

    def test_currency_included(self):
        results = list(iter_numbers("cost $4,500.00 today"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert val == Decimal("4500.00")
        assert raw == "$4,500.00"
        assert end - start == len(raw)

    def test_start_points_at_non_whitespace(self):
        text = "  42"
        results = list(iter_numbers(text))
        assert len(results) == 1
        _, start, _, _ = results[0]
        assert text[start] == "4"

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_empty(self):
        assert list(iter_numbers("")) == []

    def test_spans_consistent_with_raw(self):
        text = "values: 1,234.56 and -7"
        for raw, start, end, _ in iter_numbers(text):
            assert text[start:end] == raw

    def test_unicode_fold_nbsp(self):
        results = list(iter_numbers("1\u00a0000"))
        # folded to "1 000" which regex matches as thousands
        assert len(results) >= 1
        assert any(r[3] == Decimal("1000") for r in results)

    def test_negative_sign(self):
        results = list(iter_numbers("x -5 y"))
        vals = [r[3] for r in results]
        assert Decimal("-5") in vals


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_basic(self):
        results = list(iter_dates("on 2026-03-04 we met"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "2026-03-04"
        assert start == 3
        assert end == 13
        assert val == _dt.date(2026, 3, 4)

    def test_multiple_formats(self):
        text = "2026-03-04 and 4 March 2026"
        results = list(iter_dates(text))
        assert len(results) == 2
        dates = sorted(r[3] for r in results)
        assert dates == [_dt.date(2026, 3, 4), _dt.date(2026, 3, 4)]

    def test_ambiguous_skipped_by_default(self):
        assert list(iter_dates("03/04/2026")) == []

    def test_ambiguous_with_day_first(self):
        results = list(iter_dates("03/04/2026", day_first=True))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 4, 3)

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_empty(self):
        assert list(iter_dates("")) == []

    def test_spans_consistent_with_raw(self):
        text = "depart 2026-12-25 arrive Mar 4 2027"
        for raw, start, end, _ in iter_dates(text):
            assert text[start:end] == raw

    def test_invalid_date_skipped(self):
        # 31 February matches pattern but fails in date()
        assert list(iter_dates("31 February 2026")) == []

    def test_unicode_fold(self):
        # curly quotes shouldn't interfere; ensure dash folding works
        results = list(iter_dates("date\u20132026-03-04"))
        assert len(results) == 1
        assert results[0][3] == _dt.date(2026, 3, 4)
