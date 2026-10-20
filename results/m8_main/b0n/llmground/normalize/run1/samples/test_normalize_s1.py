"""Tests for llmground.normalize."""

import datetime as dt
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
        assert fold_unicode("\u2018a\u2019\u201cb\u201d") == "'a'\"b\""

    def test_dashes(self):
        assert fold_unicode("a\u2013b\u2014c\u2212d") == "a-b-c-d"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"

    def test_zero_width(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_ligature(self):
        assert fold_unicode("\ufb01") == "fi"

    def test_plain_text_unchanged(self):
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
        assert normalize_text("HeLLo") == "hello"

    def test_casefold_disabled(self):
        assert normalize_text("HeLLo", casefold=False) == "HeLLo"

    def test_folds_unicode(self):
        assert normalize_text("a\u2013b") == "a-b"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_casefold_german(self):
        assert normalize_text("STRASSE") == "strasse"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_punctuation(self):
        assert strip_punctuation("hello, world!") == "hello world"

    def test_keeps_underscore_and_alnum(self):
        assert strip_punctuation("a_b-c") == "a_bc"

    def test_keeps_whitespace(self):
        assert strip_punctuation("a b") == "a b"

    def test_empty(self):
        assert strip_punctuation("") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_int(self):
        assert parse_number("42") == Decimal("42")

    def test_negative(self):
        assert parse_number("-42") == Decimal("-42")

    def test_plus_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_us_thousands(self):
        assert parse_number("4,500.00") == Decimal("4500.00")

    def test_european_thousands(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_currency_symbol(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")
        assert parse_number("\u20ac100") == Decimal("100")

    def test_currency_code(self):
        assert parse_number("USD 100") == Decimal("100")
        assert parse_number("100 EUR") == Decimal("100")

    def test_plain_decimal(self):
        assert parse_number("3.14") == Decimal("3.14")

    def test_european_comma_decimal_only(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_thousands_comma_only(self):
        assert parse_number("1,234") == Decimal("1234")

    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty(self):
        assert parse_number("") is None
        assert parse_number("   ") is None

    def test_no_digits(self):
        assert parse_number("abc") is None

    def test_only_currency(self):
        assert parse_number("$") is None

    def test_strips_spaces(self):
        assert parse_number("1 234") == Decimal("1234")


class TestNumbersEqual:
    def test_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("1.0")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1"), Decimal("2")) is False

    def test_one_none(self):
        assert numbers_equal(None, Decimal("1")) is False
        assert numbers_equal(Decimal("1"), None) is False
        assert numbers_equal(None, None) is False

    def test_tolerance(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.05"), Decimal("0.1")) is True

    def test_tolerance_exceeded(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.5"), Decimal("0.1")) is False

    def test_tolerance_zero(self):
        assert numbers_equal(Decimal("1"), Decimal("2"), Decimal("0")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_ymd(self):
        assert parse_date("2026-03-04") == dt.date(2026, 3, 4)

    def test_ymd_slash(self):
        assert parse_date("2026/03/04") == dt.date(2026, 3, 4)

    def test_unambiguous_dmy(self):
        assert parse_date("25/03/2026") == dt.date(2026, 3, 25)

    def test_unambiguous_mdy(self):
        assert parse_date("03/25/2026") == dt.date(2026, 3, 25)

    def test_ambiguous_returns_none(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first(self):
        assert parse_date("03/04/2026", day_first=True) == dt.date(2026, 4, 3)

    def test_ambiguous_month_first(self):
        assert parse_date("03/04/2026", day_first=False) == dt.date(2026, 3, 4)

    def test_dmonthy(self):
        assert parse_date("4 March 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_ordinal(self):
        assert parse_date("4th March 2026") == dt.date(2026, 3, 4)

    def test_monthdy(self):
        assert parse_date("March 4, 2026") == dt.date(2026, 3, 4)

    def test_month_abbrev(self):
        assert parse_date("Mar 4, 2026") == dt.date(2026, 3, 4)

    def test_invalid_date(self):
        assert parse_date("31 February 2026") is None

    def test_none(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None

    def test_no_date(self):
        assert parse_date("hello world") is None


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("hello world", "hello world") == 1.0

    def test_order_insensitive(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial(self):
        # ta={a,b}, tb={a,c}; intersection={a}, union={a,b,c}
        assert token_set_ratio("a b", "a c") == pytest.approx(1 / 3)

    def test_disjoint(self):
        assert token_set_ratio("foo", "bar") == 0.0

    def test_empty(self):
        assert token_set_ratio("", "foo") == 0.0
        assert token_set_ratio("foo", "") == 0.0
        assert token_set_ratio("", "") == 0.0

    def test_punctuation_ignored(self):
        assert token_set_ratio("hello, world!", "hello world") == 1.0

    def test_dup_tokens(self):
        assert token_set_ratio("a a b", "a b") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_equal(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_first(self):
        assert levenshtein("", "abc") == 3

    def test_empty_second(self):
        assert levenshtein("abc", "") == 3

    def test_substitution(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_insertion(self):
        assert levenshtein("abc", "abcd") == 1

    def test_deletion(self):
        assert levenshtein("abcd", "abc") == 1

    def test_both_empty(self):
        assert levenshtein("", "") == 0


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("", "abc") == 0.0
        assert similarity("abc", "") == 0.0

    def test_typo(self):
        # levenshtein=1 over length 5
        assert similarity("hello", "hallo") == pytest.approx(1 - 1 / 5)

    def test_normalizes_case_and_ws(self):
        assert similarity("Hello  World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_basic(self):
        results = list(iter_numbers("Value is 4,500.00 today"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert raw == "4,500.00"
        assert val == Decimal("4500.00")
        assert "Value is 4,500.00 today"[start:end] == "4,500.00"

    def test_multiple(self):
        results = list(iter_numbers("5 apples and 3 oranges"))
        assert [r[0] for r in results] == ["5", "3"]
        assert [r[3] for r in results] == [Decimal("5"), Decimal("3")]

    def test_span_excludes_leading_space(self):
        text = "cost:  123"
        results = list(iter_numbers(text))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert not raw.startswith(" ")
        assert text[start:end] == raw

    def test_currency(self):
        results = list(iter_numbers("only $99 here"))
        assert len(results) == 1
        assert results[0][3] == Decimal("99")

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_unicode_folding(self):
        # full-width digits
        results = list(iter_numbers("price \uff14\uff12"))
        assert len(results) == 1
        assert results[0][3] == Decimal("42")


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_basic(self):
        results = list(iter_dates("On 2026-03-04 we met"))
        assert len(results) == 1
        raw, start, end, val = results[0]
        assert val == dt.date(2026, 3, 4)
        assert "On 2026-03-04 we met"[start:end] == raw

    def test_multiple(self):
        results = list(iter_dates("2026-03-04 and 4 March 2026"))
        assert len(results) == 2
        assert results[0][3] == dt.date(2026, 3, 4)
        assert results[1][3] == dt.date(2026, 3, 4)

    def test_ambiguous_skipped(self):
        # ambiguous dmy/mdy without hint -> parse_date returns None -> not yielded
        results = list(iter_dates("on 03/04/2026"))
        assert results == []

    def test_ambiguous_day_first(self):
        results = list(iter_dates("on 03/04/2026", day_first=True))
        assert len(results) == 1
        assert results[0][3] == dt.date(2026, 4, 3)

    def test_no_dates(self):
        assert list(iter_dates("nothing here")) == []

    def test_invalid_date_skipped(self):
        assert list(iter_dates("31 February 2026")) == []
