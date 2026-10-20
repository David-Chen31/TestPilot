"""Unit tests for llmground.normalize."""

from __future__ import annotations

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

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("\u2018a\u2019", "'a'"),
        ("\u201cb\u201d", '"b"'),
        ("a\u2010b\u2011c\u2012d\u2013e\u2014f\u2015g\u2212h", "a-b-c-d-e-f-g-h"),
        ("a\u00a0b", "a b"),
        ("\u200b\u200c\u200d\ufeff", ""),
        ("wait\u2026", "wait..."),
    ],
)
def test_fold_unicode_lookalikes(raw, expected):
    assert fold_unicode(raw) == expected


def test_fold_unicode_nfkc_fullwidth():
    assert fold_unicode("\uff21\uff22\uff23") == "ABC"


def test_fold_unicode_ligature():
    assert fold_unicode("\ufb01") == "fi"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_dedup_whitespace_and_case():
    assert normalize_text("  Hello   World\t\n") == "hello world"


def test_normalize_text_preserves_case():
    assert normalize_text("  Hello   World  ", casefold=False) == "Hello World"


def test_normalize_text_folds_unicode():
    assert normalize_text("\u201cHi\u201d") == '"hi"'


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

def test_strip_punctuation_removes_punct():
    assert strip_punctuation("Hello, World! (v2.0)") == "Hello World v20"


def test_strip_punctuation_keeps_underscore():
    assert strip_punctuation("a_b-c") == "a_bc"


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$4,500.00", Decimal("4500.00")),
        ("4,500", Decimal("4500")),
        ("1.234,56", Decimal("1234.56")),
        ("12,34", Decimal("12.34")),
        ("1,234.56", Decimal("1234.56")),
        ("(1,234.56)", Decimal("-1234.56")),
        ("-42", Decimal("-42")),
        ("+42", Decimal("42")),
        ("USD 1,000", Decimal("1000")),
        ("1 000 000", Decimal("1000000")),
        ("\u20ac1.234,56", Decimal("1234.56")),
        ("4500.0", Decimal("4500.0")),
    ],
)
def test_parse_number_valid(raw, expected):
    result = parse_number(raw)
    assert result == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "abc", "USD", "()", "1.2.3", "?", "--5"],
)
def test_parse_number_invalid(raw):
    assert parse_number(raw) is None


def test_parse_number_accepts_non_string():
    assert parse_number(42) == Decimal("42")


def test_parse_number_european_thousands():
    assert parse_number("1.234.567") == Decimal("1234567")


def test_parse_number_us_thousands():
    assert parse_number("1,234,567") == Decimal("1234567")


def test_parse_number_negative_with_currency():
    assert parse_number("-$1,000") == Decimal("-1000")


def test_parse_number_accounting_with_currency():
    assert parse_number("($1,000)") == Decimal("-1000")


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

def test_numbers_equal_none_returns_false():
    assert numbers_equal(None, Decimal("1")) is False
    assert numbers_equal(Decimal("1"), None) is False
    assert numbers_equal(None, None) is False


def test_numbers_equal_exact():
    assert numbers_equal(Decimal("4500.0"), Decimal("4500.00")) is True
    assert numbers_equal(Decimal("1"), Decimal("2")) is False


def test_numbers_equal_tolerance():
    assert numbers_equal(Decimal("100.0"), Decimal("100.5"), tolerance=Decimal("0.5")) is True
    assert numbers_equal(Decimal("100.0"), Decimal("100.5"), tolerance=Decimal("0.4")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

def test_parse_date_iso():
    assert parse_date("2026-03-04") == dt.date(2026, 3, 4)


def test_parse_date_slash_ymd():
    assert parse_date("2026/03/04") == dt.date(2026, 3, 4)


def test_parse_date_unambiguous_dmy():
    # 13/04/2026 -> first > 12 means day-first
    assert parse_date("13/04/2026") == dt.date(2026, 4, 13)


def test_parse_date_unambiguous_mdy():
    # 04/13/2026 -> second > 12 means month-first
    assert parse_date("04/13/2026") == dt.date(2026, 4, 13)


def test_parse_date_ambiguous_returns_none():
    assert parse_date("03/04/2026") is None


def test_parse_date_ambiguous_day_first_true():
    assert parse_date("03/04/2026", day_first=True) == dt.date(2026, 4, 3)


def test_parse_date_ambiguous_day_first_false():
    assert parse_date("03/04/2026", day_first=False) == dt.date(2026, 3, 4)


def test_parse_date_dmonthy():
    assert parse_date("4 March 2026") == dt.date(2026, 3, 4)


def test_parse_date_dmonthy_with_suffix_and_abbrev():
    assert parse_date("4th Mar 2026") == dt.date(2026, 3, 4)


def test_parse_date_monthdy():
    assert parse_date("March 4, 2026") == dt.date(2026, 3, 4)


def test_parse_date_monthdy_abbrev_dot():
    assert parse_date("Sept. 4 2026") == dt.date(2026, 9, 4)


def test_parse_date_invalid_value_returns_none():
    assert parse_date("31 February 2026") is None


def test_parse_date_no_match_returns_none():
    assert parse_date("not a date") is None


def test_parse_date_none_input():
    assert parse_date(None) is None


def test_parse_date_empty():
    assert parse_date("") is None


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

def test_token_set_ratio_reordered_equal():
    assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0


def test_token_set_ratio_partial():
    # {a, b, c} vs {a, b}: intersection 2, union 3
    assert token_set_ratio("a b c", "a b") == pytest.approx(2 / 3)


def test_token_set_ratio_empty():
    assert token_set_ratio("", "abc") == 0.0
    assert token_set_ratio("abc", "") == 0.0
    assert token_set_ratio("", "") == 0.0


def test_token_set_ratio_case_and_punct_insensitive():
    assert token_set_ratio("Hello, World!", "world hello") == 1.0


def test_token_set_ratio_disjoint():
    assert token_set_ratio("abc", "xyz") == 0.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("a", "b", 1),
    ],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

def test_similarity_identical():
    assert similarity("Hello World", "hello   world") == 1.0


def test_similarity_both_empty():
    assert similarity("", "") == 1.0


def test_similarity_one_empty():
    assert similarity("", "abc") == 0.0
    assert similarity("abc", "") == 0.0


def test_similarity_typo():
    # "hello" vs "helo": dist 1, longest 5
    assert similarity("hello", "helo") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

def test_iter_numbers_basic():
    results = list(iter_numbers("total is $4,500.00 today"))
    assert len(results) == 1
    raw, start, end, val = results[0]
    assert raw == "$4,500.00"
    assert val == Decimal("4500.00")
    assert "total is $4,500.00 today"[start:end] == "$4,500.00"


def test_iter_numbers_multiple_and_spans():
    text = "a 10 b 20 c"
    results = list(iter_numbers(text))
    assert [r[3] for r in results] == [Decimal("10"), Decimal("20")]
    for raw, start, end, _ in results:
        assert text[start:end] == raw


def test_iter_numbers_span_excludes_leading_space():
    text = "10 and 20"
    for raw, start, end, _ in iter_numbers(text):
        assert not raw.startswith(" ")
        assert text[start:end] == raw


def test_iter_numbers_no_numbers():
    assert list(iter_numbers("no digits here")) == []


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

def test_iter_dates_basic():
    text = "on 2026-03-04 we ship"
    results = list(iter_dates(text))
    assert len(results) == 1
    raw, start, end, val = results[0]
    assert raw == "2026-03-04"
    assert val == dt.date(2026, 3, 4)
    assert text[start:end] == "2026-03-04"


def test_iter_dates_multiple():
    text = "from 4 March 2026 to March 5, 2026"
    results = list(iter_dates(text))
    dates = [r[3] for r in results]
    assert dt.date(2026, 3, 4) in dates
    assert dt.date(2026, 3, 5) in dates


def test_iter_dates_day_first_kwarg():
    text = "deadline 03/04/2026"
    results = list(iter_dates(text, day_first=True))
    assert len(results) == 1
    assert results[0][3] == dt.date(2026, 4, 3)


def test_iter_dates_ambiguous_default_skips():
    text = "deadline 03/04/2026"
    assert list(iter_dates(text)) == []


def test_iter_dates_none_found():
    assert list(iter_dates("no dates here")) == []


def test_iter_dates_span_matches_text():
    text = "due 4 March 2026!"
    for raw, start, end, _ in iter_dates(text):
        assert text[start:end] == raw
