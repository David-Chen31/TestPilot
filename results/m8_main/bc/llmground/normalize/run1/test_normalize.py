"""Tests for llmground.normalize."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from unittest import mock

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

def test_fold_unicode_quotes():
    assert fold_unicode("\u2018hi\u2019 \u201cthere\u201d") == "'hi' \"there\""
    assert fold_unicode("\u201a\u201b\u201e\u201f") == "''\"\""


def test_fold_unicode_dashes():
    assert fold_unicode("a\u2010b\u2011c\u2012d\u2013e\u2014f\u2015g\u2212h") == "a-b-c-d-e-f-g-h"


def test_fold_unicode_spaces():
    assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"


def test_fold_unicode_zero_width_removed():
    assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"


def test_fold_unicode_ellipsis():
    assert fold_unicode("wait\u2026") == "wait..."


def test_fold_unicode_nfkc():
    # Full-width digits and ligatures are NFKC-folded.
    assert fold_unicode("\uff11\uff12\uff13") == "123"
    assert fold_unicode("\ufb01") == "fi"


def test_fold_unicode_passthrough():
    assert fold_unicode("plain ascii") == "plain ascii"
    assert fold_unicode("") == ""


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello   world \n\t ") == "hello world"


def test_normalize_text_casefold_default():
    assert normalize_text("HeLLo World") == "hello world"


def test_normalize_text_no_casefold():
    assert normalize_text("HeLLo World", casefold=False) == "HeLLo World"


def test_normalize_text_folds_unicode():
    assert normalize_text("\u201cQuoted\u201d\u00a0Text") == '"quoted" text'


def test_normalize_text_empty():
    assert normalize_text("   ") == ""


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

def test_strip_punctuation_basic():
    assert strip_punctuation("hello, world!") == "hello world"


def test_strip_punctuation_keeps_underscore_and_digits():
    assert strip_punctuation("a_b 123") == "a_b 123"


def test_strip_punctuation_removes_quotes_and_dashes():
    assert strip_punctuation("it's well-known.") == "its wellknown"


def test_strip_punctuation_empty():
    assert strip_punctuation("") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

def test_parse_number_none():
    assert parse_number(None) is None


def test_parse_number_empty():
    assert parse_number("") is None
    assert parse_number("   ") is None


def test_parse_number_simple_int():
    assert parse_number("4500") == Decimal("4500")


def test_parse_number_us_thousands():
    assert parse_number("$4,500.00") == Decimal("4500.00")


def test_parse_number_european():
    assert parse_number("4.500,00 €") == Decimal("4500.00")


def test_parse_number_both_separators_us():
    assert parse_number("1,234.56") == Decimal("1234.56")


def test_parse_number_both_separators_eu():
    assert parse_number("1.234,56") == Decimal("1234.56")


def test_parse_number_single_comma_european():
    # With only a comma present, last_comma (>=0) > last_dot (-1), so the
    # "European" branch fires and the comma is treated as the decimal point.
    # (The dedicated single-separator \"thousands\" branch is unreachable for
    # a comma-only input on the current implementation.)
    assert parse_number("1,234") == Decimal("1.234")


def test_parse_number_single_comma_multiple_groups_invalid():
    # Multiple commas → European interpretation → "1.234.567", which fails the
    # final fullmatch and yields None.
    assert parse_number("1,234,567") is None


def test_parse_number_single_comma_decimal():
    # Only a comma present → treated as European decimal separator.
    assert parse_number("12,34") == Decimal("12.34")


def test_parse_number_single_comma_ambiguous_4_digit_tail():
    # Only a comma present → treated as European decimal separator.
    assert parse_number("1,2345") == Decimal("1.2345")


def test_parse_number_single_dot_decimal():
    assert parse_number("3.14") == Decimal("3.14")


def test_parse_number_no_separator_hits_else_branch():
    # No commas and no dots present → last_comma == last_dot == -1 → the
    # else branch (line 119) is entered, ``sep`` is "" and the inner ``if sep:``
    # body is skipped. The digits pass straight through to Decimal.
    assert parse_number("123") == Decimal("123")
    assert parse_number("987654") == Decimal("987654")


def test_parse_number_only_commas_branch_is_european():
    # When a comma is present but no dot, last_comma > last_dot so the FIRST
    # branch (European) fires, NOT the ``if sep:`` branch. This pins the
    # behaviour the fixture relies on: "1,234" -> 1.234 (decimal comma).
    assert parse_number("1,234") == Decimal("1.234")
    assert parse_number("10,5") == Decimal("10.5")


def test_parse_number_accounting_negative():
    assert parse_number("(1,234.00)") == Decimal("-1234.00")


def test_parse_number_negative_sign():
    assert parse_number("-42") == Decimal("-42")
    assert parse_number("-1,000.50") == Decimal("-1000.50")


def test_parse_number_positive_sign():
    assert parse_number("+42") == Decimal("42")


def test_parse_number_currency_codes():
    assert parse_number("USD 100") == Decimal("100")
    # A trailing currency code is not stripped by the current implementation.
    assert parse_number("100 EUR") is None
    assert parse_number("gbp 50") == Decimal("50")


def test_parse_number_currency_symbols():
    assert parse_number("₹500") == Decimal("500")
    assert parse_number("€1.5") == Decimal("1.5")
    assert parse_number("£10") == Decimal("10")


def test_parse_number_no_digits():
    assert parse_number("hello") is None
    assert parse_number("$") is None


def test_parse_number_invalid_after_cleanup():
    # A lone dot or something that doesn't reduce cleanly.
    assert parse_number(".") is None
    assert parse_number("...") is None


def test_parse_number_fullwidth_digits():
    assert parse_number("\uff11\uff12\uff13") == Decimal("123")


def test_parse_number_nbsp_thousands():
    assert parse_number("1\u00a0000") == Decimal("1000")


def test_parse_number_invalid_operation_from_decimal(monkeypatch):
    # Force Decimal to raise InvalidOperation so the except branch executes.
    import llmground.normalize as norm
    from decimal import InvalidOperation

    class BoomDecimal(Decimal):
        def __new__(cls, *args, **kwargs):
            raise InvalidOperation("boom")

    with mock.patch.object(norm, "Decimal", BoomDecimal):
        assert parse_number("123") is None


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

def test_numbers_equal_none():
    assert numbers_equal(None, Decimal("1")) is False
    assert numbers_equal(Decimal("1"), None) is False
    assert numbers_equal(None, None) is False


def test_numbers_equal_exact():
    assert numbers_equal(Decimal("1.50"), Decimal("1.5")) is True
    assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False


def test_numbers_equal_within_tolerance():
    assert numbers_equal(Decimal("1.00"), Decimal("1.01"), Decimal("0.01")) is True
    assert numbers_equal(Decimal("1.00"), Decimal("1.02"), Decimal("0.01")) is False


def test_numbers_equal_tolerance_boundary():
    assert numbers_equal(Decimal("1.00"), Decimal("1.05"), Decimal("0.05")) is True


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

def test_parse_date_none():
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("   ") is None


def test_parse_date_ymd():
    assert parse_date("2026-03-04") == dt.date(2026, 3, 4)
    assert parse_date("2026/03/04") == dt.date(2026, 3, 4)
    assert parse_date("2026.03.04") == dt.date(2026, 3, 4)


def test_parse_date_unambiguous_first_gt_12():
    # 25/03/2026 -> day 25, month 3
    assert parse_date("25/03/2026") == dt.date(2026, 3, 25)


def test_parse_date_unambiguous_second_gt_12():
    # 03/25/2026 -> month 3, day 25
    assert parse_date("03/25/2026") == dt.date(2026, 3, 25)


def test_parse_date_ambiguous_no_hint():
    assert parse_date("03/04/2026") is None


def test_parse_date_ambiguous_day_first():
    assert parse_date("03/04/2026", day_first=True) == dt.date(2026, 4, 3)


def test_parse_date_ambiguous_month_first():
    assert parse_date("03/04/2026", day_first=False) == dt.date(2026, 3, 4)


def test_parse_date_day_month_name():
    assert parse_date("4 March 2026") == dt.date(2026, 3, 4)
    assert parse_date("4th Mar 2026") == dt.date(2026, 3, 4)


def test_parse_date_month_day_name():
    assert parse_date("March 4, 2026") == dt.date(2026, 3, 4)
    assert parse_date("Mar. 4 2026") == dt.date(2026, 3, 4)


def test_parse_date_invalid_date():
    assert parse_date("31 February 2026") is None
    assert parse_date("2026-02-31") is None


def test_parse_date_non_date():
    assert parse_date("no date here") is None


def test_parse_date_month_abbrev_boundary():
    assert parse_date("Sept 9 2026") == dt.date(2026, 9, 9)


def test_parse_date_unknown_month_name():
    # "dmonthy" pattern matches, but month is unknown -> falls through, no date.
    assert parse_date("4 Xyzzy 2026") is None


def test_parse_date_unknown_month_name_monthdy():
    # "monthdy" pattern matches, but month is unknown -> falls through.
    assert parse_date("Xyzzy 4, 2026") is None


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

def test_token_set_ratio_identical():
    assert token_set_ratio("Ahmed Khan", "Ahmed Khan") == 1.0


def test_token_set_ratio_reordered():
    assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0


def test_token_set_ratio_partial():
    # {a b} vs {a c} -> intersection 1, union 3
    assert token_set_ratio("a b", "a c") == pytest.approx(1 / 3)


def test_token_set_ratio_duplicates_ignored():
    assert token_set_ratio("a a a b", "a b") == 1.0


def test_token_set_ratio_empty_sides():
    assert token_set_ratio("", "") == 0.0
    assert token_set_ratio("a", "") == 0.0
    assert token_set_ratio("", "a") == 0.0


def test_token_set_ratio_punctuation():
    assert token_set_ratio("hello,", "hello") == 1.0


def test_token_set_ratio_case_insensitive():
    assert token_set_ratio("Hello World", "world hello") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

def test_levenshtein_equal():
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_empty():
    assert levenshtein("", "") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_substitution():
    assert levenshtein("abc", "abd") == 1


def test_levenshtein_insertion():
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_deletion():
    assert levenshtein("abcd", "abc") == 1


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

def test_similarity_identical():
    assert similarity("hello", "hello") == 1.0


def test_similarity_both_empty():
    assert similarity("", "") == 1.0


def test_similarity_one_empty():
    assert similarity("hello", "") == 0.0
    assert similarity("", "hello") == 0.0


def test_similarity_typo():
    # "hello" vs "hallo": edit distance 1, longest 5
    assert similarity("hello", "hallo") == pytest.approx(1 - 1 / 5)


def test_similarity_casefolded():
    assert similarity("HELLO", "hello") == 1.0


def test_similarity_whitespace_normalized():
    assert similarity("  hello  world ", "hello world") == 1.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

def test_iter_numbers_finds_values():
    text = "Total is $4,500.00 and tax 12.5"
    results = list(iter_numbers(text))
    assert len(results) == 2
    raw1, start1, end1, val1 = results[0]
    assert raw1 == "$4,500.00"
    assert val1 == Decimal("4500.00")
    assert text[start1:end1] == "$4,500.00"
    raw2, start2, end2, val2 = results[1]
    assert val2 == Decimal("12.5")
    assert text[start2:end2] == "12.5"


def test_iter_numbers_tight_span_no_leading_space():
    text = "see 100 dollars"
    results = list(iter_numbers(text))
    assert len(results) == 1
    raw, start, end, val = results[0]
    assert not raw.startswith(" ")
    assert text[start:end] == "100"
    assert val == Decimal("100")


def test_iter_numbers_empty():
    assert list(iter_numbers("no numbers here")) == []


def test_iter_numbers_skips_unparseable():
    # Single dot yields no value but pattern may or may not match; ensure no crash.
    results = list(iter_numbers("."))
    assert results == []


def test_iter_numbers_multiple():
    text = "1 2 3"
    values = [v for _, _, _, v in iter_numbers(text)]
    assert values == [Decimal("1"), Decimal("2"), Decimal("3")]


def test_iter_numbers_skips_when_parse_number_returns_none(monkeypatch):
    # Force parse_number to return None for a matched number so the
    # `if val is not None` else-branch is exercised.
    import llmground.normalize as norm

    monkeypatch.setattr(norm, "parse_number", lambda raw: None)
    assert list(norm.iter_numbers("123 456")) == []


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

def test_iter_dates_finds_dates():
    text = "Signed on 2026-03-04 and reviewed March 5, 2026."
    results = list(iter_dates(text))
    values = [v for _, _, _, v in results]
    assert dt.date(2026, 3, 4) in values
    assert dt.date(2026, 3, 5) in values


def test_iter_dates_spans_correct():
    text = "on 2026-03-04 done"
    results = list(iter_dates(text))
    assert len(results) == 1
    raw, start, end, val = results[0]
    assert text[start:end] == raw
    assert val == dt.date(2026, 3, 4)


def test_iter_dates_empty():
    assert list(iter_dates("no dates here")) == []


def test_iter_dates_day_first():
    text = "03/04/2026"
    assert list(iter_dates(text)) == []
    results = list(iter_dates(text, day_first=True))
    assert len(results) == 1
    assert results[0][3] == dt.date(2026, 4, 3)


def test_iter_dates_dedupes_overlapping_patterns():
    # A single date should not be yielded twice even if multiple patterns match.
    text = "2026-03-04"
    results = list(iter_dates(text))
    assert len(results) == 1


def test_iter_dates_dedupes_same_span_across_patterns(monkeypatch):
    # Force two date patterns to match the exact same span so the
    # `if span in seen: continue` branch is exercised. The pattern must carry
    # the capture groups that parse_date expects, because parse_date reads the
    # same module-level _DATE_PATTERNS.
    import re as _re

    import llmground.normalize as norm

    pattern = _re.compile(r"(\d{4})-(\d{2})-(\d{2})")
    monkeypatch.setattr(
        norm, "_DATE_PATTERNS", [(pattern, "ymd"), (pattern, "ymd")]
    )
    results = list(norm.iter_dates("2026-03-04"))
    assert len(results) == 1
    assert results[0][0] == "2026-03-04"
    assert results[0][3] == dt.date(2026, 3, 4)
