"""Unit tests for llmground.matchers (matching engine)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from llmground.matchers import (
    Match,
    MatchTier,
    NOT_FOUND,
    Span,
    is_trivial,
    match_date,
    match_number,
    match_text,
)


# --------------------------------------------------------------------------- #
# MatchTier
# --------------------------------------------------------------------------- #


class TestMatchTier:
    def test_values(self):
        assert MatchTier.EXACT.value == "exact"
        assert MatchTier.NORMALIZED.value == "normalized"
        assert MatchTier.SEMANTIC.value == "semantic"
        assert MatchTier.FUZZY.value == "fuzzy"
        assert MatchTier.NONE.value == "none"

    def test_rank_ordering(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_ranks_strictly_descending(self):
        tiers = [
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        ]
        ranks = [t.rank for t in tiers]
        assert ranks == sorted(ranks, reverse=True)
        assert len(set(ranks)) == len(ranks)

    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"


# --------------------------------------------------------------------------- #
# Span
# --------------------------------------------------------------------------- #


class TestSpan:
    def test_fields(self):
        s = Span(3, 7, "abcd")
        assert s.start == 3
        assert s.end == 7
        assert s.text == "abcd"

    def test_to_dict(self):
        s = Span(3, 7, "abcd")
        assert s.to_dict() == {"start": 3, "end": 7, "text": "abcd"}

    def test_to_dict_empty_text(self):
        s = Span(0, 0, "")
        assert s.to_dict() == {"start": 0, "end": 0, "text": ""}


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #


class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.EXACT)
        assert m.tier is MatchTier.EXACT
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_found_property(self):
        assert Match(MatchTier.EXACT).found is True
        assert Match(MatchTier.NORMALIZED).found is True
        assert Match(MatchTier.SEMANTIC).found is True
        assert Match(MatchTier.FUZZY).found is True
        assert Match(MatchTier.NONE).found is False

    def test_to_dict(self):
        m = Match(
            MatchTier.FUZZY,
            0.87654,
            [Span(0, 2, "ab")],
            note="hi",
        )
        d = m.to_dict()
        assert d["tier"] == "fuzzy"
        assert d["score"] == 0.8765
        assert d["spans"] == [{"start": 0, "end": 2, "text": "ab"}]
        assert d["note"] == "hi"

    def test_to_dict_rounds_score(self):
        m = Match(MatchTier.EXACT, 0.99999)
        assert m.to_dict()["score"] == 1.0

    def test_to_dict_empty(self):
        d = Match(MatchTier.NONE).to_dict()
        assert d == {"tier": "none", "score": 0.0, "spans": [], "note": ""}

    def test_to_dict_zero_rounds_to_zero(self):
        m = Match(MatchTier.NONE, 0.00001)
        assert m.to_dict()["score"] == 0.0

    def test_spans_default_not_shared(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []


# --------------------------------------------------------------------------- #
# NOT_FOUND sentinel
# --------------------------------------------------------------------------- #


class TestNotFoundSentinel:
    def test_tier_and_score(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.spans == []

    def test_found_false(self):
        assert NOT_FOUND.found is False


# --------------------------------------------------------------------------- #
# is_trivial
# --------------------------------------------------------------------------- #


class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", " ", "0", "1", "true", "false", "yes", "no", "n/a", "na",
         "none", "null", "-", "TRUE", "  Yes  ", "N/A", "False", "0"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "42", "hello", "acme", "0.5", "y", "n", "not found?",
         "Zero", "1.0"],
    )
    def test_non_trivial_values(self, value):
        assert is_trivial(value) is False

    def test_int_input(self):
        assert is_trivial(1) is True
        assert is_trivial(0) is True
        assert is_trivial(2) is False

    def test_decimal_like_input(self):
        # str(Decimal("1")) == "1"
        assert is_trivial(Decimal("1")) is True
        assert is_trivial(Decimal("2")) is False

    def test_none_input(self):
        # str(None) == "None"; casefold -> "none" -> trivial
        assert is_trivial(None) is True


# --------------------------------------------------------------------------- #
# match_text: trivial / empty
# --------------------------------------------------------------------------- #


class TestMatchTextTrivial:
    @pytest.mark.parametrize("value", ["", "   ", "yes", "NO", "none", "-"])
    def test_trivial_returns_none_tier(self, value):
        m = match_text(value, "some source yes no")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.found is False
        assert m.note == "value too trivial to verify"


# --------------------------------------------------------------------------- #
# match_text: EXACT
# --------------------------------------------------------------------------- #


class TestMatchTextExact:
    def test_simple_exact(self):
        source = "The company is Acme Corp today."
        m = match_text("Acme Corp", source)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        span = m.spans[0]
        assert span.text == "Acme Corp"
        assert source[span.start:span.end] == "Acme Corp"

    def test_multiple_occurrences(self):
        source = "apple then apple again apple"
        m = match_text("apple", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3
        for span in m.spans:
            assert source[span.start:span.end] == "apple"

    def test_occurrence_limit_five(self):
        source = "x " * 10  # no match
        source = "apple " * 10
        m = match_text("apple", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_overlapping_occurrences(self):
        # "aaa" in "aaaaa" -> overlapping matches
        source = "aaaaa"
        m = match_text("aaa", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3
        assert [s.start for s in m.spans] == [0, 1, 2]

    def test_case_sensitive_exact_falls_through(self):
        # Exact search is case-sensitive; falls to normalized.
        m = match_text("Acme", "the acme company")
        assert m.tier is MatchTier.NORMALIZED

    def test_exact_no_note(self):
        m = match_text("Acme", "Acme here")
        assert m.note == ""


# --------------------------------------------------------------------------- #
# match_text: NORMALIZED
# --------------------------------------------------------------------------- #


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        source = "The ACME Corporation"
        m = match_text("acme corporation", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert m.note == "matched after whitespace/case/unicode normalisation"
        assert len(m.spans) == 1

    def test_whitespace_collapsing(self):
        source = "Acme   \n\t  Corp"
        m = match_text("acme corp", source)
        assert m.tier is MatchTier.NORMALIZED
        assert len(m.spans) == 1

    def test_span_points_into_source(self):
        source = "prefix Acme Corp suffix"
        m = match_text("acme corp", source)
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert 0 <= span.start < span.end <= len(source)
        assert source[span.start:span.end] == "Acme Corp"

    def test_span_length_uses_value_length(self):
        source = "xx Hello World yy"
        m = match_text("hello world", source)
        # end = start + len(value); start maps to "Hello"
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert span.end - span.start == len("hello world")

    def test_unicode_folding(self):
        # ligature "ﬁ" folds to "fi"
        source = "the ﬁnal result"
        m = match_text("final", source)
        assert m.tier is MatchTier.NORMALIZED

    def test_punctuation_ignored(self):
        source = "Acme, Inc. is here"
        m = match_text("Acme Inc", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        assert m.spans == []


# --------------------------------------------------------------------------- #
# match_text: FUZZY
# --------------------------------------------------------------------------- #


class TestMatchTextFuzzy:
    def test_fuzzy_ocr_noise(self):
        source = "The Acrne Corporation is big"
        m = match_text("Acme Corporation", source)
        assert m.tier is MatchTier.FUZZY
        assert m.found is True
        assert 0.0 < m.score < 1.0
        assert m.note.startswith("closest text in source is")
        assert len(m.spans) == 1
        span = m.spans[0]
        assert source[span.start:span.end] == span.text

    def test_fuzzy_threshold_respected(self):
        source = "completely unrelated words here"
        m = match_text("Acme Corporation", source, fuzzy_threshold=0.99)
        assert m.tier is MatchTier.NONE

    def test_low_threshold_allows_weak_match(self):
        source = "The Acrne Corporation is big"
        m = match_text("Acme Corporation", source, fuzzy_threshold=0.1)
        assert m.tier is MatchTier.FUZZY

    def test_no_match_returns_none_with_best_score(self):
        source = "completely unrelated words here"
        m = match_text("Acme Corporation", source)
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_token_reorder_match(self):
        # token_set_ratio helps reordered tokens
        source = "Corporation Acme"
        m = match_text("Acme Corporation", source)
        assert m.tier is MatchTier.FUZZY
        assert m.spans

    def test_window_slack_zero(self):
        source = "Acrne Corporation"
        m = match_text("Acme Corporation", source, window_slack=0.0)
        # With slack 0 the window must be ~same length as value.
        assert m.tier in (MatchTier.FUZZY, MatchTier.NONE)


# --------------------------------------------------------------------------- #
# match_text: NONE
# --------------------------------------------------------------------------- #


class TestMatchTextNone:
    def test_not_in_source(self):
        m = match_text("Zebra", "the quick brown fox")
        assert m.tier is MatchTier.NONE
        assert m.found is False
        assert m.spans == []

    def test_empty_source(self):
        m = match_text("Zebra", "")
        assert m.tier is MatchTier.NONE

    def test_value_longer_than_source(self):
        m = match_text("a very long value indeed", "short")
        assert m.tier is MatchTier.NONE


# --------------------------------------------------------------------------- #
# match_number
# --------------------------------------------------------------------------- #


class TestMatchNumberTrivial:
    @pytest.mark.parametrize("value", [1, 0, "1", "0", "yes", "none"])
    def test_trivial_returns_none(self, value):
        m = match_number(value, "anything 1 0 yes")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"
        assert m.found is False

    def test_non_numeric_string(self):
        m = match_number("not-a-number", "source text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"


class TestMatchNumberExact:
    def test_exact_verbatim(self):
        source = "The total is 4500 here."
        m = match_number("4500", source)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].text == "4500"

    def test_exact_int(self):
        source = "count: 42"
        m = match_number(42, source)
        assert m.tier is MatchTier.EXACT
        assert m.spans[0].text == "42"

    def test_exact_multiple(self):
        source = "12 and 12 and 12"
        m = match_number(12, source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3

    def test_exact_limit_five(self):
        source = "7 " * 10
        m = match_number(7, source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5


class TestMatchNumberSemantic:
    def test_currency_formatting(self):
        source = "Total: $4,500.00 payable."
        m = match_number(4500, source)
        # raw "4500" is not literally present -> semantic
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert len(m.spans) >= 1
        assert any("4,500" in s.text for s in m.spans)
        assert m.note == "numeric value present, formatted differently"

    def test_decimal_formatting(self):
        source = "price is 12.50 dollars"
        m = match_number("12.5", source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) >= 1

    def test_thousands_separator(self):
        source = "value of 1,234,567 units"
        m = match_number(1234567, source)
        assert m.tier is MatchTier.SEMANTIC

    def test_negative_number(self):
        source = "balance: -250.00"
        m = match_number(-250, source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) >= 1

    def test_accounting_negative(self):
        source = "loss (1,234.00)"
        m = match_number(-1234, source)
        assert m.tier is MatchTier.SEMANTIC

    def test_decimal_value_object(self):
        source = "amount $99.99"
        m = match_number(Decimal("99.99"), source)
        assert m.tier is MatchTier.SEMANTIC


class TestMatchNumberTolerance:
    def test_tolerance_matches_near_value(self):
        source = "approx 100 units"
        m = match_number(105, source, tolerance=Decimal("10"))
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 1
        assert m.spans[0].text == "100"

    def test_tolerance_too_tight(self):
        source = "approx 100 units"
        m = match_number(105, source, tolerance=Decimal("1"))
        assert m.tier is MatchTier.NONE

    def test_tolerance_none(self):
        source = "approx 100 units"
        m = match_number(105, source)
        assert m.tier is MatchTier.NONE


class TestMatchNumberNone:
    def test_no_number_in_source(self):
        m = match_number(1234, "no digits here at all")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "no matching number in source"

    def test_nearest_note(self):
        m = match_number(1000, "we have 999 apples")
        assert m.tier is MatchTier.NONE
        assert m.note == "nearest number in source is '999'"

    def test_nearest_picks_closest(self):
        m = match_number(1000, "values 10, 998, 2000")
        assert m.tier is MatchTier.NONE
        assert "998" in m.note

    def test_empty_source(self):
        m = match_number(500, "")
        assert m.tier is MatchTier.NONE

    def test_unparseable_value_string(self):
        m = match_number("abc", "123")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"


# --------------------------------------------------------------------------- #
# match_date
# --------------------------------------------------------------------------- #


class TestMatchDate:
    def test_unparseable_value(self):
        m = match_date("not a date", "some source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"
        assert m.found is False

    def test_exact_verbatim(self):
        source = "due on 2026-01-15 please"
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].text == "2026-01-15"

    def test_exact_multiple(self):
        source = "2026-01-15 and 2026-01-15"
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2

    def test_exact_limit_five(self):
        source = "2026-01-15 " * 10
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_semantic_different_format(self):
        source = "Due date: Jan 15, 2026."
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.note == "date present, written in a different format"

    def test_semantic_dmy(self):
        source = "signed on 15/01/2026"
        m = match_date("2026-01-15", source, day_first=True)
        assert m.tier is MatchTier.SEMANTIC

    def test_semantic_month_name(self):
        source = "January 15, 2026"
        m = match_date("15 January 2026", source)
        assert m.tier is MatchTier.SEMANTIC

    def test_no_matching_date(self):
        source = "no dates around"
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_different_date_present(self):
        source = "2026-02-20"
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.NONE

    def test_day_first_disambiguation(self):
        source = "date: 03/04/2026"
        # ambiguous, day_first True -> April 3
        m = match_date("2026-04-03", source, day_first=True)
        assert m.tier is MatchTier.SEMANTIC
        # day_first False -> March 4; not the same target
        m2 = match_date("2026-04-03", source, day_first=False)
        assert m2.tier is MatchTier.NONE

    def test_datetime_value(self):
        source = "on 2026-01-15 we meet"
        m = match_date(dt.date(2026, 1, 15), source)
        assert m.tier is MatchTier.EXACT

    def test_span_points_into_source(self):
        source = "Due date: Jan 15, 2026."
        m = match_date("2026-01-15", source)
        assert m.tier is MatchTier.SEMANTIC
        span = m.spans[0]
        assert source[span.start:span.end] == span.text

    def test_empty_source(self):
        m = match_date("2026-01-15", "")
        assert m.tier is MatchTier.NONE
