"""Unit tests for llmground.matchers."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import pytest

from llmground.matchers import (
    NOT_FOUND,
    Match,
    MatchTier,
    Span,
    is_trivial,
    match_date,
    match_number,
    match_text,
)


# --------------------------------------------------------------------------- #
# MatchTier
# --------------------------------------------------------------------------- #


def test_match_tier_values():
    assert MatchTier.EXACT.value == "exact"
    assert MatchTier.NORMALIZED.value == "normalized"
    assert MatchTier.SEMANTIC.value == "semantic"
    assert MatchTier.FUZZY.value == "fuzzy"
    assert MatchTier.NONE.value == "none"


def test_match_tier_is_str_enum():
    assert isinstance(MatchTier.EXACT, str)
    assert MatchTier.EXACT == "exact"


@pytest.mark.parametrize(
    "tier, rank",
    [
        (MatchTier.EXACT, 4),
        (MatchTier.NORMALIZED, 3),
        (MatchTier.SEMANTIC, 2),
        (MatchTier.FUZZY, 1),
        (MatchTier.NONE, 0),
    ],
)
def test_match_tier_rank(tier, rank):
    assert tier.rank == rank


def test_match_tier_rank_ordering():
    assert (
        MatchTier.EXACT.rank
        > MatchTier.NORMALIZED.rank
        > MatchTier.SEMANTIC.rank
        > MatchTier.FUZZY.rank
        > MatchTier.NONE.rank
    )


# --------------------------------------------------------------------------- #
# Span
# --------------------------------------------------------------------------- #


def test_span_to_dict():
    span = Span(2, 5, "abc")
    assert span.to_dict() == {"start": 2, "end": 5, "text": "abc"}


def test_span_defaults_and_positional():
    span = Span(0, 0, "")
    assert span.start == 0
    assert span.end == 0
    assert span.text == ""


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #


def test_match_defaults():
    m = Match(MatchTier.NONE)
    assert m.score == 0.0
    assert m.spans == []
    assert m.note == ""


def test_match_spans_default_are_independent():
    a = Match(MatchTier.NONE)
    b = Match(MatchTier.NONE)
    a.spans.append(Span(0, 1, "x"))
    assert b.spans == []


def test_match_found_property():
    assert Match(MatchTier.EXACT, 1.0).found is True
    assert Match(MatchTier.NORMALIZED).found is True
    assert Match(MatchTier.SEMANTIC).found is True
    assert Match(MatchTier.FUZZY, 0.5).found is True
    assert Match(MatchTier.NONE).found is False


def test_match_to_dict_rounds_score():
    m = Match(
        MatchTier.FUZZY,
        0.123456789,
        [Span(1, 2, "ab")],
        note="hi",
    )
    d = m.to_dict()
    assert d == {
        "tier": "fuzzy",
        "score": 0.1235,
        "spans": [{"start": 1, "end": 2, "text": "ab"}],
        "note": "hi",
    }


def test_match_to_dict_empty_spans():
    d = Match(MatchTier.NONE).to_dict()
    assert d == {"tier": "none", "score": 0.0, "spans": [], "note": ""}


def test_not_found_constant():
    assert NOT_FOUND.tier is MatchTier.NONE
    assert NOT_FOUND.score == 0.0
    assert NOT_FOUND.spans == []
    assert NOT_FOUND.found is False


# --------------------------------------------------------------------------- #
# is_trivial
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    ["", "0", "1", "true", "false", "yes", "no", "n/a", "na", "none", "null", "-"],
)
def test_is_trivial_known_values(value):
    assert is_trivial(value) is True


@pytest.mark.parametrize("value", ["TRUE", "  Yes  ", "N/A", "None", "False"])
def test_is_trivial_case_and_whitespace_insensitive(value):
    assert is_trivial(value) is True


@pytest.mark.parametrize("value", [0, 1, True, False, None])
def test_is_trivial_non_string_inputs(value):
    # str(0) -> "0", str(True) -> "True" all resolve to trivial tokens.
    assert is_trivial(value) is True


@pytest.mark.parametrize("value", ["Acme Corp", "1234", "2026-01-01", "hello"])
def test_is_trivial_false_for_meaningful_values(value):
    assert is_trivial(value) is False


# --------------------------------------------------------------------------- #
# match_text
# --------------------------------------------------------------------------- #


def test_match_text_trivial_value():
    m = match_text("1", "the value is 1 here")
    assert m.tier is MatchTier.NONE
    assert m.note == "value too trivial to verify"


def test_match_text_empty_value():
    m = match_text("", "some source")
    assert m.tier is MatchTier.NONE
    assert m.note == "value too trivial to verify"


def test_match_text_exact():
    source = "The company Acme Corp is located here."
    m = match_text("Acme Corp", source)
    assert m.tier is MatchTier.EXACT
    assert m.score == 1.0
    assert len(m.spans) == 1
    assert m.spans[0].text == "Acme Corp"
    assert source[m.spans[0].start:m.spans[0].end] == "Acme Corp"


def test_match_text_exact_multiple_spans_capped_at_five():
    source = "foo " * 10
    m = match_text("foo", source)
    assert m.tier is MatchTier.EXACT
    assert len(m.spans) == 5


def test_match_text_exact_all_spans_positions():
    source = "ab ab ab"
    m = match_text("ab", source)
    assert [s.start for s in m.spans] == [0, 3, 6]


def test_match_text_normalized_whitespace():
    source = "The   value  is   Acme    Corp today."
    m = match_text("Acme Corp", source)
    assert m.tier is MatchTier.NORMALIZED
    assert m.score == 0.95
    assert m.note == "matched after whitespace/case/unicode normalisation"
    assert len(m.spans) == 1


def test_match_text_normalized_case():
    source = "We love ACME corp very much."
    m = match_text("acme CORP", source)
    assert m.tier is MatchTier.NORMALIZED
    assert m.score == 0.95


def test_match_text_normalized_span_text_in_source():
    source = "prefix ACME   CORP suffix"
    m = match_text("acme corp", source)
    assert m.tier is MatchTier.NORMALIZED
    span = m.spans[0]
    assert source[span.start:span.end] == span.text


def test_match_text_punctuation_insensitive():
    source = "We ordered from Acme Inc today."
    m = match_text("Acme, Inc.", source)
    assert m.tier is MatchTier.NORMALIZED
    assert m.score == 0.9
    assert m.note == "matched ignoring punctuation"


def test_match_text_punctuation_insensitive_other_direction():
    source = "Visit Acme, Inc. right now"
    m = match_text("Acme Inc", source)
    # "Acme Inc" normalises to "acme inc"; stripped source has "Acme, Inc."
    # which strips to "Acme Inc". Stripped value is "Acme Inc".
    # But exact ("Acme Inc" not in source) fails, normalized fails ("acme inc"
    # not in "visit acme, inc. right now"), so punctuation tier should hit.
    assert m.tier is MatchTier.NORMALIZED
    assert m.score == 0.9
    assert m.note == "matched ignoring punctuation"


def test_match_text_fuzzy_typo():
    source = "Acme Corporation is a large business."
    m = match_text("Acme Corporaton", source, fuzzy_threshold=0.8)
    assert m.tier is MatchTier.FUZZY
    assert 0.8 <= m.score <= 1.0
    assert "similar" in m.note


def test_match_text_fuzzy_below_threshold_returns_none():
    source = "totally unrelated text about cats"
    m = match_text("Acme Corporation", source, fuzzy_threshold=0.85)
    assert m.tier is MatchTier.NONE
    assert m.spans == []
    # score should reflect the best fuzzy score found, or 0.0
    assert 0.0 <= m.score < 0.85


def test_match_text_fuzzy_threshold_boundary_exact():
    source = "the quick brown fox jumps"
    # identical value within window with slack 0 -> exact would match; use
    # slightly-off value with high threshold.
    m = match_text(
        "the quik brown fox jumps", source, fuzzy_threshold=0.9
    )
    # similarity is high enough, should be fuzzy (not exact, not normalized).
    assert m.tier is MatchTier.FUZZY
    assert m.score >= 0.9


def test_match_text_no_match_empty_source():
    m = match_text("Acme Corp", "")
    assert m.tier is MatchTier.NONE


def test_match_text_fuzzy_span_aligns_to_words():
    source = "hello world this is text"
    m = match_text("hello worlds", source, fuzzy_threshold=0.5)
    assert m.tier is MatchTier.FUZZY
    span = m.spans[0]
    # span should align to word boundaries
    assert source[span.start:span.end] == span.text
    assert not span.text.startswith(" ")
    assert not span.text.endswith(" ")


def test_match_text_window_slack_limits_candidate_length():
    # window_slack=0 means candidate length must equal len(value) exactly
    source = "some much longer text that will not match"
    m = match_text("abc", source, window_slack=0.0, fuzzy_threshold=0.99)
    assert m.tier is MatchTier.NONE


# --------------------------------------------------------------------------- #
# match_number
# --------------------------------------------------------------------------- #


def test_match_number_trivial_value():
    m = match_number(1, "there is a 1 here")
    assert m.tier is MatchTier.NONE
    assert m.note == "value too trivial to verify"


def test_match_number_non_numeric():
    m = match_number("not a number", "source")
    assert m.tier is MatchTier.NONE
    assert m.note == "value is not numeric"


def test_match_number_exact_verbatim():
    source = "The total is 4500.0 dollars."
    m = match_number(4500.0, source)
    assert m.tier is MatchTier.EXACT
    assert m.score == 1.0
    assert len(m.spans) == 1
    assert m.spans[0].text == "4500.0"


def test_match_number_exact_multiple_capped_at_five():
    source = "42 " * 8
    m = match_number(42, source)
    assert m.tier is MatchTier.EXACT
    assert len(m.spans) == 5


def test_match_number_semantic_currency_formatting():
    source = "The invoice total is $4,500.00 due today."
    m = match_number(4500.0, source)
    assert m.tier is MatchTier.SEMANTIC
    assert m.score == 1.0
    assert m.note == "numeric value present, formatted differently"
    assert len(m.spans) >= 1
    # ensure a matching span contains digits
    assert any(any(ch.isdigit() for ch in s.text) for s in m.spans)


def test_match_number_semantic_thousands_separator():
    source = "Revenue was 1,234,567 for the year."
    m = match_number(1234567, source)
    assert m.tier is MatchTier.SEMANTIC


def test_match_number_semantic_european_format():
    source = "Preis: 1.234,56 EUR"
    m = match_number(Decimal("1234.56"), source)
    assert m.tier is MatchTier.SEMANTIC


def test_match_number_tolerance_match():
    source = "Approximately 100.4 units were sold."
    m = match_number(100.0, source, tolerance=Decimal("0.5"))
    assert m.tier is MatchTier.SEMANTIC
    assert m.score == 1.0


def test_match_number_tolerance_boundary_inclusive():
    source = "value is 100.5 exactly"
    m = match_number(100, source, tolerance=Decimal("0.5"))
    assert m.tier is MatchTier.SEMANTIC


def test_match_number_tolerance_exceeded():
    source = "value is 101 here"
    m = match_number(100, source, tolerance=Decimal("0.5"))
    assert m.tier is MatchTier.NONE
    assert "nearest number" in m.note
    assert "'101'" in m.note


def test_match_number_no_match_reports_nearest():
    source = "The answer is 999."
    m = match_number(1000, source)
    assert m.tier is MatchTier.NONE
    assert m.score == 0.0
    assert m.note == "nearest number in source is '999'"


def test_match_number_no_match_no_numbers_in_source():
    source = "there are no digits here"
    m = match_number(42, source)
    assert m.tier is MatchTier.NONE
    assert m.note == "no matching number in source"


def test_match_number_nearest_picks_closest():
    source = "values are 10, 200, 3000"
    m = match_number(500, source)
    assert m.tier is MatchTier.NONE
    assert m.note == "nearest number in source is '200'"


def test_match_number_negative_semantic():
    source = "The balance was (1,234.56) which is negative."
    m = match_number(Decimal("-1234.56"), source)
    assert m.tier is MatchTier.SEMANTIC


def test_match_number_string_input():
    source = "Cost: 12.50"
    m = match_number("12.50", source)
    # "12.50" appears verbatim -> EXACT
    assert m.tier is MatchTier.EXACT


# --------------------------------------------------------------------------- #
# match_date
# --------------------------------------------------------------------------- #


def test_match_date_unrecognisable():
    m = match_date("not a date", "source text")
    assert m.tier is MatchTier.NONE
    assert m.note == "value is not a recognisable date"


def test_match_date_exact_verbatim():
    source = "The event is on 2026-01-15 at noon."
    m = match_date("2026-01-15", source)
    assert m.tier is MatchTier.EXACT
    assert m.score == 1.0
    assert len(m.spans) == 1
    assert m.spans[0].text == "2026-01-15"


def test_match_date_exact_multiple_capped_at_five():
    source = "2026-01-15 " * 8
    m = match_date("2026-01-15", source)
    assert m.tier is MatchTier.EXACT
    assert len(m.spans) == 5


def test_match_date_semantic_different_format():
    source = "Meeting scheduled for January 15, 2026."
    m = match_date("15 January 2026", source)
    assert m.tier is MatchTier.SEMANTIC
    assert m.score == 1.0
    assert m.note == "date present, written in a different format"
    assert len(m.spans) >= 1


def test_match_date_semantic_date_object_input():
    source = "The deadline is 2026-03-01."
    m = match_date(_dt.date(2026, 3, 1), source)
    assert m.tier in (MatchTier.EXACT, MatchTier.SEMANTIC)


def test_match_date_no_match():
    source = "No dates in this text at all."
    m = match_date("2026-01-15", source)
    assert m.tier is MatchTier.NONE
    assert m.note == "no matching date in source"


def test_match_date_day_first_disambiguation():
    source = "The date is 03/04/2026."
    # day_first=True -> 3 April 2026
    m = match_date("2026-04-03", source, day_first=True)
    assert m.tier == MatchTier.SEMANTIC
    # day_first=False -> March 4 2026
    m2 = match_date("2026-04-03", source, day_first=False)
    assert m2.tier == MatchTier.NONE


def test_match_date_ambiguous_without_flag():
    # "03/04/2026" ambiguous and day_first=None -> parse_date returns None
    m = match_date("03/04/2026", "text with 03/04/2026 in it")
    assert m.tier is MatchTier.NONE
    assert m.note == "value is not a recognisable date"


def test_match_date_semantic_span_text():
    source = "Some preamble then Jan 15, 2026 follows."
    m = match_date("2026-01-15", source)
    assert m.tier == MatchTier.SEMANTIC
    span = m.spans[0]
    assert source[span.start:span.end] == span.text
