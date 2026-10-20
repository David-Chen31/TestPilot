"""Unit tests for llmground.matchers."""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import pytest

from llmground.matchers import (
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


class TestMatchTier:
    def test_values(self):
        assert MatchTier.EXACT.value == "exact"
        assert MatchTier.NORMALIZED.value == "normalized"
        assert MatchTier.SEMANTIC.value == "semantic"
        assert MatchTier.FUZZY.value == "fuzzy"
        assert MatchTier.NONE.value == "none"

    def test_is_str_enum(self):
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
    def test_rank(self, tier, rank):
        assert tier.rank == rank

    def test_rank_is_strictly_ordered(self):
        tiers = [
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        ]
        ranks = [t.rank for t in tiers]
        assert ranks == sorted(ranks, reverse=True)


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

    def test_default_spans_are_independent(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []

    @pytest.mark.parametrize(
        "tier, found",
        [
            (MatchTier.EXACT, True),
            (MatchTier.NORMALIZED, True),
            (MatchTier.SEMANTIC, True),
            (MatchTier.FUZZY, True),
            (MatchTier.NONE, False),
        ],
    )
    def test_found(self, tier, found):
        assert Match(tier).found is found

    def test_to_dict(self):
        m = Match(
            MatchTier.FUZZY,
            score=0.87654,
            spans=[Span(10, 15, "hello")],
            note="similar",
        )
        assert m.to_dict() == {
            "tier": "fuzzy",
            "score": 0.8765,
            "spans": [{"start": 10, "end": 15, "text": "hello"}],
            "note": "similar",
        }

    def test_to_dict_rounds_score(self):
        m = Match(MatchTier.EXACT, score=0.99999)
        assert m.to_dict()["score"] == 1.0

    def test_to_dict_no_spans(self):
        m = Match(MatchTier.NONE)
        assert m.to_dict() == {
            "tier": "none",
            "score": 0.0,
            "spans": [],
            "note": "",
        }


# --------------------------------------------------------------------------- #
# is_trivial
# --------------------------------------------------------------------------- #


class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na", "none",
         "null", "-"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["TRUE", "Yes", "None", "  n/a  ", "False", "N/A"],
    )
    def test_trivial_is_case_and_whitespace_insensitive(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "42", "hello", "0.0", "1.0", "yes please", "a", "trueish"],
    )
    def test_non_trivial(self, value):
        assert is_trivial(value) is False

    def test_non_string_input_coerced_to_str(self):
        # str(1) -> "1" is trivial; str(True) -> "True" is trivial.
        assert is_trivial(1) is True
        assert is_trivial(True) is True
        assert is_trivial(0) is True
        assert is_trivial(None) is True  # str(None) == "None"
        assert is_trivial(2) is False

    def test_none_is_trivial(self):
        # str(None).strip().casefold() == "none", in _TRIVIAL
        assert is_trivial(None) is True


# --------------------------------------------------------------------------- #
# match_text
# --------------------------------------------------------------------------- #


class TestMatchTextExact:
    def test_exact_match(self):
        source = "The quick brown fox."
        m = match_text("quick", source)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].start == 4
        assert m.spans[0].end == 9
        assert m.spans[0].text == "quick"
        assert m.found is True

    def test_exact_multiple_occurrences(self):
        source = "foo bar foo baz foo"
        m = match_text("foo", source)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert [s.start for s in m.spans] == [0, 8, 16]

    def test_exact_caps_at_five_spans(self):
        source = " ".join(["x"] * 10)
        m = match_text("x", source)
        # "x" is *not* trivial (it's a single letter, not in _TRIVIAL)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_span_text_matches_source(self):
        source = "prefix needle suffix"
        m = match_text("needle", source)
        span = m.spans[0]
        assert source[span.start:span.end] == "needle"


class TestMatchTextTrivial:
    @pytest.mark.parametrize("value", ["", "   ", "1", "0", "yes", "TRUE"])
    def test_trivial_returns_none(self, value):
        m = match_text(value, "some document 1 0 yes")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == "value too trivial to verify"

    def test_none_value_returns_none(self):
        # matches `not value`
        m = match_text("", "abc")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("Quick", "the QUICK brown")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert "normalisation" in m.note
        assert len(m.spans) == 1

    def test_whitespace_collapse(self):
        source = "hello    world   again"
        m = match_text("hello world", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_span_points_into_original(self):
        source = "intro   Hello   World outro"
        m = match_text("hello world", source)
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert source[span.start:span.end] == "Hello   World"

    def test_unicode_fold(self):
        # Fullwidth / lookalike folding. "café" via NFKC.
        source = "The cafe\u0301 is open"  # combining accent
        m = match_text("café", source)
        # verbatim may or may not match; accept either EXACT or NORMALIZED
        assert m.tier in (MatchTier.EXACT, MatchTier.NORMALIZED)
        assert m.found is True


class TestMatchTextPunctuation:
    def test_punctuation_insensitive(self):
        source = "We hired Acme Inc today."
        m = match_text("Acme, Inc.", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        # Per the implementation, no spans are attached in this tier.
        assert m.spans == []

    def test_punctuation_with_trailing_period(self):
        source = "registered at Example Corp."
        m = match_text("Example Corp.", source)
        # "Example Corp." is verbatim in source -> EXACT
        assert m.tier is MatchTier.EXACT

    def test_punctuation_ignored_when_verbatim_absent(self):
        source = "the company is Widgets R Us"
        m = match_text("Widgets R Us.", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"


class TestMatchTextFuzzy:
    def test_fuzzy_ocr_noise(self):
        m = match_text("Elephant", "The elephont is grey", fuzzy_threshold=0.85)
        assert m.tier is MatchTier.FUZZY
        assert 0.85 <= m.score < 1.0
        assert m.note.startswith("closest text in source is")
        assert len(m.spans) == 1

    def test_fuzzy_span_is_substring(self):
        source = "The elephont is grey"
        m = match_text("Elephant", source, fuzzy_threshold=0.85)
        assert m.tier is MatchTier.FUZZY
        span = m.spans[0]
        assert source[span.start:span.end] == span.text

    def test_below_threshold_returns_none(self):
        m = match_text("Elephant", "totally unrelated content here",
                       fuzzy_threshold=0.85)
        assert m.tier is MatchTier.NONE
        assert m.spans == []
        assert m.score < 0.85

    def test_none_tier_score_reflects_best(self):
        # No near match; score should be the best window score seen (>= 0).
        m = match_text("abcdefghij", "zzzzzzzzzzzzzzzzzzzz", fuzzy_threshold=0.95)
        assert m.tier is MatchTier.NONE
        assert 0.0 <= m.score < 0.95

    def test_custom_threshold_respected(self):
        source = "roughly similar text"
        low = match_text("roughly similar txt", source, fuzzy_threshold=0.5)
        high = match_text("roughly similar txt", source, fuzzy_threshold=0.99)
        assert low.tier is MatchTier.FUZZY
        assert high.tier is MatchTier.NONE

    def test_token_set_ratio_catches_reorder(self):
        source = "Khan Ahmed"
        m = match_text("Ahmed Khan!", source, fuzzy_threshold=0.6)
        # Without exact/normalized/punct match we should reach fuzzy.
        # Punctuation-stripped form is "ahmed khan" vs "khan ahmed" — not substring,
        # so fuzzy tier with token_set_ratio bonus should fire.
        assert m.tier is MatchTier.FUZZY

    def test_no_words_in_source(self):
        # whitespace-only source yields no windows -> NONE with score 0.0
        m = match_text("something", "    ", fuzzy_threshold=0.1)
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0


# --------------------------------------------------------------------------- #
# match_number
# --------------------------------------------------------------------------- #


class TestMatchNumberExact:
    def test_exact_string(self):
        m = match_number("42", "answer is 42 here")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        span = m.spans[0]
        assert span.text == "42"
        assert span.start == 10
        assert span.end == 12

    def test_exact_caps_at_five(self):
        source = " ".join(["42"] * 8)
        m = match_number("42", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_numeric_input(self):
        # str(42) == "42", verbatim
        m = match_number(42, "value: 42")
        assert m.tier is MatchTier.EXACT

    def test_exact_decimal_input(self):
        # str(Decimal("4500.0")) -> "4500.0"; source has it verbatim
        m = match_number(Decimal("4500.0"), "amount 4500.0 total")
        assert m.tier is MatchTier.EXACT


class TestMatchNumberSemantic:
    def test_formatted_differently(self):
        m = match_number(4500, "the price is $4,500.00 today")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "numeric value present, formatted differently"
        span = m.spans[0]
        assert "$4,500.00" in span.text

    def test_semantic_negative_accounting(self):
        m = match_number(-1234.5, "net: (1,234.50)")
        assert m.tier is MatchTier.SEMANTIC

    def test_semantic_multiple_matches(self):
        source = "prices: 4,500 and 4500.00 and 4500"
        m = match_number(4500, source)
        assert m.tier is MatchTier.SEMANTIC
        # Raw "4500" is verbatim -> EXACT is taken first.
        # Re-run with a value that forces semantic:
        m2 = match_number(Decimal("4500.00"), "$4,500 and 4,500.0 and 4,500")
        assert m2.tier is MatchTier.SEMANTIC
        assert len(m2.spans) == 3

    def test_tolerance(self):
        m = match_number(100, "value is 105", tolerance=Decimal("10"))
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0

    def test_tolerance_too_tight(self):
        m = match_number(100, "value is 105", tolerance=Decimal("1"))
        assert m.tier is MatchTier.NONE

    def test_no_tolerance_zero(self):
        m = match_number(100, "value is 105")
        assert m.tier is MatchTier.NONE


class TestMatchNumberNotFound:
    def test_not_found(self):
        m = match_number(42, "nothing numeric here")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "no matching number in source"

    def test_not_found_with_nearest(self):
        m = match_number(100, "value is 50")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "nearest number in source is '50'"

    def test_nearest_picks_closest(self):
        m = match_number(100, "values: 10 and 90 and 200")
        assert m.tier is MatchTier.NONE
        assert m.note == "nearest number in source is '90'"


class TestMatchNumberTrivialAndInvalid:
    @pytest.mark.parametrize("value", ["0", "1", "yes", ""])
    def test_trivial(self, value):
        m = match_number(value, "source contains 0 1 yes")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_not_numeric(self):
        m = match_number("hello", "the word hello appears")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_none_is_trivial(self):
        m = match_number(None, "any source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"


# --------------------------------------------------------------------------- #
# match_date
# --------------------------------------------------------------------------- #


class TestMatchDateExact:
    def test_exact_string(self):
        m = match_date("2026-03-04", "The date is 2026-03-04 today")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        span = m.spans[0]
        assert span.text == "2026-03-04"

    def test_exact_caps_at_five(self):
        source = " ".join(["2026-03-04"] * 8)
        m = match_date("2026-03-04", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5


class TestMatchDateSemantic:
    def test_different_format(self):
        m = match_date("2026-03-04", "Due on 4 March 2026.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        span = m.spans[0]
        assert "4 March 2026" in span.text or "March" in span.text

    def test_semantic_with_object(self):
        m = match_date(_dt.date(2026, 3, 4), "due 04/03/2026", day_first=True)
        assert m.tier is MatchTier.SEMANTIC

    def test_day_first_disambiguation(self):
        # 03/04/2026 ambiguous without day_first
        m = match_date("2026-03-04", "date: 03/04/2026", day_first=True)
        assert m.tier is MatchTier.SEMANTIC
        span_text = m.spans[0].text
        assert "03/04/2026" in span_text

    def test_day_first_false_does_not_match(self):
        # 03/04/2026 with day_first=False -> 2026-04-03, not target 2026-03-04
        m = match_date("2026-03-04", "date: 03/04/2026", day_first=False)
        assert m.tier is MatchTier.NONE

    def test_no_matching_date(self):
        m = match_date("2026-03-04", "nothing date-like here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_wrong_date(self):
        m = match_date("2026-03-04", "and the date is 2026-03-05")
        assert m.tier is MatchTier.NONE

    def test_semantic_caps_at_five(self):
        # Many differently-formatted occurrences of same date.
        source = " ".join(["4 March 2026"] * 8)
        m = match_date("2026-03-04", source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5


class TestMatchDateInvalid:
    def test_unrecognisable(self):
        m = match_date("not a date", "the source says nothing")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_empty_string(self):
        m = match_date("", "any source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_invalid_calendar_date(self):
        # parse_date rejects 31 Feb
        m = match_date("31/02/2026", "today is 31/02/2026", day_first=True)
        assert m.tier is MatchTier.NONE

    def test_none(self):
        m = match_date(None, "any source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"
