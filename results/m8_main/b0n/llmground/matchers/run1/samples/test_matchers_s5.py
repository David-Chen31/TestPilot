"""Unit tests for llmground.matchers."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# MatchTier
# ---------------------------------------------------------------------------

class TestMatchTier:
    def test_is_string_enum(self):
        assert MatchTier.EXACT == "exact"
        assert MatchTier.NONE == "none"

    def test_rank_order(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_rank_strictly_decreasing(self):
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


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

class TestSpan:
    def test_fields(self):
        s = Span(start=3, end=7, text="abcd")
        assert s.start == 3
        assert s.end == 7
        assert s.text == "abcd"

    def test_to_dict(self):
        s = Span(1, 4, "abc")
        assert s.to_dict() == {"start": 1, "end": 4, "text": "abc"}


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.NONE)
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_found_true_for_non_none(self):
        for tier in (
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
        ):
            assert Match(tier).found is True

    def test_found_false_for_none(self):
        assert Match(MatchTier.NONE).found is False

    def test_spans_default_not_shared(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []

    def test_to_dict(self):
        m = Match(
            MatchTier.SEMANTIC,
            score=0.98765,
            spans=[Span(2, 5, "abc")],
            note="hello",
        )
        d = m.to_dict()
        assert d == {
            "tier": "semantic",
            "score": 0.9877,
            "spans": [{"start": 2, "end": 5, "text": "abc"}],
            "note": "hello",
        }

    def test_to_dict_rounds_score_to_four_places(self):
        assert Match(MatchTier.EXACT, score=1.0).to_dict()["score"] == 1.0
        assert Match(MatchTier.FUZZY, score=0.123456).to_dict()["score"] == 0.1235

    def test_not_found_constant(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.found is False


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------

class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            "0",
            "1",
            "true",
            "True",
            "TRUE",
            "false",
            "yes",
            "no",
            "n/a",
            "na",
            "none",
            "null",
            "-",
        ],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            " 1 ",
            " 0 ",
            " YES ",
            "\ttrue\n",
        ],
    )
    def test_trivial_with_whitespace(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "42", "Acme Inc", "hello", "12", "10", "-1", "n/a.", "maybe"],
    )
    def test_non_trivial_values(self, value):
        assert is_trivial(value) is False

    def test_numeric_input(self):
        assert is_trivial(1) is True
        assert is_trivial(0) is True
        assert is_trivial(2) is False

    def test_none_input(self):
        # str(None).strip().casefold() == "none"
        assert is_trivial(None) is True


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------

class TestMatchTextExact:
    def test_exact_match_single(self):
        result = match_text("Acme Inc", "The company is Acme Inc here.")
        assert result.tier is MatchTier.EXACT
        assert result.score == 1.0
        assert result.note == ""
        assert len(result.spans) == 1
        span = result.spans[0]
        assert span.text == "Acme Inc"
        assert span.start == 15
        assert span.end == 23
        assert "The company is Acme Inc here."[span.start:span.end] == "Acme Inc"

    def test_exact_match_multiple_spans(self):
        source = "Acme Inc and again Acme Inc twice"
        result = match_text("Acme Inc", source)
        assert result.tier is MatchTier.EXACT
        assert [s.start for s in result.spans] == [0, 22]

    def test_exact_spans_capped_at_five(self):
        source = "foo " * 10
        # "foo " repeated 10 times -> 10 occurrences of "foo"
        result = match_text("foo", source)
        assert result.tier is MatchTier.EXACT
        assert len(result.spans) == 5

    def test_exact_case_sensitive(self):
        result = match_text("Acme", "the ACME company")
        assert result.tier is not MatchTier.EXACT
        assert result.tier is MatchTier.NORMALIZED


class TestMatchTextTrivial:
    @pytest.mark.parametrize("value", ["", "1", "yes", "none", "  ", "-"])
    def test_trivial_returns_none(self, value):
        result = match_text(value, "some source")
        assert result.tier is MatchTier.NONE
        assert result.score == 0.0
        assert result.note == "value too trivial to verify"
        assert result.spans == []


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        source = "The company is ACME INC."
        result = match_text("acme inc", source)
        assert result.tier is MatchTier.NORMALIZED
        assert result.score == 0.95
        assert result.note == "matched after whitespace/case/unicode normalisation"
        assert len(result.spans) == 1
        span = result.spans[0]
        assert span.end - span.start == len("acme inc")
        assert source[span.start:span.end].lower() == "acme inc."

    def test_whitespace_normalized(self):
        source = "Acme    Inc   Corp"
        result = match_text("Acme Inc Corp", source)
        assert result.tier is MatchTier.NORMALIZED
        assert result.score == 0.95

    def test_unicode_fold(self):
        # full-width letters should fold to ascii
        source = "\uff21\uff43\uff4d\uff45 \uff29\uff4e\uff43"
        result = match_text("Acme Inc", source)
        assert result.tier is MatchTier.NORMALIZED
        assert result.score == 0.95

    def test_normalized_span_maps_to_original(self):
        source = "    The   ACME    Inc    company"
        result = match_text("acme inc", source)
        assert result.tier is MatchTier.NORMALIZED
        span = result.spans[0]
        assert 0 <= span.start <= len(source)
        assert span.end <= len(source)
        assert span.end - span.start == len("acme inc")


class TestMatchTextPunctuation:
    def test_punctuation_insensitive(self):
        source = "Acme, Inc. is great"
        result = match_text("Acme Inc", source)
        assert result.tier is MatchTier.NORMALIZED
        assert result.score == 0.9
        assert result.note == "matched ignoring punctuation"
        assert result.spans == []

    def test_punctuation_only_match(self):
        result = match_text("Acme-Inc", "the ACME inc company")
        # exact fails (different exact text), normalized: "acme-inc" not in
        # "the acme inc company", but stripped both -> "acmeinc" matches
        assert result.tier is MatchTier.NORMALIZED
        assert result.score == 0.9


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        source = "The Acrne Inc company"
        result = match_text("Acme Inc", source)
        assert result.tier is MatchTier.FUZZY
        assert 0.85 <= result.score <= 1.0
        assert result.note.startswith("closest text in source is")
        assert len(result.spans) == 1

    def test_fuzzy_threshold_respected(self):
        # Extremely dissimilar value -> NONE
        result = match_text("zzzzzzzzzz", "The Acme Inc company")
        assert result.tier is MatchTier.NONE
        assert result.score < 0.85

    def test_custom_low_threshold(self):
        source = "The Acme Company here"
        result = match_text("Acme Coorp", source, fuzzy_threshold=0.5)
        assert result.tier is MatchTier.FUZZY
        assert result.score >= 0.5

    def test_custom_high_threshold_rejects(self):
        source = "The Acrne Inc company"
        result = match_text("Acme Inc", source, fuzzy_threshold=0.999)
        assert result.tier is MatchTier.NONE
        assert result.score < 0.999

    def test_word_order_insensitive_fuzzy(self):
        # token_set_ratio should help reordered text
        result = match_text("Khan Ahmed", "Ahmed Khan", fuzzy_threshold=0.8)
        assert result.tier is MatchTier.FUZZY


class TestMatchTextNone:
    def test_no_match(self):
        result = match_text("Completely Different", "The Acme Inc company")
        assert result.tier is MatchTier.NONE
        assert result.spans == []

    def test_empty_source(self):
        result = match_text("Acme Inc", "")
        assert result.tier is MatchTier.NONE
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------

class TestMatchNumberTrivial:
    def test_trivial(self):
        result = match_number("0", "0 appears here")
        assert result.tier is MatchTier.NONE
        assert result.note == "value too trivial to verify"

    def test_trivial_one(self):
        result = match_number(1, "there is a 1")
        assert result.tier is MatchTier.NONE

    def test_not_numeric(self):
        result = match_number("abc", "nothing here")
        assert result.tier is MatchTier.NONE
        assert result.note == "value is not numeric"


class TestMatchNumberExact:
    def test_verbatim(self):
        result = match_number("4500", "the number 4500 here")
        assert result.tier is MatchTier.EXACT
        assert result.score == 1.0
        assert len(result.spans) == 1
        assert result.spans[0].text == "4500"
        assert result.spans[0].start == 11

    def test_verbatim_multiple_capped(self):
        source = " 42 " * 10
        result = match_number("42", source)
        assert result.tier is MatchTier.EXACT
        assert len(result.spans) == 5


class TestMatchNumberSemantic:
    def test_formatted_differently(self):
        # "$4,500.00" parsed as 4500.00; target 4500.0
        result = match_number("4500.0", "the price is $4,500.00 today")
        assert result.tier is MatchTier.SEMANTIC
        assert result.score == 1.0
        assert len(result.spans) >= 1
        # raw text should include the formatted version
        assert any("4,500.00" in s.text for s in result.spans)
        assert result.note == "numeric value present, formatted differently"

    def test_european_format(self):
        result = match_number(Decimal("1234.5"), "the value is 1.234,5")
        assert result.tier is MatchTier.SEMANTIC

    def test_accounting_negative(self):
        result = match_number(-1234.56, "losses of (1,234.56) this year")
        assert result.tier is MatchTier.SEMANTIC

    def test_tolerance_match(self):
        result = match_number(
            100, "value near 105 here", tolerance=Decimal("10")
        )
        assert result.tier is MatchTier.SEMANTIC
        assert result.spans[0].text == "105"

    def test_tolerance_excludes_far(self):
        result = match_number(100, "value near 200 here", tolerance=Decimal("10"))
        assert result.tier is MatchTier.NONE

    def test_semantic_multiple_capped(self):
        source = "4,500.00 " * 10
        result = match_number("4500", source)
        assert result.tier is MatchTier.SEMANTIC
        assert len(result.spans) == 5


class TestMatchNumberNotFound:
    def test_no_number_in_source(self):
        result = match_number("4500", "no digits at all here")
        assert result.tier is MatchTier.NONE
        assert result.note == "no matching number in source"

    def test_nearest_note(self):
        result = match_number(4500, "the value 100 appears")
        assert result.tier is MatchTier.NONE
        assert result.note == "nearest number in source is '100'"

    def test_nearest_note_picks_closest(self):
        result = match_number(4500, "values 100 and 4000 and 999 here")
        assert result.tier is MatchTier.NONE
        assert "4000" in result.note

    def test_unparseable_source_numbers_ignored(self):
        result = match_number("4500", "no real numbers, just text")
        assert result.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------

class TestMatchDateExact:
    def test_verbatim(self):
        source = "published on 2026-01-15 here"
        result = match_date("2026-01-15", source)
        assert result.tier is MatchTier.EXACT
        assert result.score == 1.0
        assert result.spans[0].text == "2026-01-15"
        assert result.spans[0].start == 13

    def test_verbatim_multiple_capped(self):
        source = "2026-01-15 " * 10
        result = match_date("2026-01-15", source)
        assert result.tier is MatchTier.EXACT
        assert len(result.spans) == 5


class TestMatchDateSemantic:
    def test_different_format(self):
        result = match_date("2026-01-15", "dated January 15, 2026.")
        assert result.tier is MatchTier.SEMANTIC
        assert result.score == 1.0
        assert len(result.spans) == 1
        assert result.note == "date present, written in a different format"

    def test_month_name_format(self):
        result = match_date("15 January 2026", "on 2026-01-15 we met")
        assert result.tier is MatchTier.SEMANTIC

    def test_day_first_disambiguation(self):
        result = match_date("03/04/2026", "date: 4/3/2026", day_first=True)
        assert result.tier is MatchTier.SEMANTIC

    def test_semantic_multiple_capped(self):
        source = "January 15, 2026 and 15 Jan 2026 and 2026-01-15 and more 01/15/2026"
        result = match_date("2026-01-15", source)
        # exact would also find 2026-01-15 inside source, so remove that
        assert result.tier is MatchTier.SEMANTIC or result.tier is MatchTier.EXACT
        assert len(result.spans) <= 5


class TestMatchDateNotFound:
    def test_ambiguous_date_returns_none(self):
        result = match_date("03/04/2026", "date: 03/04/2026")
        assert result.tier is MatchTier.NONE
        assert result.note == "value is not a recognisable date"

    def test_not_a_date(self):
        result = match_date("hello", "some text")
        assert result.tier is MatchTier.NONE
        assert result.note == "value is not a recognisable date"

    def test_no_matching_date(self):
        result = match_date("2026-01-15", "the date is 2020-05-05")
        assert result.tier is MatchTier.NONE
        assert result.note == "no matching date in source"

    def test_empty_source(self):
        result = match_date("2026-01-15", "")
        assert result.tier is MatchTier.NONE

    def test_invalid_date(self):
        # 31 February parses to None
        result = match_date("31/02/2026", "date 31/02/2026", day_first=True)
        assert result.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# Cross-cutting behavior
# ---------------------------------------------------------------------------

class TestTieredEvidence:
    def test_exact_beats_normalized(self):
        source = "Acme Inc"
        assert match_text("Acme Inc", source).tier is MatchTier.EXACT
        assert match_text("acme inc", source).tier is MatchTier.NORMALIZED

    def test_number_exact_beats_semantic(self):
        assert match_number("42", "value 42 here").tier is MatchTier.EXACT
        assert match_number("42.0", "value 42 here").tier is MatchTier.SEMANTIC

    def test_return_types(self):
        for result in (
            match_text("Acme Inc", "Acme Inc"),
            match_number(42, "42"),
            match_date("2026-01-15", "2026-01-15"),
        ):
            assert isinstance(result, Match)
            assert isinstance(result.tier, MatchTier)
            assert isinstance(result.score, float)
            assert isinstance(result.spans, list)
            for span in result.spans:
                assert isinstance(span, Span)
