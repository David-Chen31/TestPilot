"""Tests for llmground.matchers."""

from __future__ import annotations

import datetime as _dt
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


# ---------------------------------------------------------------------------
# MatchTier
# ---------------------------------------------------------------------------


class TestMatchTier:
    def test_values(self):
        assert MatchTier.EXACT.value == "exact"
        assert MatchTier.NORMALIZED.value == "normalized"
        assert MatchTier.SEMANTIC.value == "semantic"
        assert MatchTier.FUZZY.value == "fuzzy"
        assert MatchTier.NONE.value == "none"

    def test_ranks(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_rank_strictly_ordered(self):
        order = [
            MatchTier.NONE,
            MatchTier.FUZZY,
            MatchTier.SEMANTIC,
            MatchTier.NORMALIZED,
            MatchTier.EXACT,
        ]
        ranks = [t.rank for t in order]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_is_string_enum(self):
        assert MatchTier.EXACT == "exact"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_to_dict(self):
        s = Span(3, 7, "abcd")
        assert s.to_dict() == {"start": 3, "end": 7, "text": "abcd"}

    def test_fields(self):
        s = Span(start=0, end=2, text="hi")
        assert (s.start, s.end, s.text) == (0, 2, "hi")


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.EXACT)
        assert m.tier is MatchTier.EXACT
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_spans_default_factory_independent(self):
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
            spans=[Span(1, 4, "abc")],
            note="hi",
        )
        assert m.to_dict() == {
            "tier": "fuzzy",
            "score": 0.8765,
            "spans": [{"start": 1, "end": 4, "text": "abc"}],
            "note": "hi",
        }

    def test_to_dict_empty_spans(self):
        d = Match(MatchTier.NONE).to_dict()
        assert d == {"tier": "none", "score": 0.0, "spans": [], "note": ""}

    def test_not_found_constant(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.spans == []
        assert NOT_FOUND.note == ""
        assert NOT_FOUND.found is False


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------


class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na", "none", "null", "-"],
    )
    def test_trivial_literals(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        [" 1 ", "TRUE", "False", "N/A", "None", "NULL", "  -  ", "\tYES\n"],
    )
    def test_trivial_case_and_whitespace(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize("value", ["2", "0.0", "10", "maybe", "abc", "-1", "0.5"])
    def test_non_trivial(self, value):
        assert is_trivial(value) is False

    def test_non_string_input(self):
        assert is_trivial(1) is True
        assert is_trivial(True) is True
        assert is_trivial(0) is True
        assert is_trivial(2) is False
        assert is_trivial(False) is True
        assert is_trivial(None) is True  # str(None) == "None"


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------


class TestMatchTextExact:
    def test_verbatim(self):
        m = match_text("hello", "well hello there")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].to_dict() == {
            "start": 5,
            "end": 10,
            "text": "hello",
        }
        assert m.note == ""

    def test_multiple_occurrences(self):
        m = match_text("x", "aaa")  # trivial single char
        assert m.tier is MatchTier.NONE
        m = match_text("ab", "ab ab ab ab ab ab ab")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5  # capped at 5
        assert all(s.text == "ab" for s in m.spans)

    def test_occurrences_offsets(self):
        m = match_text("the", "the cat the dog the")
        assert [s.start for s in m.spans] == [0, 8, 16]

    def test_exact_is_case_sensitive(self):
        m = match_text("Hello", "hello world")
        assert m.tier is MatchTier.NORMALIZED


class TestMatchTextTrivial:
    @pytest.mark.parametrize(
        "value", ["", "   ", "1", "0", "true", "yes", "no", "none", "null", "-", "n/a"]
    )
    def test_trivial_returns_none(self, value):
        m = match_text(value, "some 1 true false yes no content here")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"
        assert m.found is False

    def test_whitespace_only(self):
        m = match_text("   \n\t ", "anything at all")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("Hello World", "the hello   world text")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert m.note == "matched after whitespace/case/unicode normalisation"
        assert len(m.spans) == 1
        span = m.spans[0]
        assert span.start == 4
        # span text maps to source[start:start+len(value)]
        assert span.text == "hello   world"[: len(span.text)]

    def test_whitespace_collapse(self):
        m = match_text("foo bar", "a   foo    bar   baz")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_unicode_folding(self):
        # Ligature \ufb01 folded to "fi"
        m = match_text("find", "please \ufb01nd it")
        assert m.tier is MatchTier.NORMALIZED

    def test_span_start_maps_to_source(self):
        m = match_text("Alpha Beta", "  ALPHA    BETA  ")
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert 0 <= span.start < len("  ALPHA    BETA  ")
        assert span.end <= len("  ALPHA    BETA  ")


class TestMatchTextPunctuation:
    def test_punctuation_insensitive(self):
        m = match_text("Acme Inc", "Welcome to Acme, Inc. today")
        # "acme inc" -> normalize -> "acme inc"; stripped both -> "acme inc"
        # Note: exact "Acme Inc" not in source (comma between), so falls through.
        assert m.tier is MatchTier.NORMALIZED
        # Either whitespace-normalized path or punctuation path
        assert m.score in (0.95, 0.9)

    def test_punctuation_only_match(self):
        m = match_text("Acme Inc", "Company: Acme,Inc. founded")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score in (0.95, 0.9)


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        m = match_text("accommodation", "we booked accomodation nearby")
        assert m.tier is MatchTier.FUZZY
        assert 0.85 <= m.score <= 1.0
        assert "similar" in m.note
        assert len(m.spans) == 1
        assert m.spans[0].text == "accomodation"

    def test_fuzzy_threshold_custom(self):
        # Similar but below default threshold
        value = "abcdefghij"
        source = "abcdeFGHIJ"
        m_strict = match_text(value, source, fuzzy_threshold=0.95)
        assert m_strict.tier is MatchTier.NONE

    def test_fuzzy_above_custom_threshold(self):
        value = "international"
        source = "internatonal"
        m = match_text(value, source, fuzzy_threshold=0.8)
        assert m.tier is MatchTier.FUZZY
        assert m.score >= 0.8

    def test_near_below_threshold_reports_score(self):
        m = match_text("averylonguniquestringvalue", "averyverydifferentstringhere")
        assert m.tier is MatchTier.NONE
        assert m.score >= 0.0


class TestMatchTextNone:
    def test_no_match(self):
        m = match_text("xylophone", "the quick brown fox jumps")
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_empty_source(self):
        m = match_text("hello", "")
        assert m.tier is MatchTier.NONE

    def test_empty_source_trivial_value(self):
        m = match_text("hello world", "")
        assert m.tier is MatchTier.NONE
        assert m.spans == []


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------


class TestMatchNumberTrivial:
    def test_trivial_value(self):
        m = match_number("1", "we paid 1 dollar")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    @pytest.mark.parametrize("value", ["", "   ", "0", "yes", "no", "true"])
    def test_trivial_variants(self, value):
        assert match_number(value, "some text").tier is MatchTier.NONE

    def test_numeric_int_zero_is_trivial(self):
        assert match_number(0, "0 in text").tier is MatchTier.NONE


class TestMatchNumberNonNumeric:
    def test_unparseable(self):
        m = match_number("not-a-number", "nothing here")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_abc(self):
        m = match_number("abc", "abc def")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"


class TestMatchNumberExact:
    def test_verbatim(self):
        m = match_number("4500", "we charged 4500 dollars")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].to_dict() == {"start": 11, "end": 15, "text": "4500"}

    def test_verbatim_float_string(self):
        m = match_number("4500.0", "value was 4500.0 exactly")
        assert m.tier is MatchTier.EXACT

    def test_multiple_exact_capped(self):
        m = match_number("42", "42 42 42 42 42 42 42")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5


class TestMatchNumberSemantic:
    def test_formatted_differently(self):
        m = match_number("4500.0", "total: $4,500.00 paid")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "numeric value present, formatted differently"
        assert len(m.spans) == 1
        assert m.spans[0].text == "4,500.00"

    def test_int_vs_decimal(self):
        m = match_number(4500, "we paid 4500.00 dollars")
        assert m.tier is MatchTier.SEMANTIC

    def test_currency_symbol(self):
        m = match_number("100", "cost was $100")
        assert m.tier is MatchTier.SEMANTIC

    def test_multiple_semantic_capped(self):
        m = match_number("7", "7.0 and 7.00 and 7.000 and 7.0000 and 7.00000 and 7.000000")
        # 7 is not trivial? "7" is not in _TRIVIAL set... check
        # _TRIVIAL includes "0" and "1" but not "7".
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) <= 5

    def test_tolerance(self):
        m = match_number("100", "cost was 101", tolerance=Decimal("2"))
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance_too_small(self):
        m = match_number("100", "cost was 105", tolerance=Decimal("2"))
        assert m.tier is MatchTier.NONE

    def test_tolerance_zero(self):
        m = match_number("100", "cost was 100.0", tolerance=Decimal("0"))
        assert m.tier is MatchTier.SEMANTIC

    def test_accounting_negative(self):
        m = match_number("-1234.56", "adjustment: (1,234.56)")
        assert m.tier is MatchTier.SEMANTIC

    def test_semantic_span_text_matches_raw(self):
        m = match_number("2000", "revenue of 2,000 dollars")
        assert m.tier is MatchTier.SEMANTIC
        assert m.spans[0].text == "2,000"


class TestMatchNumberNone:
    def test_no_number_present(self):
        m = match_number("9999", "no numbers at all here")
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_near_reported(self):
        m = match_number("100", "cost was 150")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert "150" in m.note
        assert m.note == "nearest number in source is '150'"

    def test_no_numbers_note(self):
        m = match_number("100", "abc def ghi")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------


class TestMatchDateNonDate:
    def test_unparseable(self):
        m = match_date("not a date", "no dates here")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_empty_string(self):
        m = match_date("", "some text")
        assert m.tier is MatchTier.NONE

    def test_ambiguous_date_refused(self):
        m = match_date("03/04/2026", "nothing")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"


class TestMatchDateExact:
    def test_verbatim(self):
        m = match_date("2026-03-04", "dated 2026-03-04 here")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].to_dict() == {
            "start": 6,
            "end": 16,
            "text": "2026-03-04",
        }

    def test_verbatim_multiple(self):
        m = match_date("2026-03-04", "2026-03-04 and 2026-03-04 and 2026-03-04")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3


class TestMatchDateSemantic:
    def test_different_format(self):
        m = match_date("2026-03-04", "dated March 4, 2026")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        assert len(m.spans) >= 1
        assert m.spans[0].text == "March 4, 2026"

    def test_slash_format(self):
        m = match_date("2026-03-04", "on 03/04/2026", day_first=False)
        assert m.tier is MatchTier.SEMANTIC

    def test_day_first_disambiguation(self):
        # 13/04/2026 unambiguous (13 > 12)
        m = match_date("2026-04-13", "meeting 13/04/2026")
        assert m.tier is MatchTier.SEMANTIC

    def test_multiple_semantic_spans(self):
        m = match_date(
            "2026-03-04",
            "March 4, 2026 and 4 March 2026 and 03/04/2026",
            day_first=False,
        )
        assert m.tier is MatchTier.SEMANTIC
        # Exact textual string "2026-03-04" not present, so semantic
        assert len(m.spans) >= 1


class TestMatchDateNone:
    def test_no_matching_date(self):
        m = match_date("2026-03-04", "no dates at all")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_different_date(self):
        m = match_date("2026-03-04", "dated 2025-01-01")
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_invalid_date_value(self):
        m = match_date("31 February 2026", "February has fewer days")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# Interaction / misc
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_found_property_consistent(self):
        assert match_text("hello", "hello there").found is True
        assert match_text("zzzzz", "hello there").found is False

    def test_match_to_dict_roundtrip_keys(self):
        m = match_text("hello", "hello there")
        d = m.to_dict()
        assert set(d) == {"tier", "score", "spans", "note"}
        assert d["tier"] == "exact"
        assert d["score"] == 1.0

    def test_rank_ordering_via_matches(self):
        exact = match_text("hello", "hello")
        normalized = match_text("hello", "HELLO")
        assert exact.tier.rank > normalized.tier.rank
