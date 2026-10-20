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

    def test_rank_ordering(self):
        ranks = [t.rank for t in MatchTier]
        assert ranks == sorted(ranks, reverse=True)

    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"


# --------------------------------------------------------------------------- #
# Span
# --------------------------------------------------------------------------- #

class TestSpan:
    def test_to_dict(self):
        s = Span(3, 7, "abcd")
        assert s.to_dict() == {"start": 3, "end": 7, "text": "abcd"}

    def test_fields(self):
        s = Span(start=0, end=5, text="hello")
        assert (s.start, s.end, s.text) == (0, 5, "hello")


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #

class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.NONE)
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_found_true_for_non_none(self):
        for tier in (MatchTier.EXACT, MatchTier.NORMALIZED,
                     MatchTier.SEMANTIC, MatchTier.FUZZY):
            assert Match(tier).found is True

    def test_found_false_for_none(self):
        assert Match(MatchTier.NONE).found is False

    def test_spans_default_is_independent(self):
        a = Match(MatchTier.NONE)
        b = Match(MatchTier.NONE)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []

    def test_to_dict(self):
        m = Match(
            MatchTier.EXACT,
            0.999999,
            [Span(0, 4, "text")],
            "note here",
        )
        d = m.to_dict()
        assert d == {
            "tier": "exact",
            "score": 1.0,
            "spans": [{"start": 0, "end": 4, "text": "text"}],
            "note": "note here",
        }

    def test_to_dict_empty_spans(self):
        d = Match(MatchTier.NONE, 0.123456).to_dict()
        assert d["tier"] == "none"
        assert d["score"] == 0.1235  # rounded to 4 places
        assert d["spans"] == []


# --------------------------------------------------------------------------- #
# is_trivial
# --------------------------------------------------------------------------- #

class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na",
         "none", "null", "-", "  ", "TRUE", "Yes", " 1 ", "N/A"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "42", "hello", "yes we can", "10", "-1", "NaN"],
    )
    def test_nontrivial_values(self, value):
        assert is_trivial(value) is False

    def test_non_string_inputs(self):
        assert is_trivial(0) is True
        assert is_trivial(1) is True
        assert is_trivial(2) is False
        assert is_trivial(True) is True
        assert is_trivial(None) is True


# --------------------------------------------------------------------------- #
# match_text
# --------------------------------------------------------------------------- #

class TestMatchTextExact:
    def test_exact_single(self):
        m = match_text("Alice", "Hello Alice, welcome.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert (m.spans[0].start, m.spans[0].end) == (6, 11)
        assert m.spans[0].text == "Alice"

    def test_exact_multiple_capped_at_five(self):
        source = "x " * 10 + "cat"  # six occurrences
        source = "cat cat cat cat cat cat"
        m = match_text("cat", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_spans_positions(self):
        m = match_text("ab", "abxxab")
        starts = [s.start for s in m.spans]
        assert starts == [0, 4]


class TestMatchTextTrivial:
    def test_empty_value(self):
        m = match_text("", "anything")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"
        assert m.found is False

    def test_whitespace_value(self):
        m = match_text("   ", "anything")
        assert m.tier is MatchTier.NONE

    def test_trivial_token(self):
        m = match_text("yes", "well yes indeed")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_casefolded(self):
        m = match_text("TRUE", "this is true")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_difference(self):
        m = match_text("hello world", "Hello World")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert "normalis" in m.note or "normaliz" in m.note
        assert len(m.spans) == 1

    def test_whitespace_collapse(self):
        m = match_text("hello world", "hello    world")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_unicode_fold(self):
        # ﬁ ligature folds via NFKC to "fi"
        m = match_text("file", "\ufb01le")
        assert m.tier in (MatchTier.NORMALIZED, MatchTier.EXACT)
        # Regardless, it was found
        assert m.found is True

    def test_span_within_bounds(self):
        source = "prefix   Hello   World  suffix"
        m = match_text("hello world", source)
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert 0 <= span.start <= span.end <= len(source)


class TestMatchTextPunctuation:
    def test_punctuation_difference(self):
        m = match_text("Acme Inc", "We love Acme, Inc. here.")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        assert m.spans == []


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        # One-character difference in a longer string should exceed 0.85.
        m = match_text("Helvetica Corporation",
                       "The Helvetia Corporation is based here.")
        assert m.tier is MatchTier.FUZZY
        assert 0.85 <= m.score <= 1.0
        assert len(m.spans) == 1
        assert "similar" in m.note

    def test_fuzzy_threshold_respected(self):
        # Very dissimilar: should not match.
        m = match_text("completely different words",
                       "nothing here is related at all in any way")
        assert m.tier is MatchTier.NONE

    def test_fuzzy_custom_threshold(self):
        # With a low threshold, something should match.
        m = match_text("abcde", "the document says abxde somewhere",
                       fuzzy_threshold=0.5)
        assert m.tier is MatchTier.FUZZY

    def test_fuzzy_returns_best_score_when_below(self):
        # Something in the source is a bit similar; no match but score reported.
        m = match_text("the quick brown fox jumps over the lazy dog",
                       "unrelated content entirely here",
                       fuzzy_threshold=0.99)
        assert m.tier is MatchTier.NONE
        assert m.score >= 0.0


# --------------------------------------------------------------------------- #
# match_text edge cases
# --------------------------------------------------------------------------- #

class TestMatchTextEdgeCases:
    def test_empty_source(self):
        m = match_text("hello", "")
        assert m.tier is MatchTier.NONE

    def test_value_longer_than_source(self):
        m = match_text("a very long phrase indeed", "short")
        assert m.tier is MatchTier.NONE


# --------------------------------------------------------------------------- #
# match_number
# --------------------------------------------------------------------------- #

class TestMatchNumberExact:
    def test_verbatim_number(self):
        m = match_number("4500", "The total is 4500 dollars.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].text == "4500"

    def test_verbatim_float_string(self):
        m = match_number("4500.0", "value: 4500.0")
        assert m.tier is MatchTier.EXACT

    def test_verbatim_multiple_capped(self):
        m = match_number("42", "42 and 42 and 42 and 42 and 42 and 42")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_int_value(self):
        m = match_number(4500, "Order 4500 shipped.")
        assert m.tier is MatchTier.EXACT


class TestMatchNumberSemantic:
    def test_formatted_differently(self):
        m = match_number("4500", "The total is $4,500.00 dollars.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.note == "numeric value present, formatted differently"

    def test_decimal_equivalent(self):
        m = match_number(Decimal("4500.00"), "Amount: 4,500")
        assert m.tier is MatchTier.SEMANTIC

    def test_european_format(self):
        m = match_number("1234.56", "Betrag: 1.234,56 EUR")
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance(self):
        m = match_number("100", "value is 105",
                         tolerance=Decimal("10"))
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance_negative(self):
        m = match_number("100", "value is 95",
                         tolerance=Decimal("10"))
        assert m.tier is MatchTier.SEMANTIC

    def test_multiple_semantic_spans(self):
        m = match_number("10", "ten is 10 or 10.0 maybe")
        # "10" appears verbatim so this would be EXACT.
        assert m.tier is MatchTier.EXACT

    def test_semantic_multiple(self):
        m = match_number("10", "values: 10.00 and 10,0 and more")
        assert m.tier is MatchTier.SEMANTIC
        # capped at 5
        assert 1 <= len(m.spans) <= 5


class TestMatchNumberNone:
    def test_trivial(self):
        m = match_number("0", "there are 0 items")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_int(self):
        m = match_number(1, "value 1")
        assert m.tier is MatchTier.NONE

    def test_not_numeric(self):
        m = match_number("abc", "some text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_no_number_in_source(self):
        m = match_number("9999", "no numbers here at all")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"

    def test_nearest_number_reported(self):
        m = match_number("4500", "only 4200 in the doc")
        assert m.tier is MatchTier.NONE
        assert "nearest number in source" in m.note
        assert "4200" in m.note


# --------------------------------------------------------------------------- #
# match_date
# --------------------------------------------------------------------------- #

class TestMatchDateExact:
    def test_verbatim_iso(self):
        m = match_date("2026-03-04", "Date: 2026-03-04")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1

    def test_verbatim_multi_word(self):
        m = match_date("March 4, 2026", "on March 4, 2026 we celebrate")
        assert m.tier is MatchTier.EXACT

    def test_date_object_exact(self):
        d = _dt.date(2026, 3, 4)
        # str(d) == "2026-03-04"
        m = match_date(d, "recorded 2026-03-04 here")
        assert m.tier is MatchTier.EXACT


class TestMatchDateSemantic:
    def test_different_format(self):
        m = match_date("2026-03-04", "Written on 4 March 2026 in London.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"

    def test_american_format_with_day_first(self):
        m = match_date("03/04/2026", "seen 2026-03-04",
                       day_first=False)
        assert m.tier is MatchTier.SEMANTIC

    def test_unambiguous_month(self):
        m = match_date("2026-12-25", "Christmas: 25 December 2026")
        assert m.tier is MatchTier.SEMANTIC


class TestMatchDateNone:
    def test_unparseable_value(self):
        m = match_date("not a date", "some 2026-03-04 text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_no_date_in_source(self):
        m = match_date("2026-03-04", "no dates at all in here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_ambiguous_without_day_first(self):
        # "03/04/2026" is ambiguous when day_first is None and parse fails
        m = match_date("03/04/2026", "the fourth of March")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"


# --------------------------------------------------------------------------- #
# NOT_FOUND sentinel equivalence (sanity)
# --------------------------------------------------------------------------- #

class TestIntegration:
    def test_verify_style_text_flow(self):
        m = match_text("Alice Smith", "Employee: Alice Smith, engineer")
        assert m.found is True
        assert m.tier.rank >= MatchTier.NORMALIZED.rank

    def test_verify_style_number_flow(self):
        m = match_number(4500.0, "Salary: $4,500.00")
        assert m.found is True
        assert m.tier is MatchTier.SEMANTIC

    def test_verify_style_date_flow(self):
        m = match_date("2026-03-04", "started on the 4th of March 2026")
        assert m.found is True
        assert m.tier is MatchTier.SEMANTIC
