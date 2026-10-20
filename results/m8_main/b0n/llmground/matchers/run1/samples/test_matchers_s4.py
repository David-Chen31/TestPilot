"""Tests for llmground.matchers."""

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

    def test_rank_ordering(self):
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

    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_fields(self):
        s = Span(3, 7, "abcd")
        assert s.start == 3
        assert s.end == 7
        assert s.text == "abcd"

    def test_to_dict(self):
        s = Span(3, 7, "abcd")
        assert s.to_dict() == {"start": 3, "end": 7, "text": "abcd"}

    def test_to_dict_keys(self):
        d = Span(0, 0, "").to_dict()
        assert set(d.keys()) == {"start", "end", "text"}


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.NONE)
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_default_spans_independent(self):
        a = Match(MatchTier.NONE)
        b = Match(MatchTier.NONE)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []

    def test_found_true_for_non_none(self):
        for tier in [
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
        ]:
            assert Match(tier).found is True

    def test_found_false_for_none(self):
        assert Match(MatchTier.NONE).found is False

    def test_to_dict(self):
        m = Match(
            MatchTier.EXACT,
            1.0,
            [Span(0, 3, "abc")],
            "note here",
        )
        assert m.to_dict() == {
            "tier": "exact",
            "score": 1.0,
            "spans": [{"start": 0, "end": 3, "text": "abc"}],
            "note": "note here",
        }

    def test_to_dict_rounds_score(self):
        m = Match(MatchTier.FUZZY, 0.123456789)
        assert m.to_dict()["score"] == 0.1235

    def test_to_dict_rounds_banker(self):
        # round() uses banker's rounding; 0.00005 -> 0.0001 (4 dp)
        m = Match(MatchTier.FUZZY, 0.987654)
        assert m.to_dict()["score"] == 0.9877

    def test_to_dict_empty_spans(self):
        assert Match(MatchTier.NONE).to_dict()["spans"] == []


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------


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
        ["TRUE", "True", "No", "YES", "N/A", "None", "NULL", "  1  "],
    )
    def test_case_and_whitespace_insensitive(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "0.0", "0.5", "abc", "Acme Inc", "true-ish", "10", "yes indeed"],
    )
    def test_non_trivial(self, value):
        assert is_trivial(value) is False

    def test_non_string_input(self):
        assert is_trivial(1) is True
        assert is_trivial(0) is True
        assert is_trivial(2) is False

    def test_none_input(self):
        # str(None).strip().casefold() == "none" -> trivial
        assert is_trivial(None) is True


# ---------------------------------------------------------------------------
# match_text: trivial / empty
# ---------------------------------------------------------------------------


class TestMatchTextTrivial:
    def test_empty_value(self):
        m = match_text("", "some source")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "value too trivial to verify"

    def test_whitespace_value(self):
        m = match_text("   ", "some source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    @pytest.mark.parametrize("value", ["0", "1", "yes", "no", "true", "none"])
    def test_trivial_values(self, value):
        m = match_text(value, "0 1 yes no true none")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_case_insensitive(self):
        m = match_text("TRUE", "TRUE is here")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# match_text: exact
# ---------------------------------------------------------------------------


class TestMatchTextExact:
    def test_exact_single(self):
        src = "The Acme Corporation reported revenue."
        m = match_text("Acme", src)
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        sp = m.spans[0]
        assert src[sp.start:sp.end] == "Acme"
        assert sp.text == "Acme"

    def test_exact_span_offsets(self):
        src = "prefix Acme suffix"
        m = match_text("Acme", src)
        sp = m.spans[0]
        assert (sp.start, sp.end) == (7, 11)

    def test_exact_multiple_occurrences(self):
        src = "Acme and Acme and Acme"
        m = match_text("Acme", src)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3
        for sp in m.spans:
            assert src[sp.start:sp.end] == "Acme"

    def test_exact_capped_at_five(self):
        src = "x " * 10 + "Acme" + " Acme" * 10
        m = match_text("Acme", src)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_case_sensitive(self):
        # "acme" appears but not "Acme"; the exact tier must not fire.
        m = match_text("Acme", "the acme company")
        assert m.tier is not MatchTier.EXACT

    def test_exact_overlapping_finds_start_step(self):
        src = "aaaa"
        m = match_text("aa", src)
        assert m.tier is MatchTier.EXACT
        # _find_all advances by 1, so it finds positions 0,1,2
        assert [(s.start, s.end) for s in m.spans] == [(0, 2), (1, 3), (2, 4)]


# ---------------------------------------------------------------------------
# match_text: normalized
# ---------------------------------------------------------------------------


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("Acme Corporation", "the ACME CORPORATION is here")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert len(m.spans) == 1
        assert m.note == "matched after whitespace/case/unicode normalisation"

    def test_whitespace_collapse(self):
        m = match_text("Acme   Corporation", "the Acme Corporation is here")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_window_slack_does_not_matter_for_normalized(self):
        m = match_text("Acme Corporation", "ACME corporation")
        assert m.tier is MatchTier.NORMALIZED

    def test_newline_vs_space(self):
        m = match_text("Acme Corporation", "Acme\nCorporation")
        assert m.tier is MatchTier.NORMALIZED

    def test_span_points_into_original(self):
        src = "the ACME Corporation is here"
        m = match_text("Acme Corporation", src)
        sp = m.spans[0]
        # Span text should come from the original source.
        assert sp.text == src[sp.start:sp.end]
        assert "ACME" in sp.text

    def test_message_same_after_normalization(self):
        m = match_text("hello world", "HELLO   WORLD")
        assert m.tier is MatchTier.NORMALIZED


# ---------------------------------------------------------------------------
# match_text: punctuation-insensitive normalized
# ---------------------------------------------------------------------------


class TestMatchTextPunctuation:
    def test_punctuation_ignored(self):
        m = match_text("Acme Inc", "Acme, Inc. reported")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        assert m.spans == []

    def test_punctuation_trailing_dot(self):
        m = match_text("Acme Inc.", "Acme Inc is a company")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9


# ---------------------------------------------------------------------------
# match_text: fuzzy
# ---------------------------------------------------------------------------


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        m = match_text("Corporaton", "The Acme Corporation.")
        assert m.tier is MatchTier.FUZZY
        assert 0.0 < m.score < 1.0
        assert m.spans
        assert "similar" in m.note

    def test_fuzzy_score_meets_threshold(self):
        m = match_text("Corporaton", "The Acme Corporation.", fuzzy_threshold=0.5)
        assert m.tier is MatchTier.FUZZY
        assert m.score >= 0.5

    def test_fuzzy_below_threshold_returns_none(self):
        m = match_text(
            "Corporaton", "The Acme Corporation.", fuzzy_threshold=1.01
        )
        assert m.tier is MatchTier.NONE
        assert m.score > 0.0  # best score reported
        assert m.spans == []

    def test_fuzzy_span_within_source(self):
        src = "The Acme Corporaton is large."
        m = match_text("Corporaton", src)
        assert m.tier is MatchTier.FUZZY
        sp = m.spans[0]
        assert src[sp.start:sp.end] == sp.text

    def test_fuzzy_token_set(self):
        # Word-order reversal: token_set_ratio should help.
        m = match_text("Khan Ahmed", "Ahmed Khan is here", fuzzy_threshold=0.5)
        assert m.tier is MatchTier.FUZZY

    def test_no_words_in_source_returns_none(self):
        m = match_text("something", "   ")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.spans == []

    def test_empty_source(self):
        m = match_text("Acme", "")
        assert m.tier is MatchTier.NONE

    def test_default_threshold_strict(self):
        # Very dissimilar text should not be fuzzy-matched at default threshold.
        m = match_text("zzzzzzzz", "abcdefgh")
        assert m.tier is MatchTier.NONE
        assert m.score < 0.85

    def test_best_window_larger_than_value(self):
        src = "the quick brown fox jumps"
        m = match_text("quick brwn", src, fuzzy_threshold=0.5)
        assert m.tier is MatchTier.FUZZY


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------


class TestMatchNumber:
    def test_trivial_value(self):
        m = match_number(1, "source has 1 in it")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_zero_trivial(self):
        m = match_number(0, "source has 0 in it")
        assert m.tier is MatchTier.NONE

    def test_non_numeric(self):
        m = match_number("abc", "source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_exact_verbatim(self):
        m = match_number(4500, "The price is 4500 total.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        sp = m.spans[0]
        assert sp.text == "4500"

    def test_exact_string_value(self):
        m = match_number("4500", "value: 4500")
        assert m.tier is MatchTier.EXACT

    def test_exact_capped_at_five(self):
        src = " ".join(["4500"] * 10)
        m = match_number(4500, src)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_semantic_currency_format(self):
        m = match_number(4500, "The price is $4,500.00 total.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.spans
        assert m.note == "numeric value present, formatted differently"
        found_text = m.spans[0].text
        assert "4,500" in found_text

    def test_semantic_decimal_value(self):
        m = match_number("4500.0", "The value was 4,500.")
        assert m.tier is MatchTier.SEMANTIC

    def test_semantic_multiple_matches(self):
        src = "first $1,000 and again $1000 and 1,000"
        m = match_number(1000, src)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) >= 2

    def test_semantic_capped_at_five(self):
        src = " ".join(["$1,000"] * 10)
        m = match_number(1000, src)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5

    def test_tolerance_match(self):
        m = match_number(100, "value is 105", tolerance=10)
        assert m.tier is MatchTier.SEMANTIC
        assert m.spans[0].text == "105"

    def test_tolerance_boundary_inclusive(self):
        m = match_number(100, "value is 110", tolerance=10)
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance_out_of_range(self):
        m = match_number(100, "value is 111", tolerance=10)
        assert m.tier is MatchTier.NONE

    def test_not_found_note_no_numbers(self):
        m = match_number(999, "no digits here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"

    def test_not_found_note_nearest(self):
        m = match_number(1000, "the number is 42")
        assert m.tier is MatchTier.NONE
        assert m.note == "nearest number in source is '42'"

    def test_not_found_score_zero(self):
        m = match_number(1000, "42")
        assert m.score == 0.0

    def test_negative_numbers(self):
        m = match_number(-5, "the balance is -5 dollars")
        assert m.tier is MatchTier.EXACT

    def test_negative_semantic_accounting(self):
        m = match_number(-1234, "loss of ($1,234.00)")
        assert m.tier is MatchTier.SEMANTIC

    def test_float_value(self):
        m = match_number(1234.56, "amount $1,234.56")
        assert m.tier is MatchTier.SEMANTIC

    def test_none_like_trivial_via_str(self):
        # None -> str(None) = "None" -> trivial
        m = match_number(None, "None here")
        assert m.tier is MatchTier.NONE

    def test_empty_string_value(self):
        m = match_number("", "some 123")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------


class TestMatchDate:
    def test_unparseable_value(self):
        m = match_date("not a date", "2024-01-01")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_exact_verbatim(self):
        m = match_date("2024-01-15", "date: 2024-01-15 here")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        sp = m.spans[0]
        assert sp.text == "2024-01-15"

    def test_exact_string_value(self):
        m = match_date("15 January 2024", "meeting 15 January 2024")
        assert m.tier is MatchTier.EXACT

    def test_exact_capped_at_five(self):
        src = " ".join(["2024-01-15"] * 10)
        m = match_date("2024-01-15", src)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_semantic_different_format(self):
        m = match_date("2024-01-15", "meeting on 15/01/2024")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        assert m.spans
        assert "15/01/2024" in m.spans[0].text

    def test_semantic_month_name(self):
        m = match_date("2024-01-15", "dated Jan 15, 2024")
        assert m.tier is MatchTier.SEMANTIC

    def test_semantic_multiple_matches(self):
        src = "on 15/01/2024 and also January 15, 2024"
        m = match_date("2024-01-15", src)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) >= 2

    def test_not_found(self):
        m = match_date("2024-01-15", "the year was 1999")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_day_first_disambiguation(self):
        # 03/04/2026 is ambiguous without day_first
        m_ambig = match_date("03/04/2026", "date is 03/04/2026")
        # Verbatim match still hits exact regardless.
        assert m_ambig.tier is MatchTier.EXACT

    def test_day_first_semantic(self):
        m = match_date(
            "2026-04-03", "date 03/04/2026", day_first=True
        )
        assert m.tier is MatchTier.SEMANTIC
        assert "03/04/2026" in m.spans[0].text

    def test_month_first_semantic(self):
        m = match_date(
            "2026-03-04", "date 03/04/2026", day_first=False
        )
        assert m.tier is MatchTier.SEMANTIC

    def test_invalid_date_returns_none(self):
        m = match_date("31 February 2024", "31 February 2024")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_empty_value(self):
        m = match_date("", "2024-01-15")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# Full integration-style checks via to_dict()
# ---------------------------------------------------------------------------


class TestToDictIntegration:
    def test_exact_match_dict(self):
        m = match_text("Acme", "Acme is here")
        d = m.to_dict()
        assert d["tier"] == "exact"
        assert d["score"] == 1.0
        assert d["spans"] == [{"start": 0, "end": 4, "text": "Acme"}]

    def test_none_match_dict(self):
        m = match_text("", "anything")
        d = m.to_dict()
        assert d["tier"] == "none"
        assert d["score"] == 0.0
        assert d["spans"] == []

    def test_found_property_consistency(self):
        m = match_number(4500, "$4,500.00")
        assert m.found is True
        assert m.to_dict()["tier"] == "semantic"
