"""Tests for llmground.matchers."""

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


# ---------------------------------------------------------------------------
# MatchTier
# ---------------------------------------------------------------------------


class TestMatchTier:
    def test_rank_ordering(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_rank_ordering_is_strictly_decreasing(self):
        tiers = [
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        ]
        ranks = [t.rank for t in tiers]
        assert ranks == sorted(ranks, reverse=True)

    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"
        assert MatchTier.NONE.value == "none"

    def test_all_values(self):
        assert {t.value for t in MatchTier} == {
            "exact",
            "normalized",
            "semantic",
            "fuzzy",
            "none",
        }


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
        s = Span(0, 5, "hello")
        assert s.to_dict() == {"start": 0, "end": 5, "text": "hello"}

    def test_equality(self):
        assert Span(1, 2, "x") == Span(1, 2, "x")
        assert Span(1, 2, "x") != Span(1, 2, "y")


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


class TestMatch:
    def test_defaults(self):
        m = Match(MatchTier.NONE)
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_spans_default_is_not_shared(self):
        m1 = Match(MatchTier.NONE)
        m2 = Match(MatchTier.NONE)
        m1.spans.append(Span(0, 1, "a"))
        assert m2.spans == []

    def test_found_true(self):
        for tier in (
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
        ):
            assert Match(tier).found is True

    def test_found_false_for_none(self):
        assert Match(MatchTier.NONE).found is False

    def test_to_dict(self):
        m = Match(
            MatchTier.FUZZY,
            score=0.123456,
            spans=[Span(2, 5, "abc")],
            note="hi",
        )
        assert m.to_dict() == {
            "tier": "fuzzy",
            "score": 0.1235,
            "spans": [{"start": 2, "end": 5, "text": "abc"}],
            "note": "hi",
        }

    def test_to_dict_rounds_score(self):
        assert Match(MatchTier.EXACT, score=1.0).to_dict()["score"] == 1.0

    def test_not_found_singleton(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.spans == []
        assert NOT_FOUND.found is False


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------


class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na", "none", "null", "-"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["", " ", "TRUE", "False", "N/A", " None ", "-", "1", "0"],
    )
    def test_case_and_whitespace_insensitive(self, value):
        # Empty/whitespace strings and casefolded tokens are all treated as trivial.
        assert is_trivial(value) is True

    @pytest.mark.parametrize("value", ["2", "abc", "hello world", "12", "0.0", "Falsey"])
    def test_non_trivial(self, value):
        assert is_trivial(value) is False

    def test_accepts_non_string(self):
        assert is_trivial(1) is True
        assert is_trivial(0) is True
        assert is_trivial(None) is True
        assert is_trivial(42) is False


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------


class TestMatchTextExact:
    def test_exact_found(self):
        m = match_text("hello", "well hello there")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0] == Span(5, 10, "hello")

    def test_exact_multiple_spans(self):
        m = match_text("cat", "cat dog cat cat")
        assert m.tier is MatchTier.EXACT
        assert [s.start for s in m.spans] == [0, 8, 12]
        assert all(s.text == "cat" for s in m.spans)

    def test_exact_caps_spans_at_five(self):
        m = match_text("a1", "a1 " * 10)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_span_offsets_into_original(self):
        source = "prefix-HELLO-suffix"
        m = match_text("HELLO", source)
        assert m.tier is MatchTier.EXACT
        sp = m.spans[0]
        assert source[sp.start:sp.end] == "HELLO"


class TestMatchTextTrivial:
    @pytest.mark.parametrize("value", ["", "1", "true", "None", "n/a", "-"])
    def test_trivial_returns_none(self, value):
        m = match_text(value, "some source text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_whitespace_only(self):
        m = match_text("   ", "some source")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("Hello World", "the HELLO world here")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert m.note == "matched after whitespace/case/unicode normalisation"
        assert len(m.spans) == 1

    def test_whitespace_collapse(self):
        m = match_text("hello    world", "say hello world now")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_newlines_collapse(self):
        m = match_text("hello world", "a hello\n\nworld b")
        assert m.tier is MatchTier.NORMALIZED

    def test_span_end_bounded(self):
        # end is min(len(source), start + len(value)); ensure within bounds.
        source = "abc HELLO"
        m = match_text("hello", source)
        assert m.tier is MatchTier.NORMALIZED
        sp = m.spans[0]
        assert 0 <= sp.start <= len(source)
        assert 0 <= sp.end <= len(source)


class TestMatchTextPunctuation:
    def test_punctuation_insensitive_match(self):
        m = match_text("Acme Inc", "we use Acme, Inc. here")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        assert m.spans == []


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        # "recieve" vs "receive" — edit distance 1 over 7 chars = ~0.857 >= 0.85
        m = match_text("receive payment", "please recieve payment today")
        assert m.tier is MatchTier.FUZZY
        assert 0.85 <= m.score <= 1.0
        assert m.note.startswith("closest text in source is")

    def test_fuzzy_span_returns_text(self):
        m = match_text("receive payment", "please recieve payment today")
        assert m.tier is MatchTier.FUZZY
        sp = m.spans[0]
        assert sp.start <= sp.end
        assert sp.text == m.spans[0].text

    def test_below_threshold_returns_none(self):
        m = match_text("completely different", "nothing alike", fuzzy_threshold=0.99)
        assert m.tier is MatchTier.NONE

    def test_custom_threshold_low_accepts(self):
        m = match_text("hello", "helo world", fuzzy_threshold=0.5)
        assert m.tier is MatchTier.FUZZY

    def test_custom_threshold_high_rejects(self):
        m = match_text("hello", "helo world", fuzzy_threshold=0.99)
        # "helo" vs "hello" similarity is ~0.8, below 0.99
        assert m.tier is MatchTier.NONE

    def test_no_words_in_source(self):
        m = match_text("hello", "   ")
        assert m.tier is MatchTier.NONE

    def test_empty_source(self):
        m = match_text("hello", "")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# _find_all / normalised offset tracking (behaviour reached via public API)
# ---------------------------------------------------------------------------


class TestFindAllBoundaries:
    """Pin the exact span enumeration done by _find_all.

    Matches advance by one character per hit (``start = idx + 1``), so
    overlapping occurrences are all reported.  A change to the stride or the
    initial offset alters the reported start positions.
    """

    def test_overlapping_matches(self):
        m = match_text("aa", "aaaa")
        assert m.tier is MatchTier.EXACT
        # Overlapping occurrences: one at each of 0, 1, 2.
        assert [s.start for s in m.spans] == [0, 1, 2]
        assert [s.end for s in m.spans] == [2, 3, 4]
        assert [s.text for s in m.spans] == ["aa", "aa", "aa"]

    def test_overlapping_capped_at_five(self):
        m = match_text("aa", "a" * 8)  # "aa" occurrences at 0..6
        assert m.tier is MatchTier.EXACT
        assert [s.start for s in m.spans] == [0, 1, 2, 3, 4]

    def test_first_occurrence_at_zero(self):
        m = match_text("needle", "needle in haystack")
        assert m.spans[0] == Span(0, 6, "needle")


class TestMapToOriginal:
    """NORMALIZED spans are derived from a best-effort offset translation.

    These pin the exact start offsets produced by ``_map_to_original`` when the
    normalised text differs from the original in whitespace or case.
    """

    def test_leading_whitespace_start_zero(self):
        # "hello" is not a verbatim substring of "   hello" (case differs);
        # normalised match is at index 0 so the mapped start is 0.
        m = match_text("HELLO", "   hello")
        assert m.tier is MatchTier.NORMALIZED
        assert len(m.spans) == 1
        assert m.spans[0].start == 0
        assert m.spans[0].end == 5

    def test_single_char_prefix(self):
        # normalised "hello" starts at index 1 of "xhello" so original offset is 1.
        m = match_text("hello", "xHELLO")
        assert m.tier is MatchTier.NORMALIZED
        assert m.spans[0].start == 1
        assert m.spans[0].end == 6

    def test_multiple_collapsed_gaps(self):
        # Original has runs of spaces that normalise to single spaces; the walk
        # back to the original must consume the collapsed whitespace exactly.
        source = "one two   three   four"
        m = match_text("FOUR", source)
        assert m.tier is MatchTier.NORMALIZED
        assert m.spans[0].start == 16
        assert m.spans[0].end == 20

    def test_multiple_collapsed_gaps_single_gap(self):
        # A single collapsed gap: "ab   HELLO" -> normalised "ab hello".
        m = match_text("HELLO", "ab   hello")
        assert m.tier is MatchTier.NORMALIZED
        assert m.spans[0].start == 3
        assert m.spans[0].end == 8


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------


class TestMatchNumberExact:
    def test_verbatim_exact(self):
        m = match_number("4500", "total is 4500 dollars")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0] == Span(9, 13, "4500")

    def test_verbatim_multiple(self):
        m = match_number("42", "42 and 42")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2


class TestMatchNumberTrivial:
    @pytest.mark.parametrize("value", ["0", "1", "-", "n/a"])
    def test_trivial(self, value):
        m = match_number(value, "1 2 3")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_not_numeric(self):
        m = match_number("abc", "abc 123")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"


class TestMatchNumberSemantic:
    def test_formatted_differently(self):
        """Exercise the SEMANTIC tier without an intervening verbatim EXACT hit.

        The value ``"4500"`` is not a literal substring of ``"$4,500.00"``
        because of the comma, so the verbatim check misses, but the parsed
        numeric value equals the parsed number found in the source.
        """
        m = match_number("4500", "total is $4,500.00 only")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "numeric value present, formatted differently"
        assert len(m.spans) >= 1

    def test_european_format(self):
        m = match_number("1234.5", "wert ist 1.234,50 eur")
        assert m.tier is MatchTier.SEMANTIC

    def test_accounting_negative(self):
        # iter_numbers does not absorb the surrounding parentheses, so the
        # candidate is the positive 1234.56 which does not equal the target
        # -1234.56, and the parens block numeric detection entirely here.
        m = match_number("-1234.56", "adjustment (1,234.56)")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "no matching number in source"

    def test_spans_capped_at_five(self):
        m = match_number("7", "7 7 7 7 7 7 7")
        # 7 is not trivial, first is verbatim exact
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_semantic_spans_capped(self):
        # "8.0" is not a substring of "$8" so verbatim misses, hitting the
        # semantic path; spans are capped at five.
        source = "$8 " * 8
        m = match_number("8.0", source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5


class TestMatchNumberTolerance:
    def test_accepts_within_tolerance(self):
        m = match_number("100", "value is 99.5", tolerance=Decimal("1"))
        assert m.tier is MatchTier.SEMANTIC

    def test_boundary_inclusive(self):
        # Exactly at the tolerance boundary (abs == tolerance) is accepted.
        m = match_number("100", "value is 99", tolerance=Decimal("1"))
        assert m.tier is MatchTier.SEMANTIC

    def test_rejects_outside_tolerance(self):
        m = match_number("100", "value is 90", tolerance=Decimal("1"))
        assert m.tier is MatchTier.NONE

    def test_verbatim_wins_over_tolerance(self):
        # The verbatim "100" substring is present, so the exact tier is returned
        # before the tolerance/semantic path is reached.
        m = match_number("100", "value is 100.0", tolerance=Decimal("5"))
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0] == Span(9, 12, "100")


class TestMatchNumberNotFound:
    def test_no_number(self):
        m = match_number("999", "no digits here at all")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "no matching number in source"

    def test_nearest_note(self):
        m = match_number("500", "the answer is 42")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert m.note == "nearest number in source is '42'"

    def test_nearest_is_closest(self):
        m = match_number("1000", "values 200 and 950")
        assert m.tier is MatchTier.NONE
        assert "950" in m.note


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------


class TestMatchDateExact:
    def test_verbatim(self):
        m = match_date("2026-03-04", "due on 2026-03-04 sharp")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0] == Span(7, 17, "2026-03-04")

    def test_verbatim_multiple(self):
        m = match_date("2026-03-04", "2026-03-04 and 2026-03-04")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2


class TestMatchDateNotDate:
    def test_unrecognisable(self):
        m = match_date("not a date", "anything")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_ambiguous_refused(self):
        # 03/04/2026 is ambiguous without day_first.
        m = match_date("03/04/2026", "on 2026-03-04")
        assert m.tier is MatchTier.NONE


class TestMatchDateSemantic:
    def test_different_format(self):
        m = match_date("2026-03-04", "due on March 4, 2026")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        assert len(m.spans) >= 1

    def test_ymd_vs_word_month(self):
        m = match_date("2026-12-25", "Christmas is 25 December 2026")
        assert m.tier is MatchTier.SEMANTIC

    def test_day_first_does_not_match_april_4(self):
        # parse_date("03/04/2026", day_first=True) resolves to 3 April 2026,
        # so a source containing "4 April 2026" refers to a different date.
        m = match_date("03/04/2026", "on 4 April 2026", day_first=True)
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_day_first_matches_april_3(self):
        m = match_date("03/04/2026", "on 3 April 2026", day_first=True)
        assert m.tier is MatchTier.SEMANTIC

    def test_month_first_disambiguation(self):
        m = match_date("03/04/2026", "on March 4, 2026", day_first=False)
        assert m.tier is MatchTier.SEMANTIC

    def test_spans_capped_at_five(self):
        source = "; ".join(["4 March 2026"] * 8)
        m = match_date("2026-03-04", source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5


class TestMatchDateNotFound:
    def test_no_matching_date(self):
        m = match_date("2026-03-04", "on 2020-01-01")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"
        assert m.score == 0.0

    def test_source_without_dates(self):
        m = match_date("2026-03-04", "no dates at all here")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# Cross-cutting: dataclass / integration behaviors
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_exact_beats_fuzzy(self):
        m = match_text("hello", "hello and helo")
        assert m.tier is MatchTier.EXACT

    def test_number_returns_tuple_shape(self):
        m = match_number("12", "twelve is 12")
        assert isinstance(m, Match)
        # not trivial (12 not in _TRIVIAL), so verbatim exact
        assert m.tier is MatchTier.EXACT

    def test_match_date_date_object_input(self):
        m = match_date(_dt.date(2026, 3, 4), "due 2026-03-04")
        # str(date) is "2026-03-04", exact substring present
        assert m.tier is MatchTier.EXACT

    def test_all_matches_have_valid_span_bounds(self):
        m = match_text("world", "hello world")
        for sp in m.spans:
            assert 0 <= sp.start <= sp.end
