import re
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
    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT.value == "exact"

    def test_all_members(self):
        assert set(MatchTier) == {
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        }

    @pytest.mark.parametrize(
        "tier,expected_rank",
        [
            (MatchTier.EXACT, 4),
            (MatchTier.NORMALIZED, 3),
            (MatchTier.SEMANTIC, 2),
            (MatchTier.FUZZY, 1),
            (MatchTier.NONE, 0),
        ],
    )
    def test_rank(self, tier, expected_rank):
        assert tier.rank == expected_rank

    def test_rank_order_strictly_descending(self):
        ranks = [t.rank for t in (
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        )]
        assert ranks == sorted(ranks, reverse=True)
        assert len(set(ranks)) == len(ranks)


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

class TestSpan:
    def test_fields(self):
        s = Span(2, 5, "abc")
        assert s.start == 2
        assert s.end == 5
        assert s.text == "abc"

    def test_to_dict(self):
        s = Span(0, 3, "foo")
        assert s.to_dict() == {"start": 0, "end": 3, "text": "foo"}

    def test_to_dict_roundtrip_length_consistency(self):
        s = Span(10, 17, "abcdefg")
        d = s.to_dict()
        assert d["end"] - d["start"] == len(d["text"])


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

    def test_found_property(self):
        assert Match(MatchTier.EXACT).found is True
        assert Match(MatchTier.NORMALIZED).found is True
        assert Match(MatchTier.SEMANTIC).found is True
        assert Match(MatchTier.FUZZY).found is True
        assert Match(MatchTier.NONE).found is False

    def test_to_dict(self):
        m = Match(
            MatchTier.FUZZY,
            score=0.87654,
            spans=[Span(1, 4, "abc")],
            note="close",
        )
        assert m.to_dict() == {
            "tier": "fuzzy",
            "score": 0.8765,
            "spans": [{"start": 1, "end": 4, "text": "abc"}],
            "note": "close",
        }

    def test_to_dict_rounds_score(self):
        assert Match(MatchTier.EXACT, score=1.0).to_dict()["score"] == 1.0
        # round() uses banker's rounding: 0.12345 -> 0.1235 (to even last digit)
        assert Match(MatchTier.FUZZY, score=0.12345).to_dict()["score"] == 0.1235
        assert Match(MatchTier.FUZZY, score=0.12346).to_dict()["score"] == 0.1235

    def test_default_spans_not_shared_between_instances(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []
        assert a.spans != b.spans

    def test_not_found_sentinel(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.found is False


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------

class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na",
         "none", "null", "-"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        [" TRUE ", "False", "Yes", "N/A", "None", "NULL"],
    )
    def test_trivial_is_case_and_whitespace_insensitive(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "0.5", "hello", "42", "Acme Inc", "2026-01-01"],
    )
    def test_non_trivial_values(self, value):
        assert is_trivial(value) is False

    def test_non_string_inputs_coerced(self):
        assert is_trivial(1) is True
        assert is_trivial(0) is True
        assert is_trivial(2) is False
        assert is_trivial(None) is True

    def test_returns_bool(self):
        assert isinstance(is_trivial("hello"), bool)
        assert isinstance(is_trivial("yes"), bool)


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------

class TestMatchTextExact:
    def test_exact_single_occurrence(self):
        m = match_text("Acme", "We love Acme Inc.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].start == 8
        assert m.spans[0].end == 12
        assert m.spans[0].text == "Acme"

    def test_exact_multiple_occurrences(self):
        m = match_text("Acme", "Acme and Acme again")
        assert m.tier is MatchTier.EXACT
        assert [s.start for s in m.spans] == [0, 9]
        assert all(s.text == "Acme" for s in m.spans)

    def test_exact_spans_capped_at_five(self):
        source = "Acme " * 10
        m = match_text("Acme", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_preserves_original_offsets(self):
        source = "prefix Acme suffix"
        m = match_text("Acme", source)
        assert m.spans[0].start == 7
        assert source[m.spans[0].start:m.spans[0].end] == "Acme"


class TestMatchTextTrivial:
    def test_empty_value(self):
        m = match_text("", "some source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_whitespace_value(self):
        m = match_text("   ", "some source")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    @pytest.mark.parametrize("value", ["1", "true", "N/A", "None", "-"])
    def test_trivial_tokens(self, value):
        m = match_text(value, f"source containing {value}")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_case_insensitive(self):
        m = match_text("  YES ", "yes it is")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("ACME", "acme corporation")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert len(m.spans) == 1
        assert m.note == "matched after whitespace/case/unicode normalisation"

    def test_whitespace_collapsed(self):
        m = match_text("Acme   Inc", "Acme Inc")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_span_within_source_bounds(self):
        source = "   Big   Co   "
        m = match_text("Big Co", source)
        assert m.tier is MatchTier.NORMALIZED
        span = m.spans[0]
        assert 0 <= span.start <= len(source)
        assert 0 <= span.end <= len(source)
        assert span.start <= span.end

    def test_unicode_folding(self):
        # ﬁ ligature folds to "fi" under NFKC
        m = match_text("find", "ﬁnd the value")
        assert m.tier in (MatchTier.NORMALIZED, MatchTier.SEMANTIC, MatchTier.FUZZY)
        assert m.found


class TestMatchTextPunctuation:
    def test_punctuation_insensitive(self):
        m = match_text("Acme, Inc.", "We work with Acme Inc today")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"
        assert m.spans == []

    def test_punctuation_removed_from_source(self):
        m = match_text("hello world", "hello, world!")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        m = match_text("Muhammad Ali", "Muhammad Aly was here")
        assert m.tier is MatchTier.FUZZY
        assert 0.0 < m.score < 1.0
        assert len(m.spans) == 1
        assert re.search(r"similar", m.note)

    def test_fuzzy_disabled_when_threshold_high(self):
        m = match_text("Muhammad Ali", "Muhammad Aly was here",
                       fuzzy_threshold=0.9999)
        assert m.tier is MatchTier.NONE

    def test_lower_threshold_finds_fuzzy(self):
        m = match_text("Muhammad Ali", "Muhammad Aly was here",
                       fuzzy_threshold=0.1)
        assert m.tier is MatchTier.FUZZY

    def test_custom_threshold_boundary(self):
        # Find the score first with a low threshold, then reuse it.
        probe = match_text("Muhammad Ali", "Muhammad Aly was here",
                           fuzzy_threshold=0.0)
        assert probe.tier is MatchTier.FUZZY
        score = probe.score
        at_score = match_text("Muhammad Ali", "Muhammad Aly was here",
                              fuzzy_threshold=score)
        assert at_score.tier is MatchTier.FUZZY
        above = match_text("Muhammad Ali", "Muhammad Aly was here",
                           fuzzy_threshold=min(score + 0.01, 1.0))
        if score + 0.01 <= 1.0:
            assert above.tier is MatchTier.NONE


class TestMatchTextNotFound:
    def test_no_match(self):
        m = match_text("completely unrelated", "nothing to see here")
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_no_match_score_reported(self):
        m = match_text("completely unrelated", "nothing to see here")
        # score reflects best fuzzy attempt, in [0, 1)
        assert 0.0 <= m.score < 1.0

    def test_empty_source(self):
        m = match_text("Acme", "")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------

class TestMatchNumberExact:
    def test_verbatim_exact(self):
        m = match_number("4500", "Total: 4500 units")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "4500"

    def test_exact_spans_capped_at_five(self):
        source = "100 " * 10
        m = match_number("100", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5


class TestMatchNumberSemantic:
    def test_currency_and_commas(self):
        m = match_number(4500.0, "The price is $4,500.00 today")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "numeric value present, formatted differently"
        # iter_numbers keeps the currency symbol in the raw span text
        assert m.spans[0].text == "$4,500.00"

    def test_decimal_equivalence(self):
        m = match_number(Decimal("12.50"), "amount 12.5 EUR")
        assert m.tier is MatchTier.SEMANTIC
        assert m.spans[0].text == "12.5"

    def test_multiple_matches_exact(self):
        # When the raw string appears verbatim in the source it reports EXACT.
        m = match_number(7, "we have 7 and also 7 more, plus $7")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3
        assert all(s.text == "7" for s in m.spans)

    def test_multiple_matches_semantic(self):
        # Use a value whose raw string ("7.0") does not appear verbatim so the
        # semantic path collects all numeric matches.
        m = match_number(Decimal("7.0"), "we have 7 and also 7 more, plus $7")
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 3
        # iter_numbers keeps the currency symbol attached to its number.
        assert [s.text for s in m.spans] == ["7", "7", "$7"]

    def test_semantic_span_offsets(self):
        source = "The price is $4,500.00 today"
        m = match_number(4500, source)
        span = m.spans[0]
        assert source[span.start:span.end] == span.text


class TestMatchNumberTolerance:
    def test_within_tolerance(self):
        # "100" appears verbatim in the source, so EXACT takes priority.
        m = match_number(100, "value is 100.5",
                         tolerance=Decimal("1"))
        assert m.tier is MatchTier.EXACT

    def test_within_tolerance_semantic(self):
        # Value's raw string "100.0" does not appear verbatim in the source, so
        # we reach the semantic path; 100.4 is within tolerance 1 of 100.
        m = match_number(Decimal("100.0"), "value is 100.4",
                         tolerance=Decimal("1"))
        assert m.tier is MatchTier.SEMANTIC
        assert m.spans[0].text == "100.4"

    def test_outside_tolerance(self):
        m = match_number(100, "value is 105",
                         tolerance=Decimal("1"))
        assert m.tier is MatchTier.NONE

    def test_tolerance_not_used_for_exact(self):
        m = match_number(100, "value is 100")
        assert m.tier is MatchTier.EXACT

    def test_tolerance_none_requires_exact_value(self):
        m = match_number(100, "value is 101")
        assert m.tier is MatchTier.NONE


class TestMatchNumberNotFound:
    def test_no_number(self):
        m = match_number(42, "no digits here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"

    def test_no_number_in_source_with_numbers(self):
        m = match_number(42, "the numbers are 1 and 2")
        assert m.tier is MatchTier.NONE
        assert "nearest number" in m.note
        assert "'1'" in m.note or "'2'" in m.note

    def test_nearest_reported(self):
        m = match_number(100, "we have 90 and 200")
        assert m.tier is MatchTier.NONE
        assert "'90'" in m.note

    def test_score_zero_on_not_found(self):
        m = match_number(42, "no digits")
        assert m.score == 0.0

    def test_trivial_value(self):
        m = match_number(1, "lots of 1s everywhere")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_zero(self):
        m = match_number(0, "there are 0 things")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_non_numeric_value(self):
        m = match_number("not a number", "abc 123")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_empty_string_value(self):
        m = match_number("", "source")
        assert m.tier is MatchTier.NONE


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------

class TestMatchDateExact:
    def test_verbatim_exact(self):
        m = match_date("2026-01-15", "Report dated 2026-01-15 today")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "2026-01-15"

    def test_exact_multiple(self):
        m = match_date("2026-01-15", "2026-01-15 and 2026-01-15")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2


class TestMatchDateSemantic:
    def test_month_name_format(self):
        m = match_date("2026-01-15", "dated January 15, 2026")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        assert m.spans

    def test_numeric_format(self):
        m = match_date("2026-01-15", "dated 15/01/2026")
        assert m.tier is MatchTier.SEMANTIC
        assert m.spans

    def test_day_first_disambiguation(self):
        m = match_date("01/02/2026", "dated 01/02/2026", day_first=True)
        # Source contains identical text -> exact
        assert m.tier is MatchTier.EXACT

    def test_day_first_semantic(self):
        m = match_date("2026-02-01", "dated 1 February 2026", day_first=True)
        assert m.tier is MatchTier.SEMANTIC

    def test_spans_capped_at_five_semantic(self):
        source = "on 2026-01-15 " * 10
        # "January 15, 2026" is not verbatim in the source, so semantic path.
        m = match_date("January 15, 2026", source)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5
        assert all(s.text == "2026-01-15" for s in m.spans)

    def test_unparseable_source_value(self):
        source = "on 2026-01-15 " * 10
        m2 = match_date("2026-01-15x", source)
        assert m2.tier is MatchTier.NONE


class TestMatchDateNotFound:
    def test_no_date(self):
        m = match_date("2026-01-15", "no dates in this text")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_different_date(self):
        m = match_date("2026-01-15", "dated January 16, 2026")
        assert m.tier is MatchTier.NONE

    def test_unparseable_value(self):
        m = match_date("not a date", "dated 2026-01-15")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_empty_value(self):
        m = match_date("", "dated 2026-01-15")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_ambiguous_date_not_guessed(self):
        # 03/04/2026 is ambiguous; with day_first=None parse_date returns None
        m = match_date("03/04/2026", "dated 03/04/2026")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_invalid_date_value(self):
        # 31 February does not exist
        m = match_date("2026-02-31", "dated 2026-02-28")
        assert m.tier is MatchTier.NONE
