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
    def test_rank_ordering(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_rank_strictly_descending(self):
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

    def test_values(self):
        assert MatchTier.EXACT.value == "exact"
        assert MatchTier.NORMALIZED.value == "normalized"
        assert MatchTier.SEMANTIC.value == "semantic"
        assert MatchTier.FUZZY.value == "fuzzy"
        assert MatchTier.NONE.value == "none"

    def test_is_str_enum(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_to_dict(self):
        span = Span(start=3, end=8, text="hello")
        assert span.to_dict() == {"start": 3, "end": 8, "text": "hello"}

    def test_positions(self):
        span = Span(0, 4, "abcd")
        assert span.start == 0
        assert span.end == 4
        assert span.text == "abcd"


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


class TestMatch:
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

    def test_defaults(self):
        m = Match(MatchTier.EXACT)
        assert m.score == 0.0
        assert m.spans == []
        assert m.note == ""

    def test_to_dict(self):
        m = Match(
            MatchTier.EXACT,
            score=0.123456,
            spans=[Span(1, 3, "ab")],
            note="hi",
        )
        d = m.to_dict()
        assert d == {
            "tier": "exact",
            "score": 0.123456,
            "spans": [{"start": 1, "end": 3, "text": "ab"}],
            "note": "hi",
        }

    def test_to_dict_rounds_score(self):
        m = Match(MatchTier.FUZZY, score=0.123456789)
        assert m.to_dict()["score"] == 0.1235

    def test_spans_default_factory_independent(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        a.spans.append(Span(0, 1, "x"))
        assert b.spans == []

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
        ["", "  ", "0", "1", "true", "TRUE", " False ", "yes", "no",
         "n/a", "na", "none", "null", "-", "None", "NULL"],
    )
    def test_trivial_values(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize("value", ["2", "42", "Acme", "hello world", "12.5"])
    def test_non_trivial_values(self, value):
        assert is_trivial(value) is False

    def test_non_string(self):
        assert is_trivial(0) is False  # "0" -> trivial, wait
        # str(0).strip().casefold() == "0" which is trivial
        # Correcting: 0 as int stringifies to "0" and IS trivial
        # so test the actual behavior:
        assert is_trivial(0) is True
        assert is_trivial(1) is True
        assert is_trivial(42) is False


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------


class TestMatchTextExact:
    def test_simple_exact(self):
        m = match_text("Acme Corp", "We like Acme Corp a lot.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].start == 8
        assert m.spans[0].end == 17
        assert m.spans[0].text == "Acme Corp"

    def test_multiple_occurrences_capped_at_five(self):
        source = " ".join(["Word"] * 10)
        m = match_text("Word", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_multiple_occurrences_fewer_than_five(self):
        m = match_text("cat", "cat sat on cat mat cat")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 3
        assert [s.start for s in m.spans] == [0, 11, 19]


class TestMatchTextTrivial:
    @pytest.mark.parametrize("value", ["", "  ", "1", "yes", "n/a", "-"])
    def test_trivial_returns_none(self, value):
        m = match_text(value, "any source 1 yes n/a")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_does_not_look_for_spans(self):
        m = match_text("1", "there is a 1 here")
        assert m.tier is MatchTier.NONE
        assert m.spans == []


class TestMatchTextNormalized:
    def test_case_insensitive(self):
        m = match_text("Acme Corp", "we like ACME corp.")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert len(m.spans) == 1
        assert m.note == "matched after whitespace/case/unicode normalisation"

    def test_whitespace_collapse(self):
        m = match_text("hello world", "say   hello    world  now")
        assert m.tier is MatchTier.NORMALIZED
        assert m.spans[0].text.lower() == "hello    world"

    def test_punctuation_insensitive(self):
        m = match_text("Acme, Inc.", "The company Acme Inc is big.")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"


class TestMatchTextFuzzy:
    def test_fuzzy_typo(self):
        m = match_text(
            "Acme Corporation",
            "We like Acme Corporaton a lot and more text here.",
        )
        assert m.tier is MatchTier.FUZZY
        assert 0.85 <= m.score <= 1.0
        assert len(m.spans) == 1
        assert "closest text in source is" in m.note

    def test_fuzzy_threshold_custom_high_rejects(self):
        m = match_text(
            "Acme Corporation",
            "We like Acme Corporaton a lot.",
            fuzzy_threshold=0.99,
        )
        assert m.tier is MatchTier.NONE
        assert m.score < 0.99

    def test_fuzzy_threshold_low_accepts(self):
        m = match_text(
            "abcdefgh",
            "abcdefgx",
            fuzzy_threshold=0.5,
        )
        assert m.tier is MatchTier.FUZZY


class TestMatchTextNotFound:
    def test_completely_different(self):
        m = match_text("xyzzy", "completely unrelated document text")
        assert m.tier is MatchTier.NONE
        assert m.found is False

    def test_no_fuzzy_window_when_source_empty(self):
        m = match_text("something", "")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------


class TestMatchNumberExact:
    def test_verbatim(self):
        m = match_number("4500.0", "The value is 4500.0 here.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "4500.0"

    def test_verbatim_multiple(self):
        m = match_number("42", "42 and 42 again")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2


class TestMatchNumberSemantic:
    def test_currency_formatting(self):
        m = match_number("4500.0", "Total: $4,500.00 due today.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "numeric value present, formatted differently"
        assert len(m.spans) == 1

    def test_european_format(self):
        m = match_number("1234.56", "Betrag: 1.234,56 EUR.")
        assert m.tier is MatchTier.SEMANTIC

    def test_negative_accounting(self):
        m = match_number(-1234.56, "Value: (1,234.56) here.")
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance(self):
        from decimal import Decimal

        m = match_number(
            "1000", "approximately 1005 dollars", tolerance=Decimal("10")
        )
        assert m.tier is MatchTier.SEMANTIC


class TestMatchNumberNotFound:
    def test_not_present(self):
        m = match_number("999", "the source has 100 and 200 only")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert "nearest number in source is" in m.note

    def test_no_numbers_at_all(self):
        m = match_number("999", "no numbers here at all")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"

    def test_trivial_value(self):
        m = match_number("1", "there is a 1 here")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_not_numeric(self):
        m = match_number("abc", "source text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------


class TestMatchDateExact:
    def test_verbatim(self):
        m = match_date("2026-03-04", "The date 2026-03-04 appears.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "2026-03-04"


class TestMatchDateSemantic:
    def test_different_format(self):
        m = match_date("2026-03-04", "On March 4, 2026 something happened.")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.note == "date present, written in a different format"
        assert len(m.spans) == 1

    def test_day_first(self):
        m = match_date("2026-03-04", "date is 04/03/2026", day_first=True)
        assert m.tier is MatchTier.SEMANTIC


class TestMatchDateNotFound:
    def test_no_date(self):
        m = match_date("2026-03-04", "no dates in this text")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_unrecognisable_value(self):
        m = match_date("not a date", "some text")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"


# ---------------------------------------------------------------------------
# Span offsets and edge cases
# ---------------------------------------------------------------------------


class TestSpanOffsets:
    def test_span_slices_source(self):
        source = "prefix Acme suffix"
        m = match_text("Acme", source)
        span = m.spans[0]
        assert source[span.start:span.end] == span.text == "Acme"

    def test_multiline_source(self):
        source = "line one\nAcme Corp\nline three"
        m = match_text("Acme Corp", source)
        span = m.spans[0]
        assert source[span.start:span.end] == "Acme Corp"
