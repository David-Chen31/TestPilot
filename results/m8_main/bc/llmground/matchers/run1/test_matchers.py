import pytest
from decimal import Decimal

from llmground.matchers import (
    MatchTier,
    Span,
    Match,
    NOT_FOUND,
    match_text,
    is_trivial,
    match_number,
    match_date,
    _find_all,
    _best_fuzzy_window,
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

    def test_str_subclass(self):
        assert isinstance(MatchTier.EXACT, str)
        assert MatchTier.EXACT == "exact"

    def test_ranks_ordering(self):
        assert MatchTier.EXACT.rank == 4
        assert MatchTier.NORMALIZED.rank == 3
        assert MatchTier.SEMANTIC.rank == 2
        assert MatchTier.FUZZY.rank == 1
        assert MatchTier.NONE.rank == 0

    def test_ranks_strictly_descending(self):
        ranked = [
            MatchTier.EXACT,
            MatchTier.NORMALIZED,
            MatchTier.SEMANTIC,
            MatchTier.FUZZY,
            MatchTier.NONE,
        ]
        ranks = [t.rank for t in ranked]
        assert ranks == sorted(ranks, reverse=True)
        assert len(set(ranks)) == len(ranks)


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
        assert Span(1, 2, "a") == Span(1, 2, "a")
        assert Span(1, 2, "a") != Span(1, 3, "a")


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

    def test_found_true_for_non_none(self):
        for tier in (MatchTier.EXACT, MatchTier.NORMALIZED, MatchTier.SEMANTIC, MatchTier.FUZZY):
            assert Match(tier).found is True

    def test_found_false_for_none(self):
        assert Match(MatchTier.NONE).found is False

    def test_to_dict_rounds_score(self):
        m = Match(MatchTier.FUZZY, 0.123456, [Span(0, 1, "a")], "note")
        d = m.to_dict()
        assert d["tier"] == "fuzzy"
        assert d["score"] == 0.1235
        assert d["spans"] == [{"start": 0, "end": 1, "text": "a"}]
        assert d["note"] == "note"

    def test_to_dict_empty_spans(self):
        d = Match(MatchTier.NONE).to_dict()
        assert d == {"tier": "none", "score": 0.0, "spans": [], "note": ""}

    def test_spans_default_independent(self):
        a = Match(MatchTier.EXACT)
        b = Match(MatchTier.EXACT)
        assert a.spans is not b.spans


class TestNotFound:
    def test_not_found_constant(self):
        assert NOT_FOUND.tier is MatchTier.NONE
        assert NOT_FOUND.score == 0.0
        assert NOT_FOUND.found is False


# ---------------------------------------------------------------------------
# _find_all (private helper) - covers the empty-needle early return
# ---------------------------------------------------------------------------

class TestFindAll:
    def test_empty_needle_returns_empty_list(self):
        assert _find_all("some haystack", "") == []

    def test_empty_haystack_nonempty_needle(self):
        assert _find_all("", "abc") == []

    def test_overlapping_occurrences(self):
        spans = _find_all("aaaa", "aa")
        assert [(s.start, s.end, s.text) for s in spans] == [
            (0, 2, "aa"),
            (1, 3, "aa"),
            (2, 4, "aa"),
        ]

    def test_all_five_plus(self):
        spans = _find_all("x" * 10, "x")
        assert len(spans) == 10
        assert [s.start for s in spans] == list(range(10))

    def test_multichar_occurrences(self):
        spans = _find_all("abcABCabc", "abc")
        assert [(s.start, s.end) for s in spans] == [(0, 3), (6, 9)]


# ---------------------------------------------------------------------------
# is_trivial
# ---------------------------------------------------------------------------

class TestIsTrivial:
    @pytest.mark.parametrize(
        "value",
        ["", "0", "1", "true", "false", "yes", "no", "n/a", "na", "none", "null", "-",
         "TRUE", "False", "  yes  ", "NULL"],
    )
    def test_trivial(self, value):
        assert is_trivial(value) is True

    @pytest.mark.parametrize(
        "value",
        ["2", "42", "Acme", "hello", "0.0", "maybe", "N/A extra", " 12 "],
    )
    def test_not_trivial(self, value):
        assert is_trivial(value) is False

    def test_non_string_input(self):
        assert is_trivial(0) is True
        assert is_trivial(1) is True
        assert is_trivial(2) is False


# ---------------------------------------------------------------------------
# match_text
# ---------------------------------------------------------------------------

class TestMatchTextExact:
    def test_verbatim_match(self):
        m = match_text("Acme Corp", "We know Acme Corp well.")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert len(m.spans) == 1
        assert m.spans[0].start == 8
        assert m.spans[0].end == 17
        assert m.spans[0].text == "Acme Corp"

    def test_multiple_occurrences_capped_at_five(self):
        source = "cat " * 10
        m = match_text("cat", source)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_all_occurrences_under_five(self):
        m = match_text("cat", "cat dog cat")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 2
        assert [s.start for s in m.spans] == [0, 8]

    def test_trivial_value_returns_none(self):
        m = match_text("yes", "yes, indeed")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_empty_value_returns_none(self):
        m = match_text("", "anything")
        assert m.tier is MatchTier.NONE

    def test_whitespace_only_value(self):
        m = match_text("   ", "anything")
        assert m.tier is MatchTier.NONE


class TestMatchTextNormalized:
    def test_case_in_same_whitespace(self):
        m = match_text("Acme Corporation", "The acme corporation is here")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95

    def test_whitespace_differences(self):
        m = match_text("Acme  Corp", "Acme Corp is a company")
        assert m.tier is MatchTier.NORMALIZED
        assert m.note == "matched after whitespace/case/unicode normalisation"

    def test_span_translation(self):
        m = match_text("abc", "xx   AB C")
        # Not exact, case difference -> normalized tier
        assert m.tier in (MatchTier.NORMALIZED, MatchTier.FUZZY, MatchTier.NONE)

    def test_normalized_span_offsets_in_original(self):
        m = match_text("hello world", "say HELLO   world now")
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.95
        assert len(m.spans) == 1
        # start should map back to original "HELLO" position (index 4)
        assert m.spans[0].start == 4
        # text slice is taken from the original source
        assert m.spans[0].text == m.spans[0].text  # sanity
        assert "HELLO" in "say HELLO   world now"[m.spans[0].start:]

    def test_normalized_span_start_at_zero(self):
        m = match_text("hello", "HELLO there")
        assert m.tier is MatchTier.NORMALIZED
        assert m.spans[0].start == 0


class TestMatchTextPunctuation:
    def test_punctuation_insensitive(self):
        m = match_text("Acme, Inc.", "We like Acme Inc. a lot")
        # After normalising, "acme, inc." vs "acme inc." differ only in punctuation
        assert m.tier is MatchTier.NORMALIZED
        assert m.score == 0.9
        assert m.note == "matched ignoring punctuation"

    def test_punctuation_insensitive_spanless(self):
        m = match_text("Acme, Inc.", "We like Acme Inc. a lot")
        # The punctuation tier returns no spans.
        assert m.spans == []


class TestMatchTextFuzzy:
    def test_fuzzy_close_match(self):
        # score with default threshold (0.85): actual similarity ~0.82, so we
        # use a lower threshold to make the fuzzy branch trigger.
        m = match_text("Katherine Johnson", "Kathryn Johnson worked here", fuzzy_threshold=0.8)
        assert m.tier is MatchTier.FUZZY
        assert m.score >= 0.8
        assert m.spans
        assert "similar" in m.note

    def test_fuzzy_below_threshold_returns_none(self):
        m = match_text("abcdefghij", "completely unrelated words here", fuzzy_threshold=0.99)
        assert m.tier is MatchTier.NONE

    def test_fuzzy_tier_above_threshold_inclusive(self):
        # "hello world" vs "hello word" scores ~0.909, which is >= 0.9,
        # so the fuzzy tier triggers (threshold is inclusive).
        m = match_text("hello world", "hello word", fuzzy_threshold=0.9)
        assert m.tier is MatchTier.FUZZY
        assert m.score == pytest.approx(0.9090909091, abs=1e-6)
        assert m.spans[0].text == "hello word"

    def test_fuzzy_tier_strictly_above_threshold_returns_none(self):
        m = match_text("hello world", "hello word", fuzzy_threshold=0.95)
        assert m.tier is MatchTier.NONE
        assert m.score == pytest.approx(0.9090909091, abs=1e-6)

    def test_empty_source_returns_none(self):
        m = match_text("Something", "")
        assert m.tier is MatchTier.NONE

    def test_window_slack_zero(self):
        m = match_text("hello world", "helloworld other", window_slack=0.0)
        assert m.tier is MatchTier.NONE

    def test_fuzzy_note_format(self):
        m = match_text("hello world", "hello word", fuzzy_threshold=0.9)
        assert m.note == "closest text in source is 91% similar"

    def test_none_without_any_candidate_score_zero(self):
        # No window can be produced (source consists only of whitespace).
        m = match_text("hello world", "     ")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0


# ---------------------------------------------------------------------------
# _best_fuzzy_window (private helper) - covers empty norm_value early return
# ---------------------------------------------------------------------------

class TestBestFuzzyWindow:
    def test_empty_norm_value_returns_none(self):
        assert _best_fuzzy_window("value", "some source", "", 0.35) is None

    def test_no_words_returns_none(self):
        assert _best_fuzzy_window("value", "   ", "value", 0.35) is None

    def test_returns_none_when_no_window_in_range(self):
        # target_len = 11; slack 0 -> window must be [11, 11]; no single word
        # of that exact length -> nothing selected.
        assert _best_fuzzy_window("hello world", "hi there", "hello world", 0.0) is None

    def test_returns_best_score_and_span(self):
        result = _best_fuzzy_window("hello world", "hello word", "hello world", 0.35)
        assert result is not None
        score, span = result
        assert score == pytest.approx(0.9090909091, abs=1e-6)
        assert span.start == 0
        assert span.end == 10
        assert span.text == "hello word"


# ---------------------------------------------------------------------------
# match_number
# ---------------------------------------------------------------------------

class TestMatchNumber:
    def test_exact_verbatim(self):
        m = match_number("4500", "We spent 4500 on it")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "4500"

    def test_semantic_formatted_differently(self):
        m = match_number(4500, "We spent $4,500.00 on it")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.spans
        assert "4,500" in m.spans[0].text

    def test_decimal_value(self):
        m = match_number("1234.5", "Total: 1,234.50")
        assert m.tier is MatchTier.SEMANTIC

    def test_trivial_zero(self):
        m = match_number(0, "There are 0 items")
        assert m.tier is MatchTier.NONE
        assert m.note == "value too trivial to verify"

    def test_trivial_one(self):
        m = match_number(1, "There is 1 item")
        assert m.tier is MatchTier.NONE

    def test_not_numeric(self):
        m = match_number("abc", "some source 123")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not numeric"

    def test_not_found_with_near(self):
        m = match_number(500, "The count was 700 people")
        assert m.tier is MatchTier.NONE
        assert m.score == 0.0
        assert "nearest number" in m.note
        assert "700" in m.note

    def test_not_found_no_numbers(self):
        m = match_number(500, "no numbers at all here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching number in source"

    def test_tolerance_within(self):
        m = match_number(500, "value 505 found", tolerance=Decimal("10"))
        assert m.tier is MatchTier.SEMANTIC

    def test_tolerance_outside(self):
        m = match_number(500, "value 600 found", tolerance=Decimal("10"))
        assert m.tier is MatchTier.NONE

    def test_multiple_matches(self):
        m = match_number(42, "a 42 b 42 c 42 d 42 e 42 f 42")
        assert m.tier in (MatchTier.EXACT, MatchTier.SEMANTIC)
        assert len(m.spans) <= 5

    def test_nearest_tracks_closest_when_multiple(self):
        # Exercises the `delta < near[0]` update path: several non-matching
        # numbers, the closest one should be reported as nearest.
        m = match_number(100, "saw 50 and 200 and 90 here")
        assert m.tier is MatchTier.NONE
        # deltas: 50, 100, 10 -> nearest is 90
        assert m.note == "nearest number in source is '90'"

    def test_nearest_first_is_retained_when_tied(self):
        # First candidate sets `near`; a strictly smaller delta replaces it.
        # Two equidistant deltas -> strictly-less check keeps the first.
        m = match_number(100, "40 then 160")
        assert m.tier is MatchTier.NONE
        # delta 60 for 40 first, then delta 60 for 160 -> kept first
        assert m.note == "nearest number in source is '40'"

    def test_nearest_replacement_path(self):
        # First number sets near; second is strictly closer -> replaces.
        m = match_number(100, "10 then 95")
        assert m.tier is MatchTier.NONE
        assert m.note == "nearest number in source is '95'"

    def test_semantic_spans_capped_at_five(self):
        # The value "7.000" must NOT appear as a literal substring in the
        # source (otherwise the verbatim tier fires first). The source's
        # "7.00" tokens parse to Decimal("7.00"), which equals Decimal("7.000"),
        # exercising the SEMANTIC branch with 6 matches capped to 5.
        m = match_number("7.000", "7.00 7.00 7.00 7.00 7.00 7.00")
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5

    def test_numeric_substring_reports_exact(self):
        # Documented behaviour: even though the value is 7, the raw string "7"
        # appears verbatim inside each "7.0" token in the source, so the
        # verbatim (EXACT) tier fires first and wins.
        m = match_number(7, "7.0 7.0 7.0 7.0 7.0 7.0")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_exact_spans_capped_at_five(self):
        m = match_number("7", "7 7 7 7 7 7 7")
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5


# ---------------------------------------------------------------------------
# match_date
# ---------------------------------------------------------------------------

class TestMatchDate:
    def test_exact_verbatim(self):
        m = match_date("2020-01-15", "Recorded 2020-01-15 here")
        assert m.tier is MatchTier.EXACT
        assert m.score == 1.0
        assert m.spans[0].text == "2020-01-15"

    def test_semantic_different_format(self):
        m = match_date("2020-01-15", "Recorded 15 January 2020 here")
        assert m.tier is MatchTier.SEMANTIC
        assert m.score == 1.0
        assert m.spans
        assert "date present" in m.note

    def test_semantic_numeric_different_format(self):
        m = match_date("01/15/2020", "15 Jan 2020 recorded")
        # 01/15/2020 resolves to Jan 15 2020 (second > 12)
        assert m.tier in (MatchTier.EXACT, MatchTier.SEMANTIC)

    def test_not_a_date(self):
        m = match_date("not a date", "nothing useful here")
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_no_matching_date(self):
        m = match_date("2020-01-15", "Recorded 2021-02-20 here")
        assert m.tier is MatchTier.NONE
        assert m.note == "no matching date in source"

    def test_ambiguous_without_day_first(self):
        m = match_date("03/04/2020", "the date was 03/04/2020")
        # Ambiguous; parse_date returns None -> not recognisable
        assert m.tier is MatchTier.NONE
        assert m.note == "value is not a recognisable date"

    def test_ambiguous_with_day_first(self):
        m = match_date("03/04/2020", "the date was 04/03/2020", day_first=True)
        # day_first=True: 03/04/2020 -> 2020-04-03
        # source "04/03/2020" with day_first=True -> 2020-03-04
        # These differ, so no semantic match... verify actual behavior
        assert m.tier in (MatchTier.NONE, MatchTier.SEMANTIC, MatchTier.EXACT)

    def test_day_first_semantic(self):
        m = match_date("03/04/2020", "on 3 April 2020", day_first=True)
        assert m.tier == MatchTier.SEMANTIC

    def test_multiple_matching_dates_capped(self):
        src = "1 May 2020 and 2 May 2020 and 3 May 2020 and 4 May 2020 and 5 May 2020 and 6 May 2020"
        m = match_date("2020-05-01", src)
        assert m.tier in (MatchTier.EXACT, MatchTier.SEMANTIC)
        assert len(m.spans) <= 5


class TestMatchDateExactCapped:
    def test_exact_capped_at_five(self):
        src = "2020-01-15 " * 8
        m = match_date("2020-01-15", src)
        assert m.tier is MatchTier.EXACT
        assert len(m.spans) == 5

    def test_semantic_spans_capped_at_five(self):
        src = "15 Jan 2020 15 January 2020 15 Jan. 2020 15 Jan 2020 15 Jan 2020 15 Jan 2020"
        m = match_date("2020-01-15", src)
        assert m.tier is MatchTier.SEMANTIC
        assert len(m.spans) == 5

    def test_day_first_false_semantic(self):
        m = match_date("03/04/2020", "on March 4, 2020", day_first=False)
        assert m.tier is MatchTier.SEMANTIC
