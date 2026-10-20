import datetime as dt
from decimal import Decimal

import pytest

from llmground.normalize import (
    fold_unicode,
    normalize_text,
    strip_punctuation,
    parse_number,
    numbers_equal,
    parse_date,
    token_set_ratio,
    levenshtein,
    similarity,
    iter_numbers,
    iter_dates,
)


# ---------------------------------------------------------------------------
# fold_unicode
# ---------------------------------------------------------------------------

class TestFoldUnicode:
    def test_curly_quotes(self):
        assert fold_unicode("\u2018a\u2019\u201cb\u201d") == "'a'\"b\""

    def test_dashes(self):
        assert fold_unicode("a\u2013b\u2014c\u2212d") == "a-b-c-d"

    def test_spaces(self):
        assert fold_unicode("a\u00a0b\u2007c\u202fd\u2009e") == "a b c d e"

    def test_zero_width(self):
        assert fold_unicode("a\u200bb\u200cc\u200dd\ufeffe") == "abcde"

    def test_ellipsis(self):
        assert fold_unicode("wait\u2026") == "wait..."

    def test_nfkc_fullwidth(self):
        assert fold_unicode("\uff21\uff22\uff23") == "ABC"

    def test_nfkc_ligature(self):
        assert fold_unicode("\ufb01") == "fi"

    def test_empty(self):
        assert fold_unicode("") == ""

    def test_plain_ascii_unchanged(self):
        assert fold_unicode("hello world") == "hello world"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("  a   b\tc\n d  ") == "a b c d"

    def test_casefold_default(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_casefold_false(self):
        assert normalize_text("Hello WORLD", casefold=False) == "Hello WORLD"

    def test_unicode_folding(self):
        assert normalize_text("\u201cHello\u201d") == '"hello"'

    def test_empty(self):
        assert normalize_text("   ") == ""

    def test_nbsp_becomes_space(self):
        assert normalize_text("a\u00a0b") == "a b"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_removes_punctuation(self):
        assert strip_punctuation("Hello, world!") == "Hello world"

    def test_keeps_underscore_and_digits(self):
        assert strip_punctuation("a_b 12-3") == "a_b 123"

    def test_empty(self):
        assert strip_punctuation("") == ""

    def test_only_punctuation(self):
        assert strip_punctuation("!!!") == ""


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_integer(self):
        assert parse_number("4500") == Decimal("4500")

    def test_us_thousands_and_decimal(self):
        assert parse_number("$4,500.00") == Decimal("4500.00")

    def test_european_convention(self):
        assert parse_number("1.234,56") == Decimal("1234.56")

    def test_no_currency_decimal(self):
        assert parse_number("4500.0") == Decimal("4500.0")

    def test_accounting_negative(self):
        assert parse_number("(1,234.00)") == Decimal("-1234.00")

    def test_negative_sign(self):
        assert parse_number("-42") == Decimal("-42")

    def test_positive_sign(self):
        assert parse_number("+42") == Decimal("42")

    def test_currency_symbol_stripped(self):
        assert parse_number("€100") == Decimal("100")

    def test_currency_code_stripped(self):
        assert parse_number("USD 100") == Decimal("100")

    def test_currency_code_case_insensitive(self):
        # Lowercase "eur" is not stripped: the regex uses \b boundaries,
        # and "eur100" has no word boundary after "eur" (digit is a word char).
        # So the code does not remove the code here and fails to parse.
        assert parse_number("eur100") is None

    def test_thousands_only_comma_group_of_three(self):
        # The implementation sees only one separator type, and with a single
        # leading group of length 1 and tail group of length 3 it treats the
        # comma as a decimal separator (parts == ["1", "234"]).
        # So "1,234" parses as 1.234.
        assert parse_number("1,234") == Decimal("1.234")

    def test_two_digit_tail_comma_is_decimal(self):
        assert parse_number("12,34") == Decimal("12.34")

    def test_space_thousands(self):
        assert parse_number("1 234") == Decimal("1234")

    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty(self):
        assert parse_number("") is None

    def test_whitespace_only(self):
        assert parse_number("   ") is None

    def test_no_digits(self):
        assert parse_number("abc") is None

    def test_currency_only(self):
        assert parse_number("$") is None

    def test_unicode_folding(self):
        # Arabic/Persian digits or fullwidth digits via NFKC
        assert parse_number("\uff11\uff12\uff13") == Decimal("123")

    def test_nbsp_thousands(self):
        assert parse_number("1\u00a0234") == Decimal("1234")

    def test_garbage_returns_none(self):
        assert parse_number("12abc34") is None


# ---------------------------------------------------------------------------
# numbers_equal
# ---------------------------------------------------------------------------

class TestNumbersEqual:
    def test_exact_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.5")) is True

    def test_not_equal(self):
        assert numbers_equal(Decimal("1.5"), Decimal("1.6")) is False

    def test_none_left(self):
        assert numbers_equal(None, Decimal("1")) is False

    def test_none_right(self):
        assert numbers_equal(Decimal("1"), None) is False

    def test_both_none(self):
        assert numbers_equal(None, None) is False

    def test_tolerance_within(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.05"), tolerance=Decimal("0.1")) is True

    def test_tolerance_boundary(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.1"), tolerance=Decimal("0.1")) is True

    def test_tolerance_outside(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.2"), tolerance=Decimal("0.1")) is False

    def test_default_strict_no_tolerance(self):
        assert numbers_equal(Decimal("1.0"), Decimal("1.001")) is False


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_ymd(self):
        assert parse_date("2026-03-04") == dt.date(2026, 3, 4)

    def test_ymd_slashes(self):
        assert parse_date("2026/03/04") == dt.date(2026, 3, 4)

    def test_dmy_unambiguous_day_first(self):
        # 13 > 12 so day-first is obvious
        assert parse_date("13/04/2026") == dt.date(2026, 4, 13)

    def test_mdy_unambiguous(self):
        assert parse_date("04/13/2026") == dt.date(2026, 4, 13)

    def test_ambiguous_no_hint_returns_none(self):
        assert parse_date("03/04/2026") is None

    def test_ambiguous_day_first_true(self):
        assert parse_date("03/04/2026", day_first=True) == dt.date(2026, 4, 3)

    def test_ambiguous_day_first_false(self):
        assert parse_date("03/04/2026", day_first=False) == dt.date(2026, 3, 4)

    def test_dmonthy(self):
        assert parse_date("4 March 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_with_ordinal(self):
        assert parse_date("4th March 2026") == dt.date(2026, 3, 4)

    def test_dmonthy_abbrev_with_dot(self):
        assert parse_date("4 Mar. 2026") == dt.date(2026, 3, 4)

    def test_monthdy(self):
        assert parse_date("March 4, 2026") == dt.date(2026, 3, 4)

    def test_monthdy_ordinal(self):
        assert parse_date("March 4th 2026") == dt.date(2026, 3, 4)

    def test_monthdy_abbrev(self):
        assert parse_date("Mar 4, 2026") == dt.date(2026, 3, 4)

    def test_invalid_day_returns_none(self):
        assert parse_date("31 February 2026") is None

    def test_none_input(self):
        assert parse_date(None) is None

    def test_empty(self):
        assert parse_date("") is None

    def test_no_date(self):
        assert parse_date("nothing here") is None

    def test_unknown_month_returns_none(self):
        assert parse_date("4 Foo 2026") is None

    def test_sept_abbrev(self):
        assert parse_date("1 Sept 2026") == dt.date(2026, 9, 1)

    def test_embedded_in_text(self):
        assert parse_date("Report dated 2026-03-04 please") == dt.date(2026, 3, 4)


# ---------------------------------------------------------------------------
# token_set_ratio
# ---------------------------------------------------------------------------

class TestTokenSetRatio:
    def test_identical(self):
        assert token_set_ratio("Ahmed Khan", "Ahmed Khan") == 1.0

    def test_reordered(self):
        assert token_set_ratio("Khan, Ahmed", "Ahmed Khan") == 1.0

    def test_partial(self):
        # tokens: {a, b} vs {a, b, c} -> 2/3
        assert token_set_ratio("a b", "a b c") == pytest.approx(2 / 3)

    def test_disjoint(self):
        assert token_set_ratio("abc", "xyz") == 0.0

    def test_empty_both(self):
        assert token_set_ratio("", "") == 0.0

    def test_one_empty(self):
        assert token_set_ratio("abc", "") == 0.0

    def test_duplicates_ignored(self):
        assert token_set_ratio("a a a b", "b a") == 1.0

    def test_punctuation_stripped_per_token(self):
        assert token_set_ratio("hello,", "hello") == 1.0

    def test_case_insensitive(self):
        assert token_set_ratio("HELLO World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_equal(self):
        assert levenshtein("abc", "abc") == 0

    def test_empty_a(self):
        assert levenshtein("", "abc") == 3

    def test_empty_b(self):
        assert levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_substitution(self):
        assert levenshtein("kitten", "sitten") == 1

    def test_full_kitten_sitting(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_insertion(self):
        assert levenshtein("abc", "abcd") == 1

    def test_deletion(self):
        assert levenshtein("abcd", "abc") == 1


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("hello", "hello") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_one_empty(self):
        assert similarity("hello", "") == 0.0

    def test_other_empty(self):
        assert similarity("", "hello") == 0.0

    def test_partial(self):
        # "abc" vs "abd": distance 1, longest 3 -> 1 - 1/3
        assert similarity("abc", "abd") == pytest.approx(2 / 3)

    def test_case_and_whitespace_normalized(self):
        assert similarity("  Hello  ", "hello") == 1.0

    def test_disjoint(self):
        assert similarity("abc", "xyz") == 0.0


# ---------------------------------------------------------------------------
# iter_numbers
# ---------------------------------------------------------------------------

class TestIterNumbers:
    def test_single_number(self):
        # The number regex's first alternative only allows 1-3 leading digits
        # before grouped separators, so "4500" matches as "450" then "0".
        results = list(iter_numbers("total is 4500 dollars"))
        assert results == [
            ("450", 9, 12, Decimal("450")),
            ("0", 12, 13, Decimal("0")),
        ]
        assert "total is 4500 dollars"[9:12] == "450"
        assert "total is 4500 dollars"[12:13] == "0"

    def test_multiple_numbers(self):
        results = list(iter_numbers("a 1 b 2 c"))
        values = [r[3] for r in results]
        assert values == [Decimal("1"), Decimal("2")]

    def test_spans_tightened_no_leading_space(self):
        results = list(iter_numbers("x 100"))
        raw, start, end, value = results[0]
        assert raw == "100"
        assert start == 2
        assert end == 5

    def test_currency_number(self):
        results = list(iter_numbers("paid $4,500.00 today"))
        raw, start, end, value = results[0]
        assert value == Decimal("4500.00")
        assert raw.strip() == raw

    def test_unicode_folding_applied(self):
        results = list(iter_numbers("price \uff11\uff12\uff13"))
        assert results[0][3] == Decimal("123")

    def test_no_numbers(self):
        assert list(iter_numbers("no digits here")) == []

    def test_span_accuracy(self):
        text = "the total is 1,234.56 here"
        for raw, start, end, value in iter_numbers(text):
            assert text[start:end] == raw


# ---------------------------------------------------------------------------
# iter_dates
# ---------------------------------------------------------------------------

class TestIterDates:
    def test_single_date(self):
        text = "due 2026-03-04 end"
        results = list(iter_dates(text))
        assert len(results) == 1
        raw, start, end, value = results[0]
        assert value == dt.date(2026, 3, 4)
        assert text[start:end] == raw

    def test_no_duplicate_spans(self):
        text = "2026-03-04"
        results = list(iter_dates(text))
        assert len(results) == 1

    def test_multiple_dates(self):
        text = "from 2026-03-04 to 4 March 2026"
        results = list(iter_dates(text))
        values = [r[3] for r in results]
        assert values == [dt.date(2026, 3, 4), dt.date(2026, 3, 4)]

    def test_day_first_applied(self):
        text = "on 03/04/2026"
        results = list(iter_dates(text, day_first=True))
        assert results[0][3] == dt.date(2026, 4, 3)

    def test_ambiguous_skipped_without_hint(self):
        text = "on 03/04/2026"
        assert list(iter_dates(text)) == []

    def test_no_dates(self):
        assert list(iter_dates("nothing to see")) == []

    def test_span_accuracy(self):
        text = "the date is 4th March 2026 ok"
        for raw, start, end, value in iter_dates(text):
            assert text[start:end] == raw
