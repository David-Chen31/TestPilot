import re

import pytest

from funcy.strings import (
    re_iter,
    re_all,
    re_find,
    re_test,
    re_finder,
    re_tester,
    str_join,
    cut_prefix,
    cut_suffix,
)


# ---------------------------------------------------------------------------
# re_iter / re_all
# ---------------------------------------------------------------------------

def test_re_iter_no_groups():
    assert list(re_iter(r"\d+", "a1b22c333")) == ["1", "22", "333"]


def test_re_iter_one_group():
    assert list(re_iter(r"(\d+)", "a1b22c333")) == ["1", "22", "333"]


def test_re_iter_multiple_groups():
    assert list(re_iter(r"(\d+)([a-z])", "1a22b333c")) == [
        ("1", "a"),
        ("22", "b"),
        ("333", "c"),
    ]


def test_re_iter_named_groups_all_named():
    regex = r"(?P<num>\d+)(?P<letter>[a-z])"
    assert list(re_iter(regex, "1a22b")) == [
        {"num": "1", "letter": "a"},
        {"num": "22", "letter": "b"},
    ]


def test_re_iter_mixed_groups_returns_match_objects():
    regex = r"(?P<num>\d+)([a-z])"
    result = list(re_iter(regex, "1a22b"))
    assert len(result) == 2
    for m in result:
        assert isinstance(m, re.Match)
    assert result[0].group("num") == "1"
    assert result[0].group(2) == "a"


def test_re_iter_is_lazy_mapping():
    it = re_iter(r"\d", "123")
    assert hasattr(it, "__iter__")
    assert list(it) == ["1", "2", "3"]


def test_re_iter_no_matches():
    assert list(re_iter(r"\d+", "abc")) == []


def test_re_iter_flags():
    assert list(re_iter(r"[a-z]+", "AbC def", re.IGNORECASE)) == ["AbC", "def"]


def test_re_iter_accepts_compiled_pattern():
    pattern = re.compile(r"\d+")
    assert list(re_iter(pattern, "a1b22")) == ["1", "22"]


def test_re_iter_compiled_pattern_ignores_flags():
    pattern = re.compile(r"[a-z]+")
    # flags ignored for already-compiled patterns; pattern is case-sensitive,
    # so only lowercase runs survive ("b" inside "AbC" and the trailing "def")
    assert list(re_iter(pattern, "AbC def", re.IGNORECASE)) == ["b", "def"]


def test_re_all_basic():
    assert re_all(r"\d+", "a1b22c333") == ["1", "22", "333"]


def test_re_all_empty():
    assert re_all(r"\d+", "abc") == []


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

def test_re_find_no_groups():
    assert re_find(r"\d+", "a1b22c") == "1"


def test_re_find_one_group():
    assert re_find(r"(\d+)", "a1b22c") == "1"


def test_re_find_multiple_groups():
    assert re_find(r"(\d+)([a-z])", "x1a22b") == ("1", "a")


def test_re_find_named_groups_all_named():
    assert re_find(r"(?P<num>\d+)(?P<letter>[a-z])", "x1a") == {
        "num": "1",
        "letter": "a",
    }


def test_re_find_mixed_groups_returns_match():
    m = re_find(r"(?P<num>\d+)([a-z])", "x1a")
    assert isinstance(m, re.Match)
    assert m.group("num") == "1"
    assert m.group(2) == "a"


def test_re_find_no_match_returns_none():
    assert re_find(r"\d+", "abc") is None


def test_re_find_flags():
    assert re_find(r"[a-z]+", "AbC", re.IGNORECASE) == "AbC"


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

def test_re_test_true():
    assert re_test(r"\d+", "a1b") is True


def test_re_test_false():
    assert re_test(r"\d+", "abc") is False


def test_re_test_returns_bool():
    assert isinstance(re_test(r"\d+", "a1b"), bool)


def test_re_test_flags():
    assert re_test(r"[a-z]+", "ABC", re.IGNORECASE) is True
    assert re_test(r"[a-z]+", "ABC") is False


def test_re_test_compiled():
    pattern = re.compile(r"\d+")
    assert re_test(pattern, "a1b") is True


# ---------------------------------------------------------------------------
# re_finder
# ---------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r"\d+")
    assert callable(finder)
    assert finder("a1b22") == "1"


def test_re_finder_no_match():
    finder = re_finder(r"\d+")
    assert finder("abc") is None


def test_re_finder_with_groups():
    finder = re_finder(r"(\d+)-([a-z]+)")
    assert finder("x12-ab") == ("12", "ab")


def test_re_finder_flags():
    finder = re_finder(r"[a-z]+", re.IGNORECASE)
    assert finder("AbC") == "AbC"


def test_re_finder_compiled_ignores_flags():
    pattern = re.compile(r"[a-z]+")
    finder = re_finder(pattern, re.IGNORECASE)
    assert finder("ABC") is None


# ---------------------------------------------------------------------------
# re_tester
# ---------------------------------------------------------------------------

def test_re_tester_returns_callable():
    tester = re_tester(r"\d+")
    assert callable(tester)
    assert tester("a1b") is True
    assert tester("abc") is False


def test_re_tester_flags():
    tester = re_tester(r"[a-z]+", re.IGNORECASE)
    assert tester("ABC") is True


def test_re_tester_compiled():
    pattern = re.compile(r"\d+")
    tester = re_tester(pattern)
    assert tester("a1b") is True
    assert tester("abc") is False


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join(",", ["a", "b", "c"]) == "a,b,c"


def test_str_join_stringifies_items():
    assert str_join("-", [1, 2, 3]) == "1-2-3"


def test_str_join_empty_seq():
    assert str_join(",", []) == ""


def test_str_join_single_arg_returns_joined_sep_string():
    # With a single argument, str_join treats the argument as a sequence
    # and joins its items with '' (str_join('', seq) under the hood).
    # For a plain string, that yields the same string back.
    assert str_join(",") == ","
    assert str_join("abc") == "abc"


def test_str_join_single_arg_iterates_string_chars():
    # Each char is passed through str, then joined with '' -> identity.
    assert str_join("hello") == "hello"
    assert str_join("") == ""


def test_str_join_uses_sep_class_for_stringification():
    # For non-str sep (e.g. bytes), items are converted via sep.__class__.
    result = str_join(b",", [b"a", b"b", b"c"])
    assert result == b"a,b,c"


def test_str_join_with_tuple():
    assert str_join("+", ("x", "y")) == "x+y"


def test_str_join_single_element():
    assert str_join(",", ["only"]) == "only"


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix("foobar", "foo") == "bar"


def test_cut_prefix_absent():
    assert cut_prefix("foobar", "baz") == "foobar"


def test_cut_prefix_empty_prefix():
    assert cut_prefix("foobar", "") == "foobar"


def test_cut_prefix_whole_string():
    assert cut_prefix("foo", "foo") == ""


def test_cut_prefix_longer_than_string():
    assert cut_prefix("fo", "foo") == "fo"


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix("foobar", "bar") == "foo"


def test_cut_suffix_absent():
    assert cut_suffix("foobar", "baz") == "foobar"


def test_cut_suffix_empty_suffix():
    assert cut_suffix("foobar", "") == "foobar"


def test_cut_suffix_whole_string():
    assert cut_suffix("bar", "bar") == ""


def test_cut_suffix_longer_than_string():
    assert cut_suffix("ar", "bar") == "ar"
