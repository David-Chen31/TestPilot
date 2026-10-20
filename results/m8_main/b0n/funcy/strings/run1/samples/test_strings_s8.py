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

def test_re_iter_no_groups_returns_full_matches():
    assert list(re_iter(r"\d+", "a1b22c333")) == ["1", "22", "333"]


def test_re_iter_single_group_returns_group():
    assert list(re_iter(r"(\d+)", "a1b22c333")) == ["1", "22", "333"]


def test_re_iter_multiple_groups_returns_tuples():
    assert list(re_iter(r"(\w)(\d)", "a1 b2")) == [("a", "1"), ("b", "2")]


def test_re_iter_named_groups_only_returns_dicts():
    regex = r"(?P<letter>\w)(?P<digit>\d)"
    assert list(re_iter(regex, "a1 b2")) == [
        {"letter": "a", "digit": "1"},
        {"letter": "b", "digit": "2"},
    ]


def test_re_iter_mixed_named_and_unnamed_groups_returns_match_objects():
    regex = r"(\w)(?P<digit>\d)"
    result = list(re_iter(regex, "a1 b2"))
    assert len(result) == 2
    for m in result:
        assert isinstance(m, re.Match)
    assert result[0].group(1) == "a"
    assert result[0].group("digit") == "1"


def test_re_iter_empty_when_no_match():
    assert list(re_iter(r"x", "abc")) == []


def test_re_iter_accepts_compiled_pattern():
    pattern = re.compile(r"\d+")
    assert list(re_iter(pattern, "a1b22")) == ["1", "22"]


def test_re_iter_uses_flags_when_compiling():
    assert list(re_iter(r"[a-z]+", "AbC", re.IGNORECASE)) == ["AbC"]


def test_re_all_is_list():
    result = re_all(r"\d+", "a1b22c333")
    assert result == ["1", "22", "333"]


def test_re_all_with_no_match():
    assert re_all(r"x", "abc") == []


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

def test_re_find_no_groups():
    assert re_find(r"\d+", "abc123def") == "123"


def test_re_find_single_group():
    assert re_find(r"(\d+)", "abc123def") == "123"


def test_re_find_multiple_groups():
    assert re_find(r"(\w)(\d)", "a1 b2") == ("a", "1")


def test_re_find_named_groups():
    assert re_find(r"(?P<letter>\w)(?P<digit>\d)", "a1 b2") == {
        "letter": "a",
        "digit": "1",
    }


def test_re_find_mixed_groups_returns_match():
    m = re_find(r"(\w)(?P<digit>\d)", "a1")
    assert isinstance(m, re.Match)
    assert m.group("digit") == "1"


def test_re_find_no_match_returns_none():
    assert re_find(r"\d+", "abc") is None


def test_re_find_with_flags():
    assert re_find(r"[a-z]+", "AbC", re.IGNORECASE) == "AbC"


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

def test_re_test_true():
    assert re_test(r"\d+", "abc123") is True


def test_re_test_false():
    assert re_test(r"\d+", "abc") is False


def test_re_test_with_flags():
    assert re_test(r"[a-z]+", "ABC", re.IGNORECASE) is True


def test_re_test_returns_bool_not_match():
    result = re_test(r"a", "a")
    assert result is True
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# re_finder / re_tester
# ---------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r"\d+")
    assert finder("a1b22") == "1"
    assert finder("nothing") is None


def test_re_finder_single_group():
    finder = re_finder(r"(\d+)")
    assert finder("abc123") == "123"


def test_re_finder_with_flags():
    finder = re_finder(r"[a-z]+", re.IGNORECASE)
    assert finder("ABC") == "ABC"


def test_re_finder_accepts_compiled_regex():
    finder = re_finder(re.compile(r"\d+"))
    assert finder("x9") == "9"


def test_re_tester_returns_callable():
    tester = re_tester(r"\d+")
    assert tester("a1") is True
    assert tester("abc") is False


def test_re_tester_with_flags():
    tester = re_tester(r"[a-z]+", re.IGNORECASE)
    assert tester("ABC") is True


def test_re_tester_accepts_compiled_regex():
    tester = re_tester(re.compile(r"\d+"))
    assert tester("x9") is True


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join("-", ["a", "b", "c"]) == "a-b-c"


def test_str_join_empty_sequence():
    assert str_join("-", []) == ""


def test_str_join_forces_stringification():
    assert str_join("-", [1, 2, 3]) == "1-2-3"


def test_str_join_single_arg_treats_seq_as_sep():
    # With one arg, seq defaults to EMPTY, so 'abc' is treated as sep
    # and joined with ''.
    assert str_join("abc") == "abc"


def test_str_join_single_arg_single_char():
    assert str_join("x") == "x"


def test_str_join_with_iterable_generator():
    assert str_join(",", (i for i in range(3))) == "0,1,2"


def test_str_join_with_mixed_types():
    assert str_join("|", [1, "a", 2.5]) == "1|a|2.5"


def test_str_join_subclass_of_str_sep():
    class MyStr(str):
        pass

    # sep.__class__ is used for stringification; MyStr("1") == "1"
    assert str_join(MyStr("-"), [1, 2]) == "1-2"


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix("foobar", "foo") == "bar"


def test_cut_prefix_absent():
    assert cut_prefix("foobar", "bar") == "foobar"


def test_cut_prefix_empty_prefix():
    assert cut_prefix("foobar", "") == "foobar"


def test_cut_prefix_whole_string():
    assert cut_prefix("foo", "foo") == ""


def test_cut_prefix_longer_than_string():
    assert cut_prefix("foo", "foobar") == "foo"


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix("foobar", "bar") == "foo"


def test_cut_suffix_absent():
    assert cut_suffix("foobar", "foo") == "foobar"


def test_cut_suffix_empty_suffix():
    assert cut_suffix("foobar", "") == "foobar"


def test_cut_suffix_whole_string():
    assert cut_suffix("foo", "foo") == ""


def test_cut_suffix_longer_than_string():
    assert cut_suffix("foo", "foobar") == "foo"


def test_cut_suffix_empty_string_and_empty_suffix():
    assert cut_suffix("", "") == ""
