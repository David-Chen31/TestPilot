import re

import pytest

from funcy.primitives import EMPTY
from funcy.strings import (
    cut_prefix,
    cut_suffix,
    re_all,
    re_find,
    re_finder,
    re_iter,
    re_test,
    re_tester,
    str_join,
)


# ---------------------------------------------------------------------------
# re_iter / re_all
# ---------------------------------------------------------------------------

def test_re_iter_no_groups_returns_strings():
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_single_group_returns_group():
    assert list(re_iter(r'(\d+)', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_multiple_groups_returns_tuples():
    assert list(re_iter(r'(\d)(\d)', 'a12b34')) == [('1', '2'), ('3', '4')]


def test_re_iter_named_groups_only_returns_dicts():
    # groups == len(groupindex) and all groups are named -> groupdict
    assert list(re_iter(r'(?P<a>\d)(?P<b>\d)', 'a12b34')) == [
        {'a': '1', 'b': '2'},
        {'a': '3', 'b': '4'},
    ]


def test_re_iter_mixed_groups_returns_match_objects():
    # one named and one unnamed -> falls into the fallback branch
    result = list(re_iter(r'(?P<a>\d)(\d)', 'a12b34'))
    assert len(result) == 2
    assert all(isinstance(m, re.Match) for m in result)
    assert [m.group(0) for m in result] == ['12', '34']


def test_re_iter_is_lazy_map():
    it = re_iter(r'\d', '123')
    assert iter(it) is it  # map object


def test_re_iter_with_flags():
    assert list(re_iter(r'abc', 'ABC abc', re.I)) == ['ABC', 'abc']


def test_re_iter_accepts_compiled_pattern():
    pattern = re.compile(r'\d+')
    assert list(re_iter(pattern, 'a1b22')) == ['1', '22']


def test_re_all_basic():
    assert re_all(r'\w+', 'hello world') == ['hello', 'world']


def test_re_all_empty():
    assert re_all(r'\d+', 'no digits') == []


# ---------------------------------------------------------------------------
# re_find / re_test
# ---------------------------------------------------------------------------

def test_re_find_no_groups():
    assert re_find(r'\d+', 'abc123def') == '123'


def test_re_find_single_group():
    assert re_find(r'(\d+)', 'abc123def') == '123'


def test_re_find_multiple_groups():
    assert re_find(r'(\d)(\d)', 'abc123') == ('1', '2')


def test_re_find_named_groups_only():
    assert re_find(r'(?P<a>\d)(?P<b>\d)', 'ab12') == {'a': '1', 'b': '2'}


def test_re_find_no_match_returns_none():
    assert re_find(r'\d+', 'abcdef') is None


def test_re_find_with_flags():
    assert re_find(r'abc', 'XYZABC', re.I) == 'ABC'


def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abcdef') is False


def test_re_test_with_flags():
    assert re_test(r'abc', 'XYZABC', re.I) is True
    assert re_test(r'abc', 'XYZABC') is False


def test_re_test_returns_bool_not_match():
    result = re_test(r'\d', '1')
    assert result is True


# ---------------------------------------------------------------------------
# re_finder / re_tester
# ---------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert finder('abc123') == '123'
    assert finder('abc') is None


def test_re_finder_with_groups():
    finder = re_finder(r'(\d)(\d)')
    assert finder('x12') == ('1', '2')
    assert finder('x') is None


def test_re_finder_with_flags():
    finder = re_finder(r'abc', re.I)
    assert finder('XYZABC') == 'ABC'


def test_re_finder_accepts_compiled():
    finder = re_finder(re.compile(r'\d+'))
    assert finder('a42') == '42'


def test_re_tester_returns_callable_predicate():
    tester = re_tester(r'\d+')
    assert tester('abc123') is True
    assert tester('abc') is False


def test_re_tester_with_flags():
    tester = re_tester(r'abc', re.I)
    assert tester('ABC') is True


def test_re_tester_accepts_compiled():
    tester = re_tester(re.compile(r'\d'))
    assert tester('a1') is True
    assert tester('a') is False


def test_re_finder_reused_over_multiple_strings():
    finder = re_finder(r'(\w+)@(\w+)')
    assert finder('me@here') == ('me', 'here')
    assert finder('nothing') is None
    assert finder('a@b') == ('a', 'b')


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join(',', ['a', 'b', 'c']) == 'a,b,c'


def test_str_join_numeric_items_stringified():
    assert str_join('-', [1, 2, 3]) == '1-2-3'


def test_str_join_mixed_types():
    assert str_join(', ', [1, 'two', 3.0]) == '1, two, 3.0'


def test_str_join_single_arg_uses_empty_sep():
    assert str_join(['a', 'b', 'c']) == 'abc'


def test_str_join_single_arg_single_item():
    assert str_join(['x']) == 'x'


def test_str_join_single_arg_empty_sequence():
    assert str_join([]) == ''


def test_str_join_empty_sequence_with_sep():
    assert str_join('+', []) == ''


def test_str_join_uses_sep_class_for_str_items():
    # sep is a str subclass; items should be coerced through sep.__class__
    class S(str):
        pass

    sep = S('-')
    result = str_join(sep, [1, 2])
    assert result == '1-2'


def test_str_join_explicit_empty_sep():
    assert str_join('', ['a', 'b']) == 'ab'


def test_str_join_with_generator():
    assert str_join(',', (str(i) for i in range(3))) == '0,1,2'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'baz') == 'foobar'


def test_cut_prefix_full_string():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_prefix_empty_prefix():
    assert cut_prefix('foo', '') == 'foo'


def test_cut_prefix_longer_prefix_than_string():
    assert cut_prefix('ab', 'abcdef') == 'ab'


def test_cut_prefix_not_at_start():
    assert cut_prefix('barfoo', 'foo') == 'barfoo'


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'baz') == 'foobar'


def test_cut_suffix_full_string():
    assert cut_suffix('bar', 'bar') == ''


def test_cut_suffix_empty_suffix_returns_original():
    # guarded by `suffix and ...`
    assert cut_suffix('foo', '') == 'foo'


def test_cut_suffix_longer_suffix_than_string():
    assert cut_suffix('ab', 'abcdef') == 'ab'


def test_cut_suffix_not_at_end():
    assert cut_suffix('foobar', 'foo') == 'foobar'


def test_cut_suffix_single_char():
    assert cut_suffix('abc', 'c') == 'ab'


# ---------------------------------------------------------------------------
# _prepare via public API: compiled pattern + flags interaction
# ---------------------------------------------------------------------------

def test_compiled_pattern_flags_argument_passed_through():
    # When a compiled pattern is given, flags arg is passed to _prepare but
    # not applied (compile is skipped) -- behavior check via public API
    pattern = re.compile(r'abc', re.I)
    assert re_find(pattern, 'XYZABC') == 'ABC'


def test_invalid_regex_raises():
    with pytest.raises(re.error):
        re_find('(', 'abc')


def test_invalid_regex_in_finder_raises():
    with pytest.raises(re.error):
        re_finder('(')


def test_invalid_regex_in_tester_raises():
    with pytest.raises(re.error):
        re_tester('(')


# ---------------------------------------------------------------------------
# EMPTY sentinel semantics for str_join
# ---------------------------------------------------------------------------

def test_str_join_EMPTY_is_default_sentinel():
    # Passing EMPTY explicitly as seq is treated as "not provided"
    assert str_join(EMPTY, ['a', 'b']) == 'ab'


def test_str_join_sep_default_not_used_when_seq_given():
    assert str_join(',', ['a']) == 'a'
