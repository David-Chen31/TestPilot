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


# ----------------------------- re_iter / re_all -----------------------------

def test_re_iter_no_groups():
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_single_group():
    assert re_all(r'(\d+)', 'a1b22') == ['1', '22']


def test_re_iter_multiple_groups():
    assert re_all(r'(\d)(\w)', 'a1b2c') == [('1', 'b'), ('2', 'c')]


def test_re_iter_named_groups_only():
    result = re_all(r'(?P<x>\d+)-(?P<y>\d+)', '1-2 x 3-4')
    assert result == [
        {'x': '1', 'y': '2'},
        {'x': '3', 'y': '4'},
    ]


def test_re_iter_named_and_unnamed_groups_returns_match():
    result = re_all(r'(\d+)-(?P<y>\d+)', '1-2')
    assert len(result) == 1
    m = result[0]
    assert isinstance(m, re.Match)
    assert m.group(1) == '1'
    assert m.group('y') == '2'


def test_re_iter_accepts_compiled_regex():
    pattern = re.compile(r'\d+')
    assert re_all(pattern, 'a1b2') == ['1', '2']


def test_re_iter_flags():
    assert re_all('hello', 'HELLO world', re.IGNORECASE) == ['HELLO']


def test_re_iter_returns_iterator():
    it = re_iter(r'\d+', '1 2')
    assert not isinstance(it, list)
    assert list(it) == ['1', '2']


def test_re_iter_no_matches():
    assert re_all(r'\d+', 'abc') == []


# ----------------------------- re_find --------------------------------------

def test_re_find_no_match_returns_none():
    assert re_find(r'\d+', 'abc') is None


def test_re_find_no_groups():
    assert re_find(r'\d+', 'a12b') == '12'


def test_re_find_single_group():
    assert re_find(r'(\d+)', 'a12b') == '12'


def test_re_find_multiple_groups():
    assert re_find(r'(\d)(\w)', 'a1b') == ('1', 'b')


def test_re_find_named_groups_only():
    assert re_find(r'(?P<a>\d+)-(?P<b>\d+)', 'x1-2y') == {'a': '1', 'b': '2'}


def test_re_find_mixed_groups_returns_match():
    m = re_find(r'(\d+)-(?P<b>\d+)', 'x1-2y')
    assert isinstance(m, re.Match)
    assert m.group(1) == '1'


def test_re_find_with_flags():
    assert re_find('abc', '--ABC--', re.IGNORECASE) == 'ABC'


# ----------------------------- re_test --------------------------------------

def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abc') is False


def test_re_test_accepts_compiled():
    assert re_test(re.compile(r'\d+'), '1') is True


def test_re_test_flags():
    assert re_test('abc', 'ABC', re.IGNORECASE) is True


# ----------------------------- re_finder / re_tester ------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert finder('a12b') == '12'
    assert finder('abc') is None


def test_re_finder_with_flags():
    finder = re_finder('abc', re.IGNORECASE)
    assert finder('--ABC--') == 'ABC'


def test_re_finder_accepts_compiled():
    finder = re_finder(re.compile(r'(\d+)'))
    assert finder('a5b') == '5'


def test_re_tester_returns_callable():
    tester = re_tester(r'\d+')
    assert tester('a1') is True
    assert tester('ab') is False


def test_re_tester_with_flags():
    tester = re_tester('abc', re.IGNORECASE)
    assert tester('ABC') is True
    assert tester('xyz') is False


def test_re_tester_accepts_compiled():
    tester = re_tester(re.compile(r'\d+'))
    assert tester('1') is True


# ----------------------------- str_join -------------------------------------

def test_str_join_basic():
    assert str_join(',', ['a', 'b', 'c']) == 'a,b,c'


def test_str_join_empty_seq():
    assert str_join(',', []) == ''


def test_str_join_forces_stringification():
    assert str_join('-', [1, 2, 3]) == '1-2-3'


def test_str_join_mixed_types():
    assert str_join(' ', ['x', 1, 2.5]) == 'x 1 2.5'


def test_str_join_single_arg_is_join_of_characters():
    # seq is EMPTY -> str_join('', sep): joins characters of sep with ''
    assert str_join('abc') == 'abc'


def test_str_join_single_arg_non_string_raises():
    with pytest.raises(TypeError):
        str_join(123)


def test_str_join_uses_sep_class():
    # sep.__class__ is used as the str conversion
    assert str_join(',', ['a', 'b']) == 'a,b'


# ----------------------------- cut_prefix / cut_suffix ---------------------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'bar') == 'foobar'


def test_cut_prefix_empty_prefix():
    assert cut_prefix('foobar', '') == 'foobar'


def test_cut_prefix_whole_string():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'foo') == 'foobar'


def test_cut_suffix_empty_suffix():
    assert cut_suffix('foobar', '') == 'foobar'


def test_cut_suffix_whole_string():
    assert cut_suffix('foo', 'foo') == ''


def test_cut_prefix_with_class_name_pattern():
    # Matches the naming pattern used in funcy.objects to strip a class prefix.
    assert cut_prefix('MyClass__foo', '%s__' % 'MyClass') == 'foo'
