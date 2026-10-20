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

def test_re_iter_no_groups_yields_full_match():
    assert re_all(r'\d+', 'a1b22c333') == ['1', '22', '333']
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_single_group_yields_group():
    assert re_all(r'(\d+)', 'a1b22c333') == ['1', '22', '333']


def test_re_iter_multiple_groups_yields_tuples():
    assert re_all(r'(\w)(\d+)', 'a1 b22 c333') == [
        ('a', '1'), ('b', '22'), ('c', '333')
    ]


def test_re_iter_named_groups_only_yields_dict():
    result = re_all(r'(?P<letter>\w)(?P<num>\d+)', 'a1 b22')
    assert result == [
        {'letter': 'a', 'num': '1'},
        {'letter': 'b', 'num': '22'},
    ]


def test_re_iter_mixed_named_and_positional_groups_yields_match():
    pattern = r'(\w)(?P<num>\d+)'
    matches = re_all(pattern, 'a1 b22')
    assert len(matches) == 2
    assert all(isinstance(m, re.Match) for m in matches)
    assert matches[0].group(0) == 'a1'
    assert matches[1].group('num') == '22'


def test_re_iter_with_compiled_pattern():
    pattern = re.compile(r'\d+')
    assert re_all(pattern, 'a1b22') == ['1', '22']


def test_re_iter_with_flags():
    # case-insensitive via flags
    assert re_all('abc', 'ABC abc', re.IGNORECASE) == ['ABC', 'abc']


def test_re_iter_empty_when_no_match():
    assert re_all(r'\d+', 'no digits here') == []
    assert list(re_iter(r'\d+', '')) == []


def test_re_iter_returns_iterator():
    it = re_iter(r'\d+', 'a1b2')
    # map object is a lazy iterator
    assert iter(it) is it
    assert list(it) == ['1', '2']


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

def test_re_find_no_groups():
    assert re_find(r'\d+', 'abc123def') == '123'


def test_re_find_single_group():
    assert re_find(r'(\d+)', 'abc123def') == '123'


def test_re_find_multiple_groups():
    assert re_find(r'(\w)(\d+)', 'xa12b') == ('a', '12')


def test_re_find_named_groups():
    assert re_find(r'(?P<x>\w)(?P<y>\d+)', 'qa7') == {'x': 'a', 'y': '7'}


def test_re_find_mixed_groups_returns_match():
    m = re_find(r'(\w)(?P<y>\d+)', 'qa7z')
    assert isinstance(m, re.Match)
    assert m.group(0) == 'a7'


def test_re_find_no_match_returns_none():
    assert re_find(r'\d+', 'no digits') is None


def test_re_find_with_flags():
    assert re_find('abc', 'xxABCyy', re.IGNORECASE) == 'ABC'


def test_re_find_with_compiled_pattern():
    assert re_find(re.compile(r'\d+'), 'a42b') == '42'


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abcdef') is False


def test_re_test_empty_string():
    assert re_test(r'.', '') is False
    assert re_test(r'.*', '') is True


def test_re_test_with_flags():
    assert re_test('abc', 'xxABC', re.IGNORECASE) is True


def test_re_test_with_compiled_pattern():
    assert re_test(re.compile(r'\d+'), 'a1') is True


# ---------------------------------------------------------------------------
# re_finder / re_tester
# ---------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert finder('abc123def') == '123'
    assert finder('no digits') is None


def test_re_finder_with_groups():
    finder = re_finder(r'(\d+)-(\d+)')
    assert finder('x 12-34 y') == ('12', '34')


def test_re_finder_with_flags():
    finder = re_finder('abc', re.IGNORECASE)
    assert finder('xxABC') == 'ABC'


def test_re_finder_with_compiled_pattern():
    finder = re_finder(re.compile(r'\d+'))
    assert finder('a99') == '99'


def test_re_tester_returns_callable():
    tester = re_tester(r'\d+')
    assert tester('abc123') is True
    assert tester('abc') is False


def test_re_tester_with_flags():
    tester = re_tester('abc', re.IGNORECASE)
    assert tester('XXABC') is True


def test_re_tester_with_compiled_pattern():
    tester = re_tester(re.compile(r'\d+'))
    assert tester('x1') is True
    assert tester('xy') is False


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join(',', ['a', 'b', 'c']) == 'a,b,c'


def test_str_join_empty_seq():
    assert str_join('-', []) == ''


def test_str_join_stringifies_items():
    assert str_join('-', [1, 2, 3]) == '1-2-3'
    assert str_join('', [1, None, 2.5]) == '1None2.5'


def test_str_join_single_arg_uses_empty_sep():
    assert str_join(['a', 'b', 'c']) == 'abc'
    assert str_join([1, 2, 3]) == '123'


def test_str_join_single_non_list_iterable():
    assert str_join(('x', 'y')) == 'xy'
    assert str_join(range(3)) == '012'


def test_str_join_with_generator():
    assert str_join(',', (str(i) for i in range(3))) == '0,1,2'


def test_str_join_preserves_sep_type():
    # sep.__class__ is used to stringify items; a bytes sep yields bytes
    assert str_join(b',', [b'a', b'b']) == b'a,b'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'baz') == 'foobar'


def test_cut_prefix_equals_string():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_prefix_empty_prefix():
    assert cut_prefix('foobar', '') == 'foobar'


def test_cut_prefix_longer_than_string():
    assert cut_prefix('foo', 'foobar') == 'foo'


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'baz') == 'foobar'


def test_cut_suffix_equals_string():
    assert cut_suffix('foo', 'foo') == ''


def test_cut_suffix_empty_suffix():
    assert cut_suffix('foobar', '') == 'foobar'


def test_cut_suffix_longer_than_string():
    assert cut_suffix('foo', 'foobar') == 'foo'


def test_cut_suffix_empty_string_input():
    assert cut_suffix('', '') == ''
    assert cut_suffix('', 'x') == ''
