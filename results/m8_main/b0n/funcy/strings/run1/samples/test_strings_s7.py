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
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_one_group():
    assert list(re_iter(r'(\d+)', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_multiple_groups():
    result = list(re_iter(r'(\d+)-(\w+)', '1-a 22-bb'))
    assert result == [('1', 'a'), ('22', 'bb')]


def test_re_iter_named_groups_only():
    result = list(re_iter(r'(?P<num>\d+)-(?P<word>\w+)', '1-a 22-bb'))
    assert result == [{'num': '1', 'word': 'a'}, {'num': '22', 'word': 'bb'}]


def test_re_iter_mixed_named_and_unnamed_groups_returns_match():
    pattern = r'(\d+)-(?P<word>\w+)'
    result = list(re_iter(pattern, '1-a'))
    assert len(result) == 1
    match = result[0]
    assert isinstance(match, re.Match)
    assert match.group(0) == '1-a'
    assert match.group(1) == '1'
    assert match.group('word') == 'a'


def test_re_iter_named_group_count_mismatch_returns_match():
    # two groups, two named -> groupdict; but here 2 groups and 1 named -> mismatch
    pattern = r'(?P<first>\d+)-(\w+)'
    result = list(re_iter(pattern, '1-a'))
    assert len(result) == 1
    assert isinstance(result[0], re.Match)


def test_re_iter_no_matches():
    assert list(re_iter(r'\d+', 'abc')) == []


def test_re_iter_with_flags():
    assert list(re_iter(r'[a-z]+', 'ABC def', flags=re.IGNORECASE)) == ['ABC', 'def']


def test_re_iter_accepts_compiled_pattern():
    pattern = re.compile(r'\d+')
    assert list(re_iter(pattern, 'a1b22')) == ['1', '22']


def test_re_iter_returns_iterator():
    result = re_iter(r'\d+', 'a1b2')
    assert iter(result) is result


def test_re_all_basic():
    assert re_all(r'\d+', 'a1b22c333') == ['1', '22', '333']


def test_re_all_empty():
    assert re_all(r'\d+', 'abc') == []


def test_re_all_groups():
    assert re_all(r'(\w)(\d)', 'a1 b2') == [('a', '1'), ('b', '2')]


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

def test_re_find_no_groups():
    assert re_find(r'\d+', 'abc123def') == '123'


def test_re_find_one_group():
    assert re_find(r'(\d+)', 'abc123def') == '123'


def test_re_find_multiple_groups():
    assert re_find(r'(\d+)-(\w+)', 'abc 12-x def') == ('12', 'x')


def test_re_find_named_groups():
    assert re_find(r'(?P<a>\d+)-(?P<b>\w+)', '12-x') == {'a': '12', 'b': 'x'}


def test_re_find_no_match_returns_none():
    assert re_find(r'\d+', 'abc') is None


def test_re_find_with_flags():
    assert re_find(r'[a-z]+', '123ABC', flags=re.IGNORECASE) == 'ABC'


def test_re_find_compiled():
    pattern = re.compile(r'\d+')
    assert re_find(pattern, 'abc123') == '123'


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abc') is False


def test_re_test_with_flags():
    assert re_test(r'abc', 'ABC', flags=re.IGNORECASE) is True


def test_re_test_compiled():
    pattern = re.compile(r'\d+')
    assert re_test(pattern, 'abc123') is True


# ---------------------------------------------------------------------------
# re_finder / re_tester
# ---------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert callable(finder)
    assert finder('a1b22') == '1'
    assert finder('abc') is None


def test_re_finder_group():
    finder = re_finder(r'(\d+)-(\w+)')
    assert finder('x 12-a y') == ('12', 'a')


def test_re_finder_named_groups():
    finder = re_finder(r'(?P<n>\d+)')
    assert finder('x 5 y') == {'n': '5'}


def test_re_finder_with_flags():
    finder = re_finder(r'[a-z]+', flags=re.IGNORECASE)
    assert finder('123ABC') == 'ABC'


def test_re_finder_compiled():
    finder = re_finder(re.compile(r'\d+'))
    assert finder('a1') == '1'


def test_re_tester_returns_callable():
    tester = re_tester(r'\d+')
    assert callable(tester)
    assert tester('a1') is True
    assert tester('abc') is False


def test_re_tester_with_flags():
    tester = re_tester(r'abc', flags=re.IGNORECASE)
    assert tester('ABC') is True


def test_re_tester_compiled():
    tester = re_tester(re.compile(r'\d+'))
    assert tester('a1') is True


def test_invalid_regex_raises():
    with pytest.raises(re.error):
        re_finder(r'(')
    with pytest.raises(re.error):
        re_tester(r'(')


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join(', ', ['a', 'b', 'c']) == 'a, b, c'


def test_str_join_empty_seq():
    assert str_join(', ', []) == ''


def test_str_join_forces_stringification():
    assert str_join('-', [1, 2, 3]) == '1-2-3'


def test_str_join_mixed_types():
    assert str_join(',', [1, 'a', None]) == '1,a,None'


def test_str_join_curried():
    join_with_comma = str_join(', ')
    assert join_with_comma(['a', 'b']) == 'a, b'


def test_str_join_curried_scalar_items():
    join_with_dash = str_join('-')
    assert join_with_dash([1, 2]) == '1-2'


def test_str_join_sep_class_used_for_items():
    # Items are stringified using sep.__class__
    assert str_join('-', [1, 2]) == '1-2'
    assert str_join(b'-', [b'a', b'b']) == b'a-b'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'baz') == 'foobar'


def test_cut_prefix_empty_prefix():
    assert cut_prefix('foobar', '') == 'foobar'


def test_cut_prefix_whole_string():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_prefix_longer_than_string():
    assert cut_prefix('foo', 'foobar') == 'foo'


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'baz') == 'foobar'


def test_cut_suffix_empty_suffix():
    assert cut_suffix('foobar', '') == 'foobar'


def test_cut_suffix_whole_string():
    assert cut_suffix('foo', 'foo') == ''


def test_cut_suffix_longer_than_string():
    assert cut_suffix('foo', 'foobar') == 'foo'
