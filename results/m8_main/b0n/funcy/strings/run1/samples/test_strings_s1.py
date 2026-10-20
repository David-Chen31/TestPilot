import re
import pytest

from funcy.strings import (
    re_iter, re_all, re_find, re_test, re_finder, re_tester,
    str_join, cut_prefix, cut_suffix,
)


# --------------------------------------------------------------------------
# re_iter
# --------------------------------------------------------------------------

def test_re_iter_no_groups():
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_single_group():
    assert list(re_iter(r'(\d+)', 'a1b22c333')) == ['1', '22', '333']


def test_re_iter_multiple_groups():
    assert list(re_iter(r'(\d)(\d)', 'a12b34')) == [('1', '2'), ('3', '4')]


def test_re_iter_named_groups_only():
    result = list(re_iter(r'(?P<x>\d+)-(?P<y>\d+)', '12-34 56-78'))
    assert result == [
        {'x': '12', 'y': '34'},
        {'x': '56', 'y': '78'},
    ]


def test_re_iter_named_and_unnamed_groups_returns_match():
    matches = list(re_iter(r'(?P<x>\d+)(\d)', '12 34'))
    assert len(matches) == 2
    for m in matches:
        assert hasattr(m, 'group')
        assert hasattr(m, 'groupdict')


def test_re_iter_no_matches():
    assert list(re_iter(r'\d+', 'abc')) == []


def test_re_iter_with_flags():
    assert list(re_iter(r'[a-z]+', 'AB cd EF gh', re.IGNORECASE)) == ['AB', 'cd', 'EF', 'gh']


def test_re_iter_compiled_pattern():
    pattern = re.compile(r'\d+')
    assert list(re_iter(pattern, 'a1b22')) == ['1', '22']


def test_re_iter_returns_iterator():
    it = re_iter(r'\d+', '1 2 3')
    assert iter(it) is it
    assert next(it) == '1'


# --------------------------------------------------------------------------
# re_all
# --------------------------------------------------------------------------

def test_re_all_basic():
    assert re_all(r'\d+', 'a1b22c333') == ['1', '22', '333']


def test_re_all_no_matches():
    assert re_all(r'\d+', 'abc') == []


def test_re_all_groups():
    assert re_all(r'(\w)(\w)', 'ab cd') == [('a', 'b'), ('c', 'd')]


def test_re_all_with_flags():
    assert re_all(r'[a-z]+', 'AB cd EF', re.IGNORECASE) == ['AB', 'cd', 'EF']


def test_re_all_compiled_pattern():
    assert re_all(re.compile(r'\d'), 'a1b2') == ['1', '2']


def test_re_all_returns_list():
    result = re_all(r'\d', '12')
    assert isinstance(result, list)


# --------------------------------------------------------------------------
# re_find
# --------------------------------------------------------------------------

def test_re_find_basic():
    assert re_find(r'\d+', 'abc123def456') == '123'


def test_re_find_no_match_returns_none():
    assert re_find(r'\d+', 'abc') is None


def test_re_find_single_group():
    assert re_find(r'(\d+)', 'abc123') == '123'


def test_re_find_multiple_groups():
    assert re_find(r'(\d)(\d)', 'a12') == ('1', '2')


def test_re_find_named_groups_only():
    assert re_find(r'(?P<x>\d+)-(?P<y>\d+)', '12-34') == {'x': '12', 'y': '34'}


def test_re_find_named_and_unnamed_returns_match():
    m = re_find(r'(?P<x>\d+)(\d)', '12')
    assert m.group(0) == '12'


def test_re_find_with_flags():
    assert re_find(r'[a-z]+', 'AB cd', re.IGNORECASE) == 'AB'


def test_re_find_compiled_pattern():
    assert re_find(re.compile(r'\d+'), 'x99') == '99'


# --------------------------------------------------------------------------
# re_test
# --------------------------------------------------------------------------

def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abc') is False


def test_re_test_with_flags():
    assert re_test(r'[a-z]+', 'ABC', re.IGNORECASE) is True


def test_re_test_compiled_pattern():
    assert re_test(re.compile(r'\d'), 'a1') is True


# --------------------------------------------------------------------------
# re_finder
# --------------------------------------------------------------------------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert callable(finder)
    assert finder('abc123') == '123'
    assert finder('abc') is None


def test_re_finder_with_flags():
    finder = re_finder(r'[a-z]+', re.IGNORECASE)
    assert finder('AB cd') == 'AB'


def test_re_finder_compiled_pattern():
    finder = re_finder(re.compile(r'\d+'))
    assert finder('x42') == '42'


def test_re_finder_groups():
    finder = re_finder(r'(\d)(\d)')
    assert finder('a12') == ('1', '2')


# --------------------------------------------------------------------------
# re_tester
# --------------------------------------------------------------------------

def test_re_tester_returns_callable():
    tester = re_tester(r'\d+')
    assert callable(tester)
    assert tester('abc123') is True
    assert tester('abc') is False


def test_re_tester_with_flags():
    tester = re_tester(r'[a-z]+', re.IGNORECASE)
    assert tester('ABC') is True


def test_re_tester_compiled_pattern():
    tester = re_tester(re.compile(r'\d'))
    assert tester('a1') is True


# --------------------------------------------------------------------------
# str_join
# --------------------------------------------------------------------------

def test_str_join_basic():
    assert str_join('-', ['a', 'b', 'c']) == 'a-b-c'


def test_str_join_empty_seq():
    assert str_join('-', []) == ''


def test_str_join_converts_items():
    assert str_join(',', [1, 2, 3]) == '1,2,3'


def test_str_join_unicode_sep():
    assert str_join('→', ['x', 'y']) == 'x→y'


def test_str_join_bytes_sep():
    assert str_join(b',', [b'1', b'2']) == b'1,2'


def test_str_join_single_arg_uses_empty_sep():
    assert str_join(['a', 'b', 'c']) == 'abc'


def test_str_join_single_arg_converts_items():
    assert str_join([1, 2, 3]) == '123'


def test_str_join_single_arg_empty():
    assert str_join([]) == ''


# --------------------------------------------------------------------------
# cut_prefix
# --------------------------------------------------------------------------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'baz') == 'foobar'


def test_cut_prefix_full_string():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_prefix_empty_prefix():
    assert cut_prefix('foo', '') == 'foo'


def test_cut_prefix_empty_string():
    assert cut_prefix('', '') == ''
    assert cut_prefix('', 'x') == ''


# --------------------------------------------------------------------------
# cut_suffix
# --------------------------------------------------------------------------

def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'baz') == 'foobar'


def test_cut_suffix_full_string():
    assert cut_suffix('foo', 'foo') == ''


def test_cut_suffix_empty_suffix():
    assert cut_suffix('foo', '') == 'foo'


def test_cut_suffix_empty_string():
    assert cut_suffix('', '') == ''
    assert cut_suffix('', 'x') == ''


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

def test_re_iter_invalid_regex():
    with pytest.raises(re.error):
        re_iter('(', 'abc')


def test_re_find_invalid_regex():
    with pytest.raises(re.error):
        re_find('(', 'abc')


def test_re_tester_invalid_regex():
    with pytest.raises(re.error):
        re_tester('(')


def test_str_join_non_string_sep_raises():
    # A non-str separator that isn't EMPTY; sep.join won't exist for ints.
    with pytest.raises(AttributeError):
        str_join(1, ['a', 'b'])


def test_cut_prefix_non_string_raises():
    with pytest.raises(AttributeError):
        cut_prefix(123, 'x')


def test_cut_suffix_non_string_raises():
    with pytest.raises(AttributeError):
        cut_suffix(123, 'x')
