import re
import pytest

from funcy.strings import (
    re_iter, re_all, re_find, re_test, re_finder, re_tester,
    str_join, cut_prefix, cut_suffix,
    _make_getter, _prepare, _re_type,
)


# ---------- re_iter / re_all ----------

def test_re_iter_no_groups():
    assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']


def test_re_all_no_groups():
    assert re_all(r'\d+', 'a1b22c333') == ['1', '22', '333']


def test_re_all_no_matches():
    assert re_all(r'\d+', 'abc') == []


def test_re_iter_single_group():
    assert list(re_iter(r'(\d+)', 'a1b22')) == ['1', '22']


def test_re_iter_multiple_groups():
    assert list(re_iter(r'(\d)(\d)', 'a12b34')) == [('1', '2'), ('3', '4')]


def test_re_iter_named_groups_all():
    result = list(re_iter(r'(?P<a>\d)(?P<b>\d)', 'x12y34'))
    assert result == [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]


def test_re_iter_mixed_groups_returns_match():
    matches = list(re_iter(r'(?P<a>\d)(\d)', 'x12y34'))
    assert len(matches) == 2
    for m in matches:
        assert isinstance(m, _re_type.match(type(m)))
        assert m.lastindex is not None


def test_re_iter_compiled_pattern():
    pattern = re.compile(r'\d+')
    assert list(re_iter(pattern, 'a1b22')) == ['1', '22']


def test_re_iter_flags():
    assert re_all(r'[a-z]+', 'ABC def GHI jkl', re.IGNORECASE) == ['ABC', 'def', 'GHI', 'jkl']


def test_re_iter_returns_iterator():
    it = re_iter(r'\d', '123')
    assert not isinstance(it, list)
    assert list(it) == ['1', '2', '3']


# ---------- re_find ----------

def test_re_find_no_groups():
    assert re_find(r'\d+', 'abc 123 def') == '123'


def test_re_find_no_match():
    assert re_find(r'\d+', 'abcdef') is None


def test_re_find_single_group():
    assert re_find(r'(\w+)@(\w+)', 'mail: foo@bar') is None
    assert re_find(r'(\w+)@(\w+)', 'foo@bar') == ('foo', 'bar')


def test_re_find_one_group():
    assert re_find(r'(\d+)', 'abc 42 def') == '42'


def test_re_find_named_groups():
    assert re_find(r'(?P<user>\w+)@(?P<host>\w+)', 'foo@bar') == {'user': 'foo', 'host': 'bar'}


def test_re_find_compiled_and_flags():
    assert re_find(re.compile(r'\d+'), 'x9') == '9'
    assert re_find(r'[a-z]+', 'ABC', re.IGNORECASE) == 'ABC'


def test_re_find_mixed_groups_returns_match_object():
    m = re_find(r'(?P<a>\d)(\d)', 'x12')
    assert m is not None
    assert m.group(0) == '12'


# ---------- re_test ----------

def test_re_test_true():
    assert re_test(r'\d+', 'abc123') is True


def test_re_test_false():
    assert re_test(r'\d+', 'abcdef') is False


def test_re_test_compiled_and_flags():
    assert re_test(re.compile(r'\d+'), 'a1') is True
    assert re_test(r'[a-z]+', 'ABC', re.IGNORECASE) is True


# ---------- re_finder / re_tester ----------

def test_re_finder_returns_callable():
    finder = re_finder(r'\d+')
    assert finder('abc 123') == '123'
    assert finder('abc') is None


def test_re_finder_compiled_and_flags():
    finder = re_finder(re.compile(r'\d+'))
    assert finder('x7') == '7'
    finder2 = re_finder(r'[a-z]+', re.IGNORECASE)
    assert finder2('ABC') == 'ABC'


def test_re_finder_named_groups():
    finder = re_finder(r'(?P<k>\w+)=(?P<v>\w+)')
    assert finder('a=1') == {'k': 'a', 'v': '1'}
    assert finder('nothing') is None


def test_re_tester_returns_predicate():
    tester = re_tester(r'\d+')
    assert tester('abc123') is True
    assert tester('abcdef') is False


def test_re_tester_compiled_and_flags():
    tester = re_tester(re.compile(r'foo'))
    assert tester('foobar') is True
    tester2 = re_tester(r'foo', re.IGNORECASE)
    assert tester2('FOOBAR') is True


# ---------- _make_getter / _prepare (internal helpers, via public behavior) ----------

def test_prepare_returns_compiled_pattern():
    pattern, getter = _prepare(r'\d+', 0)
    assert pattern.search('a1') is not None
    m = pattern.search('a1')
    assert getter(m) == '1'


def test_prepare_with_compiled_ignores_flags():
    compiled = re.compile(r'[a-z]+')
    pattern, getter = _prepare(compiled, re.IGNORECASE)
    assert pattern is compiled


def test_make_getter_zero_groups():
    regex = re.compile(r'\d+')
    getter = _make_getter(regex)
    assert getter(regex.search('a1b22')) == '22'


# ---------- str_join ----------

def test_str_join_basic():
    assert str_join(',', ['a', 'b', 'c']) == 'a,b,c'


def test_str_join_empty_seq():
    assert str_join(',', []) == ''


def test_str_join_stringifies_items():
    assert str_join('-', [1, 2, 3]) == '1-2-3'


def test_str_join_with_non_str_sep():
    assert str_join(b'-', ['a', 'b']) == 'a-b'
    assert str_join(0, [1, 2, 3]) == '10203'


def test_str_join_single_arg_returns_callable():
    joiner = str_join(',')
    assert callable(joiner)
    assert joiner(['x', 'y']) == 'x,y'


def test_str_join_single_arg_stringifies_with_str():
    joiner = str_join(b',')
    assert joiner([1, 2]) == '1,2'


# ---------- cut_prefix ----------

def test_cut_prefix_present():
    assert cut_prefix('foobar', 'foo') == 'bar'


def test_cut_prefix_absent():
    assert cut_prefix('foobar', 'baz') == 'foobar'


def test_cut_prefix_empty():
    assert cut_prefix('foobar', '') == 'foobar'


def test_cut_prefix_whole():
    assert cut_prefix('foo', 'foo') == ''


def test_cut_prefix_longer_than_s():
    assert cut_prefix('foo', 'foobar') == 'foo'


# ---------- cut_suffix ----------

def test_cut_suffix_present():
    assert cut_suffix('foobar', 'bar') == 'foo'


def test_cut_suffix_absent():
    assert cut_suffix('foobar', 'baz') == 'foobar'


def test_cut_suffix_empty():
    assert cut_suffix('foobar', '') == 'foobar'


def test_cut_suffix_whole():
    assert cut_suffix('foo', 'foo') == ''


def test_cut_suffix_longer_than_s():
    assert cut_suffix('foo', 'foobar') == 'foo'
