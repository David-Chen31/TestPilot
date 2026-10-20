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

class TestReIter:
    def test_no_groups_returns_full_matches(self):
        assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']

    def test_single_group_returns_group(self):
        assert list(re_iter(r'(\d+)', 'a1b22c333')) == ['1', '22', '333']

    def test_single_group_returns_group_1_not_group_0(self):
        # Distinguish group(1) from group(0): pick a pattern where they differ.
        # With a single capturing group, group(1) is the inner match,
        # while group(0) would be the whole match.
        assert list(re_iter(r'(\d+)x', 'a12xb34x')) == ['12', '34']
        # sanity: whole match includes the trailing 'x'
        assert re_all(r'(\d+)x', 'a12xb34x') == ['12', '34']

    def test_single_group_full_match_differs(self):
        # group(0) would be 'abc12', group(1) is '12'
        assert re_find(r'abc(\d+)', 'xxabc12yy') == '12'
        # if replaced with group(0) the result would be 'abc12'
        assert re_find(r'abc(\d+)', 'xxabc12yy') != 'abc12'

    def test_multiple_groups_returns_tuples(self):
        assert list(re_iter(r'(\d)(\d)', 'a12b34')) == [('1', '2'), ('3', '4')]

    def test_named_groups_only_returns_dict(self):
        result = list(re_iter(r'(?P<x>\d+)-(?P<y>\d+)', 'a1-2 b3-4'))
        assert result == [{'x': '1', 'y': '2'}, {'x': '3', 'y': '4'}]

    def test_mixed_named_and_unnamed_returns_match_object(self):
        pat = r'(?P<x>\d+)-(\d+)'
        result = list(re_iter(pat, 'a1-2'))
        assert len(result) == 1
        m = result[0]
        assert isinstance(m, re.Match)
        assert m.group('x') == '1'
        assert m.group(2) == '2'

    def test_partial_named_returns_match_object(self):
        # Regression branch: some but not all groups are named.
        pat = r'(\d+)-(?P<y>\d+)'
        result = list(re_iter(pat, 'a1-2'))
        assert len(result) == 1
        assert isinstance(result[0], re.Match)

    def test_compiled_regex_accepted(self):
        compiled = re.compile(r'\d+')
        assert list(re_iter(compiled, 'a1b22')) == ['1', '22']

    def test_flags(self):
        assert list(re_iter(r'[a-z]+', 'AbC', re.IGNORECASE)) == ['AbC']

    def test_no_matches(self):
        assert list(re_iter(r'\d+', 'abc')) == []

    def test_returns_map_object(self):
        import collections.abc
        result = re_iter(r'\d+', 'a1')
        assert isinstance(result, collections.abc.Iterator)


class TestReAll:
    def test_basic(self):
        assert re_all(r'\d+', 'a1b22c333') == ['1', '22', '333']

    def test_single_group(self):
        assert re_all(r'(\d+)', 'a1b22') == ['1', '22']

    def test_single_group_differs_from_full_match(self):
        # group(1) must be used, not group(0)
        assert re_all(r'<(\w+)>', '<a><b>') == ['a', 'b']

    def test_multiple_groups(self):
        assert re_all(r'(\d)(\d)', 'a12b34') == [('1', '2'), ('3', '4')]

    def test_named_only(self):
        assert re_all(r'(?P<x>\d+)', 'a1b2') == [{'x': '1'}, {'x': '2'}]

    def test_no_matches(self):
        assert re_all(r'zzz', 'abc') == []

    def test_flags(self):
        assert re_all(r'[a-z]+', 'AbC', re.IGNORECASE) == ['AbC']

    def test_compiled(self):
        assert re_all(re.compile(r'\d+'), 'a1b2') == ['1', '2']


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

class TestReFind:
    def test_no_groups(self):
        assert re_find(r'\d+', 'abc123def') == '123'

    def test_single_group(self):
        assert re_find(r'(\d+)', 'abc123def') == '123'

    def test_single_group_not_full_match(self):
        # group(1) used, not group(0)
        assert re_find(r'abc(\d+)', 'xxabc12yy') == '12'
        assert re_find(r'x(a|b)y', 'ixayj') == 'a'

    def test_multiple_groups(self):
        assert re_find(r'(\d)(\d)', 'abc12def') == ('1', '2')

    def test_named_only(self):
        assert re_find(r'(?P<x>\d+)', 'abc123') == {'x': '123'}

    def test_partial_named_returns_match(self):
        result = re_find(r'(\d)-(?P<y>\d)', 'a1-2')
        assert isinstance(result, re.Match)
        assert result.group(0) == '1-2'

    def test_no_match_returns_none(self):
        assert re_find(r'\d+', 'abc') is None

    def test_flags(self):
        assert re_find(r'[a-z]+', 'AbC', re.IGNORECASE) == 'AbC'

    def test_compiled(self):
        assert re_find(re.compile(r'\d+'), 'abc123') == '123'

    def test_first_match_only(self):
        assert re_find(r'\d+', 'a1b2') == '1'


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

class TestReTest:
    def test_true(self):
        assert re_test(r'\d+', 'abc123') is True

    def test_false(self):
        assert re_test(r'\d+', 'abc') is False

    def test_flags(self):
        assert re_test(r'[a-z]+', 'ABC', re.IGNORECASE) is True

    def test_compiled(self):
        assert re_test(re.compile(r'\d+'), 'abc123') is True

    def test_anchored(self):
        assert re_test(r'^\d+$', '123') is True
        assert re_test(r'^\d+$', '123a') is False


# ---------------------------------------------------------------------------
# re_finder
# ---------------------------------------------------------------------------

class TestReFinder:
    def test_basic(self):
        finder = re_finder(r'\d+')
        assert finder('abc123') == '123'
        assert finder('abc') is None

    def test_groups(self):
        finder = re_finder(r'(\d)(\d)')
        assert finder('a12') == ('1', '2')

    def test_single_group_returns_group_1(self):
        # group(1) not group(0)
        finder = re_finder(r'abc(\d+)')
        assert finder('xxabc12yy') == '12'

    def test_named(self):
        finder = re_finder(r'(?P<x>\d+)')
        assert finder('a12') == {'x': '12'}

    def test_partial_named_returns_match(self):
        finder = re_finder(r'(\d)-(?P<y>\d)')
        m = finder('a1-2')
        assert isinstance(m, re.Match)

    def test_flags(self):
        finder = re_finder(r'[a-z]+', re.IGNORECASE)
        assert finder('AbC') == 'AbC'

    def test_compiled(self):
        finder = re_finder(re.compile(r'\d+'))
        assert finder('abc1') == '1'

    def test_callable(self):
        finder = re_finder(r'\d+')
        assert callable(finder)


# ---------------------------------------------------------------------------
# re_tester
# ---------------------------------------------------------------------------

class TestReTester:
    def test_basic(self):
        tester = re_tester(r'\d+')
        assert tester('abc123') is True
        assert tester('abc') is False

    def test_flags(self):
        tester = re_tester(r'[a-z]+', re.IGNORECASE)
        assert tester('ABC') is True

    def test_compiled(self):
        tester = re_tester(re.compile(r'\d+'))
        assert tester('a1') is True

    def test_returns_bool(self):
        tester = re_tester(r'\d+')
        assert tester('a1') is True
        assert tester('a') is False


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

class TestStrJoin:
    def test_basic(self):
        assert str_join(', ', ['a', 'b', 'c']) == 'a, b, c'

    def test_forces_stringification(self):
        assert str_join('-', [1, 2, 3]) == '1-2-3'

    def test_empty_seq(self):
        assert str_join(', ', []) == ''

    def test_single_element(self):
        assert str_join(', ', ['x']) == 'x'

    def test_single_argument_joins_with_empty(self):
        # When seq is EMPTY, the first argument becomes the sequence.
        assert str_join(['a', 'b', 'c']) == 'abc'

    def test_single_argument_numeric(self):
        assert str_join([1, 2, 3]) == '123'

    def test_mixed_types(self):
        assert str_join(',', [1, 'a', 2.5, None]) == '1,a,2.5,None'

    def test_sep_class_used_for_stringification(self):
        # sep.__class__ is used as the converter; str for str sep.
        class MyStr(str):
            pass

        sep = MyStr('-')
        result = str_join(sep, [1, 2, 3])
        assert result == '1-2-3'

    def test_generator(self):
        assert str_join('-', (x for x in range(3))) == '0-1-2'

    def test_tuple(self):
        assert str_join('', ('a', 'b')) == 'ab'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

class TestCutPrefix:
    def test_present(self):
        assert cut_prefix('foobar', 'foo') == 'bar'

    def test_absent(self):
        assert cut_prefix('foobar', 'bar') == 'foobar'

    def test_empty_prefix(self):
        assert cut_prefix('foobar', '') == 'foobar'

    def test_whole_string(self):
        assert cut_prefix('foo', 'foo') == ''

    def test_longer_prefix(self):
        assert cut_prefix('ab', 'abc') == 'ab'

    def test_empty_string(self):
        assert cut_prefix('', 'abc') == ''
        assert cut_prefix('', '') == ''


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

class TestCutSuffix:
    def test_present(self):
        assert cut_suffix('foobar', 'bar') == 'foo'

    def test_absent(self):
        assert cut_suffix('foobar', 'foo') == 'foobar'

    def test_empty_suffix(self):
        # Empty suffix must not strip the whole string.
        assert cut_suffix('foobar', '') == 'foobar'

    def test_whole_string(self):
        assert cut_suffix('bar', 'bar') == ''

    def test_longer_suffix(self):
        assert cut_suffix('ab', 'cab') == 'ab'

    def test_empty_string(self):
        assert cut_suffix('', 'abc') == ''
        assert cut_suffix('', '') == ''
