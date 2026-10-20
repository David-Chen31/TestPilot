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
# re_iter
# ---------------------------------------------------------------------------

class TestReIter:
    def test_no_groups(self):
        assert list(re_iter(r'\d+', 'a1b22c333')) == ['1', '22', '333']

    def test_one_group(self):
        assert list(re_iter(r'(\d+)', 'a1b22')) == ['1', '22']

    def test_multiple_groups(self):
        assert list(re_iter(r'(\d+)-(\d+)', 'a 1-2 b 3-4')) == [('1', '2'), ('3', '4')]

    def test_named_groups_only(self):
        assert list(re_iter(r'(?P<x>\d+)-(?P<y>\d+)', 'a 1-2')) == [{'x': '1', 'y': '2'}]

    def test_mixed_named_unnamed_returns_match(self):
        result = list(re_iter(r'(\d+)-(?P<y>\d+)', 'a 1-2'))
        assert len(result) == 1
        assert isinstance(result[0], re.Match)
        assert result[0].group(0) == '1-2'

    def test_returns_iterator(self):
        it = re_iter(r'\d+', '123')
        assert iter(it) is it
        assert list(it) == ['123']

    def test_no_match(self):
        assert list(re_iter(r'\d+', 'abc')) == []

    def test_flags(self):
        assert list(re_iter(r'[a-z]+', 'abc DEF', re.IGNORECASE)) == ['abc', 'DEF']

    def test_compiled_pattern(self):
        pat = re.compile(r'\d+')
        assert list(re_iter(pat, 'a1b2')) == ['1', '2']

    def test_compiled_pattern_flags_ignored(self):
        # When pattern is already compiled, flags argument is ignored.
        pat = re.compile(r'[a-z]+')
        assert list(re_iter(pat, 'abc DEF', re.IGNORECASE)) == ['abc']

    def test_overlapping_groups_full_named(self):
        # All groups are named -> groupdict
        assert list(re_iter(r'(?P<a>x)(?P<b>y)', 'xy')) == [{'a': 'x', 'b': 'y'}]


# ---------------------------------------------------------------------------
# re_all
# ---------------------------------------------------------------------------

class TestReAll:
    def test_returns_list(self):
        result = re_all(r'\d+', 'a1b22')
        assert result == ['1', '22']
        assert isinstance(result, list)

    def test_no_match(self):
        assert re_all(r'\d+', 'abc') == []

    def test_flags(self):
        assert re_all(r'[a-z]+', 'aBc', re.IGNORECASE) == ['aBc']


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

class TestReFind:
    def test_no_groups(self):
        assert re_find(r'\d+', 'a1b22') == '1'

    def test_one_group(self):
        assert re_find(r'(\d+)-(\d+)', 'a1-2b') == ('1', '2')

    def test_single_group_returns_group(self):
        assert re_find(r'a(\d+)b', 'xxa12byy') == '12'

    def test_named_groups(self):
        assert re_find(r'(?P<x>\d+)-(?P<y>\d+)', 'a1-2b') == {'x': '1', 'y': '2'}

    def test_no_match_returns_none(self):
        assert re_find(r'\d+', 'abc') is None

    def test_flags(self):
        assert re_find(r'[a-z]+', 'ABC def', re.IGNORECASE) == 'ABC'

    def test_compiled(self):
        assert re_find(re.compile(r'\d+'), 'a42b') == '42'


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

class TestReTest:
    def test_true(self):
        assert re_test(r'\d+', 'abc123') is True

    def test_false(self):
        assert re_test(r'\d+', 'abc') is False

    def test_returns_bool(self):
        result = re_test(r'\d+', 'abc123')
        assert result is True
        assert isinstance(result, bool)

    def test_flags(self):
        assert re_test(r'[a-z]+', 'ABC', re.IGNORECASE) is True
        assert re_test(r'^[a-z]+$', 'ABC', re.IGNORECASE) is True
        assert re_test(r'^[a-z]+$', 'ABC') is False

    def test_compiled(self):
        assert re_test(re.compile(r'\d'), 'a1') is True


# ---------------------------------------------------------------------------
# re_finder
# ---------------------------------------------------------------------------

class TestReFinder:
    def test_returns_callable(self):
        finder = re_finder(r'\d+')
        assert callable(finder)
        assert finder('a1b22') == '1'
        assert finder('no digits') is None

    def test_returns_simplest_form(self):
        finder = re_finder(r'(\d+)-(\d+)')
        assert finder('a1-2b') == ('1', '2')

    def test_named_groups(self):
        finder = re_finder(r'(?P<x>\d+)-(?P<y>\d+)')
        assert finder('a1-2b') == {'x': '1', 'y': '2'}

    def test_flags(self):
        finder = re_finder(r'[a-z]+', re.IGNORECASE)
        assert finder('ABC') == 'ABC'

    def test_compiled(self):
        finder = re_finder(re.compile(r'\d+'))
        assert finder('a7') == '7'

    def test_no_match_none(self):
        finder = re_finder(r'\d+')
        assert finder('abc') is None


# ---------------------------------------------------------------------------
# re_tester
# ---------------------------------------------------------------------------

class TestReTester:
    def test_returns_callable(self):
        tester = re_tester(r'\d+')
        assert callable(tester)
        assert tester('a1') is True
        assert tester('abc') is False

    def test_returns_bool(self):
        tester = re_tester(r'\d')
        assert tester('1') is True
        assert tester('x') is False

    def test_flags(self):
        tester = re_tester(r'[a-z]+', re.IGNORECASE)
        assert tester('ABC') is True

    def test_compiled(self):
        tester = re_tester(re.compile(r'\d'))
        assert tester('a1') is True
        assert tester('ab') is False

    def test_anchored(self):
        tester = re_tester(r'^abc$')
        assert tester('abc') is True
        assert tester('xabcx') is False


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

class TestStrJoin:
    def test_basic(self):
        assert str_join(',', ['a', 'b', 'c']) == 'a,b,c'

    def test_empty_seq(self):
        assert str_join(',', []) == ''

    def test_single_arg_joins_into_empty_sep(self):
        assert str_join(['a', 'b', 'c']) == 'abc'

    def test_stringifies_items(self):
        assert str_join('-', [1, 2, 3]) == '1-2-3'

    def test_stringifies_mixed(self):
        assert str_join(', ', [1, 'a', None]) == '1, a, None'

    def test_sep_type_used_for_items(self):
        # Items are str()-ed via sep.__class__, i.e. str()
        assert str_join('|', [True, 1.5]) == 'True|1.5'

    def test_bytes_sep(self):
        assert str_join(b',', [b'a', b'b']) == b'a,b'

    def test_sep_in_sequence(self):
        # Passing single arg that is a sequence -> joined with ''
        assert str_join('abc') == 'abc'

    def test_generator(self):
        assert str_join('-', (x for x in ['a', 'b'])) == 'a-b'

    def test_tuple_sequence(self):
        assert str_join('+', ('x', 'y')) == 'x+y'


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

    def test_full_match(self):
        assert cut_prefix('foo', 'foo') == ''

    def test_prefix_longer(self):
        assert cut_prefix('foo', 'foobar') == 'foo'

    def test_empty_string(self):
        assert cut_prefix('', 'x') == ''
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
        assert cut_suffix('foobar', '') == 'foobar'

    def test_full_match(self):
        assert cut_suffix('foo', 'foo') == ''

    def test_suffix_longer(self):
        assert cut_suffix('foo', 'foobar') == 'foo'

    def test_empty_string(self):
        assert cut_suffix('', 'x') == ''
        assert cut_suffix('', '') == ''

    def test_object_with_empty_suffix(self):
        # Empty suffix should short-circuit and return original (truthy check).
        class S(str):
            pass
        s = S('abc')
        assert cut_suffix(s, '') is s


# ---------------------------------------------------------------------------
# Integration with funcmakers-style use
# ---------------------------------------------------------------------------

class TestFuncmakerStyle:
    def test_tester_and_finder_usage(self):
        finder = re_finder(r'\d+')
        tester = re_tester(r'\d+')
        assert finder('abc123') == '123'
        assert tester('abc123') is True
        assert tester('abc') is False

    def test_cut_prefix_via_objects_style(self):
        name = 'MyClass__method'
        assert cut_prefix(name, 'MyClass__') == 'method'
