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
# re_iter / re_all — simplest possible presentation of matches
# ---------------------------------------------------------------------------

class TestReIterAndReAll:
    def test_no_groups_returns_matches(self):
        assert re_all(r'\d+', 'a1 b22 c333') == ['1', '22', '333']

    def test_no_groups_iter_is_iterator(self):
        result = re_iter(r'\d+', 'a1 b22')
        # It's a map object, iterable and lazy
        assert list(result) == ['1', '22']

    def test_single_group_returns_group(self):
        assert re_all(r'(\d+)', 'a1 b22 c333') == ['1', '22', '333']

    def test_multiple_unnamed_groups_returns_tuple(self):
        assert re_all(r'(\d)(\w)', '1a 2b') == [('1', 'a'), ('2', 'b')]

    def test_all_named_groups_returns_dict(self):
        result = re_all(r'(?P<num>\d)(?P<letter>\w)', '1a 2b')
        assert result == [
            {'num': '1', 'letter': 'a'},
            {'num': '2', 'letter': 'b'},
        ]

    def test_named_and_unnamed_returns_match_object(self):
        result = re_all(r'(?P<num>\d)(\w)', '1a 2b')
        assert len(result) == 2
        for m in result:
            assert isinstance(m, re.Match)

    def test_with_compiled_pattern(self):
        pattern = re.compile(r'\d+')
        assert re_all(pattern, 'a1 b22') == ['1', '22']

    def test_with_flags(self):
        assert re_all(r'[a-z]+', 'ABC def GHI', flags=re.IGNORECASE) == ['ABC', 'def', 'GHI']

    def test_flags_ignored_for_compiled_pattern(self):
        pattern = re.compile(r'[a-z]+', re.IGNORECASE)
        assert re_all(pattern, 'ABC def', flags=re.MULTILINE) == ['ABC', 'def']

    def test_no_matches_returns_empty(self):
        assert re_all(r'\d+', 'abc') == []
        assert re_iter(r'\d+', 'abc') is not None
        assert list(re_iter(r'\d+', 'abc')) == []

    def test_empty_string(self):
        assert re_all(r'\d+', '') == []


# ---------------------------------------------------------------------------
# re_find
# ---------------------------------------------------------------------------

class TestReFind:
    def test_no_groups(self):
        assert re_find(r'\d+', 'abc 123 def') == '123'

    def test_single_group(self):
        assert re_find(r'(\d+)', 'abc 123 def') == '123'

    def test_multiple_groups(self):
        assert re_find(r'(\d)(\w)', '1a 2b') == ('1', 'a')

    def test_named_groups(self):
        assert re_find(r'(?P<num>\d)(?P<letter>\w)', 'x 1a y') == {
            'num': '1',
            'letter': 'a',
        }

    def test_mixed_named_unnamed_returns_match(self):
        m = re_find(r'(?P<num>\d)(\w)', 'x 1a')
        assert isinstance(m, re.Match)
        assert m.group() == '1a'

    def test_no_match_returns_none(self):
        assert re_find(r'\d+', 'abc') is None

    def test_with_flags(self):
        assert re_find(r'[a-z]+', 'ABC def', flags=re.IGNORECASE) == 'ABC'

    def test_with_compiled_pattern(self):
        assert re_find(re.compile(r'\d+'), 'abc 123') == '123'


# ---------------------------------------------------------------------------
# re_test
# ---------------------------------------------------------------------------

class TestReTest:
    def test_match_returns_true(self):
        assert re_test(r'\d+', 'abc 123') is True

    def test_no_match_returns_false(self):
        assert re_test(r'\d+', 'abc') is False

    def test_with_flags(self):
        assert re_test(r'[a-z]+', 'ABC', flags=re.IGNORECASE) is True
        assert re_test(r'[a-z]+', 'ABC') is False

    def test_with_compiled(self):
        assert re_test(re.compile(r'\d'), 'abc 1') is True


# ---------------------------------------------------------------------------
# re_finder / re_tester
# ---------------------------------------------------------------------------

class TestReFinder:
    def test_returns_callable(self):
        finder = re_finder(r'\d+')
        assert callable(finder)

    def test_finds_match(self):
        finder = re_finder(r'\d+')
        assert finder('a 123 b 45') == '123'

    def test_no_match_returns_none(self):
        finder = re_finder(r'\d+')
        assert finder('abc') is None

    def test_named_groups(self):
        finder = re_finder(r'(?P<n>\d)(?P<l>\w)')
        assert finder('1a') == {'n': '1', 'l': 'a'}

    def test_with_flags(self):
        finder = re_finder(r'[a-z]+', flags=re.IGNORECASE)
        assert finder('ABC') == 'ABC'

    def test_with_compiled(self):
        finder = re_finder(re.compile(r'\d+'))
        assert finder('abc 42') == '42'


class TestReTester:
    def test_returns_callable(self):
        tester = re_tester(r'\d+')
        assert callable(tester)

    def test_match_true(self):
        tester = re_tester(r'\d+')
        assert tester('abc 123') is True

    def test_no_match_false(self):
        tester = re_tester(r'\d+')
        assert tester('abc') is False

    def test_with_flags(self):
        tester = re_tester(r'[a-z]+', flags=re.IGNORECASE)
        assert tester('ABC') is True

    def test_with_compiled(self):
        tester = re_tester(re.compile(r'\d+'))
        assert tester('abc 1') is True


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

class TestStrJoin:
    def test_basic(self):
        assert str_join('-', ['a', 'b', 'c']) == 'a-b-c'

    def test_empty_sequence(self):
        assert str_join('-', []) == ''

    def test_stringifies_items(self):
        assert str_join(',', [1, 2, 3]) == '1,2,3'

    def test_none_items(self):
        assert str_join(',', [None, 1]) == 'None,1'

    def test_single_arg_joins_characters(self):
        assert str_join('abc') == 'a-b-c'

    def test_single_arg_empty_string(self):
        assert str_join('') == ''

    def test_sep_used_as_type(self):
        # sep.__class__ is used for stringification, so list items become lists
        # joined by the separator.
        assert str_join('-', [1, 2]) == '1-2'

    def test_with_generator(self):
        assert str_join(',', (str(i) for i in range(3))) == '0,1,2'

    def test_with_tuple(self):
        assert str_join('', ('a', 'b')) == 'ab'

    def test_custom_sep(self):
        assert str_join('::', ['x', 'y']) == 'x::y'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

class TestCutPrefix:
    def test_prefix_present(self):
        assert cut_prefix('foobar', 'foo') == 'bar'

    def test_prefix_absent(self):
        assert cut_prefix('foobar', 'baz') == 'foobar'

    def test_empty_prefix(self):
        assert cut_prefix('foobar', '') == 'foobar'

    def test_full_string_prefix(self):
        assert cut_prefix('foo', 'foo') == ''

    def test_empty_string_with_prefix(self):
        assert cut_prefix('', 'foo') == ''

    def test_empty_string_with_empty_prefix(self):
        assert cut_prefix('', '') == ''


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

class TestCutSuffix:
    def test_suffix_present(self):
        assert cut_suffix('foobar', 'bar') == 'foo'

    def test_suffix_absent(self):
        assert cut_suffix('foobar', 'baz') == 'foobar'

    def test_empty_suffix(self):
        # Guard against negative-slice bug: empty suffix returns string unchanged
        assert cut_suffix('foobar', '') == 'foobar'

    def test_full_string_suffix(self):
        assert cut_suffix('foo', 'foo') == ''

    def test_empty_string_with_suffix(self):
        assert cut_suffix('', 'foo') == ''

    def test_empty_string_with_empty_suffix(self):
        assert cut_suffix('', '') == ''
