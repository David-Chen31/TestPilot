import re
import pytest

from funcy.strings import (
    re_iter, re_all, re_find, re_test, re_finder, re_tester,
    str_join, cut_prefix, cut_suffix,
)


# ---------------------------------------------------------------------------
# re_iter / re_all
# ---------------------------------------------------------------------------

class TestReIter:
    def test_no_groups(self):
        assert list(re_iter(r'\d+', 'a12b345c')) == ['12', '345']

    def test_one_group(self):
        assert list(re_iter(r'(\d+)', 'a12b345c')) == ['12', '345']

    def test_multiple_groups(self):
        assert list(re_iter(r'(\d)(\d)', 'a12b34c')) == [('1', '2'), ('3', '4')]

    def test_named_groups_only(self):
        result = list(re_iter(r'(?P<num>\d+)', 'a12b345c'))
        assert result == [{'num': '12'}, {'num': '345'}]

    def test_mixed_named_and_unnamed(self):
        # groups != number of named groups -> raw match objects
        result = list(re_iter(r'(\d)(?P<x>\d)', 'a12b34c'))
        assert len(result) == 2
        for m in result:
            assert isinstance(m, re.Match)

    def test_compiled_pattern(self):
        pat = re.compile(r'\d+')
        assert list(re_iter(pat, 'a12b345c')) == ['12', '345']

    def test_flags(self):
        assert list(re_iter(r'[a-z]+', 'ABCdef', flags=re.IGNORECASE)) == \
            ['ABCdef']

    def test_flags_ignored_for_compiled(self):
        pat = re.compile(r'\d+')
        assert list(re_iter(pat, 'a12b', flags=re.IGNORECASE)) == ['12']

    def test_no_matches(self):
        assert list(re_iter(r'\d+', 'abc')) == []

    def test_returns_iterator(self):
        result = re_iter(r'\d+', 'a12b')
        assert not isinstance(result, list)
        assert list(result) == ['12']


class TestReAll:
    def test_basic(self):
        assert re_all(r'\d+', 'a12b345c') == ['12', '345']

    def test_no_matches(self):
        assert re_all(r'\d+', 'abc') == []

    def test_returns_list(self):
        assert isinstance(re_all(r'\d+', 'a12b'), list)

    def test_named_groups(self):
        assert re_all(r'(?P<num>\d+)', 'a12b345c') == \
            [{'num': '12'}, {'num': '345'}]


# ---------------------------------------------------------------------------
# re_find / re_finder
# ---------------------------------------------------------------------------

class TestReFind:
    def test_no_groups(self):
        assert re_find(r'\d+', 'a12b345c') == '12'

    def test_one_group(self):
        assert re_find(r'(\d+)', 'a12b345c') == '12'

    def test_multiple_groups(self):
        assert re_find(r'(\d)(\d)', 'a12b') == ('1', '2')

    def test_named_groups_only(self):
        assert re_find(r'(?P<num>\d+)', 'a12b') == {'num': '12'}

    def test_mixed_groups_returns_match(self):
        m = re_find(r'(\d)(?P<x>\d)', 'a12b')
        assert isinstance(m, re.Match)
        assert m.group(0) == '12'

    def test_no_match(self):
        assert re_find(r'\d+', 'abc') is None

    def test_first_match_only(self):
        assert re_find(r'\d+', 'a12b345c') == '12'

    def test_flags(self):
        assert re_find(r'[a-z]+', '123XYZ', flags=re.IGNORECASE) == 'XYZ'


class TestReFinder:
    def test_creates_callable(self):
        finder = re_finder(r'\d+')
        assert callable(finder)
        assert finder('a12b') == '12'
        assert finder('abc') is None

    def test_groups(self):
        finder = re_finder(r'(\d+)')
        assert finder('a12b') == '12'

    def test_flags(self):
        finder = re_finder(r'[a-z]+', flags=re.IGNORECASE)
        assert finder('123XYZ') == 'XYZ'

    def test_compiled_regex(self):
        finder = re_finder(re.compile(r'\d+'))
        assert finder('a12b') == '12'


# ---------------------------------------------------------------------------
# re_test / re_tester
# ---------------------------------------------------------------------------

class TestReTest:
    def test_match(self):
        assert re_test(r'\d+', 'a12b') is True

    def test_no_match(self):
        assert re_test(r'\d+', 'abc') is False

    def test_flags(self):
        assert re_test(r'[a-z]+', '123', flags=re.IGNORECASE) is False
        assert re_test(r'[a-z]+', '123XYZ', flags=re.IGNORECASE) is True

    def test_compiled(self):
        assert re_test(re.compile(r'\d+'), 'a12b') is True


class TestReTester:
    def test_creates_predicate(self):
        tester = re_tester(r'\d+')
        assert callable(tester)
        assert tester('a12b') is True
        assert tester('abc') is False

    def test_flags(self):
        tester = re_tester(r'[a-z]+', flags=re.IGNORECASE)
        assert tester('123XYZ') is True
        assert tester('123') is False

    def test_compiled(self):
        tester = re_tester(re.compile(r'\d+'))
        assert tester('a12b') is True

    def test_returns_bool(self):
        tester = re_tester(r'\d+')
        assert tester('a12b') is True
        assert tester('abc') is False


# ---------------------------------------------------------------------------
# str_join
# ---------------------------------------------------------------------------

class TestStrJoin:
    def test_basic(self):
        assert str_join(', ', ['a', 'b', 'c']) == 'a, b, c'

    def test_empty_seq(self):
        assert str_join(', ', []) == ''

    def test_single_arg_uses_empty_sep(self):
        assert str_join(['a', 'b', 'c']) == 'abc'

    def test_single_arg_empty(self):
        assert str_join([]) == ''

    def test_stringifies_items(self):
        assert str_join('-', [1, 2, 3]) == '1-2-3'

    def test_stringifies_all(self):
        assert str_join(',', [1, 'x', 2.5, None, True]) == '1,x,2.5,None,True'

    def test_sep_not_string(self):
        # Uses sep.__class__ to coerce items, so a non-str sep fails to join
        with pytest.raises(TypeError):
            str_join(1, [1, 2])

    def test_generator(self):
        assert str_join(',', (x for x in ['a', 'b'])) == 'a,b'

    def test_single_arg_generator(self):
        assert str_join((x for x in ['a', 'b'])) == 'ab'


# ---------------------------------------------------------------------------
# cut_prefix
# ---------------------------------------------------------------------------

class TestCutPrefix:
    def test_present(self):
        assert cut_prefix('foobar', 'foo') == 'bar'

    def test_not_present(self):
        assert cut_prefix('foobar', 'bar') == 'foobar'

    def test_empty_prefix(self):
        assert cut_prefix('foobar', '') == 'foobar'

    def test_full_string(self):
        assert cut_prefix('foo', 'foo') == ''

    def test_empty_string(self):
        assert cut_prefix('', 'foo') == ''

    def test_empty_string_empty_prefix(self):
        assert cut_prefix('', '') == ''

    def test_case_sensitive(self):
        assert cut_prefix('Foobar', 'foo') == 'Foobar'

    def test_middle_not_cut(self):
        assert cut_prefix('afoo', 'foo') == 'afoo'


# ---------------------------------------------------------------------------
# cut_suffix
# ---------------------------------------------------------------------------

class TestCutSuffix:
    def test_present(self):
        assert cut_suffix('foobar', 'bar') == 'foo'

    def test_not_present(self):
        assert cut_suffix('foobar', 'foo') == 'foobar'

    def test_empty_suffix(self):
        assert cut_suffix('foobar', '') == 'foobar'

    def test_full_string(self):
        assert cut_suffix('bar', 'bar') == ''

    def test_empty_string(self):
        assert cut_suffix('', 'bar') == ''

    def test_empty_string_empty_suffix(self):
        assert cut_suffix('', '') == ''

    def test_case_sensitive(self):
        assert cut_suffix('foobar', 'BAR') == 'foobar'

    def test_middle_not_cut(self):
        assert cut_suffix('bara', 'bar') == 'bara'

    def test_does_not_return_minus_zero(self):
        # ensure empty suffix handled specially (avoids s[:-0] returning '')
        assert cut_suffix('abc', '') == 'abc'
