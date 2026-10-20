import pytest
from unittest import mock

from sqlparse.utils import (
    split_unquoted_newlines,
    remove_quotes,
    recurse,
    imt,
    consume,
    offset,
    indent,
)


# ---------------------------------------------------------------------------
# split_unquoted_newlines
# ---------------------------------------------------------------------------

class TestSplitUnquotedNewlines:
    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_no_newlines(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_single_lf(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_single_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_newline_inside_double_quotes(self):
        stmt = 'a "x\ny" b'
        assert split_unquoted_newlines(stmt) == ['a "x\ny" b']

    def test_newline_inside_single_quotes(self):
        stmt = "a 'x\ny' b"
        assert split_unquoted_newlines(stmt) == ["a 'x\ny' b"]

    def test_mixed_quoted_and_unquoted(self):
        stmt = 'x\n"a\nb"\ny'
        assert split_unquoted_newlines(stmt) == ['x', '"a\nb"', 'y']

    def test_escaped_quote_in_double_quotes(self):
        stmt = '"a\\"b\nc"'
        assert split_unquoted_newlines(stmt) == ['"a\\"b\nc"']

    def test_escaped_quote_in_single_quotes(self):
        stmt = "'a\\'b\nc'"
        assert split_unquoted_newlines(stmt) == ["'a\\'b\nc'"]

    def test_non_string_uses_str(self):
        # str(12345) == "12345"
        assert split_unquoted_newlines(12345) == ["12345"]

    def test_windows_lineendings_in_unquoted(self):
        assert split_unquoted_newlines("a\r\nb\rc\nd") == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------

class TestRemoveQuotes:
    def test_none(self):
        assert remove_quotes(None) is None

    def test_no_quotes(self):
        assert remove_quotes("abc") == "abc"

    def test_double_quotes(self):
        assert remove_quotes('"abc"') == "abc"

    def test_single_quotes(self):
        assert remove_quotes("'abc'") == "abc"

    def test_backticks(self):
        assert remove_quotes("`abc`") == "abc"

    def test_mismatched_quotes(self):
        assert remove_quotes('"abc\'') == '"abc\''

    def test_empty_single_quotes(self):
        # '""' -> '' after stripping; careful about empty string input
        assert remove_quotes("''") == ""

    def test_single_character(self):
        # val[0] == val[-1] and it's the same char; stripping gives ""
        assert remove_quotes("'") == ""

    def test_unquoted_single_char(self):
        assert remove_quotes("a") == "a"

    def test_quote_only_at_start(self):
        assert remove_quotes('"abc') == '"abc'

    def test_quote_only_at_end(self):
        assert remove_quotes('abc"') == 'abc"'


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class FakeGroup:
    def __init__(self, sublists=None, name="group"):
        self._sublists = sublists or []
        self.name = name

    def get_sublists(self):
        return iter(self._sublists)


class SpecialGroup(FakeGroup):
    pass


class TestRecurse:
    def test_calls_function_on_root(self):
        calls = []

        @recurse()
        def visit(tlist):
            calls.append(tlist)

        root = FakeGroup()
        visit(root)
        assert calls == [root]

    def test_recurses_into_sublists(self):
        visited = []
        child1 = FakeGroup(name="c1")
        child2 = FakeGroup(name="c2")
        root = FakeGroup([child1, child2], name="root")

        @recurse()
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        # children visited first, then root
        assert visited == ["c1", "c2", "root"]

    def test_nested_recursion(self):
        visited = []
        grand = FakeGroup(name="grand")
        child = FakeGroup([grand], name="child")
        root = FakeGroup([child], name="root")

        @recurse()
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        assert visited == ["grand", "child", "root"]

    def test_does_not_recurse_into_excluded_classes(self):
        visited = []
        grand = SpecialGroup(name="grand")
        child = SpecialGroup([grand], name="child")
        root = FakeGroup([child], name="root")

        @recurse(SpecialGroup)
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        # child is SpecialGroup -> skipped recursion and not visited;
        # grand is only reachable through child, so not visited.
        assert visited == ["root"]

    def test_excluded_class_still_reachable_from_other_path(self):
        visited = []
        special = SpecialGroup(name="special")
        normal = FakeGroup([special], name="normal")
        root = FakeGroup([normal], name="root")

        @recurse(SpecialGroup)
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        assert visited == ["normal", "root"]

    def test_multiple_excluded_classes(self):
        visited = []
        s1 = SpecialGroup(name="s1")
        root = FakeGroup([s1], name="root")

        @recurse(SpecialGroup, FakeGroup)
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        # root is FakeGroup -> not recursed (but still called at top)
        assert visited == ["root"]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype=None, value="", match_result=False):
        self.ttype = ttype
        self.value = value
        self._match_result = match_result
        self.match_calls = []

    def match(self, *args):
        self.match_calls.append(args)
        return self._match_result


class TestImt:
    def test_none_token(self):
        assert imt(None) is False

    def test_none_token_with_match_args(self):
        assert imt(None, i=str, m=("a",), t=("b",)) is False

    def test_instance_match(self):
        token = FakeToken()
        assert imt(token, i=FakeToken) is True

    def test_instance_tuple_match(self):
        token = FakeToken()
        assert imt(token, i=(int, str)) is False
        assert imt(token, i=(int, FakeToken)) is True

    def test_no_match_returns_false(self):
        token = FakeToken()
        assert imt(token) is False

    def test_match_single(self):
        token = FakeToken(match_result=True)
        assert imt(token, m=("a", "b")) is True
        assert token.match_calls == [("a", "b")]

    def test_match_single_false(self):
        token = FakeToken(match_result=False)
        assert imt(token, m=("a", "b")) is False

    def test_match_list_any_true(self):
        token = FakeToken(match_result=True)
        assert imt(token, m=[("a",), ("b",)]) is True
        # any() should short-circuit after first True
        assert token.match_calls == [("a",)]

    def test_match_list_all_false(self):
        token = FakeToken(match_result=False)
        assert imt(token, m=[("a",), ("b",)]) is False
        assert token.match_calls == [("a",), ("b",)]

    def test_ttype_single(self):
        token = FakeToken(ttype="foo")
        assert imt(token, t="foo") is True
        assert imt(token, t="bar") is False

    def test_ttype_in_tuple(self):
        # token.ttype in t where t is a tuple of strings
        token = FakeToken(ttype="foo")
        assert imt(token, t=("foo", "bar")) is True
        assert imt(token, t=("baz", "bar")) is False

    def test_ttype_list_of_tuples(self):
        token = FakeToken(ttype="foo")
        assert imt(token, t=[("foo", "bar"), ("baz",)]) is True
        assert imt(token, t=[("qux",), ("baz",)]) is False

    def test_instance_takes_precedence(self):
        token = FakeToken(match_result=True)
        # instance check passes; should not consult match
        assert imt(token, i=FakeToken, m=("x",)) is True
        assert token.match_calls == []

    def test_instance_fails_then_match(self):
        token = FakeToken(match_result=True)
        assert imt(token, i=int, m=("x",)) is True
        assert token.match_calls == [("x",)]

    def test_match_and_ttype(self):
        token = FakeToken(ttype="foo", match_result=False)
        assert imt(token, m=("x",), t="foo") is True

    def test_empty_args_falsy(self):
        token = FakeToken()
        # i, m, t all falsy means return False
        assert imt(token, i=None, m=None, t=None) is False


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_consume_partial(self):
        it = iter([1, 2, 3, 4, 5])
        consume(it, 2)
        assert list(it) == [3, 4, 5]

    def test_consume_zero(self):
        it = iter([1, 2, 3])
        consume(it, 0)
        assert list(it) == [1, 2, 3]

    def test_consume_all_with_none(self):
        it = iter([1, 2, 3])
        consume(it, None)
        assert list(it) == []

    def test_consume_more_than_available(self):
        it = iter([1, 2])
        consume(it, 10)
        assert list(it) == []

    def test_consume_negative(self):
        # islice with negative stop raises ValueError
        it = iter([1, 2, 3])
        with pytest.raises(ValueError):
            consume(it, -1)


# ---------------------------------------------------------------------------
# offset context manager
# ---------------------------------------------------------------------------

class TestOffset:
    def test_default_n(self):
        f = mock.Mock()
        f.offset = 5
        with offset(f):
            assert f.offset == 5
        assert f.offset == 5

    def test_increases_and_restores(self):
        f = mock.Mock()
        f.offset = 0
        with offset(f, 3):
            assert f.offset == 3
        assert f.offset == 0

    def test_negative(self):
        f = mock.Mock()
        f.offset = 10
        with offset(f, -4):
            assert f.offset == 6
        assert f.offset == 10

    def test_restores_on_exception(self):
        f = mock.Mock()
        f.offset = 2
        with pytest.raises(RuntimeError):
            with offset(f, 5):
                assert f.offset == 7
                raise RuntimeError("boom")
        assert f.offset == 2

    def test_nested(self):
        f = mock.Mock()
        f.offset = 0
        with offset(f, 1):
            with offset(f, 2):
                assert f.offset == 3
            assert f.offset == 1
        assert f.offset == 0


# ---------------------------------------------------------------------------
# indent context manager
# ---------------------------------------------------------------------------

class TestIndent:
    def test_default_n(self):
        f = mock.Mock()
        f.indent = 4
        with indent(f):
            assert f.indent == 5
        assert f.indent == 4

    def test_custom_n(self):
        f = mock.Mock()
        f.indent = 0
        with indent(f, 3):
            assert f.indent == 3
        assert f.indent == 0

    def test_negative(self):
        f = mock.Mock()
        f.indent = 5
        with indent(f, -2):
            assert f.indent == 3
        assert f.indent == 5

    def test_restores_on_exception(self):
        f = mock.Mock()
        f.indent = 1
        with pytest.raises(ValueError):
            with indent(f, 2):
                assert f.indent == 3
                raise ValueError("nope")
        assert f.indent == 1

    def test_nested(self):
        f = mock.Mock()
        f.indent = 0
        with indent(f):
            with indent(f):
                assert f.indent == 2
            assert f.indent == 1
        assert f.indent == 0
