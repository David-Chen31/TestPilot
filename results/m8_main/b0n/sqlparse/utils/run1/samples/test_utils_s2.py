"""Tests for sqlparse.utils."""

import pytest

from sqlparse import sql, tokens as T
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
    def test_single_line(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_split_lf(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_split_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_split_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_newline_in_single_quotes(self):
        assert split_unquoted_newlines("'a\nb'") == ["'a\nb'"]

    def test_newline_in_double_quotes(self):
        assert split_unquoted_newlines('"a\nb"') == ['"a\nb"']

    def test_newline_in_quotes_with_surrounding(self):
        assert split_unquoted_newlines("x 'a\nb' y") == ["x 'a\nb' y"]

    def test_newline_after_quotes(self):
        assert split_unquoted_newlines("'a'\nb") == ["'a'", "b"]

    def test_escaped_quote_in_string(self):
        assert split_unquoted_newlines(r"'a\'b'") == [r"'a\'b'"]

    def test_accepts_non_str_via_str_conversion(self):
        assert split_unquoted_newlines(42) == ["42"]

    def test_multiple_lines_mixed(self):
        result = split_unquoted_newlines("a\n'b\nc'\r\nd")
        assert result == ["a", "'b\nc'", "d"]


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
        assert remove_quotes("\"abc'") == "\"abc'"

    def test_only_one_char(self):
        # val[0] == val[-1] is True but len 1 would strip both edges -> ''
        assert remove_quotes("'") == ""

    def test_inner_quotes_kept(self):
        assert remove_quotes("'a'b'") == "a'b"

    def test_empty_string(self):
        assert remove_quotes("") == ""

    def test_unquoted_edge(self):
        assert remove_quotes("ab\"") == "ab\""


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class _FakeToken:
    def __init__(self, sublists=None):
        self._sublists = sublists or []

    def get_sublists(self):
        return self._sublists


class _A(_FakeToken):
    pass


class _B(_FakeToken):
    pass


class TestRecurse:
    def test_applies_to_leaf(self):
        calls = []

        @recurse()
        def visitor(tlist):
            calls.append(tlist)

        node = _FakeToken()
        visitor(node)
        assert calls == [node]

    def test_recurses_into_sublists(self):
        child = _FakeToken()
        parent = _FakeToken([child])
        calls = []

        @recurse()
        def visitor(tlist):
            calls.append(tlist)

        visitor(parent)
        # children visited before parent
        assert calls == [child, parent]

    def test_skips_excluded_class(self):
        excluded = _A()
        child = _FakeToken()
        parent = _FakeToken([excluded, child])
        calls = []

        @recurse(_A)
        def visitor(tlist):
            calls.append(tlist)

        visitor(parent)
        assert excluded not in calls
        assert calls == [child, parent]

    def test_skips_excluded_class_multiple(self):
        a = _A()
        b = _B()
        parent = _FakeToken([a, b])
        calls = []

        @recurse(_A, _B)
        def visitor(tlist):
            calls.append(tlist)

        visitor(parent)
        assert calls == [parent]

    def test_deeply_nested(self):
        grandchild = _FakeToken()
        child = _FakeToken([grandchild])
        root = _FakeToken([child])
        calls = []

        @recurse()
        def visitor(tlist):
            calls.append(tlist)

        visitor(root)
        assert calls == [grandchild, child, root]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class TestImt:
    def _tz(self, ttype, value="x"):
        return sql.Token(ttype, value)

    def test_none_token(self):
        assert imt(None) is False

    def test_no_criteria(self):
        tok = self._tz(T.Keyword)
        assert imt(tok) is False

    def test_instance_match(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, i=sql.Token) is True

    def test_instance_no_match(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, i=sql.Identifier) is False

    def test_instance_tuple(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, i=(sql.Identifier, sql.Token)) is True

    def test_match_single(self):
        tok = self._tz(T.Keyword, "SELECT")
        assert imt(tok, m=(T.Keyword, "SELECT")) is True

    def test_match_single_no(self):
        tok = self._tz(T.Keyword, "SELECT")
        assert imt(tok, m=(T.Keyword, "FROM")) is False

    def test_match_list(self):
        tok = self._tz(T.Keyword, "FROM")
        patterns = [(T.Keyword, "SELECT"), (T.Keyword, "FROM")]
        assert imt(tok, m=patterns) is True

    def test_match_list_no(self):
        tok = self._tz(T.Keyword, "WHERE")
        patterns = [(T.Keyword, "SELECT"), (T.Keyword, "FROM")]
        assert imt(tok, m=patterns) is False

    def test_ttype_single(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, t=T.Keyword) is True

    def test_ttype_single_no(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, t=T.Name) is False

    def test_ttype_list(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, t=[T.Name, T.Keyword]) is True

    def test_ttype_list_no(self):
        tok = self._tz(T.Keyword)
        assert imt(tok, t=[T.Name, T.String]) is False

    def test_instance_short_circuits(self):
        tok = self._tz(T.Keyword, "SELECT")
        # Instance matches => True without checking match/ttype
        assert imt(tok, i=sql.Token, m=(T.Keyword, "NOPE")) is True

    def test_match_then_ttype(self):
        tok = self._tz(T.Keyword, "SELECT")
        # match fails but ttype matches
        assert imt(tok, m=(T.Keyword, "FROM"), t=T.Keyword) is True


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_consume_n(self):
        it = iter(range(10))
        consume(it, 3)
        assert next(it) == 3

    def test_consume_all(self):
        it = iter(range(5))
        consume(it, None)
        assert list(it) == []

    def test_consume_zero(self):
        it = iter(range(5))
        consume(it, 0)
        assert next(it) == 0

    def test_consume_more_than_available(self):
        it = iter(range(3))
        consume(it, 10)
        assert list(it) == []

    def test_consume_on_list_iterator(self):
        data = [1, 2, 3, 4]
        it = iter(data)
        consume(it, 2)
        assert list(it) == [3, 4]


# ---------------------------------------------------------------------------
# offset context manager
# ---------------------------------------------------------------------------

class _Filter:
    def __init__(self):
        self.offset = 0
        self.indent = 0


class TestOffset:
    def test_default_n(self):
        f = _Filter()
        with offset(f):
            assert f.offset == 0
        assert f.offset == 0

    def test_adds_and_restores(self):
        f = _Filter()
        with offset(f, 5):
            assert f.offset == 5
        assert f.offset == 0

    def test_nested(self):
        f = _Filter()
        with offset(f, 2):
            assert f.offset == 2
            with offset(f, 3):
                assert f.offset == 5
            assert f.offset == 2
        assert f.offset == 0

    def test_restores_on_exception(self):
        f = _Filter()
        with pytest.raises(ValueError):
            with offset(f, 4):
                assert f.offset == 4
                raise ValueError("boom")
        assert f.offset == 0

    def test_negative(self):
        f = _Filter()
        with offset(f, -3):
            assert f.offset == -3
        assert f.offset == 0

    def test_existing_offset(self):
        f = _Filter()
        f.offset = 10
        with offset(f, 5):
            assert f.offset == 15
        assert f.offset == 10


# ---------------------------------------------------------------------------
# indent context manager
# ---------------------------------------------------------------------------

class TestIndent:
    def test_default(self):
        f = _Filter()
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_explicit_n(self):
        f = _Filter()
        with indent(f, 3):
            assert f.indent == 3
        assert f.indent == 0

    def test_nested(self):
        f = _Filter()
        with indent(f):
            assert f.indent == 1
            with indent(f):
                assert f.indent == 2
            assert f.indent == 1
        assert f.indent == 0

    def test_restores_on_exception(self):
        f = _Filter()
        with pytest.raises(RuntimeError):
            with indent(f, 2):
                assert f.indent == 2
                raise RuntimeError("boom")
        assert f.indent == 0

    def test_existing_indent(self):
        f = _Filter()
        f.indent = 4
        with indent(f, 2):
            assert f.indent == 6
        assert f.indent == 4

    def test_negative(self):
        f = _Filter()
        with indent(f, -1):
            assert f.indent == -1
        assert f.indent == 0
