import pytest

from sqlparse import sql
from sqlparse import tokens as T
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

    def test_single_line(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_lf(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\n\nb") == ["a", "", "", "b"]

    def test_only_newline(self):
        assert split_unquoted_newlines("\n") == ["", ""]

    def test_newline_in_single_quotes(self):
        assert split_unquoted_newlines("a'x\ny'b") == ["a'x\ny'b"]

    def test_newline_in_double_quotes(self):
        assert split_unquoted_newlines('a"x\ny"b') == ['a"x\ny"b']

    def test_newline_mixed_with_quotes(self):
        result = split_unquoted_newlines("a\n'b\nc'\nd")
        assert result == ["a", "'b\nc'", "d"]

    def test_escaped_quote_in_string(self):
        assert split_unquoted_newlines("'a\\'b\nc'") == ["'a\\'b\nc'"]

    def test_non_string_input_converted(self):
        # str() is applied to the input
        class Foo:
            def __str__(self):
                return "x\ny"

        assert split_unquoted_newlines(Foo()) == ["x", "y"]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------

class TestRemoveQuotes:
    def test_none(self):
        assert remove_quotes(None) is None

    def test_double_quotes(self):
        assert remove_quotes('"abc"') == "abc"

    def test_single_quotes(self):
        assert remove_quotes("'abc'") == "abc"

    def test_backticks(self):
        assert remove_quotes("`abc`") == "abc"

    def test_no_quotes(self):
        assert remove_quotes("abc") == "abc"

    def test_mismatched_quotes(self):
        assert remove_quotes('"abc\'') == '"abc\''

    def test_only_opening_quote(self):
        # single char: val[0] == val[-1], so it gets stripped to empty
        assert remove_quotes('"') == ""

    def test_empty_string(self):
        # val[0] would raise IndexError on empty string
        with pytest.raises(IndexError):
            remove_quotes("")

    def test_inner_quotes_kept(self):
        assert remove_quotes('"a\'b"') == "a'b"


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class DummyList:
    def __init__(self, name, sublists=None):
        self.name = name
        self._sublists = sublists or []

    def get_sublists(self):
        return self._sublists


class OtherToken:
    """Used together with DummyList; not a DummyList."""


class Leaf:
    def __init__(self, name):
        self.name = name

    def get_sublists(self):
        return []


def test_recurse_visits_all_sublists():
    visited = []

    @recurse()
    def visit(tlist):
        visited.append(tlist.name)

    d = DummyList("root", [DummyList("a"), DummyList("b")])
    visit(d)
    assert visited == ["a", "b", "root"]


def test_recurse_skips_excluded_class():
    visited = []

    @recurse(DummyList)
    def visit(tlist):
        visited.append(tlist.name)

    d = DummyList("root", [DummyList("a")])
    visit(d)
    # excluded sublist not recursed into, but top-level f(DummyList) still called
    assert visited == ["root"]


def test_recurse_with_non_sublist_children():
    visited = []

    @recurse()
    def visit(tlist):
        visited.append(tlist.name)

    class Mixed(DummyList):
        pass

    leaf = Leaf("leaf")
    root = Mixed("root", [leaf])
    # Leaf is not DummyList-like, but recurse doesn't check type, just calls
    # wrapped_f which calls get_sublists() -> fine
    visit(root)
    assert visited == ["leaf", "root"]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype, value=""):
        self.ttype = ttype
        self.value = value

    def match(self, ttype, value):
        return self.ttype == ttype and self.value == value


class TestImt:
    def test_none_token(self):
        assert imt(None) is False

    def test_instance_match(self):
        tok = FakeToken(T.Name)
        assert imt(tok, i=FakeToken) is True

    def test_instance_no_match(self):
        tok = FakeToken(T.Name)
        assert imt(tok, i=sql.Identifier) is False

    def test_instance_tuple(self):
        tok = FakeToken(T.Name)
        assert imt(tok, i=(sql.Identifier, FakeToken)) is True

    def test_match_tuple(self):
        tok = FakeToken(T.Name, "x")
        assert imt(tok, m=(T.Name, "x")) is True
        assert imt(tok, m=(T.Name, "y")) is False

    def test_match_list(self):
        tok = FakeToken(T.Name, "x")
        assert imt(tok, m=[(T.Name, "y"), (T.Name, "x")]) is True
        assert imt(tok, m=[(T.Name, "y"), (T.Name, "z")]) is False

    def test_ttype_match(self):
        tok = FakeToken(T.Name)
        assert imt(tok, t=T.Name) is True
        assert imt(tok, t=T.String) is False

    def test_ttype_list(self):
        tok = FakeToken(T.Name)
        assert imt(tok, t=[T.String, T.Name]) is True
        assert imt(tok, t=[T.String, T.Number]) is False

    def test_ttype_token_type_group(self):
        # T.Name is T.Token.Name; membership in T.Token should match
        tok = FakeToken(T.Name)
        assert imt(tok, t=T.Token) is True

    def test_ttype_list_of_groups(self):
        tok = FakeToken(T.Name)
        assert imt(tok, t=[T.Token, T.Keyword]) is True

    def test_all_false(self):
        tok = FakeToken(T.Name)
        assert imt(tok) is False


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

def test_consume_partial():
    it = iter([1, 2, 3, 4, 5])
    consume(it, 2)
    assert next(it) == 3


def test_consume_all_with_none():
    it = iter([1, 2, 3, 4, 5])
    consume(it, None)
    assert list(it) == []


def test_consume_more_than_available():
    it = iter([1, 2])
    consume(it, 10)
    assert list(it) == []


def test_consume_zero():
    it = iter([1, 2, 3])
    consume(it, 0)
    assert next(it) == 1


# ---------------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------------

class OffsetHolder:
    def __init__(self, offset_val=0):
        self.offset = offset_val


def test_offset_adds_and_restores():
    holder = OffsetHolder(5)
    with offset(holder, 3):
        assert holder.offset == 8
    assert holder.offset == 5


def test_offset_default_n():
    holder = OffsetHolder(0)
    with offset(holder):
        assert holder.offset == 0
    assert holder.offset == 0


def test_offset_restores_on_exception():
    holder = OffsetHolder(2)
    with pytest.raises(RuntimeError):
        with offset(holder, 4):
            assert holder.offset == 6
            raise RuntimeError("boom")
    assert holder.offset == 2


def test_offset_negative():
    holder = OffsetHolder(10)
    with offset(holder, -3):
        assert holder.offset == 7
    assert holder.offset == 10


# ---------------------------------------------------------------------------
# indent
# ---------------------------------------------------------------------------

class IndentHolder:
    def __init__(self, indent_val=0):
        self.indent = indent_val


def test_indent_adds_and_restores():
    holder = IndentHolder(1)
    with indent(holder, 2):
        assert holder.indent == 3
    assert holder.indent == 1


def test_indent_default_n():
    holder = IndentHolder(0)
    with indent(holder):
        assert holder.indent == 1
    assert holder.indent == 0


def test_indent_restores_on_exception():
    holder = IndentHolder(0)
    with pytest.raises(ValueError):
        with indent(holder, 5):
            assert holder.indent == 5
            raise ValueError("boom")
    assert holder.indent == 0


def test_indent_negative():
    holder = IndentHolder(3)
    with indent(holder, -1):
        assert holder.indent == 2
    assert holder.indent == 3
