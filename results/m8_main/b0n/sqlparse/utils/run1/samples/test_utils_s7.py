import pytest
from collections import deque

from sqlparse.utils import (
    split_unquoted_newlines,
    remove_quotes,
    recurse,
    imt,
    consume,
    offset,
    indent,
    SPLIT_REGEX,
    LINE_MATCH,
)


# ---------------------------------------------------------------------------
# split_unquoted_newlines
# ---------------------------------------------------------------------------

class TestSplitUnquotedNewlines:
    def test_simple_no_newline(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_simple_lf(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_simple_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_simple_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_double_quoted_string_keeps_newline(self):
        result = split_unquoted_newlines('SELECT "a\nb"')
        assert result == ['SELECT "a\nb"']

    def test_single_quoted_string_keeps_newline(self):
        result = split_unquoted_newlines("SELECT 'a\nb'")
        assert result == ["SELECT 'a\nb'"]

    def test_escaped_quote_in_double_quotes(self):
        # A backslash-escaped quote must not terminate the string early.
        result = split_unquoted_newlines('"a\\"b\nc"')
        assert result == ['"a\\"b\nc"']

    def test_escaped_quote_in_single_quotes(self):
        result = split_unquoted_newlines("'a\\'b\nc'")
        assert result == ["'a\\'b\nc'"]

    def test_mixed_quoted_and_unquoted(self):
        text = "'a\nb'\n'c\nd'"
        assert split_unquoted_newlines(text) == ["'a\nb'", "'c\nd'"]

    def test_non_string_converted(self):
        # str(stmt) applied to the argument.
        assert split_unquoted_newlines(123) == ["123"]

    def test_bytes_converted(self):
        assert split_unquoted_newlines(b"abc") == ["b'abc'"]

    def test_quoted_with_surrounding_text(self):
        text = 'x = "a\nb" AND y = 1'
        assert split_unquoted_newlines(text) == ['x = "a\nb" AND y = 1']

    def test_only_newline(self):
        assert split_unquoted_newlines("\n") == ["", ""]


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

    def test_only_leading_quote(self):
        assert remove_quotes('"abc') == '"abc'

    def test_only_trailing_quote(self):
        assert remove_quotes('abc"') == 'abc"'

    def test_empty_string(self):
        # val[0] on empty string raises IndexError.
        with pytest.raises(IndexError):
            remove_quotes("")

    def test_single_character_quote(self):
        # A single quote char: val[0] == val[-1], so result is '' after slicing.
        assert remove_quotes('"') == ""

    def test_double_quoted_empty(self):
        assert remove_quotes('""') == ""

    def test_nested_quotes_inside(self):
        assert remove_quotes('"a\'b"') == "a'b"

    def test_non_quote_prefix(self):
        assert remove_quotes('x"y"') == 'x"y"'


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class FakeList:
    """Minimal stand-in for a TokenList usable by recurse."""

    def __init__(self, sublists=None, name="root"):
        self._sublists = sublists or []
        self.hits = 0
        self.name = name

    def get_sublists(self):
        return self._sublists


class FakeSub:
    def __init__(self, name):
        self.name = name


class TestRecurse:
    def test_applies_function_on_each_node(self):
        leaf1 = FakeList(name="leaf1")
        leaf2 = FakeList(name="leaf2")
        mid = FakeList(sublists=[leaf1, leaf2], name="mid")
        root = FakeList(sublists=[mid], name="root")

        visited = []

        @recurse()
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        # Post-order: leaves first, then their parent, then root.
        assert visited == ["leaf1", "leaf2", "mid", "root"]

    def test_skip_given_class(self):
        leaf = FakeList(name="leaf")
        skipped = FakeSub(name="skipped")
        root = FakeList(sublists=[leaf, skipped], name="root")

        visited = []

        @recurse(FakeSub)
        def visit(tlist):
            visited.append(tlist)

        visit(root)
        # The FakeSub should be not visited and not recursed into.
        assert visited == [leaf, root]
        assert skipped not in visited

    def test_skip_prevents_recursion_into_skipped(self):
        inner = FakeList(name="inner")
        skipped = FakeSub(name="skipped")
        # Give the skipped node a get_sublists that would blow up if called.
        def bad_get_sublists():
            raise AssertionError("should not recurse into skipped type")
        skipped.get_sublists = bad_get_sublists

        root = FakeList(sublists=[skipped], name="root")
        visited = []

        @recurse(FakeSub)
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        assert visited == ["root"]

    def test_no_sublists(self):
        root = FakeList(name="root")
        visited = []

        @recurse()
        def visit(tlist):
            visited.append(tlist.name)

        visit(root)
        assert visited == ["root"]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype=None, value="", match_result=False):
        self.ttype = ttype
        self.value = value
        self._match_result = match_result

    def match(self, ttype, value):
        # Record the arguments used and return the configured result.
        self.last_match_args = (ttype, value)
        return self._match_result


class TestImt:
    def test_none_token(self):
        assert imt(None) is False

    def test_none_token_with_conditions(self):
        assert imt(None, i=object, m=(None, None), t=None) is False

    def test_instance_match(self):
        tok = FakeToken()
        assert imt(tok, i=FakeToken) is True

    def test_instance_no_match(self):
        tok = FakeToken()
        assert imt(tok, i=str) is False

    def test_instance_tuple(self):
        tok = FakeToken()
        assert imt(tok, i=(str, FakeToken)) is True

    def test_nothing_provided(self):
        tok = FakeToken()
        assert imt(tok) is False

    def test_match_single(self):
        tok = FakeToken(match_result=True)
        assert imt(tok, m=("TOKEN", "val")) is True
        assert tok.last_match_args == ("TOKEN", "val")

    def test_match_single_no_match(self):
        tok = FakeToken(match_result=False)
        assert imt(tok, m=("TOKEN", "val")) is False

    def test_match_list_any(self):
        tok = FakeToken(match_result=True)
        assert imt(tok, m=[("A", "a"), ("B", "b")]) is True

    def test_match_list_none_match(self):
        tok = FakeToken(match_result=False)
        assert imt(tok, m=[("A", "a"), ("B", "b")]) is False

    def test_ttype_single(self):
        tok = FakeToken(ttype="myttype")
        assert imt(tok, t="myttype") is True

    def test_ttype_single_no_match(self):
        tok = FakeToken(ttype="other")
        assert imt(tok, t="myttype") is False

    def test_ttype_tuple(self):
        tok = FakeToken(ttype="b")
        assert imt(tok, t=("a", "b")) is True

    def test_ttype_list_containing_sets(self):
        tok = FakeToken(ttype="a")
        assert imt(tok, t=[("a", "b"), ("c", "d")]) is True

    def test_ttype_list_no_match(self):
        tok = FakeToken(ttype="z")
        assert imt(tok, t=[("a", "b"), ("c", "d")]) is False

    def test_precedence_instance_first(self):
        # Instance check wins before match check.
        tok = FakeToken(match_result=False)
        assert imt(tok, i=FakeToken, m=("A", "a")) is True

    def test_instance_false_then_match(self):
        tok = FakeToken(match_result=True)
        assert imt(tok, i=str, m=("A", "a")) is True

    def test_match_then_ttype(self):
        tok = FakeToken(match_result=False, ttype="t")
        assert imt(tok, m=("A", "a"), t="t") is True

    def test_all_false(self):
        tok = FakeToken(match_result=False, ttype="t")
        assert imt(tok, i=str, m=("A", "a"), t="other") is False


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_consume_none_consumes_all(self):
        it = iter([1, 2, 3, 4])
        consume(it, None)
        assert list(it) == []

    def test_consume_some(self):
        it = iter([1, 2, 3, 4])
        consume(it, 2)
        assert list(it) == [3, 4]

    def test_consume_zero(self):
        it = iter([1, 2, 3])
        consume(it, 0)
        assert list(it) == [1, 2, 3]

    def test_consume_more_than_available(self):
        it = iter([1, 2])
        consume(it, 10)
        assert list(it) == []

    def test_consume_generator(self):
        consumed = []

        def gen():
            for i in range(5):
                consumed.append(i)
                yield i

        g = gen()
        consume(g, 3)
        assert consumed == [0, 1, 2]
        assert list(g) == [3, 4]

    def test_consume_empty(self):
        it = iter([])
        consume(it, 5)
        assert list(it) == []


# ---------------------------------------------------------------------------
# offset / indent contexts
# ---------------------------------------------------------------------------

class FakeFilter:
    def __init__(self):
        self.offset = 10
        self.indent = 5


class TestOffset:
    def test_offset_default(self):
        f = FakeFilter()
        with offset(f):
            assert f.offset == 10
        assert f.offset == 10

    def test_offset_positive(self):
        f = FakeFilter()
        with offset(f, 3):
            assert f.offset == 13
        assert f.offset == 10

    def test_offset_negative(self):
        f = FakeFilter()
        with offset(f, -4):
            assert f.offset == 6
        assert f.offset == 10

    def test_offset_restored_on_exception(self):
        f = FakeFilter()
        with pytest.raises(ValueError):
            with offset(f, 7):
                assert f.offset == 17
                raise ValueError("boom")
        assert f.offset == 10

    def test_offset_nested(self):
        f = FakeFilter()
        with offset(f, 1):
            assert f.offset == 11
            with offset(f, 2):
                assert f.offset == 13
            assert f.offset == 11
        assert f.offset == 10


class TestIndent:
    def test_indent_default(self):
        f = FakeFilter()
        with indent(f):
            assert f.indent == 6
        assert f.indent == 5

    def test_indent_custom(self):
        f = FakeFilter()
        with indent(f, 3):
            assert f.indent == 8
        assert f.indent == 5

    def test_indent_negative(self):
        f = FakeFilter()
        with indent(f, -2):
            assert f.indent == 3
        assert f.indent == 5

    def test_indent_restored_on_exception(self):
        f = FakeFilter()
        with pytest.raises(RuntimeError):
            with indent(f, 4):
                assert f.indent == 9
                raise RuntimeError("boom")
        assert f.indent == 5

    def test_indent_nested(self):
        f = FakeFilter()
        with indent(f, 1):
            assert f.indent == 6
            with indent(f, 2):
                assert f.indent == 8
            assert f.indent == 6
        assert f.indent == 5
