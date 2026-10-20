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
)
from sqlparse import sql, tokens as T


# --------------------------------------------------------------------------
# split_unquoted_newlines
# --------------------------------------------------------------------------

class TestSplitUnquotedNewlines:
    def test_empty_string(self):
        assert split_unquoted_newlines('') == ['']

    def test_single_line_no_newline(self):
        assert split_unquoted_newlines('SELECT 1') == ['SELECT 1']

    def test_simple_lf(self):
        assert split_unquoted_newlines('a\nb') == ['a', 'b']

    def test_simple_cr(self):
        assert split_unquoted_newlines('a\rb') == ['a', 'b']

    def test_simple_crlf(self):
        assert split_unquoted_newlines('a\r\nb') == ['a', 'b']

    def test_trailing_newline(self):
        assert split_unquoted_newlines('a\n') == ['a', '']

    def test_leading_newline(self):
        assert split_unquoted_newlines('\na') == ['', 'a']

    def test_multiple_newlines(self):
        assert split_unquoted_newlines('a\n\nb') == ['a', '', 'b']

    def test_crlf_not_doubled(self):
        # CRLF should count as a single newline.
        assert split_unquoted_newlines('a\r\nb') == ['a', 'b']

    def test_newline_in_single_quotes(self):
        assert split_unquoted_newlines("a'x\ny'b") == ["a'x\ny'b"]

    def test_newline_in_double_quotes(self):
        assert split_unquoted_newlines('a"x\ny"b') == ['a"x\ny"b']

    def test_escaped_quote_in_string(self):
        assert split_unquoted_newlines("a'x\\'y\nz'b") == ["a'x\\'y\nz'b"]

    def test_multiple_lines_with_quotes(self):
        text = "select 'a\nb' from t\nwhere x='c\r\nd'"
        assert split_unquoted_newlines(text) == [
            "select 'a\nb' from t",
            "where x='c\r\nd'",
        ]

    def test_mixed_quote_types(self):
        assert split_unquoted_newlines("'a'\n\"b\"") == ["'a'", '"b"']

    def test_converts_to_str(self):
        class StrLike:
            def __str__(self):
                return 'a\nb'

        assert split_unquoted_newlines(StrLike()) == ['a', 'b']


# --------------------------------------------------------------------------
# remove_quotes
# --------------------------------------------------------------------------

class TestRemoveQuotes:
    def test_none(self):
        assert remove_quotes(None) is None

    def test_empty_string(self):
        # val[0] / val[-1] would raise IndexError but empty string can't get
        # past the check... actually it does access val[0].
        # Given the implementation accesses val[0] and val[-1], an empty
        # string is not handled specially; it will raise IndexError.
        with pytest.raises(IndexError):
            remove_quotes('')

    def test_double_quotes(self):
        assert remove_quotes('"abc"') == 'abc'

    def test_single_quotes(self):
        assert remove_quotes("'abc'") == 'abc'

    def test_backticks(self):
        assert remove_quotes('`abc`') == 'abc'

    def test_no_quotes(self):
        assert remove_quotes('abc') == 'abc'

    def test_mismatched_quotes(self):
        assert remove_quotes('"abc\'') == '"abc\''

    def test_only_opening_quote(self):
        assert remove_quotes('"abc') == '"abc'

    def test_only_closing_quote(self):
        assert remove_quotes('abc"') == 'abc"'

    def test_single_char_quote(self):
        # A single quote char: val[0] == val[-1] and both are quote chars.
        assert remove_quotes('"') == ''

    def test_nested_quotes(self):
        assert remove_quotes('"\'a\'"') == "'a'"

    def test_non_quote_wrapping(self):
        assert remove_quotes('(abc)') == '(abc)'


# --------------------------------------------------------------------------
# recurse
# --------------------------------------------------------------------------

class _FakeGroup:
    """Minimal stand-in for a token list supporting get_sublists()."""

    def __init__(self, sublists=None):
        self._sublists = list(sublists or [])

    def get_sublists(self):
        return iter(self._sublists)


class TestRecurse:
    def test_visits_self(self):
        visited = []

        @recurse()
        def visit(tlist):
            visited.append(tlist)

        root = _FakeGroup()
        visit(root)
        assert visited == [root]

    def test_visits_children(self):
        visited = []

        @recurse()
        def visit(tlist):
            visited.append(tlist)

        child = _FakeGroup()
        root = _FakeGroup([child])
        visit(root)
        assert visited == [child, root]

    def test_skips_excluded_class(self):
        visited = []

        @recurse(sql.Identifier)
        def visit(tlist):
            visited.append(tlist)

        ident = sql.Identifier  # class reference
        dummy = _FakeGroup()

        class Excluded(_FakeGroup):
            pass

        # Build a hierarchy: root -> excluded subtree, root -> normal child.
        # Use sql.Identifier as the excluded type; we need instances of it.
        ident_node = sql.Identifier(sql.Token(T.Name, 'x'))
        # Give the identifier a sublist so recurse has to decide to skip it.
        root = _FakeGroup([ident_node, dummy])
        visit(root)

        # dummy is visited, ident_node (an Identifier) is not recursed into,
        # but its own f() is still not called since the check wraps recursion.
        # Actually the exclusion prevents descending into ident_node; its
        # sublists aren't traversed but ident_node itself isn't visited either
        # because the loop only calls wrapped_f on non-excluded sublists.
        assert dummy in visited
        assert root in visited
        assert ident_node not in visited

    def test_post_order_visit(self):
        order = []

        @recurse()
        def visit(tlist):
            order.append(tlist)

        grandchild = _FakeGroup()
        child = _FakeGroup([grandchild])
        root = _FakeGroup([child])
        visit(root)
        assert order == [grandchild, child, root]


# --------------------------------------------------------------------------
# imt
# --------------------------------------------------------------------------

def _tok(ttype, value='x'):
    return sql.Token(ttype, value)


class TestImt:
    def test_none_token(self):
        assert imt(None) is False

    def test_none_token_with_all_args(self):
        assert imt(None, i=sql.Token, m=(T.Name, 'x'), t=T.Name) is False

    def test_instance_match(self):
        tok = _tok(T.Name)
        assert imt(tok, i=sql.Token) is True
        assert imt(tok, i=sql.Identifier) is False

    def test_instance_tuple(self):
        tok = _tok(T.Name)
        assert imt(tok, i=(sql.Identifier, sql.Token)) is True

    def test_match_single(self):
        tok = _tok(T.Name, 'foo')
        assert imt(tok, m=(T.Name, 'foo')) is True
        assert imt(tok, m=(T.Name, 'bar')) is False

    def test_match_list(self):
        tok = _tok(T.Name, 'foo')
        assert imt(tok, m=[(T.Name, 'bar'), (T.Name, 'foo')]) is True
        assert imt(tok, m=[(T.Name, 'bar'), (T.Name, 'baz')]) is False

    def test_ttype_single(self):
        tok = _tok(T.Name)
        assert imt(tok, t=T.Name) is True
        assert imt(tok, t=T.String) is False

    def test_ttype_list(self):
        tok = _tok(T.Name)
        assert imt(tok, t=[T.String, T.Name]) is True
        assert imt(tok, t=[T.String, T.Number]) is False

    def test_ttype_tuple(self):
        tok = _tok(T.Name)
        assert imt(tok, t=(T.String, T.Name)) is True
        assert imt(tok, t=(T.String, T.Number)) is False

    def test_combined_args_instance_first(self):
        tok = _tok(T.Name)
        assert imt(tok, i=sql.Token, t=T.String) is True

    def test_all_false(self):
        tok = _tok(T.Name)
        assert imt(tok, i=sql.Identifier, m=(T.Name, 'other'), t=T.String) is False

    def test_match_after_instance_fails(self):
        tok = _tok(T.Name, 'foo')
        assert imt(tok, i=sql.Identifier, m=(T.Name, 'foo')) is True

    def test_ttype_after_match_fails(self):
        tok = _tok(T.Name, 'foo')
        assert imt(tok, m=(T.Name, 'bar'), t=T.Name) is True


# --------------------------------------------------------------------------
# consume
# --------------------------------------------------------------------------

class TestConsume:
    def test_consume_parts(self):
        it = iter([1, 2, 3, 4, 5])
        consume(it, 2)
        assert list(it) == [3, 4, 5]

    def test_consume_zero(self):
        it = iter([1, 2, 3])
        consume(it, 0)
        assert list(it) == [1, 2, 3]

    def test_consume_more_than_available(self):
        it = iter([1, 2])
        consume(it, 5)
        assert list(it) == []

    def test_consume_none_consumes_entirely(self):
        it = iter([1, 2, 3])
        consume(it, None)
        assert list(it) == []

    def test_consume_generator(self):
        def gen():
            yield from range(10)

        g = gen()
        consume(g, 4)
        assert list(g) == [4, 5, 6, 7, 8, 9]

    def test_consume_negative(self):
        it = iter([1, 2, 3])
        consume(it, -1)
        assert list(it) == [1, 2, 3]


# --------------------------------------------------------------------------
# offset / indent context managers
# --------------------------------------------------------------------------

class _FilterStub:
    def __init__(self, offset_val=0, indent_val=0):
        self.offset = offset_val
        self.indent = indent_val


class TestOffset:
    def test_default_n(self):
        f = _FilterStub()
        with offset(f):
            assert f.offset == 0
        assert f.offset == 0

    def test_positive_n(self):
        f = _FilterStub()
        with offset(f, 3):
            assert f.offset == 3
        assert f.offset == 0

    def test_restores_original(self):
        f = _FilterStub(offset_val=5)
        with offset(f, 2):
            assert f.offset == 7
        assert f.offset == 5

    def test_restores_on_exception(self):
        f = _FilterStub(offset_val=1)
        with pytest.raises(ValueError):
            with offset(f, 4):
                assert f.offset == 5
                raise ValueError('boom')
        assert f.offset == 1

    def test_nested(self):
        f = _FilterStub()
        with offset(f, 1):
            with offset(f, 2):
                assert f.offset == 3
            assert f.offset == 1
        assert f.offset == 0

    def test_negative_n(self):
        f = _FilterStub(offset_val=3)
        with offset(f, -1):
            assert f.offset == 2
        assert f.offset == 3


class TestIndent:
    def test_default_n(self):
        f = _FilterStub()
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_positive_n(self):
        f = _FilterStub()
        with indent(f, 3):
            assert f.indent == 3
        assert f.indent == 0

    def test_restores_original(self):
        f = _FilterStub(indent_val=4)
        with indent(f, 2):
            assert f.indent == 6
        assert f.indent == 4

    def test_restores_on_exception(self):
        f = _FilterStub(indent_val=1)
        with pytest.raises(RuntimeError):
            with indent(f, 2):
                assert f.indent == 3
                raise RuntimeError('boom')
        assert f.indent == 1

    def test_nested(self):
        f = _FilterStub()
        with indent(f):
            with indent(f, 3):
                assert f.indent == 4
            assert f.indent == 1
        assert f.indent == 0

    def test_negative_n(self):
        f = _FilterStub(indent_val=2)
        with indent(f, -1):
            assert f.indent == 1
        assert f.indent == 2
