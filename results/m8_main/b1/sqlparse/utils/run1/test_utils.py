import pytest

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
    def test_simple_single_line(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_split_on_lf(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_split_on_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_split_on_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_only_newline(self):
        assert split_unquoted_newlines("\n") == ["", ""]

    def test_newline_inside_double_quotes(self):
        assert split_unquoted_newlines('a "b\nc" d') == ['a "b\nc" d']

    def test_newline_inside_single_quotes(self):
        assert split_unquoted_newlines("a 'b\nc' d") == ["a 'b\nc' d"]

    def test_newline_inside_crlf_quotes(self):
        assert split_unquoted_newlines('"x\r\ny"') == ['"x\r\ny"']

    def test_escaped_quote_inside_string(self):
        text = '"a\\"b\nc"'
        assert split_unquoted_newlines(text) == [text]

    def test_mixed_quoted_and_unquoted(self):
        result = split_unquoted_newlines("a\n'b\nc'\nd")
        assert result == ["a", "'b\nc'", "d"]

    def test_string_conversion(self):
        class WithStr:
            def __str__(self):
                return "x\ny"

        assert split_unquoted_newlines(WithStr()) == ["x", "y"]

    def test_non_string_object_with_str(self):
        assert split_unquoted_newlines(123) == ["123"]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------

class TestRemoveQuotes:
    def test_none_returns_none(self):
        assert remove_quotes(None) is None

    def test_double_quotes(self):
        assert remove_quotes('"abc"') == "abc"

    def test_single_quotes(self):
        assert remove_quotes("'abc'") == "abc"

    def test_backtick_quotes(self):
        assert remove_quotes("`abc`") == "abc"

    def test_no_quotes(self):
        assert remove_quotes("abc") == "abc"

    def test_mismatched_quotes(self):
        assert remove_quotes('"abc\'') == '"abc\''

    def test_only_opening(self):
        assert remove_quotes('"abc') == '"abc'

    def test_empty_string(self):
        # val[0] would raise IndexError; ensure behavior on empty input
        with pytest.raises(IndexError):
            remove_quotes("")

    def test_single_char_quote(self):
        # single char: val[0] == val[-1] both are the quote char -> strips both,
        # but that would be val[1:-1] of length 1 -> empty
        assert remove_quotes('"') == ""

    def test_nested_quotes_not_stripped_fully(self):
        assert remove_quotes('"abc"def"') == 'abc"def'

    def test_wrong_outer_with_inner(self):
        assert remove_quotes("'abc\"") == "'abc\""


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class FakeTokenList:
    def __init__(self, sublists=None):
        self._sublists = sublists or []

    def get_sublists(self):
        return self._sublists


class ExcludedType(FakeTokenList):
    pass


class IncludedType(FakeTokenList):
    pass


class TestRecurse:
    def test_applies_to_root(self):
        calls = []

        @recurse()
        def visit(tlist):
            calls.append(tlist)

        root = FakeTokenList()
        visit(root)
        assert calls == [root]

    def test_recurses_into_sublists(self):
        child = FakeTokenList()
        root = FakeTokenList([child])
        calls = []

        @recurse()
        def visit(tlist):
            calls.append(tlist)

        visit(root)
        # child visited first (post-order), then root
        assert calls == [child, root]

    def test_does_not_recurse_into_excluded(self):
        excluded = ExcludedType()
        root = FakeTokenList([excluded])
        calls = []

        @recurse(ExcludedType)
        def visit(tlist):
            calls.append(tlist)

        visit(root)
        assert calls == [root]

    def test_recurses_into_nonexcluded_among_excluded(self):
        excluded = ExcludedType()
        included = IncludedType([FakeTokenList()])
        root = FakeTokenList([excluded, included])
        calls = []

        @recurse(ExcludedType)
        def visit(tlist):
            calls.append(tlist)

        visit(root)
        # included and its child visited, excluded skipped, root visited
        assert calls[0] is included._sublists[0]
        assert included in calls
        assert excluded not in calls
        assert calls[-1] is root

    def test_multiple_classes(self):
        e1 = ExcludedType()
        e2 = ExcludedType()
        child = FakeTokenList()
        root = FakeTokenList([e1, e2, child])
        calls = []

        @recurse(ExcludedType)
        def visit(tlist):
            calls.append(tlist)

        visit(root)
        assert e1 not in calls
        assert e2 not in calls
        assert child in calls
        assert calls[-1] is root


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype=None, match_result=None):
        self.ttype = ttype
        self._match_result = match_result

    def match(self, *args):
        self._last_match_args = args
        return self._match_result


class TestImt:
    def test_none_token(self):
        assert imt(None) is False

    def test_instance_match(self):
        tok = FakeToken()
        assert imt(tok, i=FakeToken) is True

    def test_instance_no_match(self):
        tok = FakeToken()
        assert imt(tok, i=str) is False

    def test_instance_tuple(self):
        tok = FakeToken()
        assert imt(tok, i=(str, FakeToken)) is True

    def test_match_single(self):
        tok = FakeToken(match_result=True)
        assert imt(tok, m=("X", "y")) is True

    def test_match_single_false(self):
        tok = FakeToken(match_result=False)
        assert imt(tok, m=("X", "y")) is False

    def test_match_list_any_true(self):
        tok = FakeToken(match_result=True)
        assert imt(tok, m=[("A", "b"), ("C", "d")]) is True

    def test_match_list_none_true(self):
        tok = FakeToken(match_result=False)
        assert imt(tok, m=[("A", "b"), ("C", "d")]) is False

    def test_ttype_single_eq(self):
        tok = FakeToken(ttype="TT")
        assert imt(tok, t="TT") is True

    def test_ttype_no_eq(self):
        tok = FakeToken(ttype="TT")
        assert imt(tok, t="OTHER") is False

    def test_ttype_list_membership(self):
        tok = FakeToken(ttype="B")
        assert imt(tok, t=["A", "B"]) is True

    def test_ttype_list_no_membership(self):
        tok = FakeToken(ttype="C")
        assert imt(tok, t=["A", "B"]) is False

    def test_i_takes_precedence_returns_true(self):
        tok = FakeToken(ttype="WRONG", match_result=False)
        assert imt(tok, i=FakeToken, m=("A", "b"), t="X") is True

    def test_match_and_ttype(self):
        tok = FakeToken(ttype="TT", match_result=False)
        # m fails but t succeeds
        assert imt(tok, m=("A", "b"), t="TT") is True

    def test_nothing_given_false(self):
        tok = FakeToken(ttype="TT", match_result=True)
        assert imt(tok) is False


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_consume_partial(self):
        it = iter(range(10))
        consume(it, 3)
        assert next(it) == 3

    def test_consume_zero(self):
        it = iter(range(5))
        consume(it, 0)
        assert next(it) == 0

    def test_consume_all_remaining(self):
        it = iter(range(3))
        consume(it, 10)
        assert list(it) == []

    def test_consume_none_entirely(self):
        it = iter(range(5))
        consume(it, None)
        assert list(it) == []

    def test_consume_list_iterator(self):
        data = iter(["a", "b", "c"])
        consume(data, 2)
        assert next(data) == "c"

    def test_consume_larger_than_length(self):
        it = iter([1])
        consume(it, 5)
        assert list(it) == []


# ---------------------------------------------------------------------------
# offset context manager
# ---------------------------------------------------------------------------

class _Filter:
    def __init__(self):
        self.offset = 0


class TestOffset:
    def test_default_increases_by_zero(self):
        f = _Filter()
        with offset(f):
            assert f.offset == 0
        assert f.offset == 0

    def test_increases_inside_and_restores(self):
        f = _Filter()
        with offset(f, 5):
            assert f.offset == 5
        assert f.offset == 0

    def test_restores_on_exception(self):
        # The context manager has no try/finally, so on exception the offset
        # is NOT restored (the decrement after `yield` is skipped).
        f = _Filter()
        with pytest.raises(RuntimeError):
            with offset(f, 3):
                assert f.offset == 3
                raise RuntimeError("boom")
        assert f.offset == 3

    def test_relative_to_existing_offset(self):
        f = _Filter()
        f.offset = 10
        with offset(f, 2):
            assert f.offset == 12
        assert f.offset == 10

    def test_negative_offset(self):
        f = _Filter()
        with offset(f, -3):
            assert f.offset == -3
        assert f.offset == 0


# ---------------------------------------------------------------------------
# indent context manager
# ---------------------------------------------------------------------------

class _IndentFilter:
    def __init__(self):
        self.indent = 0


class TestIndent:
    def test_default_increases_by_one(self):
        f = _IndentFilter()
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_custom_amount(self):
        f = _IndentFilter()
        with indent(f, 4):
            assert f.indent == 4
        assert f.indent == 0

    def test_restores_on_exception(self):
        # Like `offset`, `indent` lacks try/finally, so the decrement is not
        # executed when the body raises.
        f = _IndentFilter()
        with pytest.raises(ValueError):
            with indent(f, 2):
                assert f.indent == 2
                raise ValueError("bad")
        assert f.indent == 2

    def test_relative_to_existing_indent(self):
        f = _IndentFilter()
        f.indent = 3
        with indent(f, 2):
            assert f.indent == 5
        assert f.indent == 3

    def test_nested(self):
        f = _IndentFilter()
        with indent(f):
            assert f.indent == 1
            with indent(f):
                assert f.indent == 2
            assert f.indent == 1
        assert f.indent == 0
