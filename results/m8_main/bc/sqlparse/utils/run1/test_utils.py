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


# ---------------------------------------------------------------- split_unquoted_newlines

class TestSplitUnquotedNewlines:
    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_no_newlines(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_simple_newline(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_multiple_newlines(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_newline_inside_single_quotes(self):
        assert split_unquoted_newlines("'a\nb'") == ["'a\nb'"]

    def test_newline_inside_double_quotes(self):
        assert split_unquoted_newlines('"a\nb"') == ['"a\nb"']

    def test_mixed_quoted_and_unquoted(self):
        result = split_unquoted_newlines("a 'x\ny'\nb")
        assert result == ["a 'x\ny'", "b"]

    def test_escaped_quote_inside_string(self):
        # the backslash escape keeps the quote inside the string
        result = split_unquoted_newlines("'a\\'b\nc'")
        assert result == ["'a\\'b\nc'"]

    def test_non_str_input_is_converted(self):
        class FakeStmt:
            def __str__(self):
                return "a\nb"

        assert split_unquoted_newlines(FakeStmt()) == ["a", "b"]

    def test_multiple_lines_with_strings(self):
        text = "SELECT '\n'\nFROM t"
        assert split_unquoted_newlines(text) == ["SELECT '\n'", "FROM t"]


# ---------------------------------------------------------------- remove_quotes

class TestRemoveQuotes:
    def test_none_returns_none(self):
        assert remove_quotes(None) is None

    def test_empty_string_raises_index_error(self):
        # val[0] is accessed unconditionally, so empty string raises
        with pytest.raises(IndexError):
            remove_quotes("")

    def test_no_quotes(self):
        assert remove_quotes("abc") == "abc"

    def test_double_quotes(self):
        assert remove_quotes('"abc"') == "abc"

    def test_single_quotes(self):
        assert remove_quotes("'abc'") == "abc"

    def test_backticks(self):
        assert remove_quotes("`abc`") == "abc"

    def test_unmatched_quotes(self):
        assert remove_quotes('"abc') == '"abc'

    def test_mismatched_quotes(self):
        assert remove_quotes('"abc\'') == '"abc\''

    def test_only_quotes(self):
        assert remove_quotes('""') == ""

    def test_single_char(self):
        # val[0] == val[-1] for single char, so it strips to empty
        assert remove_quotes('"') == ""

    def test_plain_quote_char_in_middle(self):
        assert remove_quotes('a"b') == 'a"b'


# ---------------------------------------------------------------- recurse

class FakeGroup:
    """Minimal token-like object supporting get_sublists."""
    def __init__(self, cls, sublists=None):
        self.cls = cls
        self._sublists = sublists or []
        self.calls = 0

    def get_sublists(self):
        return self._sublists


class Excluded(FakeGroup):
    pass


class TestRecurse:
    def test_recurse_applies_to_sublists(self):
        leaf1 = FakeGroup("Leaf")
        leaf2 = FakeGroup("Leaf")
        parent = FakeGroup("Parent", [leaf1, leaf2])

        seen = []

        @recurse()
        def f(tlist):
            seen.append(tlist)

        f(parent)

        assert seen == [leaf1, leaf2, parent]

    def test_recurse_skips_excluded_class(self):
        # When a sublist is an instance of an excluded class, it is neither
        # recursed into nor passed to f. Only the deepest non-excluded
        # nodes and the top-level object are visited.
        leaf = FakeGroup("Leaf")
        deepest = FakeGroup("Deepest")
        excluded_child = Excluded("Excluded", [deepest])
        parent = FakeGroup("Parent", [leaf, excluded_child])

        seen = []

        @recurse(Excluded)
        def f(tlist):
            seen.append(tlist)

        f(parent)

        # excluded_child is skipped entirely (not recursed, not visited),
        # so neither excluded_child nor deepest appears in seen.
        assert seen == [leaf, parent]

    def test_recurse_nested(self):
        deepest = FakeGroup("Deep")
        mid = FakeGroup("Mid", [deepest])
        top = FakeGroup("Top", [mid])

        seen = []

        @recurse()
        def f(tlist):
            seen.append(tlist)

        f(top)
        assert seen == [deepest, mid, top]

    def test_recurse_returns_decorated_function(self):
        @recurse()
        def f(tlist):
            pass

        assert callable(f)


# ---------------------------------------------------------------- imt

class FakeToken:
    def __init__(self, ttype=None, match_result=False):
        self.ttype = ttype
        self._match_result = match_result

    def match(self, *args):
        return self._match_result


class TType:
    # Simulate enum-like token types; `in` operations on TType iterate its
    # members. We model that by making TType a container with a single
    # member (itself), so `ttype in ttype` -> True and `other in ttype` -> False.
    def __init__(self, name):
        self.name = name

    def __contains__(self, item):
        return item is self

    def __iter__(self):
        yield self

    def __repr__(self):
        return f"TType({self.name})"


class TestImt:
    def test_none_token_returns_false(self):
        assert imt(None, i=int) is False

    def test_instance_match(self):
        token = FakeToken()
        assert imt(token, i=FakeToken) is True

    def test_instance_no_match(self):
        token = FakeToken()
        assert imt(token, i=str) is False

    def test_instance_tuple(self):
        token = FakeToken()
        assert imt(token, i=(str, FakeToken)) is True

    def test_match_true(self):
        token = FakeToken(match_result=True)
        assert imt(token, m=("anything",)) is True

    def test_match_false(self):
        token = FakeToken(match_result=False)
        assert imt(token, m=("anything",)) is False

    def test_match_list_true(self):
        token = FakeToken(match_result=True)
        assert imt(token, m=[("a",), ("b",)]) is True

    def test_match_list_false(self):
        token = FakeToken(match_result=False)
        assert imt(token, m=[("a",), ("b",)]) is False

    def test_ttype_single(self):
        tt = TType("A")
        token = FakeToken(ttype=tt)
        # single-item branch: token.ttype in t -> tt in tt -> True
        assert imt(token, t=tt) is True

    def test_ttype_single_no_match(self):
        token = FakeToken(ttype=TType("A"))
        other = TType("B")
        # token.ttype in other -> False
        assert imt(token, t=other) is False

    def test_ttype_list(self):
        a = TType("A")
        b = TType("B")
        token = FakeToken(ttype=a)
        # list branch: any(token.ttype in ttype for ttype in t)
        assert imt(token, t=[a, b]) is True

    def test_ttype_list_no_match(self):
        a = TType("A")
        b = TType("B")
        token = FakeToken(ttype=TType("C"))
        assert imt(token, t=[a, b]) is False

    def test_no_criteria_returns_false(self):
        token = FakeToken()
        assert imt(token) is False

    def test_instance_precedence_over_match(self):
        token = FakeToken(match_result=False)
        assert imt(token, i=FakeToken, m=("x",)) is True


# ---------------------------------------------------------------- consume

class TestConsume:
    def test_consume_n(self):
        it = iter([1, 2, 3, 4])
        consume(it, 2)
        assert list(it) == [3, 4]

    def test_consume_all_none(self):
        it = iter([1, 2, 3])
        consume(it, None)
        assert list(it) == []

    def test_consume_more_than_available(self):
        it = iter([1, 2])
        consume(it, 10)
        assert list(it) == []

    def test_consume_zero(self):
        it = iter([1, 2, 3])
        consume(it, 0)
        assert list(it) == [1, 2, 3]

    def test_consume_negative(self):
        # islice with negative stop raises ValueError
        it = iter([1, 2, 3])
        with pytest.raises(ValueError):
            consume(it, -1)


# ---------------------------------------------------------------- offset

class FilterStub:
    def __init__(self, offset_val=0):
        self.offset = offset_val


class TestOffset:
    def test_offset_increases_within_context(self):
        f = FilterStub(offset_val=5)
        with offset(f, 3):
            assert f.offset == 8
        assert f.offset == 5

    def test_offset_default_n(self):
        f = FilterStub(offset_val=1)
        with offset(f):
            assert f.offset == 1
        assert f.offset == 1

    def test_offset_not_restored_on_exception(self):
        # the contextmanager has no try/finally, so on exception the
        # decrement line is skipped and the offset stays increased.
        f = FilterStub(offset_val=2)
        with pytest.raises(RuntimeError):
            with offset(f, 4):
                assert f.offset == 6
                raise RuntimeError("boom")
        assert f.offset == 6


# ---------------------------------------------------------------- indent

class TestIndent:
    def test_indent_increases_within_context(self):
        f = FilterStub()
        f.indent = 0
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_indent_custom_n(self):
        f = FilterStub()
        f.indent = 5
        with indent(f, 3):
            assert f.indent == 8
        assert f.indent == 5

    def test_indent_not_restored_on_exception(self):
        # the contextmanager has no try/finally, so on exception the
        # decrement line is skipped and the indent stays increased.
        f = FilterStub()
        f.indent = 1
        with pytest.raises(ValueError):
            with indent(f, 2):
                assert f.indent == 3
                raise ValueError("oops")
        assert f.indent == 3
