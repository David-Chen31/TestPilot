#
# Tests for sqlparse/utils.py
#
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
    def test_empty_string(self):
        assert split_unquoted_newlines("") == [""]

    def test_single_line(self):
        assert split_unquoted_newlines("SELECT 1") == ["SELECT 1"]

    def test_simple_newline(self):
        assert split_unquoted_newlines("a\nb") == ["a", "b"]

    def test_crlf(self):
        assert split_unquoted_newlines("a\r\nb") == ["a", "b"]

    def test_cr(self):
        assert split_unquoted_newlines("a\rb") == ["a", "b"]

    def test_multiple_lines(self):
        assert split_unquoted_newlines("a\nb\nc") == ["a", "b", "c"]

    def test_trailing_newline(self):
        assert split_unquoted_newlines("a\n") == ["a", ""]

    def test_leading_newline(self):
        assert split_unquoted_newlines("\na") == ["", "a"]

    def test_double_newline(self):
        assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]

    def test_newline_inside_single_quotes(self):
        assert split_unquoted_newlines("'a\nb'") == ["'a\nb'"]

    def test_newline_inside_double_quotes(self):
        assert split_unquoted_newlines('"a\nb"') == ['"a\nb"']

    def test_mixed_quoted_and_newlines(self):
        stmt = "'a\nb'\nc"
        assert split_unquoted_newlines(stmt) == ["'a\nb'", "c"]

    def test_escaped_quote_inside_string(self):
        stmt = r"'a\'b\nc'" + "\nd"
        result = split_unquoted_newlines(stmt)
        assert result == [r"'a\'b\nc'", "d"]

    def test_non_string_input_converted(self):
        # convert via str()
        result = split_unquoted_newlines(123)
        assert result == ["123"]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------
class TestRemoveQuotes:
    def test_none_returns_none(self):
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

    def test_only_opening_quote(self):
        assert remove_quotes('"abc') == '"abc'

    def test_single_char_double_quote(self):
        # val[0] == val[-1] and in the set; slicing "" content
        assert remove_quotes('"') == ""

    def test_empty_string(self):
        # val[0] would raise IndexError on empty string
        with pytest.raises(IndexError):
            remove_quotes("")

    def test_quotes_only_pair(self):
        assert remove_quotes("''") == ""

    def test_nested_quotes(self):
        assert remove_quotes('"\'abc\'"') == "'abc'"


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------
class DummyGroup:
    def __init__(self, sublists=None):
        self._sublists = sublists if sublists is not None else []

    def get_sublists(self):
        return iter(self._sublists)


class TestRecurse:
    def test_calls_on_leaf(self):
        calls = []

        @recurse()
        def visit(group):
            calls.append(group)

        leaf = DummyGroup()
        visit(leaf)
        assert calls == [leaf]

    def test_recurses_into_children(self):
        child1 = DummyGroup()
        child2 = DummyGroup()
        parent = DummyGroup([child1, child2])
        calls = []

        @recurse()
        def visit(group):
            calls.append(group)

        visit(parent)
        # children are visited first (post-order), then the parent
        assert calls == [child1, child2, parent]

    def test_skips_excluded_type(self):
        class Excluded(DummyGroup):
            pass

        excluded = Excluded()
        child = DummyGroup()
        parent = DummyGroup([excluded, child])
        calls = []

        @recurse(Excluded)
        def visit(group):
            calls.append(group)

        visit(parent)
        # excluded is not recursed into but still... it is not passed to f
        # because only get_sublists of parent recurses into each; excluded
        # itself has no sublists. The excluded group is NOT passed to f
        # because f is only called on the top tlist, not on sublists.
        # So calls should just contain parent (excluded has no children).
        assert parent in calls
        assert excluded not in calls
        assert child in calls

    def test_excluded_not_recursed_children_skipped(self):
        class Excluded(DummyGroup):
            pass

        grandchild = DummyGroup()
        excluded = Excluded([grandchild])
        parent = DummyGroup([excluded])
        calls = []

        @recurse(Excluded)
        def visit(group):
            calls.append(group)

        visit(parent)
        # grandchild should NOT be visited because excluded is not recursed
        assert calls == [parent]
        assert grandchild not in calls


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------
class FakeToken:
    def __init__(self, ttype, value, match_result=None):
        self.ttype = ttype
        self.value = value
        self._match_result = match_result

    def match(self, *args):
        if self._match_result is not None:
            return self._match_result
        return False


class TestImt:
    def test_none_token_returns_false(self):
        assert imt(None) is False

    def test_no_criteria_returns_false(self):
        tok = FakeToken("x", "val")
        assert imt(tok) is False

    def test_instance_match(self):
        tok = FakeToken("x", "val")
        assert imt(tok, i=FakeToken) is True

    def test_instance_tuple_match(self):
        tok = FakeToken("x", "val")
        assert imt(tok, i=(int, FakeToken)) is True

    def test_instance_no_match(self):
        tok = FakeToken("x", "val")
        assert imt(tok, i=int) is False

    def test_match_single(self):
        tok = FakeToken("x", "val", match_result=True)
        assert imt(tok, m=("a", "b")) is True

    def test_match_single_false(self):
        tok = FakeToken("x", "val", match_result=False)
        assert imt(tok, m=("a", "b")) is False

    def test_match_list_true(self):
        tok = FakeToken("x", "val", match_result=True)
        assert imt(tok, m=[("a", "b")]) is True

    def test_match_list_false(self):
        tok = FakeToken("x", "val", match_result=False)
        assert imt(tok, m=[("a", "b")]) is False

    def test_ttype_single_match(self):
        tok = FakeToken("tokentype1", "val")
        assert imt(tok, t="tokentype1") is True

    def test_ttype_single_no_match(self):
        tok = FakeToken("tokentype1", "val")
        assert imt(tok, t="tokentype2") is False

    def test_ttype_list_match(self):
        tok = FakeToken("tokentype2", "val")
        assert imt(tok, t=["tokentype1", "tokentype2"]) is True

    def test_ttype_tuple_tuple_match(self):
        # a tuple of tuples (in-style)
        tok = FakeToken("ttype1", "val")
        # token.ttype in ttype for ttype in t -> "ttype1" in ("ttype1",)
        assert imt(tok, t=[("ttype1",), ("ttype2",)]) is True

    def test_ttype_tuple_tuple_no_match(self):
        tok = FakeToken("other", "val")
        assert imt(tok, t=[("ttype1",), ("ttype2",)]) is False

    def test_combined_instance_hit(self):
        tok = FakeToken("x", "val", match_result=False)
        assert imt(tok, i=FakeToken, m=("a", "b")) is True

    def test_combined_match_hit(self):
        tok = FakeToken("x", "val", match_result=True)
        assert imt(tok, m=("a", "b"), t="nope") is True

    def test_combined_ttype_hit(self):
        tok = FakeToken("ttype", "val", match_result=False)
        assert imt(tok, i=int, m=("a", "b"), t="ttype") is True


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------
class TestConsume:
    def test_consume_partial(self):
        it = iter([1, 2, 3, 4, 5])
        consume(it, 2)
        assert list(it) == [3, 4, 5]

    def test_consume_none_consumes_all(self):
        it = iter([1, 2, 3])
        consume(it, None)
        assert list(it) == []

    def test_consume_zero(self):
        it = iter([1, 2, 3])
        consume(it, 0)
        assert list(it) == [1, 2, 3]

    def test_consume_more_than_length(self):
        it = iter([1, 2])
        consume(it, 10)
        assert list(it) == []

    def test_consume_on_generator(self):
        def gen():
            for i in range(5):
                yield i

        g = gen()
        consume(g, 3)
        assert list(g) == [3, 4]


# ---------------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------------
class FakeFilter:
    def __init__(self, offset=0, indent=0):
        self.offset = offset
        self.indent = indent


class TestOffset:
    def test_default(self):
        f = FakeFilter()
        with offset(f):
            assert f.offset == 0

    def test_positive(self):
        f = FakeFilter()
        with offset(f, 5):
            assert f.offset == 5
        assert f.offset == 0

    def test_negative(self):
        f = FakeFilter(offset=10)
        with offset(f, -3):
            assert f.offset == 7
        assert f.offset == 10

    def test_restores_on_exception(self):
        f = FakeFilter(offset=2)
        with pytest.raises(ValueError):
            with offset(f, 4):
                assert f.offset == 6
                raise ValueError("boom")
        assert f.offset == 2


# ---------------------------------------------------------------------------
# indent
# ---------------------------------------------------------------------------
class TestIndent:
    def test_default(self):
        f = FakeFilter()
        with indent(f):
            assert f.indent == 1
        assert f.indent == 0

    def test_custom_n(self):
        f = FakeFilter()
        with indent(f, 3):
            assert f.indent == 3
        assert f.indent == 0

    def test_negative(self):
        f = FakeFilter(indent=5)
        with indent(f, -2):
            assert f.indent == 3
        assert f.indent == 5

    def test_restores_on_exception(self):
        f = FakeFilter(indent=1)
        with pytest.raises(RuntimeError):
            with indent(f, 2):
                assert f.indent == 3
                raise RuntimeError("boom")
        assert f.indent == 1
