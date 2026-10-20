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

def test_split_unquoted_newlines_simple():
    assert split_unquoted_newlines("a\nb\nc") == ["a", "b", "c"]


def test_split_unquoted_newlines_no_newline():
    assert split_unquoted_newlines("abc") == ["abc"]


def test_split_unquoted_newlines_empty():
    assert split_unquoted_newlines("") == [""]


def test_split_unquoted_newlines_crlf():
    assert split_unquoted_newlines("a\r\nb") == ["a", "b"]


def test_split_unquoted_newlines_cr():
    assert split_unquoted_newlines("a\rb") == ["a", "b"]


def test_split_unquoted_newlines_trailing_newline():
    assert split_unquoted_newlines("a\n") == ["a", ""]


def test_split_unquoted_newlines_leading_newline():
    assert split_unquoted_newlines("\na") == ["", "a"]


def test_split_unquoted_newlines_multiple_consecutive():
    assert split_unquoted_newlines("a\n\nb") == ["a", "", "b"]


def test_split_unquoted_newlines_double_quoted():
    assert split_unquoted_newlines('a "x\ny" b') == ['a "x\ny" b']


def test_split_unquoted_newlines_single_quoted():
    assert split_unquoted_newlines("a 'x\ny' b") == ["a 'x\ny' b"]


def test_split_unquoted_newlines_escaped_quote_in_string():
    assert split_unquoted_newlines('a "x\\"\ny" b') == ['a "x\\"\ny" b']


def test_split_unquoted_newlines_mixed():
    result = split_unquoted_newlines("a\n\"b\nc\"\nd")
    assert result == ['a', '"b\nc"', 'd']


def test_split_unquoted_newlines_coerces_to_str():
    class S:
        def __str__(self):
            return "x\ny"
    assert split_unquoted_newlines(S()) == ["x", "y"]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------

def test_remove_quotes_none():
    assert remove_quotes(None) is None


def test_remove_quotes_double():
    assert remove_quotes('"abc"') == "abc"


def test_remove_quotes_single():
    assert remove_quotes("'abc'") == "abc"


def test_remove_quotes_backtick():
    assert remove_quotes("`abc`") == "abc"


def test_remove_quotes_unquoted():
    assert remove_quotes("abc") == "abc"


def test_remove_quotes_mismatched():
    assert remove_quotes('"abc\'') == '"abc\''


def test_remove_quotes_only_opening():
    assert remove_quotes('"') == '"'


def test_remove_quotes_other_char():
    assert remove_quotes("[abc]") == "[abc]"


def test_remove_quotes_empty_string():
    # empty string: accessing val[0] would raise IndexError
    with pytest.raises(IndexError):
        remove_quotes("")


def test_remove_quotes_nested():
    assert remove_quotes('""abc""') == '"abc"'


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class FakeTokenList:
    def __init__(self, sublists=None):
        self._sublists = sublists or []

    def get_sublists(self):
        return iter(self._sublists)


class FakeIdentifier(FakeTokenList):
    pass


class FakeOther(FakeTokenList):
    pass


def test_recurse_visits_all():
    leaf1 = FakeOther()
    leaf2 = FakeOther()
    mid = FakeOther([leaf1, leaf2])
    root = FakeOther([mid])

    visited = []

    @recurse()
    def visit(tlist):
        visited.append(tlist)

    visit(root)
    # post-order: leaves first, then mid, then root
    assert visited == [leaf1, leaf2, mid, root]


def test_recurse_skips_given_class():
    leaf = FakeIdentifier()
    other = FakeOther([leaf])
    root = FakeOther([other])

    visited = []

    @recurse(FakeIdentifier)
    def visit(tlist):
        visited.append(tlist)

    visit(root)
    # FakeIdentifier is skipped entirely (not visited), FakeOther node still visited
    assert leaf not in visited
    assert other in visited
    assert root in visited


def test_recurse_returns_wrapped_function():
    @recurse()
    def visit(tlist):
        return None

    assert callable(visit)


def test_recurse_no_sublists():
    root = FakeOther()
    visited = []

    @recurse()
    def visit(tlist):
        visited.append(tlist)

    visit(root)
    assert visited == [root]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype=None, value="", matches=None):
        self.ttype = ttype
        self.value = value
        self._matches = matches or []

    def match(self, *args):
        return self._matches == list(args)


def test_imt_none_token():
    assert imt(None) is False


def test_imt_i_match():
    assert imt(FakeToken(), i=FakeToken) is True


def test_imt_i_no_match():
    assert imt(FakeToken(), i=str) is False


def test_imt_m_tuple_match():
    tok = FakeToken(matches=[1, 2])
    assert imt(tok, m=(1, 2)) is True


def test_imt_m_tuple_no_match():
    tok = FakeToken(matches=[1, 2])
    assert imt(tok, m=(1, 3)) is False


def test_imt_m_list_match():
    tok = FakeToken(matches=[1, 2])
    assert imt(tok, m=[(1, 3), (1, 2)]) is True


def test_imt_m_list_no_match():
    tok = FakeToken(matches=[1, 2])
    assert imt(tok, m=[(1, 3), (1, 4)]) is False


def test_imt_t_direct_match():
    class T:
        pass
    ttype = T()
    tok = FakeToken(ttype=ttype)
    assert imt(tok, t=ttype) is True


def test_imt_t_direct_no_match():
    class T:
        pass
    tok = FakeToken(ttype=T())
    assert imt(tok, t=T()) is False


def test_imt_t_list_match():
    class T:
        pass
    t1, t2 = T(), T()
    tok = FakeToken(ttype=t2)
    assert imt(tok, t=[t1, t2]) is True


def test_imt_t_list_no_match():
    class T:
        pass
    tok = FakeToken(ttype=T())
    assert imt(tok, t=[T(), T()]) is False


def test_imt_nothing_given():
    assert imt(FakeToken()) is False


def test_imt_multiple_conditions_or():
    tok = FakeToken(ttype=None)
    assert imt(tok, i=FakeToken) is True


# Test the ttype-in-tuple behavior used for ttype membership. Since ttype is
# compared via ``token.ttype in t``, make ttype a special object that supports
# membership checks.
class InSet:
    def __init__(self, value):
        self.value = value

    def __contains__(self, item):
        return item == self.value


def test_imt_t_single_contains():
    class T:
        pass
    ttype = T()
    marker = object()
    tok = FakeToken(ttype=ttype)
    # token.ttype in t -> uses t.__contains__
    container = InSet(marker)
    assert imt(tok, t=container) is (ttype == marker)


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

def test_consume_partial():
    it = iter([1, 2, 3, 4])
    consume(it, 2)
    assert list(it) == [3, 4]


def test_consume_all_with_none():
    it = iter([1, 2, 3])
    consume(it, None)
    assert list(it) == []


def test_consume_zero():
    it = iter([1, 2, 3])
    consume(it, 0)
    assert list(it) == [1, 2, 3]


def test_consume_more_than_available():
    it = iter([1, 2])
    consume(it, 10)
    assert list(it) == []


def test_consume_generic_iterable():
    lst = [1, 2, 3, 4, 5]
    consume(iter(lst), 3)
    # original list untouched, just consuming the iterator
    assert lst == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# offset / indent context managers
# ---------------------------------------------------------------------------

class FakeFilter:
    def __init__(self):
        self.offset = 0
        self.indent = 0


def test_offset_default():
    f = FakeFilter()
    with offset(f):
        assert f.offset == 0
    assert f.offset == 0


def test_offset_increments_within():
    f = FakeFilter()
    with offset(f, 5):
        assert f.offset == 5
    assert f.offset == 0


def test_offset_restores_on_exception():
    f = FakeFilter()
    with pytest.raises(RuntimeError):
        with offset(f, 3):
            assert f.offset == 3
            raise RuntimeError("boom")
    assert f.offset == 0


def test_offset_relative_to_existing():
    f = FakeFilter()
    f.offset = 10
    with offset(f, 2):
        assert f.offset == 12
    assert f.offset == 10


def test_indent_default():
    f = FakeFilter()
    with indent(f):
        assert f.indent == 1
    assert f.indent == 0


def test_indent_custom():
    f = FakeFilter()
    with indent(f, 4):
        assert f.indent == 4
    assert f.indent == 0


def test_indent_nested():
    f = FakeFilter()
    with indent(f):
        assert f.indent == 1
        with indent(f):
            assert f.indent == 2
        assert f.indent == 1
    assert f.indent == 0


def test_indent_restores_on_exception():
    f = FakeFilter()
    with pytest.raises(ValueError):
        with indent(f, 2):
            assert f.indent == 2
            raise ValueError("bad")
    assert f.indent == 0


def test_offset_nested():
    f = FakeFilter()
    with offset(f, 1):
        assert f.offset == 1
        with offset(f, 2):
            assert f.offset == 3
        assert f.offset == 1
    assert f.offset == 0
