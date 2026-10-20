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

def test_split_unquoted_newlines_plain():
    assert split_unquoted_newlines("a\nb") == ["a", "b"]


def test_split_unquoted_newlines_empty():
    # No content results in a single empty string element.
    assert split_unquoted_newlines("") == [""]


def test_split_unquoted_newlines_no_newline():
    assert split_unquoted_newlines("select 1") == ["select 1"]


def test_split_unquoted_newlines_crlf():
    assert split_unquoted_newlines("a\r\nb") == ["a", "b"]


def test_split_unquoted_newlines_cr():
    assert split_unquoted_newlines("a\rb") == ["a", "b"]


def test_split_unquoted_newlines_multiple():
    assert split_unquoted_newlines("a\nb\r\nc\rd") == ["a", "b", "c", "d"]


def test_split_unquoted_newlines_trailing_newline():
    # Behavior of the implementation: trailing newline adds an empty line.
    assert split_unquoted_newlines("a\n") == ["a", ""]


def test_split_unquoted_newlines_leading_newline():
    assert split_unquoted_newlines("\na") == ["", "a"]


def test_split_unquoted_newlines_ignores_single_quoted():
    assert split_unquoted_newlines("'a\nb'") == ["'a\nb'"]


def test_split_unquoted_newlines_ignores_double_quoted():
    assert split_unquoted_newlines('"a\nb"') == ['"a\nb"']


def test_split_unquoted_newlines_quoted_then_newline():
    assert split_unquoted_newlines("'a\nb'\nc") == ["'a\nb'", "c"]


def test_split_unquoted_newlines_escaped_quote():
    # Escaped quote inside a single-quoted string should not terminate it.
    assert split_unquoted_newlines(r"'a\'b'c") == ["'a\\'b'c"]


def test_split_unquoted_newlines_accepts_non_str():
    # str() is applied to the input.
    assert split_unquoted_newlines(123) == ["123"]


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


def test_remove_quotes_no_quotes():
    assert remove_quotes("abc") == "abc"


def test_remove_quotes_mismatched_quotes():
    assert remove_quotes("'abc\"") == "'abc\""


def test_remove_quotes_only_opening():
    # Single char: val[0] == val[-1], so it is stripped -> empty string.
    assert remove_quotes("'") == ""


def test_remove_quotes_empty_string():
    # Current behavior: indexing into an empty string raises IndexError.
    with pytest.raises(IndexError):
        remove_quotes("")


def test_remove_quotes_only_partial():
    assert remove_quotes("'ab") == "'ab"


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class DummyNode:
    def __init__(self, sublists=None):
        self._sublists = sublists or []

    def get_sublists(self):
        return self._sublists


def test_recurse_visits_children_before_parent():
    leaf = DummyNode()
    mid = DummyNode([leaf])
    root = DummyNode([mid])

    visited = []
    recurse()(lambda tlist: visited.append(tlist))(root)

    assert visited == [leaf, mid, root]


def test_recurse_skips_instances_of_given_class():
    skipped = DummyNode()
    other = DummyNode()
    root = DummyNode([skipped, other])

    visited = []
    recurse(DummyNode)(lambda tlist: visited.append(tlist))(root)

    # Both children are instances of DummyNode -> skipped, root still visited.
    assert visited == [root]


def test_recurse_skips_only_matching_type():
    class Special(DummyNode):
        pass

    special = Special()
    normal = DummyNode()
    root = DummyNode([special, normal])

    visited = []
    recurse(Special)(lambda tlist: visited.append(tlist))(root)

    assert visited == [normal, root]


def test_recurse_multiple_classes():
    class SpecialA(DummyNode):
        pass

    class SpecialB(DummyNode):
        pass

    a = SpecialA()
    b = SpecialB()
    n = DummyNode()
    root = DummyNode([a, b, n])

    visited = []
    recurse(SpecialA, SpecialB)(lambda tlist: visited.append(tlist))(root)

    assert visited == [n, root]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

class FakeToken:
    def __init__(self, ttype=None, value=None):
        self.ttype = ttype
        self.value = value

    def match(self, ttype, value):
        return self.ttype == ttype and self.value == value


def test_imt_none_token():
    assert imt(None) is False


def test_imt_instance_check():
    class T:
        pass

    token = T()
    assert imt(token, i=T) is True
    assert imt(token, i=(T,)) is True
    assert imt(token, i=int) is False


def test_imt_no_criteria():
    assert imt(FakeToken()) is False


def test_imt_match_single():
    token = FakeToken(ttype="a", value="b")
    assert imt(token, m=("a", "b")) is True
    assert imt(token, m=("a", "c")) is False


def test_imt_match_list():
    token = FakeToken(ttype="a", value="b")
    assert imt(token, m=[("a", "b")]) is True
    assert imt(token, m=[("a", "x"), ("a", "b")]) is True
    assert imt(token, m=[("a", "x"), ("a", "y")]) is False


def test_imt_ttype_single():
    token = FakeToken(ttype="typeA")
    assert imt(token, t="typeA") is True
    assert imt(token, t="typeB") is False


def test_imt_ttype_list():
    token = FakeToken(ttype="typeA")
    assert imt(token, t=["typeA", "typeB"]) is True
    assert imt(token, t=["typeB", "typeC"]) is False


def test_imt_combined():
    token = FakeToken(ttype="a", value="b")
    # instance matches short-circuits
    assert imt(token, i=FakeToken, t="wrong") is True
    # match path
    assert imt(token, m=("a", "b"), t="wrong") is True
    # ttype path
    assert imt(token, m=("x", "y"), t="a") is True


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

def test_consume_entire_when_none():
    it = iter([1, 2, 3, 4])
    consume(it, None)
    assert list(it) == []


def test_consume_partial():
    it = iter([1, 2, 3, 4])
    consume(it, 2)
    assert list(it) == [3, 4]


def test_consume_more_than_available():
    it = iter([1, 2])
    consume(it, 5)
    assert list(it) == []


def test_consume_zero():
    it = iter([1, 2, 3])
    consume(it, 0)
    assert list(it) == [1, 2, 3]


def test_consume_full_consumption_returns_none():
    # The function returns None; its only effect is on the iterator.
    it = iter([1, 2, 3])
    result = consume(it, None)
    assert result is None
    assert list(it) == []


def test_consume_partial_returns_none():
    it = iter([1, 2, 3, 4, 5])
    result = consume(it, 2)
    assert result is None
    assert list(it) == [3, 4, 5]


def test_consume_consumes_exactly_n():
    # Ensures consumption stops exactly after n items: with a deque of
    # maxlen != 0 or a non-fully-consuming variant, the number of consumed
    # items would differ, which is detectable by counting pulls from a
    # generator that records how many times it was advanced.
    pulled = []

    def gen():
        for i in range(10):
            pulled.append(i)
            yield i

    it = gen()
    consume(it, 3)
    # Exactly 3 items pulled from the generator (islice pulls n items).
    assert pulled == [0, 1, 2]
    # The remaining items are still available.
    assert list(it) == [3, 4, 5, 6, 7, 8, 9]


def test_consume_none_pulls_everything():
    pulled = []

    def gen():
        for i in range(5):
            pulled.append(i)
            yield i

    it = gen()
    consume(it, None)
    assert pulled == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# offset context manager
# ---------------------------------------------------------------------------

class FakeFilter:
    def __init__(self, offset_value=0, indent_value=0):
        self.offset = offset_value
        self.indent = indent_value


def test_offset_increments_and_restores():
    f = FakeFilter(offset_value=5)
    with offset(f, 3):
        assert f.offset == 8
    assert f.offset == 5


def test_offset_default_zero():
    f = FakeFilter(offset_value=7)
    with offset(f):
        assert f.offset == 7
    assert f.offset == 7


def test_offset_negative():
    f = FakeFilter(offset_value=10)
    with offset(f, -4):
        assert f.offset == 6
    assert f.offset == 10


def test_offset_restores_on_exception():
    # Current behavior: the generator-based context manager does not restore
    # state when an exception propagates out of the body (no try/finally).
    f = FakeFilter(offset_value=1)
    with pytest.raises(ValueError):
        with offset(f, 2):
            assert f.offset == 3
            raise ValueError("boom")
    assert f.offset == 3


# ---------------------------------------------------------------------------
# indent context manager
# ---------------------------------------------------------------------------

def test_indent_increments_and_restores():
    f = FakeFilter(indent_value=2)
    with indent(f):
        assert f.indent == 3
    assert f.indent == 2


def test_indent_custom_n():
    f = FakeFilter(indent_value=0)
    with indent(f, 5):
        assert f.indent == 5
    assert f.indent == 0


def test_indent_negative():
    f = FakeFilter(indent_value=4)
    with indent(f, -2):
        assert f.indent == 2
    assert f.indent == 4


def test_indent_restores_on_exception():
    # Current behavior: the generator-based context manager does not restore
    # state when an exception propagates out of the body (no try/finally).
    f = FakeFilter(indent_value=0)
    with pytest.raises(RuntimeError):
        with indent(f, 3):
            assert f.indent == 3
            raise RuntimeError("boom")
    assert f.indent == 3
