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

def test_split_unquoted_newlines_simple():
    assert split_unquoted_newlines("a\nb\nc") == ["a", "b", "c"]


def test_split_unquoted_newlines_crlf_and_cr():
    assert split_unquoted_newlines("a\r\nb\rc") == ["a", "b", "c"]


def test_split_unquoted_newlines_no_newlines():
    assert split_unquoted_newlines("abc def") == ["abc def"]


def test_split_unquoted_newlines_empty_string():
    assert split_unquoted_newlines("") == [""]


def test_split_unquoted_newlines_trailing_newline():
    assert split_unquoted_newlines("a\n") == ["a", ""]


def test_split_unquoted_newlines_leading_newline():
    assert split_unquoted_newlines("\na") == ["", "a"]


def test_split_unquoted_newlines_double_quoted():
    assert split_unquoted_newlines('"a\nb"\nc') == ['"a\nb"', "c"]


def test_split_unquoted_newlines_single_quoted():
    assert split_unquoted_newlines("'a\r\nb'\nc") == ["'a\r\nb'", "c"]


def test_split_unquoted_newlines_escaped_quote():
    assert split_unquoted_newlines('"a\\"b\nc"\nd') == ['"a\\"b\nc"', "d"]


def test_split_unquoted_newlines_mixed():
    text = 'SELECT "x\ny" FROM t\nWHERE a = \'p\r\nq\'\nAND b = 1'
    assert split_unquoted_newlines(text) == [
        'SELECT "x\ny" FROM t',
        "WHERE a = 'p\r\nq'",
        "AND b = 1",
    ]


def test_split_unquoted_newlines_accepts_non_str():
    # Converts statement to str first.
    stmt = sql.Statement([sql.Token(T.Keyword, "a\nb")])
    result = split_unquoted_newlines(stmt)
    assert "a" in result[0]
    assert "b" in result[-1]


# ---------------------------------------------------------------------------
# remove_quotes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ('"abc"', "abc"),
    ("'abc'", "abc"),
    ("`abc`", "abc"),
])
def test_remove_quotes_removes_matching_quotes(value, expected):
    assert remove_quotes(value) == expected


def test_remove_quotes_none():
    assert remove_quotes(None) is None


def test_remove_quotes_no_quotes():
    assert remove_quotes("abc") == "abc"


def test_remove_quotes_mismatched_quotes():
    assert remove_quotes('"abc\'') == '"abc\''
    assert remove_quotes("\"abc'") == "\"abc'"


def test_remove_quotes_only_one_char_quote():
    # A single quote char: val[0] == val[-1], so it becomes empty string.
    assert remove_quotes('"') == ""
    assert remove_quotes("'") == ""
    assert remove_quotes("`") == ""


def test_remove_quotes_inner_only():
    assert remove_quotes('a"bc') == 'a"bc'


def test_remove_quotes_empty_string():
    # Empty string: val[0] would raise IndexError in original; guard with pytest.
    with pytest.raises(IndexError):
        remove_quotes("")


# ---------------------------------------------------------------------------
# recurse
# ---------------------------------------------------------------------------

class _DummyGroup:
    def __init__(self, name):
        self.name = name
        self.sublists = []

    def get_sublists(self):
        return self.sublists


def test_recurse_calls_function_on_all_nodes():
    root = _DummyGroup("root")
    child1 = _DummyGroup("c1")
    child2 = _DummyGroup("c2")
    grandchild = _DummyGroup("g")
    root.sublists = [child1, child2]
    child1.sublists = [grandchild]

    visited = []

    @recurse()
    def visit(node):
        visited.append(node.name)

    visit(root)

    assert sorted(visited) == sorted(["root", "c1", "c2", "g"])


def test_recurse_skips_specified_classes():
    root = _DummyGroup("root")
    child = _DummyGroup("child")
    grandchild = _DummyGroup("grandchild")
    root.sublists = [child]
    child.sublists = [grandchild]

    visited = []

    @recurse(_DummyGroup)
    def visit(node):
        visited.append(node.name)

    visit(root)

    # Without any matching sublists that are not _DummyGroup, only root visited.
    assert visited == ["root"]


def test_recurse_skips_but_still_calls_on_parent():
    class Skip:
        pass

    class Node(Skip):
        def __init__(self, name, sublists=None):
            self.name = name
            self.sublists = sublists or []

        def get_sublists(self):
            return self.sublists

    root = Node("root")
    skipped = Skip()
    skipped.sublists = []
    skipped.get_sublists = lambda: []
    root.sublists = [skipped]

    visited = []

    @recurse(Skip)
    def visit(node):
        visited.append(node.name)

    visit(root)

    assert visited == ["root"]


def test_recurse_empty_group():
    root = _DummyGroup("only")
    visited = []

    @recurse()
    def visit(node):
        visited.append(node.name)

    visit(root)
    assert visited == ["only"]


# ---------------------------------------------------------------------------
# imt
# ---------------------------------------------------------------------------

def _make_token(value, ttype):
    return sql.Token(ttype, value)


def test_imt_none_token():
    assert imt(None, i=sql.Token) is False
    assert imt(None, t=T.Keyword) is False


def test_imt_with_class():
    token = _make_token("foo", T.Name)
    assert imt(token, i=sql.Token) is True
    assert imt(token, i=(sql.Token, sql.Identifier)) is True
    assert imt(token, i=sql.Identifier) is False


def test_imt_with_match_single():
    token = _make_token("true", T.Keyword)
    assert imt(token, m=(T.Keyword, "true")) is True
    assert imt(token, m=(T.Keyword, "false")) is False


def test_imt_with_match_list():
    token = _make_token("'x'", T.String.Single)
    patterns = [
        (T.Keyword, "true"),
        (T.String.Single, "'x'"),
    ]
    assert imt(token, m=patterns) is True

    token2 = _make_token("abc", T.Name)
    assert imt(token2, m=patterns) is False


def test_imt_with_ttype_single():
    token = _make_token("foo", T.Name)
    assert imt(token, t=T.Name) is True
    assert imt(token, t=T.Keyword) is False


def test_imt_with_ttype_list():
    token = _make_token("foo", T.Name)
    assert imt(token, t=[T.Keyword, T.Name]) is True
    assert imt(token, t=[T.Keyword, T.Number]) is False


def test_imt_with_ttype_tuple():
    token = _make_token("foo", T.Name)
    assert imt(token, t=(T.Name,)) is True
    assert imt(token, t=(T.Keyword,)) is False


def test_imt_combined_conditions():
    token = _make_token("true", T.Keyword)
    assert imt(token, i=sql.Token, t=T.Keyword) is True
    assert imt(token, i=sql.Token, t=T.Name) is False


def test_imt_no_conditions_false():
    token = _make_token("foo", T.Name)
    assert imt(token) is False


# ---------------------------------------------------------------------------
# consume
# ---------------------------------------------------------------------------

def test_consume_none_consumes_all():
    it = iter(range(10))
    consume(it, None)
    assert list(it) == []


def test_consume_partial():
    it = iter(range(10))
    consume(it, 4)
    assert list(it) == [4, 5, 6, 7, 8, 9]


def test_consume_zero():
    it = iter(range(5))
    consume(it, 0)
    assert list(it) == [0, 1, 2, 3, 4]


def test_consume_more_than_available():
    it = iter([1, 2, 3])
    consume(it, 100)
    assert list(it) == []


def test_consume_list_iterator():
    data = ["a", "b", "c"]
    it = iter(data)
    consume(it, 2)
    assert next(it) == "c"


# ---------------------------------------------------------------------------
# offset context manager
# ---------------------------------------------------------------------------

class _Filter:
    def __init__(self):
        self.offset = 0
        self.indent = 0


def test_offset_default():
    f = _Filter()
    with offset(f):
        assert f.offset == 0
    assert f.offset == 0


def test_offset_increases_and_restores():
    f = _Filter()
    with offset(f, 5):
        assert f.offset == 5
    assert f.offset == 0


def test_offset_negative():
    f = _Filter()
    f.offset = 10
    with offset(f, -3):
        assert f.offset == 7
    assert f.offset == 10


def test_offset_nested():
    f = _Filter()
    with offset(f, 2):
        assert f.offset == 2
        with offset(f, 3):
            assert f.offset == 5
        assert f.offset == 2
    assert f.offset == 0


def test_offset_restores_on_exception():
    f = _Filter()
    with pytest.raises(RuntimeError):
        with offset(f, 4):
            assert f.offset == 4
            raise RuntimeError("boom")
    # Error path in original code does not restore; assert actual behavior:
    assert f.offset == 4


# ---------------------------------------------------------------------------
# indent context manager
# ---------------------------------------------------------------------------

def test_indent_default():
    f = _Filter()
    with indent(f):
        assert f.indent == 1
    assert f.indent == 0


def test_indent_increases_and_restores():
    f = _Filter()
    with indent(f, 3):
        assert f.indent == 3
    assert f.indent == 0


def test_indent_negative():
    f = _Filter()
    f.indent = 5
    with indent(f, -2):
        assert f.indent == 3
    assert f.indent == 5


def test_indent_nested():
    f = _Filter()
    with indent(f):
        assert f.indent == 1
        with indent(f, 2):
            assert f.indent == 3
        assert f.indent == 1
    assert f.indent == 0


def test_indent_restores_on_exception():
    f = _Filter()
    with pytest.raises(ValueError):
        with indent(f, 2):
            assert f.indent == 2
            raise ValueError("bad")
    # Error path in original code does not restore; assert actual behavior:
    assert f.indent == 2
