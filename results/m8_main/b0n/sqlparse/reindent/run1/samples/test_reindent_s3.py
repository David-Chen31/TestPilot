import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter
from sqlparse.utils import indent, offset


def parse(stmt_str):
    from sqlparse.engine import FilterStack
    from sqlparse.lexer import Lexer
    from sqlparse.parser import Parser
    stack = FilterStack()
    return stack.run(stmt_str, None)


# ---------------------------------------------------------------------------
# Construction / attributes
# ---------------------------------------------------------------------------

def test_default_attributes():
    f = ReindentFilter()
    assert f.n == '\n'
    assert f.width == 2
    assert f.char == ' '
    assert f.indent == 0
    assert f.offset == 0
    assert f.wrap_after == 0
    assert f.comma_first is False
    assert f.indent_columns is False
    assert f.compact is False
    assert f._curr_stmt is None
    assert f._last_stmt is None
    assert f._last_func is None


def test_custom_attributes():
    f = ReindentFilter(width=4, char='\t', wrap_after=10, n='\r\n',
                       comma_first=True, indent_after_first=True,
                       indent_columns=True, compact=True)
    assert f.width == 4
    assert f.char == '\t'
    assert f.wrap_after == 10
    assert f.n == '\r\n'
    assert f.comma_first is True
    assert f.indent == 1
    assert f.indent_columns is True
    assert f.compact is True


# ---------------------------------------------------------------------------
# leading_ws and nl
# ---------------------------------------------------------------------------

def test_leading_ws_default():
    f = ReindentFilter()
    assert f.leading_ws == 0


def test_leading_ws_with_indent():
    f = ReindentFilter(width=3, indent_after_first=True)
    assert f.indent == 1
    assert f.leading_ws == 3


def test_leading_ws_with_offset():
    f = ReindentFilter(width=2)
    with offset(f, 4):
        assert f.offset == 4
        assert f.leading_ws == 4
    assert f.offset == 0


def test_nl_default():
    f = ReindentFilter()
    tok = f.nl()
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n'


def test_nl_with_offset():
    f = ReindentFilter(width=2)
    tok = f.nl(offset=4)
    assert tok.value == '\n' + ' ' * 4


def test_nl_negative_offset_clamped():
    f = ReindentFilter(width=2)
    tok = f.nl(offset=-10)
    assert tok.value == '\n'


def test_nl_custom_char_and_n():
    f = ReindentFilter(width=3, char='.', n='|')
    tok = f.nl()
    assert tok.value == '|'


def test_nl_respects_indent():
    f = ReindentFilter(width=2, indent_after_first=True)
    tok = f.nl()
    assert tok.value == '\n  '


# ---------------------------------------------------------------------------
# _current_line_len / _get_offset
# ---------------------------------------------------------------------------

def _make_simple_group():
    """Build a small TokenList: [a, ws, b] and return (tlist, a, ws, b)."""
    a = sql.Token(T.Name, 'aa')
    ws = sql.Token(T.Whitespace, ' ')
    b = sql.Token(T.Name, 'bb')
    tl = sql.TokenList([a, ws, b])
    return tl, a, ws, b


def test_current_line_len_first_token():
    tl, a, ws, b = _make_simple_group()
    f = ReindentFilter()
    f._curr_stmt = tl
    # First token: nothing before it on the line.
    assert f._current_line_len(a) == 0


def test_current_line_len_middle_token():
    tl, a, ws, b = _make_simple_group()
    f = ReindentFilter()
    f._curr_stmt = tl
    assert f._current_line_len(ws) == 2
    assert f._current_line_len(b) == 3


def test_current_line_len_stops_at_newline():
    a = sql.Token(T.Name, 'aa')
    nl = sql.Token(T.Whitespace, '\n  ')
    b = sql.Token(T.Name, 'bb')
    tl = sql.TokenList([a, nl, b])
    f = ReindentFilter()
    f._curr_stmt = tl
    # After the newline, only the trailing indentation (2 chars) + "bb".
    # For b, current line is "\n  " then b -> tail is len("  ") == 2.
    assert f._current_line_len(b) == 2


def test_current_line_len_trailing_newline():
    a = sql.Token(T.Name, 'aa')
    nl = sql.Token(T.Whitespace, 'xx\n')
    b = sql.Token(T.Name, 'bb')
    tl = sql.TokenList([a, nl, b])
    f = ReindentFilter()
    f._curr_stmt = tl
    # For b, the token before is "xx\n": line before is "xx".
    assert f._current_line_len(b) == 2


def test_get_offset_simple():
    tl, a, ws, b = _make_simple_group()
    f = ReindentFilter(width=2)
    f._curr_stmt = tl
    # b at position 3, leading_ws is 0 -> offset 3.
    assert f._get_offset(b) == 3


def test_get_offset_with_leading_ws():
    tl, a, ws, b = _make_simple_group()
    f = ReindentFilter(width=2, indent_after_first=True)
    f._curr_stmt = tl
    # leading_ws = 2; b position 3 -> 3 - 2 = 1
    assert f._get_offset(b) == 1


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_from():
    tlist = sql.TokenList([
        sql.Token(T.Name, 'x'),
        sql.Token(T.Keyword, 'FROM'),
    ])
    f = ReindentFilter()
    idx, tok = f._next_token(tlist)
    assert idx == 1
    assert tok.normalized == 'FROM'


def test_next_token_none():
    tlist = sql.TokenList([sql.Token(T.Name, 'abc')])
    f = ReindentFilter()
    idx, tok = f._next_token(tlist)
    assert idx is None
    assert tok is None


def test_next_token_between_skips_to_and():
    tlist = sql.TokenList([
        sql.Token(T.Keyword, 'BETWEEN'),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Keyword, 'AND'),
    ])
    f = ReindentFilter()
    idx, tok = f._next_token(tlist)
    # BETWEEN -> recursively find after; finds AND.
    assert tok is not None
    assert tok.normalized == 'AND'


# ---------------------------------------------------------------------------
# process() / public behavior on parsed statements
# ---------------------------------------------------------------------------

def test_process_returns_stmt():
    stmt = sql.Statement([sql.Token(T.Name, 'foo')])
    f = ReindentFilter()
    result = f.process(stmt)
    assert result is stmt
    assert f._curr_stmt is stmt
    assert f._last_stmt is stmt


def test_process_two_statements_inserts_separator():
    s1 = sql.Statement([sql.Token(T.Name, 'a')])
    s2 = sql.Statement([sql.Token(T.Name, 'b')])
    f = ReindentFilter()
    f.process(s1)
    f.process(s2)
    # Second statement should get a leading whitespace separator inserted.
    assert s2.tokens[0].ttype is T.Whitespace
    assert s2.tokens[0].value == '\n\n'


def test_process_second_stmt_after_trailing_newline():
    s1 = sql.Statement([sql.Token(T.Name, 'a')])
    # Force trailing newline on str(s1)
    s1.tokens.append(sql.Token(T.Whitespace, '\n'))
    s2 = sql.Statement([sql.Token(T.Name, 'b')])
    f = ReindentFilter()
    f.process(s1)
    f.process(s2)
    assert s2.tokens[0].ttype is T.Whitespace
    assert s2.tokens[0].value == '\n'


# ---------------------------------------------------------------------------
# _split_statements
# ---------------------------------------------------------------------------

def test_split_statements_inserts_newline_before_select():
    select = sql.Token(T.Keyword.DML, 'SELECT')
    name = sql.Token(T.Name, 'x')
    tlist = sql.TokenList([name, select])
    f = ReindentFilter()
    f._split_statements(tlist)
    # A newline should have been inserted before SELECT.
    idx, tok = tlist.token_next_by(m=(T.Keyword, 'SELECT'))
    prev_idx, prev = tlist.token_prev(idx, skip_ws=False)
    assert prev is not None
    assert prev.ttype is T.Whitespace
    assert prev.value.startswith('\n')


def test_split_statements_no_break_for_first_token():
    select = sql.Token(T.Keyword.DML, 'SELECT')
    tlist = sql.TokenList([select])
    f = ReindentFilter()
    f._split_statements(tlist)
    # Only token -> no insert; the TokenList should still hold SELECT.
    assert tlist.tokens[0] is select


def test_split_statements_removes_ws_before_select():
    ws = sql.Token(T.Whitespace, '   ')
    select = sql.Token(T.Keyword.DML, 'SELECT')
    tlist = sql.TokenList([ws, select])
    f = ReindentFilter()
    f._split_statements(tlist)
    # Original ws token should be replaced by a filtered-out sequence; newline
    # token should precede SELECT.
    idx = tlist.token_index(select)
    assert idx == 1
    assert tlist.tokens[0].ttype is T.Whitespace
    assert tlist.tokens[0].value.startswith('\n')


# ---------------------------------------------------------------------------
# _split_kwds
# ---------------------------------------------------------------------------

def test_split_kwds_inserts_newline_before_keyword():
    name = sql.Token(T.Name, 'x')
    kw = sql.Token(T.Keyword, 'JOIN')
    other = sql.Token(T.Name, 'y')
    tlist = sql.TokenList([name, sql.Token(T.Whitespace, ' '), kw, other])
    f = ReindentFilter()
    f._split_kwds(tlist)
    idx = tlist.token_index(kw)
    assert idx > 0
    prev = tlist.tokens[idx - 1]
    assert prev.ttype is T.Whitespace
    assert prev.value.startswith('\n')


def test_split_kwds_no_extra_newline_if_already():
    name = sql.Token(T.Name, 'x')
    kw = sql.Token(T.Keyword, 'JOIN')
    tlist = sql.TokenList([name, sql.Token(T.Whitespace, '\n'), kw])
    f = ReindentFilter()
    f._split_kwds(tlist)
    idx = tlist.token_index(kw)
    prev = tlist.tokens[idx - 1]
    assert prev.ttype is T.Whitespace
    assert prev.value == '\n'


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

def test_process_where_no_where_keyword():
    tlist = sql.TokenList([sql.Token(T.Name, 'x')])
    f = ReindentFilter()
    # Should not raise nor modify; no WHERE token.
    before = len(tlist.tokens)
    f._process_where(tlist)
    assert len(tlist.tokens) == before


def test_process_where_inserts_newline():
    where = sql.Token(T.Keyword, 'WHERE')
    x = sql.Token(T.Name, 'x')
    tlist = sql.TokenList([sql.Token(T.Whitespace, ' '), where, x])
    f = ReindentFilter()
    f._process_where(tlist)
    idx = tlist.token_index(where)
    prev = tlist.tokens[idx - 1]
    assert prev.ttype is T.Whitespace
    assert prev.value.startswith('\n')


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_open():
    tlist = sql.Parenthesis([])
    f = ReindentFilter()
    before = len(tlist.tokens)
    f._process_parenthesis(tlist)
    assert len(tlist.tokens) == before


def test_process_parenthesis_simple():
    open_p = sql.Token(T.Punctuation, '(')
    x = sql.Token(T.Name, 'x')
    close_p = sql.Token(T.Punctuation, ')')
    tlist = sql.Parenthesis([open_p, x, close_p])
    f = ReindentFilter()
    f._process_parenthesis(tlist)
    # No error; offset restored.
    assert f.offset == 0


def test_process_parenthesis_with_dml():
    open_p = sql.Token(T.Punctuation, '(')
    select = sql.Token(T.Keyword.DML, 'SELECT')
    x = sql.Token(T.Name, 'x')
    close_p = sql.Token(T.Punctuation, ')')
    tlist = sql.Parenthesis([open_p, select, sql.Token(T.Whitespace, ' '),
                             x, close_p])
    f = ReindentFilter()
    f._process_parenthesis(tlist)
    # DML present -> a leading newline inserted at index 0.
    assert tlist.tokens[0].ttype is T.Whitespace
    assert tlist.tokens[0].value.startswith('\n')


# ---------------------------------------------------------------------------
# _process_function
# ---------------------------------------------------------------------------

def test_process_function_sets_last_func():
    fn = sql.Token(T.Name, 'foo')
    tlist = sql.Function([fn])
    f = ReindentFilter()
    f._process_function(tlist)
    assert f._last_func is fn


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def _make_identifier_list(n=3):
    tokens = []
    for i in range(n):
        if i > 0:
            tokens.append(sql.Token(T.Punctuation, ','))
            tokens.append(sql.Token(T.Whitespace, ' '))
        tokens.append(sql.Token(T.Name, 'c%d' % i))
    return sql.IdentifierList(tokens)


def test_process_identifierlist_no_wrap():
    tlist = _make_identifier_list(3)
    f = ReindentFilter(wrap_after=0)
    # Should not raise; nothing wrapped.
    before = str(tlist)
    f._process_identifierlist(tlist)
    assert '\n' not in str(tlist)
    assert str(tlist) == before


def test_process_identifierlist_wraps():
    tlist = _make_identifier_list(3)
    f = ReindentFilter(wrap_after=1)
    f._process_identifierlist(tlist)
    # With wrap_after=1 and offset arithmetic, newline whitespace should appear.
    assert '\n' in str(tlist)


def test_process_identifierlist_indent_columns():
    tlist = _make_identifier_list(3)
    f = ReindentFilter(wrap_after=1, indent_columns=True)
    f._process_identifierlist(tlist)
    assert '\n' in str(tlist)


def test_process_identifierlist_indent_columns_tab():
    tlist = _make_identifier_list(3)
    f = ReindentFilter(wrap_after=1, indent_columns=True, char='\t')
    f._process_identifierlist(tlist)
    assert '\n' in str(tlist)


def test_process_identifierlist_comma_first():
    tlist = _make_identifier_list(3)
    f = ReindentFilter(wrap_after=1, comma_first=True)
    f._process_identifierlist(tlist)
    # A newline must have been inserted.
    assert '\n' in str(tlist)


def test_process_identifierlist_within_function_ensure_ws():
    fn = sql.Token(T.Name, 'f')
    inner = _make_identifier_list(2)
    # Values within Function branch exercised via .within(sql.Function)
    tlist = sql.Function([sql.Token(T.Name, 'f'),
                          sql.Token(T.Punctuation, '('),
                          inner,
                          sql.Token(T.Punctuation, ')')])
    f = ReindentFilter()
    f._last_func = fn
    f._process_identifierlist(inner)
    # _process_default runs; ensure no exception.
    assert inner is not None


def test_process_identifierlist_function_wraps():
    inner = _make_identifier_list(3)
    fn = sql.Token(T.Name, 'long_function_name')
    tlist = sql.Function([fn,
                          sql.Token(T.Punctuation, '('),
                          inner,
                          sql.Token(T.Punctuation, ')')])
    f = ReindentFilter(wrap_after=5)
    f._last_func = fn
    f._process_identifierlist(inner)
    assert '\n' in str(inner)


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case_basic():
    # CASE WHEN ... THEN ... ELSE ... END
    case = sql.Case([
        sql.Token(T.Keyword, 'CASE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'WHEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'THEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'ELSE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '2'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'END'),
    ])
    f = ReindentFilter(wrap_after=0)
    f._process_case(case)
    # Should not raise.
    assert f.offset == 0


def test_process_case_compact_no_newlines():
    case = sql.Case([
        sql.Token(T.Keyword, 'CASE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'WHEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'x'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'THEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'END'),
    ])
    f = ReindentFilter(compact=True)
    before = str(case)
    f._process_case(case)
    # compact mode suppresses the wrap logic; contents should be unchanged
    # (still no newlines inserted).
    assert '\n' not in str(case)


def test_process_case_wraps():
    case = sql.Case([
        sql.Token(T.Keyword, 'CASE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'WHEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Name, 'condition'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'THEN'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'ELSE'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '2'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword, 'END'),
    ])
    f = ReindentFilter(wrap_after=1)
    f._process_case(case)
    assert '\n' in str(case)


# ---------------------------------------------------------------------------
# _process_values
# ---------------------------------------------------------------------------

def test_process_values():
    p1 = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Punctuation, ')'),
    ])
    comma = sql.Token(T.Punctuation, ',')
    p2 = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Number.Integer, '2'),
        sql.Token(T.Punctuation, ')'),
    ])
    tlist = sql.Values([
        sql.Token(T.Keyword, 'VALUES'),
        sql.Token(T.Whitespace, ' '),
        p1, comma,
        sql.Token(T.Whitespace, ' '),
        p2,
    ])
    f = ReindentFilter()
    f._process_values(tlist)
    # Leading newline inserted at index 0.
    assert tlist.tokens[0].ttype is T.Whitespace
    assert tlist.tokens[0].value.startswith('\n')
    # A newline inserted after the comma (non comma_first)
    idx_comma = tlist.token_index(comma)
    nxt = tlist.tokens[idx_comma + 1]
    assert nxt.ttype is T.Whitespace
    assert nxt.value.startswith('\n')


def test_process_values_comma_first():
    p1 = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Punctuation, ')'),
    ])
    comma = sql.Token(T.Punctuation, ',')
    p2 = sql.Parenthesis([
        sql.Token(T.Punctuation, '('),
        sql.Token(T.Number.Integer, '2'),
        sql.Token(T.Punctuation, ')'),
    ])
    tlist = sql.Values([
        sql.Token(T.Keyword, 'VALUES'),
        sql.Token(T.Whitespace, ' '),
        p1, comma,
        sql.Token(T.Whitespace, ' '),
        p2,
    ])
    f = ReindentFilter(comma_first=True)
    f._process_values(tlist)
    # With comma_first the newline is inserted before the comma.
    idx_comma = tlist.token_index(comma)
    prev = tlist.tokens[idx_comma - 1]
    assert prev.ttype is T.Whitespace
    assert prev.value.startswith('\n')


# ---------------------------------------------------------------------------
# _process_default dispatch
# ---------------------------------------------------------------------------

def test_process_default_splits_keywords():
    name = sql.Token(T.Name, 'x')
    where_kw = sql.Token(T.Keyword, 'WHERE')
    val = sql.Token(T.Name, 'y')
    tlist = sql.TokenList([name, sql.Token(T.Whitespace, ' '),
                           where_kw, sql.Token(T.Whitespace, ' '), val])
    f = ReindentFilter()
    f._process_default(tlist, stmts=False)
    idx = tlist.token_index(where_kw)
    prev = tlist.tokens[idx - 1]
    assert prev.ttype is T.Whitespace
    assert prev.value.startswith('\n')


def test_process_default_recurses_into_sublists():
    inner = sql.TokenList([sql.Token(T.Name, 'inner')])
    outer = sql.TokenList([inner])
    f = ReindentFilter()
    called = []
    orig = f._process

    def spy(tlist):
        called.append(tlist)
        return orig(tlist)
    f._process = spy
    f._process_default(outer, stmts=False)
    assert inner in called


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatches_default():
    tlist = sql.TokenList([sql.Token(T.Name, 'x')])
    f = ReindentFilter()
    called = []
    orig = f._process_default

    def spy(tlist_, stmts=True):
        called.append(tlist_)
        return orig(tlist_, stmts)
    f._process_default = spy
    f._process(tlist)
    assert called == [tlist]


def test_process_dispatches_named_handler():
    where = sql.Token(T.Keyword, 'WHERE')
    tlist = sql.Where([sql.Token(T.Whitespace, ' '), where,
                       sql.Token(T.Name, 'x')])
    f = ReindentFilter()
    called = []
    orig = f._process_where

    def spy(tlist_):
        called.append(tlist_)
        return orig(tlist_)
    f._process_where = spy
    f._process(tlist)
    assert called == [tlist]


# ---------------------------------------------------------------------------
# indent/offset context managers sanity (used by the filter)
# ---------------------------------------------------------------------------

def test_indent_context_manager_restores():
    f = ReindentFilter(width=2)
    assert f.indent == 0
    with indent(f):
        assert f.indent == 1
        with indent(f, 2):
            assert f.indent == 3
        assert f.indent == 1
    assert f.indent == 0


def test_offset_context_manager_restores():
    f = ReindentFilter()
    assert f.offset == 0
    with offset(f, 5):
        assert f.offset == 5
    assert f.offset == 0


# ---------------------------------------------------------------------------
# End-to-end via process on real parsed statements
# ---------------------------------------------------------------------------

def test_end_to_end_select_simple():
    stmt = parse("SELECT a, b FROM foo")
    f = ReindentFilter()
    out = f.process(stmt)
    text = str(out)
    # FROM gets moved to a new line.
    assert '\nFROM' in text


def test_end_to_end_where():
    stmt = parse("SELECT a FROM foo WHERE a = 1")
    f = ReindentFilter()
    out = f.process(stmt)
    text = str(out)
    assert '\nWHERE' in text


def test_end_to_end_two_statements():
    stmts = parse("SELECT 1; SELECT 2")
    f = ReindentFilter()
    out = [f.process(s) for s in stmts]
    # Second statement should carry a leading whitespace separator.
    assert out[1].tokens[0].ttype is T.Whitespace
    assert out[1].tokens[0].value in ('\n', '\n\n')


def test_end_to_end_compact_width_char():
    stmt = parse("SELECT a, b FROM foo")
    f = ReindentFilter(width=4, char='\t', n='\r\n')
    out = f.process(stmt)
    text = str(out)
    assert '\r\n' in text
