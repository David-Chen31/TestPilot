import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter


def build(tokens):
    """Build an sql.TokenList from raw token tuples (ttype, value)."""
    tlist = sql.TokenList()
    for ttype, value in tokens:
        tlist.tokens.append(sql.Token(ttype, value))
    return tlist


def parse(sql_text):
    import sqlparse
    return sqlparse.parse(sql_text)[0]


# ---------------------------------------------------------------------------
# Construction / attribute defaults
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
    assert f.n == '\r\n'
    assert f.width == 4
    assert f.char == '\t'
    assert f.indent == 1
    assert f.wrap_after == 10
    assert f.comma_first is True
    assert f.indent_columns is True
    assert f.compact is True


# ---------------------------------------------------------------------------
# leading_ws property
# ---------------------------------------------------------------------------

def test_leading_ws_default():
    f = ReindentFilter()
    # offset 0 + indent 0 * width 2 == 0
    assert f.leading_ws == 0


def test_leading_ws_with_indent():
    f = ReindentFilter(width=4, indent_after_first=True)
    # offset 0 + indent 1 * width 4 == 4
    assert f.leading_ws == 4


def test_leading_ws_with_offset():
    f = ReindentFilter(width=3, indent_after_first=True)
    f.offset = 5
    # 5 + 1 * 3 == 8
    assert f.leading_ws == 8


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default():
    f = ReindentFilter()
    tok = f.nl()
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n'


def test_nl_with_offset():
    f = ReindentFilter(width=2, indent_after_first=True)
    tok = f.nl(offset=3)
    # leading_ws = 0 + 1*2 = 2, plus offset 3 == 5 spaces
    assert tok.value == '\n     '


def test_nl_negative_offset_clamped():
    f = ReindentFilter()
    tok = f.nl(offset=-10)
    # max(0, 0 + -10) == 0 -> just newline
    assert tok.value == '\n'


def test_nl_char():
    f = ReindentFilter(width=3, char='\t', indent_after_first=True)
    tok = f.nl()
    assert tok.value == '\n' + '\t' * 3


def test_nl_custom_newline():
    f = ReindentFilter(n='\r\n', width=1, indent_after_first=True)
    tok = f.nl()
    assert tok.value == '\r\n '


# ---------------------------------------------------------------------------
# _next_token / _split_kwds / _split_statements
# ---------------------------------------------------------------------------

def test_next_token_finds_keyword():
    tlist = build([
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Keyword, 'FROM'),
        (T.Whitespace, ' '),
        (T.Keyword, 'WHERE'),
    ])
    f = ReindentFilter()
    idx, token = f._next_token(tlist)
    assert idx == 2
    assert token.value == 'FROM'


def test_next_token_between_skips():
    tlist = build([
        (T.Keyword, 'WHERE'),
        (T.Whitespace, ' '),
        (T.Keyword, 'a'),
        (T.Whitespace, ' '),
        (T.Keyword, 'BETWEEN'),
        (T.Whitespace, ' '),
        (T.Keyword, 'b'),
        (T.Whitespace, ' '),
        (T.Keyword, 'AND'),
        (T.Whitespace, ' '),
        (T.Keyword, 'c'),
        (T.Keyword, 'LIMIT'),
    ])
    f = ReindentFilter()
    idx, token = f._next_token(tlist)
    # 'WHERE' matches first? WHERE is not in split_words so it's skipped;
    # BETWEEN is in split_words so first match. Then BETWEEN skips its AND,
    # so the next token after that sequence is LIMIT... but _next_token
    # returns one token only per call. First call returns BETWEEN.
    assert token.value == 'BETWEEN'
    idx2, token2 = f._next_token(tlist, idx)
    # After BETWEEN, it processes: at BETWEEN, skip its AND, returns next.
    assert token2.value == 'LIMIT'


def test_next_token_none():
    tlist = build([
        (T.Whitespace, ' '),
        (T.Number.Integer, '5'),
    ])
    f = ReindentFilter()
    idx, token = f._next_token(tlist)
    assert idx is None
    assert token is None


def test_split_kwds_inserts_newline():
    tlist = build([
        (T.Keyword.DML, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Keyword, 'FROM'),
        (T.Whitespace, ' '),
        (T.Identifier, 'tbl'),
    ])
    f = ReindentFilter()
    f._split_kwds(tlist)
    # A newline whitespace token should now precede FROM
    values = [t.value for t in tlist.tokens]
    idx = values.index('FROM')
    assert values[idx - 1] == '\n'


def test_split_kwds_existing_newline_untouched():
    tlist = build([
        (T.Keyword, 'FROM'),
    ])
    # insert a leading newline whitespace
    tlist.tokens.insert(0, sql.Token(T.Whitespace, '\n'))
    f = ReindentFilter()
    f._split_kwds(tlist)
    assert tlist.tokens[0].value == '\n'
    # no extra newline inserted before FROM
    assert tlist.tokens[1].value == 'FROM'


def test_split_statements_inserts_newline_between():
    tlist = build([
        (T.Keyword.DML, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Keyword.DML, 'SELECT'),
    ])
    f = ReindentFilter()
    f._split_statements(tlist)
    # second SELECT should be preceded by newline whitespace
    idx = [i for i, t in enumerate(tlist.tokens) if t.value == 'SELECT'][1]
    assert tlist.tokens[idx - 1].value == '\n'


def test_split_statements_first_token_not_preceded():
    tlist = build([
        (T.Keyword.DML, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Identifier, 'x'),
    ])
    f = ReindentFilter()
    f._split_statements(tlist)
    assert tlist.tokens[0].value == 'SELECT'


# ---------------------------------------------------------------------------
# _process dispatch
# ---------------------------------------------------------------------------

def test_process_dispatch_uses_process_default(monkeypatch):
    f = ReindentFilter()
    called = {}

    def fake_default(tlist, stmts=True):
        called['tlist'] = tlist
        called['stmts'] = stmts

    monkeypatch.setattr(f, '_process_default', fake_default)
    tlist = sql.TokenList()
    f._process(tlist)
    assert called['tlist'] is tlist


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

def test_process_where_returns_without_where():
    f = ReindentFilter()
    tlist = build([
        (T.Keyword, 'SELECT'),
        (T.Whitespace, ' '),
        (T.Identifier, 'x'),
    ])
    f._process_where(tlist)
    # No newline inserted
    assert all(t.value != '\n' for t in tlist.tokens)


def test_process_where_inserts_newline():
    tlist = build([
        (T.Keyword, 'WHERE'),
        (T.Whitespace, ' '),
        (T.Identifier, 'x'),
    ])
    f = ReindentFilter()
    f._process_where(tlist)
    assert tlist.tokens[0].value == '\n'
    assert tlist.tokens[1].value == 'WHERE'


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_open_returns():
    tlist = build([
        (T.Keyword.DML, 'SELECT'),
    ])
    f = ReindentFilter()
    # No parenthesis open token -> should return without error
    f._process_parenthesis(tlist)
    # nothing inserted
    assert tlist.tokens[0].value == 'SELECT'


# ---------------------------------------------------------------------------
# _process_default
# ---------------------------------------------------------------------------

def test_process_default_recurses_sublists():
    f = ReindentFilter()
    visited = []

    def fake_process(sgroup):
        visited.append(sgroup)

    f._process = fake_process
    inner = sql.TokenList()
    tlist = sql.TokenList()
    tlist.tokens.append(inner)
    f._process_default(tlist, stmts=False)
    assert visited == [inner]


# ---------------------------------------------------------------------------
# process() (public entry point)
# ---------------------------------------------------------------------------

def test_process_returns_stmt():
    f = ReindentFilter()
    stmt = parse('SELECT a FROM b')
    result = f.process(stmt)
    assert result is stmt
    assert f._last_stmt is stmt
    assert f._curr_stmt is stmt


def test_process_second_stmt_gets_leading_blank_line():
    f = ReindentFilter()
    stmt1 = parse('SELECT 1')
    stmt2 = parse('SELECT 2')
    f.process(stmt1)
    f.process(stmt2)
    first = stmt2.tokens[0]
    assert first.is_whitespace
    assert first.value == '\n\n'


def test_process_second_stmt_after_trailing_newline_uses_single_nl():
    f = ReindentFilter()
    stmt1 = parse('SELECT 1')
    # simulate that last_stmt's string ends with newline
    f._last_stmt = sql.Statement([sql.Token(T.Whitespace, '\n')])
    stmt2 = parse('SELECT 2')
    f.process(stmt2)
    first = stmt2.tokens[0]
    assert first.is_whitespace
    assert first.value == '\n'


# ---------------------------------------------------------------------------
# Integration-ish checks via process()
# ---------------------------------------------------------------------------

def test_process_simple_from_newline():
    f = ReindentFilter()
    stmt = parse('SELECT a FROM b')
    out = str(f.process(stmt))
    assert 'FROM' in out
    assert '\nFROM' in out


def test_process_with_where_indented():
    f = ReindentFilter(width=2)
    stmt = parse('SELECT a FROM b WHERE x = 1')
    out = str(f.process(stmt))
    assert '\nWHERE' in out


def test_process_parenthesis_subquery():
    f = ReindentFilter()
    stmt = parse('SELECT * FROM (SELECT a FROM b) x')
    out = str(f.process(stmt))
    # subquery DML gets a newline in front
    assert '(\n' in out


def test_process_comma_first_wrapping():
    f = ReindentFilter(wrap_after=5, comma_first=True)
    stmt = parse('SELECT aaa, bbb, ccc FROM tbl')
    out = str(f.process(stmt))
    assert ',' in out


def test_process_indent_columns():
    f = ReindentFilter(indent_columns=True, width=4)
    stmt = parse('SELECT a, b FROM tbl')
    out = str(f.process(stmt))
    assert 'a' in out and 'b' in out


def test_process_values():
    f = ReindentFilter()
    stmt = parse('INSERT INTO t VALUES (1, 2), (3, 4)')
    out = str(f.process(stmt))
    assert 'VALUES' in out


def test_process_function_last_func_set():
    f = ReindentFilter()
    stmt = parse('SELECT COUNT(a) FROM t')
    f.process(stmt)
    # a function was processed, so _last_func should have been set
    assert f._last_func is not None


def test_process_case_compact():
    f = ReindentFilter(compact=True)
    stmt = parse('SELECT CASE WHEN a THEN b ELSE c END FROM t')
    out = str(f.process(stmt))
    assert 'CASE' in out


def test_process_case_non_compact():
    f = ReindentFilter(compact=False, wrap_after=10)
    stmt = parse('SELECT CASE WHEN aaaa THEN bbbb ELSE cccc END FROM t')
    out = str(f.process(stmt))
    assert 'CASE' in out


def test_process_multiple_statements():
    f = ReindentFilter()
    stmt = sql.Statement([
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '1'),
        sql.Token(T.Punctuation, ';'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Number.Integer, '2'),
    ])
    out = str(f.process(stmt))
    assert 'SELECT' in out


# ---------------------------------------------------------------------------
# _current_line_len edge cases
# ---------------------------------------------------------------------------

def test_current_line_len_single_token():
    f = ReindentFilter()
    tok = sql.Token(T.Identifier, 'abc')
    tlist = sql.TokenList()
    tlist.tokens.append(tok)
    f._curr_stmt = tlist
    # single token, no previous tokens
    assert f._current_line_len(tok) == 0


def test_current_line_len_with_previous():
    f = ReindentFilter()
    tlist = sql.TokenList()
    t1 = sql.Token(T.Identifier, 'abc')
    t2 = sql.Token(T.Identifier, 'de')
    tlist.tokens.extend([t1, t2])
    f._curr_stmt = tlist
    # t1 is 3 chars, so length at t2 is 3
    assert f._current_line_len(t2) == 3


def test_current_line_len_after_newline():
    f = ReindentFilter()
    tlist = sql.TokenList()
    t1 = sql.Token(T.Identifier, 'abc')
    nl = sql.Token(T.Whitespace, '\n  ')
    t2 = sql.Token(T.Identifier, 'xy')
    tlist.tokens.extend([t1, nl, t2])
    f._curr_stmt = tlist
    # After newline with 2 spaces => current line length 2
    assert f._current_line_len(t2) == 2


# ---------------------------------------------------------------------------
# _get_offset
# ---------------------------------------------------------------------------

def test_get_offset_subtracts_leading_ws():
    f = ReindentFilter(width=2, indent_after_first=True)
    tok = sql.Token(T.Identifier, 'abc')
    tlist = sql.TokenList()
    tlist.tokens.append(tok)
    f._curr_stmt = tlist
    # _current_line_len == 0, leading_ws == 2 -> 0 - 2 == -2
    assert f._get_offset(tok) == -2
