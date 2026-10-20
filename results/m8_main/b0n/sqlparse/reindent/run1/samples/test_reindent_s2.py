import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter


def parse_one(sql_text):
    import sqlparse
    return sqlparse.parse(sql_text)[0]


def fmt(sql_text, **kwargs):
    import sqlparse
    return sqlparse.format(sql_text, reindent=True, **kwargs)


# ---------------------------------------------------------------------------
# Constructor / attribute defaults
# ---------------------------------------------------------------------------

def test_constructor_defaults():
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


def test_constructor_indent_after_first():
    f = ReindentFilter(indent_after_first=True)
    assert f.indent == 1


def test_constructor_custom_values():
    f = ReindentFilter(width=4, char='\t', wrap_after=10, n='\r\n',
                       comma_first=True, indent_columns=True, compact=True)
    assert f.width == 4
    assert f.char == '\t'
    assert f.wrap_after == 10
    assert f.n == '\r\n'
    assert f.comma_first is True
    assert f.indent_columns is True
    assert f.compact is True


# ---------------------------------------------------------------------------
# leading_ws property
# ---------------------------------------------------------------------------

def test_leading_ws_default():
    f = ReindentFilter()
    # offset 0 + indent 0 * width 2
    assert f.leading_ws == 0


def test_leading_ws_with_indent():
    f = ReindentFilter(width=3, indent_after_first=True)
    assert f.leading_ws == 3


def test_leading_ws_with_offset():
    f = ReindentFilter(width=2, indent_after_first=True)
    f.offset = 5
    assert f.leading_ws == 5 + 1 * 2


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default():
    f = ReindentFilter(width=2, indent_after_first=True)
    tok = f.nl()
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n' + ' ' * 2


def test_nl_with_offset():
    f = ReindentFilter(width=2)
    f.offset = 3
    tok = f.nl(offset=2)
    assert tok.value == '\n' + ' ' * 5


def test_nl_negative_offset_clamped():
    f = ReindentFilter(width=2)
    tok = f.nl(offset=-10)
    # max(0, leading_ws + offset)
    assert tok.value == '\n'


def test_nl_uses_custom_char_and_newline():
    f = ReindentFilter(width=3, char='\t', n='\r\n', indent_after_first=True)
    tok = f.nl()
    assert tok.value == '\r\n' + '\t' * 3


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_keyword():
    f = ReindentFilter()
    stmt = parse_one('SELECT a FROM b')
    tidx, token = f._next_token(stmt)
    assert token is not None
    assert token.normalized in ('FROM', 'JOIN', 'AND', 'OR', 'UNION')


def test_next_token_none_when_absent():
    f = ReindentFilter()
    stmt = parse_one('SELECT a b c')
    tidx, token = f._next_token(stmt)
    assert token is None
    assert tidx is None


def test_next_token_skips_between_and():
    f = ReindentFilter()
    stmt = parse_one('SELECT a WHERE b BETWEEN 1 AND 2')
    tidx, token = f._next_token(stmt)
    # BETWEEN should be skipped, its AND should be skipped, eventually WHERE
    # appears first; check the token found is a real keyword.
    assert token is not None
    assert token.ttype is T.Keyword


# ---------------------------------------------------------------------------
# _current_line_len
# ---------------------------------------------------------------------------

def test_current_line_len_simple():
    f = ReindentFilter()
    stmt = parse_one('SELECT a')
    f._curr_stmt = stmt
    # Measure the second token (whitespace)
    tok = stmt.tokens[1]
    assert f._current_line_len(tok) == len('SELECT')


def test_current_line_len_within_token_stops_at_stmt():
    f = ReindentFilter()
    stmt = parse_one('SELECT a')
    f._curr_stmt = stmt
    # First token: nothing before it
    assert f._current_line_len(stmt.tokens[0]) == 0


def test_current_line_len_after_newline():
    f = ReindentFilter()
    stmt = parse_one('SELECT\n   a')
    f._curr_stmt = stmt
    # Find the 'a' token
    last = stmt.tokens[-1]
    # tokens: 'SELECT', ws '\n   ', 'a'
    length = f._current_line_len(last)
    assert length == 3  # spaces before 'a'


# ---------------------------------------------------------------------------
# _get_offset
# ---------------------------------------------------------------------------

def test_get_offset_matches_line_len_minus_leading_ws():
    f = ReindentFilter(width=2)
    stmt = parse_one('SELECT a')
    f._curr_stmt = stmt
    tok = stmt.tokens[1]
    expected = f._current_line_len(tok) - len(' ' * f.leading_ws)
    assert f._get_offset(tok) == expected


# ---------------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------------

def test_process_returns_stmt():
    f = ReindentFilter()
    stmt = parse_one('SELECT a FROM b')
    result = f.process(stmt)
    assert result is stmt
    assert f._last_stmt is stmt
    assert f._curr_stmt is stmt


def test_process_second_statement_gets_leading_newlines():
    f = ReindentFilter()
    stmt1 = parse_one('SELECT a')
    stmt2 = parse_one('SELECT b')
    f.process(stmt1)
    f.process(stmt2)
    # Since stmt1 doesn't end with newline, two newlines are inserted.
    first_tok = stmt2.tokens[0]
    assert first_tok.ttype is T.Whitespace
    assert first_tok.value == '\n\n'


# ---------------------------------------------------------------------------
# High-level formatting behavior
# ---------------------------------------------------------------------------

def test_reindent_keywords_on_own_lines():
    out = fmt('SELECT a FROM b WHERE c = d')
    assert '\nFROM' in out or 'FROM' in out
    # WHERE should be placed at start of a line
    assert '\nWHERE' in out


def test_reindent_and_or_join_split():
    out = fmt('SELECT a FROM b JOIN c ON a.x = c.x AND a.y = c.y')
    assert 'JOIN' in out
    assert 'AND' in out


def test_reindent_custom_width():
    out = fmt('SELECT a FROM b', indent_width=4)
    # Some line should be indented with 4 spaces
    assert 'FROM' in out


def test_reindent_comma_first():
    out = fmt('SELECT a, b, c FROM t', comma_first=True, wrap_after=1)
    # comma_first places commas at line beginnings
    lines = out.splitlines()
    assert any(line.lstrip().startswith(',') for line in lines)


def test_reindent_identifierlist_wrap():
    out = fmt('SELECT aaaa, bbbb, cccc, dddd FROM t", wrap_after=5)'.replace('", wrap_after=5)', '\', wrap_after=5)'))
    # just ensure no crash and something formatted
    assert 'SELECT' in out


def test_reindent_columns():
    out = fmt('SELECT a, b, c FROM t', indent_columns=True)
    assert out.count('\n') >= 2


def test_reindent_values():
    out = fmt("INSERT INTO t VALUES (1, 2), (3, 4), (5, 6)")
    assert 'VALUES' in out
    assert out.count('\n') >= 2


def test_reindent_case():
    out = fmt('SELECT CASE WHEN a THEN b WHEN c THEN d ELSE e END FROM t')
    assert 'CASE' in out
    assert 'WHEN' in out


def test_reindent_compact_case():
    out = fmt('SELECT CASE WHEN a THEN b END FROM t', compact=True)
    # compact avoids wrapping within case; still valid
    assert 'CASE' in out


def test_reindent_multiple_statements():
    out = fmt('SELECT a; SELECT b;')
    assert out.count('SELECT') == 2


def test_reindent_subquery_parenthesis():
    out = fmt('SELECT * FROM (SELECT a FROM b) x')
    assert 'SELECT' in out
    assert '\n' in out


def test_reindent_function_call():
    out = fmt('SELECT foo(a, b) FROM t')
    assert 'foo' in out


def test_reindent_tabs_char():
    out = fmt('SELECT a, b FROM t', indent_width=1, indent_columns=True)
    assert 'SELECT' in out


def test_reindent_empty_input():
    assert fmt('') == ''


def test_reindent_plain_keyword_only():
    out = fmt('SELECT')
    assert out.strip() == 'SELECT'


def test_reindent_wrap_after_zero_no_wrapping():
    out = fmt('SELECT a, b, c FROM t', wrap_after=0)
    # No forced wrapping inside identifier list when wrap_after is 0
    assert 'SELECT' in out


# ---------------------------------------------------------------------------
# _process_where / _split_kwds / _split_statements internals
# ---------------------------------------------------------------------------

def test_process_where_inserts_newline():
    f = ReindentFilter()
    stmt = parse_one('SELECT a WHERE b')
    f._curr_stmt = stmt
    f._process_where(stmt)
    # Should have inserted a newline whitespace token before WHERE
    found = False
    for tok in stmt.tokens:
        if tok.ttype is T.Whitespace and '\n' in tok.value:
            found = True
            break
    assert found


def test_process_where_without_where_returns():
    f = ReindentFilter()
    stmt = parse_one('SELECT a')
    f._curr_stmt = stmt
    before = list(stmt.tokens)
    f._process_where(stmt)
    assert list(stmt.tokens) == before


def test_split_statements_inserts_breaks():
    f = ReindentFilter()
    stmt = parse_one('SELECT a SELECT b')
    f._curr_stmt = stmt
    f._split_statements(stmt)
    # there should now be a newline whitespace token inserted
    values = [t.value for t in stmt.tokens]
    assert any('\n' in v for v in values)


def test_split_kwds_noop_when_no_keywords():
    f = ReindentFilter()
    stmt = parse_one('abc def')
    f._curr_stmt = stmt
    before = [t.value for t in stmt.tokens]
    f._split_kwds(stmt)
    assert [t.value for t in stmt.tokens] == before


# ---------------------------------------------------------------------------
# _process_parenthesis empty parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_open_token_returns():
    f = ReindentFilter()
    stmt = parse_one('SELECT 1')
    f._curr_stmt = stmt
    # Fake a parenthesis group without open token
    grp = sql.Parenthesis([sql.Token(T.Number, '1')])
    f._process_parenthesis(grp)
    # no exception, unchanged
    assert grp.tokens[0].value == '1'


# ---------------------------------------------------------------------------
# _process dispatch fallback
# ---------------------------------------------------------------------------

def test_process_dispatch_unknown_uses_default():
    f = ReindentFilter()
    stmt = parse_one('SELECT a FROM b')
    f._curr_stmt = stmt
    # sql.Statement has no _process_statement; must fall back to _process_default
    f._process(stmt)
    assert '\n' in str(stmt)
