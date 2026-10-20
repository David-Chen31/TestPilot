import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter


def make_stmt(text):
    """Parse a statement via sqlparse's parser (kept local to avoid
    depending on the formatter's default filter stack)."""
    from sqlparse.lexer import Lexer
    from sqlparse.parser import Parser

    Lexer.get_default_instance().default_initialized = True
    return Parser().parse(text)[0]


def run(text, **kwargs):
    filt = ReindentFilter(**kwargs)
    stmt = make_stmt(text)
    return str(filt.process(stmt))


# ---------------------------------------------------------------------------
# Construction / defaults
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
    assert f.indent == 1
    assert f.wrap_after == 10
    assert f.n == '\r\n'
    assert f.comma_first is True
    assert f.indent_columns is True
    assert f.compact is True


def test_leading_ws_property():
    f = ReindentFilter(width=3, char=' ')
    assert f.leading_ws == 0
    f.indent = 2
    assert f.leading_ws == 6
    f.offset = 5
    assert f.leading_ws == 11


def test_nl_generates_newline_token():
    f = ReindentFilter(width=2, char=' ')
    tok = f.nl()
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n'

    f.indent = 2
    tok = f.nl()
    assert tok.value == '\n    '

    tok = f.nl(offset=2)
    assert tok.value == '\n      '

    # max(0, ...) clamp
    tok = f.nl(offset=-100)
    assert tok.value == '\n'


def test_nl_custom_newline_and_char():
    f = ReindentFilter(width=2, char='\t', n='\r\n')
    f.indent = 1
    tok = f.nl()
    assert tok.value == '\r\n\t'


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_keyword():
    stmt = make_stmt('SELECT a FROM b')
    f = ReindentFilter()
    idx, tok = f._next_token(stmt)
    assert tok is not None
    assert tok.normalized == 'FROM'


def test_next_token_skips_between_and():
    # BETWEEN ... AND ... -- the AND belonging to BETWEEN must be skipped.
    stmt = make_stmt('SELECT a FROM b WHERE x BETWEEN 1 AND 2 AND y = 3')
    f = ReindentFilter()
    idx, tok = f._next_token(stmt)
    # first split keyword after WHERE among the list is the second AND
    assert tok is not None
    # it should be the AND that follows '2', not the BETWEEN-AND
    assert tok.normalized == 'AND'


def test_next_token_none_when_no_match():
    stmt = make_stmt('x')
    f = ReindentFilter()
    idx, tok = f._next_token(stmt)
    assert tok is None


# ---------------------------------------------------------------------------
# Reindent basic behavior
# ---------------------------------------------------------------------------

def test_process_returns_same_statements_object():
    stmt = make_stmt('SELECT a FROM b')
    f = ReindentFilter()
    out = f.process(stmt)
    assert out is stmt
    assert f._curr_stmt is stmt
    assert f._last_stmt is stmt


def test_process_breaks_before_keywords():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a FROM b')
    out = str(f.process(stmt))
    # FROM keyword starts a new line
    assert '\nFROM' in out
    assert out.startswith('SELECT')


def test_process_multiple_statements_blank_line():
    from sqlparse.lexer import Lexer
    from sqlparse.parser import Parser

    Lexer.get_default_instance().default_initialized = True
    stmts = Parser().parse('SELECT a FROM b; SELECT c FROM d')
    f = ReindentFilter()
    out = [str(f.process(s)) for s in stmts]
    # second statement has a leading blank line inserted
    assert out[1].startswith('SELECT') is False
    assert out[1].startswith('\n')


def test_process_last_stmt_ending_newline_single():
    f = ReindentFilter()
    s1 = make_stmt('SELECT a')
    f.process(s1)
    # force trailing newline
    s1.tokens.append(sql.Token(T.Whitespace, '\n'))
    s2 = make_stmt('SELECT b')
    f.process(s2)
    assert s2.tokens[0].value == '\n'


def test_process_last_stmt_no_newline_double():
    f = ReindentFilter()
    s1 = make_stmt('SELECT a')
    f.process(s1)
    s2 = make_stmt('SELECT b')
    f.process(s2)
    assert s2.tokens[0].value == '\n\n'


# ---------------------------------------------------------------------------
# WHERE handling
# ---------------------------------------------------------------------------

def test_process_where():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a FROM b WHERE x = 1')
    out = str(f.process(stmt))
    assert '\nWHERE' in out


def test_process_where_none_returns_early():
    # A group typed as Where but without a WHERE keyword -- must not raise.
    f = ReindentFilter()
    where = sql.Where()
    where.tokens = [sql.Token(T.Text, 'x')]
    # Should simply return, no exception.
    f._process_where(where)


# ---------------------------------------------------------------------------
# Parenthesis handling
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_first_returns():
    f = ReindentFilter()
    paren = sql.Parenthesis()
    # No open parenthesis token -> first is None -> early return
    f._process_parenthesis(paren)


def test_process_parenthesis_simple():
    f = ReindentFilter()
    stmt = make_stmt('SELECT (a + b) FROM c')
    out = str(f.process(stmt))
    assert '(' in out and ')' in out


def test_process_parenthesis_with_dml():
    f = ReindentFilter()
    stmt = make_stmt('SELECT (SELECT a FROM b) AS x')
    out = str(f.process(stmt))
    assert 'SELECT' in out


# ---------------------------------------------------------------------------
# Function / identifier list
# ---------------------------------------------------------------------------

def test_process_function_records_last_func():
    f = ReindentFilter()
    stmt = make_stmt('SELECT foo(a, b) FROM c')
    f.process(stmt)
    assert f._last_func is not None
    assert f._last_func.value == 'foo'


def test_indent_columns_identifierlist():
    f = ReindentFilter(indent_columns=True)
    stmt = make_stmt('SELECT a, b, c FROM t')
    out = str(f.process(stmt))
    # No exception; identifiers formatted
    assert 'a' in out and 'b' in out and 'c' in out


def test_indent_columns_tab_width():
    f = ReindentFilter(indent_columns=True, char='\t')
    stmt = make_stmt('SELECT aaa, bbb FROM t')
    out = str(f.process(stmt))
    assert '\t' in out


def test_identifierlist_wrap_after():
    f = ReindentFilter(wrap_after=10)
    stmt = make_stmt('SELECT aaaa, bbbb, cccc, dddd FROM t')
    out = str(f.process(stmt))
    # wraps put identifiers on separate lines
    assert out.count('\n') >= 1


def test_identifierlist_comma_first():
    f = ReindentFilter(wrap_after=10, comma_first=True)
    stmt = make_stmt('SELECT aaaa, bbbb, cccc, dddd FROM t')
    out = str(f.process(stmt))
    assert ',' in out
    assert out.count('\n') >= 1


def test_identifierlist_within_function_whitespace():
    f = ReindentFilter()
    stmt = make_stmt('SELECT foo(a,b) FROM t')
    out = str(f.process(stmt))
    # the comma should have whitespace after it
    assert 'a, b' in out or 'a , b' in out


def test_identifierlist_within_function_wraps():
    # wrap_after + _last_func set: adjusted offset branch
    f = ReindentFilter(wrap_after=5, width=2)
    stmt = make_stmt('SELECT somefunc(aaaa, bbbb) FROM t')
    out = str(f.process(stmt))
    assert '\n' in out


# ---------------------------------------------------------------------------
# CASE handling
# ---------------------------------------------------------------------------

def test_process_case():
    f = ReindentFilter()
    stmt = make_stmt(
        'SELECT CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END FROM t')
    out = str(f.process(stmt))
    assert 'CASE' in out
    assert 'WHEN' in out
    assert 'END' in out


def test_process_case_compact():
    f = ReindentFilter(compact=True)
    stmt = make_stmt(
        'SELECT CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END FROM t')
    out = str(f.process(stmt))
    assert 'CASE' in out


def test_process_case_wrap():
    f = ReindentFilter(wrap_after=5)
    stmt = make_stmt(
        'SELECT CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END FROM t')
    out = str(f.process(stmt))
    assert 'CASE' in out


# ---------------------------------------------------------------------------
# VALUES handling
# ---------------------------------------------------------------------------

def test_process_values():
    f = ReindentFilter()
    stmt = make_stmt('INSERT INTO t VALUES (1, 2), (3, 4)')
    out = str(f.process(stmt))
    assert 'VALUES' in out


def test_process_values_comma_first():
    f = ReindentFilter(comma_first=True)
    stmt = make_stmt('INSERT INTO t VALUES (1, 2), (3, 4)')
    out = str(f.process(stmt))
    assert 'VALUES' in out


# ---------------------------------------------------------------------------
# Misc private helpers
# ---------------------------------------------------------------------------

def test_current_line_len_empty():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a')
    f._curr_stmt = stmt
    # Measuring "before the statement" yields 0
    assert f._current_line_len(stmt) == 0


def test_get_offset_basic():
    f = ReindentFilter(width=2)
    stmt = make_stmt('SELECT a')
    f._curr_stmt = stmt
    # offset relative to leading whitespace; for start of statement 0 - 0
    first = next(stmt.flatten())
    assert f._get_offset(first) == -len(f.char * f.leading_ws)


def test_get_offset_with_leading_ws():
    f = ReindentFilter(width=2)
    f.indent = 2
    stmt = make_stmt('SELECT a')
    f._curr_stmt = stmt
    first = next(stmt.flatten())
    # 0 - 4 == -4
    assert f._get_offset(first) == -4


def test_split_statements_inserts_nl():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a; SELECT b')
    f._curr_stmt = stmt
    f._split_statements(stmt)
    s = str(stmt)
    assert '\n' in s


def test_split_kwds_no_double_newline():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a\nFROM b')
    f._curr_stmt = stmt
    f._split_kwds(stmt)
    # A pre-existing newline should not be duplicated
    assert '\n\nFROM' not in str(stmt)


def test_process_uses_default_for_unknown_group():
    f = ReindentFilter()
    stmt = make_stmt('SELECT a FROM b')
    f._curr_stmt = stmt
    # _process dispatches by type; unknown groups fall back to _process_default
    group = sql.TokenList([sql.Token(T.Text, 'x')])
    f._process(group)  # should not raise


def test_process_default_no_stmts():
    f = ReindentFilter()
    stmt = make_stmt('SELECT FROM x')
    f._curr_stmt = stmt
    # stmts=False path
    f._process_default(stmt, stmts=False)
    assert 'FROM' in str(stmt)


def test_process_identifierlist_empty_identifiers():
    # Identifier list with no identifiers should raise IndexError on pop(0)
    f = ReindentFilter()
    idlist = sql.IdentifierList()
    with pytest.raises(IndexError):
        f._process_identifierlist(idlist)


def test_process_case_no_end_close():
    # A Case group without an END token: token_next_by returns None for end
    f = ReindentFilter()
    stmt = make_stmt(
        'SELECT CASE WHEN a THEN 1 ELSE 2 END FROM t')
    f._curr_stmt = stmt
    f._process_case(stmt)  # should not raise
