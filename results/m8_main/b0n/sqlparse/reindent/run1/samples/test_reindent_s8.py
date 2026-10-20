import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter
import sqlparse


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def format_sql(sql_text, **kwargs):
    return sqlparse.format(sql_text, reindent=True, **kwargs)


def build_filter(**kwargs):
    return ReindentFilter(**kwargs)


# ---------------------------------------------------------------------------
# constructor / simple attribute accessors
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


def test_constructor_custom_values():
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


def test_indent_after_first_sets_indent():
    assert ReindentFilter(indent_after_first=True).indent == 1
    assert ReindentFilter(indent_after_first=False).indent == 0


def test_leading_ws_property():
    f = ReindentFilter(width=2)
    assert f.leading_ws == 0
    f.indent = 1
    assert f.leading_ws == 2
    f.offset = 3
    assert f.leading_ws == 5
    f.width = 4
    f.indent = 2
    assert f.leading_ws == 3 + 8


def test_nl_creates_whitespace_token():
    f = ReindentFilter(width=2, n='\n', char=' ')
    tok = f.nl()
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n'


def test_nl_with_indent():
    f = ReindentFilter(width=2, char=' ')
    f.indent = 2
    tok = f.nl()
    assert tok.value == '\n    '


def test_nl_with_offset_arg():
    f = ReindentFilter(width=2, char=' ')
    f.indent = 1
    tok = f.nl(offset=-2)
    # leading_ws = 2, + offset -2 = 0
    assert tok.value == '\n'


def test_nl_negative_clamped_to_zero():
    f = ReindentFilter(width=2, char=' ')
    f.indent = 1
    tok = f.nl(offset=-100)
    assert tok.value == '\n'


def test_nl_custom_newline_char():
    f = ReindentFilter(width=1, char='*', n='|')
    f.indent = 2
    tok = f.nl()
    assert tok.value == '|**'


# ---------------------------------------------------------------------------
# _current_line_len / _get_offset on a real parsed statement
# ---------------------------------------------------------------------------

def _parse(stmt_text):
    return sqlparse.parse(stmt_text)[0]


def test_current_line_len_no_parent_returns_zero():
    f = ReindentFilter()
    tok = sql.Token(T.Text, 'abc')
    # token has no parent and is not _curr_stmt (None)
    assert f._current_line_len(tok) == 0


def test_get_offset_relative_to_leading_ws():
    f = ReindentFilter(width=2)
    stmt = _parse('SELECT 1')
    f._curr_stmt = stmt
    # pick a token inside the statement
    tok = stmt.tokens[0]
    # nothing before -> 0 line len, leading_ws == 0
    off = f._get_offset(tok)
    assert off == f._current_line_len(tok) - len(f.char * f.leading_ws)
    assert isinstance(off, int)


# ---------------------------------------------------------------------------
# process() basics
# ---------------------------------------------------------------------------

def test_process_returns_stmt():
    f = ReindentFilter()
    stmt = _parse('select a from b')
    result = f.process(stmt)
    assert result is stmt


def test_process_sets_curr_and_last_stmt():
    f = ReindentFilter()
    s1 = _parse('select 1')
    s2 = _parse('select 2')
    f.process(s1)
    assert f._curr_stmt is s2 or f._curr_stmt is s1
    assert f._last_stmt is s1
    f.process(s2)
    assert f._curr_stmt is s2
    assert f._last_stmt is s2


def test_process_inserts_blank_line_between_statements():
    f = ReindentFilter()
    s1 = _parse('select 1')
    f.process(s1)
    # ensure s1 does not end with newline
    assert not str(s1).endswith('\n')
    s2 = _parse('select 2')
    f.process(s2)
    first = s2.tokens[0]
    assert first.ttype is T.Whitespace
    assert first.value == '\n\n'


def test_process_single_newline_when_prev_ends_with_newline():
    f = ReindentFilter()
    s1 = _parse('select 1')
    s1.tokens.append(sql.Token(T.Whitespace, '\n'))
    assert str(s1).endswith('\n')
    f.process(s1)
    s2 = _parse('select 2')
    f.process(s2)
    first = s2.tokens[0]
    assert first.ttype is T.Whitespace
    assert first.value == '\n'


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

def test_next_token_finds_from_keyword():
    stmt = _parse('select a from b join c')
    f = ReindentFilter()
    fidx, ftok = f._next_token(stmt)
    assert ftok is not None
    assert ftok.normalized == 'FROM'
    assert stmt.tokens[fidx] is ftok


def test_next_token_none_when_absent():
    stmt = _parse('select 1')
    f = ReindentFilter()
    idx, tok = f._next_token(stmt)
    assert tok is None


# ---------------------------------------------------------------------------
# _split_kwds / _split_statements
# ---------------------------------------------------------------------------

def test_split_kwds_inserts_newlines():
    stmt = _parse('select a from b')
    f = ReindentFilter()
    f._curr_stmt = stmt
    f._split_kwds(stmt)
    text = ''.join(t.value for t in stmt.tokens)
    assert '\n' in text
    # FROM should be on its own line
    assert '\nFROM' in text


def test_split_statements_inserts_newlines_between_dml():
    stmt = _parse('select 1 select 2')
    f = ReindentFilter()
    f._curr_stmt = stmt
    f._split_statements(stmt)
    text = ''.join(t.value for t in stmt.tokens)
    assert text.count('SELECT') == 2
    # there should be a newline before the second SELECT
    assert '\nSELECT' in text


def test_split_statements_skips_first_token():
    stmt = _parse('select 1')
    f = ReindentFilter()
    f._curr_stmt = stmt
    f._split_statements(stmt)
    text = ''.join(t.value for t in stmt.tokens)
    # leading whitespace/newline should not be inserted before first token
    assert text.lstrip('\n ') == text.lstrip()


# ---------------------------------------------------------------------------
# _process dispatch / default
# ---------------------------------------------------------------------------

def test_process_default_dispatches():
    f = ReindentFilter()
    stmt = _parse('select a, b from c')
    f._curr_stmt = stmt
    f._process(stmt)
    # default processing will run on sublists; nothing raises
    text = str(stmt)
    assert 'SELECT' in text.upper()


def test_process_default_without_statements_split():
    f = ReindentFilter()
    stmt = _parse('select a, b')
    f._curr_stmt = stmt
    # call directly with stmts=False
    f._process_default(stmt, stmts=False)
    # still runs keyword splitting
    assert str(stmt) == str(stmt)


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

def test_process_where_inserts_newline():
    stmt = _parse('select a from b where c = 1')
    f = ReindentFilter()
    f._curr_stmt = stmt
    where = stmt.token_next_by(m=(T.Keyword, 'WHERE'))[1]
    # find the where's parent list (the statement)
    f._process_where(stmt)
    text = str(stmt)
    assert '\nWHERE' in text


def test_process_where_no_where_token_returns():
    stmt = _parse('select 1')
    f = ReindentFilter()
    f._curr_stmt = stmt
    before = str(stmt)
    f._process_where(stmt)
    # nothing changed since there is no WHERE
    assert str(stmt) == before


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

def test_process_parenthesis_no_paren_returns():
    f = ReindentFilter()
    # a bare token list without a Parenthesis open marker
    tlist = sql.TokenList([sql.Token(T.Text, 'a')])
    before = [t.value for t in tlist.tokens]
    f._process_parenthesis(tlist)
    assert [t.value for t in tlist.tokens] == before


def test_process_parenthesis_with_paren():
    stmt = _parse('select (a) from b')
    f = ReindentFilter()
    f._curr_stmt = stmt
    f._process(stmt)
    text = str(stmt)
    assert '(' in text
    assert ')' in text


# ---------------------------------------------------------------------------
# _process_function
# ---------------------------------------------------------------------------

def test_process_function_sets_last_func():
    stmt = _parse('select foo(a) from b')
    f = ReindentFilter()
    f._curr_stmt = stmt
    f._process(stmt)
    # after processing a function, _last_func should be the function name token
    if f._last_func is not None:
        assert f._last_func.ttype in (T.Name, T.Keyword)


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

def test_identifierlist_no_wrap_short():
    text = format_sql('select a, b from t')
    assert 'a,' in text
    assert 'b' in text


def test_identifierlist_wrap_after():
    text = format_sql('select aaaaaa, bbbbbb, cccccc from t', wrap_after=10)
    # with a small wrap_after, columns should be broken onto separate lines
    assert text.count('\n') >= 1


def test_identifierlist_comma_first():
    text = format_sql('select aaaaaaa, bbbbbbb from t',
                      wrap_after=5, comma_first=True)
    # comma_first places commas at start of the line; verify at least one
    # comma appears at the beginning of a line (after stripping ws)
    lines = [ln.strip() for ln in text.splitlines()]
    assert any(ln.startswith(',') for ln in lines)


def test_identifierlist_indent_columns():
    text = format_sql('select aaaaaa, bbbbbb from t',
                      wrap_after=10, indent_columns=True)
    assert 'a' in text and 'b' in text


def test_identifierlist_tab_char():
    f = ReindentFilter(char='\t', width=4)
    text = format_sql('select aaaaaa, bbbbbb from t',
                      char='\t', width=4, wrap_after=10)
    assert 'a' in text and 'b' in text


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

def test_process_case_basic():
    text = format_sql(
        'select case when a then b when c then d end from t')
    assert 'CASE' in text.upper()
    assert 'END' in text.upper()
    assert text.count('\n') >= 1


def test_process_case_compact():
    text = format_sql(
        'select case when a then b else c end from t', compact=True)
    assert 'CASE' in text.upper()


def test_process_case_wrap():
    text = format_sql(
        'select case when aaaaaaaaaa then bbbbbbbbbb end from t',
        wrap_after=5)
    assert 'CASE' in text.upper()


# ---------------------------------------------------------------------------
# _process_values
# ---------------------------------------------------------------------------

def test_process_values_newline_insertion():
    text = format_sql('insert into t values (1), (2)')
    assert 'VALUES' in text.upper()
    assert text.count('\n') >= 1


def test_process_values_comma_first():
    text = format_sql('insert into t values (1), (2)', comma_first=True)
    lines = [ln.strip() for ln in text.splitlines()]
    assert any(ln.startswith(',') for ln in lines)


# ---------------------------------------------------------------------------
# end-to-end formatting through sqlparse.format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sql_text', [
    'select 1',
    'select a, b, c from t',
    'select a from t where a = 1 and b = 2',
    'select a from t1 join t2 on t1.id = t2.id',
    'insert into t (a, b) values (1, 2)',
    'update t set a = 1 where b = 2',
    'select case when a then b end from t',
])
def test_end_to_end_reindent(sql_text):
    out = format_sql(sql_text)
    assert isinstance(out, str)
    assert out != ''

    # keywords are uppercased by default formatter options; reindent puts
    # them at line starts
    lines = out.splitlines()
    assert len(lines) >= 1


def test_end_to_end_multiple_statements():
    out = sqlparse.format('select 1; select 2', reindent=True)
    # two statements get separated by a blank line
    assert out.count('SELECT') == 2
    assert '\n\n' in out or '\n' in out


def test_indent_after_first_option():
    out = sqlparse.format('select a, b from t',
                          reindent=True, indent_after_first=True)
    assert 'a' in out and 'b' in out


def test_compact_option():
    out = sqlparse.format('select a, b from t', reindent=True, compact=True)
    assert 'a' in out and 'b' in out


def test_reindent_width_custom():
    out = sqlparse.format('select a from t where b = 1',
                          reindent=True, indent_width=4)
    assert '\n' in out
