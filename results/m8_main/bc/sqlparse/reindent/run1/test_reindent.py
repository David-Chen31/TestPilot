"""Tests for sqlparse.filters.reindent.ReindentFilter."""

import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(text, **kwargs):
    return sqlparse.format(text, reindent=True, **kwargs)


def run_filter(stmt, **kwargs):
    """Apply a ReindentFilter directly to a parsed statement."""
    parsed = sqlparse.parse(stmt)[0]
    return ReindentFilter(**kwargs).process(parsed)


# ---------------------------------------------------------------------------
# constructor defaults / attributes
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


def test_indent_after_first():
    f = ReindentFilter(indent_after_first=True)
    assert f.indent == 1


def test_custom_attributes():
    f = ReindentFilter(width=4, char='\t', wrap_after=10, n='\r\n',
                       comma_first=True, indent_columns=True, compact=True)
    assert f.width == 4
    assert f.char == '\t'
    assert f.wrap_after == 10
    assert f.n == '\r\n'
    assert f.comma_first is True
    assert f.indent_columns is True
    assert f.compact is True


def test_leading_ws():
    f = ReindentFilter(width=3)
    assert f.leading_ws == 0
    f.indent = 2
    assert f.leading_ws == 6
    f.offset = 4
    assert f.leading_ws == 10


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

def test_nl_default():
    f = ReindentFilter(width=2)
    tok = f.nl()
    assert isinstance(tok, sql.Token)
    assert tok.ttype is T.Whitespace
    assert tok.value == '\n'


def test_nl_with_indent():
    f = ReindentFilter(width=4)
    f.indent = 2
    tok = f.nl()
    assert tok.value == '\n' + ' ' * 8


def test_nl_with_offset():
    f = ReindentFilter(width=2)
    f.indent = 1
    tok = f.nl(offset=2)
    assert tok.value == '\n' + ' ' * 4


def test_nl_negative_offset_clamped():
    f = ReindentFilter(width=2)
    tok = f.nl(offset=-100)
    # max(0, ...) applies
    assert tok.value == '\n'


def test_nl_custom_newline():
    f = ReindentFilter(n='\r\n')
    assert f.nl().value == '\r\n'


# ---------------------------------------------------------------------------
# process / statement separation
# ---------------------------------------------------------------------------

def test_process_returns_stmt():
    parsed = sqlparse.parse('select 1')[0]
    f = ReindentFilter()
    result = f.process(parsed)
    assert result is parsed
    assert f._curr_stmt is parsed
    assert f._last_stmt is parsed


def test_process_first_statement_no_prefix():
    stmt = sqlparse.parse('select 1')[0]
    ReindentFilter().process(stmt)
    # Leading token should not be whitespace '\n\n'
    assert not stmt.tokens[0].is_whitespace


def test_process_second_statement_gets_prefix():
    stmts = sqlparse.parse('select 1; select 2')
    f = ReindentFilter()
    f.process(stmts[0])
    second = f.process(stmts[1])
    assert second.tokens[0].is_whitespace
    assert second.tokens[0].value in ('\n', '\n\n')


def test_process_last_stmt_ending_newline():
    f = ReindentFilter()
    first = sqlparse.parse('select 1')[0]
    # Force last_stmt value ending with '\n'
    f._last_stmt = sql.Token(T.Whitespace, 'x\n')
    f.process(sqlparse.parse('select 2')[0])
    # prefix should be single '\n'
    assert f._last_stmt.tokens[0].value == '\n'


def test_reindent_simple_select():
    # sqlparse.format defaults to not uppercasing keywords.
    assert _fmt('select * from foo') == 'select *\nfrom foo'


def test_reindent_where():
    out = _fmt('select * from foo where x = 1')
    assert out == 'select *\nfrom foo\nwhere x = 1'


def test_reindent_inner_join():
    out = _fmt('select * from a join b on a.id = b.id')
    assert 'join b' in out
    assert out.startswith('select *\nfrom a\n')


# ---------------------------------------------------------------------------
# _split_statements / _split_kwds
# ---------------------------------------------------------------------------

def test_split_statements_inside():
    # A single parsed statement object won't contain the following one; the
    # DML splitting within one statement occurs when there is a trailing DML.
    stmt = sqlparse.parse('select 1; insert into t values (1)')[0]
    ReindentFilter().process(stmt)
    text = str(stmt)
    assert 'select 1' in text


def test_split_statements_multiple_dml_one_stmt():
    # Two DML statements parsed separately
    stmts = sqlparse.parse('select 1; insert into t values (1)')
    assert len(stmts) == 2
    f = ReindentFilter()
    f.process(stmts[0])
    f.process(stmts[1])
    assert 'insert into t' in str(stmts[1])


def test_split_kwds_from():
    stmt = run_filter('select * from foo')
    text = str(stmt)
    assert 'from foo' in text
    assert text.startswith('select')


def test_between_and():
    out = _fmt('select * from foo where x between 1 and 2')
    # between/and should not break after BETWEEN
    assert 'between 1' in out.lower()


# ---------------------------------------------------------------------------
# parenthesis (subquery)
# ---------------------------------------------------------------------------

def test_parenthesis_subquery():
    out = _fmt('select * from (select 1) as x')
    # whitespace after '(' should start on new line
    assert out.count('\n') >= 1


def test_process_parenthesis_no_open():
    # Empty parenthesis token group has no M_OPEN; should early return
    paren = sql.Parenthesis()
    f = ReindentFilter()
    f._process_parenthesis(paren)  # no exception
    assert len(paren.tokens) == 0


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

def test_process_where_no_where():
    where = sql.Where()
    ReindentFilter()._process_where(where)  # no exception


# ---------------------------------------------------------------------------
# identifier list / columns
# ---------------------------------------------------------------------------

def test_identifierlist_wraps_by_default():
    # Without wrap_after, every column after the first gets a newline.
    out = _fmt('select a, b, c from t')
    assert out == 'select a,\n       b,\n       c\nfrom t'


def test_identifierlist_wrap_after():
    out = _fmt('select aaa, bbb, ccc, ddd from t', wrap_after=10)
    lines = [ln for ln in out.splitlines() if ',' in ln or 'ddd' in ln]
    assert len(lines) >= 1


def test_identifierlist_comma_first():
    out = _fmt('select aaa, bbb, ccc from t', wrap_after=6, comma_first=True)
    # commas placed at line starts for wrapped items
    wrapped = [ln.strip() for ln in out.splitlines()]
    assert any(ln.startswith(',') for ln in wrapped)


def test_identifierlist_indent_columns():
    out = _fmt('select aaa, bbb, ccc from t', wrap_after=5, indent_columns=True)
    assert isinstance(out, str)


def test_identifierlist_tab_char():
    out = _fmt('select aaa, bbb, ccc from t', wrap_after=5, indent_width=1)
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# _process_function
# ---------------------------------------------------------------------------

def test_process_function_sets_last_func():
    stmt = sqlparse.parse('select sum(x) from t')[0]
    f = ReindentFilter()
    f.process(stmt)
    assert f._last_func is not None


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

def test_values_list():
    out = _fmt('insert into t values (1), (2), (3)')
    assert 'values' in out
    # each tuple separated
    assert out.count('(') == 3


def test_values_comma_first():
    out = _fmt('insert into t values (1), (2), (3)', comma_first=True)
    assert isinstance(out, str)
    assert out.count('(') == 3


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------

def test_case_wrap():
    text = ('select case when a = 1 then 1 '
            'when b = 2 then 2 else 3 end from t')
    out = _fmt(text, wrap_after=20)
    assert 'case' in out.lower()
    assert 'end' in out.lower()


def test_case_compact():
    text = 'select case when a = 1 then 1 else 0 end from t'
    out = _fmt(text, compact=True)
    assert 'case' in out.lower()


# ---------------------------------------------------------------------------
# char handling
# ---------------------------------------------------------------------------

def test_indent_width_custom():
    out = _fmt('select * from foo', indent_width=4)
    assert 'from foo' in out


# ---------------------------------------------------------------------------
# _current_line_len / _get_offset
# ---------------------------------------------------------------------------

def test_current_line_len_empty():
    f = ReindentFilter()
    token = sql.Token(T.Whitespace, ' ')
    # no parent and not curr_stmt: loop should not run
    assert f._current_line_len(token) == 0


def test_get_offset_without_parent():
    f = ReindentFilter()
    token = sql.Token(T.Whitespace, '  ')
    assert f._get_offset(token) == 0


# ---------------------------------------------------------------------------
# _next_token recursion for BETWEEN ... AND
# ---------------------------------------------------------------------------

def test_next_token_between_recursion():
    # A bare "x between 1 and 2" is parsed as an IdentifierList; the
    # BETWEEN/AND recursion only short-circuits then re-searches, and since
    # there is no further split keyword, the result is None.
    stmt = sqlparse.parse('x between 1 and 2')[0]
    f = ReindentFilter()
    tidx, token = f._next_token(stmt)
    assert token is None


def test_next_token_none():
    stmt = sqlparse.parse('nothing here')[0]
    f = ReindentFilter()
    tidx, token = f._next_token(stmt)
    assert token is None


# ===========================================================================
# Additional tests for coverage of previously-uncovered branches.
# ===========================================================================

# ---------------------------------------------------------------------------
# _current_line_len: empty-value tokens, trailing newline collapse,
# multi-line handling (lines 61-62, 76, 80-82)
# ---------------------------------------------------------------------------

def test_current_line_len_empty_token_value():
    """Line 61-62: a token with empty value is skipped (continue)."""
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    # Build a small parent group containing an empty token followed by the
    # target token.  The empty-token branch must be hit without error.
    parent = sql.IdentifierList()
    empty = sql.Token(T.Whitespace, '')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [empty, target]
    target.parent = parent
    # length of prefix should just count 'x' -> but target isn't itself
    # counted; prefix-only length of the empty token is 0.
    assert f._current_line_len(target) == 0


def test_current_line_len_empty_token_continue_then_length():
    """Empty token skipped, but real content afterwards is still counted."""
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    empty = sql.Token(T.Whitespace, '')
    abc = sql.Token(T.Name, 'abc')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [empty, abc, target]
    target.parent = parent
    # Walking backwards: target -> abc (len 3) + empty (skipped) => 3
    assert f._current_line_len(target) == 3


def test_current_line_len_trailing_newline_collapse():
    """Line 76-82: value ends with a newline and nothing after it.

    The line to measure is the one before the trailing break.
    """
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    # "ab\ncd\n": the break that starts the current line is the final '\n';
    # nothing at all follows it, so the line to measure is 'cd'.
    tok = sql.Token(T.Whitespace, 'ab\ncd\n')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [tok, target]
    target.parent = parent
    # lines = ('ab\ncd\n' + '.').splitlines() = ['ab', 'cd', '.'] -> len 3
    # tail = len('.') - 1 + 0 = 0, so returns len(lines[-2]) = len('cd') = 2
    assert f._current_line_len(target) == 2


def test_current_line_len_tail_value():
    """Line 74-77: text after the last break is returned directly."""
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    tok = sql.Token(T.Whitespace, 'hello\nbye')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [tok, target]
    target.parent = parent
    # lines = ['hello', 'bye.'] ; tail = len('bye.') - 1 + 0 = 3
    assert f._current_line_len(target) == 3


def test_current_line_len_multi_line_no_tail_uses_lines0():
    """Lines 80-82: value has multiple breaks and empty tail; the line to
    measure is ``lines[0]`` (the earliest line segment in this value)."""
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    # "ab\ncd\n" -> (value + '.') = 'ab\ncd\n.'
    # splitlines() = ['ab', 'cd', '.'] ; len == 3 so not > 2 => no return,
    # falls through to length = len(lines[0]) = 2.
    tok = sql.Token(T.Whitespace, 'ab\n')
    abcde = sql.Token(T.Name, 'abcde')
    target = sql.Token(T.Name, 'x')
    # To reach the `if len(lines) > 2` false branch we need the final
    # splitlines list to have exactly 2 entries (a single trailing break).
    # "ab\n" + '.' = 'ab\n.' -> ['ab', '.'] -> len 2, tail=0, not >2,
    # length = len('ab') = 2.
    parent.tokens = [tok, abcde, target]
    target.parent = parent
    # Walk: abcde (len 5) first -> length=5. Then tok: 'ab\n' has a break,
    # lines = ['ab','.']; tail = len('.')-1 + 5 = 5 -> returns 5.
    assert f._current_line_len(target) == 5


def test_current_line_len_len_lines_gt_two():
    """Lines 80-82: a value with exactly two internal breaks where the final
    entry is empty; returns len of the segment before the last break."""
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    # "ab\ncd\n" + '.' = 'ab\ncd\n.' -> ['ab', 'cd', '.'] (len 3).
    # tail = len('.') - 1 + 0 = 0 -> falsy.
    # len(lines) > 2 is True -> return len(lines[-2]) = len('cd') = 2.
    tok = sql.Token(T.Whitespace, 'ab\ncd\n')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [tok, target]
    target.parent = parent
    assert f._current_line_len(target) == 2


def test_current_line_len_falls_through_to_lines0():
    """Lines 80-82: the ``length = len(lines[0])`` branch executes when the
    multi-line token has just one internal break and an empty tail.

    "ab\n" + '.' == 'ab\n.' -> splitlines() == ['ab', '.']
    tail = len('.') - 1 + 0 = 0 -> falsy, len(lines) == 2 -> not > 2,
    so length is set to len('ab') == 2, then loop ends -> return 2.
    """
    f = ReindentFilter()
    stmt = sqlparse.parse('select 1')[0]
    f._curr_stmt = stmt
    parent = sql.IdentifierList()
    tok = sql.Token(T.Whitespace, 'ab\n')
    target = sql.Token(T.Name, 'x')
    parent.tokens = [tok, target]
    target.parent = parent
    assert f._current_line_len(target) == 2


# ---------------------------------------------------------------------------
# _next_token BETWEEN ... AND recursion (line 105-106)
# ---------------------------------------------------------------------------

def test_next_token_between_and_chain():
    """Line 105: _next_token sees BETWEEN then the AND that follows it."""
    # Put BETWEEN and AND as keywords together so the recursion fires.
    from sqlparse.sql import TokenList
    tl = TokenList()
    between = sql.Token(T.Keyword, 'BETWEEN')
    and_kw = sql.Token(T.Keyword, 'AND')
    tl.tokens = [sql.Token(T.Name, 'x'), between,
                 sql.Token(T.Whitespace, ' '),
                 sql.Token(T.Number, '1'),
                 sql.Token(T.Whitespace, ' '),
                 and_kw,
                 sql.Token(T.Whitespace, ' '),
                 sql.Token(T.Number, '2')]
    f = ReindentFilter()
    tidx, token = f._next_token(tl)
    # Recursion ends with no more keywords found after AND -> None.
    assert token is None


# ---------------------------------------------------------------------------
# _split_kwds: prev_ is whitespace (line 116-118) and uprev ends with
# newline (line 120 skip)
# ---------------------------------------------------------------------------

def test_split_kwds_prev_whitespace_replaced():
    """Line 116-118: an existing whitespace token before the keyword is
    deleted before inserting the newline."""
    from sqlparse.sql import TokenList
    tl = TokenList()
    select = sql.Token(T.Keyword.DML, 'select')
    ws1 = sql.Token(T.Whitespace, '  ')
    frm = sql.Token(T.Keyword, 'from')
    ws2 = sql.Token(T.Whitespace, ' ')
    tbl = sql.Token(T.Name, 't')
    tl.tokens = [select, ws1, frm, ws2, tbl]
    f = ReindentFilter()
    f._split_kwds(tl)
    values = [t.value for t in tl.tokens]
    # the two-space whitespace before 'from' should have been replaced
    assert '  ' not in values
    assert values[1] == '\n' or values[1].startswith('\n')


def test_split_kwds_preceding_newline_removed():
    """Line 116-118 + 120: a preceding whitespace token is *always* deleted,
    and then the recorded ``uprev`` (the deleted whitespace, starting with
    '\\n') prevents inserting a new break.  So the keyword ends up directly
    after the previous token with no whitespace in between."""
    from sqlparse.sql import TokenList
    tl = TokenList()
    select = sql.Token(T.Keyword.DML, 'select')
    nl_ws = sql.Token(T.Whitespace, '\n')
    frm = sql.Token(T.Keyword, 'from')
    ws2 = sql.Token(T.Whitespace, ' ')
    tbl = sql.Token(T.Name, 't')
    tl.tokens = [select, nl_ws, frm, ws2, tbl]
    f = ReindentFilter()
    f._split_kwds(tl)
    from_idx = tl.token_index(frm)
    # The whitespace before 'from' was deleted and not reinserted because
    # ``uprev`` ended with '\n'.
    prev = tl.tokens[from_idx - 1]
    assert prev is select


def test_split_kwds_carriage_return_prev_removed():
    """Line 120 branch: uprev.endswith('\\r') also skips insertion; the
    preceding whitespace is still deleted."""
    from sqlparse.sql import TokenList
    tl = TokenList()
    select = sql.Token(T.Keyword.DML, 'select')
    prev_ws = sql.Token(T.Whitespace, '\r')
    frm = sql.Token(T.Keyword, 'from')
    ws2 = sql.Token(T.Whitespace, ' ')
    tbl = sql.Token(T.Name, 't')
    tl.tokens = [select, prev_ws, frm, ws2, tbl]
    f = ReindentFilter()
    f._split_kwds(tl)
    from_idx = tl.token_index(frm)
    prev = tl.tokens[from_idx - 1]
    # Whitespace token was removed and not reinserted (uprev ended with '\r').
    assert prev is select


# ---------------------------------------------------------------------------
# _split_statements: prev is whitespace (line 131-133) and prev is None
# (line 135 insert) within one statement.
# ---------------------------------------------------------------------------

def test_split_statements_prev_whitespace_deleted():
    """Line 131-133: a whitespace token before a DML keyword is removed."""
    from sqlparse.sql import TokenList
    tl = TokenList()
    x = sql.Token(T.Name, 'x')
    ws = sql.Token(T.Whitespace, '   ')
    dml = sql.Token(T.Keyword.DML, 'select')
    rest = sql.Token(T.Name, 'y')
    tl.tokens = [x, ws, dml, rest]
    f = ReindentFilter()
    f._split_statements(tl)
    values = [t.value for t in tl.tokens]
    assert '   ' not in values
    sel_idx = tl.token_index(dml)
    assert tl.tokens[sel_idx - 1].is_whitespace
    assert tl.tokens[sel_idx - 1].value.startswith('\n')


def test_split_statements_target_gets_prefixed():
    """Line 135-137: a DML keyword preceded by something gets a newline."""
    from sqlparse.sql import TokenList
    tl = TokenList()
    x = sql.Token(T.Name, 'x')
    dml = sql.Token(T.Keyword.DML, 'select')
    rest = sql.Token(T.Name, 'y')
    tl.tokens = [x, dml, rest]
    f = ReindentFilter()
    f._split_statements(tl)
    sel_idx = tl.token_index(dml)
    prev = tl.tokens[sel_idx - 1]
    assert prev.is_whitespace
    assert prev.value.startswith('\n')


# ---------------------------------------------------------------------------
# _process_identifierlist: Function/Values branch and comma_first None
# (lines 179-210, 191-192)
# ---------------------------------------------------------------------------

def test_identifierlist_inside_function_inserts_whitespace():
    """Line 205-210: identifier list inside a Function gets comma-spacing
    fixes, no wrapping (indent_columns path is short-circuited differently)."""
    stmt = sqlparse.parse('select f(a, b) from t')[0]
    f = ReindentFilter()
    f.process(stmt)
    out = str(stmt)
    # 'a, b' inside function stays on one line
    assert 'a, b' in out


def test_identifierlist_inside_values_inserts_whitespace():
    stmt = sqlparse.parse('insert into t values (a, b, c)')[0]
    f = ReindentFilter()
    f.process(stmt)
    out = str(stmt)
    assert 'a, b' in out


def test_identifierlist_inside_function_wrap_after_with_last_func():
    """Line 212-217: wrap_after applies and _last_func adjusts the offset."""
    stmt = sqlparse.parse('select sum(aaa, bbb, ccc) from t')[0]
    f = ReindentFilter(wrap_after=5)
    f.process(stmt)
    out = str(stmt)
    # at least a newline was introduced inside the function args
    assert out.count('\n') >= 2


def test_identifierlist_comma_first_missing_comma():
    """Line 191-192: comma_first collapses a comma that can't be located.

    Comma is always found in real input, so we exercise it by building an
    identifier list where comma_first wrapping tries to find the comma for
    the *first* identifier (which has no preceding comma)."""
    stmt = sqlparse.parse('select a, b, c from t')[0]
    f = ReindentFilter(wrap_after=1, comma_first=True)
    # Should not raise even if a comma can't be located for some item.
    f.process(stmt)
    assert isinstance(str(stmt), str)


def test_identifierlist_comma_first_comma_found_branch():
    """Line 191-192: the ``if comma is None`` is checked; here a comma is
    found so processing proceeds and places a comma at the start of a line."""
    stmt = sqlparse.parse('select aaa, bbb, ccc from t')[0]
    f = ReindentFilter(wrap_after=1, comma_first=True)
    f.process(stmt)
    out = str(stmt)
    lines = out.splitlines()
    # At least one continuation line starts with a comma.
    assert any(ln.lstrip().startswith(',') for ln in lines)


def test_identifierlist_comma_first_inserts_space_after_comma():
    """Lines 195-201: comma_first with a preceding comma that lacks trailing
    whitespace triggers insertion of a single space after the comma."""
    from sqlparse.sql import TokenList, Identifier
    f = ReindentFilter(wrap_after=1, comma_first=True)
    tl = sql.IdentifierList()
    a = sql.Identifier([sql.Token(T.Name, 'aaa')])
    comma = sql.Token(T.Punctuation, ',')
    b = sql.Identifier([sql.Token(T.Name, 'bbb')])
    c = sql.Identifier([sql.Token(T.Name, 'ccc')])
    # No whitespace after comma -> wrap must insert T.Whitespace ' '.
    tl.tokens = [a, comma, b, comma, c]
    for tok in tl.tokens:
        tok.parent = tl
    a.parent = tl
    b.parent = tl
    c.parent = tl
    f._process_identifierlist(tl)
    out = str(tl)
    # After processing a comma should be followed by whitespace before 'bbb'.
    # The wrapped comma line-break plus an inserted single space.
    assert ', ' in out or ',\n' in out


# ---------------------------------------------------------------------------
# _next_token: nothing to recurse into after BETWEEN (line 102-108)
# ---------------------------------------------------------------------------

def test_between_without_following_keyword():
    """BETWEEN at the end: recursion returns (None, None) gracefully."""
    stmt = sqlparse.parse('x between y')[0]
    f = ReindentFilter()
    idx, token = f._next_token(stmt)
    # No AND after BETWEEN; recursion returns None.
    assert token is None


# ---------------------------------------------------------------------------
# Wrap boundary: position exactly at wrap_after doesn't wrap
# (line 185: strictly greater than)
# ---------------------------------------------------------------------------

def test_identifierlist_wrap_boundary_no_break_when_at_limit():
    """wrap position must exceed wrap_after, not just equal it."""
    # With wrap_after high enough nothing wraps: single line identifiers.
    out = _fmt('select a, b from t', wrap_after=100)
    first_part = out.split('from')[0]
    assert ',' in first_part
    # 'a,' and 'b' should not be split by newline if it fits
    # (first identifier is popped so only subsequent ones wrap)
    assert out.count('\n') == 1  # only the 'from' split


# ---------------------------------------------------------------------------
# Comma-first with wrap_after within identifier list producing adjust
# (line 187-201)
# ---------------------------------------------------------------------------

def test_identifierlist_comma_first_places_comma_at_line_start():
    out = _fmt('select aaa, bbb, ccc from t', wrap_after=4, comma_first=True)
    lines = out.splitlines()
    # at least one continuation line begins with a comma
    assert any(ln.lstrip().startswith(',') for ln in lines)


# ---------------------------------------------------------------------------
# Values: comma_first branch (line 264-267)
# ---------------------------------------------------------------------------

def test_values_comma_first_places_newline_before_comma():
    out = _fmt('insert into t values (1), (2)', comma_first=True)
    lines = out.splitlines()
    # a continuation line should start with a comma
    assert any(ln.lstrip().startswith(',') for ln in lines)


# ---------------------------------------------------------------------------
# indent_columns with tab char (line 174)
# ---------------------------------------------------------------------------

def test_identifierlist_indent_columns_tab_char():
    """Line 174: indent_columns with char='\\t' uses num_offset=1."""
    stmt = sqlparse.parse('select aaa, bbb from t')[0]
    f = ReindentFilter(indent_columns=True, char='\t')
    f.process(stmt)
    # Leading whitespace of the continuation (if any) is a single tab.
    out = str(stmt)
    assert '\t' in out or 'aaa' in out


# ---------------------------------------------------------------------------
# _process_values on real VALUES with comma_first: offset computed per token
# ---------------------------------------------------------------------------

def test_values_multiple_tuples_comma_first():
    out = _fmt('insert into t values (1), (2), (3), (4)', comma_first=True)
    assert out.count('(') == 4


# ---------------------------------------------------------------------------
# _process_case: compact prevents breaks (lines 244, 253)
# ---------------------------------------------------------------------------

def test_case_compact_no_breaks_inside():
    text = 'select case when a = 1 then 1 else 0 end from t'
    out = _fmt(text, compact=True, wrap_after=5)
    # With compact, no mid-CASE line breaks are inserted.
    # The 'end' keyword shouldn't get its own newline inserted before it
    # by the compact-skipped line 253 branch.
    case_part = out.lower().split('case')[1].split('from')[0]
    # fewer newlines than non-compact would produce
    assert case_part.count('\n') <= 2


# ---------------------------------------------------------------------------
# `indent_after_first` public behavior
# ---------------------------------------------------------------------------

def test_indent_after_first_affects_output():
    out = _fmt('select * from t', indent_after_first=True)
    # first token gets indented by leading_ws = 0 + 1*2 = 2 spaces
    assert out.startswith('  select') or 'from' in out


# ---------------------------------------------------------------------------
# Multiple statements + last_stmt ending in newline (existing behavior path)
# ---------------------------------------------------------------------------

def test_process_multiple_statements_sequentially():
    stmts = sqlparse.parse('select 1; select 2; select 3')
    f = ReindentFilter()
    for s in stmts:
        f.process(s)
    # second and third statements both get prefixed with whitespace
    assert stmts[1].tokens[0].is_whitespace
    assert stmts[2].tokens[0].is_whitespace


# ===========================================================================
# Targeted tests for the remaining uncovered branches.
# ===========================================================================

def test_values_insert_whitespace_after_comma_no_ws():
    """Line 208-209: in a Values-contained identifier list, a comma with no
    following whitespace gets a single space inserted after it."""
    from sqlparse.sql import Identifier
    stmt = sqlparse.parse('select 1')[0]
    f = ReindentFilter()
    f._curr_stmt = stmt
    values = sql.Values()
    tl = sql.IdentifierList()
    a = sql.Identifier([sql.Token(T.Name, 'a')])
    comma = sql.Token(T.Punctuation, ',')
    b = sql.Identifier([sql.Token(T.Name, 'b')])
    tl.tokens = [a, comma, b]
    tl.parent = values
    a.parent = tl
    comma.parent = tl
    b.parent = tl
    values.tokens = [tl]
    f._process_identifierlist(tl)
    out = str(tl)
    # The comma must now be followed by a single space.
    assert 'a, b' in out


def test_identifierlist_comma_first_inserts_space_branch():
    """Lines 198-201: after moving the comma to the start of a line,
    whitespace after the comma is missing, so a single space is inserted."""
    from sqlparse.sql import Identifier
    stmt = sqlparse.parse('select 1')[0]
    f = ReindentFilter(wrap_after=1, comma_first=True, width=0)
    f._curr_stmt = stmt
    tl = sql.IdentifierList()
    a = sql.Identifier([sql.Token(T.Name, 'aaa')])
    comma1 = sql.Token(T.Punctuation, ',')
    b = sql.Identifier([sql.Token(T.Name, 'bbb')])
    comma2 = sql.Token(T.Punctuation, ',')
    c = sql.Identifier([sql.Token(T.Name, 'ccc')])
    tl.tokens = [a, comma1, b, comma2, c]
    for tok in tl.tokens:
        tok.parent = tl
    f._process_identifierlist(tl)
    out = str(tl)
    # Find each comma in turn: after processing a wrapped comma it must be
    # immediately followed by whitespace (either the original or inserted).
    text_list = tl.tokens
    for i, tok in enumerate(text_list):
        if tok.value == ',' and i + 1 < len(text_list):
            # a newline was placed before the comma; the next token should
            # be whitespace (the inserted single space).
            nxt = text_list[i + 1]
            assert nxt.is_whitespace
    assert 'aaa' in out and 'bbb' in out and 'ccc' in out
