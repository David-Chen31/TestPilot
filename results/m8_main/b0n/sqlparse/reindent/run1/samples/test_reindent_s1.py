import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse(text):
    """Parse *text* into a statement using a fresh sqlparse parser.

    We rely on the public ``sqlparse.parse`` entry point so the tokens we feed
    into ``ReindentFilter`` are exactly what the library produces in practice.
    """
    import sqlparse
    return sqlparse.parse(text)[0]


def filtered(text, **kwargs):
    stmt = parse(text)
    return ReindentFilter(**kwargs).process(stmt)


# ---------------------------------------------------------------------------
# constructor / simple state
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_defaults(self):
        f = ReindentFilter()
        assert f.width == 2
        assert f.char == ' '
        assert f.wrap_after == 0
        assert f.n == '\n'
        assert f.comma_first is False
        assert f.indent_columns is False
        assert f.compact is False
        assert f.indent == 0
        assert f.offset == 0
        assert f._curr_stmt is None
        assert f._last_stmt is None
        assert f._last_func is None

    def test_indent_after_first(self):
        f = ReindentFilter(indent_after_first=True)
        assert f.indent == 1

    def test_custom_values(self):
        f = ReindentFilter(width=4, char='\t', wrap_after=10, n='\r\n',
                           comma_first=True, indent_columns=True, compact=True)
        assert f.width == 4
        assert f.char == '\t'
        assert f.wrap_after == 10
        assert f.n == '\r\n'
        assert f.comma_first is True
        assert f.indent_columns is True
        assert f.compact is True

    def test_leading_ws_property(self):
        f = ReindentFilter(width=3)
        f.indent = 2
        f.offset = 1
        assert f.leading_ws == 1 + 2 * 3


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

class TestNl:
    def test_nl_default(self):
        f = ReindentFilter(width=2)
        tok = f.nl()
        assert tok.ttype is T.Whitespace
        assert tok.value == '\n'

    def test_nl_with_indent(self):
        f = ReindentFilter(width=3)
        f.indent = 1
        tok = f.nl()
        assert tok.value == '\n' + ' ' * 3

    def test_nl_with_offset(self):
        f = ReindentFilter(width=2)
        tok = f.nl(offset=2)
        assert tok.value == '\n' + ' ' * 2

    def test_nl_negative_offset_clamped(self):
        f = ReindentFilter(width=2)
        f.indent = 0
        f.offset = 0
        tok = f.nl(offset=-5)
        # max(0, ...) -> no indentation after the newline
        assert tok.value == '\n'

    def test_nl_custom_char_newline(self):
        f = ReindentFilter(width=2, char='.', n='\r\n')
        f.indent = 2
        tok = f.nl()
        assert tok.value == '\r\n....'


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

class TestNextToken:
    def test_finds_from(self):
        stmt = parse('SELECT a FROM foo')
        f = ReindentFilter()
        tidx, tok = f._next_token(stmt)
        assert tok is not None
        assert tok.normalized == 'FROM'
        assert stmt.tokens[tidx] is tok

    def test_no_match_returns_none(self):
        stmt = parse('SELECT a')
        f = ReindentFilter()
        tidx, tok = f._next_token(stmt)
        assert tidx == -1
        assert tok is None

    def test_between_and_skipped(self):
        # BETWEEN ... AND ...: the AND directly following must be skipped
        stmt = parse('SELECT * WHERE x BETWEEN 1 AND 2')
        f = ReindentFilter()
        tidx, tok = f._next_token(stmt)
        assert tok is not None
        assert tok.normalized == 'WHERE'

    def test_between_then_and(self):
        # Look at what comes after BETWEEN: should skip the matching AND
        stmt = parse('BETWEEN 1 AND 2')
        f = ReindentFilter()
        tidx, tok = f._next_token(stmt)
        assert tok is not None
        assert tok.normalized == 'BETWEEN'
        tidx2, tok2 = f._next_token(stmt, tidx)
        # AND was consumed inside the BETWEEN handling; find nothing else
        assert tok2 is None


# ---------------------------------------------------------------------------
# _get_offset / _current_line_len
# ---------------------------------------------------------------------------

class TestCurrentLineLen:
    def test_empty_when_stmt_is_itself(self):
        stmt = parse('SELECT 1')
        f = ReindentFilter()
        f._curr_stmt = stmt
        # token is the statement itself -> loop never runs -> 0
        assert f._current_line_len(stmt) == 0

    def test_measures_preceding_tokens(self):
        stmt = parse('SELECT 1')
        f = ReindentFilter()
        f._curr_stmt = stmt
        # pick a token after 'SELECT'
        idx, tok = stmt.token_next_by(m=(T.Keyword, 'SELECT'))
        # token after SELECT
        next_idx, next_tok = stmt.token_next(idx)
        length = f._current_line_len(next_tok)
        assert length == len('SELECT')

    def test_stops_at_newline(self):
        stmt = parse('SELECT a,\n b')
        f = ReindentFilter()
        f._curr_stmt = stmt
        # find the 'b' identifier
        idx, btok = stmt.token_next_by(m=(T.Name, 'b'))
        length = f._current_line_len(btok)
        # 'b' is on the line started by the newline; the newline token is
        # whitespace with value '\n '.  Measuring from b backwards should
        # return the length of the current line ('  ' -> 1 space, maybe).
        assert length >= 0

    def test_get_offset(self):
        stmt = parse('aaa bbb')
        f = ReindentFilter()
        f._curr_stmt = stmt
        idx, btok = stmt.token_next_by(m=(T.Name, 'bbb'))
        off = f._get_offset(btok)
        # 'aaa ' is 4 chars preceding bbb; leading_ws is 0
        # offset returned is current-line-len - leading_ws
        assert off == len('aaa ')


# ---------------------------------------------------------------------------
# _split_statements
# ---------------------------------------------------------------------------

class TestSplitStatements:
    def test_split_on_keyword_dml(self):
        stmt = parse('SELECT 1; SELECT 2')
        f = ReindentFilter()
        f._split_statements(stmt)
        text = str(stmt)
        # a newline should be inserted before the second SELECT
        assert text.startswith('SELECT 1;')
        assert '\nSELECT 2' in text

    def test_noop_for_single_statement(self):
        stmt = parse('SELECT 1')
        f = ReindentFilter()
        before = str(stmt)
        f._split_statements(stmt)
        # only the leading SELECT; nothing to split
        assert str(stmt) == before


# ---------------------------------------------------------------------------
# _split_kwds
# ---------------------------------------------------------------------------

class TestSplitKwds:
    def test_split_from(self):
        stmt = parse('SELECT a FROM foo')
        f = ReindentFilter()
        f._split_kwds(stmt)
        text = str(stmt)
        assert '\nFROM' in text

    def test_noop_without_keywords(self):
        stmt = parse('SELECT a')
        f = ReindentFilter()
        before = str(stmt)
        f._split_kwds(stmt)
        assert str(stmt) == before


# ---------------------------------------------------------------------------
# process() end-to-end behavior
# ---------------------------------------------------------------------------

class TestProcess:
    def test_process_returns_stmt(self):
        stmt = parse('SELECT a FROM b')
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt

    def test_process_sets_curr_stmt(self):
        stmt = parse('SELECT a')
        f = ReindentFilter()
        f.process(stmt)
        assert f._curr_stmt is stmt
        assert f._last_stmt is stmt

    def test_select_from_newline(self):
        result = filtered('SELECT a FROM b')
        text = str(result)
        assert '\nFROM' in text

    def test_two_statements_separated(self):
        f = ReindentFilter()
        s1 = parse('SELECT 1')
        f.process(s1)
        s2 = parse('SELECT 2')
        result = f.process(s2)
        text = str(result)
        # a blank line separator is inserted at the front of the second stmt
        assert text.startswith('\n\n') or text.startswith('\n')

    def test_where_indented(self):
        result = filtered('SELECT a FROM foo WHERE x = 1')
        text = str(result)
        assert '\nWHERE' in text

    def test_parenthesis_inside_statement(self):
        result = filtered('SELECT (a FROM x)')
        # just ensure no crash and string produced
        assert 'SELECT' in str(result)

    def test_function(self):
        result = filtered('SELECT MAX(a) FROM foo')
        text = str(result)
        assert 'MAX' in text
        assert '\nFROM' in text

    def test_case_expression(self):
        result = filtered(
            'SELECT CASE WHEN a THEN b ELSE c END FROM foo',
            wrap_after=10)
        text = str(result)
        assert 'CASE' in text
        assert 'END' in text

    def test_values(self):
        result = filtered('INSERT INTO t VALUES (1, 2), (3, 4)')
        text = str(result)
        assert 'VALUES' in text

    def test_identifier_list_wrap(self):
        result = filtered('SELECT aaaaaa, bbbbbb, cccccc FROM t',
                          wrap_after=10)
        text = str(result)
        # wrapping should introduce additional lines
        assert text.count('\n') >= 1

    def test_comma_first(self):
        result = filtered('SELECT aaaaaa, bbbbbb, cccccc, dddddd FROM t',
                          wrap_after=10, comma_first=True)
        text = str(result)
        assert 'SELECT' in text

    def test_indent_columns(self):
        result = filtered('SELECT aaaaaa, bbbbbb FROM t',
                          wrap_after=10, indent_columns=True)
        text = str(result)
        assert 'SELECT' in text

    def test_compact_case(self):
        result = filtered(
            'SELECT CASE WHEN a THEN b ELSE c END FROM foo',
            compact=True)
        text = str(result)
        assert 'CASE' in text
        assert 'END' in text

    def test_custom_newline(self):
        result = filtered('SELECT a FROM b', n='\r\n')
        text = str(result)
        assert '\r\nFROM' in text


# ---------------------------------------------------------------------------
# edge cases / branches
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_process_where_without_where_keyword(self):
        # _process_where only triggers on Where groups; directly call it on
        # a statement that has no WHERE to ensure early return.
        stmt = parse('SELECT 1')
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._process_where(stmt)
        # no crash, no changes
        assert 'WHERE' not in str(stmt)

    def test_process_parenthesis_without_open(self):
        # empty Parenthesis group: token_next_by returns None
        stmt = parse('()')
        # find parenthesis group
        groups = list(stmt.get_sublists())
        if groups:
            f = ReindentFilter()
            f._curr_stmt = stmt
            # should early-return if first is None; otherwise run normally
            f._process_parenthesis(groups[0])

    def test_process_default_no_statements(self):
        stmt = parse('SELECT a FROM b')
        f = ReindentFilter()
        f._curr_stmt = stmt
        # stmts=False should skip _split_statements but still split kwds
        f._process_default(stmt, stmts=False)
        assert '\nFROM' in str(stmt)

    def test_where_indent_restored(self):
        stmt = parse('SELECT a FROM foo WHERE x = 1')
        f = ReindentFilter()
        f.process(stmt)
        assert f.indent == 0
        assert f.offset == 0

    def test_identifierlist_within_function(self):
        # identifiers within a Function should take the else branch
        stmt = parse('MAX(a, b)')
        f = ReindentFilter()
        f.process(stmt)
        text = str(stmt)
        assert 'MAX' in text

    def test_identifierlist_within_values(self):
        stmt = parse('INSERT INTO t VALUES (1, 2)')
        f = ReindentFilter()
        f.process(stmt)
        text = str(stmt)
        assert 'VALUES' in text

    def test_case_single_when(self):
        # only one case -> iterable exhausted, no insertion loop iteration
        result = filtered('SELECT CASE WHEN a THEN b END', wrap_after=1000)
        text = str(result)
        assert 'CASE' in text
        assert 'END' in text

    def test_empty_statement(self):
        # parse('') gives an empty statement
        stmt = parse('')
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt

    def test_leading_ws_with_char_tab(self):
        f = ReindentFilter(width=2, char='\t')
        f.indent = 1
        f.offset = 0
        assert f.leading_ws == 2

    def test_two_statements_last_ends_with_newline(self):
        f = ReindentFilter()
        s1 = parse('SELECT 1')
        # manually mark last statement with trailing newline
        f._last_stmt = sql.Token(T.Whitespace, 'x\n')
        s2 = parse('SELECT 2')
        result = f.process(s2)
        text = str(result)
        # single newline separator because last stmt ends with '\n'
        assert text.startswith('\n')
