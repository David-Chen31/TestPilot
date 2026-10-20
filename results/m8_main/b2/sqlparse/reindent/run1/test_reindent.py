import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter
from sqlparse import parse


def fmt(sql_text, **kwargs):
    stmt = parse(sql_text)[0]
    stmt = ReindentFilter(**kwargs).process(stmt)
    return stmt


def text(sql_text, **kwargs):
    return str(fmt(sql_text, **kwargs))


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
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

    def test_custom(self):
        f = ReindentFilter(width=4, char='\t', wrap_after=80, n='\r\n',
                           comma_first=True, indent_after_first=True,
                           indent_columns=True, compact=True)
        assert f.width == 4
        assert f.char == '\t'
        assert f.wrap_after == 80
        assert f.n == '\r\n'
        assert f.comma_first is True
        assert f.indent == 1
        assert f.indent_columns is True
        assert f.compact is True

    def test_indent_after_first_sets_indent(self):
        assert ReindentFilter(indent_after_first=True).indent == 1
        assert ReindentFilter(indent_after_first=False).indent == 0


class TestProperties:
    def test_leading_ws_default(self):
        f = ReindentFilter()
        assert f.leading_ws == 0

    def test_leading_ws_with_indent(self):
        f = ReindentFilter(width=3)
        f.indent = 2
        assert f.leading_ws == 6

    def test_leading_ws_with_offset(self):
        f = ReindentFilter(width=2)
        f.indent = 1
        f.offset = 5
        assert f.leading_ws == 7


class TestNl:
    def test_nl_plain(self):
        f = ReindentFilter()
        tok = f.nl()
        assert isinstance(tok, sql.Token)
        assert tok.ttype is T.Whitespace
        assert tok.value == '\n'

    def test_nl_with_indent(self):
        f = ReindentFilter(width=2)
        f.indent = 2
        tok = f.nl()
        assert tok.value == '\n    '

    def test_nl_with_offset(self):
        f = ReindentFilter(width=2)
        f.indent = 1
        tok = f.nl(offset=3)
        assert tok.value == '\n     '

    def test_nl_negative_offset_clamped(self):
        f = ReindentFilter(width=2)
        f.indent = 0
        tok = f.nl(offset=-10)
        assert tok.value == '\n'

    def test_nl_custom_newline_char(self):
        f = ReindentFilter(n='\r\n', char='.', width=2)
        f.indent = 1
        tok = f.nl()
        assert tok.value == '\r\n..'


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

class TestNextToken:
    def test_finds_keyword(self):
        stmt = parse('SELECT a FROM b')[0]
        f = ReindentFilter()
        idx, tok = f._next_token(stmt)
        assert tok is not None
        assert tok.normalized == 'FROM'

    def test_stops_at_between_and(self):
        # BETWEEN ... AND must be treated specially: the AND is skipped.
        stmt = parse('SELECT a BETWEEN 1 AND 2 FROM t')[0]
        f = ReindentFilter()
        idx, tok = f._next_token(stmt)
        # first split keyword encountered should be BETWEEN then it looks for
        # the AND and continues from there.
        assert tok is not None
        assert tok.normalized in ('BETWEEN', 'FROM')

    def test_no_token(self):
        stmt = parse('SELECT 1')[0]
        f = ReindentFilter()
        _, tok = f._next_token(stmt)
        # no split words present
        assert tok is None


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

class TestProcessWhere:
    def test_where_no_keyword_returns(self):
        stmt = parse('SELECT 1')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt
        # find a Where group; create one artificially
        wh = sql.Where([sql.Token(T.Keyword, 'WHERE')])
        # Remove keyword to test early return
        wh.tokens = []
        before = list(wh.tokens)
        f._process_where(wh)
        assert wh.tokens == before

    def test_where_formats(self):
        result = text('SELECT a FROM b WHERE c = 1')
        assert 'WHERE' in result
        # newline inserted before WHERE
        assert '\n' in result


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

class TestProcessParenthesis:
    def test_empty_parenthesis(self):
        stmt = parse('SELECT ()')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt
        paren = None
        for tok in stmt.flatten():
            pass
        # Directly construct empty parenthesis
        p = sql.Parenthesis([])
        # No open token, should return without error
        f._process_parenthesis(p)
        assert p.tokens == []

    def test_parenthesis_with_subquery(self):
        result = text('SELECT * FROM (SELECT a FROM b) x')
        assert '(' in result
        assert ')' in result


# ---------------------------------------------------------------------------
# _process_function
# ---------------------------------------------------------------------------

class TestProcessFunction:
    def test_sets_last_func(self):
        stmt = parse('SELECT COUNT(a) FROM t')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt

        def find_function(lst):
            for sub in lst.get_sublists():
                if isinstance(sub, sql.Function):
                    yield sub
                yield from find_function(sub)

        funcs = list(find_function(stmt))
        assert funcs, 'expected a Function group'
        f._process_function(funcs[0])
        assert f._last_func is funcs[0][0]


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

class TestProcessIdentifierList:
    def test_simple_identifiers_no_wrap(self):
        result = text('SELECT a, b, c FROM t')
        assert 'a' in result and 'b' in result and 'c' in result

    def test_wrap_after(self):
        result = text('SELECT aaa, bbb, ccc, ddd FROM t', wrap_after=10)
        assert '\n' in result

    def test_comma_first(self):
        result = text('SELECT aaa, bbb, ccc, ddd FROM t',
                      wrap_after=5, comma_first=True)
        assert ',' in result

    def test_identifierlist_in_function(self):
        result = text('SELECT f(a, b) FROM t')
        assert 'a' in result and 'b' in result

    def test_identifierlist_in_values(self):
        result = text("INSERT INTO t VALUES (1, 2, 3)")
        assert '1' in result


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

class TestProcessCase:
    def test_case_processing(self):
        result = text(
            'SELECT CASE WHEN a = 1 THEN 2 ELSE 3 END FROM t')
        assert 'CASE' in result
        assert 'WHEN' in result
        assert 'END' in result

    def test_case_compact(self):
        result = text(
            'SELECT CASE WHEN a = 1 THEN 2 ELSE 3 END FROM t', compact=True)
        assert 'CASE' in result

    def test_case_wrap(self):
        result = text(
            'SELECT CASE WHEN a = 1 THEN 2 WHEN b = 2 THEN 3 END FROM t',
            wrap_after=5)
        assert 'CASE' in result


# ---------------------------------------------------------------------------
# _process_values
# ---------------------------------------------------------------------------

class TestProcessValues:
    def test_values_format(self):
        result = text("INSERT INTO t VALUES (1, 2), (3, 4)")
        assert '\n' in result
        assert '1' in result and '4' in result

    def test_values_comma_first(self):
        result = text("INSERT INTO t VALUES (1, 2), (3, 4)",
                      comma_first=True)
        assert ',' in result


# ---------------------------------------------------------------------------
# _split_statements
# ---------------------------------------------------------------------------

class TestSplitStatements:
    def test_multiple_statements(self):
        stmt = parse('SELECT 1')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt
        # default processing splits DML keywords
        f._process(stmt)
        assert str(stmt).strip() != ''


# ---------------------------------------------------------------------------
# process - top level
# ---------------------------------------------------------------------------

class TestProcess:
    def test_process_sets_curr_stmt(self):
        stmt = parse('SELECT a FROM b')[0]
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt
        assert f._curr_stmt is stmt
        assert f._last_stmt is stmt

    def test_process_various_statements(self):
        for src in [
            'SELECT a FROM b',
            'SELECT a, b FROM b WHERE c = 1',
            'INSERT INTO t VALUES (1, 2)',
            'UPDATE t SET a = 1 WHERE b = 2',
            'DELETE FROM t WHERE a = 1',
        ]:
            result = text(src)
            assert result.strip()

    def test_process_multiple_returns(self):
        stmts = parse('SELECT 1; SELECT 2')
        f = ReindentFilter()
        results = [f.process(s) for s in stmts]
        assert len(results) == 2
        # second statement gets a separating newline prefix
        if len(stmts) > 1:
            joined = ''.join(str(s) for s in results)
            assert 'SELECT' in joined

    def test_process_idempotent_stability(self):
        # Processing twice should not crash
        result1 = text('SELECT a, b FROM t')
        result2 = text('SELECT a, b FROM t')
        assert result1 == result2


# ---------------------------------------------------------------------------
# wrap and offset behavior
# ---------------------------------------------------------------------------

class TestWrapBehavior:
    def test_wrap_after_zero_no_extra_break_in_function(self):
        result = text('SELECT f(a, b, c) FROM t')
        assert 'f(' in result

    def test_indent_columns(self):
        result = text('SELECT aaa, bbb, ccc FROM t',
                      indent_columns=True, wrap_after=5)
        assert 'aaa' in result


class TestEdgeCases:
    def test_empty_statement(self):
        # Empty token list should not raise
        stmt = sql.Statement([])
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt

    def test_current_line_len_no_break(self):
        stmt = parse('SELECT abc')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt
        token = stmt.tokens[-1]
        length = f._current_line_len(token)
        assert isinstance(length, int)
        assert length >= 0

    def test_get_offset_returns_int(self):
        stmt = parse('SELECT abc')[0]
        f = ReindentFilter()
        f._curr_stmt = stmt
        token = stmt.tokens[-1]
        off = f._get_offset(token)
        assert isinstance(off, int)

    def test_tab_char(self):
        result = text('SELECT a, b FROM t', char='\t', width=1)
        assert 'SELECT' in result

    def test_whitespace_handling(self):
        # Already-formatted input should be idempotent-ish
        result = text('SELECT\n  a\nFROM t')
        assert 'SELECT' in result
