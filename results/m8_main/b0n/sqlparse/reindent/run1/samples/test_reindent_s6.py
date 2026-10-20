import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter
from sqlparse.utils import indent, offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tok(ttype, value):
    return sql.Token(ttype, value)


def make_stmt(*tokens):
    return sql.Statement(tokens)


# ---------------------------------------------------------------------------
# Constructor / simple attribute defaults
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

    def test_custom_values(self):
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


class TestLeadingWs:
    def test_default(self):
        f = ReindentFilter()
        assert f.leading_ws == 0

    def test_with_indent(self):
        f = ReindentFilter()
        f.indent = 2
        f.offset = 3
        assert f.leading_ws == 2 * 2 + 3

    def test_indent_after_first(self):
        f = ReindentFilter(indent_after_first=True, width=3)
        assert f.indent == 1
        assert f.leading_ws == 3


class TestNl:
    def test_nl_default(self):
        f = ReindentFilter()
        t = f.nl()
        assert t.is_whitespace
        assert str(t) == '\n'

    def test_nl_with_indent(self):
        f = ReindentFilter()
        f.indent = 2
        f.width = 4
        t = f.nl()
        assert str(t) == '\n' + ' ' * 8

    def test_nl_negative_offset_clamped(self):
        f = ReindentFilter()
        f.indent = 1
        f.width = 2
        t = f.nl(offset=-100)
        assert str(t) == '\n'

    def test_nl_positive_offset(self):
        f = ReindentFilter()
        f.indent = 1
        f.width = 2
        t = f.nl(offset=3)
        assert str(t) == '\n' + ' ' * 5

    def test_nl_custom_n_char(self):
        f = ReindentFilter(n='\r\n', char='_', width=2)
        f.indent = 1
        assert str(f.nl()) == '\r\n__'


# ---------------------------------------------------------------------------
# _next_token / _split_kwds / _split_statements
# ---------------------------------------------------------------------------

class TestNextToken:
    def test_finds_from(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'FROM'))
        f = ReindentFilter()
        idx, token = f._next_token(stmt)
        assert token.normalized == 'FROM'

    def test_no_match_returns_none(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        f = ReindentFilter()
        idx, token = f._next_token(stmt)
        assert token is None

    def test_between_and_skip(self):
        # BETWEEN AND should skip over the AND and find the next split keyword.
        stmt = make_stmt(tok(T.Keyword, 'BETWEEN'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'AND'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'FROM'))
        f = ReindentFilter()
        idx, token = f._next_token(stmt)
        # After BETWEEN -> recurse (AND), after AND -> recurse (FROM)
        assert token.normalized == 'FROM'


class TestSplitKwds:
    def test_splits_from(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'FROM'))
        f = ReindentFilter()
        f._split_kwds(stmt)
        assert any(t.is_whitespace and str(t) == '\n' for t in stmt.tokens)

    def test_no_keyword_noop(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        f = ReindentFilter()
        before = list(stmt.tokens)
        f._split_kwds(stmt)
        assert [id(t) for t in stmt.tokens] == [id(t) for t in before]


class TestSplitStatements:
    def test_splits_second_dml(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'INSERT'))
        f = ReindentFilter()
        f._split_statements(stmt)
        # A newline should be inserted before INSERT
        values = [str(t) for t in stmt.tokens if t.is_whitespace]
        assert any(v.endswith('\n') for v in values)

    def test_first_dml_no_newline(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        f = ReindentFilter()
        f._split_statements(stmt)
        assert len(stmt.tokens) == 1


# ---------------------------------------------------------------------------
# _current_line_len / _get_offset
# ---------------------------------------------------------------------------

class TestCurrentLineLen:
    def _setup(self):
        f = ReindentFilter()
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'FROM'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'bar'))
        f._curr_stmt = stmt
        return f, stmt

    def test_current_line_len_all_on_one_line(self):
        f, stmt = self._setup()
        last = stmt.tokens[-1]
        # "SELECT FROM " is 12 characters before bar
        assert f._current_line_len(last) == 12

    def test_current_line_len_after_newline(self):
        f, stmt = self._setup()
        # insert a newline before FROM
        stmt.tokens.insert(2, tok(T.Whitespace, '\n'))
        bar = stmt.tokens[-1]
        # line is "\n bar"? actually "FROM bar": F R O M space = 5 chars
        assert f._current_line_len(bar) == 5

    def test_current_line_len_is_curr_stmt(self):
        f = ReindentFilter()
        stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        f._curr_stmt = stmt
        assert f._current_line_len(stmt) == 0


class TestGetOffset:
    def test_get_offset_simple(self):
        f = ReindentFilter()
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'foo'))
        f._curr_stmt = stmt
        # line length before 'foo' is len('SELECT ') == 7, leading_ws == 0
        assert f._get_offset(stmt.tokens[-1]) == 7

    def test_get_offset_with_indent(self):
        f = ReindentFilter()
        f.indent = 1
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'foo'))
        f._curr_stmt = stmt
        assert f._get_offset(stmt.tokens[-1]) == 7 - 2


# ---------------------------------------------------------------------------
# process() and integration tests
# ---------------------------------------------------------------------------

class TestProcess:
    def test_process_basic_select(self):
        stmt = make_stmt(tok(T.Keyword.DML, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'foo'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'FROM'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'bar'))
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt
        assert f._curr_stmt is stmt
        assert f._last_stmt is stmt
        # FROM should be at the start of a line
        s = str(stmt)
        assert '\nFROM' in s

    def test_process_two_statements_adds_blank_line(self):
        f = ReindentFilter()

        stmt1 = make_stmt(tok(T.Keyword.DML, 'SELECT'),
                          tok(T.Whitespace, ' '),
                          tok(T.Name, 'a'))
        f.process(stmt1)

        stmt2 = make_stmt(tok(T.Keyword.DML, 'SELECT'),
                          tok(T.Whitespace, ' '),
                          tok(T.Name, 'b'))
        f.process(stmt2)
        # A whitespace token should be prepended. Since stmt1 doesn't end
        # with newline, the prepended token is '\n\n'.
        first = stmt2.tokens[0]
        assert first.is_whitespace
        assert str(first) == '\n\n'

    def test_process_second_stmt_single_newline_when_prev_ends_with_nl(self):
        f = ReindentFilter()

        # First statement ends with a newline
        stmt1 = make_stmt(tok(T.Keyword.DML, 'SELECT'),
                          tok(T.Whitespace, ' '),
                          tok(T.Name, 'a'),
                          tok(T.Whitespace, '\n'))
        f.process(stmt1)
        assert str(stmt1).endswith('\n')

        stmt2 = make_stmt(tok(T.Keyword.DML, 'SELECT'),
                          tok(T.Whitespace, ' '),
                          tok(T.Name, 'b'))
        f.process(stmt2)
        assert str(stmt2.tokens[0]) == '\n'


class TestProcessWhere:
    def test_where_gets_newline(self):
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'x'),
                         tok(T.Whitespace, ' '),
                         tok(T.Keyword, 'WHERE'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'y'))
        f = ReindentFilter()
        f.process(stmt)
        assert '\nWHERE' in str(stmt)

    def test_where_without_where_returns(self):
        # A Where group without WHERE token should return early.
        where = sql.Where([
            tok(T.Keyword, 'SELECT'),
            tok(T.Name, 'x'),
        ])
        f = ReindentFilter()
        f._process_where(where)
        assert len(where.tokens) == 2


class TestProcessCase:
    def test_case_simple(self):
        # Build a simple CASE expression
        text = "CASE WHEN a = 1 THEN 'x' ELSE 'y' END"
        from sqlparse import parse
        stmt = parse(text)[0]
        f = ReindentFilter(wrap_after=10)
        f.process(stmt)
        out = str(stmt)
        assert 'CASE' in out
        assert 'END' in out

    def test_case_compact_no_newlines(self):
        from sqlparse import parse
        text = "CASE WHEN a = 1 THEN 'x' ELSE 'y' END"
        stmt = parse(text)[0]
        f = ReindentFilter(wrap_after=1, compact=True)
        f.process(stmt)
        # In compact mode no extra newlines around WHEN/END should be added
        # from the case-specific logic (still one-line).
        out = str(stmt)
        assert out.count('\n') == 0


class TestProcessParenthesis:
    def test_parenthesis_basic(self):
        from sqlparse import parse
        stmt = parse("SELECT (1 + 2) AS x")[0]
        f = ReindentFilter()
        f.process(stmt)
        assert '(' in str(stmt)
        assert ')' in str(stmt)

    def test_parenthesis_no_open_returns(self):
        # Parenthesis without '(' token returns early.
        paren = sql.Parenthesis([tok(T.Name, 'x')])
        f = ReindentFilter()
        f._process_parenthesis(paren)
        # No modification should have occurred.
        assert [str(t) for t in paren.tokens] == ['x']


class TestProcessFunction:
    def test_function_records_last_func(self):
        from sqlparse import parse
        stmt = parse("SELECT COUNT(*) FROM t")[0]
        f = ReindentFilter()
        f.process(stmt)
        assert f._last_func is not None
        assert 'COUNT' in f._last_func.value


class TestProcessIdentifierlist:
    def test_identifierlist_within_function(self):
        from sqlparse import parse
        stmt = parse("SELECT f(a, b, c) FROM t")[0]
        f = ReindentFilter(wrap_after=5)
        f.process(stmt)
        assert 'f' in str(stmt)

    def test_identifierlist_wrapping(self):
        from sqlparse import parse
        stmt = parse("SELECT aaaaa, bbbbb, ccccc, ddddd FROM t")[0]
        f = ReindentFilter(wrap_after=10)
        f.process(stmt)
        out = str(stmt)
        # At least some wrapping must have happened
        assert out.count('\n') >= 1

    def test_identifierlist_comma_first(self):
        from sqlparse import parse
        stmt = parse("SELECT aaaaa, bbbbb, ccccc, ddddd FROM t")[0]
        f = ReindentFilter(wrap_after=10, comma_first=True)
        f.process(stmt)
        out = str(stmt)
        assert ',' in out

    def test_identifierlist_indent_columns(self):
        from sqlparse import parse
        stmt = parse("SELECT aaa, bbb, ccc FROM t")[0]
        f = ReindentFilter(wrap_after=5, indent_columns=True)
        f.process(stmt)
        assert 'aaa' in str(stmt)


class TestProcessValues:
    def test_values_basic(self):
        from sqlparse import parse
        stmt = parse("INSERT INTO t VALUES (1), (2), (3)")[0]
        f = ReindentFilter()
        f.process(stmt)
        out = str(stmt)
        assert 'VALUES' in out

    def test_values_comma_first(self):
        from sqlparse import parse
        stmt = parse("INSERT INTO t VALUES (1), (2), (3)")[0]
        f = ReindentFilter(comma_first=True)
        f.process(stmt)
        assert 'VALUES' in str(stmt)


# ---------------------------------------------------------------------------
# _process dispatch and _process_default
# ---------------------------------------------------------------------------

class TestProcessDispatch:
    def test_process_dispatch_unknown_uses_default(self):
        f = ReindentFilter()
        f._curr_stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        stmt = make_stmt(tok(T.Keyword, 'SELECT'))
        f._curr_stmt = stmt
        f._process(stmt)
        assert f._curr_stmt is stmt

    def test_process_default_without_trailing_statements(self):
        f = ReindentFilter()
        stmt = make_stmt(tok(T.Keyword, 'SELECT'),
                         tok(T.Whitespace, ' '),
                         tok(T.Name, 'x'))
        f._curr_stmt = stmt
        f._process_default(stmt, stmts=False)
        # No DML split should have happened; still tokens present
        assert any(t.value == 'SELECT' for t in stmt.flatten()
                   if not t.is_whitespace)


# ---------------------------------------------------------------------------
# External formatter integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_reindent_via_parse_and_filter(self):
        from sqlparse import parse
        stmt = parse("SELECT a b FROM foo WHERE x = 1")[0]
        f = ReindentFilter()
        f.process(stmt)
        out = str(stmt)
        assert '\n' in out

    def test_indent_and_offset_context_managers_restore(self):
        f = ReindentFilter()
        start_indent = f.indent
        with indent(f, 3):
            assert f.indent == start_indent + 3
        assert f.indent == start_indent

        start_offset = f.offset
        with offset(f, 5):
            assert f.offset == start_offset + 5
        assert f.offset == start_offset
