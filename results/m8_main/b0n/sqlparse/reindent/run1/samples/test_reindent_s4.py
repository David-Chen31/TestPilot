import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.reindent import ReindentFilter
from sqlparse import parse


def tokenize(sql_text):
    return parse(sql_text)[0]


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
        assert f.n == '\r\n'
        assert f.width == 4
        assert f.char == '\t'
        assert f.indent == 1
        assert f.wrap_after == 80
        assert f.comma_first is True
        assert f.indent_columns is True
        assert f.compact is True


class TestLeadingWs:
    def test_leading_ws(self):
        f = ReindentFilter(width=2)
        assert f.leading_ws == 0
        f.indent = 1
        assert f.leading_ws == 2
        f.indent = 3
        assert f.leading_ws == 6
        f.offset = 4
        assert f.leading_ws == 10

    def test_leading_ws_with_char(self):
        f = ReindentFilter(width=3, char='-')
        f.indent = 2
        assert f.leading_ws == 6


class TestNl:
    def test_nl_basic(self):
        f = ReindentFilter(width=2)
        tok = f.nl()
        assert isinstance(tok, sql.Token)
        assert tok.ttype is T.Whitespace
        assert tok.value == '\n'

    def test_nl_with_indent(self):
        f = ReindentFilter(width=2)
        f.indent = 2
        assert f.nl().value == '\n    '

    def test_nl_with_offset(self):
        f = ReindentFilter(width=2)
        f.indent = 1
        assert f.nl(offset=3).value == '\n     '

    def test_nl_negative_clamped(self):
        f = ReindentFilter(width=2)
        assert f.nl(offset=-10).value == '\n'

    def test_nl_custom_newline_and_char(self):
        f = ReindentFilter(width=1, char='.', n='\r\n')
        f.indent = 3
        assert f.nl().value == '\r\n...'


class TestCurrentLineLen:
    def test_empty_statement(self):
        stmt = tokenize("SELECT 1")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # measuring the first token: nothing before
        first = stmt.tokens[0]
        assert f._current_line_len(first) == 0

    def test_counts_preceding_tokens(self):
        stmt = tokenize("SELECT 1")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # find a token deep in statement - measure all preceding text length
        total = len(str(stmt))
        last = next(reversed(list(stmt.flatten())))
        # last token's preceding length equals total minus its own value
        assert f._current_line_len(last) == total - len(last.value)

    def test_stops_at_newline(self):
        stmt = tokenize("SELECT 1\nFROM foo")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # FROM keyword should only count back to the newline (0)
        _, from_tok = stmt.token_next_by(m=(T.Keyword, 'FROM'))
        assert f._current_line_len(from_tok) == 0

    def test_counts_after_newline_text(self):
        stmt = tokenize("SELECT 1\nFROM foo")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # last token is 'foo' identifier; preceding on line is 'FROM '
        last = next(reversed(list(stmt.flatten())))
        assert f._current_line_len(last) == 5  # 'FROM '

    def test_returns_zero_when_node_is_curr_stmt(self):
        stmt = tokenize("SELECT 1")
        f = ReindentFilter()
        f._curr_stmt = stmt
        assert f._current_line_len(stmt) == 0


class TestGetOffset:
    def test_basic(self):
        stmt = tokenize("SELECT foo")
        f = ReindentFilter(width=2)
        f._curr_stmt = stmt
        first = stmt.tokens[0]
        # leading_ws is 0 -> offset = 0 - 0 = 0
        assert f._get_offset(first) == 0

    def test_with_leading_ws(self):
        stmt = tokenize("SELECT foo")
        f = ReindentFilter(width=2)
        f._curr_stmt = stmt
        f.indent = 1
        # leading_ws = 2, current line len 0 -> 0 - 2 = -2
        first = stmt.tokens[0]
        assert f._get_offset(first) == -2

    def test_line_len_minus_ws(self):
        stmt = tokenize("abc def")
        f = ReindentFilter(width=2)
        f._curr_stmt = stmt
        f.indent = 1
        last = next(reversed(list(stmt.flatten())))
        # preceding = 'abc ' -> 4 ; minus leading_ws (2) = 2
        assert f._get_offset(last) == 2


class TestNextToken:
    def test_finds_select(self):
        f = ReindentFilter()
        stmt = tokenize("SELECT 1")
        # SELECT not in split words so not found
        idx, tok = f._next_token(stmt)
        assert (idx, tok) == (-1, None)

    def test_finds_from(self):
        f = ReindentFilter()
        stmt = tokenize("SELECT * FROM foo")
        idx, tok = f._next_token(stmt)
        assert tok.value == 'FROM'

    def test_skips_between_and(self):
        f = ReindentFilter()
        stmt = tokenize("a BETWEEN 1 AND 2 AND b")
        idx, tok = f._next_token(stmt)
        # BETWEEN triggers recursion, then AND triggers recursion,
        # then final token is the second AND
        assert tok.normalized == 'AND'

    def test_between_at_end(self):
        f = ReindentFilter()
        stmt = tokenize("SELECT a BETWEEN")
        idx, tok = f._next_token(stmt)
        # BETWEEN followed by recursion; nothing after -> returns -1/None
        assert tok is None


class TestSplitStatements:
    def test_simple_select(self):
        f = ReindentFilter()
        stmt = tokenize("SELECT 1")
        f._curr_stmt = stmt
        f._split_statements(stmt)
        # Only one DML; no break inserted
        assert 'SELECT' in str(stmt)
        assert not str(stmt).startswith('\n')

    def test_two_statements(self):
        stmt = tokenize("SELECT 1; SELECT 2")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_statements(stmt)
        # Second SELECT gets a newline before it
        assert 'SELECT 1;\nSELECT 2' in str(stmt)


class TestSplitKwds:
    def test_from_split(self):
        stmt = tokenize("SELECT * FROM foo")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_kwds(stmt)
        assert 'FROM' in str(stmt)
        assert '\nFROM' in str(stmt)

    def test_already_newlined(self):
        stmt = tokenize("SELECT *\nFROM foo")
        f = ReindentFilter()
        f._curr_stmt = stmt
        before = str(stmt)
        f._split_kwds(stmt)
        # FROM already after a newline, so no change there
        assert str(stmt).count('\nFROM') == 1

    def test_and_split(self):
        stmt = tokenize("a AND b AND c")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_kwds(stmt)
        s = str(stmt)
        assert '\nAND' in s


class TestProcessWhere:
    def test_where_split(self):
        stmt = tokenize("SELECT * FROM foo WHERE x = 1")
        f = ReindentFilter()
        f._curr_stmt = stmt
        _, where = stmt.token_next_by(m=(T.Keyword, 'WHERE'))
        where.parent._process_where(where.parent)  # not typical but no crash
        # Use full processing instead

    def test_where_within_statement(self):
        stmt = tokenize("SELECT * FROM foo WHERE x = 1")
        f = ReindentFilter()
        result = f.process(stmt)
        assert '\nWHERE' in str(result)

    def test_no_where(self):
        stmt = tokenize("SELECT 1")
        f = ReindentFilter()
        f.process(stmt)
        assert 'WHERE' not in str(stmt)


class TestProcessParenthesis:
    def test_simple_paren(self):
        stmt = tokenize("SELECT (1)")
        f = ReindentFilter()
        result = f.process(stmt)
        assert '(' in str(result)

    def test_dml_in_paren(self):
        stmt = tokenize("SELECT * FROM (SELECT 1) x")
        f = ReindentFilter()
        result = f.process(stmt)
        assert 'SELECT' in str(result)


class TestProcessFunction:
    def test_function_sets_last_func(self):
        stmt = tokenize("SELECT COUNT(a)")
        f = ReindentFilter()
        f._curr_stmt = stmt
        func = stmt.get_sublists()
        # find Function by walking
        found = None
        for token in stmt.flatten():
            pass
        # simulate process
        f.process(stmt)
        assert f._last_func is not None
        assert f._last_func.value.lower() == 'count'


class TestProcessIdentifierlist:
    def test_wrap_after(self):
        filter_ = ReindentFilter(wrap_after=20)
        stmt = tokenize(
            "SELECT aaaaa, bbbbb, ccccc, ddddd, eeeee, fffff FROM foo")
        result = filter_.process(stmt)
        s = str(result)
        assert '\n' in s.split('FROM')[0]

    def test_comma_first(self):
        filter_ = ReindentFilter(wrap_after=20, comma_first=True)
        stmt = tokenize(
            "SELECT aaaaa, bbbbb, ccccc, ddddd, eeeee, fffff FROM foo")
        result = filter_.process(stmt)
        s = str(result)
        assert s.count(',') == 5

    def test_indent_columns(self):
        filter_ = ReindentFilter(wrap_after=20, indent_columns=True)
        stmt = tokenize("SELECT a, b, c, d, e, f, g, h FROM foo")
        result = filter_.process(stmt)
        assert 'SELECT' in str(result)

    def test_tab_char(self):
        filter_ = ReindentFilter(wrap_after=20, char='\t', indent_columns=True)
        stmt = tokenize("SELECT a, b, c, d, e, f FROM foo")
        result = filter_.process(stmt)
        assert 'SELECT' in str(result)

    def test_in_function_no_wrap(self):
        filter_ = ReindentFilter(wrap_after=10)
        stmt = tokenize("SELECT COALESCE(a, b, c, d, e) FROM foo")
        result = filter_.process(stmt)
        assert 'COALESCE' in str(result)

    def test_in_values(self):
        filter_ = ReindentFilter(wrap_after=10)
        stmt = tokenize("INSERT INTO t VALUES (a, b, c, d, e)")
        result = filter_.process(stmt)
        assert 'VALUES' in str(result)


class TestProcessCase:
    def test_case_split(self):
        stmt = tokenize(
            "SELECT CASE WHEN aaaa THEN bbbb WHEN cccc THEN dddd END FROM t")
        f = ReindentFilter(wrap_after=20)
        result = f.process(stmt)
        s = str(result)
        assert 'CASE' in s
        assert 'END' in s

    def test_compact_case(self):
        stmt = tokenize(
            "SELECT CASE WHEN aaaa THEN bbbb WHEN cccc THEN dddd END FROM t")
        f = ReindentFilter(wrap_after=20, compact=True)
        result = f.process(stmt)
        assert 'CASE' in str(result)


class TestProcessValues:
    def test_values_split(self):
        stmt = tokenize("INSERT INTO t VALUES (1, 2), (3, 4)")
        f = ReindentFilter()
        result = f.process(stmt)
        assert 'VALUES' in str(result)

    def test_values_comma_first(self):
        stmt = tokenize("INSERT INTO t VALUES (1, 2), (3, 4)")
        f = ReindentFilter(comma_first=True)
        result = f.process(stmt)
        assert 'VALUES' in str(result)


class TestProcessDefault:
    def test_stmts_false(self):
        stmt = tokenize("SELECT 1; SELECT 2")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # call with stmts=False -> should not split statements
        f._process_default(stmt, stmts=False)
        # statements not split at top level by this call
        # just verify no crash and content preserved
        assert 'SELECT 1' in str(stmt)


class TestProcess:
    def test_returns_stmt(self):
        stmt = tokenize("SELECT 1")
        f = ReindentFilter()
        result = f.process(stmt)
        assert result is stmt

    def test_sets_curr_and_last(self):
        stmt1 = tokenize("SELECT 1")
        stmt2 = tokenize("SELECT 2")
        f = ReindentFilter()
        f.process(stmt1)
        assert f._curr_stmt is stmt1
        assert f._last_stmt is stmt1
        f.process(stmt2)
        assert f._curr_stmt is stmt2
        assert f._last_stmt is stmt2

    def test_multiple_statements_get_separator(self):
        stmt1 = tokenize("SELECT 1")
        stmt2 = tokenize("SELECT 2")
        f = ReindentFilter()
        f.process(stmt1)
        f.process(stmt2)
        # stmt2 gets a leading whitespace token
        assert stmt2.tokens[0].ttype is T.Whitespace
        assert stmt2.tokens[0].value in ('\n', '\n\n')

    def test_last_stmt_ending_newline(self):
        stmt1 = tokenize("SELECT 1")
        # force stmt1 repr to end with newline
        stmt1.tokens.append(sql.Token(T.Whitespace, '\n'))
        stmt2 = tokenize("SELECT 2")
        f = ReindentFilter()
        f.process(stmt1)
        f.process(stmt2)
        assert stmt2.tokens[0].value == '\n'

    def test_last_stmt_no_trailing_newline(self):
        stmt1 = tokenize("SELECT 1")
        stmt2 = tokenize("SELECT 2")
        f = ReindentFilter()
        f.process(stmt1)
        f.process(stmt2)
        assert stmt2.tokens[0].value == '\n\n'

    def test_full_select_formatting(self):
        stmt = tokenize("select a, b from foo where x = 1 and y = 2")
        f = ReindentFilter()
        result = f.process(stmt)
        s = str(result)
        assert '\nFROM' in s
        assert '\nWHERE' in s

    def test_join_split(self):
        stmt = tokenize("SELECT a FROM t JOIN u ON t.id = u.id")
        f = ReindentFilter()
        result = f.process(stmt)
        assert '\nJOIN' in str(result)

    def test_group_by_split(self):
        stmt = tokenize("SELECT a FROM t GROUP BY a ORDER BY a")
        f = ReindentFilter()
        result = f.process(stmt)
        s = str(result)
        assert '\nGROUP BY' in s
        assert '\nORDER BY' in s


class TestIndentOffsetContext:
    def test_indent_context_applied(self):
        stmt = tokenize("SELECT * FROM foo WHERE x = 1")
        f = ReindentFilter()
        f.process(stmt)
        # indent restored to 0 after process
        assert f.indent == 0
        assert f.offset == 0

    def test_previous_filter_state_clean(self):
        f = ReindentFilter(indent_after_first=True)
        stmt = tokenize("SELECT 1")
        f.process(stmt)
        assert f.indent == 1
        assert f.offset == 0
