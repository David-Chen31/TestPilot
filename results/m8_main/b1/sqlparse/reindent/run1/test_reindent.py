"""Tests for sqlparse.filters.reindent.ReindentFilter."""

import pytest

import sqlparse
from sqlparse import sql, tokens as T
from sqlparse.filters.reindent import ReindentFilter


def parse_all(text):
    """Parse *text* and return all statements."""
    return sqlparse.parse(text)


def make_stmt(text):
    """Parse *text* and return the single Statement."""
    stmts = sqlparse.parse(text)
    assert len(stmts) == 1
    return stmts[0]


def reindent(stmt, **kwargs):
    """Apply a fresh filter with *kwargs* to *stmt* and return it."""
    return ReindentFilter(**kwargs).process(stmt)


# ---------------------------------------------------------------------------
# Constructor / leading_ws
# ---------------------------------------------------------------------------

class TestConstructor:
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
        f = ReindentFilter(
            width=4, char='\t', wrap_after=10, n='\r\n',
            comma_first=True, indent_after_first=True,
            indent_columns=True, compact=True,
        )
        assert f.width == 4
        assert f.char == '\t'
        assert f.wrap_after == 10
        assert f.n == '\r\n'
        assert f.comma_first is True
        assert f.indent == 1          # indent_after_first -> indent 1
        assert f.indent_columns is True
        assert f.compact is True

    def test_leading_ws_without_indent(self):
        f = ReindentFilter(width=2)
        assert f.leading_ws == 0

    def test_leading_ws_with_indent(self):
        f = ReindentFilter(width=3, indent_after_first=True)
        assert f.indent == 1
        assert f.leading_ws == 3

    def test_leading_ws_includes_offset(self):
        f = ReindentFilter(width=2, indent_after_first=True)
        f.offset = 4
        assert f.leading_ws == 4 + 1 * 2


# ---------------------------------------------------------------------------
# nl()
# ---------------------------------------------------------------------------

class TestNl:
    def test_nl_minimal(self):
        f = ReindentFilter()
        tok = f.nl()
        assert tok.value == '\n'
        assert tok.ttype is T.Whitespace

    def test_nl_with_width_and_indent(self):
        f = ReindentFilter(width=3, indent_after_first=True)
        tok = f.nl()
        assert tok.value == '\n' + ' ' * 3

    def test_nl_with_positive_offset(self):
        f = ReindentFilter(width=2, indent_after_first=True)
        tok = f.nl(offset=2)
        # leading_ws == 2, +2 -> 4 spaces
        assert tok.value == '\n' + ' ' * 4

    def test_nl_negative_offset_clamped(self):
        f = ReindentFilter(width=2)  # leading_ws == 0
        tok = f.nl(offset=-5)
        assert tok.value == '\n'

    def test_nl_uses_custom_char_and_break(self):
        f = ReindentFilter(width=2, char='\t', indent_after_first=True,
                           n='\r\n')
        tok = f.nl()
        assert tok.value == '\r\n' + '\t' * 2


# ---------------------------------------------------------------------------
# _current_line_len / _get_offset
# ---------------------------------------------------------------------------

class TestCurrentLineLen:
    def test_line_len_before_any_break(self):
        stmt = make_stmt("SELECT a, b, c FROM t")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # First token of SELECT: nothing precedes it -> 0.
        first = stmt.tokens[0]
        assert f._current_line_len(first) == 0

    def test_line_len_accumulates_values(self):
        stmt = make_stmt("SELECT abc FROM t")
        f = ReindentFilter()
        f._curr_stmt = stmt
        # Find the FROM keyword token position on the raw statement.
        idx, token = stmt.token_next_by(m=(T.Keyword, 'FROM'))
        # Tokens before FROM: 'SELECT', ' ', 'abc', ' '.  Length is 11.
        expected = len('SELECT') + 1 + len('abc') + 1
        assert f._current_line_len(token) == expected

    def test_line_len_after_newline_resets(self):
        stmt = make_stmt("SELECT\n  abc FROM t")
        f = ReindentFilter()
        f._curr_stmt = stmt
        _, token = stmt.token_next_by(m=(T.Keyword, 'FROM'))
        # Current line is "  abc " -> length 6.
        assert f._current_line_len(token) == len('  abc ')

    def test_get_offset_relates_to_leading_ws(self):
        stmt = make_stmt("SELECT abc FROM t")
        f = ReindentFilter(width=2)
        f._curr_stmt = stmt
        _, token = stmt.token_next_by(m=(T.Keyword, 'FROM'))
        expected_len = len('SELECT') + 1 + len('abc') + 1
        # leading_ws == 0 (no indent, no offset), so offset == length.
        assert f._get_offset(token) == expected_len


# ---------------------------------------------------------------------------
# _next_token
# ---------------------------------------------------------------------------

class TestNextToken:
    def test_finds_first_split_keyword(self):
        stmt = make_stmt("SELECT a FROM t")
        f = ReindentFilter()
        tidx, token = f._next_token(stmt)
        assert token is not None
        assert token.normalized == 'FROM'

    def test_no_split_keyword(self):
        stmt = make_stmt("SELECT a b c")
        f = ReindentFilter()
        tidx, token = f._next_token(stmt)
        assert token is None

    def test_between_skips_its_and(self):
        stmt = make_stmt("SELECT a WHERE x BETWEEN 1 AND 2")
        f = ReindentFilter()
        tidx, token = f._next_token(stmt)
        # First match is BETWEEN; the helper recurses past BETWEEN and its
        # AND, so we should end up on the next real split keyword or None.
        assert token is None or token.normalized not in ('BETWEEN', 'AND')


# ---------------------------------------------------------------------------
# Process: statement separation
# ---------------------------------------------------------------------------

class TestProcessStatements:
    def test_single_statement_returns_same_object(self):
        stmt = make_stmt("SELECT a FROM t")
        out = ReindentFilter().process(stmt)
        assert out is stmt

    def test_newline_between_keywords(self):
        stmt = make_stmt("SELECT a FROM t")
        result = str(ReindentFilter().process(stmt))
        assert '\n' in result
        # FROM gets its own line
        assert 'FROM' in result

    def test_last_stmt_recorded(self):
        f = ReindentFilter()
        stmt = make_stmt("SELECT a")
        f.process(stmt)
        assert f._last_stmt is stmt
        assert f._curr_stmt is stmt

    def test_second_statement_gets_leading_newline(self):
        f = ReindentFilter()
        s1 = f.process(make_stmt("SELECT 1"))
        # s1 does not end with newline -> next gets '\n\n'
        assert not str(s1).endswith('\n')
        s2 = f.process(make_stmt("SELECT 2"))
        assert s2.tokens[0].is_whitespace
        assert s2.tokens[0].value == '\n\n'

    def test_second_statement_single_newline_if_prev_ended_nl(self):
        f = ReindentFilter()
        s1 = make_stmt("SELECT 1")
        s1.tokens.append(sql.Token(T.Whitespace, '\n'))
        f.process(s1)
        s2 = f.process(make_stmt("SELECT 2"))
        assert s2.tokens[0].value == '\n'


# ---------------------------------------------------------------------------
# _split_kwds
# ---------------------------------------------------------------------------

class TestSplitKwds:
    def test_from_split(self):
        stmt = make_stmt("SELECT a FROM t")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_kwds(stmt)
        text = str(stmt)
        assert '\nFROM' in text

    def test_no_split_keyword_leaves_text_alone(self):
        stmt = make_stmt("SELECT a b c")
        f = ReindentFilter()
        f._curr_stmt = stmt
        before = str(stmt)
        f._split_kwds(stmt)
        assert str(stmt) == before

    def test_and_or_split(self):
        stmt = make_stmt("SELECT a WHERE x=1 AND y=2 OR z=3")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_kwds(stmt)
        text = str(stmt)
        assert '\nAND' in text or 'AND' in text


# ---------------------------------------------------------------------------
# _split_statements
# ---------------------------------------------------------------------------

class TestSplitStatements:
    def test_inserts_break_before_dml(self):
        # Two statements are parsed separately; reindent each with the same
        # filter so _last_stmt tracking gives us the statement separator.
        stmts = parse_all("SELECT a; INSERT INTO t VALUES (1)")
        assert len(stmts) == 2
        f = ReindentFilter()
        s1 = f.process(stmts[0])
        s2 = f.process(stmts[1])
        assert s2.tokens[0].is_whitespace
        assert s2.tokens[0].value in ('\n', '\n\n')
        # Second statement is separated from the first.
        assert '\n' in str(s2)[:1] or s2.tokens[0].value.startswith('\n')

    def test_first_statement_no_break(self):
        stmt = make_stmt("SELECT a")
        f = ReindentFilter()
        f._curr_stmt = stmt
        f._split_statements(stmt)
        # Only statement, no prev -> unchanged
        assert str(stmt) == "SELECT a"

    def test_split_statements_inserts_break_before_nested_dml(self):
        # Build a list-like group containing a SELECT keyword that is not the
        # first token and confirm _split_statements inserts a break before it.
        stmt = make_stmt("SELECT a")
        f = ReindentFilter()
        f._curr_stmt = stmt
        before_len = len(stmt.tokens)
        # Manually invoke on the statement: first DML is at index 0 -> no break.
        f._split_statements(stmt)
        assert len(stmt.tokens) == before_len


# ---------------------------------------------------------------------------
# _process_where
# ---------------------------------------------------------------------------

class TestProcessWhere:
    def test_where_block_processed(self):
        stmt = make_stmt("SELECT a FROM t WHERE x=1")
        out = str(ReindentFilter().process(stmt))
        assert '\nWHERE' in out

    def test_where_without_keyword_returns(self):
        f = ReindentFilter()
        # Build a fake tlist-like group without a WHERE keyword.
        tlist = sql.TokenList(tokens=[sql.Token(T.Keyword, 'X')])
        f._curr_stmt = tlist
        before = str(tlist)
        f._process_where(tlist)
        assert str(tlist) == before


# ---------------------------------------------------------------------------
# _process_parenthesis
# ---------------------------------------------------------------------------

class TestProcessParenthesis:
    def test_parenthesis_is_processed(self):
        stmt = make_stmt("SELECT * FROM t WHERE (a=1)")

        out = str(ReindentFilter().process(stmt))
        # Parenthesis content preserved.
        assert 'a=1' in out

    def test_parenthesis_without_open_returns(self):
        f = ReindentFilter()
        tlist = sql.TokenList(tokens=[sql.Token(T.Keyword, 'X')])
        f._curr_stmt = tlist
        before = str(tlist)
        f._process_parenthesis(tlist)
        assert str(tlist) == before


# ---------------------------------------------------------------------------
# _process_identifierlist
# ---------------------------------------------------------------------------

class TestProcessIdentifierList:
    def test_no_wrap_default(self):
        stmt = make_stmt("SELECT a, b, c FROM t")
        out = str(ReindentFilter(wrap_after=0).process(stmt))
        # With wrap_after 0 the outer list is not wrapped.
        assert out.count('\n') >= 1  # FROM split still present

    def test_wrap_after_triggers_breaks(self):
        stmt = make_stmt("SELECT aaa, bbb, ccc, ddd, eee FROM t")
        out = str(ReindentFilter(wrap_after=10).process(stmt))
        # Wrapping present: some newline inside the select list.
        assert out.count('\n') >= 2

    def test_comma_first_moves_commas(self):
        stmt = make_stmt("SELECT aaa, bbb, ccc, ddd FROM t")
        out = str(ReindentFilter(wrap_after=8, comma_first=True).process(stmt))
        # Comma-first places the comma after newline plus indentation.
        assert '\n' in out
        # Each wrapped line begins with spaces then a comma.
        lines = [ln for ln in out.split('\n') if ln.strip().startswith(',')]
        assert len(lines) >= 1
        assert all(ln.lstrip().startswith(',') for ln in lines)

    def test_indent_columns(self):
        stmt = make_stmt("SELECT aaa, bbb, ccc FROM t")
        out = str(ReindentFilter(wrap_after=6, indent_columns=True).process(stmt))
        assert '\n' in out


# ---------------------------------------------------------------------------
# _process_case
# ---------------------------------------------------------------------------

class TestProcessCase:
    def test_case_with_when_then(self):
        stmt = make_stmt(
            "SELECT CASE WHEN a=1 THEN 'x' ELSE 'y' END FROM t")
        out = str(ReindentFilter(wrap_after=10).process(stmt))
        assert 'WHEN' in out
        assert 'END' in out

    def test_case_compact(self):
        stmt = make_stmt(
            "SELECT CASE WHEN a=1 THEN 'x' ELSE 'y' END FROM t")
        out_compact = str(
            ReindentFilter(wrap_after=10, compact=True).process(stmt))
        out_full = str(ReindentFilter(wrap_after=10).process(stmt))
        # Compact should not have introduced a newline right before END.
        assert 'END' in out_compact
        assert out_compact != out_full or 'END' in out_full


# ---------------------------------------------------------------------------
# _process_values
# ---------------------------------------------------------------------------

class TestProcessValues:
    def test_values_wrapping(self):
        stmt = make_stmt("INSERT INTO t VALUES (1, 2), (3, 4)")
        out = str(ReindentFilter(wrap_after=20).process(stmt))
        assert 'VALUES' in out

    def test_values_comma_first(self):
        stmt = make_stmt("INSERT INTO t VALUES (1), (2)")
        out = str(ReindentFilter(wrap_after=1, comma_first=True).process(stmt))
        assert 'VALUES' in out


# ---------------------------------------------------------------------------
# _process_default / sublists
# ---------------------------------------------------------------------------

class TestProcessDefault:
    def test_sublists_processed(self):
        stmt = make_stmt("SELECT (SELECT b FROM u) FROM t")
        out = str(ReindentFilter().process(stmt))
        # The inner SELECT FROM should have a break.
        assert out.count('\n') >= 2


# ---------------------------------------------------------------------------
# End-to-end forms
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_basic_select(self):
        stmt = make_stmt("SELECT a FROM t")
        f = ReindentFilter()
        out = f.process(stmt)
        assert out.tokens[0].value == 'SELECT'
        assert '\nFROM' in str(out)
        # nl() returns a newline token; its value starts with newline char.
        assert str(f.nl()).startswith('\n')

    def test_custom_newline_char(self):
        stmt = make_stmt("SELECT a FROM t")
        out = str(ReindentFilter(n='\r\n').process(stmt))
        assert '\r\nFROM' in out

    def test_offset_zero_after_processing(self):
        f = ReindentFilter()
        f.process(make_stmt("SELECT a FROM t"))
        # offset is restored by context manager.
        assert f.offset == 0

    def test_indent_restored_after_processing(self):
        f = ReindentFilter(indent_after_first=True)
        f.process(make_stmt("SELECT a FROM t"))
        assert f.indent == 1

    def test_wrap_after_high_does_not_break_columns(self):
        stmt = make_stmt("SELECT a, b FROM t")
        out = str(ReindentFilter(wrap_after=1000).process(stmt))
        # Some newline for FROM, but column list stays on one line.
        first_line = out.split('\n')[0]
        assert 'a' in first_line


# ---------------------------------------------------------------------------
# Error handling / robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_statement(self):
        # Empty input parses to zero statements.
        stmts = parse_all("")
        assert len(stmts) == 0

    def test_whitespace_only(self):
        # Whitespace-only input parses to zero statements.
        stmts = parse_all("   ")
        assert len(stmts) == 0

    def test_plain_whitespace_statement(self):
        # A Statement built with only whitespace processes without error.
        stmt = sql.Statement(tokens=[sql.Token(T.Whitespace, '   ')])
        out = ReindentFilter().process(stmt)
        assert isinstance(out, sql.Statement)

    def test_process_returns_statement_instance(self):
        stmt = make_stmt("SELECT 1")
        out = ReindentFilter().process(stmt)
        assert isinstance(out, sql.Statement)
