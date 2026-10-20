import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def make_token(ttype, value):
    return sql.Token(ttype, value)


def make_stmt(*tokens):
    stmt = sql.Statement()
    stmt.tokens = list(tokens)
    return stmt


def render_tokens(tokens):
    return ''.join(str(t) for t in tokens)


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

class TestOutputFilterBase:
    def test_init_default_varname(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0

    def test_init_custom_varname(self):
        f = OutputFilter(varname='query')
        assert f.varname == 'query'
        assert f.count == 0

    def test_varname_prefix_used(self):
        class Sub(OutputFilter):
            varname_prefix = 'pre_'
        f = Sub(varname='x')
        assert f.varname == 'pre_x'

    def test_process_raises_not_implemented(self):
        f = OutputFilter()
        stmt = make_stmt(make_token(T.Text, 'select 1'))
        with pytest.raises(NotImplementedError):
            f.process(stmt)

    def test_process_increments_count(self):
        f = OutputFilter()
        stmt = make_stmt()
        with pytest.raises(NotImplementedError):
            f.process(stmt)
        assert f.count == 1


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

class TestOutputPythonFilter:
    def test_single_line(self):
        f = OutputPythonFilter()
        stmt = make_stmt(make_token(T.Keyword, 'SELECT 1'))
        result = f.process(stmt)
        assert str(result) == "sql = 'SELECT 1'"

    def test_varname_custom(self):
        f = OutputPythonFilter(varname='q')
        stmt = make_stmt(make_token(T.Keyword, 'SELECT 1'))
        result = f.process(stmt)
        assert str(result) == "q = 'SELECT 1'"

    def test_multiline_wraps_in_parens(self):
        f = OutputPythonFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'FROM'),
        )
        result = f.process(stmt)
        out = str(result)
        assert out.startswith("sql = (")
        assert out.endswith(")")
        assert '\n' in out

    def test_multiline_content(self):
        f = OutputPythonFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'FROM'),
        )
        out = str(f.process(stmt))
        # first line inside quote
        assert "sql = ('SELECT '" in out
        # continuation line indented by len(varname)+4 = 3+4 = 7
        assert "\n       'FROM" in out

    def test_escape_single_quote(self):
        f = OutputPythonFilter()
        stmt = make_stmt(make_token(T.Text, "it's"))
        out = str(f.process(stmt))
        assert out == "sql = 'it\\'s'"

    def test_escape_backslash(self):
        f = OutputPythonFilter()
        stmt = make_stmt(make_token(T.Text, "a\\b"))
        out = str(f.process(stmt))
        assert out == "sql = 'a\\\\b'"

    def test_escape_backslash_before_quote(self):
        f = OutputPythonFilter()
        stmt = make_stmt(make_token(T.Text, "a\\'b"))
        out = str(f.process(stmt))
        # backslash doubled then quote escaped
        assert "a\\\\\\'b" in out

    def test_second_call_gets_numbered_varname(self):
        f = OutputPythonFilter()
        stmt1 = make_stmt(make_token(T.Keyword, 'A'))
        stmt2 = make_stmt(make_token(T.Keyword, 'B'))
        out1 = str(f.process(stmt1))
        out2 = str(f.process(stmt2))
        assert out1 == "sql = 'A'"
        assert out2 == "\nsql2 = 'B'"
        assert f.count == 2

    def test_second_call_multiline_no_leading_newline_issue(self):
        f = OutputPythonFilter()
        stmt1 = make_stmt(make_token(T.Keyword, 'A'))
        stmt2 = make_stmt(
            make_token(T.Keyword, 'B'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'C'),
        )
        f.process(stmt1)
        out2 = str(f.process(stmt2))
        assert out2.startswith("\nsql2 = (")
        assert out2.endswith(")")

    def test_multiline_indentation_after_newline(self):
        f = OutputPythonFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '\n    '),
            make_token(T.Keyword, 'B'),
        )
        out = str(f.process(stmt))
        # after newline, indentation from token value "    " preserved
        assert "\n           '    B" in out or "\n       '    B" in out
        # continuation line prefix is len(varname)+4 spaces, then quote
        assert "       '    B" in out

    def test_multiline_no_indentation(self):
        f = OutputPythonFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'B'),
        )
        out = str(f.process(stmt))
        assert "\n       'B" in out

    def test_modifies_token_values_in_place(self):
        f = OutputPythonFilter()
        tok = make_token(T.Text, "it's")
        stmt = make_stmt(tok)
        f.process(stmt)
        assert tok.value == "it\\'s"

    def test_whitespace_without_newline_not_treated_as_break(self):
        f = OutputPythonFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, ' '),
            make_token(T.Keyword, 'B'),
        )
        out = str(f.process(stmt))
        assert out == "sql = 'A B'"
        assert out.count("'") == 2

    def test_empty_stream(self):
        f = OutputPythonFilter()
        stmt = make_stmt()
        out = str(f.process(stmt))
        assert out == "sql = ''"


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

class TestOutputPHPFilter:
    def test_single_line(self):
        f = OutputPHPFilter()
        stmt = make_stmt(make_token(T.Keyword, 'SELECT 1'))
        out = str(f.process(stmt))
        assert out == '$sql = "SELECT 1";'

    def test_varname_prefix(self):
        f = OutputPHPFilter(varname='q')
        assert f.varname == '$q'
        stmt = make_stmt(make_token(T.Keyword, 'SELECT 1'))
        out = str(f.process(stmt))
        assert out == '$q = "SELECT 1";'

    def test_multiline(self):
        f = OutputPHPFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'FROM'),
        )
        out = str(f.process(stmt))
        assert out.startswith('$sql =  "')
        assert out.endswith('";')
        assert ' ";' in out
        assert '$sql .= "' in out

    def test_multiline_indentation(self):
        f = OutputPHPFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '\n    '),
            make_token(T.Keyword, 'B'),
        )
        out = str(f.process(stmt))
        # continuation has indentation preserved after the quote header
        assert '$sql .= "    B' in out

    def test_escape_double_quote(self):
        f = OutputPHPFilter()
        stmt = make_stmt(make_token(T.Text, 'say "hi"'))
        out = str(f.process(stmt))
        assert out == '$sql = "say \\"hi\\"";'

    def test_escape_backslash(self):
        f = OutputPHPFilter()
        stmt = make_stmt(make_token(T.Text, "a\\b"))
        out = str(f.process(stmt))
        assert 'a\\\\b' in out

    def test_escape_backslash_before_quote(self):
        f = OutputPHPFilter()
        stmt = make_stmt(make_token(T.Text, 'a\\"b'))
        out = str(f.process(stmt))
        assert 'a\\\\\\"b' in out

    def test_second_call_numbered(self):
        f = OutputPHPFilter()
        stmt1 = make_stmt(make_token(T.Keyword, 'A'))
        stmt2 = make_stmt(make_token(T.Keyword, 'B'))
        out1 = str(f.process(stmt1))
        out2 = str(f.process(stmt2))
        assert out1 == '$sql = "A";'
        assert out2 == '\n$sql2 = "B";'
        assert f.count == 2

    def test_multiline_has_extra_space_before_eq(self):
        f = OutputPHPFilter()
        stmt = make_stmt(
            make_token(T.Keyword, 'A'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'B'),
        )
        out = str(f.process(stmt))
        assert '$sql =  "' in out

    def test_single_line_no_extra_space_before_eq(self):
        f = OutputPHPFilter()
        stmt = make_stmt(make_token(T.Keyword, 'A'))
        out = str(f.process(stmt))
        assert '$sql = "' in out
        assert '$sql =  "' not in out

    def test_empty_stream(self):
        f = OutputPHPFilter()
        stmt = make_stmt()
        out = str(f.process(stmt))
        assert out == '$sql = "";'

    def test_modifies_token_values_in_place(self):
        f = OutputPHPFilter()
        tok = make_token(T.Text, 'a"b')
        stmt = make_stmt(tok)
        f.process(stmt)
        assert tok.value == 'a\\"b'


# ---------------------------------------------------------------------------
# Return value / mutation contracts shared by both filters
# ---------------------------------------------------------------------------

class TestProcessContract:
    @pytest.mark.parametrize('cls', [OutputPythonFilter, OutputPHPFilter])
    def test_process_returns_same_statement_object(self, cls):
        f = cls()
        stmt = make_stmt(make_token(T.Keyword, 'A'))
        result = f.process(stmt)
        assert result is stmt

    @pytest.mark.parametrize('cls', [OutputPythonFilter, OutputPHPFilter])
    def test_process_sets_tokens(self, cls):
        f = cls()
        stmt = make_stmt(make_token(T.Keyword, 'A'))
        f.process(stmt)
        assert isinstance(stmt.tokens, list)
        assert len(stmt.tokens) > 0
        assert all(isinstance(t, sql.Token) for t in stmt.tokens)
</final>
