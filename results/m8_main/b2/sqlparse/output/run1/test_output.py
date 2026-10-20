import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_token(ttype, value):
    return sql.Token(ttype, value)


def build_statement(*tokens):
    """Create a real sql.Statement-like object from tokens."""
    stmt = sql.Statement(tokens)
    return stmt


def result_string(stmt):
    return str(stmt)


# ---------------------------------------------------------------------------
# OutputFilter (abstract base)
# ---------------------------------------------------------------------------

class TestOutputFilter:

    def test_default_varname(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0
        assert f.varname_prefix == ''

    def test_custom_varname(self):
        f = OutputFilter(varname='query')
        assert f.varname == 'query'
        assert f.count == 0

    def test_process_raises_not_implemented(self):
        f = OutputFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT'))
        with pytest.raises(NotImplementedError):
            f.process(stmt)

    def test_process_increments_count_even_on_error(self):
        f = OutputFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT'))
        with pytest.raises(NotImplementedError):
            f.process(stmt)
        assert f.count == 1

    def test_count_increments_on_each_call(self):
        f = OutputFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT'))
        for expected in (1, 2, 3):
            with pytest.raises(NotImplementedError):
                f.process(stmt)
            assert f.count == expected


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

class TestOutputPythonFilter:

    def test_varname_prefix_empty(self):
        f = OutputPythonFilter()
        assert f.varname_prefix == ''
        assert f.varname == 'sql'

    def test_custom_varname(self):
        f = OutputPythonFilter(varname='q')
        assert f.varname == 'q'

    def test_single_line_output(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert str(out) == "sql = 'SELECT 1'"

    def test_varname_used_in_output(self):
        f = OutputPythonFilter(varname='myvar')
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert str(out) == "myvar = 'SELECT 1'"

    def test_second_call_uses_numbered_varname(self):
        f = OutputPythonFilter()
        stmt1 = build_statement(make_token(T.Keyword, 'SELECT 1'))
        stmt2 = build_statement(make_token(T.Keyword, 'SELECT 2'))
        out1 = f.process(stmt1)
        out2 = f.process(stmt2)
        # The implementation mutates the statement in place; out1 is the same
        # object as stmt1. After the second process() call, both statements
        # start with a newline because count > 1 at _process time.
        assert str(out1) == "\nsql = 'SELECT 1'"
        # Second statement is prefixed by a newline and uses sql2
        assert str(out2) == "\nsql2 = 'SELECT 2'"

    def test_third_call_uses_sql3(self):
        f = OutputPythonFilter()
        for _ in range(2):
            f.process(build_statement(make_token(T.Keyword, 'SELECT 1')))
        stmt3 = build_statement(make_token(T.Keyword, 'SELECT 3'))
        out3 = f.process(stmt3)
        assert str(out3) == "\nsql3 = 'SELECT 3'"

    def test_single_quotes_escaped(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.String.Single, "'abc'"))
        out = f.process(stmt)
        assert str(out) == "sql = '\\'abc\\''"

    def test_backslashes_escaped(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.String.Single, "a\\b"))
        out = f.process(stmt)
        assert str(out) == "sql = 'a\\\\b'"

    def test_backslash_before_quote_escaped(self):
        # The security fix: backslash before quote must not break out.
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Text, "\\'"))
        out = f.process(stmt)
        # backslash -> \\, quote -> \'  =>  \\\'
        assert str(out) == "sql = '\\\\\\''"

    def test_plain_double_quotes_not_escaped(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Text, '"abc"'))
        out = f.process(stmt)
        assert str(out) == "sql = '\"abc\"'"

    def test_no_newline_no_parentheses(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert '(' not in str(out)
        assert ')' not in str(out)

    def test_multiline_output_has_parentheses(self):
        f = OutputPythonFilter()
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Name, 'a'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        assert out.startswith('sql = (')
        assert out.endswith(')')
        assert "'" in out

    def test_multiline_indentation_and_quotes(self):
        f = OutputPythonFilter(varname='sql')
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n    '),
            make_token(T.Name, 'a'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        # After newline, quote header is indented by len(varname) + 3 spaces
        # (one space is stripped because the previous line ends with " '").
        assert "\n       '    a'" in out

    def test_multiline_preserves_indentation_after_newline(self):
        f = OutputPythonFilter(varname='sql')
        ws = make_token(T.Whitespace, '\n    ')
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            ws,
            make_token(T.Name, 'a'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        lines = out.split('\n')
        # second line should contain quote header then indentation then name
        assert 'a' in lines[1]

    def test_empty_statement(self):
        f = OutputPythonFilter()
        stmt = build_statement()
        out = f.process(stmt)
        assert str(out) == "sql = ''"

    def test_returns_same_statement_object(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert out is stmt

    def test_token_types_of_output(self):
        f = OutputPythonFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT'))
        f.process(stmt)
        types = [tok.ttype for tok in stmt.tokens]
        assert T.Name in types
        assert T.Operator in types
        assert T.Text in types


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

class TestOutputPHPFilter:

    def test_varname_prefix(self):
        f = OutputPHPFilter()
        assert f.varname_prefix == '$'
        assert f.varname == '$sql'

    def test_custom_varname(self):
        f = OutputPHPFilter(varname='q')
        assert f.varname == '$q'

    def test_single_line_output(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert str(out) == '$sql = "SELECT 1";'

    def test_custom_varname_in_output(self):
        f = OutputPHPFilter(varname='x')
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert str(out) == '$x = "SELECT 1";'

    def test_second_call_uses_dollar_sql2(self):
        f = OutputPHPFilter()
        f.process(build_statement(make_token(T.Keyword, 'SELECT 1')))
        out2 = f.process(build_statement(make_token(T.Keyword, 'SELECT 2')))
        assert str(out2) == '\n$sql2 = "SELECT 2";'

    def test_double_quotes_escaped(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Text, '"abc"'))
        out = f.process(stmt)
        assert str(out) == '$sql = "\\"abc\\"";'

    def test_backslashes_escaped(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Text, "a\\b"))
        out = f.process(stmt)
        assert str(out) == '$sql = "a\\\\b";'

    def test_backslash_before_quote_escaped(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Text, '\\"'))
        out = f.process(stmt)
        # backslash -> \\ ; quote -> \"  =>  \\\"  inside "..."
        assert str(out) == '$sql = "\\\\\\"";'

    def test_single_quotes_not_escaped(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Text, "'abc'"))
        out = f.process(stmt)
        assert str(out) == "$sql = \"'abc'\";"

    def test_multiline_output_has_extra_space_and_dot_equals(self):
        f = OutputPHPFilter(varname='sql')
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Name, 'a'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        # Header has extra space: "$sql  ="
        assert '$sql  =' in out
        # continuation uses .= concatenation
        assert '$sql .=' in out
        assert out.endswith(';')

    def test_multiline_linebreak_produces_concat(self):
        f = OutputPHPFilter()
        tokens = [
            make_token(T.Keyword, 'SELECT'),
            make_token(T.Whitespace, '\n    '),
            make_token(T.Name, 'a'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        lines = out.split('\n')
        # line 1 closes with ' ";'
        assert lines[0].rstrip().endswith('";')
        # line 2 begins with continuation header
        assert '.=' in lines[1]

    def test_empty_statement(self):
        f = OutputPHPFilter()
        stmt = build_statement()
        out = f.process(stmt)
        assert str(out) == '$sql = "";'

    def test_returns_same_statement_object(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT 1'))
        out = f.process(stmt)
        assert out is stmt

    def test_ends_with_punctuation_semicolon(self):
        f = OutputPHPFilter()
        stmt = build_statement(make_token(T.Keyword, 'SELECT'))
        # stmt.tokens after process() is a generator; materialize it.
        f.process(stmt)
        tokens = list(stmt.tokens)
        assert tokens[-1].ttype is T.Punctuation
        assert tokens[-1].value == ';'


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

class TestFiltersTogether:

    def test_independent_counts(self):
        py = OutputPythonFilter()
        php = OutputPHPFilter()
        stmt1 = build_statement(make_token(T.Keyword, 'SELECT 1'))
        stmt2 = build_statement(make_token(T.Keyword, 'SELECT 1'))
        py.process(stmt1)
        py.process(stmt2)
        assert py.count == 2
        assert php.count == 0

    def test_has_nl_detection(self):
        # A statement whose stripped string spans multiple lines must be
        # treated as multiline (parentheses for python filter).
        f = OutputPythonFilter()
        tokens = [
            make_token(T.Keyword, 'a'),
            make_token(T.Whitespace, '\n'),
            make_token(T.Keyword, 'b'),
        ]
        stmt = build_statement(*tokens)
        out = str(f.process(stmt))
        assert out.startswith('sql = (')

    def test_is_whitespace_newline_token_handled(self):
        # Whitespace token containing newline triggers continuation logic.
        f = OutputPythonFilter()
        ws = make_token(T.Whitespace, '\n')
        assert ws.is_whitespace
        stmt = build_statement(
            make_token(T.Keyword, 'a'), ws, make_token(T.Keyword, 'b'))
        out = str(f.process(stmt))
        assert '\n' in out
