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

def make_stmt(*tokens):
    """Build a sql.Statement whose tokens are the given tokens/token-lists."""
    stmt = sql.Statement()
    stmt.tokens = list(tokens)
    return stmt


def tok(ttype, value):
    return sql.Token(ttype, value)


def render(tokens):
    return "".join(str(t) for t in tokens)


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

def test_output_filter_init_defaults():
    f = OutputFilter()
    assert f.varname == "sql"
    assert f.count == 0
    assert f.varname_prefix == ""


def test_output_filter_init_custom_varname():
    f = OutputFilter(varname="query")
    assert f.varname == "query"
    assert f.count == 0


def test_output_filter_process_raises_not_implemented():
    f = OutputFilter()
    stmt = make_stmt(tok(T.Text, "select 1"))
    with pytest.raises(NotImplementedError):
        f.process(stmt)
    # count was incremented before the error
    assert f.count == 1
    assert stmt.tokens  # untouched


def test_output_filter_process_increments_count():
    class Recorder(OutputFilter):
        def _process(self, stream, varname, has_nl):
            return list(stream)

    f = Recorder()
    stmt = make_stmt(tok(T.Text, "x"))
    f.process(stmt)
    assert f.count == 1
    f.process(stmt)
    assert f.count == 2


def test_output_filter_varname_prefix_applies():
    class Prefixed(OutputFilter):
        varname_prefix = ">>"

        def _process(self, stream, varname, has_nl):
            return list(stream)

    f = Prefixed(varname="abc")
    assert f.varname == ">>abc"


def test_output_filter_multiple_calls_varname_numbering():
    captured = []

    class Capture(OutputFilter):
        def _process(self, stream, varname, has_nl):
            captured.append((varname, has_nl))
            return list(stream)

    f = Capture()
    f.process(make_stmt(tok(T.Text, "a")))
    f.process(make_stmt(tok(T.Text, "b")))
    f.process(make_stmt(tok(T.Text, "c")))
    assert captured == [("sql", False), ("sql2", False), ("sql3", False)]


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

def test_python_filter_single_line():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"), tok(T.Whitespace, " "),
                     tok(T.Text, "*"))
    result = f.process(stmt)
    assert render(result.tokens) == "sql = 'SELECT *'"


def test_python_filter_custom_varname():
    f = OutputPythonFilter(varname="q")
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    assert render(result.tokens) == "q = 'SELECT'"


def test_python_filter_multiline():
    f = OutputPythonFilter()
    stmt = make_stmt(
        tok(T.Keyword, "SELECT"),
        tok(T.Whitespace, " "),
        tok(T.Whitespace, "\n"),
        tok(T.Keyword, "FROM"),
    )
    result = f.process(stmt)
    assert render(result.tokens) == "sql = ('SELECT '\n       'FROM')"


def test_python_filter_multiline_indentation():
    f = OutputPythonFilter()
    # '\n  ' newline with trailing indentation
    stmt = make_stmt(
        tok(T.Keyword, "SELECT"),
        tok(T.Whitespace, "\n  "),
        tok(T.Keyword, "FROM"),
    )
    result = f.process(stmt)
    # varname 'sql' -> 3 + 4 = 7 spaces header
    assert render(result.tokens) == "sql = ('SELECT '\n       '  FROM')"


def test_python_filter_multiple_statements_numbering():
    f = OutputPythonFilter()
    stmt1 = make_stmt(tok(T.Keyword, "SELECT"))
    stmt2 = make_stmt(tok(T.Keyword, "FROM"))
    r1 = f.process(stmt1)
    r2 = f.process(stmt2)
    assert render(r1.tokens) == "sql = 'SELECT'"
    assert render(r2.tokens) == "\nsql2 = 'FROM'"


def test_python_filter_escapes_single_quote():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Text, "O'Reilly"))
    result = f.process(stmt)
    assert render(result.tokens) == "sql = 'O\\'Reilly'"


def test_python_filter_escapes_backslash():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Text, "a\\b"))
    result = f.process(stmt)
    assert render(result.tokens) == "sql = 'a\\\\b'"


def test_python_filter_escapes_backslash_before_quote():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Text, "\\'"))
    result = f.process(stmt)
    # backslash doubled, then single quote escaped
    assert render(result.tokens) == "sql = '\\\\\\''"


def test_python_filter_does_not_escape_double_quote():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Text, 'a"b'))
    result = f.process(stmt)
    assert render(result.tokens) == "sql = 'a\"b'"


def test_python_filter_returns_same_stmt_object():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    assert result is stmt


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

def test_php_filter_single_line():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"), tok(T.Whitespace, " "),
                     tok(T.Text, "*"))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "SELECT *";'


def test_php_filter_varname_prefix():
    f = OutputPHPFilter(varname="q")
    assert f.varname == "$q"
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    assert render(result.tokens) == '$q = "SELECT";'


def test_php_filter_multiline_header():
    f = OutputPHPFilter()
    stmt = make_stmt(
        tok(T.Keyword, "SELECT"),
        tok(T.Whitespace, " \n"),
        tok(T.Keyword, "FROM"),
    )
    result = f.process(stmt)
    # has_nl True -> extra whitespace after varname
    assert render(result.tokens) == (
        '$sql  = "SELECT "\n'
        '$sql .= "FROM";'
    )


def test_php_filter_multiline_indentation():
    f = OutputPHPFilter()
    stmt = make_stmt(
        tok(T.Keyword, "SELECT"),
        tok(T.Whitespace, "\n    "),
        tok(T.Keyword, "FROM"),
    )
    result = f.process(stmt)
    assert render(result.tokens) == (
        '$sql  = "SELECT "\n'
        '$sql .= "    FROM";'
    )


def test_php_filter_multiple_statements_numbering():
    f = OutputPHPFilter()
    r1 = f.process(make_stmt(tok(T.Keyword, "SELECT")))
    r2 = f.process(make_stmt(tok(T.Keyword, "FROM")))
    assert render(r1.tokens) == '$sql = "SELECT";'
    assert render(r2.tokens) == '\n$sql2 = "FROM";'


def test_php_filter_escapes_double_quote():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Text, 'a"b'))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "a\\"b";'


def test_php_filter_escapes_backslash():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Text, "a\\b"))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "a\\\\b";'


def test_php_filter_escapes_backslash_before_quote():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Text, '\\"'))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "\\\\\\"";'


def test_php_filter_does_not_escape_single_quote():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Text, "O'Reilly"))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "O\'Reilly";'


def test_php_filter_returns_same_stmt_object():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    assert result is stmt


# ---------------------------------------------------------------------------
# Token type checks (side effects on produced tokens)
# ---------------------------------------------------------------------------

def test_python_filter_token_types():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    types = [t.ttype for t in result.tokens]
    assert types == [T.Name, T.Whitespace, T.Operator, T.Whitespace, T.Text, T.Text]


def test_php_filter_token_types():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Keyword, "SELECT"))
    result = f.process(stmt)
    types = [t.ttype for t in result.tokens]
    assert types == [
        T.Name, T.Whitespace, T.Operator, T.Whitespace,
        T.Text, T.Text, T.Punctuation,
    ]


def test_python_filter_whitespace_without_newline_not_split():
    f = OutputPythonFilter()
    stmt = make_stmt(tok(T.Keyword, "A"), tok(T.Whitespace, " "),
                     tok(T.Keyword, "B"))
    result = f.process(stmt)
    assert render(result.tokens) == "sql = 'A B'"


def test_php_filter_whitespace_without_newline_not_split():
    f = OutputPHPFilter()
    stmt = make_stmt(tok(T.Keyword, "A"), tok(T.Whitespace, " "),
                     tok(T.Keyword, "B"))
    result = f.process(stmt)
    assert render(result.tokens) == '$sql = "A B";'
