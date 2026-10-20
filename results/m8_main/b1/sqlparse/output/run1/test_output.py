import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def _make_token(token_type, value):
    return sql.Token(token_type, value)


def _make_stmt(tokens):
    return sql.Statement(tokens)


class TestOutputFilter:
    def test_abstract_process_raises(self):
        f = OutputFilter()
        with pytest.raises(NotImplementedError):
            f._process([], 'sql', False)

    def test_init_defaults(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0
        assert f.varname_prefix == ''

    def test_init_custom_varname(self):
        f = OutputFilter('foo')
        assert f.varname == 'foo'
        assert f.count == 0

    def test_process_calls_abstract_raises(self):
        f = OutputFilter()
        stmt = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        with pytest.raises(NotImplementedError):
            f.process(stmt)


class TestOutputPythonFilter:
    def test_single_line(self):
        f = OutputPythonFilter()
        stmt = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        result = f.process(stmt)
        assert str(result) == "sql = 'SELECT'"
        assert f.count == 1

    def test_count_increments(self):
        f = OutputPythonFilter()
        stmt1 = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        f.process(stmt1)
        assert f.count == 1
        stmt2 = _make_stmt([_make_token(T.Keyword, 'FROM')])
        f.process(stmt2)
        assert f.count == 2

    def test_second_call_uses_numbered_varname(self):
        f = OutputPythonFilter()
        stmt1 = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        f.process(stmt1)
        stmt2 = _make_stmt([_make_token(T.Keyword, 'FROM')])
        result = f.process(stmt2)
        assert str(result) == "\nsql2 = 'FROM'"

    def test_custom_varname(self):
        f = OutputPythonFilter('query')
        stmt = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        result = f.process(stmt)
        assert str(result) == "query = 'SELECT'"

    def test_multiline_wrapped_in_parens(self):
        f = OutputPythonFilter()
        tokens = [
            _make_token(T.Keyword, 'SELECT'),
            _make_token(T.Whitespace, '\n  '),
            _make_token(T.Keyword, 'FROM'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        # Indentation after newline is preserved and emitted after the quote.
        assert str(result) == "sql = ('SELECT '\n       '  FROM')"

    def test_escape_single_quote(self):
        f = OutputPythonFilter()
        stmt = _make_stmt([_make_token(T.String.Single, "'abc'")])
        result = f.process(stmt)
        assert str(result) == "sql = '\\'abc\\''"

    def test_escape_backslash_before_quote(self):
        f = OutputPythonFilter()
        stmt = _make_stmt([_make_token(T.Text, "\\'")])
        result = f.process(stmt)
        # backslash escaped first, then quote
        assert str(result) == "sql = '\\\\\\''"

    def test_escape_backslash(self):
        f = OutputPythonFilter()
        stmt = _make_stmt([_make_token(T.Text, '\\')])
        result = f.process(stmt)
        assert str(result) == "sql = '\\\\'"

    def test_multiple_newline_tokens(self):
        f = OutputPythonFilter()
        tokens = [
            _make_token(T.Keyword, 'A'),
            _make_token(T.Whitespace, '\n'),
            _make_token(T.Keyword, 'B'),
            _make_token(T.Whitespace, '\n'),
            _make_token(T.Keyword, 'C'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        assert str(result) == "sql = ('A '\n       'B '\n       'C')"

    def test_multiline_indentation_preserved(self):
        f = OutputPythonFilter()
        tokens = [
            _make_token(T.Keyword, 'A'),
            _make_token(T.Whitespace, '\n    '),
            _make_token(T.Keyword, 'B'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        assert str(result) == "sql = ('A '\n       '    B')"

    def test_has_nl_uses_wrapped(self):
        # has_nl is computed from str(stmt).strip() lines
        f = OutputPythonFilter()
        tokens = [
            _make_token(T.Keyword, 'A'),
            _make_token(T.Whitespace, '\n'),
            _make_token(T.Keyword, 'B'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        out = str(result)
        assert out.startswith("sql = ('")
        assert out.endswith("')")


class TestOutputPHPFilter:
    def test_varname_prefix(self):
        f = OutputPHPFilter()
        assert f.varname_prefix == '$'
        assert f.varname == '$sql'

    def test_single_line(self):
        f = OutputPHPFilter()
        stmt = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        result = f.process(stmt)
        assert str(result) == '$sql = "SELECT";'

    def test_custom_varname(self):
        f = OutputPHPFilter('query')
        stmt = _make_stmt([_make_token(T.Keyword, 'SELECT')])
        result = f.process(stmt)
        assert str(result) == '$query = "SELECT";'

    def test_second_call_numbered(self):
        f = OutputPHPFilter()
        f.process(_make_stmt([_make_token(T.Keyword, 'A')]))
        stmt = _make_stmt([_make_token(T.Keyword, 'B')])
        result = f.process(stmt)
        assert str(result) == '\n$sql2 = "B";'

    def test_multiline(self):
        f = OutputPHPFilter()
        tokens = [
            _make_token(T.Keyword, 'A'),
            _make_token(T.Whitespace, '\n'),
            _make_token(T.Keyword, 'B'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        assert str(result) == '$sql  = "A ";\n$sql .= "B";'

    def test_escape_double_quote(self):
        f = OutputPHPFilter()
        stmt = _make_stmt([_make_token(T.Text, '"abc"')])
        result = f.process(stmt)
        assert str(result) == '$sql = "\\"abc\\"";'

    def test_escape_backslash(self):
        f = OutputPHPFilter()
        stmt = _make_stmt([_make_token(T.Text, '\\')])
        result = f.process(stmt)
        assert str(result) == '$sql = "\\\\";'

    def test_escape_backslash_before_quote(self):
        f = OutputPHPFilter()
        stmt = _make_stmt([_make_token(T.Text, '\\"')])
        result = f.process(stmt)
        assert str(result) == '$sql = "\\\\\\"";'

    def test_multiline_indentation(self):
        f = OutputPHPFilter()
        tokens = [
            _make_token(T.Keyword, 'A'),
            _make_token(T.Whitespace, '\n    '),
            _make_token(T.Keyword, 'B'),
        ]
        stmt = _make_stmt(tokens)
        result = f.process(stmt)
        assert str(result) == '$sql  = "A ";\n$sql .= "    B";'
