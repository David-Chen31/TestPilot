import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def make_stmt(text):
    """Build a Statement whose top-level token is a single token of text."""
    stmt = sql.Statement()
    token = sql.Token(T.Text, text)
    stmt.tokens = [token]
    return stmt


def token_values(tokens):
    return [t.value for t in tokens]


def token_types(tokens):
    return [t.ttype for t in tokens]


class TestOutputFilterBase:
    def test_varname_prefix_empty(self):
        assert OutputFilter.varname_prefix == ''

    def test_init_default(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0

    def test_init_custom_varname(self):
        f = OutputFilter(varname='q')
        assert f.varname == 'q'
        assert f.count == 0

    def test_process_raises_not_implemented(self):
        f = OutputFilter()
        stmt = make_stmt('select 1')
        with pytest.raises(NotImplementedError):
            f.process(stmt)

    def test_process_increments_count(self):
        f = OutputFilter()
        stmt = make_stmt('select 1')
        with pytest.raises(NotImplementedError):
            f.process(stmt)
        assert f.count == 1


class TestOutputPythonFilter:
    def test_is_subclass(self):
        assert issubclass(OutputPythonFilter, OutputFilter)

    def test_single_line(self):
        f = OutputPythonFilter()
        stmt = make_stmt('select 1')
        result = f.process(stmt)
        assert result is stmt
        assert token_values(stmt.tokens) == [
            'sql', ' ', '=', ' ', "'", 'select 1', "'",
        ]
        assert token_types(stmt.tokens) == [
            T.Name, T.Whitespace, T.Operator, T.Whitespace,
            T.Text, T.Text, T.Text,
        ]

    def test_custom_varname(self):
        f = OutputPythonFilter(varname='query')
        stmt = make_stmt('select 1')
        f.process(stmt)
        assert token_values(stmt.tokens)[0] == 'query'

    def test_count_and_second_call(self):
        f = OutputPythonFilter()
        stmt1 = make_stmt('select 1')
        f.process(stmt1)
        assert token_values(stmt1.tokens)[0] == 'sql'

        stmt2 = make_stmt('select 2')
        f.process(stmt2)
        # Second call: leading newline and varname sql2
        assert token_values(stmt2.tokens)[0] == '\n'
        assert token_values(stmt2.tokens)[1] == 'sql2'
        assert f.count == 2

    def test_multiline(self):
        f = OutputPythonFilter()
        stmt = make_stmt('select 1\nfrom t')
        f.process(stmt)
        assert token_values(stmt.tokens) == [
            'sql', ' ', '=', ' ', '(', "'",
            'select 1',
            " '", '\n', '     ', "'",
            'from t',
            "'", ')',
        ]

    def test_quote_escaping(self):
        f = OutputPythonFilter()
        stmt = make_stmt("select 'a'")
        f.process(stmt)
        values = token_values(stmt.tokens)
        # Apostrophes in content are escaped
        assert "select \\'a\\'" in values

    def test_backslash_escaping(self):
        f = OutputPythonFilter()
        stmt = make_stmt('select \\')
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert 'select \\\\' in values

    def test_backslash_before_quote_escaping(self):
        # GHSA-3496-9g83-7v6x
        f = OutputPythonFilter()
        stmt = make_stmt("a\\'b")
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert "a\\\\\\'b" in values

    def test_tokens_marked_text(self):
        f = OutputPythonFilter()
        stmt = make_stmt('select 1')
        f.process(stmt)
        # All content tokens should be Text type
        for tok in stmt.tokens:
            assert isinstance(tok, sql.Token)

    def test_whitespace_without_newline_kept(self):
        f = OutputPythonFilter()
        stmt = sql.Statement()
        stmt.tokens = [
            sql.Token(T.Text, 'a'),
            sql.Token(T.Whitespace, ' '),
            sql.Token(T.Text, 'b'),
        ]
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert "a b" in values  # leading? content tokens preserved

    def test_newline_with_indentation(self):
        f = OutputPythonFilter()
        stmt = make_stmt('a\n    b')
        f.process(stmt)
        values = token_values(stmt.tokens)
        # indentation after newline preserved as whitespace token
        assert '    ' in values


class TestOutputPHPFilter:
    def test_is_subclass(self):
        assert issubclass(OutputPHPFilter, OutputFilter)

    def test_varname_prefix(self):
        assert OutputPHPFilter.varname_prefix == '$'

    def test_init(self):
        f = OutputPHPFilter()
        assert f.varname == '$sql'
        assert f.count == 0

    def test_single_line(self):
        f = OutputPHPFilter()
        stmt = make_stmt('select 1')
        result = f.process(stmt)
        assert result is stmt
        assert token_values(stmt.tokens) == [
            '$sql', ' ', '=', ' ', '"', 'select 1', '"', ';',
        ]
        assert token_types(stmt.tokens) == [
            T.Name, T.Whitespace, T.Operator, T.Whitespace,
            T.Text, T.Text, T.Text, T.Punctuation,
        ]

    def test_custom_varname(self):
        f = OutputPHPFilter(varname='query')
        stmt = make_stmt('select 1')
        f.process(stmt)
        assert token_values(stmt.tokens)[0] == '$query'

    def test_count_second_call(self):
        f = OutputPHPFilter()
        stmt1 = make_stmt('select 1')
        f.process(stmt1)
        stmt2 = make_stmt('select 2')
        f.process(stmt2)
        assert token_values(stmt2.tokens)[0] == '\n'
        assert token_values(stmt2.tokens)[1] == '$sql2'

    def test_multiline(self):
        f = OutputPHPFilter()
        stmt = make_stmt('select 1\nfrom t')
        f.process(stmt)
        assert token_values(stmt.tokens) == [
            '$sql', ' ', ' ', '=', ' ', '"',
            'select 1',
            ' ";', '\n', '$sql', ' ', '.=', ' ', '"',
            'from t',
            '"', ';',
        ]

    def test_double_quote_escaping(self):
        f = OutputPHPFilter()
        stmt = make_stmt('select "a"')
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert 'select \\"a\\"' in values

    def test_backslash_escaping(self):
        f = OutputPHPFilter()
        stmt = make_stmt('select \\')
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert 'select \\\\' in values

    def test_backslash_before_quote_escaping(self):
        # GHSA-3496-9g83-7v6x
        f = OutputPHPFilter()
        stmt = make_stmt('a\\"b')
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert 'a\\\\\\"b' in values

    def test_newline_with_indentation(self):
        f = OutputPHPFilter()
        stmt = make_stmt('a\n    b')
        f.process(stmt)
        values = token_values(stmt.tokens)
        assert '    ' in values
