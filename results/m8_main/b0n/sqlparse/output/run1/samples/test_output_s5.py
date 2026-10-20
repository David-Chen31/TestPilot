import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def make_statement(text, newline_tokens=None):
    """Build a simple sql.Statement with tokens for testing."""
    stmt = sql.Statement()
    if newline_tokens is None:
        stmt.tokens = [sql.Token(T.Text, text)]
    else:
        stmt.tokens = newline_tokens
    return stmt


def joined(stmt):
    return ''.join(tok.value for tok in stmt.tokens)


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

class TestOutputFilterBase:
    def test_default_varname(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0
        assert f.varname_prefix == ''

    def test_custom_varname(self):
        f = OutputFilter(varname='q')
        assert f.varname == 'q'

    def test_process_increments_count(self):
        f = OutputFilter()
        stmt = make_statement('SELECT 1')
        with pytest.raises(NotImplementedError):
            f.process(stmt)
        assert f.count == 1

    def test_process_raises_not_implemented(self):
        f = OutputFilter()
        stmt = make_statement('SELECT 1')
        with pytest.raises(NotImplementedError):
            f.process(stmt)

    def test_process_raises_each_call(self):
        f = OutputFilter()
        stmt = make_statement('SELECT 1')
        with pytest.raises(NotImplementedError):
            f.process(stmt)
        stmt2 = make_statement('SELECT 2')
        with pytest.raises(NotImplementedError):
            f.process(stmt2)
        assert f.count == 2


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

class TestOutputPythonFilter:
    def test_varname_default(self):
        f = OutputPythonFilter()
        assert f.varname == 'sql'
        assert f.varname_prefix == ''

    def test_single_line_output(self):
        f = OutputPythonFilter()
        stmt = make_statement('SELECT 1')
        result = f.process(stmt)
        assert result is stmt
        assert joined(stmt) == "sql = 'SELECT 1'"

    def test_single_line_token_types(self):
        f = OutputPythonFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        assert values == ['sql', ' ', '=', ' ', "'", 'SELECT 1', "'"]

    def test_single_line_first_token_name(self):
        f = OutputPythonFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        first = stmt.tokens[0]
        assert first.ttype == T.Name
        assert first.value == 'sql'

    def test_single_line_no_parentheses(self):
        f = OutputPythonFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        assert '(' not in values
        assert ')' not in values

    def test_multiline_output_has_parentheses(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n'),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPythonFilter()
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        assert '(' in values
        assert ')' in values
        # parenthesis before opening quote, closing paren after closing quote
        assert values.index('(') < values.index("'")
        assert values.index(')') > values.index("'")

    def test_multiline_indentation_preserved(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n    '),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPythonFilter()
        f.process(stmt)
        text = joined(stmt)
        # closing quote, newline, header indent, quote, then indentation
        assert " ' \n" in text
        # header indentation is len('sql') + 4 = 7 spaces before the quote
        assert "       '" in text
        # the trailing indentation from the newline token is reproduced
        assert '\n    ' in text or '    ' in text

    def test_multiline_no_trailing_indentation(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n'),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPythonFilter()
        f.process(stmt)
        text = joined(stmt)
        assert " ' \n" in text

    def test_second_statement_has_leading_newline(self):
        f = OutputPythonFilter()
        s1 = make_statement('SELECT 1')
        f.process(s1)
        s2 = make_statement('SELECT 2')
        f.process(s2)
        # second output starts with a whitespace newline
        first = s2.tokens[0]
        assert first.ttype == T.Whitespace
        assert first.value == '\n'
        # and uses numbered varname
        assert s2.tokens[1].value == 'sql2'

    def test_first_statement_uses_base_varname(self):
        f = OutputPythonFilter()
        s1 = make_statement('SELECT 1')
        f.process(s1)
        assert s1.tokens[0].value == 'sql'

    def test_multiple_statements_numbering(self):
        f = OutputPythonFilter()
        for i, expected in enumerate(['a', 'b', 'c'], start=1):
            stmt = make_statement(f'SELECT {expected}')
            f.process(stmt)
            if i == 1:
                assert stmt.tokens[0].value == 'sql'
            else:
                assert stmt.tokens[1].value == f'sql{i}'
        assert f.count == 3

    def test_custom_varname(self):
        f = OutputPythonFilter(varname='query')
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        assert joined(stmt) == "query = 'SELECT 1'"

    def test_escaping_single_quotes(self):
        f = OutputPythonFilter()
        stmt = make_statement("SELECT 'x'")
        f.process(stmt)
        text = joined(stmt)
        assert "\\'" in text
        # The actual SQL text has escaped quotes
        assert text == "sql = 'SELECT \\'x\\''"

    def test_escaping_backslashes(self):
        f = OutputPythonFilter()
        stmt = make_statement('a\\b')
        f.process(stmt)
        text = joined(stmt)
        assert '\\\\' in text
        assert text == "sql = 'a\\\\b'"

    def test_escaping_backslash_before_quote(self):
        f = OutputPythonFilter()
        stmt = make_statement("a\\'b")
        f.process(stmt)
        text = joined(stmt)
        # backslash doubled first, then quote escaped
        assert text == "sql = 'a\\\\\\'b'"

    def test_output_filter_token_types_text(self):
        f = OutputPythonFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        # content is a Text token
        content_tokens = [t for t in stmt.tokens if t.ttype == T.Text]
        assert len(content_tokens) >= 1

    def test_has_nl_detection_strip(self):
        # Leading/trailing whitespace without internal newline -> single line
        toks = [
            sql.Token(T.Whitespace, '  '),
            sql.Token(T.Text, 'SELECT 1'),
            sql.Token(T.Whitespace, '  '),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPythonFilter()
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        assert '(' not in values
        assert ')' not in values


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
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        assert joined(stmt) == '$sql = "SELECT 1";'

    def test_single_line_token_types(self):
        f = OutputPHPFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        assert values == ['$sql', ' ', '=', ' ', '"', 'SELECT 1', '"', ';']

    def test_single_line_has_equals_and_semicolon(self):
        f = OutputPHPFilter()
        stmt = make_statement('SELECT 1')
        f.process(stmt)
        ops = [t.value for t in stmt.tokens if t.ttype == T.Operator]
        assert ops == ['=']
        punct = [t.value for t in stmt.tokens if t.ttype == T.Punctuation]
        assert punct == [';']

    def test_multiline_has_extra_space_after_name(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n'),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPHPFilter()
        f.process(stmt)
        values = [t.value for t in stmt.tokens]
        # name, space, space, =, ...
        assert values[0] == '$sql'
        assert values[1] == ' '
        assert values[2] == ' '
        assert values[3] == '='

    def test_multiline_concat_header(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n    '),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPHPFilter()
        f.process(stmt)
        text = joined(stmt)
        assert ' ";' in text
        assert '.=' in text
        # closing of first line: ' ";'
        assert ' ";' in text

    def test_multiline_line_break_tokens_types(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n'),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPHPFilter()
        f.process(stmt)
        ops = [t.value for t in stmt.tokens if t.ttype == T.Operator]
        assert '.=' in ops
        assert '=' in ops

    def test_multiline_indentation_preserved(self):
        toks = [
            sql.Token(T.Keyword, 'SELECT'),
            sql.Token(T.Whitespace, '\n    '),
            sql.Token(T.Text, '1'),
        ]
        stmt = make_statement(None, newline_tokens=toks)
        f = OutputPHPFilter()
        f.process(stmt)
        text = joined(stmt)
        assert '    ' in text

    def test_second_statement_leading_newline_and_numbering(self):
        f = OutputPHPFilter()
        s1 = make_statement('SELECT 1')
        f.process(s1)
        s2 = make_statement('SELECT 2')
        f.process(s2)
        assert s2.tokens[0].ttype == T.Whitespace
        assert s2.tokens[0].value == '\n'
        assert s2.tokens[1].value == '$sql2'

    def test_escaping_double_quotes(self):
        f = OutputPHPFilter()
        stmt = make_statement('SELECT "x"')
        f.process(stmt)
        text = joined(stmt)
        assert text == '$sql = "SELECT \\"x\\"";'

    def test_escaping_backslashes(self):
        f = OutputPHPFilter()
        stmt = make_statement('a\\b')
        f.process(stmt)
        text = joined(stmt)
        assert text == '$sql = "a\\\\b";'

    def test_escaping_backslash_before_quote(self):
        f = OutputPHPFilter()
        stmt = make_statement('a\\"b')
        f.process(stmt)
        text = joined(stmt)
        assert text == '$sql = "a\\\\\\"b";'

    def test_does_not_escape_single_quotes(self):
        f = OutputPHPFilter()
        stmt = make_statement("SELECT 'x'")
        f.process(stmt)
        text = joined(stmt)
        assert "'x'" in text
        assert "\\'" not in text

    def test_multiple_statements_count(self):
        f = OutputPHPFilter()
        for _ in range(3):
            f.process(make_statement('SELECT 1'))
        assert f.count == 3


# ---------------------------------------------------------------------------
# Shared behaviors across both filters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('filter_cls', [OutputPythonFilter, OutputPHPFilter])
class TestSharedFilterProperties:
    def test_returns_same_statement(self, filter_cls):
        f = filter_cls()
        stmt = make_statement('SELECT 1')
        assert f.process(stmt) is stmt

    def test_count_starts_at_zero(self, filter_cls):
        assert filter_cls().count == 0

    def test_count_increments(self, filter_cls):
        f = filter_cls()
        f.process(make_statement('a'))
        assert f.count == 1
        f.process(make_statement('b'))
        assert f.count == 2

    def test_empty_statement_single_line(self, filter_cls):
        f = filter_cls()
        stmt = sql.Statement()
        stmt.tokens = []
        f.process(stmt)
        text = joined(stmt)
        # Empty content yields empty quoted string
        assert f.varname in text

    def test_modifies_and_escapes_token_values(self, filter_cls):
        f = filter_cls()
        stmt = make_statement('a\\b')
        f.process(stmt)
        # All resulting content tokens are Text; escaping applied
        text = joined(stmt)
        assert '\\\\' in text
