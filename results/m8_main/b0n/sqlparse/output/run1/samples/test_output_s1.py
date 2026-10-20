import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def _tokens_str(tokens):
    return ''.join(tok.value for tok in tokens)


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

class TestOutputFilter:
    def test_attributes_default(self):
        f = OutputFilter()
        assert f.varname == 'sql'
        assert f.count == 0

    def test_attributes_custom_varname(self):
        f = OutputFilter(varname='query')
        assert f.varname == 'query'
        assert f.count == 0

    def test_process_raises_not_implemented(self):
        f = OutputFilter()
        stmt = sql.Statement([sql.Token(T.Text, 'foo')])
        with pytest.raises(NotImplementedError):
            f.process(stmt)

    def test_process_increments_count(self):
        class Dummy(OutputFilter):
            def _process(self, stream, varname, has_nl):
                return list(stream)

        f = Dummy()
        stmt = sql.Statement([sql.Token(T.Text, 'x')])
        f.process(stmt)
        assert f.count == 1
        f.process(sql.Statement([sql.Token(T.Text, 'y')]))
        assert f.count == 2

    def test_process_varname_numbering(self):
        captured = []

        class Dummy(OutputFilter):
            def _process(self, stream, varname, has_nl):
                captured.append((varname, has_nl))
                return list(stream)

        f = Dummy(varname='q')
        f.process(sql.Statement([sql.Token(T.Text, 'a')]))
        f.process(sql.Statement([sql.Token(T.Text, 'b')]))
        f.process(sql.Statement([sql.Token(T.Text, 'c')]))
        assert [c[0] for c in captured] == ['q', 'q2', 'q3']

    def test_process_sets_tokens(self):
        class Dummy(OutputFilter):
            def _process(self, stream, varname, has_nl):
                return [sql.Token(T.Text, 'replaced')]

        f = Dummy()
        stmt = sql.Statement([sql.Token(T.Text, 'orig')])
        result = f.process(stmt)
        assert result is stmt
        assert _tokens_str(stmt.tokens) == 'replaced'

    def test_process_has_nl_detection(self):
        captured = []

        class Dummy(OutputFilter):
            def _process(self, stream, varname, has_nl):
                captured.append(has_nl)
                return list(stream)

        f = Dummy()
        f.process(sql.Statement([sql.Token(T.Text, 'single')]))
        # two separate token values but no newline -> still one line
        f.process(sql.Statement([sql.Token(T.Text, 'a'),
                                 sql.Token(T.Text, 'b')]))
        f.process(sql.Statement([sql.Token(T.Whitespace, '\n'),
                                 sql.Token(T.Text, 'c')]))
        assert captured == [False, False, True]

    def test_varname_prefix(self):
        class Prefixed(OutputFilter):
            varname_prefix = '@'

        f = Prefixed(varname='foo')
        assert f.varname == '@foo'

    def test_varname_prefix_default_empty(self):
        f = OutputFilter(varname='foo')
        assert f.varname == 'foo'


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

class TestOutputPythonFilter:
    def _make_stmt(self, *tokens):
        return sql.Statement(list(tokens))

    def test_single_line(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Keyword, 'SELECT'),
                               sql.Token(T.Whitespace, ' '),
                               sql.Token(T.Text, '1'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == "sql = 'SELECT 1'"

    def test_single_line_no_parens(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'x'))
        result = f.process(stmt)
        assert '(' not in _tokens_str(result.tokens)
        assert ')' not in _tokens_str(result.tokens)

    def test_multi_line_uses_parens(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        text = _tokens_str(result.tokens)
        assert text.startswith('sql = (')
        assert text.endswith(')')

    def test_multi_line_newline_handling(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        text = _tokens_str(result.tokens)
        # first line closed with " '" then newline and quote header spacing
        expected = (
            "sql = ('a '"
            "\n"
            "       'b'"
            ")"
        )
        # length of varname 'sql' = 3 -> 7 spaces indentation
        assert text == "sql = ('a '\n       'b')"

    def test_newline_with_indentation(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n    '),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        text = _tokens_str(result.tokens)
        assert text == "sql = ('a '\n           'b')"

    def test_newline_without_indentation_no_extra_ws(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        values = [t.value for t in result.tokens]
        # indentation whitespace after quote header should be empty
        assert '' not in values

    def test_escaping_single_quote(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, "it's"))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == "sql = 'it\\'s'"

    def test_escaping_backslash(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, "a\\b"))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == "sql = 'a\\\\b'"

    def test_escaping_backslash_before_quote(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, "a\\'b"))
        result = f.process(stmt)
        # backslash doubled first, then quote escaped
        assert _tokens_str(result.tokens) == "sql = 'a\\\\\\'b'"

    def test_second_statement_numbered(self):
        f = OutputPythonFilter()
        s1 = self._make_stmt(sql.Token(T.Text, 'one'))
        s2 = self._make_stmt(sql.Token(T.Text, 'two'))
        f.process(s1)
        r2 = f.process(s2)
        text = _tokens_str(r2.tokens)
        assert text == "\nsql2 = 'two'"

    def test_third_statement_numbered(self):
        f = OutputPythonFilter()
        for i in range(2):
            f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        r = f.process(self._make_stmt(sql.Token(T.Text, 'y')))
        assert _tokens_str(r.tokens) == "\nsql3 = 'y'"

    def test_no_leading_newline_first_stmt(self):
        f = OutputPythonFilter()
        r = f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        assert _tokens_str(r.tokens) == "sql = 'x'"

    def test_custom_varname(self):
        f = OutputPythonFilter(varname='myq')
        r = f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        assert _tokens_str(r.tokens) == "myq = 'x'"

    def test_varname_prefix_empty(self):
        assert OutputPythonFilter.varname_prefix == ''

    def test_token_types(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Keyword, 'SELECT'))
        result = f.process(stmt)
        types = [t.ttype for t in result.tokens]
        assert T.Name in types
        assert T.Operator in types
        assert T.Text in types

    def test_whitespace_without_newline_escaped_as_text(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '  '),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        # a space (no newline) is not a separator, it becomes text
        assert _tokens_str(result.tokens) == "sql = 'a  b'"

    def test_multiple_newlines(self):
        f = OutputPythonFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'c'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == (
            "sql = ('a '\n       'b '\n       'c')"
        )


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

class TestOutputPHPFilter:
    def _make_stmt(self, *tokens):
        return sql.Statement(list(tokens))

    def test_varname_prefix(self):
        assert OutputPHPFilter.varname_prefix == '$'

    def test_default_varname(self):
        f = OutputPHPFilter()
        assert f.varname == '$sql'

    def test_single_line(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Keyword, 'SELECT'),
                               sql.Token(T.Whitespace, ' '),
                               sql.Token(T.Text, '1'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == '$sql = "SELECT 1";'

    def test_single_line_no_extra_ws(self):
        f = OutputPHPFilter()
        r = f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        text = _tokens_str(r.tokens)
        assert text == '$sql = "x";'

    def test_multi_line(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        text = _tokens_str(result.tokens)
        # has_nl adds extra space after varname before '='; note $sql length=4
        expected = (
            '$sql  = "a ";'
            "\n"
            "$sql .= \"b\";"
        )
        assert text == (
            '$sql  = "a ";'
            "\n"
            '$sql .= "b";'
        )

    def test_multi_line_with_indentation(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n  '),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        text = _tokens_str(result.tokens)
        assert text == '$sql  = "a ";\n$sql .= "  b";'

    def test_escaping_double_quote(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'say "hi"'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == '$sql = "say \\"hi\\"";'

    def test_escaping_backslash(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a\\b'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == '$sql = "a\\\\b";'

    def test_escaping_backslash_before_quote(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a\\"b'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == '$sql = "a\\\\\\"b";'

    def test_second_statement_numbered(self):
        f = OutputPHPFilter()
        f.process(self._make_stmt(sql.Token(T.Text, 'one')))
        r = f.process(self._make_stmt(sql.Token(T.Text, 'two')))
        assert _tokens_str(r.tokens) == '\n$sql2 = "two";'

    def test_custom_varname(self):
        f = OutputPHPFilter(varname='q')
        assert f.varname == '$q'
        r = f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        assert _tokens_str(r.tokens) == '$q = "x";'

    def test_token_types(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Keyword, 'SELECT'))
        result = f.process(stmt)
        types = [t.ttype for t in result.tokens]
        assert T.Name in types
        assert T.Operator in types
        assert T.Text in types
        assert T.Punctuation in types

    def test_whitespace_without_newline_escaped_as_text(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '  '),
                               sql.Token(T.Text, 'b'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == '$sql = "a  b";'

    def test_multiple_newlines(self):
        f = OutputPHPFilter()
        stmt = self._make_stmt(sql.Token(T.Text, 'a'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'b'),
                               sql.Token(T.Whitespace, '\n'),
                               sql.Token(T.Text, 'c'))
        result = f.process(stmt)
        assert _tokens_str(result.tokens) == (
            '$sql  = "a ";'
            "\n"
            '$sql .= "b ";'
            "\n"
            '$sql .= "c";'
        )

    def test_ends_with_semicolon(self):
        f = OutputPHPFilter()
        r = f.process(self._make_stmt(sql.Token(T.Text, 'x')))
        assert r.tokens[-1].value == ';'
        assert r.tokens[-1].ttype == T.Punctuation
