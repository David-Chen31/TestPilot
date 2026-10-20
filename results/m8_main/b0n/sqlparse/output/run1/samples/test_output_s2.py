import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def _tokens_to_text(tokens):
    return ''.join(t.value for t in tokens)


def _build_stmt():
    return sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, ' '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

def test_output_filter_default_varname():
    f = OutputFilter()
    assert f.varname == 'sql'
    assert f.varname_prefix == ''
    assert f.count == 0


def test_output_filter_custom_varname():
    f = OutputFilter(varname='query')
    assert f.varname == 'query'
    assert f.count == 0


def test_output_filter_process_raises_not_implemented():
    f = OutputFilter()
    stmt = _build_stmt()
    with pytest.raises(NotImplementedError):
        f.process(stmt)


def test_output_filter_process_increments_count_before_raise():
    f = OutputFilter()
    stmt = _build_stmt()
    with pytest.raises(NotImplementedError):
        f.process(stmt)
    assert f.count == 1


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

def test_python_filter_varname():
    f = OutputPythonFilter()
    assert f.varname == 'sql'


def test_python_filter_custom_varname():
    f = OutputPythonFilter(varname='foo')
    assert f.varname == 'foo'


def test_python_filter_single_line():
    f = OutputPythonFilter()
    stmt = _build_stmt()
    result = f.process(stmt)
    assert result is stmt
    assert _tokens_to_text(stmt.tokens) == "sql = 'SELECT 1'"
    assert f.count == 1


def test_python_filter_single_line_token_types():
    f = OutputPythonFilter()
    stmt = _build_stmt()
    f.process(stmt)
    toks = list(stmt.tokens)
    assert (toks[0].ttype, toks[0].value) == (T.Name, 'sql')
    assert (toks[1].ttype, toks[1].value) == (T.Whitespace, ' ')
    assert (toks[2].ttype, toks[2].value) == (T.Operator, '=')
    assert (toks[3].ttype, toks[3].value) == (T.Whitespace, ' ')
    assert (toks[4].ttype, toks[4].value) == (T.Text, "'")
    # last token is closing quote
    assert (toks[-1].ttype, toks[-1].value) == (T.Text, "'")


def test_python_filter_multiline():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    text = _tokens_to_text(stmt.tokens)
    # parenthesized, lines wrapped
    assert text.startswith("sql = (")
    assert text.endswith(")")
    assert "'SELECT" in text
    assert "1'" in text
    # inner closing/opening quote markers exist
    assert "' " in text


def test_python_filter_multiline_has_parens():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    assert toks[4].ttype == T.Operator
    assert toks[4].value == '('
    assert toks[-1].ttype == T.Operator
    assert toks[-1].value == ')'


def test_python_filter_escapes_single_quote():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, "a'b"),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == "sql = 'a\\'b'"


def test_python_filter_escapes_backslash():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, 'a\\b'),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == "sql = 'a\\\\b'"


def test_python_filter_escapes_backslash_before_quote():
    # GHSA-3496-9g83-7v6x: backslash preceding quote must be escaped.
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, "\\'"),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == "sql = '\\\\\\''"


def test_python_filter_count_and_second_varname():
    f = OutputPythonFilter()
    stmt1 = _build_stmt()
    f.process(stmt1)
    stmt2 = _build_stmt()
    f.process(stmt2)
    assert f.count == 2
    assert _tokens_to_text(stmt2.tokens) == "sql2 = 'SELECT 1'"
    # second statement begins with newline
    toks = list(stmt2.tokens)
    assert toks[0].ttype == T.Whitespace
    assert toks[0].value == '\n'


def test_python_filter_indentation_preserved_after_newline():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    # find the whitespace token containing the indentation "    "
    indent_tokens = [t for t in toks if t.ttype == T.Whitespace and t.value == '    ']
    assert len(indent_tokens) == 1


def test_python_filter_secondary_line_quote_header_indentation():
    f = OutputPythonFilter(varname='sql')
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    # find whitespace token of length len('sql') + 4 = 7 next to a quote
    found = False
    for i, t in enumerate(toks):
        if t.ttype == T.Whitespace and t.value == ' ' * (len('sql') + 4):
            # next token should be an opening quote
            if i + 1 < len(toks) and toks[i + 1].value == "'":
                found = True
    assert found


def test_python_filter_newline_without_trailing_whitespace():
    f = OutputPythonFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    # no empty indentation token after newline even when after_lb empty
    assert all(t.value != '' for t in toks)


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

def test_php_filter_varname_prefix():
    f = OutputPHPFilter()
    assert f.varname == '$sql'


def test_php_filter_custom_varname():
    f = OutputPHPFilter(varname='query')
    assert f.varname == '$query'


def test_php_filter_single_line():
    f = OutputPHPFilter()
    stmt = _build_stmt()
    result = f.process(stmt)
    assert result is stmt
    assert _tokens_to_text(stmt.tokens) == '$sql = "SELECT 1";'


def test_php_filter_single_line_token_types():
    f = OutputPHPFilter()
    stmt = _build_stmt()
    f.process(stmt)
    toks = list(stmt.tokens)
    assert (toks[0].ttype, toks[0].value) == (T.Name, '$sql')
    assert (toks[1].ttype, toks[1].value) == (T.Whitespace, ' ')
    assert (toks[2].ttype, toks[2].value) == (T.Operator, '=')
    assert (toks[3].ttype, toks[3].value) == (T.Whitespace, ' ')
    assert (toks[4].ttype, toks[4].value) == (T.Text, '"')
    assert (toks[-2].ttype, toks[-2].value) == (T.Text, '"')
    assert (toks[-1].ttype, toks[-1].value) == (T.Punctuation, ';')


def test_php_filter_multiline():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    text = _tokens_to_text(stmt.tokens)
    assert text.startswith('$sql =  "')  # extra space for multiline
    assert text.endswith('";')
    # continuation assignment
    assert '$sql .=' in text


def test_php_filter_multiline_extra_space_after_varname():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    assert (toks[1].ttype, toks[1].value) == (T.Whitespace, ' ')
    # second whitespace after varname (extra) before '='
    assert (toks[2].ttype, toks[2].value) == (T.Whitespace, ' ')
    assert (toks[3].ttype, toks[3].value) == (T.Operator, '=')


def test_php_filter_escapes_double_quote():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, 'a"b'),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == '$sql = "a\\"b";'


def test_php_filter_escapes_backslash():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, 'a\\b'),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == '$sql = "a\\\\b";'


def test_php_filter_escapes_backslash_before_quote():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, '\\"'),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == '$sql = "\\\\\\"";'


def test_php_filter_does_not_escape_single_quote():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.String.Single, "a'b"),
    ])
    f.process(stmt)
    assert _tokens_to_text(stmt.tokens) == '$sql = "a\'b";'


def test_php_filter_count_and_second_varname():
    f = OutputPHPFilter()
    stmt1 = _build_stmt()
    f.process(stmt1)
    stmt2 = _build_stmt()
    f.process(stmt2)
    assert f.count == 2
    assert _tokens_to_text(stmt2.tokens) == '$sql2 = "SELECT 1";'
    toks = list(stmt2.tokens)
    assert toks[0].ttype == T.Whitespace
    assert toks[0].value == '\n'


def test_php_filter_secondary_line_header():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    # Find the continuation header sequence after closing "; newline
    found = False
    for i, t in enumerate(toks):
        if t.ttype == T.Name and t.value == '$sql' and i > 0:
            # ensure this is the continuation (preceded by Operator '.=' or after whitespace)
            seq = toks[i:i + 6]
            if len(seq) >= 5 and seq[2].value == '.=':
                found = True
    assert found


def test_php_filter_indentation_preserved():
    f = OutputPHPFilter()
    stmt = sql.Statement(tokens=[
        sql.Token(T.Keyword.DML, 'SELECT'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Literal.Number.Integer, '1'),
    ])
    f.process(stmt)
    toks = list(stmt.tokens)
    indent_tokens = [t for t in toks if t.ttype == T.Whitespace and t.value == '    ']
    assert len(indent_tokens) == 1


def test_both_filters_share_independent_counts():
    py = OutputPythonFilter()
    php = OutputPHPFilter()
    py.process(_build_stmt())
    php.process(_build_stmt())
    assert py.count == 1
    assert php.count == 1
