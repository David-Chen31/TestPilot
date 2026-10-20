import pytest

import sqlparse
from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_statement(text):
    """Parse SQL text into a single Statement."""
    stmts = sqlparse.parse(text)
    assert len(stmts) == 1
    return stmts[0]


def token_types(stmt):
    # tokens may be a generator (from _process); materialise it
    if not isinstance(stmt.tokens, list):
        stmt.tokens = list(stmt.tokens)
    return [tok.ttype for tok in stmt.tokens]


def token_values(stmt):
    if not isinstance(stmt.tokens, list):
        stmt.tokens = list(stmt.tokens)
    return [tok.value for tok in stmt.tokens]


# ---------------------------------------------------------------------------
# Tests for the base OutputFilter class
# ---------------------------------------------------------------------------

def test_output_filter_is_base_class():
    assert issubclass(OutputPythonFilter, OutputFilter)
    assert issubclass(OutputPHPFilter, OutputFilter)


def test_output_filter_default_varname():
    f = OutputFilter()
    assert f.varname == 'sql'
    assert f.count == 0


def test_output_filter_custom_varname():
    f = OutputFilter(varname='query')
    assert f.varname == 'query'
    assert f.count == 0


def test_output_filter_process_raises_not_implemented():
    f = OutputFilter()
    stmt = make_statement('select 1')
    with pytest.raises(NotImplementedError):
        f.process(stmt)


def test_output_filter_process_increments_count():
    f = OutputFilter()
    stmt = make_statement('select 1')
    with pytest.raises(NotImplementedError):
        f.process(stmt)
    assert f.count == 1


# ---------------------------------------------------------------------------
# Tests for OutputPythonFilter
# ---------------------------------------------------------------------------

def test_python_filter_varname_default():
    f = OutputPythonFilter()
    assert f.varname == 'sql'
    assert f.varname_prefix == ''


def test_python_filter_custom_varname():
    f = OutputPythonFilter(varname='q')
    assert f.varname == 'q'


def test_python_filter_simple_single_line():
    f = OutputPythonFilter()
    stmt = make_statement('select 1')
    result = f.process(stmt)
    assert result is stmt
    assert str(stmt) == "sql = 'select 1'"
    assert f.count == 1


def test_python_filter_multiline_uses_parens():
    f = OutputPythonFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    rendered = str(stmt)
    assert rendered.startswith("sql = ('")
    assert rendered.endswith("')")
    assert "\n" in rendered


def test_python_filter_multiline_first_line():
    f = OutputPythonFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    lines = str(stmt).splitlines()
    assert lines[0] == "sql = ('select * '"


def test_python_filter_multiline_second_line_indented():
    f = OutputPythonFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    lines = str(stmt).splitlines()
    # len('sql') + 4 == 7 spaces then opening quote; closing ) also on line
    assert lines[1] == "       'from foo')"


def test_python_filter_multiple_statements_numbered():
    f = OutputPythonFilter()
    stmt1 = make_statement('select 1')
    stmt2 = make_statement('select 2')
    f.process(stmt1)
    f.process(stmt2)
    # _process yields a leading '\n' into stmt1's token stream when count > 1?
    # Actually stmt1 is processed first (count==1), so no leading newline in
    # its own output; the leading newline is prepended by the *second* call's
    # _process into stmt2.  But note str(stmt1) includes the assignment.
    assert str(stmt1) == "sql = 'select 1'"
    assert str(stmt2) == "sql2 = 'select 2'"
    assert f.count == 2


def test_python_filter_second_statement_leading_newline():
    f = OutputPythonFilter()
    f.process(make_statement('select 1'))
    stmt2 = make_statement('select 2')
    f.process(stmt2)
    # tokens is a generator; materialise it
    stmt2.tokens = list(stmt2.tokens)
    assert stmt2.tokens[0].value == '\n'


def test_python_filter_escaping_single_quote():
    f = OutputPythonFilter()
    stmt = make_statement("select 'x'")
    f.process(stmt)
    rendered = str(stmt)
    # Original single quotes must be escaped inside generated literal
    assert "\\'" in rendered
    # Should not break out: exactly one leading and one trailing delimiter
    assert rendered.startswith("sql = '")
    assert rendered.endswith("'")


def test_python_filter_escaping_backslash():
    f = OutputPythonFilter()
    from sqlparse.sql import Token
    from sqlparse.tokens import Text
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Text, 'a\\b')]
    f.process(stmt)
    assert '\\\\' in str(stmt)


def test_python_filter_backslash_before_quote_escaped():
    from sqlparse.sql import Token
    from sqlparse.tokens import Text
    f = OutputPythonFilter()
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Text, "a\\'b")]
    f.process(stmt)
    rendered = str(stmt)
    # Backslash doubled and quote escaped
    assert '\\\\' in rendered
    assert "\\'" in rendered


def test_python_filter_trailing_newline_in_whitespace():
    # whitespace token containing '\n' followed by indentation
    from sqlparse.sql import Token
    from sqlparse.tokens import Whitespace
    f = OutputPythonFilter()
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Whitespace, '\n    ')]
    f.process(stmt)
    rendered = str(stmt)
    # Indentation preserved after newline
    assert '    ' in rendered


def test_python_filter_token_types():
    f = OutputPythonFilter()
    stmt = make_statement('select 1')
    f.process(stmt)
    # Note: has_nl is computed from original stmt, then tokens replaced by gen
    types = token_types(stmt)
    assert T.Name in types
    assert T.Operator in types
    assert T.Text in types


def test_python_filter_returns_same_statement():
    f = OutputPythonFilter()
    stmt = make_statement('select 1')
    result = f.process(stmt)
    assert result is stmt


# ---------------------------------------------------------------------------
# Tests for OutputPHPFilter
# ---------------------------------------------------------------------------

def test_php_filter_varname_prefix():
    f = OutputPHPFilter()
    assert f.varname_prefix == '$'
    assert f.varname == '$sql'
    assert f.count == 0


def test_php_filter_custom_varname():
    f = OutputPHPFilter(varname='query')
    assert f.varname == '$query'


def test_php_filter_simple_single_line():
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    result = f.process(stmt)
    assert result is stmt
    assert str(stmt) == '$sql = "select 1";'
    assert f.count == 1


def test_php_filter_multiline_uses_dot_equal():
    f = OutputPHPFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    rendered = str(stmt)
    assert '$sql .=' in rendered
    assert rendered.startswith('$sql  = "')
    assert rendered.endswith('";')


def test_php_filter_multiline_first_line():
    f = OutputPHPFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    lines = str(stmt).splitlines()
    assert lines[0] == '$sql  = "select * ";'


def test_php_filter_multiline_second_line():
    f = OutputPHPFilter()
    stmt = make_statement('select *\nfrom foo')
    f.process(stmt)
    lines = str(stmt).splitlines()
    assert lines[1] == '$sql .= "from foo";'


def test_php_filter_multiple_statements_numbered():
    f = OutputPHPFilter()
    stmt1 = make_statement('select 1')
    stmt2 = make_statement('select 2')
    f.process(stmt1)
    f.process(stmt2)
    assert str(stmt1) == '$sql = "select 1";'
    assert str(stmt2) == '$sql2 = "select 2";'
    assert f.count == 2


def test_php_filter_second_statement_leading_newline():
    f = OutputPHPFilter()
    f.process(make_statement('select 1'))
    stmt2 = make_statement('select 2')
    f.process(stmt2)
    stmt2.tokens = list(stmt2.tokens)
    assert stmt2.tokens[0].value == '\n'


def test_php_filter_escaping_double_quote():
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    from sqlparse.sql import Token
    from sqlparse.tokens import Text
    stmt.tokens = [Token(Text, 'a"b')]
    f.process(stmt)
    rendered = str(stmt)
    assert '\\"' in rendered
    assert rendered.startswith('$sql = "')
    assert rendered.endswith('";')


def test_php_filter_escaping_backslash():
    from sqlparse.sql import Token
    from sqlparse.tokens import Text
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Text, 'a\\b')]
    f.process(stmt)
    assert '\\\\' in str(stmt)


def test_php_filter_backslash_before_quote_escaped():
    from sqlparse.sql import Token
    from sqlparse.tokens import Text
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Text, 'a\\"b')]
    f.process(stmt)
    rendered = str(stmt)
    assert '\\\\' in rendered
    assert '\\"' in rendered


def test_php_filter_returns_same_statement():
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    result = f.process(stmt)
    assert result is stmt


def test_php_filter_token_types():
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    f.process(stmt)
    types = token_types(stmt)
    assert T.Name in types
    assert T.Operator in types
    assert T.Text in types
    assert T.Punctuation in types


def test_php_filter_whitespace_after_newline_preserved():
    from sqlparse.sql import Token
    from sqlparse.tokens import Whitespace
    f = OutputPHPFilter()
    stmt = make_statement('select 1')
    stmt.tokens = [Token(Whitespace, '\n    ')]
    f.process(stmt)
    assert '    ' in str(stmt)


# ---------------------------------------------------------------------------
# Cross-check via sqlparse public API
# ---------------------------------------------------------------------------

def test_via_sqlparse_format_python():
    out = sqlparse.format('select 1', output_format='python')
    assert out == "sql = 'select 1'"


def test_via_sqlparse_format_php():
    out = sqlparse.format('select 1', output_format='php')
    assert out == '$sql = "select 1";'
