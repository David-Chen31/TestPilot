import pytest

from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.filters.output import (
    OutputFilter,
    OutputPythonFilter,
    OutputPHPFilter,
)


def make_stmt(tokens):
    stmt = sql.Statement(tokens)
    return stmt


def tokens_from(stmt):
    return stmt.tokens


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

def test_output_filter_varname_default():
    f = OutputFilter()
    assert f.varname == 'sql'
    assert f.count == 0


def test_output_filter_varname_custom():
    f = OutputFilter(varname='query')
    assert f.varname == 'query'


def test_output_filter_varname_prefix():
    class Prefixed(OutputFilter):
        varname_prefix = 'prefix_'

    f = Prefixed(varname='x')
    assert f.varname == 'prefix_x'


def test_output_filter_process_not_implemented():
    f = OutputFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    with pytest.raises(NotImplementedError):
        f.process(stmt)


def test_output_filter_process_increments_count_and_assigns_varname():
    calls = {}

    class Dummy(OutputFilter):
        def _process(self, stream, varname, has_nl):
            calls['varname'] = varname
            calls['has_nl'] = has_nl
            calls['count'] = self.count
            return list(stream)

    f = Dummy()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])

    result = f.process(stmt)
    assert result is stmt
    assert f.count == 1
    assert calls['varname'] == 'sql'
    assert calls['count'] == 1
    # single line -> has_nl False
    assert calls['has_nl'] is False

    f.process(stmt)
    assert f.count == 2
    assert calls['varname'] == 'sql2'


def test_output_filter_has_nl_true_for_multiline():
    seen = {}

    class Dummy(OutputFilter):
        def _process(self, stream, varname, has_nl):
            seen['has_nl'] = has_nl
            return list(stream)

    f = Dummy()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1\nFROM t')])
    f.process(stmt)
    assert seen['has_nl'] is True


# ---------------------------------------------------------------------------
# Helpers to render output as a list of (token_type, value)
# ---------------------------------------------------------------------------

def render(stmt):
    return [(t.ttype, t.value) for t in stmt.tokens]


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

def test_python_filter_simple_single_line():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    out = f.process(stmt)
    assert render(out) == [
        (T.Name, 'sql'),
        (T.Whitespace, ' '),
        (T.Operator, '='),
        (T.Whitespace, ' '),
        (T.Text, "'"),
        (T.Text, 'SELECT 1'),
        (T.Text, "'"),
    ]


def test_python_filter_varname_custom():
    f = OutputPythonFilter(varname='q')
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    out = f.process(stmt)
    ttypes_values = render(out)
    assert (T.Name, 'q') in ttypes_values


def test_python_filter_multiline_wraps_with_parens():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1\nFROM t')])
    out = f.process(stmt)
    rendered = render(out)
    # Opening paren present (has_nl True)
    assert (T.Operator, '(') in rendered
    assert (T.Operator, ')') in rendered
    # Leading name
    assert rendered[0] == (T.Name, 'sql')


def test_python_filter_multiline_structure():
    f = OutputPythonFilter()
    stmt = make_stmt([
        sql.Token(T.Text, 'SELECT 1'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Text, 'FROM t'),
    ])
    out = f.process(stmt)
    rendered = render(out)

    # First line: sql = ('
    assert rendered[0] == (T.Name, 'sql')
    assert (T.Operator, '(') in rendered
    assert rendered[4] == (T.Text, "'")

    # The whitespace token containing newline is replaced by " '" + newline
    assert (T.Text, " '") in rendered
    assert (T.Whitespace, '\n') in rendered
    # Quote header on secondary line: whitespace of len(varname)+4 = 3+4=7
    assert (T.Whitespace, ' ' * (len('sql') + 4)) in rendered

    # Closing
    assert rendered[-1] == (T.Operator, ')')


def test_python_filter_multiline_indentation_preserved():
    f = OutputPythonFilter()
    stmt = make_stmt([
        sql.Token(T.Text, 'SELECT 1'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Text, 'FROM t'),
    ])
    out = f.process(stmt)
    rendered = render(out)
    # after_lb should be '    '
    assert (T.Whitespace, '    ') in rendered


def test_python_filter_escapes_single_quote():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, "O'Reilly")])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Text, "O\\'Reilly") in rendered


def test_python_filter_escapes_backslash():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, "a\\b")])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Text, "a\\\\b") in rendered


def test_python_filter_escapes_backslash_before_quote():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, "a\\'b")])
    out = f.process(stmt)
    rendered = render(out)
    # backslash and quote both escaped: a -> \\ then ' -> \'
    assert (T.Text, "a\\\\\\'b") in rendered


def test_python_filter_second_call_adds_newline_and_numbered_varname():
    f = OutputPythonFilter()
    stmt1 = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    f.process(stmt1)

    stmt2 = make_stmt([sql.Token(T.Text, 'SELECT 2')])
    out = f.process(stmt2)
    rendered = render(out)
    assert rendered[0] == (T.Whitespace, '\n')
    assert rendered[1] == (T.Name, 'sql2')


def test_python_filter_first_call_no_leading_newline():
    f = OutputPythonFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    out = f.process(stmt)
    assert render(out)[0] == (T.Name, 'sql')


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

def test_php_filter_varname_prefix():
    f = OutputPHPFilter()
    assert f.varname == '$sql'


def test_php_filter_simple_single_line():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    out = f.process(stmt)
    assert render(out) == [
        (T.Name, '$sql'),
        (T.Whitespace, ' '),
        (T.Operator, '='),
        (T.Whitespace, ' '),
        (T.Text, '"'),
        (T.Text, 'SELECT 1'),
        (T.Text, '"'),
        (T.Punctuation, ';'),
    ]


def test_php_filter_multiline_has_extra_space_before_equals():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1\nFROM t')])
    out = f.process(stmt)
    rendered = render(out)
    # has_nl adds an extra whitespace ' ' between varname and '='
    assert rendered[0] == (T.Name, '$sql')
    assert rendered[1] == (T.Whitespace, ' ')
    assert rendered[2] == (T.Whitespace, ' ')
    assert rendered[3] == (T.Operator, '=')


def test_php_filter_multiline_structure():
    f = OutputPHPFilter()
    stmt = make_stmt([
        sql.Token(T.Text, 'SELECT 1'),
        sql.Token(T.Whitespace, '\n'),
        sql.Token(T.Text, 'FROM t'),
    ])
    out = f.process(stmt)
    rendered = render(out)

    # Closing quote and add new line
    assert (T.Text, ' ";') in rendered
    # Continuation header: $sql .= "
    assert (T.Name, '$sql') in rendered
    assert (T.Operator, '.=') in rendered
    # Final close
    assert rendered[-1] == (T.Punctuation, ';')


def test_php_filter_multiline_indentation_preserved():
    f = OutputPHPFilter()
    stmt = make_stmt([
        sql.Token(T.Text, 'SELECT 1'),
        sql.Token(T.Whitespace, '\n    '),
        sql.Token(T.Text, 'FROM t'),
    ])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Whitespace, '    ') in rendered


def test_php_filter_escapes_double_quote():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'say "hi"')])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Text, 'say \\"hi\\"') in rendered


def test_php_filter_escapes_backslash():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, "a\\b")])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Text, "a\\\\b") in rendered


def test_php_filter_escapes_backslash_before_quote():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'a\\"b')])
    out = f.process(stmt)
    rendered = render(out)
    assert (T.Text, 'a\\\\\\"b') in rendered


def test_php_filter_second_call_newline_varname():
    f = OutputPHPFilter()
    stmt1 = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    f.process(stmt1)

    stmt2 = make_stmt([sql.Token(T.Text, 'SELECT 2')])
    out = f.process(stmt2)
    rendered = render(out)
    assert rendered[0] == (T.Whitespace, '\n')
    assert rendered[1] == (T.Name, '$sql2')


def test_php_filter_first_call_no_leading_newline():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    out = f.process(stmt)
    assert render(out)[0] == (T.Name, '$sql')


def test_php_filter_process_returns_same_stmt():
    f = OutputPHPFilter()
    stmt = make_stmt([sql.Token(T.Text, 'SELECT 1')])
    assert f.process(stmt) is stmt
