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

def make_stmt(*token_tuples):
    """Build a TokenList statement from (tokentype, value) tuples."""
    tokens = [sql.Token(tt, value) for tt, value in token_tuples]
    return sql.TokenList(tokens)


def literal(tok):
    """Return (value) for a token converted back to text."""
    return tok.value


# ---------------------------------------------------------------------------
# OutputFilter base class
# ---------------------------------------------------------------------------

def test_output_filter_init_defaults():
    f = OutputFilter()
    assert f.varname == 'sql'
    assert f.count == 0
    assert f.varname_prefix == ''


def test_output_filter_init_custom_varname():
    f = OutputFilter(varname='q')
    assert f.varname == 'q'
    assert f.count == 0


def test_output_filter_process_raises_not_implemented():
    f = OutputFilter()
    stmt = make_stmt((T.Keyword, 'SELECT'))
    with pytest.raises(NotImplementedError):
        f.process(stmt)
    # count was incremented before raising
    assert f.count == 1


def test_output_filter_count_increments_on_each_process():
    f = OutputFilter()
    stmt = make_stmt((T.Keyword, 'SELECT'))
    with pytest.raises(NotImplementedError):
        f.process(stmt)
    with pytest.raises(NotImplementedError):
        f.process(make_stmt((T.Keyword, 'X')))
    assert f.count == 2


def test_output_filter_subclass_uses_process_and_count():
    """process() picks varname/suffixed varname based on count."""

    class RecordingFilter(OutputFilter):
        def _process(self, stream, varname, has_nl):
            self.seen = (list(stream), varname, has_nl)
            return list(stream)

    f = RecordingFilter(varname='v')
    stmt = make_stmt((T.Keyword, 'SELECT'), (T.Whitespace, ' '), (T.Name, 'a'))
    result = f.process(stmt)
    assert result is stmt
    assert f.count == 1
    assert f.seen[1] == 'v'
    assert f.seen[2] is False

    f.process(make_stmt((T.Keyword, 'SELECT')))
    assert f.seen[1] == 'v2'


def test_output_filter_has_nl_multiline():
    class RecordingFilter(OutputFilter):
        def _process(self, stream, varname, has_nl):
            self.has_nl = has_nl
            return list(stream)

    f = RecordingFilter()
    nl = sql.Token(T.Whitespace, '\n')
    stmt = make_stmt((T.Keyword, 'SELECT'), (T.Whitespace, ' '), (T.Name, 'a'))
    # Inject newline via token value
    stmt2 = sql.TokenList([sql.Token(T.Keyword, 'SELECT'),
                           sql.Token(T.Whitespace, '\n'),
                           sql.Token(T.Name, 'a')])
    f.process(stmt2)
    assert f.has_nl is True

    f.process(stmt)
    assert f.has_nl is False


def test_output_filter_attributes_on_base_not_implemented():
    """varname_prefix default used by subclasses."""
    assert OutputFilter.varname_prefix == ''
    assert OutputPHPFilter.varname_prefix == '$'


# ---------------------------------------------------------------------------
# OutputPythonFilter
# ---------------------------------------------------------------------------

def _text_of(tokens):
    return ''.join(t.value if hasattr(t, 'value') else str(t) for t in tokens)


def test_python_filter_simple_single_line():
    f = OutputPythonFilter()
    stmt = make_stmt((T.Keyword, 'SELECT'),
                     (T.Whitespace, ' '),
                     (T.Name, 'a'))
    stmt.tokens = list(f._process(stmt.tokens, 'sql', False))
    text = _text_of(stmt.tokens)
    assert text == "sql = 'SELECT a'"

    kinds = [(t.ttype, t.value) for t in stmt.tokens]
    assert kinds[0] == (T.Name, 'sql')
    assert kinds[1] == (T.Whitespace, ' ')
    assert kinds[2] == (T.Operator, '=')
    assert kinds[3] == (T.Whitespace, ' ')
    assert kinds[4] == (T.Text, "'")
    assert kinds[-1] == (T.Text, "'")


def test_python_filter_via_process_sets_count_and_returns_stmt():
    f = OutputPythonFilter()
    stmt = make_stmt((T.Keyword, 'SELECT'),
                     (T.Whitespace, ' '),
                     (T.Name, 'a'))
    result = f.process(stmt)
    assert result is stmt
    assert f.count == 1
    assert _text_of(stmt.tokens) == "sql = 'SELECT a'"


def test_python_filter_multiline_wraps_in_parens():
    f = OutputPythonFilter()
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'),
                          sql.Token(T.Whitespace, '\n'),
                          sql.Token(T.Name, 'a')])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    assert text.startswith("sql = (")
    assert text.endswith(")")
    # closing quote and paren
    assert (T.Operator, '(') in [(t.ttype, t.value) for t in stmt.tokens]
    assert (T.Operator, ')') in [(t.ttype, t.value) for t in stmt.tokens]


def test_python_filter_multiline_newline_handling():
    f = OutputPythonFilter()
    nl_token = sql.Token(T.Whitespace, '\n   ')
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'),
                          nl_token,
                          sql.Token(T.Name, 'a')])
    f.process(stmt)
    # The newline token should have been replaced; quote header repeated
    text = _text_of(stmt.tokens)
    # Second line opens with quote on next line, indented by len(varname)+4
    assert " ' \n" in text
    assert "'" in text
    # original after-lb indentation was '   ' and must be present
    assert '   a' in text or '   ' in text


def test_python_filter_escaping_backslash_and_quote():
    f = OutputPythonFilter()
    stmt = make_stmt((T.String.Single, "'it's'"),
                     (T.Whitespace, ' '),
                     (T.Keyword, 'X'))
    # Note: the raw token value is what gets escaped
    stmt = sql.TokenList([sql.Token(T.Name, "a\\b"),
                          sql.Token(T.Whitespace, ' '),
                          sql.Token(T.Name, "it's")])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    assert 'a\\\\b' in text
    assert "it\\'s" in text


def test_python_filter_escaping_backslash_before_quote():
    f = OutputPythonFilter()
    stmt = sql.TokenList([sql.Token(T.Name, "\\'")])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    # backslash doubled, quote escaped
    assert '\\\\\\\'' in text


def test_python_filter_second_call_prepends_newline():
    f = OutputPythonFilter()
    s1 = make_stmt((T.Keyword, 'A'))
    s2 = make_stmt((T.Keyword, 'B'))
    f.process(s1)
    f.process(s2)
    assert f.count == 2
    assert s2.tokens[0].value == '\n'
    assert s2.tokens[1].value == 'sql2'
    assert _text_of(s2.tokens).startswith("sql2")


def test_python_filter_custom_varname():
    f = OutputPythonFilter(varname='query')
    stmt = make_stmt((T.Keyword, 'SELECT'))
    f.process(stmt)
    assert stmt.tokens[0].value == 'query'


def test_python_filter_multiline_quote_header_indentation():
    f = OutputPythonFilter(varname='ab')
    nl = sql.Token(T.Whitespace, '\n')
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'), nl,
                          sql.Token(T.Name, 'a')])
    f.process(stmt)
    values = [t.value for t in stmt.tokens]
    # for varname 'ab', quote header after newline indented by len+4 = 6
    assert '      ' in values


# ---------------------------------------------------------------------------
# OutputPHPFilter
# ---------------------------------------------------------------------------

def test_php_filter_simple_single_line():
    f = OutputPHPFilter()
    stmt = make_stmt((T.Keyword, 'SELECT'),
                     (T.Whitespace, ' '),
                     (T.Name, 'a'))
    result = f.process(stmt)
    assert result is stmt
    assert f.count == 1
    text = _text_of(stmt.tokens)
    assert text == '$sql = "SELECT a";'
    kinds = [(t.ttype, t.value) for t in stmt.tokens]
    assert kinds[0] == (T.Name, '$sql')
    assert kinds[-2] == (T.Text, '"')
    assert kinds[-1] == (T.Punctuation, ';')


def test_php_filter_spacing_with_multiline():
    f = OutputPHPFilter()
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'),
                          sql.Token(T.Whitespace, '\n'),
                          sql.Token(T.Name, 'a')])
    f.process(stmt)
    kinds = [(t.ttype, t.value) for t in stmt.tokens]
    # has_nl -> extra space after name before '='
    idx_name = kinds.index((T.Name, '$sql'))
    assert kinds[idx_name + 1] == (T.Whitespace, ' ')
    assert kinds[idx_name + 2] == (T.Whitespace, ' ')
    assert kinds[idx_name + 3] == (T.Operator, '=')


def test_php_filter_multiline_concat_lines():
    f = OutputPHPFilter()
    nl = sql.Token(T.Whitespace, '\n  ')
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'), nl,
                          sql.Token(T.Name, 'a')])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    assert ' ";' in text
    assert '$sql .= "' in text
    # indentation from after_lb '  ' must appear
    assert '  a' in text or '\n  ' in text


def test_php_filter_escaping():
    f = OutputPHPFilter()
    stmt = sql.TokenList([sql.Token(T.Name, "a\\b"),
                          sql.Token(T.Name, ' say "hi"')])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    assert 'a\\\\b' in text
    assert 'say \\"hi\\"' in text


def test_php_filter_second_call_prepends_newline_and_uses_suffix():
    f = OutputPHPFilter()
    s1 = make_stmt((T.Keyword, 'A'))
    s2 = make_stmt((T.Keyword, 'B'))
    f.process(s1)
    f.process(s2)
    assert f.count == 2
    assert s2.tokens[0].value == '\n'
    assert s2.tokens[1].value == '$sql2'


def test_php_filter_custom_varname():
    f = OutputPHPFilter(varname='q')
    stmt = make_stmt((T.Keyword, 'SELECT'))
    f.process(stmt)
    assert stmt.tokens[0].value == '$q'
    text = _text_of(stmt.tokens)
    assert text.startswith('$q = "')


def test_php_filter_quote_header_indentation_and_concat():
    f = OutputPHPFilter(varname='ab')
    nl = sql.Token(T.Whitespace, '\n')
    stmt = sql.TokenList([sql.Token(T.Keyword, 'SELECT'), nl,
                          sql.Token(T.Name, 'x')])
    f.process(stmt)
    kinds = [(t.ttype, t.value) for t in stmt.tokens]
    # after the second quote open we should find the varname token again
    name_positions = [i for i, (tt, v) in enumerate(kinds) if tt == T.Name]
    assert len(name_positions) >= 2
    assert kinds[name_positions[1]] == (T.Name, '$ab')


# ---------------------------------------------------------------------------
# xss / injection style edge cases on both filters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flter_cls', [OutputPythonFilter, OutputPHPFilter])
def test_quote_injection_does_not_break_out(flter_cls):
    f = flter_cls()
    stmt = sql.TokenList([sql.Token(T.Name, "'; --")])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    # The inner quote must be escaped
    if flter_cls is OutputPythonFilter:
        assert "\\'" in text
    else:
        # double quote escaping; a single quote is harmless in PHP filter
        assert '"' in text


@pytest.mark.parametrize('flter_cls', [OutputPythonFilter, OutputPHPFilter])
def test_backslash_injection(flter_cls):
    f = flter_cls()
    stmt = sql.TokenList([sql.Token(T.Name, "x\\\\")])
    f.process(stmt)
    text = _text_of(stmt.tokens)
    # Every backslash in the source value should be doubled
    assert '\\\\\\\\' in text


@pytest.mark.parametrize('flter_cls', [OutputPythonFilter, OutputPHPFilter])
def test_empty_stream(flter_cls):
    f = flter_cls()
    stmt = sql.TokenList([])
    result = f.process(stmt)
    assert result is stmt
    text = _text_of(stmt.tokens)
    # Still emits the assignment scaffolding plus empty literal
    assert '=' in text


# ---------------------------------------------------------------------------
# Behaviour around whitespace tokens
# ---------------------------------------------------------------------------

def test_python_filter_whitespace_no_newline_is_escaped():
    f = OutputPythonFilter()
    stmt = sql.TokenList([sql.Token(T.Whitespace, '  ')])
    f.process(stmt)
    # The whitespace token is not a newline token, so it goes through
    # the escaping branch and is emitted as T.Text.
    text_tokens = [t for t in stmt.tokens if t.ttype is T.Text]
    assert any(t.value == '  ' for t in text_tokens)


def test_php_filter_whitespace_no_newline_is_escaped():
    f = OutputPHPFilter()
    stmt = sql.TokenList([sql.Token(T.Whitespace, '  ')])
    f.process(stmt)
    text_tokens = [t for t in stmt.tokens if t.ttype is T.Text]
    assert any(t.value == '  ' for t in text_tokens)


def test_python_filter_token_value_mutation_escapes_original():
    """_process mutates token.value in place for escaped tokens."""
    f = OutputPythonFilter()
    tok = sql.Token(T.Name, "a'b")
    stmt = sql.TokenList([tok])
    f.process(stmt)
    assert tok.value == "a\\'b"


def test_php_filter_token_value_mutation_escapes_original():
    f = OutputPHPFilter()
    tok = sql.Token(T.Name, 'a"b')
    stmt = sql.TokenList([tok])
    f.process(stmt)
    assert tok.value == 'a\\"b'
