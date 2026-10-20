import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokens(expression):
    return list(Lexer().tokenize(expression))


def types(expression):
    return [t['type'] for t in tokens(expression)]


def values(expression):
    return [t['value'] for t in tokens(expression)]


# ---------- Empty / basic ----------

def test_empty_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(''))


def test_none_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(None))


def test_eof_token_included():
    result = tokens('foo')
    assert result[-1] == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}


def test_eof_only_for_single_char():
    result = tokens('a')
    assert result == [
        {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
        {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
    ]


# ---------- Simple tokens ----------

@pytest.mark.parametrize("char,expected_type", [
    ('.', 'dot'),
    ('*', 'star'),
    (']', 'rbracket'),
    (',', 'comma'),
    (':', 'colon'),
    ('@', 'current'),
    ('(', 'lparen'),
    (')', 'rparen'),
    ('{', 'lbrace'),
    ('}', 'rbrace'),
])
def test_simple_tokens(char, expected_type):
    result = tokens(char)
    assert result[0] == {
        'type': expected_type, 'value': char, 'start': 0, 'end': 1,
    }
    assert result[1]['type'] == 'eof'


def test_simple_token_positions():
    result = tokens('foo.bar')
    dot = result[1]
    assert dot == {'type': 'dot', 'value': '.', 'start': 3, 'end': 4}


# ---------- Identifiers ----------

def test_simple_identifier():
    assert tokens('foo')[0] == {
        'type': 'unquoted_identifier', 'value': 'foo', 'start': 0, 'end': 3,
    }


def test_identifier_with_underscore_and_digits():
    result = tokens('a_b1')
    assert result[0]['type'] == 'unquoted_identifier'
    assert result[0]['value'] == 'a_b1'


def test_identifier_starting_underscore():
    result = tokens('_foo')
    assert result[0]['type'] == 'unquoted_identifier'
    assert result[0]['value'] == '_foo'


def test_two_identifiers_whitespace_separated():
    result = tokens('foo bar')
    assert [t['value'] for t in result if t['type'] == 'unquoted_identifier'] == [
        'foo', 'bar',
    ]


# ---------- Whitespace ----------

@pytest.mark.parametrize("ws", [' ', '\t', '\n', '\r'])
def test_whitespace_skipped(ws):
    result = tokens('a' + ws + 'b')
    non_eof = [t for t in result if t['type'] != 'eof']
    assert len(non_eof) == 2


# ---------- Brackets ----------

def test_lbracket():
    assert tokens('[')[0] == {
        'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1,
    }


def test_flatten():
    assert tokens('[]')[0] == {
        'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2,
    }


def test_filter():
    assert tokens('[?')[0] == {
        'type': 'filter', 'value': '[?', 'start': 0, 'end': 2,
    }


def test_index_expression():
    result = tokens('[0]')
    assert result[0]['type'] == 'lbracket'
    assert result[1]['type'] == 'number'
    assert result[1]['value'] == 0
    assert result[2]['type'] == 'rbracket'


# ---------- Numbers ----------

def test_number():
    assert tokens('123')[0] == {
        'type': 'number', 'value': 123, 'start': 0, 'end': 3,
    }


def test_negative_number():
    assert tokens('-42')[0] == {
        'type': 'number', 'value': -42, 'start': 0, 'end': 3,
    }


def test_negative_number_with_trailing():
    result = tokens('-1]')
    assert result[0] == {
        'type': 'number', 'value': -1, 'start': 0, 'end': 2,
    }
    assert result[1]['type'] == 'rbracket'


def test_lone_dash_raises():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('-'))
    assert excinfo.value.lexer_position == 0
    assert excinfo.value.lexer_value == '-'
    assert "Unknown token '-'" in excinfo.value.message


def test_single_zero_number():
    assert tokens('0')[0]['value'] == 0


# ---------- Raw string literals ----------

def test_raw_string_literal():
    result = tokens("'foo'")
    assert result[0] == {
        'type': 'literal', 'value': 'foo', 'start': 0, 'end': 4,
    }


def test_raw_string_literal_with_escaped_quote():
    result = tokens(r"'a\'b'")
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == "a'b"


def test_raw_string_literal_unclosed():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize("'foo"))
    assert 'Unclosed' in excinfo.value.message


def test_raw_string_literal_empty():
    result = tokens("''")
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == ''


# ---------- Quoted identifiers ----------

def test_quoted_identifier():
    result = tokens('"foo"')
    assert result[0] == {
        'type': 'quoted_identifier', 'value': 'foo', 'start': 0, 'end': 4,
    }


def test_quoted_identifier_with_escape():
    result = tokens(r'"fo\"o"')
    assert result[0]['type'] == 'quoted_identifier'
    assert result[0]['value'] == 'fo"o'


def test_quoted_identifier_invalid_escape():
    with pytest.raises(LexerError):
        list(Lexer().tokenize('"foo\\q"'))


def test_quoted_identifier_unclosed():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('"foo'))
    assert 'Unclosed' in excinfo.value.message


# ---------- Literals ----------

def test_literal_number():
    result = tokens('`1`')
    assert result[0] == {
        'type': 'literal', 'value': 1, 'start': 0, 'end': 2,
    }


def test_literal_string_json():
    result = tokens('`"foo"`')
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == 'foo'


def test_literal_object():
    result = tokens('`{"a": 1}`')
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == {'a': 1}


def test_literal_json_array():
    result = tokens('`[1, 2]`')
    assert result[0]['value'] == [1, 2]


def test_literal_unclosed():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('`foo'))
    assert 'Unclosed' in excinfo.value.message


def test_literal_deprecated_string_syntax():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = tokens('`foo`')
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == 'foo'
    assert any(issubclass(w.category, PendingDeprecationWarning)
               for w in caught)


def test_literal_whitespace_parses_as_deprecated_string():
    # `foo bar baz` is not valid JSON but is accepted via the deprecated
    # string fallback (loads of a quoted string).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = tokens('`foo bar baz`')
    assert result[0]['type'] == 'literal'
    assert result[0]['value'] == 'foo bar baz'
    assert any(issubclass(w.category, PendingDeprecationWarning)
               for w in caught)


def test_literal_bad_token():
    # A trailing backslash before the closing delimiter triggers a JSON
    # decoding error in both the normal and deprecated fallback paths.
    with pytest.raises(LexerError):
        list(Lexer().tokenize(r'`"\\`'))


def test_literal_escaped_backtick():
    result = tokens(r'`"a\`b"`')
    assert result[0]['type'] == 'literal'


# ---------- Or / and / expref / comparisons ----------

def test_plain_pipe():
    assert tokens('|')[0] == {'type': 'pipe', 'value': '|', 'start': 0, 'end': 0}


def test_double_pipe_or():
    assert tokens('||')[0] == {'type': 'or', 'value': '||', 'start': 0, 'end': 1}


def test_plain_ampersand_expref():
    assert tokens('&')[0] == {
        'type': 'expref', 'value': '&', 'start': 0, 'end': 0,
    }


def test_double_ampersand_and():
    assert tokens('&&')[0] == {'type': 'and', 'value': '&&', 'start': 0, 'end': 1}


def test_lt():
    assert tokens('<')[0]['type'] == 'lt'


def test_lte():
    assert tokens('<=')[0]['type'] == 'lte'


def test_gt():
    assert tokens('>')[0]['type'] == 'gt'


def test_gte():
    assert tokens('>=')[0]['type'] == 'gte'


def test_not():
    assert tokens('!')[0]['type'] == 'not'


def test_ne():
    assert tokens('!=')[0]['type'] == 'ne'


def test_eq():
    result = tokens('==')[0]
    assert result['type'] == 'eq'
    assert result['value'] == '=='


def test_single_equals_raises():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('='))
    assert excinfo.value.lexer_value == '='
    assert "Unknown token '='" in excinfo.value.message


def test_single_equals_raises_not_at_eof():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('=a'))
    # position should have been rewound back to 0
    assert excinfo.value.lexer_position == 0


# ---------- Unknown token ----------

def test_unknown_token():
    with pytest.raises(LexerError) as excinfo:
        list(Lexer().tokenize('^'))
    assert excinfo.value.lexer_position == 0
    assert excinfo.value.lexer_value == '^'
    assert "Unknown token ^" in excinfo.value.message


# ---------- Combined expression ----------

def test_complex_expression():
    result = tokens('foo.bar[?a == `1`]')
    assert result[0]['type'] == 'unquoted_identifier'
    assert result[1]['type'] == 'dot'
    assert result[2]['type'] == 'unquoted_identifier'
    assert result[3]['type'] == 'filter'


def test_multiple_simple_tokens_positions():
    result = tokens('a.b.c')
    assert [t['type'] for t in result] == [
        'unquoted_identifier', 'dot',
        'unquoted_identifier', 'dot',
        'unquoted_identifier', 'eof',
    ]
    assert result[1]['start'] == 1
    assert result[3]['start'] == 3
