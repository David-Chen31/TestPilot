import string
import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def to_tokens(expression):
    return list(Lexer().tokenize(expression))


# ---------------------------------------------------------------------------
# Initialization / empty expressions
# ---------------------------------------------------------------------------

def test_empty_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(''))


def test_none_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(None))


def test_class_level_token_sets():
    assert Lexer.START_IDENTIFIER == set(string.ascii_letters + '_')
    assert Lexer.VALID_IDENTIFIER == set(string.ascii_letters + string.digits + '_')
    assert Lexer.VALID_NUMBER == set(string.digits)
    assert Lexer.WHITESPACE == set(" \t\n\r")
    assert Lexer.SIMPLE_TOKENS['.'] == 'dot'
    assert Lexer.SIMPLE_TOKENS['*'] == 'star'
    assert Lexer.SIMPLE_TOKENS[']'] == 'rbracket'
    assert Lexer.SIMPLE_TOKENS[','] == 'comma'
    assert Lexer.SIMPLE_TOKENS[':'] == 'colon'
    assert Lexer.SIMPLE_TOKENS['@'] == 'current'
    assert Lexer.SIMPLE_TOKENS['('] == 'lparen'
    assert Lexer.SIMPLE_TOKENS[')'] == 'rparen'
    assert Lexer.SIMPLE_TOKENS['{'] == 'lbrace'
    assert Lexer.SIMPLE_TOKENS['}'] == 'rbrace'


# ---------------------------------------------------------------------------
# EOF token
# ---------------------------------------------------------------------------

def test_single_identifier_then_eof():
    tokens = to_tokens('foo')
    assert tokens == [
        {'type': 'unquoted_identifier', 'value': 'foo', 'start': 0, 'end': 3},
        {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
    ]


def test_eof_position_after_whitespace():
    tokens = to_tokens('   ')
    assert tokens == [
        {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
    ]


# ---------------------------------------------------------------------------
# Simple tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('char,token_type', [
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
def test_simple_tokens(char, token_type):
    tokens = to_tokens(char)
    assert tokens[0] == {
        'type': token_type, 'value': char, 'start': 0, 'end': 1,
    }
    assert tokens[-1]['type'] == 'eof'


def test_simple_token_positions():
    tokens = to_tokens('a.b')
    assert tokens == [
        {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
        {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
        {'type': 'unquoted_identifier', 'value': 'b', 'start': 2, 'end': 3},
        {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
    ]


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def test_identifier_with_underscore_and_digits():
    tokens = to_tokens('foo_bar123')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': 'foo_bar123',
        'start': 0, 'end': 10,
    }


def test_identifier_starting_with_underscore():
    tokens = to_tokens('_foo')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': '_foo', 'start': 0, 'end': 4,
    }


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

def test_whitespace_skipped():
    tokens = to_tokens(' a \t b \n ')
    types_values = [(t['type'], t['value']) for t in tokens]
    assert types_values == [
        ('unquoted_identifier', 'a'),
        ('unquoted_identifier', 'b'),
        ('eof', ''),
    ]


# ---------------------------------------------------------------------------
# Brackets: flatten, filter, lbracket
# ---------------------------------------------------------------------------

def test_flatten():
    tokens = to_tokens('[]')
    assert tokens[0] == {
        'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2,
    }


def test_filter():
    tokens = to_tokens('[?')
    assert tokens[0] == {
        'type': 'filter', 'value': '[?', 'start': 0, 'end': 2,
    }


def test_lbracket():
    tokens = to_tokens('[0]')
    assert tokens[0] == {
        'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1,
    }
    assert tokens[1] == {
        'type': 'number', 'value': 0, 'start': 1, 'end': 2,
    }
    assert tokens[2] == {
        'type': 'rbracket', 'value': ']', 'start': 2, 'end': 3,
    }


# ---------------------------------------------------------------------------
# Raw string literals
# ---------------------------------------------------------------------------

def test_raw_string_literal():
    tokens = to_tokens("'foo'")
    assert tokens[0] == {
        'type': 'literal', 'value': 'foo', 'start': 0, 'end': 5,
    }


def test_raw_string_literal_with_escaped_quote():
    tokens = to_tokens("'foo\\'bar'")
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == "foo'bar"


def test_raw_string_literal_empty():
    tokens = to_tokens("''")
    assert tokens[0] == {
        'type': 'literal', 'value': '', 'start': 0, 'end': 2,
    }


def test_unclosed_raw_string_literal_error():
    with pytest.raises(LexerError) as excinfo:
        to_tokens("'foo")
    assert "Unclosed ' delimiter" in excinfo.value.message


# ---------------------------------------------------------------------------
# Quoted identifiers
# ---------------------------------------------------------------------------

def test_quoted_identifier():
    tokens = to_tokens('"foo"')
    assert tokens[0] == {
        'type': 'quoted_identifier', 'value': 'foo', 'start': 0, 'end': 5,
    }


def test_quoted_identifier_with_escapes():
    tokens = to_tokens('"foo\\"bar"')
    assert tokens[0]['type'] == 'quoted_identifier'
    assert tokens[0]['value'] == 'foo"bar'


def test_unclosed_quoted_identifier():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('"foo')
    assert "Unclosed \" delimiter" in excinfo.value.message


def test_invalid_quoted_identifier_raises():
    with pytest.raises(LexerError):
        to_tokens('"\\x"')


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def test_number():
    tokens = to_tokens('123')
    assert tokens[0] == {
        'type': 'number', 'value': 123, 'start': 0, 'end': 3,
    }


def test_number_single_digit():
    tokens = to_tokens('7')
    assert tokens[0] == {
        'type': 'number', 'value': 7, 'start': 0, 'end': 1,
    }


def test_number_followed_by_token():
    tokens = to_tokens('12.')
    assert tokens[0] == {
        'type': 'number', 'value': 12, 'start': 0, 'end': 2,
    }
    assert tokens[1]['type'] == 'dot'


def test_negative_number():
    tokens = to_tokens('-42')
    assert tokens[0] == {
        'type': 'number', 'value': -42, 'start': 0, 'end': 3,
    }


def test_single_minus_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('-')
    assert excinfo.value.message == "Unknown token '-'"
    assert excinfo.value.lexer_position == 0
    assert excinfo.value.lexer_value == '-'


def test_minus_followed_by_identifier_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('-a')
    assert excinfo.value.message == "Unknown token '-'"


# ---------------------------------------------------------------------------
# Operators: or, pipe, and, expref, lte, lt, gte, gt, ne, not, eq
# ---------------------------------------------------------------------------

def test_or_operator():
    tokens = to_tokens('a || b')
    types = [t['type'] for t in tokens]
    assert 'or' in types
    or_token = [t for t in tokens if t['type'] == 'or'][0]
    assert or_token['value'] == '||'
    assert or_token['start'] == 2
    assert or_token['end'] == 3


def test_pipe_operator():
    tokens = to_tokens('a | b')
    types = [t['type'] for t in tokens]
    assert 'pipe' in types
    pipe_token = [t for t in tokens if t['type'] == 'pipe'][0]
    assert pipe_token['value'] == '|'
    assert pipe_token['start'] == 2
    assert pipe_token['end'] == 2


def test_and_operator():
    tokens = to_tokens('a && b')
    types = [t['type'] for t in tokens]
    assert 'and' in types
    and_token = [t for t in tokens if t['type'] == 'and'][0]
    assert and_token['value'] == '&&'


def test_expref_operator():
    tokens = to_tokens('&foo')
    assert tokens[0]['type'] == 'expref'
    assert tokens[0]['value'] == '&'


def test_lte_operator():
    tokens = to_tokens('a <= b')
    types = [t['type'] for t in tokens]
    assert 'lte' in types
    lte = [t for t in tokens if t['type'] == 'lte'][0]
    assert lte['value'] == '<='
    assert lte['start'] == 2
    assert lte['end'] == 3


def test_lt_operator():
    tokens = to_tokens('a < b')
    types = [t['type'] for t in tokens]
    assert 'lt' in types
    lt = [t for t in tokens if t['type'] == 'lt'][0]
    assert lt['value'] == '<'
    assert lt['start'] == 2
    assert lt['end'] == 2


def test_gte_operator():
    tokens = to_tokens('a >= b')
    types = [t['type'] for t in tokens]
    assert 'gte' in types
    gte = [t for t in tokens if t['type'] == 'gte'][0]
    assert gte['value'] == '>='


def test_gt_operator():
    tokens = to_tokens('a > b')
    types = [t['type'] for t in tokens]
    assert 'gt' in types
    gt = [t for t in tokens if t['type'] == 'gt'][0]
    assert gt['value'] == '>'


def test_ne_operator():
    tokens = to_tokens('a != b')
    types = [t['type'] for t in tokens]
    assert 'ne' in types
    ne = [t for t in tokens if t['type'] == 'ne'][0]
    assert ne['value'] == '!='


def test_not_operator():
    tokens = to_tokens('!a')
    assert tokens[0]['type'] == 'not'
    assert tokens[0]['value'] == '!'


def test_eq_operator():
    tokens = to_tokens('a == b')
    types = [t['type'] for t in tokens]
    assert 'eq' in types
    eq = [t for t in tokens if t['type'] == 'eq'][0]
    assert eq['value'] == '=='
    assert eq['start'] == 2
    assert eq['end'] == 3


def test_single_equals_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('a = b')
    assert excinfo.value.message == "Unknown token '='"
    assert excinfo.value.lexer_value == '='


def test_single_equals_at_end_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('=')
    assert excinfo.value.message == "Unknown token '='"
    # At EOF, position is not rewound.
    assert excinfo.value.lexer_position == 0


# ---------------------------------------------------------------------------
# Literals (backtick)
# ---------------------------------------------------------------------------

def test_literal_json_number():
    tokens = to_tokens('`123`')
    assert tokens[0] == {
        'type': 'literal', 'value': 123, 'start': 0, 'end': 5,
    }


def test_literal_json_string():
    tokens = to_tokens('`"foo"`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == 'foo'


def test_literal_json_object():
    tokens = to_tokens('`{"a": 1}`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == {'a': 1}


def test_literal_escaped_backtick():
    tokens = to_tokens('`"a\\`b"`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == 'a`b'


def test_literal_deprecated_string_syntax_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tokens = to_tokens('`foo`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == 'foo'
    assert any(issubclass(w.category, PendingDeprecationWarning) for w in caught)


def test_bad_literal_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('`{invalid`')
    assert excinfo.value.message.startswith("Bad token")
    assert excinfo.value.lexer_position == 0


def test_unclosed_literal_raises():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('`123')
    assert "Unclosed ` delimiter" in excinfo.value.message


# ---------------------------------------------------------------------------
# Unknown tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('expr', ['$', '#', ';', '%', '^'])
def test_unknown_token(expr):
    with pytest.raises(LexerError) as excinfo:
        to_tokens(expr)
    assert excinfo.value.lexer_value == expr
    assert excinfo.value.lexer_position == 0
    assert excinfo.value.message == "Unknown token %s" % expr


def test_unknown_token_position():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('foo$')
    assert excinfo.value.lexer_value == '$'
    assert excinfo.value.lexer_position == 3


# ---------------------------------------------------------------------------
# LexerError attributes
# ---------------------------------------------------------------------------

def test_lexer_error_attributes():
    with pytest.raises(LexerError) as excinfo:
        to_tokens('$')
    err = excinfo.value
    assert err.lexer_position == 0
    assert err.lexer_value == '$'
    assert err.message == "Unknown token $"
    # expression is not set by the lexer itself, defaults to None.
    assert err.expression is None


# ---------------------------------------------------------------------------
# Combined / integration-style token stream
# ---------------------------------------------------------------------------

def test_combined_expression():
    tokens = to_tokens('foo.bar[0]')
    assert [(t['type'], t['value']) for t in tokens] == [
        ('unquoted_identifier', 'foo'),
        ('dot', '.'),
        ('unquoted_identifier', 'bar'),
        ('lbracket', '['),
        ('number', 0),
        ('rbracket', ']'),
        ('eof', ''),
    ]


def test_tokenize_generator_reusable_across_calls():
    lexer = Lexer()
    first = list(lexer.tokenize('a'))
    second = list(lexer.tokenize('b'))
    assert first[0]['value'] == 'a'
    assert second[0]['value'] == 'b'
