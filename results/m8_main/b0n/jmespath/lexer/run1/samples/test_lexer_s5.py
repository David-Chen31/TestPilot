import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


def token_signature(expression):
    return [(t['type'], t['value']) for t in tokenize(expression)]


# ---------------------------------------------------------------------------
# Empty expression
# ---------------------------------------------------------------------------
def test_empty_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(''))


def test_empty_expression_raises_on_empty_string():
    with pytest.raises(EmptyExpressionError):
        Lexer().tokenize('')


# ---------------------------------------------------------------------------
# EOF token
# ---------------------------------------------------------------------------
def test_eof_token_appended():
    tokens = tokenize('foo')
    assert tokens[-1] == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}


def test_eof_only_for_single_char():
    tokens = tokenize('a')
    assert tokens == [
        {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
        {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
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
    tokens = tokenize(char)
    assert tokens[0] == {'type': token_type, 'value': char,
                         'start': 0, 'end': 1}
    assert tokens[1]['type'] == 'eof'


def test_simple_tokens_positions():
    tokens = tokenize('a.b')
    assert tokens == [
        {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
        {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
        {'type': 'unquoted_identifier', 'value': 'b', 'start': 2, 'end': 3},
        {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
    ]


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
def test_unquoted_identifier():
    tokens = tokenize('foo_bar123')
    assert tokens[0] == {'type': 'unquoted_identifier', 'value': 'foo_bar123',
                         'start': 0, 'end': 10}


def test_unquoted_identifier_starting_underscore():
    tokens = tokenize('_foo')
    assert tokens[0]['type'] == 'unquoted_identifier'
    assert tokens[0]['value'] == '_foo'


def test_unquoted_identifier_with_digits():
    tokens = tokenize('abc123')
    assert tokens[0]['value'] == 'abc123'


def test_identifiers_separated_by_space():
    assert token_signature('foo bar') == [
        ('unquoted_identifier', 'foo'),
        ('unquoted_identifier', 'bar'),
        ('eof', ''),
    ]


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------
def test_whitespace_is_skipped():
    assert token_signature('  \t\n\r foo \t') == [
        ('unquoted_identifier', 'foo'),
        ('eof', ''),
    ]


def test_whitespace_positions_preserved():
    tokens = tokenize('  foo')
    assert tokens[0]['start'] == 2
    assert tokens[0]['end'] == 5


# ---------------------------------------------------------------------------
# Brackets: flatten, filter, lbracket
# ---------------------------------------------------------------------------
def test_lbracket():
    tokens = tokenize('[0]')
    assert tokens[0] == {'type': 'lbracket', 'value': '[',
                         'start': 0, 'end': 1}
    assert tokens[1] == {'type': 'number', 'value': 0,
                         'start': 1, 'end': 2}
    assert tokens[2] == {'type': 'rbracket', 'value': ']',
                         'start': 2, 'end': 3}


def test_flatten():
    tokens = tokenize('a[]')
    assert tokens[1] == {'type': 'flatten', 'value': '[]',
                         'start': 1, 'end': 3}


def test_filter():
    tokens = tokenize('[?foo]')
    assert tokens[0] == {'type': 'filter', 'value': '[?',
                         'start': 0, 'end': 2}


def test_flatten_at_eof():
    tokens = tokenize('[]')
    assert tokens[0]['type'] == 'flatten'
    assert tokens[-1]['type'] == 'eof'


# ---------------------------------------------------------------------------
# Raw string literals
# ---------------------------------------------------------------------------
def test_raw_string_literal():
    tokens = tokenize("'foo'")
    assert tokens[0] == {'type': 'literal', 'value': 'foo',
                         'start': 0, 'end': 5}


def test_raw_string_with_escaped_quote():
    tokens = tokenize("'foo\\'bar'")
    assert tokens[0]['value'] == "foo'bar"


def test_raw_string_empty():
    tokens = tokenize("''")
    assert tokens[0] == {'type': 'literal', 'value': '',
                         'start': 0, 'end': 2}


def test_raw_string_unclosed_raises():
    with pytest.raises(LexerError) as exc:
        tokenize("'foo")
    assert 'Unclosed' in str(exc.value)


# ---------------------------------------------------------------------------
# Or / and / expref / pipe
# ---------------------------------------------------------------------------
def test_or():
    tokens = tokenize('a || b')
    assert tokens[1] == {'type': 'or', 'value': '||',
                         'start': 2, 'end': 3}


def test_pipe():
    tokens = tokenize('a | b')
    assert tokens[1] == {'type': 'pipe', 'value': '|',
                         'start': 2, 'end': 2}


def test_and():
    tokens = tokenize('a && b')
    assert tokens[1] == {'type': 'and', 'value': '&&',
                         'start': 2, 'end': 3}


def test_expref():
    tokens = tokenize('&foo')
    assert tokens[0] == {'type': 'expref', 'value': '&',
                         'start': 0, 'end': 0}


# ---------------------------------------------------------------------------
# Literals (backtick)
# ---------------------------------------------------------------------------
def test_literal_json_number():
    tokens = tokenize('`123`')
    assert tokens[0] == {'type': 'literal', 'value': 123,
                         'start': 0, 'end': 5}


def test_literal_json_object():
    tokens = tokenize('`{"a": 1}`')
    assert tokens[0]['value'] == {'a': 1}


def test_literal_json_string():
    tokens = tokenize('`"foo"`')
    assert tokens[0]['value'] == 'foo'


def test_literal_json_null():
    tokens = tokenize('`null`')
    assert tokens[0]['value'] is None


def test_literal_json_true():
    tokens = tokenize('`true`')
    assert tokens[0]['value'] is True


def test_literal_with_escaped_backtick():
    tokens = tokenize('`"a\\`b"`')
    assert tokens[0]['value'] == 'a`b'


def test_literal_deprecated_string_syntax():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tokens = tokenize('`foo`')
    assert tokens[0]['value'] == 'foo'
    assert len(caught) == 1
    assert issubclass(caught[0].category, PendingDeprecationWarning)


def test_literal_unclosed_raises():
    with pytest.raises(LexerError):
        tokenize('`foo')


def test_literal_bad_token_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('`{invalid`')
    assert 'Bad token' in str(exc.value)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def test_number():
    tokens = tokenize('123')
    assert tokens[0] == {'type': 'number', 'value': 123,
                         'start': 0, 'end': 3}


def test_number_zero():
    tokens = tokenize('0')
    assert tokens[0] == {'type': 'number', 'value': 0,
                         'start': 0, 'end': 1}


def test_number_with_dot():
    tokens = tokenize('a.1')
    assert tokens[2] == {'type': 'number', 'value': 1,
                         'start': 2, 'end': 3}


def test_negative_number():
    tokens = tokenize('-123')
    assert tokens[0] == {'type': 'number', 'value': -123,
                         'start': 0, 'end': 4}


def test_negative_number_single_digit():
    tokens = tokenize('-5')
    assert tokens[0] == {'type': 'number', 'value': -5}


def test_lone_minus_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('-')
    assert "Unknown token '-'" in str(exc.value)


def test_minus_followed_by_non_digit_raises():
    with pytest.raises(LexerError):
        tokenize('-a')


# ---------------------------------------------------------------------------
# Quoted identifiers
# ---------------------------------------------------------------------------
def test_quoted_identifier():
    tokens = tokenize('"foo"')
    assert tokens[0] == {'type': 'quoted_identifier', 'value': 'foo',
                         'start': 0, 'end': 5}


def test_quoted_identifier_with_escape():
    tokens = tokenize('"foo\\nbar"')
    assert tokens[0]['value'] == 'foo\nbar'


def test_quoted_identifier_empty():
    tokens = tokenize('""')
    assert tokens[0] == {'type': 'quoted_identifier', 'value': '',
                         'start': 0, 'end': 2}


def test_quoted_identifier_unclosed_raises():
    with pytest.raises(LexerError):
        tokenize('"foo')
    with pytest.raises(LexerError) as exc:
        tokenize('"foo')
    assert 'Unclosed' in str(exc.value)


def test_quoted_identifier_invalid_json_raises():
    with pytest.raises(LexerError):
        tokenize('"foo\\x"')


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------
def test_lt():
    tokens = tokenize('a < b')
    assert tokens[1] == {'type': 'lt', 'value': '<',
                         'start': 2, 'end': 2}


def test_lte():
    tokens = tokenize('a <= b')
    assert tokens[1] == {'type': 'lte', 'value': '<=',
                         'start': 2, 'end': 3}


def test_gt():
    tokens = tokenize('a > b')
    assert tokens[1] == {'type': 'gt', 'value': '>',
                         'start': 2, 'end': 2}


def test_gte():
    tokens = tokenize('a >= b')
    assert tokens[1] == {'type': 'gte', 'value': '>=',
                         'start': 2, 'end': 3}


def test_ne():
    tokens = tokenize('a != b')
    assert tokens[1] == {'type': 'ne', 'value': '!=',
                         'start': 2, 'end': 3}


def test_not():
    tokens = tokenize('!a')
    assert tokens[0] == {'type': 'not', 'value': '!',
                         'start': 0, 'end': 0}


def test_eq():
    tokens = tokenize('a == b')
    assert tokens[1] == {'type': 'eq', 'value': '==',
                         'start': 2, 'end': 3}


def test_single_equals_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('a = b')
    assert "Unknown token '='" in str(exc.value)


def test_single_equals_at_eof_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('=')
    assert "Unknown token '='" in str(exc.value)


# ---------------------------------------------------------------------------
# Unknown tokens
# ---------------------------------------------------------------------------
def test_unknown_token_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('a $ b')
    assert "Unknown token $" in str(exc.value)


def test_unknown_token_position():
    with pytest.raises(LexerError) as exc:
        tokenize('a $')
    assert exc.value.lexer_position == 2
    assert exc.value.lexer_value == '$'


# ---------------------------------------------------------------------------
# Complex expressions
# ---------------------------------------------------------------------------
def test_complex_expression():
    expr = 'foo[?bar==`1`].baz'
    sig = token_signature(expr)
    assert sig == [
        ('unquoted_identifier', 'foo'),
        ('filter', '[?'),
        ('unquoted_identifier', 'bar'),
        ('eq', '=='),
        ('literal', 1),
        ('rbracket', ']'),
        ('dot', '.'),
        ('unquoted_identifier', 'baz'),
        ('eof', ''),
    ]


def test_leading_whitespace_only_raises_nothing_empty():
    # Whitespace is not empty, so it does not raise, only yields eof.
    tokens = tokenize('   ')
    assert tokens == [{'type': 'eof', 'value': '', 'start': 3, 'end': 3}]


def test_token_positions_in_complex_expression():
    tokens = tokenize('a.b')
    for token in tokens:
        assert token['start'] >= 0
        assert token['start'] <= token['end']


# ---------------------------------------------------------------------------
# Reusability of Lexer instance
# ---------------------------------------------------------------------------
def test_lexer_instance_reusable():
    lexer = Lexer()
    first = list(lexer.tokenize('foo'))
    second = list(lexer.tokenize('bar'))
    assert first[0]['value'] == 'foo'
    assert second[0]['value'] == 'bar'


def test_tokenize_returns_generator():
    result = Lexer().tokenize('foo')
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


# ---------------------------------------------------------------------------
# LexerError details
# ---------------------------------------------------------------------------
def test_lexer_error_str_contains_message():
    with pytest.raises(LexerError) as exc:
        tokenize('$')
    assert 'Bad jmespath expression' in str(exc.value)
    assert exc.value.lexer_position == 0
    assert exc.value.lexer_value == '$'
