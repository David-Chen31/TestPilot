import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


# ---------------------------------------------------------------------------
# Initialization / empty expression
# ---------------------------------------------------------------------------

def test_empty_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(''))


def test_none_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(None))


# ---------------------------------------------------------------------------
# EOF token
# ---------------------------------------------------------------------------

def test_eof_token_appended():
    tokens = tokenize('foo')
    assert tokens[-1] == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}


def test_eof_after_single_char():
    tokens = tokenize('@')
    assert tokens == [
        {'type': 'current', 'value': '@', 'start': 0, 'end': 1},
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
    assert tokens[0] == {'type': token_type, 'value': char, 'start': 0, 'end': 1}
    assert tokens[1] == {'type': 'eof', 'value': '', 'start': 1, 'end': 1}


def test_multiple_simple_tokens():
    tokens = tokenize('.@')
    assert tokens == [
        {'type': 'dot', 'value': '.', 'start': 0, 'end': 1},
        {'type': 'current', 'value': '@', 'start': 1, 'end': 2},
        {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
    ]


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

def test_unquoted_identifier():
    tokens = tokenize('foo')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': 'foo', 'start': 0, 'end': 3}


def test_unquoted_identifier_with_underscore_and_digits():
    tokens = tokenize('_foo123')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': '_foo123',
        'start': 0, 'end': 7}


def test_unquoted_identifier_followed_by_other_token():
    tokens = tokenize('foo.bar')
    assert [t['type'] for t in tokens] == [
        'unquoted_identifier', 'dot', 'unquoted_identifier', 'eof']
    assert tokens[0]['value'] == 'foo'
    assert tokens[1]['value'] == '.'
    assert tokens[2]['value'] == 'bar'


def test_identifier_at_end_stops_correctly():
    tokens = tokenize('foo@')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': 'foo', 'start': 0, 'end': 3}
    assert tokens[1] == {
        'type': 'current', 'value': '@', 'start': 3, 'end': 4}


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ws', [' ', '\t', '\n', '\r'])
def test_whitespace_skipped(ws):
    tokens = tokenize('a' + ws + 'b')
    assert [t['type'] for t in tokens] == [
        'unquoted_identifier', 'unquoted_identifier', 'eof']
    assert tokens[0]['value'] == 'a'
    assert tokens[1]['value'] == 'b'


def test_leading_whitespace_positions():
    tokens = tokenize('  foo')
    assert tokens[0] == {
        'type': 'unquoted_identifier', 'value': 'foo', 'start': 2, 'end': 5}


# ---------------------------------------------------------------------------
# Brackets: lbracket, flatten, filter
# ---------------------------------------------------------------------------

def test_lbracket():
    tokens = tokenize('[')
    assert tokens[0] == {'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1}


def test_lbracket_with_content():
    tokens = tokenize('[0]')
    assert tokens[0] == {'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1}
    assert tokens[1] == {'type': 'number', 'value': 0, 'start': 1, 'end': 2}
    assert tokens[2] == {'type': 'rbracket', 'value': ']', 'start': 2, 'end': 3}


def test_flatten():
    tokens = tokenize('[]')
    assert tokens[0] == {'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2}
    assert tokens[1] == {'type': 'eof', 'value': '', 'start': 2, 'end': 2}


def test_filter():
    tokens = tokenize('[?')
    assert tokens[0] == {'type': 'filter', 'value': '[?', 'start': 0, 'end': 2}


def test_flatten_then_identifier():
    tokens = tokenize('[]foo')
    assert tokens[0] == {'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2}
    assert tokens[1] == {
        'type': 'unquoted_identifier', 'value': 'foo', 'start': 2, 'end': 5}


# ---------------------------------------------------------------------------
# Raw string literal
# ---------------------------------------------------------------------------

def test_raw_string_literal():
    tokens = tokenize("'foo'")
    assert tokens[0] == {'type': 'literal', 'value': 'foo', 'start': 0, 'end': 5}


def test_raw_string_literal_with_escaped_quote():
    tokens = tokenize(r"'foo\'bar'")
    assert tokens[0] == {
        'type': 'literal', 'value': "foo'bar", 'start': 0, 'end': 10}


def test_raw_string_literal_empty():
    tokens = tokenize("''")
    assert tokens[0] == {'type': 'literal', 'value': '', 'start': 0, 'end': 2}


def test_raw_string_literal_unclosed():
    with pytest.raises(LexerError) as exc:
        tokenize("'foo")
    assert exc.value.lexer_position == 0
    assert "Unclosed ' delimiter" in exc.value.message


# ---------------------------------------------------------------------------
# Pipe / Or, Expref / And
# ---------------------------------------------------------------------------

def test_pipe():
    tokens = tokenize('|')
    assert tokens[0] == {'type': 'pipe', 'value': '|', 'start': 0, 'end': 0}


def test_or():
    tokens = tokenize('||')
    assert tokens[0] == {'type': 'or', 'value': '||', 'start': 0, 'end': 1}
    assert tokens[1] == {'type': 'eof', 'value': '', 'start': 2, 'end': 2}


def test_expref():
    tokens = tokenize('&')
    assert tokens[0] == {'type': 'expref', 'value': '&', 'start': 0, 'end': 0}


def test_and():
    tokens = tokenize('&&')
    assert tokens[0] == {'type': 'and', 'value': '&&', 'start': 0, 'end': 1}


# ---------------------------------------------------------------------------
# Literals (backtick)
# ---------------------------------------------------------------------------

def test_json_literal_number():
    tokens = tokenize('`1`')
    assert tokens[0] == {'type': 'literal', 'value': 1, 'start': 0, 'end': 3}


def test_json_literal_string():
    tokens = tokenize('`"foo"`')
    assert tokens[0] == {'type': 'literal', 'value': 'foo', 'start': 0, 'end': 7}


def test_json_literal_object():
    tokens = tokenize('`{"a": 1}`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == {'a': 1}


def test_json_literal_escaped_backtick():
    tokens = tokenize(r'`"foo\`bar"`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == 'foo`bar'


def test_literal_deprecated_string_syntax_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tokens = tokenize('`foo`')
    assert tokens[0]['type'] == 'literal'
    assert tokens[0]['value'] == 'foo'
    assert any(issubclass(w.category, PendingDeprecationWarning) for w in caught)


def test_literal_unclosed():
    with pytest.raises(LexerError) as exc:
        tokenize('`1')
    assert exc.value.lexer_position == 0
    assert "Unclosed ` delimiter" in exc.value.message


def test_literal_bad_token():
    with pytest.raises(LexerError) as exc:
        tokenize('`{invalid`')
    assert 'Bad token' in exc.value.message


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def test_number():
    tokens = tokenize('123')
    assert tokens[0] == {'type': 'number', 'value': 123, 'start': 0, 'end': 3}


def test_number_with_leading_zero():
    tokens = tokenize('007')
    assert tokens[0] == {'type': 'number', 'value': 7, 'start': 0, 'end': 3}


def test_negative_number():
    tokens = tokenize('-42')
    assert tokens[0] == {'type': 'number', 'value': -42, 'start': 0, 'end': 3}


def test_lone_minus_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('-')
    assert exc.value.lexer_position == 0
    assert exc.value.lexer_value == '-'
    assert exc.value.message == "Unknown token '-'"


def test_minus_then_non_digit_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('-a')
    assert exc.value.lexer_position == 0
    assert exc.value.lexer_value == '-'
    assert "Unknown token" in exc.value.message


# ---------------------------------------------------------------------------
# Quoted identifiers
# ---------------------------------------------------------------------------

def test_quoted_identifier():
    tokens = tokenize('"foo"')
    assert tokens[0] == {
        'type': 'quoted_identifier', 'value': 'foo', 'start': 0, 'end': 5}


def test_quoted_identifier_with_escape():
    tokens = tokenize(r'"fo\"o"')
    assert tokens[0]['type'] == 'quoted_identifier'
    assert tokens[0]['value'] == 'fo"o'


def test_quoted_identifier_empty():
    tokens = tokenize('""')
    assert tokens[0] == {
        'type': 'quoted_identifier', 'value': '', 'start': 0, 'end': 2}


def test_quoted_identifier_unclosed():
    with pytest.raises(LexerError) as exc:
        tokenize('"foo')
    assert exc.value.lexer_position == 0
    assert "Unclosed \" delimiter" in exc.value.message


def test_quoted_identifier_invalid_json_escape():
    with pytest.raises(LexerError) as exc:
        tokenize(r'"\x"')
    assert exc.value.lexer_position == 0
    assert exc.value.lexer_value == '"\\x"'


# ---------------------------------------------------------------------------
# Comparison operators: < > ! =
# ---------------------------------------------------------------------------

def test_lt():
    tokens = tokenize('<')
    assert tokens[0] == {'type': 'lt', 'value': '<', 'start': 0, 'end': 0}


def test_lte():
    tokens = tokenize('<=')
    assert tokens[0] == {'type': 'lte', 'value': '<=', 'start': 0, 'end': 1}


def test_gt():
    tokens = tokenize('>')
    assert tokens[0] == {'type': 'gt', 'value': '>', 'start': 0, 'end': 0}


def test_gte():
    tokens = tokenize('>=')
    assert tokens[0] == {'type': 'gte', 'value': '>=', 'start': 0, 'end': 1}


def test_not():
    tokens = tokenize('!')
    assert tokens[0] == {'type': 'not', 'value': '!', 'start': 0, 'end': 0}


def test_ne():
    tokens = tokenize('!=')
    assert tokens[0] == {'type': 'ne', 'value': '!=', 'start': 0, 'end': 1}


def test_eq():
    tokens = tokenize('==')
    assert tokens[0] == {'type': 'eq', 'value': '==', 'start': 0, 'end': 1}
    assert tokens[1] == {'type': 'eof', 'value': '', 'start': 2, 'end': 2}


def test_single_equals_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('=')
    assert exc.value.lexer_value == '='
    assert exc.value.message == "Unknown token '='"
    # At EOF position stays at 0
    assert exc.value.lexer_position == 0


def test_single_equals_with_following_char_position():
    with pytest.raises(LexerError) as exc:
        tokenize('=a')
    assert exc.value.lexer_value == '='
    assert exc.value.message == "Unknown token '='"
    # Position is rewound by one
    assert exc.value.lexer_position == 0


# ---------------------------------------------------------------------------
# Unknown tokens
# ---------------------------------------------------------------------------

def test_unknown_token_raises():
    with pytest.raises(LexerError) as exc:
        tokenize('#')
    assert exc.value.lexer_position == 0
    assert exc.value.lexer_value == '#'
    assert exc.value.message == "Unknown token #"


# ---------------------------------------------------------------------------
# Integration-style token sequences
# ---------------------------------------------------------------------------

def test_compound_expression():
    tokens = tokenize('foo.bar[0]')
    assert [t['type'] for t in tokens] == [
        'unquoted_identifier', 'dot', 'unquoted_identifier',
        'lbracket', 'number', 'rbracket', 'eof']
    assert tokens[0]['value'] == 'foo'
    assert tokens[3]['value'] == '['
    assert tokens[4]['value'] == 0


def test_filter_expression():
    tokens = tokenize('foo[?bar == `1`]')
    assert [t['type'] for t in tokens] == [
        'unquoted_identifier', 'filter', 'unquoted_identifier',
        'eq', 'literal', 'rbracket', 'eof']


def test_token_positions_in_compound():
    tokens = tokenize('a.b')
    assert tokens[0]['start'] == 0 and tokens[0]['end'] == 1
    assert tokens[1]['start'] == 1 and tokens[1]['end'] == 2
    assert tokens[2]['start'] == 2 and tokens[2]['end'] == 3


def test_reusable_lexer_instance():
    lexer = Lexer()
    first = list(lexer.tokenize('foo'))
    second = list(lexer.tokenize('bar'))
    assert first[0]['value'] == 'foo'
    assert second[0]['value'] == 'bar'
