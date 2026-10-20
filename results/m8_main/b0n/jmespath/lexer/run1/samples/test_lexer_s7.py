import string
import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokens(expr):
    return list(Lexer().tokenize(expr))


def token_types(expr):
    return [t['type'] for t in tokens(expr)]


# ---------------------------------------------------------------------------
# Basics / initialization
# ---------------------------------------------------------------------------

def test_empty_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(""))


def test_none_expression_raises():
    with pytest.raises(EmptyExpressionError):
        list(Lexer().tokenize(None))


def test_eof_only():
    toks = tokens(" ")
    assert toks == [{'type': 'eof', 'value': '', 'start': 1, 'end': 1}]


def test_eof_positions_match_length():
    expr = "abc"
    toks = tokens(expr)
    assert toks[-1] == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}


def test_initialize_state():
    lexer = Lexer()
    list(lexer.tokenize("abc"))
    assert lexer._position == 2
    assert lexer._expression == "abc"
    assert lexer._chars == ['a', 'b', 'c']
    assert lexer._current == 'c'
    assert lexer._length == 3


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------

def test_class_constants():
    assert Lexer.START_IDENTIFIER == set(string.ascii_letters + '_')
    assert Lexer.VALID_IDENTIFIER == set(
        string.ascii_letters + string.digits + '_')
    assert Lexer.VALID_NUMBER == set(string.digits)
    assert Lexer.WHITESPACE == set(" \t\n\r")


def test_simple_tokens_mapping():
    assert Lexer.SIMPLE_TOKENS == {
        '.': 'dot', '*': 'star', ']': 'rbracket', ',': 'comma',
        ':': 'colon', '@': 'current', '(': 'lparen', ')': 'rparen',
        '{': 'lbrace', '}': 'rbrace',
    }


# ---------------------------------------------------------------------------
# Simple tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("char,kind", [
    ('.', 'dot'), ('*', 'star'), (']', 'rbracket'), (',', 'comma'),
    (':', 'colon'), ('@', 'current'), ('(', 'lparen'), (')', 'rparen'),
    ('{', 'lbrace'), ('}', 'rbrace'),
])
def test_simple_tokens(char, kind):
    toks = tokens(char)
    assert toks[0] == {'type': kind, 'value': char, 'start': 0, 'end': 1}
    assert toks[1]['type'] == 'eof'


def test_simple_token_positions():
    toks = tokens("a.b")
    assert toks == [
        {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
        {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
        {'type': 'unquoted_identifier', 'value': 'b', 'start': 2, 'end': 3},
        {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
    ]


# ---------------------------------------------------------------------------
# Unquoted identifiers
# ---------------------------------------------------------------------------

def test_unquoted_identifier_simple():
    toks = tokens("foo")
    assert toks[0] == {'type': 'unquoted_identifier', 'value': 'foo',
                       'start': 0, 'end': 3}


def test_unquoted_identifier_underscore_start():
    toks = tokens("_foo_123")
    assert toks[0] == {'type': 'unquoted_identifier', 'value': '_foo_123',
                       'start': 0, 'end': 8}


def test_unquoted_identifier_underscore_only():
    toks = tokens("_")
    assert toks[0]['type'] == 'unquoted_identifier'
    assert toks[0]['value'] == '_'


def test_unquoted_identifier_with_digits():
    toks = tokens("abc123")
    assert toks[0]['value'] == 'abc123'


def test_identifier_then_char():
    toks = tokens("abc.def")
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'dot', 'unquoted_identifier', 'eof']


# ---------------------------------------------------------------------------
# Whitespace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space", [' ', '\t', '\n', '\r'])
def test_whitespace_ignored(space):
    toks = tokens("a" + space + "b")
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'unquoted_identifier', 'eof']


def test_multiple_whitespace():
    toks = tokens("  a  b  ")
    assert len(toks) == 3
    assert toks[0]['value'] == 'a'
    assert toks[1]['value'] == 'b'


# ---------------------------------------------------------------------------
# Brackets
# ---------------------------------------------------------------------------

def test_flatten():
    toks = tokens("[]")
    assert toks[0] == {'type': 'flatten', 'value': '[]',
                       'start': 0, 'end': 2}


def test_filter():
    toks = tokens("[?")
    assert toks[0] == {'type': 'filter', 'value': '[?',
                       'start': 0, 'end': 2}


def test_lbracket_alone():
    toks = tokens("[")
    assert toks[0] == {'type': 'lbracket', 'value': '[',
                       'start': 0, 'end': 1}


def test_lbracket_then_identifier():
    toks = tokens("[a]")
    assert [t['type'] for t in toks] == [
        'lbracket', 'unquoted_identifier', 'rbracket', 'eof']


def test_lbracket_at_eof_next_char_none():
    # "[" followed by nothing: next_char is None -> lbracket
    toks = tokens("[")
    assert toks[0]['type'] == 'lbracket'


def test_flatten_positions():
    toks = tokens("x[]y")
    flatten = toks[1]
    assert flatten == {'type': 'flatten', 'value': '[]',
                       'start': 1, 'end': 3}


# ---------------------------------------------------------------------------
# Pipe / or
# ---------------------------------------------------------------------------

def test_single_pipe():
    toks = tokens("|")
    assert toks[0] == {'type': 'pipe', 'value': '|', 'start': 0, 'end': 0}


def test_double_pipe():
    toks = tokens("||")
    assert toks[0] == {'type': 'or', 'value': '||', 'start': 0, 'end': 1}


def test_pipe_in_expression():
    toks = tokens("a|b")
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'pipe', 'unquoted_identifier', 'eof']


# ---------------------------------------------------------------------------
# Ampersand / and / expref
# ---------------------------------------------------------------------------

def test_single_ampersand():
    toks = tokens("&")
    assert toks[0] == {'type': 'expref', 'value': '&', 'start': 0, 'end': 0}


def test_double_ampersand():
    toks = tokens("&&")
    assert toks[0] == {'type': 'and', 'value': '&&', 'start': 0, 'end': 1}


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

def test_lt():
    toks = tokens("<")
    assert toks[0] == {'type': 'lt', 'value': '<', 'start': 0, 'end': 0}


def test_lte():
    toks = tokens("<=")
    assert toks[0] == {'type': 'lte', 'value': '<=', 'start': 0, 'end': 1}


def test_gt():
    toks = tokens(">")
    assert toks[0] == {'type': 'gt', 'value': '>', 'start': 0, 'end': 0}


def test_gte():
    toks = tokens(">=")
    assert toks[0] == {'type': 'gte', 'value': '>=', 'start': 0, 'end': 1}


def test_not():
    toks = tokens("!")
    assert toks[0] == {'type': 'not', 'value': '!', 'start': 0, 'end': 0}


def test_ne():
    toks = tokens("!=")
    assert toks[0] == {'type': 'ne', 'value': '!=', 'start': 0, 'end': 1}


def test_eq():
    toks = tokens("==")
    assert toks[0] == {'type': 'eq', 'value': '==', 'start': 0, 'end': 1}


def test_eq_in_expression():
    toks = tokens("a==b")
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'eq', 'unquoted_identifier', 'eof']


# ---------------------------------------------------------------------------
# Single '=' error handling
# ---------------------------------------------------------------------------

def test_single_equals_raises():
    with pytest.raises(LexerError) as exc:
        tokens("=")
    assert exc.value.lexer_value == '='
    assert exc.value.message == "Unknown token '='"
    # At EOF: position equals current position (no rewind).
    assert exc.value.lexer_position == 1


def test_single_equals_mid_expression_raises():
    with pytest.raises(LexerError) as exc:
        tokens("a=b")
    assert exc.value.lexer_value == '='
    # Rewound one from current position (2) -> 1.
    assert exc.value.lexer_position == 1


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def test_single_number():
    toks = tokens("42")
    assert toks[0] == {'type': 'number', 'value': 42, 'start': 0, 'end': 2}


def test_zero_number():
    toks = tokens("0")
    assert toks[0] == {'type': 'number', 'value': 0, 'start': 0, 'end': 1}


def test_number_leading_zero():
    toks = tokens("007")
    assert toks[0] == {'type': 'number', 'value': 7, 'start': 0, 'end': 3}


def test_negative_number():
    toks = tokens("-5")
    assert toks[0] == {'type': 'number', 'value': -5, 'start': 0, 'end': 2}


def test_negative_number_multi_digit():
    toks = tokens("-123")
    assert toks[0] == {'type': 'number', 'value': -123, 'start': 0, 'end': 4}


def test_lone_minus_raises():
    with pytest.raises(LexerError) as exc:
        tokens("-")
    assert exc.value.message == "Unknown token '-'"
    assert exc.value.lexer_value == '-'
    assert exc.value.lexer_position == 0


def test_minus_in_expression_raises():
    with pytest.raises(LexerError) as exc:
        tokens("a-b")
    assert exc.value.lexer_value == '-'
    assert exc.value.lexer_position == 1


def test_number_in_expression():
    toks = tokens("a[1]")
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'lbracket', 'number', 'rbracket', 'eof']
    assert toks[2]['value'] == 1


def test_consume_number_direct():
    lexer = Lexer()
    lexer._initialize_for_expression("123abc")
    assert lexer._consume_number() == "123"


# ---------------------------------------------------------------------------
# Unknown tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("char", ['#', '?', ';', '&'.replace('&', '%')])
def test_unknown_token(char):
    with pytest.raises(LexerError) as exc:
        tokens(char)
    assert exc.value.message == "Unknown token %s" % char
    assert exc.value.lexer_value == char
    assert exc.value.lexer_position == 0


def test_unknown_token_position_mid_string():
    with pytest.raises(LexerError) as exc:
        tokens("ab#")
    assert exc.value.lexer_position == 2
    assert exc.value.lexer_value == '#'


# ---------------------------------------------------------------------------
# Raw string literals
# ---------------------------------------------------------------------------

def test_raw_string():
    toks = tokens("'abc'")
    assert toks[0] == {'type': 'literal', 'value': 'abc',
                       'start': 0, 'end': 5}


def test_raw_string_empty():
    toks = tokens("''")
    assert toks[0] == {'type': 'literal', 'value': '', 'start': 0, 'end': 2}


def test_raw_string_escaped_quote():
    toks = tokens("'a\\'b'")
    assert toks[0]['type'] == 'literal'
    assert toks[0]['value'] == "a'b"


def test_raw_string_keeps_backslash_before_other():
    toks = tokens("'a\\\\b'")
    assert toks[0]['type'] == 'literal'


def test_unclosed_raw_string():
    with pytest.raises(LexerError) as exc:
        tokens("'abc")
    assert exc.value.message == "Unclosed ' delimiter"
    assert exc.value.lexer_position == 0


# ---------------------------------------------------------------------------
# Quoted identifiers
# ---------------------------------------------------------------------------

def test_quoted_identifier():
    toks = tokens('"foo"')
    assert toks[0] == {'type': 'quoted_identifier', 'value': 'foo',
                       'start': 0, 'end': 5}


def test_quoted_identifier_with_spaces():
    toks = tokens('"foo bar"')
    assert toks[0]['value'] == 'foo bar'


def test_quoted_identifier_empty():
    toks = tokens('""')
    assert toks[0]['type'] == 'quoted_identifier'
    assert toks[0]['value'] == ''
    assert toks[0]['end'] == 2


def test_quoted_identifier_escaped():
    toks = tokens('"a\\nb"')
    assert toks[0]['value'] == 'a\nb'


def test_unclosed_quoted_identifier():
    with pytest.raises(LexerError) as exc:
        tokens('"foo')
    assert exc.value.message == "Unclosed \" delimiter"
    assert exc.value.lexer_position == 0


def test_invalid_quoted_identifier():
    with pytest.raises(LexerError) as exc:
        tokens('"\\x"')
    assert exc.value.lexer_value == '"\\x"'
    assert exc.value.lexer_position == 0


def test_quoted_identifier_in_expression():
    toks = tokens('a."b c"')
    assert [t['type'] for t in toks] == [
        'unquoted_identifier', 'dot', 'quoted_identifier', 'eof']


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

def test_literal_number():
    toks = tokens("`1`")
    assert toks[0]['type'] == 'literal'
    assert toks[0]['value'] == 1


def test_literal_string():
    toks = tokens('`"foo"`')
    assert toks[0]['type'] == 'literal'
    assert toks[0]['value'] == 'foo'


def test_literal_array():
    toks = tokens('`[1, 2, 3]`')
    assert toks[0]['value'] == [1, 2, 3]


def test_literal_boolean():
    toks = tokens('`true`')
    assert toks[0]['value'] is True


def test_literal_null():
    toks = tokens('`null`')
    assert toks[0]['value'] is None


def test_literal_escaped_backtick():
    toks = tokens('`"a\\`b"`')
    assert toks[0]['type'] == 'literal'
    assert toks[0]['value'] == 'a`b'


def test_literal_deprecated_syntax_warns():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        toks = tokens("`foo`")
    assert toks[0]['type'] == 'literal'
    assert toks[0]['value'] == 'foo'
    assert len(record) == 1
    assert record[0].category is PendingDeprecationWarning
    assert "deprecated string literal syntax" in str(record[0].message)


def test_literal_deprecated_strips_leading_whitespace():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        toks = tokens("`   foo`")
    assert toks[0]['value'] == 'foo'


def test_literal_bad_token_raises():
    with pytest.raises(LexerError) as exc:
        tokens("`{bad json}`")
    assert exc.value.message.startswith("Bad token ")
    assert exc.value.lexer_position == 0


def test_unclosed_literal():
    with pytest.raises(LexerError) as exc:
        tokens("`1")
    assert exc.value.message == "Unclosed ` delimiter"
    assert exc.value.lexer_position == 0


def test_literal_positions():
    # 'end' is relative token length (position - start).
    toks = tokens("`1`")
    assert toks[0]['start'] == 0


# ---------------------------------------------------------------------------
# _consume_until
# ---------------------------------------------------------------------------

def test_consume_until_basic():
    lexer = Lexer()
    lexer._initialize_for_expression("'abc'")
    assert lexer._consume_until("'") == "abc"


def test_consume_until_escaped_delimiter():
    lexer = Lexer()
    lexer._initialize_for_expression("'a\\'b'")
    result = lexer._consume_until("'")
    assert result == "a\\'b"


def test_consume_until_eof_raises():
    lexer = Lexer()
    lexer._initialize_for_expression("'abc")
    with pytest.raises(LexerError) as exc:
        lexer._consume_until("'")
    assert "Unclosed ' delimiter" in exc.value.message


# ---------------------------------------------------------------------------
# _next behavior
# ---------------------------------------------------------------------------

def test_next_advances():
    lexer = Lexer()
    lexer._initialize_for_expression("abc")
    assert lexer._next() == 'b'
    assert lexer._position == 1
    assert lexer._next() == 'c'
    assert lexer._position == 2


def test_next_returns_none_at_end():
    lexer = Lexer()
    lexer._initialize_for_expression("a")
    assert lexer._next() is None
    assert lexer._current is None


def test_next_single_char_expression_stays():
    lexer = Lexer()
    lexer._initialize_for_expression("a")
    # length - 1 == 0 == position
    assert lexer._next() is None
    assert lexer._position == 0


# ---------------------------------------------------------------------------
# _match_or_else
# ---------------------------------------------------------------------------

def test_match_or_else_match():
    lexer = Lexer()
    lexer._initialize_for_expression("||")
    tok = lexer._match_or_else('|', 'or', 'pipe')
    assert tok == {'type': 'or', 'value': '||', 'start': 0, 'end': 1}


def test_match_or_else_no_match():
    lexer = Lexer()
    lexer._initialize_for_expression("|a")
    tok = lexer._match_or_else('|', 'or', 'pipe')
    assert tok == {'type': 'pipe', 'value': '|', 'start': 0, 'end': 0}


def test_match_or_else_at_eof():
    lexer = Lexer()
    lexer._initialize_for_expression("|")
    tok = lexer._match_or_else('|', 'or', 'pipe')
    assert tok['type'] == 'pipe'
    assert tok['value'] == '|'


# ---------------------------------------------------------------------------
# LexerError string formatting (integration)
# ---------------------------------------------------------------------------

def test_lexer_error_positions_in_message():
    with pytest.raises(LexerError) as exc:
        list(Lexer().tokenize("="))
    assert exc.value.lexer_position == 1


# ---------------------------------------------------------------------------
# Complex expression
# ---------------------------------------------------------------------------

def test_complex_expression():
    expr = "foo[?bar==`1`].baz[]"
    toks = tokens(expr)
    types = [t['type'] for t in toks]
    assert types == [
        'unquoted_identifier',
        'filter',
        'unquoted_identifier',
        'eq',
        'literal',
        'rbracket',
        'dot',
        'unquoted_identifier',
        'flatten',
        'eof',
    ]


def test_star_simple_token():
    toks = tokens("*")
    assert toks[0]['type'] == 'star'


def test_full_simple_token_sequence():
    expr = ". * ] , : @ ( ) { }"
    types = [t['type'] for t in tokens(expr)]
    assert types == [
        'dot', 'star', 'rbracket', 'comma', 'colon', 'current',
        'lparen', 'rparen', 'lbrace', 'rbrace', 'eof']


def test_tokenize_is_generator():
    lexer = Lexer()
    gen = lexer.tokenize("abc")
    assert hasattr(gen, '__iter__')
    assert hasattr(gen, '__next__')
