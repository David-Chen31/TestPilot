import string
import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expr):
    return list(Lexer().tokenize(expr))


def toks(expr):
    """Return list of (type, value) tuples, ignoring positions."""
    return [(t['type'], t['value']) for t in tokenize(expr)]


# ---------------------------------------------------------------- basic structure

class TestTokenizationBasics:
    def test_class_attributes(self):
        assert Lexer.START_IDENTIFIER == set(string.ascii_letters + '_')
        assert Lexer.VALID_IDENTIFIER == set(string.ascii_letters + string.digits + '_')
        assert Lexer.VALID_NUMBER == set(string.digits)
        assert Lexer.WHITESPACE == set(" \t\n\r")

    def test_simple_tokens_table(self):
        assert Lexer.SIMPLE_TOKENS == {
            '.': 'dot',
            '*': 'star',
            ']': 'rbracket',
            ',': 'comma',
            ':': 'colon',
            '@': 'current',
            '(': 'lparen',
            ')': 'rparen',
            '{': 'lbrace',
            '}': 'rbrace',
        }

    def test_eof_appended_every_time(self):
        result = tokenize('foo')
        assert result[-1] == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}

    def test_eof_single_char(self):
        result = tokenize('.')
        # '.' -> dot at 0, eof at 1,1
        assert result == [
            {'type': 'dot', 'value': '.', 'start': 0, 'end': 1},
            {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
        ]

    def test_eof_position_is_length(self):
        result = tokenize('abcdef')
        assert result[-1]['start'] == 6
        assert result[-1]['end'] == 6

    def test_generator_returns_generator(self):
        gen = Lexer().tokenize('a')
        # It should be a generator; first next is unquoted identifier.
        assert next(gen) == {
            'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1
        }


# ---------------------------------------------------------------- empty expression

class TestEmptyExpression:
    @pytest.mark.parametrize('expr', ['', None])
    def test_empty_expression_raises(self, expr):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(expr))

    def test_empty_expression_message(self):
        with pytest.raises(EmptyExpressionError) as exc:
            list(Lexer().tokenize(''))
        assert 'cannot be empty' in str(exc.value)


# ---------------------------------------------------------------- simple tokens

class TestSimpleTokens:
    @pytest.mark.parametrize('char,expected_type', [
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
    def test_each_simple_token(self, char, expected_type):
        result = tokenize(char)
        assert result[0] == {
            'type': expected_type, 'value': char, 'start': 0, 'end': 1
        }

    def test_simple_token_positions(self):
        # ".*" -> dot at 0, star at 1
        result = tokenize('.*')
        assert result[0]['start'] == 0 and result[0]['end'] == 1
        assert result[1]['start'] == 1 and result[1]['end'] == 2


# ---------------------------------------------------------------- identifiers

class TestUnquotedIdentifier:
    def test_simple(self):
        assert toks('foo') == [('unquoted_identifier', 'foo'), ('eof', '')]

    def test_underscore_start(self):
        assert toks('_foo') == [('unquoted_identifier', '_foo'), ('eof', '')]

    def test_digits_after_start(self):
        assert toks('a1b2') == [('unquoted_identifier', 'a1b2'), ('eof', '')]

    def test_stops_at_non_identifier(self):
        assert toks('foo.bar') == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('unquoted_identifier', 'bar'),
            ('eof', ''),
        ]

    def test_position(self):
        t = tokenize('ab cd')[0]
        assert t == {'type': 'unquoted_identifier', 'value': 'ab',
                     'start': 0, 'end': 2}

    def test_second_identifier_position(self):
        t = tokenize('ab cd')[1]
        assert t == {'type': 'unquoted_identifier', 'value': 'cd',
                     'start': 3, 'end': 5}


# ---------------------------------------------------------------- numbers

class TestNumbers:
    def test_positive_integer(self):
        assert tokenize('123')[0] == {
            'type': 'number', 'value': 123, 'start': 0, 'end': 3
        }

    def test_single_digit(self):
        assert toks('7') == [('number', 7), ('eof', '')]

    def test_leading_zero(self):
        assert toks('007') == [('number', 7), ('eof', '')]

    def test_negative(self):
        assert toks('-123') == [('number', -123), ('eof', '')]

    def test_negative_position(self):
        t = tokenize('-12')[0]
        assert t['start'] == 0 and t['end'] == 3

    def test_lone_minus_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('-'))
        assert exc.value.lexer_position == 0
        assert exc.value.lexer_value == '-'
        assert "Unknown token" in exc.value.message

    def test_minus_followed_by_non_digit_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('-a'))
        assert exc.value.lexer_value == '-'
        assert "Unknown token" in exc.value.message


# ---------------------------------------------------------------- brackets

class TestBrackets:
    def test_lbracket(self):
        r = tokenize('[')
        assert r[0] == {'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1}

    def test_flatten(self):
        r = tokenize('[]')
        assert r[0] == {'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2}

    def test_filter(self):
        r = tokenize('[?')
        assert r[0] == {'type': 'filter', 'value': '[?', 'start': 0, 'end': 2}

    def test_nested_flatten_positions(self):
        r = tokenize('a[]')
        assert r[1] == {'type': 'flatten', 'value': '[]', 'start': 1, 'end': 3}

    def test_flatten_then_more(self):
        assert toks('[]foo') == [
            ('flatten', '[]'),
            ('unquoted_identifier', 'foo'),
            ('eof', ''),
        ]

    def test_filter_then_more(self):
        assert toks('[?a') == [
            ('filter', '[?'),
            ('unquoted_identifier', 'a'),
            ('eof', ''),
        ]

    def test_lbracket_then_content(self):
        assert toks('[0]') == [
            ('lbracket', '['),
            ('number', 0),
            ('rbracket', ']'),
            ('eof', ''),
        ]


# ---------------------------------------------------------------- raw strings

class TestRawStringLiteral:
    def test_simple(self):
        r = tokenize("'foo'")
        assert r[0] == {
            'type': 'literal', 'value': 'foo', 'start': 0, 'end': 5
        }

    def test_empty(self):
        r = tokenize("''")
        assert r[0]['type'] == 'literal'
        assert r[0]['value'] == ''

    def test_escaped_quote(self):
        r = tokenize(r"'foo\'bar'")
        assert r[0]['value'] == "foo'bar"

    def test_backslash_preserved(self):
        r = tokenize(r"'a\\b'")
        # backslash handling: \\ stays as two backslashes in value
        assert r[0]['type'] == 'literal'

    def test_unclosed_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize("'abc"))
        assert 'Unclosed' in exc.value.message
        assert exc.value.lexer_position == 0

    def test_unclosed_after_content(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize("  'abc"))
        assert 'Unclosed' in exc.value.message
        assert exc.value.lexer_position == 2


# ---------------------------------------------------------------- quark literals

class TestLiteral:
    def test_numeric_json(self):
        r = tokenize('`123`')
        assert r[0]['type'] == 'literal'
        assert r[0]['value'] == 123

    def test_string_json(self):
        r = tokenize('`"foo"`')
        assert r[0]['value'] == 'foo'

    def test_object_json(self):
        r = tokenize('`{"a": 1}`')
        assert r[0]['value'] == {'a': 1}

    def test_list_json(self):
        r = tokenize('`[1, 2]`')
        assert r[0]['value'] == [1, 2]

    def test_true_literal(self):
        assert tokenize('`true`')[0]['value'] is True

    def test_null_literal(self):
        assert tokenize('`null`')[0]['value'] is None

    def test_deprecated_string_syntax(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = tokenize('`foo`')
        assert r[0]['value'] == 'foo'
        assert any(issubclass(x.category, PendingDeprecationWarning) for x in w)

    def test_deprecated_string_strips_leading_whitespace(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            r = tokenize('`  hello`')
        assert r[0]['value'] == 'hello'

    def test_bad_literal_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('`{"a": }`'))
        assert 'Bad token' in exc.value.message

    def test_unclosed_literal(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('`123'))
        assert 'Unclosed' in exc.value.message

    def test_escaped_backtick(self):
        r = tokenize(r'`"a\`b"`')
        assert r[0]['value'] == 'a`b'


# ---------------------------------------------------------------- quoted identifiers

class TestQuotedIdentifier:
    def test_simple(self):
        r = tokenize('"foo"')
        assert r[0] == {
            'type': 'quoted_identifier', 'value': 'foo',
            'start': 0, 'end': 5,
        }

    def test_empty(self):
        r = tokenize('""')
        assert r[0]['type'] == 'quoted_identifier'
        assert r[0]['value'] == ''

    def test_escape(self):
        r = tokenize(r'"a\"b"')
        assert r[0]['value'] == 'a"b'

    def test_unicode_escape(self):
        r = tokenize(r'"\u00e9"')
        assert r[0]['value'] == '\u00e9'

    def test_invalid_json_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize(r'"\uZZZZ"'))
        assert exc.value.lexer_position == 0

    def test_unclosed(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('"abc'))
        assert 'Unclosed' in exc.value.message


# ---------------------------------------------------------------- operators

class TestOperators:
    def test_pipe(self):
        assert toks('|') == [('pipe', '|'), ('eof', '')]

    def test_or(self):
        assert toks('||') == [('or', '||'), ('eof', '')]

    def test_expref(self):
        assert toks('&') == [('expref', '&'), ('eof', '')]

    def test_and(self):
        assert toks('&&') == [('and', '&&'), ('eof', '')]

    def test_lt(self):
        assert toks('<') == [('lt', '<'), ('eof', '')]

    def test_lte(self):
        assert toks('<=') == [('lte', '<='), ('eof', '')]

    def test_gt(self):
        assert toks('>') == [('gt', '>'), ('eof', '')]

    def test_gte(self):
        assert toks('>=') == [('gte', '>='), ('eof', '')]

    def test_not(self):
        assert toks('!') == [('not', '!'), ('eof', '')]

    def test_ne(self):
        assert toks('!=') == [('ne', '!='), ('eof', '')]

    def test_eq(self):
        assert toks('==') == [('eq', '=='), ('eof', '')]

    def test_single_eq_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('='))
        assert exc.value.lexer_position == 0
        assert exc.value.lexer_value == '='
        assert "Unknown token '='" in exc.value.message

    def test_single_eq_before_other_raises(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('=a'))
        assert exc.value.lexer_position == 0
        assert "Unknown token '='" in exc.value.message

    def test_single_eq_at_eof_in_expression(self):
        # 'a=' : '=' is at position 1, _next() returns None -> position = 1
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('a='))
        assert exc.value.lexer_position == 1
        assert "Unknown token '='" in exc.value.message

    def test_single_eq_mid_expression(self):
        # '=a' -- next is 'a' so position = 0 (rewound)
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('=a'))
        assert exc.value.lexer_position == 0


# ---------------------------------------------------------------- operators positions

class TestOperatorPositions:
    def test_pipe_position(self):
        r = tokenize(' | ')[0]
        assert r['start'] == 1 and r['end'] == 2

    def test_eq_position(self):
        r = tokenize(' == ')[0]
        assert r == {'type': 'eq', 'value': '==', 'start': 1, 'end': 2}

    def test_lte_position(self):
        r = tokenize('a<=b')[1]
        assert r['type'] == 'lte'
        assert r['start'] == 1
        assert r['end'] == 2  # note implementation uses start + 1

    def test_and_position(self):
        r = tokenize('a&&b')[1]
        assert r['type'] == 'and'
        assert r['start'] == 1
        assert r['end'] == 2


# ---------------------------------------------------------------- whitespace

class TestWhitespace:
    def test_whitespace_skipped(self):
        assert toks('  a  b  ') == [
            ('unquoted_identifier', 'a'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]

    @pytest.mark.parametrize('ws', [' ', '\t', '\n', '\r'])
    def test_each_whitespace(self, ws):
        assert toks(ws + 'a') == [('unquoted_identifier', 'a'), ('eof', '')]

    def test_only_whitespace(self):
        assert toks('   ') == [('eof', '')]

    def test_positions_after_whitespace(self):
        r = tokenize('   a')
        assert r[0]['start'] == 3 and r[0]['end'] == 4


# ---------------------------------------------------------------- unknown tokens

class TestUnknownTokens:
    @pytest.mark.parametrize('char', ['#', '$', '%', '^', '~', ';', '?'])
    def test_unknown_char_raises(self, char):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize(char))
        assert exc.value.lexer_value == char
        assert 'Unknown token' in exc.value.message

    def test_unknown_char_position(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('a#'))
        assert exc.value.lexer_position == 1
        assert exc.value.lexer_value == '#'


# ---------------------------------------------------------------- complex expressions

class TestComplexExpressions:
    def test_dotted_field(self):
        assert toks('foo.bar.baz') == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('unquoted_identifier', 'bar'),
            ('dot', '.'),
            ('unquoted_identifier', 'baz'),
            ('eof', ''),
        ]

    def test_index_expression(self):
        assert toks('foo[0]') == [
            ('unquoted_identifier', 'foo'),
            ('lbracket', '['),
            ('number', 0),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_filter_expression(self):
        assert toks('foo[?a==`1`]') == [
            ('unquoted_identifier', 'foo'),
            ('filter', '[?'),
            ('unquoted_identifier', 'a'),
            ('eq', '=='),
            ('literal', 1),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_multiselect_list(self):
        assert toks('foo[a, b]') == [
            ('unquoted_identifier', 'foo'),
            ('lbracket', '['),
            ('unquoted_identifier', 'a'),
            ('comma', ','),
            ('unquoted_identifier', 'b'),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_multiselect_hash(self):
        assert toks('{sort: foo}') == [
            ('lbrace', '{'),
            ('unquoted_identifier', 'sort'),
            ('colon', ':'),
            ('unquoted_identifier', 'foo'),
            ('rbrace', '}'),
            ('eof', ''),
        ]

    def test_pipe_expression(self):
        assert toks('foo | bar') == [
            ('unquoted_identifier', 'foo'),
            ('pipe', '|'),
            ('unquoted_identifier', 'bar'),
            ('eof', ''),
        ]

    def test_function_call(self):
        assert toks('length(@)') == [
            ('unquoted_identifier', 'length'),
            ('lparen', '('),
            ('current', '@'),
            ('rparen', ')'),
            ('eof', ''),
        ]

    def test_all_simple_tokens_in_sequence(self):
        assert toks('.*],:@(){}') == [
            ('dot', '.'),
            ('star', '*'),
            ('rbracket', ']'),
            ('comma', ','),
            ('colon', ':'),
            ('current', '@'),
            ('lparen', '('),
            ('rparen', ')'),
            ('lbrace', '{'),
            ('rbrace', '}'),
            ('eof', ''),
        ]


# ---------------------------------------------------------------- LexerError string

class TestLexerErrorFormatting:
    def test_str_contains_message_and_pointer(self):
        with pytest.raises(LexerError) as exc:
            list(Lexer().tokenize('#'))
        # expression is None because lexer doesn't set it
        s = str(exc.value)
        assert 'Bad jmespath expression' in s
        assert "Unknown token #" in s
        assert '^' in s
