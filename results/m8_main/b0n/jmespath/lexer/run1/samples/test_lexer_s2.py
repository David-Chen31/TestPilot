import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokens(expression):
    return list(Lexer().tokenize(expression))


def types(expression):
    return [t['type'] for t in tokens(expression)]


def values(expression):
    return [t['value'] for t in tokens(expression)]


class TestEmptyExpression:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(''))

    def test_none_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(None))


class TestSimpleTokens:
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
    def test_simple_token(self, char, token_type):
        result = tokens(char)
        assert len(result) == 2
        assert result[0] == {
            'type': token_type, 'value': char, 'start': 0, 'end': 1,
        }
        assert result[1] == {
            'type': 'eof', 'value': '', 'start': 1, 'end': 1,
        }

    def test_multiple_simple_tokens(self):
        result = tokens('.*')
        assert result == [
            {'type': 'dot', 'value': '.', 'start': 0, 'end': 1},
            {'type': 'star', 'value': '*', 'start': 1, 'end': 2},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]


class TestIdentifiers:
    def test_single_identifier(self):
        result = tokens('foo')
        assert result[0] == {
            'type': 'unquoted_identifier', 'value': 'foo',
            'start': 0, 'end': 3,
        }
        assert result[-1]['type'] == 'eof'

    def test_identifier_with_underscore_and_digits(self):
        result = tokens('_foo123_bar')
        assert result[0]['type'] == 'unquoted_identifier'
        assert result[0]['value'] == '_foo123_bar'
        assert result[0]['start'] == 0
        assert result[0]['end'] == 10

    def test_two_identifiers_separated_by_space(self):
        result = tokens('foo bar')
        assert result[0]['value'] == 'foo'
        assert result[0]['start'] == 0
        assert result[1]['value'] == 'bar'
        assert result[1]['start'] == 4
        assert result[1]['end'] == 7

    def test_identifier_followed_by_dot(self):
        result = tokens('foo.bar')
        assert result[0]['value'] == 'foo'
        assert result[1]['type'] == 'dot'
        assert result[2]['value'] == 'bar'
        assert result[2]['start'] == 4
        assert result[2]['end'] == 7


class TestWhitespace:
    def test_whitespace_is_skipped(self):
        assert types('   ') == ['eof']

    def test_whitespace_between_tokens(self):
        result = tokens('\t foo \n . \r bar ')
        assert [t['type'] for t in result] == [
            'unquoted_identifier', 'dot', 'unquoted_identifier', 'eof',
        ]
        assert result[0]['value'] == 'foo'
        assert result[2]['value'] == 'bar'

    def test_whitespace_does_not_affect_positions(self):
        result = tokens(' a ')
        assert result[0]['start'] == 1
        assert result[0]['end'] == 2


class TestBrackets:
    def test_lbracket(self):
        result = tokens('[')
        assert result[0] == {
            'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1,
        }
        assert result[1]['type'] == 'eof'

    def test_flatten(self):
        result = tokens('[]')
        assert result[0] == {
            'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2,
        }
        assert result[1]['type'] == 'eof'

    def test_filter(self):
        result = tokens('[?')
        assert result[0] == {
            'type': 'filter', 'value': '[?', 'start': 0, 'end': 2,
        }
        assert result[1]['type'] == 'eof'

    def test_lbracket_then_number(self):
        result = tokens('[0]')
        assert [t['type'] for t in result] == [
            'lbracket', 'number', 'rbracket', 'eof',
        ]
        assert result[0]['start'] == 0
        assert result[0]['end'] == 1
        assert result[1]['value'] == 0
        assert result[1]['start'] == 1
        assert result[1]['end'] == 2
        assert result[2]['start'] == 2
        assert result[2]['end'] == 3

    def test_double_flatten(self):
        result = tokens('[][]')
        assert [t['type'] for t in result] == ['flatten', 'flatten', 'eof']
        assert result[0]['start'] == 0
        assert result[0]['end'] == 2
        assert result[1]['start'] == 2
        assert result[1]['end'] == 4

    def test_unclosed_lbracket(self):
        # '[' with nothing after is a plain lbracket token
        result = tokens('[')
        assert result[0]['type'] == 'lbracket'


class TestRawString:
    def test_simple_raw_string(self):
        result = tokens("'foo'")
        assert result[0] == {
            'type': 'literal', 'value': 'foo', 'start': 0, 'end': 5,
        }
        assert result[1]['type'] == 'eof'

    def test_raw_string_with_escaped_quote(self):
        result = tokens(r"'foo\'bar'")
        assert result[0]['type'] == 'literal'
        assert result[0]['value'] == "foo'bar"

    def test_raw_string_with_backslash(self):
        result = tokens(r"'foo\\bar'")
        assert result[0]['value'] == r'foo\\bar'

    def test_empty_raw_string(self):
        result = tokens("''")
        assert result[0]['type'] == 'literal'
        assert result[0]['value'] == ''

    def test_unclosed_raw_string(self):
        with pytest.raises(LexerError) as excinfo:
            tokens("'foo")
        assert excinfo.value.message == "Unclosed ' delimiter"
        assert excinfo.value.lexer_position == 0


class TestPipeOr:
    def test_pipe(self):
        result = tokens('|')
        assert result[0] == {
            'type': 'pipe', 'value': '|', 'start': 0, 'end': 0,
        }
        assert result[1]['type'] == 'eof'

    def test_or(self):
        result = tokens('||')
        assert result[0] == {
            'type': 'or', 'value': '||', 'start': 0, 'end': 1,
        }
        assert result[1]['type'] == 'eof'

    def test_or_positions(self):
        result = tokens('a || b')
        assert result[1]['type'] == 'or'
        assert result[1]['start'] == 2
        assert result[1]['end'] == 3


class TestAmpersand:
    def test_expref(self):
        result = tokens('&')
        assert result[0] == {
            'type': 'expref', 'value': '&', 'start': 0, 'end': 0,
        }
        assert result[1]['type'] == 'eof'

    def test_and(self):
        result = tokens('&&')
        assert result[0] == {
            'type': 'and', 'value': '&&', 'start': 0, 'end': 1,
        }
        assert result[1]['type'] == 'eof'


class TestLiteral:
    def test_json_number_literal(self):
        result = tokens('`1`')
        assert result[0]['type'] == 'literal'
        assert result[0]['value'] == 1
        assert result[0]['start'] == 0

    def test_json_bool_literal(self):
        result = tokens('`true`')
        assert result[0]['value'] is True

    def test_json_null_literal(self):
        result = tokens('`null`')
        assert result[0]['value'] is None

    def test_json_string_literal(self):
        result = tokens('`"foo"`')
        assert result[0]['value'] == 'foo'

    def test_json_array_literal(self):
        result = tokens('`[1, 2]`')
        assert result[0]['value'] == [1, 2]

    def test_json_object_literal(self):
        result = tokens('`{"a": 1}`')
        assert result[0]['value'] == {'a': 1}

    def test_escaped_backtick(self):
        result = tokens(r'`"a\`b"`')
        assert result[0]['type'] == 'literal'
        assert result[0]['value'] == 'a`b'

    def test_deprecated_string_literal_warns(self):
        with pytest.warns(PendingDeprecationWarning):
            result = tokens('`foo`')
        assert result[0]['type'] == 'literal'
        assert result[0]['value'] == 'foo'

    def test_deprecated_string_literal_strips_leading_whitespace(self):
        with pytest.warns(PendingDeprecationWarning):
            result = tokens('`  foo`')
        assert result[0]['value'] == 'foo'

    def test_bad_literal(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('`[1,`')
        assert excinfo.value.message.startswith('Bad token')

    def test_unclosed_literal(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('`foo')
        assert excinfo.value.message == 'Unclosed ` delimiter'


class TestNumbers:
    def test_single_digit(self):
        result = tokens('1')
        assert result[0] == {
            'type': 'number', 'value': 1, 'start': 0, 'end': 1,
        }
        assert result[1]['type'] == 'eof'

    def test_multi_digit(self):
        result = tokens('123')
        assert result[0]['value'] == 123
        assert result[0]['start'] == 0
        assert result[0]['end'] == 3

    def test_zero(self):
        result = tokens('0')
        assert result[0]['value'] == 0

    def test_negative_number(self):
        result = tokens('-5')
        assert result[0]['type'] == 'number'
        assert result[0]['value'] == -5
        assert result[0]['start'] == 0
        assert result[0]['end'] == 2

    def test_negative_multi_digit(self):
        result = tokens('-123')
        assert result[0]['value'] == -123

    def test_bare_minus_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('-')
        assert excinfo.value.message == "Unknown token '-'"
        assert excinfo.value.lexer_position == 0

    def test_minus_followed_by_non_digit_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('-a')
        assert excinfo.value.message == "Unknown token '-'"
        assert excinfo.value.lexer_position == 0

    def test_number_in_brackets(self):
        result = tokens('[1]')
        assert result[1]['type'] == 'number'
        assert result[1]['value'] == 1


class TestQuotedIdentifier:
    def test_simple(self):
        result = tokens('"foo"')
        assert result[0]['type'] == 'quoted_identifier'
        assert result[0]['value'] == 'foo'
        assert result[0]['start'] == 0
        assert result[0]['end'] == 5

    def test_with_escapes(self):
        result = tokens(r'"a\"b"')
        assert result[0]['type'] == 'quoted_identifier'
        assert result[0]['value'] == 'a"b'

    def test_unclosed(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('"foo')
        assert excinfo.value.message == 'Unclosed " delimiter'

    def test_invalid_json_escape_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokens(r'"\x"')
        assert excinfo.value.lexer_position == 0


class TestComparators:
    def test_lt(self):
        result = tokens('<')
        assert result[0] == {
            'type': 'lt', 'value': '<', 'start': 0, 'end': 0,
        }

    def test_lte(self):
        result = tokens('<=')
        assert result[0] == {
            'type': 'lte', 'value': '<=', 'start': 0, 'end': 1,
        }

    def test_gt(self):
        result = tokens('>')
        assert result[0]['type'] == 'gt'
        assert result[0]['value'] == '>'

    def test_gte(self):
        result = tokens('>=')
        assert result[0]['type'] == 'gte'
        assert result[0]['value'] == '>='

    def test_not(self):
        result = tokens('!')
        assert result[0]['type'] == 'not'
        assert result[0]['value'] == '!'

    def test_ne(self):
        result = tokens('!=')
        assert result[0]['type'] == 'ne'
        assert result[0]['value'] == '!='

    def test_eq(self):
        result = tokens('==')
        assert result[0] == {
            'type': 'eq', 'value': '==', 'start': 0, 'end': 1,
        }
        assert result[1]['type'] == 'eof'

    def test_single_equals_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('=')
        assert excinfo.value.message == "Unknown token '='"
        assert excinfo.value.lexer_value == '='
        # At EOF, position is not rewound.
        assert excinfo.value.lexer_position == 0

    def test_single_equals_not_at_eof_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('=a')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '='

    def test_comparator_in_expression(self):
        result = tokens('a==b')
        assert [t['type'] for t in result] == [
            'unquoted_identifier', 'eq', 'unquoted_identifier', 'eof',
        ]
        assert result[1]['start'] == 1
        assert result[1]['end'] == 2


class TestUnknownToken:
    def test_unknown_character(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('#')
        assert excinfo.value.message == 'Unknown token #'
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '#'

    def test_unknown_character_after_token(self):
        with pytest.raises(LexerError) as excinfo:
            tokens('foo#')
        assert excinfo.value.lexer_position == 3
        assert excinfo.value.lexer_value == '#'


class TestEof:
    def test_eof_position_single_char(self):
        result = tokens('a')
        assert result[-1] == {
            'type': 'eof', 'value': '', 'start': 1, 'end': 1,
        }

    def test_eof_position_multi_char(self):
        result = tokens('abc')
        assert result[-1] == {
            'type': 'eof', 'value': '', 'start': 3, 'end': 3,
        }

    def test_eof_token_always_last(self):
        result = tokens('foo.bar')
        assert result[-1]['type'] == 'eof'
        assert all(t['type'] != 'eof' for t in result[:-1])


class TestComplexExpressions:
    def test_nested_field_access(self):
        result = tokens('foo.bar[0]')
        assert [t['type'] for t in result] == [
            'unquoted_identifier', 'dot', 'unquoted_identifier',
            'lbracket', 'number', 'rbracket', 'eof',
        ]

    def test_filter_expression(self):
        result = tokens('foo[?bar>`1`]')
        assert [t['type'] for t in result] == [
            'unquoted_identifier', 'filter', 'unquoted_identifier',
            'gt', 'literal', 'rbracket', 'eof',
        ]

    def test_multiselect_list(self):
        result = tokens('foo[bar, baz]')
        assert 'comma' in [t['type'] for t in result]

    def test_multiselect_hash(self):
        result = tokens('{a: foo, b: bar}')
        assert result[0]['type'] == 'lbrace'
        assert result[-2]['type'] == 'rbrace'

    def test_positions_are_consistent(self):
        expression = 'foo.bar'
        for token in tokens(expression):
            if token['type'] == 'unquoted_identifier':
                s, e = token['start'], token['end']
                assert expression[s:e] == token['value']


class TestLexerReuse:
    def test_lexer_can_be_reused(self):
        lexer = Lexer()
        first = list(lexer.tokenize('foo'))
        second = list(lexer.tokenize('bar'))
        assert first[0]['value'] == 'foo'
        assert second[0]['value'] == 'bar'
