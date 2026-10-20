import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


def types_and_values(expression):
    return [(t['type'], t['value']) for t in tokenize(expression)]


class TestEmptyExpression:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyExpressionError):
            Lexer().tokenize('')

    def test_none_raises(self):
        with pytest.raises(EmptyExpressionError):
            Lexer().tokenize(None)

    def test_tokenize_is_generator_lazy(self):
        # tokenize is a generator, so calling it alone shouldn't raise.
        gen = Lexer().tokenize('')
        with pytest.raises(EmptyExpressionError):
            next(gen)


class TestSimpleTokens:
    @pytest.mark.parametrize('char,type_', [
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
    def test_simple_token(self, char, type_):
        tokens = tokenize(char)
        assert tokens[0] == {'type': type_, 'value': char,
                             'start': 0, 'end': 1}
        assert tokens[1] == {'type': 'eof', 'value': '',
                             'start': 1, 'end': 1}

    def test_simple_tokens_sequence(self):
        tokens = tokenize('{@:.,}')
        assert [(t['type'], t['value']) for t in tokens] == [
            ('lbrace', '{'),
            ('current', '@'),
            ('colon', ':'),
            ('dot', '.'),
            ('comma', ','),
            ('rbrace', '}'),
            ('eof', ''),
        ]

    def test_positions(self):
        tokens = tokenize('. *')
        # whitespace between
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 1
        assert tokens[1]['start'] == 2
        assert tokens[1]['end'] == 3


class TestIdentifiers:
    def test_simple(self):
        tokens = tokenize('foo')
        assert tokens[0] == {'type': 'unquoted_identifier', 'value': 'foo',
                             'start': 0, 'end': 3}
        assert tokens[1]['type'] == 'eof'

    def test_with_underscore_and_digits(self):
        tokens = tokenize('foo_bar123')
        assert tokens[0]['type'] == 'unquoted_identifier'
        assert tokens[0]['value'] == 'foo_bar123'
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 10

    def test_starts_with_underscore(self):
        tokens = tokenize('_foo')
        assert tokens[0]['value'] == '_foo'
        assert tokens[0]['type'] == 'unquoted_identifier'

    def test_multiple_identifiers(self):
        toks = types_and_values('foo.bar')
        assert toks == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('unquoted_identifier', 'bar'),
            ('eof', ''),
        ]

    def test_identifier_then_number_not_joined(self):
        toks = types_and_values('foo.0')
        assert toks == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('number', 0),
            ('eof', ''),
        ]


class TestWhitespace:
    @pytest.mark.parametrize('ws', [' ', '\t', '\n', '\r'])
    def test_whitespace_skipped(self, ws):
        tokens = tokenize(ws + 'foo' + ws)
        assert len(tokens) == 2
        assert tokens[0]['type'] == 'unquoted_identifier'


class TestBrackets:
    def test_flatten(self):
        tokens = tokenize('[]')
        assert tokens[0] == {'type': 'flatten', 'value': '[]',
                             'start': 0, 'end': 2}
        assert tokens[1]['type'] == 'eof'

    def test_filter(self):
        tokens = tokenize('[?')
        assert tokens[0] == {'type': 'filter', 'value': '[?',
                             'start': 0, 'end': 2}
        assert tokens[1]['type'] == 'eof'

    def test_lbracket(self):
        tokens = tokenize('[0]')
        assert tokens[0] == {'type': 'lbracket', 'value': '[',
                             'start': 0, 'end': 1}
        assert tokens[1] == {'type': 'number', 'value': 0,
                             'start': 1, 'end': 2}
        assert tokens[2] == {'type': 'rbracket', 'value': ']',
                             'start': 2, 'end': 3}

    def test_lbracket_at_end(self):
        tokens = tokenize('[')
        assert tokens[0]['type'] == 'lbracket'
        assert tokens[1]['type'] == 'eof'


class TestNumbers:
    def test_single_digit(self):
        tokens = tokenize('0')
        assert tokens[0] == {'type': 'number', 'value': 0,
                             'start': 0, 'end': 1}

    def test_multi_digit(self):
        tokens = tokenize('12345')
        assert tokens[0] == {'type': 'number', 'value': 12345,
                             'start': 0, 'end': 5}

    def test_leading_zeros(self):
        tokens = tokenize('007')
        assert tokens[0]['value'] == 7
        assert tokens[0]['end'] == 3

    def test_number_between_simple(self):
        toks = types_and_values('[1,2]')
        assert toks == [
            ('lbracket', '['),
            ('number', 1),
            ('comma', ','),
            ('number', 2),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_negative_number(self):
        tokens = tokenize('-5')
        assert tokens[0] == {'type': 'number', 'value': -5,
                             'start': 0, 'end': 2}

    def test_negative_multi_digit(self):
        tokens = tokenize('-123')
        assert tokens[0]['value'] == -123
        assert tokens[0]['end'] == 4


class TestQuotedIdentifier:
    def test_simple(self):
        tokens = tokenize('"foo"')
        assert tokens[0] == {'type': 'quoted_identifier', 'value': 'foo',
                             'start': 0, 'end': 5}
        assert tokens[1]['type'] == 'eof'

    def test_with_escapes(self):
        tokens = tokenize('"foo\\nbar"')
        assert tokens[0]['type'] == 'quoted_identifier'
        assert tokens[0]['value'] == 'foo\nbar'

    def test_unicode_escape(self):
        tokens = tokenize('"\\u00e9"')
        assert tokens[0]['value'] == '\u00e9'

    def test_unclosed(self):
        with pytest.raises(LexerError):
            tokenize('"foo')

    def test_invalid_escape(self):
        with pytest.raises(LexerError):
            tokenize('"\\q"')


class TestRawStringLiteral:
    def test_simple(self):
        tokens = tokenize("'foo'")
        assert tokens[0] == {'type': 'literal', 'value': 'foo',
                             'start': 0, 'end': 5}
        assert tokens[1]['type'] == 'eof'

    def test_escaped_quote(self):
        tokens = tokenize("'foo\\'bar'")
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == "foo'bar"

    def test_unclosed(self):
        with pytest.raises(LexerError):
            tokenize("'foo")

    def test_empty_raw_string(self):
        tokens = tokenize("''")
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == ''


class TestLiteral:
    def test_json_string(self):
        tokens = tokenize('`"foo"`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_json_number(self):
        tokens = tokenize('`42`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 42

    def test_json_object(self):
        tokens = tokenize('`{"a": 1}`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == {'a': 1}

    def test_json_array(self):
        tokens = tokenize('`[1, 2]`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == [1, 2]

    def test_escaped_backtick(self):
        tokens = tokenize('`"a\\`b"`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'a`b'

    def test_unclosed(self):
        with pytest.raises(LexerError):
            tokenize('`"foo"')

    def test_deprecated_string_literal(self):
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`foo`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_deprecated_leads_whitespace_strip(self):
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`  foo`')
        assert tokens[0]['value'] == 'foo'

    def test_bad_literal(self):
        with pytest.raises(LexerError):
            tokenize('`{invalid}`')


class TestOperators:
    def test_or(self):
        tokens = tokenize('||')
        assert tokens[0] == {'type': 'or', 'value': '||',
                             'start': 0, 'end': 1}

    def test_pipe(self):
        tokens = tokenize('|')
        assert tokens[0] == {'type': 'pipe', 'value': '|',
                             'start': 0, 'end': 0}

    def test_and(self):
        tokens = tokenize('&&')
        assert tokens[0] == {'type': 'and', 'value': '&&',
                             'start': 0, 'end': 1}

    def test_expref(self):
        tokens = tokenize('&')
        assert tokens[0] == {'type': 'expref', 'value': '&',
                             'start': 0, 'end': 0}

    def test_lte(self):
        tokens = tokenize('<=')
        assert tokens[0] == {'type': 'lte', 'value': '<=',
                             'start': 0, 'end': 1}

    def test_lt(self):
        tokens = tokenize('<')
        assert tokens[0] == {'type': 'lt', 'value': '<',
                             'start': 0, 'end': 0}

    def test_gte(self):
        tokens = tokenize('>=')
        assert tokens[0] == {'type': 'gte', 'value': '>=',
                             'start': 0, 'end': 1}

    def test_gt(self):
        tokens = tokenize('>')
        assert tokens[0] == {'type': 'gt', 'value': '>',
                             'start': 0, 'end': 0}

    def test_ne(self):
        tokens = tokenize('!=')
        assert tokens[0] == {'type': 'ne', 'value': '!=',
                             'start': 0, 'end': 1}

    def test_not(self):
        tokens = tokenize('!')
        assert tokens[0] == {'type': 'not', 'value': '!',
                             'start': 0, 'end': 0}

    def test_eq(self):
        tokens = tokenize('==')
        assert tokens[0] == {'type': 'eq', 'value': '==',
                             'start': 0, 'end': 1}

    def test_single_equals_raises_at_eof(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize('=')
        assert exc_info.value.lexer_position == 0
        assert exc_info.value.lexer_value == '='

    def test_single_equals_raises_in_middle(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize('=a')
        # position rewound to the '=' location
        assert exc_info.value.lexer_position == 0
        assert exc_info.value.lexer_value == '='

    def test_equals_followed_by_space(self):
        with pytest.raises(LexerError):
            tokenize('= ')


class TestUnknownTokens:
    def test_lone_minus(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize('-')
        assert exc_info.value.lexer_position == 0
        assert exc_info.value.lexer_value == '-'

    @pytest.mark.parametrize('char', ['#', '$', '%', '^', '~', ';', '?'])
    def test_unknown_char(self, char):
        with pytest.raises(LexerError) as exc_info:
            tokenize(char)
        assert exc_info.value.lexer_value == char
        assert exc_info.value.lexer_position == 0


class TestEOF:
    def test_single_char_eof(self):
        tokens = tokenize('a')
        assert tokens[-1] == {'type': 'eof', 'value': '',
                              'start': 1, 'end': 1}

    def test_eof_position(self):
        tokens = tokenize('foo.bar')
        assert tokens[-1]['start'] == 7
        assert tokens[-1]['end'] == 7

    def test_eof_includes_length(self):
        tokens = tokenize('abc')
        assert tokens[-1]['start'] == 3
        assert tokens[-1]['end'] == 3


class TestComplex:
    def test_full_expression(self):
        toks = types_and_values('foo[?bar > `1`].baz')
        assert toks == [
            ('unquoted_identifier', 'foo'),
            ('filter', '[?'),
            ('unquoted_identifier', 'bar'),
            ('gt', '>'),
            ('literal', 1),
            ('rbracket', ']'),
            ('dot', '.'),
            ('unquoted_identifier', 'baz'),
            ('eof', ''),
        ]

    def test_nested_expression(self):
        toks = types_and_values('a.b.c')
        assert toks == [
            ('unquoted_identifier', 'a'),
            ('dot', '.'),
            ('unquoted_identifier', 'b'),
            ('dot', '.'),
            ('unquoted_identifier', 'c'),
            ('eof', ''),
        ]

    def test_star_and_current(self):
        toks = types_and_values('@.*')
        assert toks == [
            ('current', '@'),
            ('dot', '.'),
            ('star', '*'),
            ('eof', ''),
        ]

    def test_lparen_rparen(self):
        toks = types_and_values('(a)')
        assert toks == [
            ('lparen', '('),
            ('unquoted_identifier', 'a'),
            ('rparen', ')'),
            ('eof', ''),
        ]
