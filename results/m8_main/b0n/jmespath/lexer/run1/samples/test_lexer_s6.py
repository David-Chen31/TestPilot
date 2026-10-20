import string
import warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


def types_and_values(expression):
    return [(t['type'], t['value']) for t in tokenize(expression)]


class TestEmptyAndEof:
    def test_empty_expression_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(''))

    def test_none_expression_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(None))

    def test_single_char_eof_positions(self):
        tokens = tokenize('a')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a', 'start': 0, 'end': 1},
            {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
        ]

    def test_eof_start_end_match_length(self):
        tokens = tokenize('foo bar')
        eof = tokens[-1]
        assert eof['type'] == 'eof'
        assert eof['value'] == ''
        assert eof['start'] == len('foo bar')
        assert eof['end'] == len('foo bar')


class TestSimpleTokens:
    def test_all_simple_tokens(self):
        expected = {
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
        for char, token_type in expected.items():
            tokens = tokenize(char)
            assert tokens[0] == {
                'type': token_type,
                'value': char,
                'start': 0,
                'end': 1,
            }
            assert tokens[1]['type'] == 'eof'

    def test_simple_token_positions(self):
        tokens = tokenize('a.b')
        assert tokens[0] == {
            'type': 'unquoted_identifier', 'value': 'a',
            'start': 0, 'end': 1,
        }
        assert tokens[1] == {
            'type': 'dot', 'value': '.', 'start': 1, 'end': 2,
        }
        assert tokens[2] == {
            'type': 'unquoted_identifier', 'value': 'b',
            'start': 2, 'end': 3,
        }


class TestIdentifiers:
    def test_identifier_letters(self):
        assert types_and_values('foo') == [
            ('unquoted_identifier', 'foo'), ('eof', ''),
        ]

    def test_identifier_with_underscore_and_digits(self):
        assert types_and_values('foo_bar123') == [
            ('unquoted_identifier', 'foo_bar123'), ('eof', ''),
        ]

    def test_identifier_starting_with_underscore(self):
        assert types_and_values('_foo') == [
            ('unquoted_identifier', '_foo'), ('eof', ''),
        ]

    def test_identifier_includes_uppercase(self):
        assert types_and_values('FooBar') == [
            ('unquoted_identifier', 'FooBar'), ('eof', ''),
        ]

    def test_identifier_stops_at_non_identifier_char(self):
        tokens = tokenize('foo.bar')
        assert tokens[0] == {
            'type': 'unquoted_identifier', 'value': 'foo',
            'start': 0, 'end': 3,
        }
        assert tokens[1]['type'] == 'dot'

    def test_whitespace_between_identifiers(self):
        assert types_and_values('a b') == [
            ('unquoted_identifier', 'a'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]

    def test_all_whitespace_variants(self):
        assert types_and_values("a \t\n\rb") == [
            ('unquoted_identifier', 'a'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]


class TestBrackets:
    def test_flatten(self):
        tokens = tokenize('[]')
        assert tokens[0] == {
            'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2,
        }
        assert tokens[1] == {'type': 'eof', 'value': '', 'start': 2, 'end': 2}

    def test_filter(self):
        tokens = tokenize('[?')
        assert tokens[0] == {
            'type': 'filter', 'value': '[?', 'start': 0, 'end': 2,
        }

    def test_lbracket_plain(self):
        tokens = tokenize('[a]')
        assert tokens[0] == {
            'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1,
        }
        assert tokens[1] == {
            'type': 'unquoted_identifier', 'value': 'a',
            'start': 1, 'end': 2,
        }
        assert tokens[2] == {
            'type': 'rbracket', 'value': ']', 'start': 2, 'end': 3,
        }

    def test_lbracket_at_eof(self):
        tokens = tokenize('[')
        assert tokens[0] == {
            'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1,
        }
        assert tokens[1] == {'type': 'eof', 'value': '', 'start': 1, 'end': 1}

    def test_flatten_follows_identifier(self):
        tokens = tokenize('foo[]')
        assert types_and_values('foo[]') == [
            ('unquoted_identifier', 'foo'), ('flatten', '[]'), ('eof', ''),
        ]
        assert tokens[1]['start'] == 3
        assert tokens[1]['end'] == 5


class TestNumbers:
    def test_positive_integer(self):
        tokens = tokenize('123')
        assert tokens[0] == {
            'type': 'number', 'value': 123, 'start': 0, 'end': 3,
        }

    def test_zero(self):
        assert types_and_values('0') == [('number', 0), ('eof', '')]

    def test_multi_digit(self):
        assert types_and_values('9876543210') == [
            ('number', 9876543210), ('eof', ''),
        ]

    def test_number_then_identifier(self):
        assert types_and_values('1a') == [
            ('number', 1), ('unquoted_identifier', 'a'), ('eof', ''),
        ]

    def test_negative_number(self):
        tokens = tokenize('-42')
        assert tokens[0] == {
            'type': 'number', 'value': -42, 'start': 0, 'end': 3,
        }

    def test_lone_minus_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('-')
        assert "Unknown token" in str(excinfo.value.message)
        assert excinfo.value.lexer_position == 0


class TestQuotedIdentifier:
    def test_basic(self):
        tokens = tokenize('"foo"')
        assert tokens[0] == {
            'type': 'quoted_identifier', 'value': 'foo',
            'start': 0, 'end': 5,
        }

    def test_with_spaces(self):
        assert types_and_values('"foo bar"') == [
            ('quoted_identifier', 'foo bar'), ('eof', ''),
        ]

    def test_escaped_quote(self):
        assert types_and_values('"foo\\"bar"') == [
            ('quoted_identifier', 'foo"bar'), ('eof', ''),
        ]

    def test_unclosed_raises(self):
        with pytest.raises(LexerError):
            tokenize('"foo')

    def test_invalid_json_escape_raises(self):
        with pytest.raises(LexerError):
            tokenize('"foo\\x"')


class TestRawStringLiteral:
    def test_basic(self):
        tokens = tokenize("'foo'")
        assert tokens[0] == {
            'type': 'literal', 'value': 'foo', 'start': 0, 'end': 5,
        }

    def test_escaped_quote(self):
        assert types_and_values("'foo\\'bar'") == [
            ('literal', "foo'bar"), ('eof', ''),
        ]

    def test_unclosed_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize("'foo")
        assert 'Unclosed' in excinfo.value.message


class TestLiteral:
    def test_json_number(self):
        assert types_and_values('`42`') == [('literal', 42), ('eof', '')]

    def test_json_string(self):
        assert types_and_values('`"foo"`') == [
            ('literal', 'foo'), ('eof', ''),
        ]

    def test_json_object(self):
        assert types_and_values('`{"a": 1}`') == [
            ('literal', {'a': 1}), ('eof', ''),
        ]

    def test_json_array(self):
        assert types_and_values('`[1, 2]`') == [
            ('literal', [1, 2]), ('eof', ''),
        ]

    def test_json_true(self):
        assert types_and_values('`true`') == [('literal', True), ('eof', '')]

    def test_literal_positions(self):
        tokens = tokenize('`42`')
        assert tokens[0]['start'] == 0
        # Implementation stores token_len as end (position - start).
        assert tokens[0]['end'] == 4

    def test_unclosed_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('`foo')
        assert 'Unclosed' in excinfo.value.message

    def test_escaped_backtick(self):
        # Escaping the backtick allows it to appear inside a literal.
        assert types_and_values('`"foo\\`bar"`') == [
            ('literal', 'foo`bar'), ('eof', ''),
        ]

    def test_deprecated_string_literal_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tokens = tokenize('`foo`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'
        assert any(
            issubclass(w.category, PendingDeprecationWarning) for w in caught
        )

    def test_bad_literal_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('`{"a"`')
        assert 'Bad token' in excinfo.value.message


class TestPipeAndOrAndExpref:
    def test_pipe(self):
        tokens = tokenize('a|b')
        assert tokens[1] == {
            'type': 'pipe', 'value': '|', 'start': 1, 'end': 2,
        }

    def test_or(self):
        tokens = tokenize('a||b')
        assert tokens[1] == {
            'type': 'or', 'value': '||', 'start': 1, 'end': 2,
        }

    def test_expref(self):
        tokens = tokenize('a&b')
        assert tokens[1] == {
            'type': 'expref', 'value': '&', 'start': 1, 'end': 2,
        }

    def test_and(self):
        tokens = tokenize('a&&b')
        assert tokens[1] == {
            'type': 'and', 'value': '&&', 'start': 1, 'end': 2,
        }


class TestComparators:
    def test_lt(self):
        tokens = tokenize('a<b')
        assert tokens[1] == {'type': 'lt', 'value': '<', 'start': 1, 'end': 1}

    def test_lte(self):
        tokens = tokenize('a<=b')
        assert tokens[1] == {
            'type': 'lte', 'value': '<=', 'start': 1, 'end': 2,
        }

    def test_gt(self):
        tokens = tokenize('a>b')
        assert tokens[1] == {'type': 'gt', 'value': '>', 'start': 1, 'end': 1}

    def test_gte(self):
        tokens = tokenize('a>=b')
        assert tokens[1] == {
            'type': 'gte', 'value': '>=', 'start': 1, 'end': 2,
        }

    def test_not(self):
        tokens = tokenize('!a')
        assert tokens[0] == {'type': 'not', 'value': '!', 'start': 0, 'end': 0}

    def test_ne(self):
        tokens = tokenize('a!=b')
        assert tokens[1] == {
            'type': 'ne', 'value': '!=', 'start': 1, 'end': 2,
        }

    def test_eq(self):
        tokens = tokenize('a==b')
        assert tokens[1] == {
            'type': 'eq', 'value': '==', 'start': 1, 'end': 2,
        }


class TestEqualsError:
    def test_single_equals_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('=')
        err = excinfo.value
        assert err.lexer_value == '='
        assert "Unknown token '='" == err.message
        assert err.lexer_position == 0

    def test_single_equals_in_middle_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('a=b')
        err = excinfo.value
        assert err.lexer_value == '='
        assert err.lexer_position == 1

    def test_single_equals_at_eof_in_middle(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('a=')
        err = excinfo.value
        # At EOF the position is not rewound.
        assert err.lexer_position == 2


class TestUnknownToken:
    def test_hash_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('#')
        err = excinfo.value
        assert err.lexer_position == 0
        assert err.lexer_value == '#'
        assert err.message == 'Unknown token #'

    def test_unknown_after_valid(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('foo#')
        err = excinfo.value
        assert err.lexer_position == 3
        assert err.lexer_value == '#'


class TestComplexExpressions:
    def test_field_access(self):
        assert types_and_values('foo.bar.baz') == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('unquoted_identifier', 'bar'),
            ('dot', '.'),
            ('unquoted_identifier', 'baz'),
            ('eof', ''),
        ]

    def test_index_expression(self):
        assert types_and_values('foo[0]') == [
            ('unquoted_identifier', 'foo'),
            ('lbracket', '['),
            ('number', 0),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_filter_expression(self):
        assert types_and_values('[?a == `1`]') == [
            ('filter', '[?'),
            ('unquoted_identifier', 'a'),
            ('eq', '=='),
            ('literal', 1),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_multiselect_list(self):
        assert types_and_values('[a, b]') == [
            ('lbracket', '['),
            ('unquoted_identifier', 'a'),
            ('comma', ','),
            ('unquoted_identifier', 'b'),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_multiselect_hash(self):
        assert types_and_values('{a: b}') == [
            ('lbrace', '{'),
            ('unquoted_identifier', 'a'),
            ('colon', ':'),
            ('unquoted_identifier', 'b'),
            ('rbrace', '}'),
            ('eof', ''),
        ]

    def test_parens(self):
        assert types_and_values('(a)') == [
            ('lparen', '('),
            ('unquoted_identifier', 'a'),
            ('rparen', ')'),
            ('eof', ''),
        ]

    def test_current_node(self):
        assert types_and_values('@') == [('current', '@'), ('eof', '')]

    def test_star(self):
        assert types_and_values('*') == [('star', '*'), ('eof', '')]


class TestInitializeStateReuse:
    def test_tokenize_resets_state(self):
        lexer = Lexer()
        first = list(lexer.tokenize('foo'))
        second = list(lexer.tokenize('bar'))
        assert first[0]['value'] == 'foo'
        assert second[0]['value'] == 'bar'
        assert second[0]['start'] == 0
        assert second[0]['end'] == 3
