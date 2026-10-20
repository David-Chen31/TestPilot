import string
import warnings as _warnings

import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


class TestLexerConstants:
    def test_start_identifier(self):
        assert Lexer.START_IDENTIFIER == set(string.ascii_letters + '_')

    def test_valid_identifier(self):
        assert Lexer.VALID_IDENTIFIER == set(
            string.ascii_letters + string.digits + '_')

    def test_valid_number(self):
        assert Lexer.VALID_NUMBER == set(string.digits)

    def test_whitespace(self):
        assert Lexer.WHITESPACE == set(" \t\n\r")

    def test_simple_tokens(self):
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


class TestEmptyExpression:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(''))

    def test_none_raises(self):
        with pytest.raises(EmptyExpressionError):
            list(Lexer().tokenize(None))


class TestSimpleTokens:
    @pytest.mark.parametrize("char,type_", [
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
        assert tokens == [
            {'type': type_, 'value': char, 'start': 0, 'end': 1},
            {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
        ]

    def test_multiple_simple_tokens(self):
        tokens = tokenize('.*')
        assert tokens == [
            {'type': 'dot', 'value': '.', 'start': 0, 'end': 1},
            {'type': 'star', 'value': '*', 'start': 1, 'end': 2},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]


class TestIdentifiers:
    def test_simple_identifier(self):
        tokens = tokenize('foo')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'foo',
             'start': 0, 'end': 3},
            {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
        ]

    def test_identifier_with_underscore_and_digits(self):
        tokens = tokenize('_foo123')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': '_foo123',
             'start': 0, 'end': 7},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]

    def test_identifier_starting_with_underscore(self):
        tokens = tokenize('_')
        assert tokens[0] == {'type': 'unquoted_identifier', 'value': '_',
                             'start': 0, 'end': 1}

    def test_two_identifiers_separated_by_dot(self):
        tokens = tokenize('foo.bar')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'foo',
             'start': 0, 'end': 3},
            {'type': 'dot', 'value': '.', 'start': 3, 'end': 4},
            {'type': 'unquoted_identifier', 'value': 'bar',
             'start': 4, 'end': 7},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]


class TestWhitespace:
    def test_leading_and_trailing_whitespace(self):
        tokens = tokenize('  foo  ')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'foo',
             'start': 2, 'end': 5},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]

    def test_all_whitespace_chars(self):
        tokens = tokenize(" \t\n\rfoo")
        assert tokens[0] == {'type': 'unquoted_identifier', 'value': 'foo',
                             'start': 4, 'end': 7}


class TestBrackets:
    def test_lbracket(self):
        tokens = tokenize('[0]')
        assert tokens == [
            {'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1},
            {'type': 'number', 'value': 0, 'start': 1, 'end': 2},
            {'type': 'rbracket', 'value': ']', 'start': 2, 'end': 3},
            {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
        ]

    def test_flatten(self):
        tokens = tokenize('[]')
        assert tokens == [
            {'type': 'flatten', 'value': '[]', 'start': 0, 'end': 2},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]

    def test_filter(self):
        tokens = tokenize('[?')
        assert tokens == [
            {'type': 'filter', 'value': '[?', 'start': 0, 'end': 2},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]

    def test_filter_full(self):
        tokens = tokenize('[?a]')
        assert tokens[0] == {'type': 'filter', 'value': '[?',
                             'start': 0, 'end': 2}
        assert tokens[1] == {'type': 'unquoted_identifier', 'value': 'a',
                             'start': 2, 'end': 3}
        assert tokens[2] == {'type': 'rbracket', 'value': ']',
                             'start': 3, 'end': 4}

    def test_lbracket_at_eof(self):
        tokens = tokenize('[')
        assert tokens == [
            {'type': 'lbracket', 'value': '[', 'start': 0, 'end': 1},
            {'type': 'eof', 'value': '', 'start': 1, 'end': 1},
        ]


class TestNumbers:
    def test_simple_number(self):
        tokens = tokenize('123')
        assert tokens == [
            {'type': 'number', 'value': 123, 'start': 0, 'end': 3},
            {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
        ]

    def test_negative_number(self):
        tokens = tokenize('-42')
        assert tokens == [
            {'type': 'number', 'value': -42, 'start': 0, 'end': 3},
            {'type': 'eof', 'value': '', 'start': 3, 'end': 3},
        ]

    def test_negative_single_digit_number(self):
        # Distinguishes len(buff) > 1 from len(buff) > 2: '-4' -> buff '-4'.
        tokens = tokenize('-4')
        assert tokens == [
            {'type': 'number', 'value': -4, 'start': 0, 'end': 2},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]

    def test_zero(self):
        tokens = tokenize('0')
        assert tokens[0] == {'type': 'number', 'value': 0,
                             'start': 0, 'end': 1}

    def test_number_mid_expression(self):
        # _consume_number uses start only via buff accumulation; ensure
        # correct token offsets for a number not starting at position 0.
        tokens = tokenize('a.123')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a',
             'start': 0, 'end': 1},
            {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
            {'type': 'number', 'value': 123, 'start': 2, 'end': 5},
            {'type': 'eof', 'value': '', 'start': 5, 'end': 5},
        ]

    def test_negative_number_mid_expression(self):
        tokens = tokenize('a[-42]')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a',
             'start': 0, 'end': 1},
            {'type': 'lbracket', 'value': '[', 'start': 1, 'end': 2},
            {'type': 'number', 'value': -42, 'start': 2, 'end': 5},
            {'type': 'rbracket', 'value': ']', 'start': 5, 'end': 6},
            {'type': 'eof', 'value': '', 'start': 6, 'end': 6},
        ]

    def test_lone_dash_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('-')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '-'
        assert excinfo.value.message == "Unknown token '-'"

    def test_dash_then_letter_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('-a')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '-'
        assert excinfo.value.message == "Unknown token '-'"


class TestOperators:
    def test_pipe(self):
        tokens = tokenize('|')
        assert tokens[0] == {'type': 'pipe', 'value': '|',
                             'start': 0, 'end': 0}

    def test_or(self):
        tokens = tokenize('||')
        assert tokens[0] == {'type': 'or', 'value': '||',
                             'start': 0, 'end': 1}

    def test_expref(self):
        tokens = tokenize('&')
        assert tokens[0] == {'type': 'expref', 'value': '&',
                             'start': 0, 'end': 0}

    def test_and(self):
        tokens = tokenize('&&')
        assert tokens[0] == {'type': 'and', 'value': '&&',
                             'start': 0, 'end': 1}

    def test_lt(self):
        tokens = tokenize('<')
        assert tokens[0] == {'type': 'lt', 'value': '<',
                             'start': 0, 'end': 0}

    def test_lte(self):
        tokens = tokenize('<=')
        assert tokens[0] == {'type': 'lte', 'value': '<=',
                             'start': 0, 'end': 1}

    def test_gt(self):
        tokens = tokenize('>')
        assert tokens[0] == {'type': 'gt', 'value': '>',
                             'start': 0, 'end': 0}

    def test_gte(self):
        tokens = tokenize('>=')
        assert tokens[0] == {'type': 'gte', 'value': '>=',
                             'start': 0, 'end': 1}

    def test_not(self):
        tokens = tokenize('!')
        assert tokens[0] == {'type': 'not', 'value': '!',
                             'start': 0, 'end': 0}

    def test_ne(self):
        tokens = tokenize('!=')
        assert tokens[0] == {'type': 'ne', 'value': '!=',
                             'start': 0, 'end': 1}


class TestEquals:
    def test_eq(self):
        tokens = tokenize('==')
        assert tokens == [
            {'type': 'eq', 'value': '==', 'start': 0, 'end': 1},
            {'type': 'eof', 'value': '', 'start': 2, 'end': 2},
        ]

    def test_eq_in_expression(self):
        tokens = tokenize('a==b')
        assert tokens[1] == {'type': 'eq', 'value': '==',
                             'start': 1, 'end': 2}

    def test_lone_equals_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('=')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '='
        assert excinfo.value.message == "Unknown token '='"

    def test_equals_followed_by_nonsense_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('=a')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '='
        assert excinfo.value.message == "Unknown token '='"


class TestQuotedIdentifiers:
    def test_simple_quoted_identifier(self):
        tokens = tokenize('"foo"')
        assert tokens == [
            {'type': 'quoted_identifier', 'value': 'foo',
             'start': 0, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 5, 'end': 5},
        ]

    def test_quoted_identifier_mid_expression(self):
        # The 'end' offset is computed as (position - start), i.e. relative
        # to start, not as an absolute position.
        tokens = tokenize('a."foo"')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a',
             'start': 0, 'end': 1},
            {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
            {'type': 'quoted_identifier', 'value': 'foo',
             'start': 2, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]

    def test_quoted_with_escape(self):
        tokens = tokenize('"fo\\no"')
        assert tokens[0]['type'] == 'quoted_identifier'
        assert tokens[0]['value'] == 'fo\no'

    def test_unclosed_quoted_identifier_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('"foo')
        assert excinfo.value.message == "Unclosed \" delimiter"
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '"foo'

    def test_unclosed_quoted_identifier_mid_expression(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('a."foo')
        # Start of the unterminated delimiter ('"') is at position 2.
        assert excinfo.value.lexer_position == 2
        assert excinfo.value.lexer_value == '"foo'
        assert excinfo.value.message == "Unclosed \" delimiter"

    def test_invalid_json_quoted_identifier_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('"\\x"')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '"\\x"'
        assert excinfo.value.message == "Invalid \\escape"


class TestRawStringLiterals:
    def test_simple_raw_string(self):
        tokens = tokenize("'foo'")
        assert tokens == [
            {'type': 'literal', 'value': 'foo', 'start': 0, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 5, 'end': 5},
        ]

    def test_raw_string_mid_expression(self):
        # The 'end' offset is relative to start (position - start).
        tokens = tokenize("a.'foo'")
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a',
             'start': 0, 'end': 1},
            {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
            {'type': 'literal', 'value': 'foo', 'start': 2, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]

    def test_escaped_single_quote(self):
        tokens = tokenize("'fo\\'o'")
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == "fo'o"

    def test_unclosed_raw_string_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize("'foo")
        assert excinfo.value.message == "Unclosed ' delimiter"
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == "'foo"

    def test_unclosed_raw_string_mid_expression(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize("a.'foo")
        assert excinfo.value.lexer_position == 2
        assert excinfo.value.lexer_value == "'foo"
        assert excinfo.value.message == "Unclosed ' delimiter"


class TestLiterals:
    def test_json_number_literal(self):
        tokens = tokenize('`123`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 123

    def test_json_number_literal_offsets(self):
        # The 'end' offset is (position - start), i.e. relative to start.
        tokens = tokenize('`123`')
        assert tokens == [
            {'type': 'literal', 'value': 123, 'start': 0, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 5, 'end': 5},
        ]

    def test_json_literal_mid_expression(self):
        # The 'end' offset is relative to start (position - start).
        tokens = tokenize('a.`123`')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'a',
             'start': 0, 'end': 1},
            {'type': 'dot', 'value': '.', 'start': 1, 'end': 2},
            {'type': 'literal', 'value': 123, 'start': 2, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 7, 'end': 7},
        ]

    def test_json_string_literal(self):
        tokens = tokenize('`"foo"`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_json_object_literal(self):
        tokens = tokenize('`{"a": 1}`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == {'a': 1}

    def test_json_array_literal(self):
        tokens = tokenize('`[1, 2]`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == [1, 2]

    def test_escaped_backtick(self):
        tokens = tokenize('`"a\\`b"`')
        assert tokens[0]['value'] == 'a`b'

    def test_deprecated_string_literal(self):
        # A bare (non-JSON) string falls into the deprecated path.
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`foo`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_deprecated_string_literal_offsets(self):
        # The 'end' offset is relative to start (position - start).
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`foo`')
        assert tokens == [
            {'type': 'literal', 'value': 'foo', 'start': 0, 'end': 4},
            {'type': 'eof', 'value': '', 'start': 5, 'end': 5},
        ]

    def test_deprecated_string_literal_warning_message(self):
        with pytest.warns(PendingDeprecationWarning) as record:
            tokenize('`foo`')
        assert len(record) == 1
        assert str(record[0].message) == "deprecated string literal syntax"

    def test_deprecated_string_literal_lstrip(self):
        # lstrip() means leading whitespace is trimmed but trailing isn't.
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`  foo  `')
        assert tokens[0]['value'] == 'foo  '

    def test_deprecated_string_literal_internal_whitespace(self):
        # ' foo ' -> lstrip -> 'foo ' (trailing space kept).
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('` foo `')
        assert tokens[0]['value'] == 'foo '

    def test_bad_literal_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('`a\x01b`')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '`a\x01b`'
        assert excinfo.value.message == "Bad token a\x01b"

    def test_bad_literal_mid_expression(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('a.`\x01`')
        assert excinfo.value.lexer_position == 2
        assert excinfo.value.lexer_value == '`\x01`'
        assert excinfo.value.message == "Bad token \x01"

    def test_unclosed_literal_raises(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('`foo')
        assert excinfo.value.message == "Unclosed ` delimiter"
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '`foo'

    def test_unclosed_literal_mid_expression(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('a.`foo')
        assert excinfo.value.lexer_position == 2
        assert excinfo.value.lexer_value == '`foo'
        assert excinfo.value.message == "Unclosed ` delimiter"


class TestUnknownTokens:
    def test_unknown_dollar(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('$')
        assert excinfo.value.lexer_position == 0
        assert excinfo.value.lexer_value == '$'
        assert excinfo.value.message == "Unknown token $"

    def test_unknown_hash_in_expression(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('foo#')
        assert excinfo.value.lexer_position == 3
        assert excinfo.value.lexer_value == '#'
        assert excinfo.value.message == "Unknown token #"

    def test_lexer_error_str(self):
        with pytest.raises(LexerError) as excinfo:
            tokenize('$')
        err = excinfo.value
        err.expression = '$'
        text = str(err)
        assert 'Bad jmespath expression:' in text
        assert '^' in text


class TestRealExpressions:
    def test_dotted_expression(self):
        tokens = tokenize('foo.bar[0]')
        assert tokens == [
            {'type': 'unquoted_identifier', 'value': 'foo',
             'start': 0, 'end': 3},
            {'type': 'dot', 'value': '.', 'start': 3, 'end': 4},
            {'type': 'unquoted_identifier', 'value': 'bar',
             'start': 4, 'end': 7},
            {'type': 'lbracket', 'value': '[', 'start': 7, 'end': 8},
            {'type': 'number', 'value': 0, 'start': 8, 'end': 9},
            {'type': 'rbracket', 'value': ']', 'start': 9, 'end': 10},
            {'type': 'eof', 'value': '', 'start': 10, 'end': 10},
        ]

    def test_filter_expression(self):
        tokens = tokenize('foo[?bar > `1`]')
        types = [t['type'] for t in tokens]
        assert types == [
            'unquoted_identifier', 'filter', 'unquoted_identifier',
            'gt', 'literal', 'rbracket', 'eof',
        ]

    def test_or_expression(self):
        tokens = tokenize('a || b')
        assert tokens[1] == {'type': 'or', 'value': '||',
                             'start': 2, 'end': 3}

    def test_always_ends_with_eof(self):
        tokens = tokenize('foo')
        assert tokens[-1] == {'type': 'eof', 'value': '',
                              'start': 3, 'end': 3}

    def test_token_positions_are_consistent(self):
        expression = 'foo.bar == baz'
        tokens = tokenize(expression)
        for token in tokens[:-1]:
            assert 0 <= token['start'] <= token['end']

    def test_warnings_are_not_emitted_for_valid_json(self):
        with _warnings.catch_warnings():
            _warnings.simplefilter('error')
            list(Lexer().tokenize('`123`'))
