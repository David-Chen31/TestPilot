import pytest

from jmespath.lexer import Lexer
from jmespath.exceptions import LexerError, EmptyExpressionError


def tokenize(expression):
    return list(Lexer().tokenize(expression))


def types_and_values(expression):
    return [(t['type'], t['value']) for t in tokenize(expression)]


class TestSimpleTokens:
    @pytest.mark.parametrize("char,token_type", [
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
    def test_single_simple_token(self, char, token_type):
        tokens = tokenize(char)
        assert len(tokens) == 2
        tok = tokens[0]
        assert tok == {'type': token_type, 'value': char, 'start': 0, 'end': 1}
        assert tokens[1] == {'type': 'eof', 'value': '', 'start': 1, 'end': 1}

    def test_simple_tokens_sequence(self):
        tokens = tokenize('.@,')
        assert [(t['type'], t['value']) for t in tokens[:3]] == [
            ('dot', '.'), ('current', '@'), ('comma', ',')]
        assert tokens[0]['start'] == 0 and tokens[0]['end'] == 1
        assert tokens[1]['start'] == 1 and tokens[1]['end'] == 2
        assert tokens[2]['start'] == 2 and tokens[2]['end'] == 3


class TestIdentifiers:
    def test_simple_identifier(self):
        tokens = tokenize('foo')
        assert tokens[0] == {'type': 'unquoted_identifier', 'value': 'foo',
                             'start': 0, 'end': 3}
        assert tokens[1]['type'] == 'eof'

    def test_identifier_with_digits_and_underscore(self):
        tokens = tokenize('a_b1c2')
        assert tokens[0] == {'type': 'unquoted_identifier',
                             'value': 'a_b1c2', 'start': 0, 'end': 6}

    def test_identifier_starting_with_underscore(self):
        tokens = tokenize('_x')
        assert tokens[0]['type'] == 'unquoted_identifier'
        assert tokens[0]['value'] == '_x'

    def test_identifier_followed_by_simple_token(self):
        tokens = tokenize('foo.bar')
        assert types_and_values('foo.bar') == [
            ('unquoted_identifier', 'foo'),
            ('dot', '.'),
            ('unquoted_identifier', 'bar'),
            ('eof', ''),
        ]

    def test_identifier_positions(self):
        tokens = tokenize('abc.def')
        assert tokens[0]['start'] == 0 and tokens[0]['end'] == 3
        assert tokens[1]['start'] == 3 and tokens[1]['end'] == 4
        assert tokens[2]['start'] == 4 and tokens[2]['end'] == 7


class TestWhitespace:
    def test_whitespace_is_skipped(self):
        assert types_and_values('  foo  ') == [
            ('unquoted_identifier', 'foo'),
            ('eof', ''),
        ]

    def test_all_whitespace_chars(self):
        assert types_and_values(' \t\n\rfoo') == [
            ('unquoted_identifier', 'foo'),
            ('eof', ''),
        ]

    def test_whitespace_only(self):
        tokens = tokenize('   ')
        assert len(tokens) == 1
        assert tokens[0]['type'] == 'eof'
        assert tokens[0]['start'] == 3 and tokens[0]['end'] == 3


class TestBrackets:
    def test_lbracket_number_rbracket(self):
        assert types_and_values('[0]') == [
            ('lbracket', '['),
            ('number', 0),
            ('rbracket', ']'),
            ('eof', ''),
        ]

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

    def test_lbracket_positions(self):
        tokens = tokenize('foo[0]')
        lbracket = tokens[1]
        assert lbracket == {'type': 'lbracket', 'value': '[',
                            'start': 3, 'end': 4}


class TestNumbers:
    def test_single_digit(self):
        tokens = tokenize('0')
        assert tokens[0] == {'type': 'number', 'value': 0,
                             'start': 0, 'end': 1}

    def test_multi_digit(self):
        tokens = tokenize('12345')
        assert tokens[0] == {'type': 'number', 'value': 12345,
                             'start': 0, 'end': 5}

    def test_number_with_leading_zeros(self):
        tokens = tokenize('007')
        assert tokens[0]['value'] == 7
        assert tokens[0]['end'] == 3

    def test_number_followed_by_token(self):
        assert types_and_values('1.2') == [
            ('number', 1),
            ('dot', '.'),
            ('number', 2),
            ('eof', ''),
        ]

    def test_negative_number(self):
        tokens = tokenize('-5')
        assert tokens[0] == {'type': 'number', 'value': -5,
                             'start': 0, 'end': 2}

    def test_negative_multi_digit_number(self):
        tokens = tokenize('-123')
        assert tokens[0] == {'type': 'number', 'value': -123,
                             'start': 0, 'end': 4}

    def test_lone_minus_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('-')
        assert exc.value.lexer_position == 0
        assert exc.value.lexer_value == '-'
        assert exc.value.message == "Unknown token '-'"

    def test_minus_before_non_number_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('-a')
        assert exc.value.lexer_position == 0
        assert exc.value.lexer_value == '-'


class TestRawStringLiterals:
    def test_simple_raw_string(self):
        tokens = tokenize("'foo'")
        # 'end' is offset from start (not absolute), so token_len == 4.
        assert tokens[0] == {'type': 'literal', 'value': 'foo',
                             'start': 0, 'end': 4}

    def test_raw_string_with_spaces(self):
        tokens = tokenize("'hello world'")
        assert tokens[0]['value'] == 'hello world'

    def test_escaped_quote_in_raw_string(self):
        tokens = tokenize("'a\\'b'")
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == "a'b"

    def test_empty_raw_string(self):
        tokens = tokenize("''")
        # token_len == position - start == 1 (position is on closing quote).
        assert tokens[0] == {'type': 'literal', 'value': '',
                             'start': 0, 'end': 1}

    def test_unclosed_raw_string_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize("'foo")
        assert exc.value.lexer_position == 0
        assert 'Unclosed' in exc.value.message


class TestPipesAndLogical:
    def test_pipe(self):
        assert types_and_values('a|b') == [
            ('unquoted_identifier', 'a'),
            ('pipe', '|'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]

    def test_or(self):
        assert types_and_values('a||b') == [
            ('unquoted_identifier', 'a'),
            ('or', '||'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]

    def test_expref(self):
        tokens = tokenize('&foo')
        assert tokens[0] == {'type': 'expref', 'value': '&',
                             'start': 0, 'end': 0}
        assert tokens[1]['value'] == 'foo'

    def test_and(self):
        assert types_and_values('a&&b') == [
            ('unquoted_identifier', 'a'),
            ('and', '&&'),
            ('unquoted_identifier', 'b'),
            ('eof', ''),
        ]


class TestComparisons:
    @pytest.mark.parametrize("expr,token_type,value", [
        ('<', 'lt', '<'),
        ('<=', 'lte', '<='),
        ('>', 'gt', '>'),
        ('>=', 'gte', '>='),
        ('!', 'not', '!'),
        ('!=', 'ne', '!='),
    ])
    def test_comparison_tokens(self, expr, token_type, value):
        tokens = tokenize(expr)
        assert tokens[0]['type'] == token_type
        assert tokens[0]['value'] == value

    def test_lt_position(self):
        tokens = tokenize('<')
        assert tokens[0]['start'] == 0
        # Single-char non-matching branch yields end == start.
        assert tokens[0]['end'] == 0

    def test_lte_position(self):
        tokens = tokenize('<=')
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 1

    def test_gt_single_position(self):
        tokens = tokenize('>')
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 0

    def test_not_single_position(self):
        tokens = tokenize('!')
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 0


class TestEquality:
    def test_eq(self):
        tokens = tokenize('==')
        assert tokens[0] == {'type': 'eq', 'value': '==',
                             'start': 0, 'end': 1}
        assert tokens[1]['type'] == 'eof'

    def test_lone_equals_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('=')
        assert exc.value.lexer_value == '='
        assert exc.value.message == "Unknown token '='"

    def test_equals_at_eof_position(self):
        with pytest.raises(LexerError) as exc:
            tokenize('=')
        # At EOF we never advanced; position should be 0.
        assert exc.value.lexer_position == 0

    def test_equals_before_non_equals_position(self):
        with pytest.raises(LexerError) as exc:
            tokenize('=a')
        assert exc.value.lexer_position == 0


class TestQuotedIdentifiers:
    def test_simple_quoted_identifier(self):
        tokens = tokenize('"foo"')
        assert tokens[0]['type'] == 'quoted_identifier'
        assert tokens[0]['value'] == 'foo'
        assert tokens[0]['start'] == 0

    def test_quoted_identifier_with_spaces(self):
        tokens = tokenize('"foo bar"')
        assert tokens[0]['value'] == 'foo bar'

    def test_empty_quoted_identifier(self):
        tokens = tokenize('""')
        assert tokens[0]['value'] == ''

    def test_unclosed_quoted_identifier_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('"foo')
        assert 'Unclosed' in exc.value.message

    def test_invalid_quoted_identifier_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('"foo\\x"')
        assert exc.value.lexer_position == 0


class TestLiterals:
    def test_integer_literal(self):
        tokens = tokenize('`1`')
        # end is token_len (relative), not absolute position: 3 - 1 == 2.
        assert tokens[0] == {'type': 'literal', 'value': 1,
                             'start': 0, 'end': 2}

    def test_string_literal(self):
        tokens = tokenize('`"foo"`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_list_literal(self):
        tokens = tokenize('`[1, 2]`')
        assert tokens[0]['value'] == [1, 2]

    def test_dict_literal(self):
        tokens = tokenize('`{"a": 1}`')
        assert tokens[0]['value'] == {'a': 1}

    def test_boolean_literal(self):
        tokens = tokenize('`true`')
        assert tokens[0]['value'] is True

    def test_null_literal(self):
        tokens = tokenize('`null`')
        assert tokens[0]['value'] is None

    def test_unclosed_literal_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('`1')
        assert 'Unclosed' in exc.value.message

    def test_escaped_backtick(self):
        tokens = tokenize('`"a\\`b"`')
        assert tokens[0]['value'] == 'a`b'

    def test_deprecated_bare_string_literal_warns(self):
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`foo`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == 'foo'

    def test_bad_literal_does_not_raise(self):
        # '`{bad`' falls back to JSON-string parsing which succeeds,
        # producing a deprecated bare-string literal rather than an error.
        with pytest.warns(PendingDeprecationWarning):
            tokens = tokenize('`{bad`')
        assert tokens[0]['type'] == 'literal'
        assert tokens[0]['value'] == '{bad'

    def test_truly_bad_literal_raises(self):
        # Unterminated string so JSON-string fallback also fails.
        # Construct a lexeme that is invalid JSON and also invalid when
        # wrapped in double quotes.  A tab character is invalid inside a
        # JSON string, so the fallback ``loads('"...\t..."')`` fails and
        # the lexer raises a LexerError with a "Bad token" message.
        with pytest.raises(LexerError) as exc:
            tokenize("`a\tb`")
        assert 'Bad token' in exc.value.message


class TestErrors:
    def test_empty_expression_raises(self):
        with pytest.raises(EmptyExpressionError):
            tokenize('')

    def test_none_expression_raises(self):
        with pytest.raises(EmptyExpressionError):
            tokenize(None)

    def test_unknown_char_raises(self):
        with pytest.raises(LexerError) as exc:
            tokenize('#')
        assert exc.value.lexer_position == 0
        assert exc.value.lexer_value == '#'
        assert exc.value.message == 'Unknown token #'

    def test_unknown_char_position(self):
        with pytest.raises(LexerError) as exc:
            tokenize('foo#')
        assert exc.value.lexer_position == 3
        assert exc.value.lexer_value == '#'


class TestEof:
    def test_eof_token(self):
        tokens = tokenize('foo')
        eof = tokens[-1]
        assert eof == {'type': 'eof', 'value': '', 'start': 3, 'end': 3}

    def test_eof_position_multi_char(self):
        tokens = tokenize('foobar')
        assert tokens[-1]['start'] == 6
        assert tokens[-1]['end'] == 6


class TestComplexExpressions:
    def test_index_expression(self):
        assert types_and_values('foo[1]') == [
            ('unquoted_identifier', 'foo'),
            ('lbracket', '['),
            ('number', 1),
            ('rbracket', ']'),
            ('eof', ''),
        ]

    def test_filter_expression(self):
        result = types_and_values('foo[?bar > `1`]')
        assert result == [
            ('unquoted_identifier', 'foo'),
            ('filter', '[?'),
            ('unquoted_identifier', 'bar'),
            ('gt', '>'),
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

    def test_multiselect_dict(self):
        assert types_and_values('{a: b}') == [
            ('lbrace', '{'),
            ('unquoted_identifier', 'a'),
            ('colon', ':'),
            ('unquoted_identifier', 'b'),
            ('rbrace', '}'),
            ('eof', ''),
        ]

    def test_flatten_then_index(self):
        assert types_and_values('foo[][0]') == [
            ('unquoted_identifier', 'foo'),
            ('flatten', '[]'),
            ('lbracket', '['),
            ('number', 0),
            ('rbracket', ']'),
            ('eof', ''),
        ]


class TestTokenPositions:
    def test_positions_are_contiguous(self):
        tokens = tokenize('foo.bar')
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 3
        assert tokens[1]['start'] == 3
        assert tokens[1]['end'] == 4
        assert tokens[2]['start'] == 4
        assert tokens[2]['end'] == 7

    def test_identifier_end_across_whitespace(self):
        tokens = tokenize('foo bar')
        assert tokens[0]['start'] == 0
        assert tokens[0]['end'] == 3
        assert tokens[1]['start'] == 4
        assert tokens[1]['end'] == 7
