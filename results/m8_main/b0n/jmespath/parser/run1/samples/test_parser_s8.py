import pytest

from jmespath import ast, exceptions
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParserBasicParsing:
    def test_parse_returns_parsed_result(self):
        result = Parser().parse('foo')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo'

    def test_parse_field(self):
        result = Parser().parse('foo')
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo'}

    def test_parse_quoted_identifier(self):
        result = Parser().parse('"foo bar"')
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo bar'}

    def test_parse_literal_number(self):
        result = Parser().parse('`123`')
        assert result.parsed == {'type': 'literal', 'children': [], 'value': 123}

    def test_parse_literal_string(self):
        result = Parser().parse('`"hello"`')
        assert result.parsed == {'type': 'literal', 'children': [], 'value': 'hello'}

    def test_parse_current_node(self):
        result = Parser().parse('@')
        assert result.parsed == {'type': 'current', 'children': []}

    def test_parse_index(self):
        result = Parser().parse('[0]')
        assert result.parsed['type'] == 'index_expression'

    def test_parse_slice(self):
        result = Parser().parse('[1:3]')
        assert result.parsed['type'] == 'projection'
        index_expr = result.parsed['children'][0]
        assert index_expr['children'][1] == {
            'type': 'slice', 'children': [],
            'value': {'start': 1, 'stop': 3, 'step': None},
        }

    def test_parse_slice_full(self):
        result = Parser().parse('[::2]')
        index_expr = result.parsed['children'][0]
        assert index_expr['children'][1]['value'] == {
            'start': None, 'stop': None, 'step': 2}

    def test_parse_subexpression(self):
        result = Parser().parse('foo.bar')
        assert result.parsed['type'] == 'subexpression'
        assert len(result.parsed['children']) == 2

    def test_parse_subexpression_append(self):
        result = Parser().parse('foo.bar.baz')
        assert result.parsed['type'] == 'subexpression'
        assert len(result.parsed['children']) == 3

    def test_parse_pipe(self):
        result = Parser().parse('foo | bar')
        assert result.parsed['type'] == 'pipe'

    def test_parse_or(self):
        result = Parser().parse('foo || bar')
        assert result.parsed['type'] == 'or_expression'

    def test_parse_and(self):
        result = Parser().parse('foo && bar')
        assert result.parsed['type'] == 'and_expression'

    def test_parse_not(self):
        result = Parser().parse('!foo')
        assert result.parsed['type'] == 'not_expression'

    def test_parse_comparators(self):
        for op, expr in [
            ('==', 'foo == bar'),
            ('!=', 'foo != bar'),
            ('<', 'foo < bar'),
            ('<=', 'foo <= bar'),
            ('>', 'foo > bar'),
            ('>=', 'foo >= bar'),
        ]:
            result = Parser().parse(expr)
            assert result.parsed == {
                'type': 'comparator',
                'children': [result.parsed['children'][0],
                             result.parsed['children'][1]],
                'value': op,
            }

    def test_parse_multi_select_list(self):
        result = Parser().parse('[foo, bar]')
        assert result.parsed['type'] == 'multi_select_list'
        assert len(result.parsed['children']) == 2

    def test_parse_multi_select_hash(self):
        result = Parser().parse('{a: foo, b: bar}')
        assert result.parsed['type'] == 'multi_select_dict'
        assert len(result.parsed['children']) == 2

    def test_parse_function(self):
        result = Parser().parse('length(foo)')
        assert result.parsed['type'] == 'function_expression'
        assert result.parsed['value'] == 'length'
        assert len(result.parsed['children']) == 1

    def test_parse_function_multiple_args(self):
        result = Parser().parse('contains(foo, bar)')
        assert result.parsed['value'] == 'contains'
        assert len(result.parsed['children']) == 2

    def test_parse_function_no_args(self):
        result = Parser().parse('now()')
        assert result.parsed['type'] == 'function_expression'
        assert result.parsed['children'] == []

    def test_parse_star_projection(self):
        result = Parser().parse('foo[*]')
        assert result.parsed['type'] == 'projection'

    def test_parse_star_nud(self):
        result = Parser().parse('*')
        assert result.parsed['type'] == 'value_projection'

    def test_parse_dot_star(self):
        result = Parser().parse('foo.*')
        assert result.parsed['type'] == 'value_projection'

    def test_parse_filter(self):
        result = Parser().parse('foo[?bar == `1`]')
        assert result.parsed['type'] == 'filter_projection'

    def test_parse_flatten(self):
        result = Parser().parse('foo[]')
        assert result.parsed['type'] == 'projection'
        assert result.parsed['children'][0]['type'] == 'flatten'

    def test_parse_parenthesized(self):
        result = Parser().parse('(foo)')
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo'}

    def test_parse_expref(self):
        result = Parser().parse('sort_by(foo, &bar)')
        assert result.parsed['value'] == 'sort_by'
        expref = result.parsed['children'][1]
        assert expref['type'] == 'expref'

    def test_parse_nested(self):
        result = Parser().parse('foo.bar[0].baz')
        assert isinstance(result.parsed, dict)


class TestParserCache:
    def test_cache_returns_same_object(self):
        parser = Parser()
        result1 = parser.parse('foo')
        result2 = parser.parse('foo')
        assert result1 is result2

    def test_cache_key_is_expression(self):
        Parser.purge()
        Parser().parse('cache_test_expr')
        assert 'cache_test_expr' in Parser._CACHE

    def test_purge_clears_cache(self):
        Parser().parse('purge_test')
        assert len(Parser._CACHE) == 1
        Parser.purge()
        assert Parser._CACHE == {}

    def test_cache_new_object_after_purge(self):
        parser = Parser()
        result1 = parser.parse('foo')
        Parser.purge()
        result2 = parser.parse('foo')
        assert result1 is not result2

    def test_cache_does_not_exceed_max_size(self):
        Parser.purge()
        parser = Parser()
        original_max = Parser._MAX_SIZE
        Parser._MAX_SIZE = 3
        try:
            for i in range(10):
                parser.parse('field_%d' % i)
            assert len(Parser._CACHE) <= 3
        finally:
            Parser._MAX_SIZE = original_max


class TestParserErrors:
    def test_invalid_token_nud(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('==')

    def test_incomplete_expression_empty(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('')

    def test_incomplete_expression_dot(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('foo.')

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo bar')

    def test_quoted_identifier_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('"foo"(bar)')
        assert 'Quoted identifier not allowed for function names' in str(exc.value)

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('`1`(foo)')
        assert "Invalid function name" in str(exc.value)

    def test_invalid_function_name_number(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('[0](foo)')
        assert "Invalid function name" in str(exc.value)

    def test_dot_rhs_error(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo.`1`')

    def test_dot_rhs_error_message(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo.`1`')
        assert 'Expecting:' in str(exc.value)

    def test_slice_syntax_error(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('[1:2:3:4]')

    def test_match_eof_raises_incomplete(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('foo[')

    def test_error_sets_expression_attr(self):
        try:
            Parser().parse('foo bar')
        except exceptions.ParseError as e:
            assert e.expression == 'foo bar'
        else:
            pytest.fail("Expected ParseError")

    def test_lexer_error_sets_expression(self):
        with pytest.raises(exceptions.LexerError) as exc:
            Parser().parse('"unterminated')
        assert exc.value.expression == '"unterminated'

    def test_error_led_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse(']')

    def test_lbrace_invalid_key(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{`1`: foo}')


class TestParsedResult:
    def test_repr(self):
        pr = ParsedResult('foo', {'type': 'field', 'children': [], 'value': 'foo'})
        assert repr(pr) == repr(pr.parsed)

    def test_expression_attribute(self):
        pr = ParsedResult('foo.bar', ast.field('foo'))
        assert pr.expression == 'foo.bar'

    def test_search_simple(self):
        pr = Parser().parse('foo')
        assert pr.search({'foo': 'bar'}) == 'bar'

    def test_search_missing_key(self):
        pr = Parser().parse('foo')
        assert pr.search({'baz': 'bar'}) is None

    def test_search_nested(self):
        pr = Parser().parse('foo.bar')
        assert pr.search({'foo': {'bar': 42}}) == 42

    def test_search_projections(self):
        pr = Parser().parse('foo[*].bar')
        assert pr.search({'foo': [{'bar': 1}, {'bar': 2}]}) == [1, 2]

    def test_search_with_options(self):
        from jmespath import visitor

        class FakeOptions:
            dict_cls = dict

        pr = Parser().parse('foo')
        # Sanity check it accepts an options argument.
        result = pr.search({'foo': 'bar'}, options=None)
        assert result == 'bar'

    def test_render_dot_file(self):
        pr = Parser().parse('foo')
        contents = pr._render_dot_file()
        assert isinstance(contents, str)
        assert 'foo' in contents


class TestParserLookaheadInit:
    def test_custom_lookahead(self):
        parser = Parser(lookahead=3)
        assert parser._buffer_size == 3
        result = parser.parse('foo.bar')
        assert result.parsed['type'] == 'subexpression'

    def test_reuse_parser(self):
        parser = Parser()
        r1 = parser.parse('foo')
        r2 = parser.parse('bar')
        assert r1.parsed['value'] == 'foo'
        assert r2.parsed['value'] == 'bar'

    def test_binding_power_values(self):
        assert Parser.BINDING_POWER['pipe'] == 1
        assert Parser.BINDING_POWER['or'] == 2
        assert Parser.BINDING_POWER['and'] == 3
        assert Parser.BINDING_POWER['eq'] == 5
        assert Parser.BINDING_POWER['flatten'] == 9
        assert Parser.BINDING_POWER['star'] == 20
        assert Parser.BINDING_POWER['dot'] == 40
        assert Parser.BINDING_POWER['lparen'] == 60
