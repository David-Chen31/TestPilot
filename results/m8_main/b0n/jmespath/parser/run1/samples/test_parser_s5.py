import pytest

from jmespath import exceptions
from jmespath import ast
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParserInit:
    def test_default_lookahead(self):
        p = Parser()
        assert p._buffer_size == 2
        assert p._tokens == [None, None]
        assert p._index == 0
        assert p.tokenizer is None

    def test_custom_lookahead(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert p._tokens == [None, None, None]


class TestParserParse:
    def test_parse_simple_field(self):
        result = Parser().parse('foo')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo'
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo'}

    def test_parse_returns_parsed_result(self):
        result = Parser().parse('foo.bar')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo.bar'

    def test_parse_caches_result(self):
        p = Parser()
        r1 = p.parse('foo')
        r2 = p.parse('foo')
        assert r1 is r2
        assert 'foo' in Parser._CACHE

    def test_parse_cache_shared_across_instances(self):
        r1 = Parser().parse('foo')
        r2 = Parser().parse('foo')
        assert r1 is r2

    def test_purge_clears_cache(self):
        Parser().parse('foo')
        assert 'foo' in Parser._CACHE
        Parser.purge()
        assert Parser._CACHE == {}

    def test_purge_classmethod(self):
        Parser().parse('foo')
        Parser.purge()
        assert len(Parser._CACHE) == 0

    def test_cache_eviction_when_full(self):
        p = Parser()
        for i in range(Parser._MAX_SIZE):
            p.parse('field%d' % i)
        assert len(Parser._CACHE) == Parser._MAX_SIZE
        # Add another, should evict one
        p.parse('extra_field')
        assert len(Parser._CACHE) == Parser._MAX_SIZE
        assert 'field0' not in Parser._CACHE
        assert 'extra_field' in Parser._CACHE

    def test_cache_eviction_does_not_grow_unbounded(self):
        p = Parser()
        for i in range(Parser._MAX_SIZE + 10):
            p.parse('field%d' % i)
        assert len(Parser._CACHE) == Parser._MAX_SIZE


class TestParserErrors:
    def test_incomplete_expression(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc_info:
            Parser().parse('')
        assert exc_info.value.expression == ''

    def test_incomplete_expression_sets_expression(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc_info:
            Parser().parse('foo.')
        assert exc_info.value.expression == 'foo.'

    def test_parse_error_sets_expression(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo[')
        assert exc_info.value.expression == 'foo['

    def test_lexer_error_sets_expression(self):
        with pytest.raises(exceptions.LexerError) as exc_info:
            Parser().parse('foo"bar"')
        assert exc_info.value.expression == 'foo"bar"'

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo bar')
        assert 'Unexpected token' in str(exc_info.value)

    def test_quoted_identifier_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('"foo"(@)')
        assert 'Quoted identifier not allowed for function names' in str(
            exc_info.value)

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo[0]()')
        assert 'Invalid function name' in str(exc_info.value)

    def test_dot_rhs_error(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo.@')
        assert 'Expecting:' in str(exc_info.value)


class TestParsedResult:
    def test_search(self):
        result = Parser().parse('foo')
        assert result.search({'foo': 'bar'}) == 'bar'

    def test_search_options(self):
        result = Parser().parse('length(@)')
        assert result.search('hello') == 5

    def test_repr(self):
        result = Parser().parse('foo')
        assert repr(result) == repr(result.parsed)

    def test_render_dot_file(self):
        result = Parser().parse('foo.bar')
        dot = result._render_dot_file()
        assert dot.startswith('digraph')
        assert '->' in dot

    def test_attributes(self):
        result = ParsedResult('expr', {'type': 'identity', 'children': []})
        assert result.expression == 'expr'
        assert result.parsed == {'type': 'identity', 'children': []}


class TestParsingConstructs:
    def test_literal(self):
        result = Parser().parse('`{"a": 1}`')
        assert result.parsed['type'] == 'literal'
        assert result.parsed['value'] == {'a': 1}

    def test_quoted_identifier(self):
        result = Parser().parse('"foo"')
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo'}

    def test_star(self):
        result = Parser().parse('*')
        assert result.parsed['type'] == 'value_projection'

    def test_star_with_rbracket(self):
        result = Parser().parse('[*]')
        assert result.parsed['type'] == 'projection'

    def test_filter_nud(self):
        result = Parser().parse('[?a > `1`]')
        assert result.parsed['type'] == 'filter_projection'

    def test_multi_select_hash(self):
        result = Parser().parse('{a: b, c: d}')
        assert result.parsed['type'] == 'multi_select_dict'
        assert len(result.parsed['children']) == 2

    def test_multi_select_list(self):
        result = Parser().parse('[a, b, c]')
        assert result.parsed['type'] == 'multi_select_list'
        assert len(result.parsed['children']) == 3

    def test_lparen(self):
        result = Parser().parse('(foo)')
        assert result.parsed == {'type': 'field', 'children': [], 'value': 'foo'}

    def test_flatten_nud(self):
        result = Parser().parse('[]')
        assert result.parsed['type'] == 'projection'

    def test_not(self):
        result = Parser().parse('!foo')
        assert result.parsed['type'] == 'not_expression'

    def test_index_expression_nud(self):
        result = Parser().parse('[0]')
        assert result.parsed['type'] == 'index_expression'

    def test_slice_nud(self):
        result = Parser().parse('[0:1]')
        assert result.parsed['type'] == 'projection'

    def test_current_node(self):
        result = Parser().parse('@')
        assert result.parsed == {'type': 'current', 'children': []}

    def test_expref(self):
        result = Parser().parse('sort_by(@, &foo)')
        assert result.parsed['type'] == 'function_expression'

    def test_dot_subexpression_merge(self):
        result = Parser().parse('a.b.c')
        assert result.parsed['type'] == 'subexpression'
        assert len(result.parsed['children']) == 3

    def test_pipe(self):
        result = Parser().parse('a | b')
        assert result.parsed['type'] == 'pipe'

    def test_or(self):
        result = Parser().parse('a || b')
        assert result.parsed['type'] == 'or_expression'

    def test_and(self):
        result = Parser().parse('a && b')
        assert result.parsed['type'] == 'and_expression'

    def test_function_expression(self):
        result = Parser().parse('length(@)')
        assert result.parsed['type'] == 'function_expression'
        assert result.parsed['value'] == 'length'

    def test_comparators(self):
        for op, name in [('==', 'eq'), ('!=', 'ne'), ('>', 'gt'),
                         ('>=', 'gte'), ('<', 'lt'), ('<=', 'lte')]:
            Parser.purge()
            result = Parser().parse('a %s b' % op)
            assert result.parsed['type'] == 'comparator'
            assert result.parsed['value'] == name

    def test_flatten_led(self):
        result = Parser().parse('a[]')
        assert result.parsed['type'] == 'projection'

    def test_lbracket_led_index(self):
        result = Parser().parse('a[0]')
        assert result.parsed['type'] == 'index_expression'

    def test_lbracket_led_index_chain(self):
        result = Parser().parse('a[0][1]')
        assert result.parsed['type'] == 'index_expression'
        assert len(result.parsed['children']) == 3

    def test_lbracket_led_slice(self):
        result = Parser().parse('a[0:1]')
        assert result.parsed['type'] == 'projection'

    def test_lbracket_led_star(self):
        result = Parser().parse('a[*]')
        assert result.parsed['type'] == 'projection'

    def test_dot_star(self):
        result = Parser().parse('a.*')
        assert result.parsed['type'] == 'value_projection'

    def test_led_filter(self):
        result = Parser().parse('a[?b > `1`]')
        assert result.parsed['type'] == 'filter_projection'

    def test_led_filter_then_flatten(self):
        result = Parser().parse('a[?b > `1`][]')
        assert result.parsed['type'] == 'projection'

    def test_dot_rhs_lbracket(self):
        result = Parser().parse('a.[b, c]')
        assert result.parsed['type'] == 'subexpression'

    def test_dot_rhs_lbrace(self):
        result = Parser().parse('a.{b: c}')
        assert result.parsed['type'] == 'subexpression'

    def test_multi_select_hash_quoted_key(self):
        result = Parser().parse('{"foo": bar}')
        assert result.parsed['type'] == 'multi_select_dict'

    def test_multi_select_hash_trailing_comma_stops(self):
        result = Parser().parse('{a: b}')
        assert result.parsed['type'] == 'multi_select_dict'


class TestSliceParsing:
    def test_slice_step_too_many(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('[0:1:2:3]')
        assert 'syntax error' in str(exc_info.value)

    def test_slice_invalid_token(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('[a:b]')
        assert 'syntax error' in str(exc_info.value)

    def test_slice_missing_rbracket(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('[0:1')


class TestErrorNudToken:
    def test_error_nud_eof(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('.')

    def test_error_nud_invalid(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse(']')
        assert 'invalid token' in str(exc_info.value)


class TestInternalHelpers:
    def test_project_if_slice_non_slice(self):
        p = Parser()
        left = ast.field('a')
        right = ast.index(0)
        result = p._project_if_slice(left, right)
        assert result['type'] == 'index_expression'

    def test_project_if_slice_slice(self):
        p = Parser()
        p.tokenizer = None
        p._tokens = [{'type': 'eof', 'value': '', 'start': 0}]
        p._index = 0
        left = ast.field('a')
        right = ast.slice(0, 1, None)
        result = p._project_if_slice(left, right)
        assert result['type'] == 'projection'

    def test_project_if_slice_slice_stops_projection(self):
        p = Parser()
        p._tokens = [
            {'type': 'eof', 'value': '', 'start': 0},
        ]
        p._index = 0
        left = ast.field('a')
        right = ast.slice(0, 1, None)
        result = p._project_if_slice(left, right)
        assert result['children'][1] == {'type': 'identity', 'children': []}
