import pytest

from jmespath import ast
from jmespath import exceptions
from jmespath import parser
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


# ---------------------------------------------------------------------------
# Parser public API
# ---------------------------------------------------------------------------

class TestParserParse:
    def test_parse_returns_parsed_result(self):
        result = Parser().parse('foo')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo'
        assert result.parsed == ast.field('foo')

    def test_parse_simple_identifier(self):
        result = Parser().parse('foo')
        assert result.parsed['type'] == 'field'
        assert result.parsed['value'] == 'foo'

    def test_parse_literal(self):
        result = Parser().parse('`42`')
        assert result.parsed == ast.literal(42)

    def test_parse_subexpression(self):
        result = Parser().parse('foo.bar')
        assert result.parsed['type'] == 'subexpression'
        assert len(result.parsed['children']) == 2
        assert result.parsed['children'][0] == ast.field('foo')
        assert result.parsed['children'][1] == ast.field('bar')

    def test_parse_index(self):
        result = Parser().parse('foo[0]')
        assert result.parsed['type'] == 'index_expression'

    def test_parse_slice(self):
        result = Parser().parse('foo[0:10:2]')
        assert result.parsed['type'] == 'projection'

    def test_parse_projection(self):
        result = Parser().parse('foo[*].bar')
        assert result.parsed['type'] == 'projection'

    def test_parse_filter(self):
        result = Parser().parse('foo[?bar > `1`]')
        assert result.parsed['type'] == 'filter_projection'

    def test_parse_multi_select_list(self):
        result = Parser().parse('foo.[bar, baz]')
        children = result.parsed['children']
        assert children[1]['type'] == 'multi_select_list'

    def test_parse_multi_select_hash(self):
        result = Parser().parse('{foo: bar, baz: qux}')
        assert result.parsed['type'] == 'multi_select_dict'
        nodes = result.parsed['children']
        assert len(nodes) == 2
        assert nodes[0]['type'] == 'key_val_pair'
        assert nodes[0]['children'][0] == 'foo'

    def test_parse_function_expression(self):
        result = Parser().parse('length(foo)')
        assert result.parsed['type'] == 'function_expression'
        assert result.parsed['value'] == 'length'
        assert result.parsed['children'] == [ast.field('foo')]

    def test_parse_or_expression(self):
        result = Parser().parse('foo || bar')
        assert result.parsed['type'] == 'or_expression'

    def test_parse_and_expression(self):
        result = Parser().parse('foo && bar')
        assert result.parsed['type'] == 'and_expression'

    def test_parse_not_expression(self):
        result = Parser().parse('!foo')
        assert result.parsed['type'] == 'not_expression'

    def test_parse_pipe(self):
        result = Parser().parse('foo | bar')
        assert result.parsed['type'] == 'pipe'

    def test_parse_comparator_eq(self):
        result = Parser().parse('foo == bar')
        assert result.parsed['type'] == 'comparator'
        assert result.parsed['value'] == 'eq'

    def test_parse_comparator_ne(self):
        assert Parser().parse('foo != bar').parsed['value'] == 'ne'

    def test_parse_comparator_lt(self):
        assert Parser().parse('foo < bar').parsed['value'] == 'lt'

    def test_parse_comparator_lte(self):
        assert Parser().parse('foo <= bar').parsed['value'] == 'lte'

    def test_parse_comparator_gt(self):
        assert Parser().parse('foo > bar').parsed['value'] == 'gt'

    def test_parse_comparator_gte(self):
        assert Parser().parse('foo >= bar').parsed['value'] == 'gte'

    def test_parse_current_node(self):
        result = Parser().parse('@')
        assert result.parsed == ast.current_node()

    def test_parse_expref(self):
        result = Parser().parse('sort_by(foo, &bar)')
        func = result.parsed
        assert func['type'] == 'function_expression'
        assert func['children'][1]['type'] == 'expref'

    def test_parse_quoted_identifier(self):
        result = Parser().parse('"foo"')
        assert result.parsed == ast.field('foo')

    def test_parse_star_nud(self):
        result = Parser().parse('*')
        assert result.parsed['type'] == 'value_projection'

    def test_parse_flatten_nud(self):
        result = Parser().parse('[]')
        assert result.parsed['type'] == 'projection'

    def test_parse_slice_start_only(self):
        result = Parser().parse('[1:]')
        assert result.parsed['type'] == 'projection'

    def test_parse_slice_step_only(self):
        result = Parser().parse('[::2]')
        assert result.parsed['type'] == 'projection'


class TestParserCache:
    def test_parse_caches_result(self):
        p = Parser()
        first = p.parse('foo')
        second = p.parse('foo')
        assert first is second

    def test_purge_clears_cache(self):
        p = Parser()
        first = p.parse('foo')
        Parser.purge()
        second = p.parse('foo')
        assert first is not second

    def test_cache_max_size(self):
        p = Parser()
        original_max = Parser._MAX_SIZE
        try:
            Parser._MAX_SIZE = 3
            Parser.purge()
            for expr in ['a', 'b', 'c', 'd']:
                p.parse(expr)
            assert len(Parser._CACHE) <= 3
            assert 'a' not in Parser._CACHE
            assert 'd' in Parser._CACHE
        finally:
            Parser._MAX_SIZE = original_max


class TestParserErrors:
    def test_empty_expression_raises_incomplete(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('')

    def test_unclosed_bracket_raises_incomplete(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('foo[')

    def test_invalid_token_raises_parse_error(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo bar')

    def test_quoted_identifier_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('"foo"(bar)')
        assert 'Quoted identifier not allowed' in str(exc_info.value)

    def test_invalid_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo.bar(baz)')
        assert "Invalid function name" in str(exc_info.value)

    def test_bad_dot_rhs(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo.1')

    def test_lexer_error_gets_expression(self):
        with pytest.raises(exceptions.LexerError) as exc_info:
            Parser().parse('`foo')
        assert exc_info.value.expression == '`foo'

    def test_parse_error_gets_expression(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo bar')
        assert exc_info.value.expression == 'foo bar'

    def test_incomplete_error_gets_expression(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc_info:
            Parser().parse('foo.')
        assert exc_info.value.expression == 'foo.'

    def test_bad_multi_select_hash_missing_colon(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{foo bar}')

    def test_bad_multi_select_hash_non_identifier_key(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{1: bar}')


# ---------------------------------------------------------------------------
# ParsedResult
# ---------------------------------------------------------------------------

class TestParsedResult:
    def test_attributes(self):
        result = ParsedResult('foo', ast.field('foo'))
        assert result.expression == 'foo'
        assert result.parsed == ast.field('foo')

    def test_search(self):
        result = Parser().parse('foo.bar')
        assert result.search({'foo': {'bar': 'baz'}}) == 'baz'

    def test_search_with_options(self):
        result = Parser().parse('foo')
        assert result.search({'foo': 'bar'}, options=None) == 'bar'

    def test_search_missing_key_returns_none(self):
        result = Parser().parse('missing')
        assert result.search({'foo': 'bar'}) is None

    def test_search_function(self):
        result = Parser().parse('length(foo)')
        assert result.search({'foo': [1, 2, 3]}) == 3

    def test_repr(self):
        result = ParsedResult('foo', ast.field('foo'))
        assert repr(result) == repr(result.parsed)

    def test_render_dot_file(self):
        result = Parser().parse('foo')
        contents = result._render_dot_file()
        assert isinstance(contents, str)
        assert 'field' in contents


# ---------------------------------------------------------------------------
# Misc parser internals exercised through public API
# ---------------------------------------------------------------------------

class TestParserBindingPowers:
    def test_binding_power_table_present(self):
        assert Parser.BINDING_POWER['eof'] == 0
        assert Parser.BINDING_POWER['pipe'] == 1
        assert Parser.BINDING_POWER['star'] == 20
        assert Parser.BINDING_POWER['lparen'] == 60
        assert Parser.BINDING_POWER['flatten'] == 9

    def test_projection_stop_value(self):
        assert Parser._PROJECTION_STOP == 10

    def test_custom_lookahead_constructor(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert len(p._tokens) == 3
        result = p.parse('foo.bar')
        assert result.parsed['type'] == 'subexpression'

    def test_precedence_pipe_lower_than_or(self):
        result = Parser().parse('a || b | c')
        # pipe has lower bp so top node should be pipe
        assert result.parsed['type'] == 'pipe'
        assert result.parsed['children'][0]['type'] == 'or_expression'
