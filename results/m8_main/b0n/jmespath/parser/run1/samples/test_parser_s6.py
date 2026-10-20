import pytest

from jmespath import ast
from jmespath import exceptions
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParsedResult:
    def test_attributes(self):
        node = ast.field('foo')
        result = ParsedResult('foo', node)
        assert result.expression == 'foo'
        assert result.parsed is node

    def test_repr(self):
        node = ast.field('foo')
        result = ParsedResult('foo', node)
        assert repr(result) == repr(node)

    def test_search_simple(self):
        parsed = Parser().parse('foo')
        result = parsed.search({'foo': 'bar'})
        assert result == 'bar'

    def test_search_nested(self):
        parsed = Parser().parse('foo.bar')
        assert parsed.search({'foo': {'bar': 42}}) == 42

    def test_search_missing(self):
        parsed = Parser().parse('foo')
        assert parsed.search({}) is None

    def test_search_options(self):
        parsed = Parser().parse('foo.bar')
        result = parsed.search({'foo': {'bar': 1}},
                               options={'dict_cls': dict})
        assert result == 1

    def test_render_dot_file(self):
        parsed = Parser().parse('foo')
        contents = parsed._render_dot_file()
        assert isinstance(contents, str)
        assert 'digraph' in contents


class TestParserParseBasics:
    def test_parse_returns_parsed_result(self):
        parsed = Parser().parse('foo')
        assert isinstance(parsed, ParsedResult)
        assert parsed.expression == 'foo'

    def test_parse_field(self):
        parsed = Parser().parse('foo')
        assert isinstance(parsed.parsed, ast.field)
        assert parsed.parsed['value'] == 'foo'

    def test_parse_quoted_identifier(self):
        parsed = Parser().parse('"foo bar"')
        assert isinstance(parsed.parsed, ast.field)
        assert parsed.parsed['value'] == 'foo bar'

    def test_parse_literal(self):
        parsed = Parser().parse('`42`')
        assert isinstance(parsed.parsed, ast.literal)
        assert parsed.parsed['value'] == 42

    def test_parse_subexpression(self):
        parsed = Parser().parse('foo.bar')
        assert parsed.parsed['type'] == 'subexpression'

    def test_parse_index(self):
        parsed = Parser().parse('foo[0]')
        assert parsed.parsed['type'] == 'index_expression'

    def test_parse_slice(self):
        parsed = Parser().parse('foo[0:1]')
        assert parsed.parsed['type'] == 'projection'

    def test_parse_pipe(self):
        parsed = Parser().parse('foo | bar')
        assert parsed.parsed['type'] == 'pipe'

    def test_parse_or(self):
        parsed = Parser().parse('foo || bar')
        assert parsed.parsed['type'] == 'or_expression'

    def test_parse_and(self):
        parsed = Parser().parse('foo && bar')
        assert parsed.parsed['type'] == 'and_expression'

    def test_parse_comparator(self):
        parsed = Parser().parse('foo == `1`')
        assert parsed.parsed['type'] == 'comparator'

    def test_parse_function(self):
        parsed = Parser().parse('length(foo)')
        assert parsed.parsed['type'] == 'function_expression'
        assert parsed.parsed['value'] == 'length'

    def test_parse_multi_select_list(self):
        parsed = Parser().parse('[foo, bar]')
        assert parsed.parsed['type'] == 'multi_select_list'

    def test_parse_multi_select_hash(self):
        parsed = Parser().parse('{foo: bar}')
        assert parsed.parsed['type'] == 'multi_select_dict'

    def test_parse_current_node(self):
        parsed = Parser().parse('@')
        assert parsed.parsed['type'] == 'current'

    def test_parse_expref(self):
        parsed = Parser().parse('sort_by(@, &foo)')
        assert parsed.parsed['type'] == 'function_expression'

    def test_parse_projection(self):
        parsed = Parser().parse('foo[*].bar')
        assert parsed.parsed['type'] == 'projection'

    def test_parse_filter(self):
        parsed = Parser().parse('foo[?bar == `1`]')
        assert parsed.parsed['type'] == 'filter_projection'

    def test_parse_flatten(self):
        parsed = Parser().parse('foo[]')
        assert parsed.parsed['type'] == 'projection'

    def test_parse_not(self):
        parsed = Parser().parse('!foo')
        assert parsed.parsed['type'] == 'not_expression'


class TestParserCaching:
    def test_cache_returns_same_object(self):
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
        original_max = Parser._MAX_SIZE
        try:
            Parser._MAX_SIZE = 2
            p = Parser()
            p.parse('foo')
            p.parse('bar')
            p.parse('baz')
            assert len(Parser._CACHE) <= 2
        finally:
            Parser._MAX_SIZE = original_max

    def test_parse_does_not_cache_on_error(self):
        p = Parser()
        with pytest.raises(exceptions.ParseError):
            p.parse('foo[')
        assert 'foo[' not in Parser._CACHE


class TestParserErrors:
    def test_incomplete_expression_eof(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc_info:
            Parser().parse('foo.')
        assert exc_info.value.expression == 'foo.'

    def test_empty_expression(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('')

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo bar')

    def test_invalid_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo.]')

    def test_quoted_identifier_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('"foo"(bar)')
        assert 'Quoted identifier not allowed' in str(exc_info.value)

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('`1`(bar)')
        assert 'Invalid function name' in str(exc_info.value)

    def test_parse_error_sets_expression(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('foo bar')
        assert exc_info.value.expression == 'foo bar'

    def test_lexer_error_sets_expression(self):
        with pytest.raises(exceptions.LexerError) as exc_info:
            Parser().parse('foo["bar]')
        assert exc_info.value.expression == 'foo["bar]'

    def test_incomplete_function_call(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('length(foo')

    def test_missing_rparen(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('length(foo bar)')

    def test_slice_syntax_error_four_colons(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo[0:1:2:3]')

    def test_slice_non_number(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo[0:a]')

    def test_dot_invalid_rhs(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo.`1`')

    def test_multi_select_hash_invalid_key(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{`1`: foo}')

    def test_multi_select_list_missing_comma(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('[foo bar]')

    def test_led_missing(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo}')


class TestParserInternals:
    def test_lookahead_default(self):
        p = Parser()
        assert p._buffer_size == 2
        assert p._tokens == [None, None]
        assert p._index == 0

    def test_lookahead_custom(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert p._tokens == [None, None, None]

    def test_parse_resets_index(self):
        p = Parser()
        p.parse('foo')
        p.parse('bar')
        assert p._index > 0
        # Should still be able to parse again correctly.
        result = p.parse('baz')
        assert result.expression == 'baz'


class TestParserProjectionBranches:
    def test_star_nud(self):
        parsed = Parser().parse('*')
        assert parsed.parsed['type'] == 'value_projection'

    def test_star_nud_with_rbracket(self):
        parsed = Parser().parse('[*]')
        assert parsed.parsed['type'] == 'projection'

    def test_filter_nud(self):
        parsed = Parser().parse('[?foo == `1`]')
        assert parsed.parsed['type'] == 'filter_projection'

    def test_flatten_nud(self):
        parsed = Parser().parse('[].foo')
        assert parsed.parsed['type'] == 'projection'

    def test_dot_star_projection(self):
        parsed = Parser().parse('foo.*')
        assert parsed.parsed['type'] == 'value_projection'

    def test_filter_led_with_flatten(self):
        parsed = Parser().parse('foo[?bar][]')
        assert parsed.parsed['type'] == 'projection'

    def test_index_expression_optimization(self):
        parsed = Parser().parse('foo[0][1]')
        assert parsed.parsed['type'] == 'index_expression'

    def test_lparen_nud(self):
        parsed = Parser().parse('(foo)')
        assert parsed.parsed['type'] == 'field'

    def test_lbrace_nud(self):
        parsed = Parser().parse('{foo: bar}')
        assert parsed.parsed['type'] == 'multi_select_dict'
</parameter>
