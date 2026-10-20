import pytest

import jmespath
from jmespath import ast
from jmespath import exceptions
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParserPublicAPI:
    def test_parse_returns_parsed_result(self):
        result = Parser().parse('foo')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo'

    def test_parsed_result_search(self):
        result = Parser().parse('foo.bar')
        assert result.search({'foo': {'bar': 'baz'}}) == 'baz'

    def test_parsed_result_search_with_options(self):
        result = Parser().parse('foo | bar')
        assert result.search({'foo': {'bar': 1}}, options=None) == 1

    def test_parsed_result_repr(self):
        result = Parser().parse('foo')
        assert repr(result) == repr(result.parsed)

    def test_parsed_result_attributes(self):
        result = Parser().parse('foo')
        assert result.expression == 'foo'
        assert result.parsed == ast.field('foo')

    def test_parsed_result_render_dot_file(self):
        result = Parser().parse('foo.bar')
        contents = result._render_dot_file()
        assert isinstance(contents, str)
        assert 'digraph' in contents


class TestCaching:
    def test_cache_hit_returns_same_object(self):
        p = Parser()
        first = p.parse('foo')
        second = p.parse('foo')
        assert first is second

    def test_cache_shared_across_instances(self):
        first = Parser().parse('foo')
        second = Parser().parse('foo')
        assert first is second

    def test_purge_clears_cache(self):
        first = Parser().parse('foo')
        Parser.purge()
        second = Parser().parse('foo')
        assert first is not second

    def test_cache_max_size_does_not_grow_unbounded(self):
        parser = Parser()
        for i in range(Parser._MAX_SIZE + 10):
            parser.parse('foo%d' % i)
        assert len(Parser._CACHE) <= Parser._MAX_SIZE

    def test_cache_cached_expression_found(self):
        parser = Parser()
        parser.parse('foo')
        assert 'foo' in Parser._CACHE

    def test_cache_evicts_but_parsed_result_returned(self):
        # Pre-populate the cache to the max, then parse a new expression.
        # The eviction succeeds (oldest key deleted), so the new expression
        # is added and the returned result must be a valid ParsedResult.
        for i in range(Parser._MAX_SIZE):
            Parser._CACHE['__dummy_%d' % i] = 'placeholder'
        parser = Parser()
        result = parser.parse('new_expr_xyz')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'new_expr_xyz'
        assert 'new_expr_xyz' in Parser._CACHE
        assert len(Parser._CACHE) == Parser._MAX_SIZE

    def test_cache_eviction_failure_returns_result_not_cached(self):
        # Force the eviction path to fail by monkeypatching `next` behavior
        # indirectly: use a dict subclass whose __iter__ raises RuntimeError,
        # simulating the concurrent-modification case.  The parse must still
        # succeed and return the ParsedResult, but must not add to the cache.
        original_cache = Parser._CACHE

        class _RaisingDict(dict):
            def __iter__(self):
                raise RuntimeError('simulated concurrent modification')

        # Fill so len >= _MAX_SIZE triggers eviction attempt
        for i in range(Parser._MAX_SIZE):
            original_cache['__fill_%d' % i] = 'placeholder'

        class _Fake(dict):
            pass

        # Replace the cache with our raising dict, preserving contents.
        raising = _RaisingDict()
        for k, v in original_cache.items():
            dict.__setitem__(raising, k, v)
        Parser._CACHE = raising
        try:
            parser = Parser()
            result = parser.parse('another_expr_abc')
            assert isinstance(result, ParsedResult)
            assert result.expression == 'another_expr_abc'
            # Not cached because eviction failed.
            assert 'another_expr_abc' not in dict.keys(raising)
            # Size unchanged.
            assert dict.__len__(raising) == Parser._MAX_SIZE
        finally:
            Parser._CACHE = original_cache
            Parser.purge()


class TestBasicExpressions:
    def test_field(self):
        assert Parser().parse('foo').parsed == ast.field('foo')

    def test_dotted_field(self):
        parsed = Parser().parse('foo.bar').parsed
        assert parsed == ast.subexpression([ast.field('foo'), ast.field('bar')])

    def test_multi_level_dotted_field_flattens(self):
        parsed = Parser().parse('foo.bar.baz').parsed
        assert parsed['type'] == 'subexpression'
        assert len(parsed['children']) == 3

    def test_index(self):
        parsed = Parser().parse('foo[0]').parsed
        assert parsed['type'] == 'index_expression'

    def test_quoted_identifier(self):
        parsed = Parser().parse('"foo"').parsed
        assert parsed == ast.field('foo')

    def test_literal(self):
        parsed = Parser().parse('`5`').parsed
        assert parsed == ast.literal(5)

    def test_current(self):
        parsed = Parser().parse('@').parsed
        assert parsed == ast.current_node()

    def test_wildcard(self):
        parsed = Parser().parse('*').parsed
        assert parsed['type'] == 'value_projection'


class TestOperators:
    def test_pipe(self):
        parsed = Parser().parse('foo | bar').parsed
        assert parsed == ast.pipe(ast.field('foo'), ast.field('bar'))

    def test_or(self):
        parsed = Parser().parse('foo || bar').parsed
        assert parsed['type'] == 'or_expression'

    def test_and(self):
        parsed = Parser().parse('foo && bar').parsed
        assert parsed['type'] == 'and_expression'

    @pytest.mark.parametrize('expr,op', [
        ('foo == bar', 'eq'),
        ('foo != bar', 'ne'),
        ('foo > bar', 'gt'),
        ('foo >= bar', 'gte'),
        ('foo < bar', 'lt'),
        ('foo <= bar', 'lte'),
    ])
    def test_comparators(self, expr, op):
        parsed = Parser().parse(expr).parsed
        assert parsed == ast.comparator(op, ast.field('foo'), ast.field('bar'))

    def test_not(self):
        parsed = Parser().parse('!foo').parsed
        assert parsed['type'] == 'not_expression'

    def test_flatten(self):
        parsed = Parser().parse('foo[]').parsed
        assert parsed['type'] == 'projection'

    def test_lparen_nud(self):
        parsed = Parser().parse('(foo)').parsed
        assert parsed == ast.field('foo')


class TestFunctions:
    def test_no_args(self):
        parsed = Parser().parse('foo()').parsed
        assert parsed == ast.function_expression('foo', [])

    def test_one_arg(self):
        parsed = Parser().parse('foo(bar)').parsed
        assert parsed == ast.function_expression('foo', [ast.field('bar')])

    def test_multiple_args(self):
        parsed = Parser().parse('foo(bar, baz)').parsed
        assert parsed == ast.function_expression(
            'foo', [ast.field('bar'), ast.field('baz')])

    def test_lparen_after_lparen_treated_as_function(self):
        parsed = Parser().parse('(foo)(bar)').parsed
        assert parsed == ast.function_expression('foo', [ast.field('bar')])

    def test_quoted_identifier_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('"foo"(bar)')
        assert 'Quoted identifier not allowed for function names' in str(exc.value)

    def test_function_name_from_subexpression(self):
        parsed = Parser().parse('foo.bar(baz)').parsed
        assert parsed['type'] == 'subexpression'
        assert parsed['children'][0] == ast.field('foo')
        assert parsed['children'][1] == ast.function_expression(
            'bar', [ast.field('baz')])

    def test_function_name_from_non_field_left(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('[a, b](c)')
        assert 'Invalid function name' in str(exc.value)


class TestMultiSelect:
    def test_multi_select_list(self):
        parsed = Parser().parse('[foo, bar]').parsed
        assert parsed == ast.multi_select_list([ast.field('foo'), ast.field('bar')])

    def test_multi_select_list_after_dot(self):
        parsed = Parser().parse('foo.[bar, baz]').parsed
        assert parsed['type'] == 'subexpression'
        assert parsed['children'][1]['type'] == 'multi_select_list'

    def test_multi_select_hash(self):
        parsed = Parser().parse('{a: foo, b: bar}').parsed
        assert parsed['type'] == 'multi_select_dict'
        assert len(parsed['children']) == 2

    def test_multi_select_hash_after_dot(self):
        parsed = Parser().parse('foo.{a: bar}').parsed
        assert parsed['type'] == 'subexpression'
        assert parsed['children'][1]['type'] == 'multi_select_dict'


class TestSlices:
    def test_simple_slice(self):
        parsed = Parser().parse('foo[0:2]').parsed
        assert parsed['type'] == 'projection'
        assert parsed['children'][0]['type'] == 'index_expression'

    def test_step_slice(self):
        parsed = Parser().parse('foo[0:2:1]').parsed
        assert parsed['type'] == 'projection'

    def test_slice_missing_parts(self):
        parsed = Parser().parse('foo[::2]').parsed
        assert parsed['children'][0]['children'][1] == ast.slice(None, None, 2)

    def test_slice_too_many_colons(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo[0:1:2:3]')
        assert 'syntax error' in str(exc.value)

    def test_index_expression_chained(self):
        parsed = Parser().parse('foo[0][1]').parsed
        assert parsed['type'] == 'index_expression'
        assert len(parsed['children']) == 3

    def test_star_after_bracket(self):
        parsed = Parser().parse('foo[*]').parsed
        assert parsed['type'] == 'projection'

    def test_slice_just_colon(self):
        parsed = Parser().parse('foo[:]').parsed
        assert parsed['type'] == 'projection'
        assert parsed['children'][0]['type'] == 'index_expression'
        assert parsed['children'][0]['children'][1] == ast.slice(None, None, None)

    def test_slice_with_start_only(self):
        # [1:] - first token is a number so _parse_index_expression called
        parsed = Parser().parse('foo[1:]').parsed
        assert parsed['type'] == 'projection'
        assert parsed['children'][0]['children'][1] == ast.slice(1, None, None)

    def test_nud_lbracket_index(self):
        # nud path: leading lbracket with a number -> index expression
        parsed = Parser().parse('[0]').parsed
        assert parsed['type'] == 'index_expression'
        assert parsed['children'][0] == ast.identity()
        assert parsed['children'][1] == ast.index(0)

    def test_nud_lbracket_slice(self):
        # nud path with colon -> _parse_index_expression -> slice +
        # _project_if_slice wraps into projection.
        parsed = Parser().parse('[:2]').parsed
        assert parsed['type'] == 'projection'
        assert parsed['children'][0]['type'] == 'index_expression'
        assert parsed['children'][0]['children'][1] == ast.slice(None, 2, None)


class TestProjectionRhs:
    def test_projection_rhs_lbracket(self):
        # Right after a projection, `[` should be parsed as rhs via
        # _parse_projection_rhs -> lbracket branch.
        parsed = Parser().parse('foo[*][0]').parsed
        assert parsed['type'] == 'projection'
        # The rhs should not be identity; there's an index expression.
        rhs = parsed['children'][1]
        assert rhs['type'] == 'index_expression'

    def test_projection_rhs_filter(self):
        # Right after a projection, `[?` should be parsed via filter branch.
        parsed = Parser().parse('foo[*][?bar]').parsed
        assert parsed['type'] == 'projection'
        rhs = parsed['children'][1]
        assert rhs['type'] == 'filter_projection'

    def test_projection_rhs_dot(self):
        # Right after a projection, `.bar` parsed via dot branch.
        parsed = Parser().parse('foo[*].bar').parsed
        assert parsed['type'] == 'projection'
        rhs = parsed['children'][1]
        assert rhs == ast.field('bar')

    def test_projection_rhs_non_compliant(self):
        # After projection, a token that can't start a rhs and has high
        # BP but isn't lbracket/filter/dot should raise.  In practice
        # "foo[*] bar": the star projection returns an identity rhs
        # because 'bar' has binding power 0 < _PROJECTION_STOP, so the
        # projection parse ends, and then the trailing 'bar' becomes an
        # unexpected token at the top level.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo[*] bar')
        msg = str(exc.value)
        assert 'Unexpected token' in msg

    def test_no_led_for_token(self):
        # Reaching a token with nonzero BP but no led function raises.
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo bar')


class TestFilters:
    def test_filter(self):
        parsed = Parser().parse('foo[?bar]').parsed
        assert parsed['type'] == 'filter_projection'

    def test_filter_nud(self):
        parsed = Parser().parse('[?bar]').parsed
        assert parsed['type'] == 'filter_projection'

    def test_filter_with_flatten(self):
        parsed = Parser().parse('foo[?bar][]').parsed
        assert parsed['children'][1] == ast.identity()


class TestProjections:
    def test_star_projection_nud(self):
        parsed = Parser().parse('*').parsed
        assert parsed == ast.value_projection(ast.identity(), ast.identity())

    def test_star_projection_led(self):
        parsed = Parser().parse('foo.*').parsed
        assert parsed['type'] == 'value_projection'

    def test_flatten_nud(self):
        parsed = Parser().parse('[]').parsed
        assert parsed['type'] == 'projection'

    def test_star_in_bracket_nud(self):
        parsed = Parser().parse('[*]').parsed
        assert parsed == ast.projection(ast.identity(), ast.identity())

    def test_star_in_bracket_led(self):
        parsed = Parser().parse('foo[*].bar').parsed
        assert parsed['type'] == 'projection'

    def test_star_nud_rbracket(self):
        # Exercises _token_nud_star's `if self._current_token() == 'rbracket'`
        # branch.  Inside a multi-select list starting with `*` followed by
        # `]`, the star nud sees rbracket next.
        parsed = Parser().parse('[*]').parsed
        # star nud at index 1 sees rbracket -> right is identity
        assert parsed == ast.projection(ast.identity(), ast.identity())


class TestExpref:
    def test_expref(self):
        parsed = Parser().parse('foo(&bar)').parsed
        arg = parsed['children'][0]
        assert arg['type'] == 'expref'


class TestErrors:
    def test_incomplete_expression(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc:
            Parser().parse('foo |')
        assert exc.value.expression == 'foo |'

    def test_invalid_token_parse_error(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('foo bar')

    def test_expression_attribute_on_error(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo bar')
        assert exc.value.expression == 'foo bar'

    def test_lexer_error_has_expression(self):
        with pytest.raises(exceptions.LexerError) as exc:
            Parser().parse('foo `bar')
        assert exc.value.expression == 'foo `bar'

    def test_invalid_dot_rhs(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo.1')
        assert 'Expecting' in str(exc.value)

    def test_missing_rparen(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('foo(bar')

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo]')
        assert 'Unexpected token' in str(exc.value)

    def test_multiselect_hash_missing_colon(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{a foo}')

    def test_multiselect_hash_invalid_key(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('{1: foo}')

    def test_multiselect_hash_missing_rbrace(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse('{a: foo')

    def test_slice_invalid_token(self):
        # _parse_slice_expression raises syntax error token when it hits
        # something that is neither colon, number, nor rbracket.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('foo[0:bar]')
        assert 'syntax error' in str(exc.value)


class TestParserInternalHelpers:
    def test_parser_init_lookahead(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert len(p._tokens) == 3
        assert p._index == 0

    def test_do_parse_directly(self):
        p = Parser()
        result = p._do_parse('foo')
        assert isinstance(result, ParsedResult)
        assert result.expression == 'foo'

    def test_led_not_available(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse('1 2')


class TestJMESPathIntegration:
    def test_module_level_search(self):
        assert jmespath.search('foo.bar', {'foo': {'bar': 42}}) == 42

    def test_module_level_compile(self):
        result = jmespath.compile('foo')
        assert isinstance(result, ParsedResult)
        assert result.search({'foo': 'x'}) == 'x'
