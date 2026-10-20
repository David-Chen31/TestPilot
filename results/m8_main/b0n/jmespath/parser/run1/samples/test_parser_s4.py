import pytest

from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


# ---------------------------------------------------------------------------
# ParsedResult
# ---------------------------------------------------------------------------

def test_parsed_result_attributes():
    parsed = ast.field('foo')
    result = ParsedResult('foo', parsed)
    assert result.expression == 'foo'
    assert result.parsed is parsed


def test_parsed_result_repr_uses_parsed():
    parsed = ast.field('foo')
    result = ParsedResult('foo', parsed)
    assert repr(result) == repr(parsed)


def test_parsed_result_search_returns_result():
    result = ParsedResult('foo', ast.field('foo'))
    assert result.search({'foo': 'bar'}) == 'bar'


def test_parsed_result_search_missing_key_returns_none():
    result = ParsedResult('foo', ast.field('foo'))
    assert result.search({'baz': 1}) is None


def test_parsed_result_search_with_options():
    result = ParsedResult('foo', ast.field('foo'))
    assert result.search({'foo': 'bar'}, options=None) == 'bar'


def test_parsed_result_render_dot_file():
    result = ParsedResult('foo', ast.field('foo'))
    contents = result._render_dot_file()
    assert isinstance(contents, str)
    assert contents.startswith('digraph')


# ---------------------------------------------------------------------------
# Parser basic construction and cache behaviour
# ---------------------------------------------------------------------------

def test_parser_init_default_lookahead():
    p = Parser()
    assert p._buffer_size == 2
    assert p._tokens == [None, None]
    assert p._index == 0
    assert p.tokenizer is None


def test_parser_init_custom_lookahead():
    p = Parser(lookahead=3)
    assert p._buffer_size == 3
    assert p._tokens == [None, None, None]


def test_parse_caches_results():
    p = Parser()
    first = p.parse('foo')
    second = p.parse('foo')
    assert first is second
    assert 'foo' in Parser._CACHE


def test_parse_returns_parsed_result():
    p = Parser()
    result = p.parse('foo')
    assert isinstance(result, ParsedResult)
    assert result.expression == 'foo'
    assert isinstance(result.parsed, ast.field)
    assert result.parsed['value'] == 'foo'


def test_purge_clears_cache():
    p = Parser()
    p.parse('foo')
    assert len(Parser._CACHE) == 1
    Parser.purge()
    assert Parser._CACHE == {}


def test_cache_eviction_when_full(monkeypatch):
    p = Parser()
    monkeypatch.setattr(Parser, '_MAX_SIZE', 2)
    p.parse('a')
    p.parse('b')
    assert len(Parser._CACHE) == 2
    # Parsing 'a' again would hit cache, so add a new one to trigger eviction.
    p.parse('c')
    assert len(Parser._CACHE) <= 2
    assert 'c' in Parser._CACHE


def test_cache_eviction_returns_result_even_if_deletion_fails(monkeypatch):
    p = Parser()
    monkeypatch.setattr(Parser, '_MAX_SIZE', 0)

    class BadCache(dict):
        def __iter__(self):
            raise RuntimeError('boom')

    monkeypatch.setattr(Parser, '_CACHE', BadCache())
    result = p.parse('foo')
    assert isinstance(result, ParsedResult)
    # Because deletion failed, nothing was cached.
    assert len(Parser._CACHE) == 0


# ---------------------------------------------------------------------------
# Basic parsing -> AST checks
# ---------------------------------------------------------------------------

def test_parse_unquoted_identifier():
    result = Parser().parse('foo')
    assert isinstance(result.parsed, ast.field)
    assert result.parsed['value'] == 'foo'


def test_parse_quoted_identifier():
    result = Parser().parse('"foo bar"')
    assert isinstance(result.parsed, ast.field)
    assert result.parsed['value'] == 'foo bar'


def test_parse_literal():
    result = Parser().parse('`true`')
    assert isinstance(result.parsed, ast.literal)
    assert result.parsed['value'] is True


def test_parse_current_node():
    result = Parser().parse('@')
    assert isinstance(result.parsed, ast.current_node)


def test_parse_dot_create_subexpression():
    result = Parser().parse('foo.bar')
    assert result.parsed['type'] == 'subexpression'
    assert len(result.parsed['children']) == 2


def test_parse_dot_extends_existing_subexpression():
    result = Parser().parse('foo.bar.baz')
    assert result.parsed['type'] == 'subexpression'
    assert len(result.parsed['children']) == 3


def test_parse_pipe():
    result = Parser().parse('foo | bar')
    assert result.parsed['type'] == 'pipe'


def test_parse_or():
    result = Parser().parse('a || b')
    assert result.parsed['type'] == 'or_expression'


def test_parse_and():
    result = Parser().parse('a && b')
    assert result.parsed['type'] == 'and_expression'


def test_parse_not():
    result = Parser().parse('!a')
    assert result.parsed['type'] == 'not_expression'


def test_parse_comparators():
    for expr, cmp_type in [('a == b', 'eq'), ('a != b', 'ne'),
                           ('a > b', 'gt'), ('a >= b', 'gte'),
                           ('a < b', 'lt'), ('a <= b', 'lte')]:
        result = Parser().parse(expr)
        assert result.parsed['type'] == 'comparator'
        assert result.parsed['children'][0] == cmp_type


def test_parse_function_expression():
    result = Parser().parse('length(@)')
    assert result.parsed['type'] == 'function_expression'
    assert result.parsed['value'] == 'length'
    assert result.parsed['children'] == []


def test_parse_function_with_args():
    result = Parser().parse('contains(foo, bar)')
    assert result.parsed['type'] == 'function_expression'
    assert result.parsed['value'] == 'contains'
    assert len(result.parsed['children']) == 2


def test_parse_multi_select_list():
    result = Parser().parse('[foo, bar]')
    assert result.parsed['type'] == 'multi_select_list'
    assert len(result.parsed['children']) == 2


def test_parse_multi_select_hash():
    result = Parser().parse('{a: foo}')
    assert result.parsed['type'] == 'multi_select_dict'
    assert result.parsed['children'][0]['type'] == 'key_val_pair'
    assert result.parsed['children'][0]['value'] == 'a'


def test_parse_multi_select_hash_multiple_pairs():
    result = Parser().parse('{a: foo, b: bar}')
    assert len(result.parsed['children']) == 2


def test_parse_index():
    result = Parser().parse('foo[0]')
    assert result.parsed['type'] == 'index_expression'


def test_parse_slice():
    result = Parser().parse('foo[0:1]')
    assert result.parsed['type'] == 'projection'


def test_parse_flatten():
    result = Parser().parse('foo[]')
    assert result.parsed['type'] == 'projection'


def test_parse_star_projection():
    result = Parser().parse('foo[*]')
    assert result.parsed['type'] == 'projection'


def test_parse_value_projection():
    result = Parser().parse('foo.*')
    assert result.parsed['type'] == 'value_projection'


def test_parse_filter():
    result = Parser().parse('foo[?bar == `1`]')
    assert result.parsed['type'] == 'filter_projection'


def test_parse_nud_star():
    result = Parser().parse('*')
    assert result.parsed['type'] == 'value_projection'


def test_parse_nud_star_with_rbracket():
    result = Parser().parse('[*]')
    assert result.parsed['type'] == 'projection'


def test_parse_nud_flatten():
    result = Parser().parse('[]')
    assert result.parsed['type'] == 'projection'


def test_parse_nud_filter():
    result = Parser().parse('[?foo]')
    assert result.parsed['type'] == 'filter_projection'


def test_parse_expref():
    result = Parser().parse('sort_by(@, &foo)')
    assert result.parsed['type'] == 'function_expression'
    assert result.parsed['children'][1]['type'] == 'expref'


def test_parse_nested_parens():
    result = Parser().parse('(foo)')
    assert result.parsed['type'] == 'field'
    assert result.parsed['value'] == 'foo'


def test_parse_dot_multi_select_list():
    result = Parser().parse('foo.[bar]')
    assert result.parsed['type'] == 'subexpression'
    assert result.parsed['children'][1]['type'] == 'multi_select_list'


def test_parse_dot_multi_select_hash():
    result = Parser().parse('foo.{a: bar}')
    assert result.parsed['type'] == 'subexpression'
    assert result.parsed['children'][1]['type'] == 'multi_select_dict'


def test_parse_dot_star():
    result = Parser().parse('foo.*.bar')
    assert result.parsed['type'] == 'value_projection'


def test_parse_filter_then_flatten():
    result = Parser().parse('[?foo][]')
    assert result.parsed['type'] == 'projection'


# ---------------------------------------------------------------------------
# Search on ParsedResult for full expressions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('expr,data,expected', [
    ('foo', {'foo': 1}, 1),
    ('foo.bar', {'foo': {'bar': 2}}, 2),
    ('foo[0]', {'foo': [3, 4]}, 3),
    ('foo[1:]', {'foo': [1, 2, 3]}, [2, 3]),
    ('length(foo)', {'foo': 'abc'}, 3),
    ('foo || bar', {'bar': 'x'}, 'x'),
    ('[foo, bar]', {'foo': 1, 'bar': 2}, [1, 2]),
    ('{a: foo}', {'foo': 5}, {'a': 5}),
])
def test_parsed_result_search_end_to_end(expr, data, expected):
    assert Parser().parse(expr).search(data) == expected


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_incomplete_expression_empty_string():
    with pytest.raises(exceptions.IncompleteExpressionError) as exc:
        Parser().parse('')
    assert exc.value.expression == ''


def test_incomplete_expression_sets_expression():
    with pytest.raises(exceptions.IncompleteExpressionError) as exc:
        Parser().parse('foo.')
    assert exc.value.expression == 'foo.'


def test_lexer_error_sets_expression():
    with pytest.raises(exceptions.LexerError) as exc:
        Parser().parse('foo]bar')
    assert exc.value.expression == 'foo]bar'


def test_parse_error_sets_expression():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse('foo bar')
    assert exc.value.expression == 'foo bar'


def test_unexpected_token_at_end():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse('foo]')
    assert 'Unexpected token' in str(exc.value)


def test_quoted_identifier_not_allowed_for_function():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse('"foo"(bar)')
    assert 'Quoted identifier not allowed for function names.' in str(exc.value)


def test_invalid_function_name():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse('(foo)(bar)')
    assert "Invalid function name" in str(exc.value)


def test_slice_syntax_error():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('[1:2:3:4]')


def test_slice_syntax_error_non_number():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('[foo:bar]')


def test_dot_rhs_error():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse('foo.@')
    assert 'Expecting' in str(exc.value)


def test_led_error():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('foo @')


def test_nud_error_invalid_token():
    with pytest.raises(exceptions.ParseError) as exc:
        Parser().parse(']')
    assert str(exc.value)


def test_match_multiple_tokens_error():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('{1: foo}')


def test_projection_rhs_error():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('foo[*]@')


def test_eof_in_match_raises_incomplete():
    with pytest.raises(exceptions.IncompleteExpressionError):
        Parser().parse('foo[')


def test_parse_index_missing_rbracket():
    with pytest.raises(exceptions.IncompleteExpressionError):
        Parser().parse('foo[0')
