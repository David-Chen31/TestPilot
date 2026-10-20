"""Tests for jmespath.parser (Parser and ParsedResult)."""
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
# Basic parsing
# ---------------------------------------------------------------------------

def test_parse_returns_parsed_result():
    result = Parser().parse('foo')
    assert isinstance(result, ParsedResult)
    assert result.expression == 'foo'
    assert result.parsed == {'type': 'field', 'value': 'foo', 'children': []}


def test_parse_unquoted_identifier():
    result = Parser().parse('foo')
    assert result.parsed['type'] == 'field'
    assert result.parsed['value'] == 'foo'


def test_parse_quoted_identifier():
    result = Parser().parse('"foo bar"')
    assert result.parsed['type'] == 'field'
    assert result.parsed['value'] == 'foo bar'


def test_parse_subexpression():
    result = Parser().parse('foo.bar')
    assert result.parsed['type'] == 'subexpression'
    assert len(result.parsed['children']) == 2
    assert result.parsed['children'][0]['value'] == 'foo'
    assert result.parsed['children'][1]['value'] == 'bar'


def test_parse_subexpression_chained_uses_append():
    result = Parser().parse('foo.bar.baz')
    node = result.parsed
    assert node['type'] == 'subexpression'
    assert [c['value'] for c in node['children']] == ['foo', 'bar', 'baz']


def test_parse_index():
    result = Parser().parse('foo[0]')
    node = result.parsed
    assert node['type'] == 'index_expression'
    assert node['children'][1]['type'] == 'index'
    assert node['children'][1]['value'] == 0


def test_parse_index_optimization_appends_children():
    result = Parser().parse('foo[0][1]')
    node = result.parsed
    assert node['type'] == 'index_expression'
    assert len(node['children']) == 3
    assert node['children'][1]['value'] == 0
    assert node['children'][2]['value'] == 1


def test_parse_slice():
    result = Parser().parse('foo[1:2:3]')
    node = result.parsed
    assert node['type'] == 'projection'
    slice_expr = node['children'][1]['children'][1]
    assert slice_expr['type'] == 'slice'
    assert slice_expr['children'] == [1, 2, 3]


def test_parse_slice_with_missing_parts():
    result = Parser().parse('foo[:2]')
    node = result.parsed
    slice_expr = node['children'][1]['children'][1]
    assert slice_expr['children'] == [None, 2, None]


def test_parse_pipe():
    result = Parser().parse('foo | bar')
    assert result.parsed['type'] == 'pipe'
    assert result.parsed['children'][0]['value'] == 'foo'
    assert result.parsed['children'][1]['value'] == 'bar'


def test_parse_or_and():
    result = Parser().parse('a || b && c')
    assert result.parsed['type'] == 'or_expression'
    assert result.parsed['children'][1]['type'] == 'and_expression'


def test_parse_comparators():
    for op, node_type in [
        ('==', 'comparator'),
        ('!=', 'comparator'),
        ('>', 'comparator'),
        ('>=', 'comparator'),
        ('<', 'comparator'),
        ('<=', 'comparator'),
    ]:
        result = Parser().parse('a %s b' % op)
        assert result.parsed['type'] == node_type


def test_parse_literal():
    result = Parser().parse('`1`')
    assert result.parsed == {'type': 'literal', 'value': 1}


def test_parse_not():
    result = Parser().parse('!foo')
    assert result.parsed['type'] == 'not_expression'


def test_parse_current_node():
    result = Parser().parse('@')
    assert result.parsed['type'] == 'current_node'


def test_parse_expref():
    result = Parser().parse('&foo')
    assert result.parsed['type'] == 'expref'
    assert result.parsed['children'][0]['value'] == 'foo'


def test_parse_paren():
    result = Parser().parse('(a || b) && c')
    assert result.parsed['type'] == 'and_expression'


def test_parse_negative_number_in_parens():
    result = Parser().parse('(-1)')
    assert result.parsed['type'] == 'literal' or result.parsed['type'] == 'number'


def test_parse_function():
    result = Parser().parse('foo()')
    node = result.parsed
    assert node['type'] == 'function_expression'
    assert node['value'] == 'foo'


def test_parse_function_with_args():
    result = Parser().parse('foo(bar, baz)')
    node = result.parsed
    assert node['type'] == 'function_expression'
    assert len(node['children']) == 2
    assert node['children'][0]['value'] == 'bar'


def test_parse_multi_select_list():
    result = Parser().parse('[foo, bar]')
    node = result.parsed
    assert node['type'] == 'multi_select_list'
    assert [c['value'] for c in node['children']] == ['foo', 'bar']


def test_parse_multi_select_hash():
    result = Parser().parse('{a: foo, b: bar}')
    node = result.parsed
    assert node['type'] == 'multi_select_dict'
    assert len(node['children']) == 2
    assert node['children'][0]['type'] == 'key_val_pair'


def test_parse_star_projection():
    result = Parser().parse('*')
    assert result.parsed['type'] == 'value_projection'


def test_parse_dot_star():
    result = Parser().parse('foo.*')
    assert result.parsed['type'] == 'value_projection'


def test_parse_flatten():
    result = Parser().parse('foo[]')
    assert result.parsed['type'] == 'projection'


def test_parse_index_projection():
    result = Parser().parse('foo[*].bar')
    assert result.parsed['type'] == 'projection'


def test_parse_filter():
    result = Parser().parse('foo[?bar==`1`]')
    assert result.parsed['type'] == 'filter_projection'


def test_parse_filter_at_nud():
    result = Parser().parse('[?bar]')
    assert result.parsed['type'] == 'filter_projection'


def test_parse_complex_expression():
    result = Parser().parse('foo.bar[0] | [a, b]')
    assert result.parsed['type'] == 'pipe'


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_parse_empty_expression_raises_incomplete():
    with pytest.raises(exceptions.IncompleteExpressionError):
        Parser().parse('')


def test_parse_incomplete_dot_raises():
    with pytest.raises(exceptions.IncompleteExpressionError):
        Parser().parse('foo.')


def test_parse_unexpected_token():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('foo bar')


def test_parse_lexer_error():
    with pytest.raises(exceptions.LexerError):
        Parser().parse('"unterminated')


def test_parse_quoted_identifier_as_function_name():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('"foo"()')


def test_parse_invalid_function_name():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('[0]()')


def test_parse_syntax_error_in_slice():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('foo[1:2:3:4]')


def test_parse_syntax_error_in_multi_select_hash_missing_colon():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('{a}')


def test_parse_invalid_after_dot():
    with pytest.raises(exceptions.ParseError):
        Parser().parse('foo.1')


def test_incomplete_expression_error_has_expression():
    parser_instance = Parser()
    try:
        parser_instance.parse('foo.')
    except exceptions.IncompleteExpressionError as e:
        assert e.expression == 'foo.'
    else:
        pytest.fail("Expected IncompleteExpressionError")


def test_parse_error_has_expression_set():
    parser_instance = Parser()
    try:
        parser_instance.parse('foo bar')
    except exceptions.ParseError as e:
        assert e.expression == 'foo bar'
    else:
        pytest.fail("Expected ParseError")


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

def test_cache_hit_returns_same_object():
    p = Parser()
    first = p.parse('foo')
    second = p.parse('foo')
    assert first is second


def test_purge_clears_cache():
    Parser().parse('foo')
    assert 'foo' in Parser._CACHE
    Parser.purge()
    assert Parser._CACHE == {}


def test_cache_max_size_respected(monkeypatch):
    monkeypatch.setattr(Parser, '_MAX_SIZE', 2)
    Parser.purge()
    Parser().parse('a')
    Parser().parse('b')
    Parser().parse('c')
    assert len(Parser._CACHE) <= 2


def test_cache_evicts_oldest(monkeypatch):
    monkeypatch.setattr(Parser, '_MAX_SIZE', 2)
    Parser.purge()
    Parser().parse('a')
    Parser().parse('b')
    Parser().parse('c')
    assert 'a' not in Parser._CACHE
    assert 'c' in Parser._CACHE


# ---------------------------------------------------------------------------
# ParsedResult
# ---------------------------------------------------------------------------

def test_parsed_result_attributes():
    result = Parser().parse('foo')
    assert isinstance(result, ParsedResult)
    assert result.expression == 'foo'
    assert isinstance(result.parsed, dict)


def test_parsed_result_repr():
    result = Parser().parse('foo')
    assert repr(result) == repr(result.parsed)


def test_parsed_result_search():
    result = Parser().parse('foo')
    assert result.search({'foo': 'bar'}) == 'bar'


def test_parsed_result_search_with_options():
    from jmespath.visitor import Options
    result = Parser().parse('foo')
    assert result.search({'foo': 'baz'}, Options()) == 'baz'


def test_parsed_result_search_missing_key():
    result = Parser().parse('foo')
    assert result.search({'bar': 'baz'}) is None


def test_parsed_result_render_dot_file():
    result = Parser().parse('foo.bar')
    contents = result._render_dot_file()
    assert isinstance(contents, str)
    assert 'foo' in contents
    assert 'bar' in contents


# ---------------------------------------------------------------------------
# Parser internals / structure
# ---------------------------------------------------------------------------

def test_parser_init_lookahead():
    p = Parser()
    assert p._buffer_size == 2
    assert p._tokens == [None, None]
    assert p._index == 0
    assert p.tokenizer is None


def test_parser_init_custom_lookahead():
    p = Parser(lookahead=3)
    assert p._buffer_size == 3
    assert p._tokens == [None, None, None]


def test_binding_power_table():
    assert Parser.BINDING_POWER['pipe'] == 1
    assert Parser.BINDING_POWER['or'] == 2
    assert Parser.BINDING_POWER['and'] == 3
    assert Parser.BINDING_POWER['eq'] == 5
    assert Parser.BINDING_POWER['eof'] == 0
    assert Parser._PROJECTION_STOP == 10


def test_parse_resets_index():
    p = Parser()
    p.parse('foo')
    p.parse('bar')
    assert p._index == len(p._tokens)


def test_current_token_and_lookahead_helpers():
    p = Parser()
    p.parse('foo')
    # After parse, index points at eof (last token).
    assert p._current_token() == 'eof'
    assert p._lookahead_token(0)['type'] == 'eof'


def test_ast_node_types_are_ast_nodes():
    result = Parser().parse('foo.bar')
    # ast nodes are dicts subclass
    assert isinstance(result.parsed, dict)
    assert 'type' in result.parsed
    assert 'children' in result.parsed
