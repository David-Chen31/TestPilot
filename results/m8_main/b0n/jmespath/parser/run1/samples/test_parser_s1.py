import pytest

from jmespath.parser import Parser, ParsedResult
from jmespath import ast
from jmespath import exceptions


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParserBasic:
    def test_parse_returns_parsed_result(self):
        p = Parser()
        result = p.parse("foo")
        assert isinstance(result, ParsedResult)
        assert result.expression == "foo"

    def test_parse_simple_field(self):
        result = Parser().parse("foo")
        assert result.parsed == ast.field("foo")

    def test_parse_literal(self):
        result = Parser().parse("`true`")
        assert result.parsed == ast.literal(True)

    def test_parse_index(self):
        result = Parser().parse("foo[0]")
        assert result.parsed == ast.index_expression([ast.field("foo"), ast.index(0)])

    def test_parse_slice(self):
        result = Parser().parse("foo[1:2:3]")
        # slice produces a projection
        assert result.parsed['type'] == 'projection'

    def test_parse_current_node(self):
        result = Parser().parse("@")
        assert result.parsed == ast.current_node()

    def test_parse_pipe(self):
        result = Parser().parse("foo | bar")
        assert result.parsed == ast.pipe(ast.field("foo"), ast.field("bar"))

    def test_parse_or(self):
        result = Parser().parse("foo || bar")
        assert result.parsed['type'] == 'or_expression'

    def test_parse_and(self):
        result = Parser().parse("foo && bar")
        assert result.parsed['type'] == 'and_expression'

    def test_parse_comparator(self):
        result = Parser().parse("foo == bar")
        assert result.parsed == ast.comparator("eq", ast.field("foo"), ast.field("bar"))

    def test_parse_multiselect_list(self):
        result = Parser().parse("[foo, bar]")
        assert result.parsed == ast.multi_select_list([ast.field("foo"), ast.field("bar")])

    def test_parse_multiselect_hash(self):
        result = Parser().parse("{a: foo, b: bar}")
        assert result.parsed['type'] == 'multi_select_dict'

    def test_parse_function(self):
        result = Parser().parse("length(@)")
        assert result.parsed['type'] == 'function_expression'
        assert result.parsed['value'] == 'length'

    def test_parse_star_projection(self):
        result = Parser().parse("foo[*]")
        assert result.parsed['type'] == 'projection'

    def test_parse_value_projection(self):
        result = Parser().parse("foo.*")
        assert result.parsed['type'] == 'value_projection'

    def test_parse_filter(self):
        result = Parser().parse("foo[?a > `1`]")
        assert result.parsed['type'] == 'filter_projection'

    def test_parse_flatten(self):
        result = Parser().parse("foo[]")
        assert result.parsed['type'] == 'projection'
        assert result.parsed['children'][0]['type'] == 'flatten'

    def test_parse_expref(self):
        result = Parser().parse("sort_by(foo, &bar)")
        assert result.parsed['type'] == 'function_expression'

    def test_parse_subexpression(self):
        result = Parser().parse("foo.bar")
        assert result.parsed == ast.subexpression([ast.field("foo"), ast.field("bar")])

    def test_parse_nested_dot_chain(self):
        result = Parser().parse("a.b.c")
        assert result.parsed['type'] == 'subexpression'
        assert len(result.parsed['children']) == 3

    def test_parse_not(self):
        result = Parser().parse("!foo")
        assert result.parsed['type'] == 'not_expression'


class TestParserCache:
    def test_cache_hit_returns_same_object(self):
        p = Parser()
        r1 = p.parse("foo")
        r2 = p.parse("foo")
        assert r1 is r2

    def test_purge_clears_cache(self):
        p = Parser()
        r1 = p.parse("foo")
        Parser.purge()
        r2 = p.parse("foo")
        assert r1 is not r2

    def test_different_expressions_different_results(self):
        p = Parser()
        r1 = p.parse("foo")
        r2 = p.parse("bar")
        assert r1 is not r2

    def test_cache_max_size_eviction(self):
        Parser._CACHE.clear()
        old_max = Parser._MAX_SIZE
        try:
            Parser._MAX_SIZE = 2
            p = Parser()
            p.parse("a")
            p.parse("b")
            # cache now full
            assert len(Parser._CACHE) == 2
            p.parse("c")
            # should have evicted one to make room
            assert len(Parser._CACHE) == 2
            assert "c" in Parser._CACHE
        finally:
            Parser._MAX_SIZE = old_max


class TestParserErrors:
    def test_incomplete_expression_error(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("")

    def test_incomplete_expression_has_expression(self):
        try:
            Parser().parse("foo ||")
        except exceptions.IncompleteExpressionError as e:
            assert e.expression == "foo ||"
        else:
            pytest.fail("Expected IncompleteExpressionError")

    def test_parse_error_unexpected_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo bar")

    def test_lexer_error(self):
        with pytest.raises(exceptions.LexerError) as exc:
            Parser().parse("'unterminated")
        assert exc.value.expression == "'unterminated"

    def test_parse_error_expression_attached(self):
        try:
            Parser().parse("foo bar")
        except exceptions.ParseError as e:
            assert e.expression == "foo bar"
        else:
            pytest.fail("Expected ParseError")

    def test_quoted_identifier_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('"foo"(@)')
        assert 'Quoted identifier not allowed' in str(exc.value)

    def test_invalid_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("@(...)")
        assert 'Invalid function name' in str(exc.value)

    def test_slice_too_many_colons(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[1:2:3:4]")

    def test_slice_invalid_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[1:2:bar]")

    def test_dot_rhs_invalid(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.1")
        assert 'Expecting' in str(exc.value)

    def test_projection_rhs_syntax_error(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[].1")

    def test_multiselect_hash_invalid_key(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("{1: foo}")

    def test_lbracket_led_invalid(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[1")

    def test_lparen_nud_missing_rparen(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("(foo")

    def test_match_eof_incomplete(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("foo[")

    def test_cache_not_populated_on_error(self):
        Parser.purge()
        p = Parser()
        with pytest.raises(exceptions.ParseError):
            p.parse("foo bar")
        assert "foo bar" not in Parser._CACHE


class TestParsedResult:
    def test_search_simple(self):
        result = Parser().parse("foo")
        assert result.search({"foo": "bar"}) == "bar"

    def test_search_missing_key_returns_none(self):
        result = Parser().parse("foo")
        assert result.search({"other": "bar"}) is None

    def test_search_projection(self):
        result = Parser().parse("foo[*].bar")
        value = {"foo": [{"bar": 1}, {"bar": 2}]}
        assert result.search(value) == [1, 2]

    def test_search_with_options(self):
        result = Parser().parse("foo")
        assert result.search({"foo": "bar"}, options=None) == "bar"

    def test_repr_returns_repr_of_parsed(self):
        result = Parser().parse("foo")
        assert repr(result) == repr(result.parsed)

    def test_expression_attribute(self):
        result = Parser().parse("foo.bar")
        assert result.expression == "foo.bar"

    def test_parsed_attribute(self):
        result = Parser().parse("foo")
        assert result.parsed == ast.field("foo")

    def test_render_dot_file(self):
        result = Parser().parse("foo")
        contents = result._render_dot_file()
        assert isinstance(contents, str)
        assert contents  # non-empty


class TestParserLookahead:
    def test_custom_lookahead_buffer_size(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert len(p._tokens) == 3
        # can still parse
        result = p.parse("foo.bar")
        assert result.parsed['type'] == 'subexpression'

    def test_default_lookahead(self):
        p = Parser()
        assert p._buffer_size == 2

    def test_index_reset_after_parse(self):
        p = Parser()
        p.parse("foo")
        # after parse the index should point at eof (last token)
        assert p._index == len(p._tokens) - 1


class TestParserBindingPower:
    def test_binding_power_values(self):
        assert Parser.BINDING_POWER['eof'] == 0
        assert Parser.BINDING_POWER['pipe'] == 1
        assert Parser.BINDING_POWER['or'] == 2
        assert Parser.BINDING_POWER['and'] == 3
        assert Parser.BINDING_POWER['eq'] == 5
        assert Parser.BINDING_POWER['flatten'] == 9
        assert Parser.BINDING_POWER['star'] == 20
        assert Parser.BINDING_POWER['filter'] == 21
        assert Parser.BINDING_POWER['dot'] == 40
        assert Parser.BINDING_POWER['not'] == 45
        assert Parser.BINDING_POWER['lbrace'] == 50
        assert Parser.BINDING_POWER['lbracket'] == 55
        assert Parser.BINDING_POWER['lparen'] == 60

    def test_projection_stop_value(self):
        assert Parser._PROJECTION_STOP == 10

    def test_max_size_value(self):
        assert Parser._MAX_SIZE == 512
