import pytest

from jmespath import ast
from jmespath import exceptions
from jmespath.parser import Parser, ParsedResult


@pytest.fixture(autouse=True)
def clear_cache():
    Parser.purge()
    yield
    Parser.purge()


class TestParserBasics:
    def test_parse_returns_parsed_result(self):
        result = Parser().parse("foo")
        assert isinstance(result, ParsedResult)
        assert result.expression == "foo"
        assert result.parsed == {"type": "field", "children": [], "value": "foo"}

    def test_parse_field(self):
        result = Parser().parse("foo.bar")
        assert result.parsed["type"] == "subexpression"
        assert len(result.parsed["children"]) == 2

    def test_parse_caches_result(self):
        parser = Parser()
        first = parser.parse("foo.bar")
        second = parser.parse("foo.bar")
        assert first is second

    def test_purge_clears_cache(self):
        Parser().parse("foo")
        assert "foo" in Parser._CACHE
        Parser.purge()
        assert Parser._CACHE == {}

    def test_cache_max_size_eviction(self):
        old_max = Parser._MAX_SIZE
        Parser._MAX_SIZE = 3
        try:
            parser = Parser()
            parser.parse("a")
            parser.parse("b")
            parser.parse("c")
            assert len(Parser._CACHE) == 3
            parser.parse("d")
            assert len(Parser._CACHE) == 3
            assert "d" in Parser._CACHE
        finally:
            Parser._MAX_SIZE = old_max


class TestParserErrors:
    def test_empty_expression_raises_incomplete(self):
        with pytest.raises(exceptions.IncompleteExpressionError) as exc_info:
            Parser().parse("")
        assert exc_info.value.expression == ""

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse("foo bar")
        assert exc_info.value.expression == "foo bar"

    def test_invalid_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("!")

    def test_quoted_identifier_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse('"foo"(@)')
        assert "Quoted identifier not allowed for function names." in str(
            exc_info.value
        )

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc_info:
            Parser().parse("{a: b}()")
        assert "Invalid function name" in str(exc_info.value)

    def test_unmatched_rparen(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("(foo")

    def test_bad_slice(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[1:2:3:4]")

    def test_incomplete_expression_bracket(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("foo[")

    def test_lexer_error_gets_expression(self):
        with pytest.raises(exceptions.LexerError) as exc_info:
            Parser().parse("foo['bar")
        assert exc_info.value.expression == "foo['bar"


class TestParsesVariousConstructs:
    def test_identity_star(self):
        result = Parser().parse("*")
        assert result.parsed["type"] == "value_projection"

    def test_flatten(self):
        result = Parser().parse("foo[]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "flatten"

    def test_index(self):
        result = Parser().parse("foo[0]")
        assert result.parsed["type"] == "index_expression"

    def test_slice_projection(self):
        result = Parser().parse("foo[0:2]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "index_expression"

    def test_multiselect_list(self):
        result = Parser().parse("[foo, bar]")
        assert result.parsed["type"] == "multi_select_list"

    def test_multiselect_hash(self):
        result = Parser().parse("{a: foo, b: bar}")
        assert result.parsed["type"] == "multi_select_dict"

    def test_comparator_eq(self):
        result = Parser().parse("foo == bar")
        assert result.parsed["type"] == "comparator"

    def test_and_or(self):
        result = Parser().parse("a && b || c")
        assert result.parsed["type"] == "or_expression"

    def test_pipe(self):
        result = Parser().parse("foo | bar")
        assert result.parsed["type"] == "pipe"

    def test_not_expression(self):
        result = Parser().parse("!foo")
        assert result.parsed["type"] == "not_expression"

    def test_function_expression(self):
        result = Parser().parse("length(foo)")
        assert result.parsed["type"] == "function_expression"
        assert result.parsed["value"] == "length"

    def test_function_no_args(self):
        result = Parser().parse("foo()")
        assert result.parsed["type"] == "function_expression"
        assert result.parsed["children"] == []

    def test_filter(self):
        result = Parser().parse("foo[?bar]")
        assert result.parsed["type"] == "filter_projection"

    def test_current_node(self):
        result = Parser().parse("@")
        assert result.parsed["type"] == "current"

    def test_expref(self):
        result = Parser().parse("sort_by(@, &foo)")
        args = result.parsed["children"]
        assert any(a["type"] == "expref" for a in args)

    def test_literal(self):
        result = Parser().parse("`1`")
        assert result.parsed["type"] == "literal"


class TestParsedResult:
    def test_search_field(self):
        result = Parser().parse("foo")
        assert result.search({"foo": "bar"}) == "bar"

    def test_search_missing(self):
        result = Parser().parse("missing")
        assert result.search({"foo": "bar"}) is None

    def test_search_with_options(self):
        result = Parser().parse("foo")
        # Just verify options are accepted and functional.
        assert result.search({"foo": 1}, options=None) == 1

    def test_repr(self):
        result = Parser().parse("foo")
        assert repr(result) == repr(result.parsed)

    def test_render_dot_file(self):
        result = Parser().parse("foo.bar")
        contents = result._render_dot_file()
        assert isinstance(contents, str)
        assert "digraph" in contents


class TestParserInternals:
    def test_init_default_lookahead(self):
        p = Parser()
        assert p._buffer_size == 2
        assert p._tokens == [None, None]
        assert p._index == 0
        assert p.tokenizer is None

    def test_init_custom_lookahead(self):
        p = Parser(lookahead=3)
        assert p._buffer_size == 3
        assert p._tokens == [None, None, None]

    def test_binding_power_values(self):
        assert Parser.BINDING_POWER["eof"] == 0
        assert Parser.BINDING_POWER["pipe"] == 1
        assert Parser.BINDING_POWER["or"] == 2
        assert Parser.BINDING_POWER["and"] == 3
        assert Parser.BINDING_POWER["eq"] == 5
        assert Parser.BINDING_POWER["star"] == 20
        assert Parser.BINDING_POWER["lparen"] == 60

    def test_projection_stop(self):
        assert Parser._PROJECTION_STOP == 10
