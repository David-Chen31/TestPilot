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

    def test_parse_simple_field(self):
        result = Parser().parse("foo")
        assert result.parsed == {"type": "field", "children": [], "value": "foo"}

    def test_parse_nested_dot(self):
        result = Parser().parse("foo.bar")
        assert result.parsed == {
            "type": "subexpression",
            "children": [
                {"type": "field", "children": [], "value": "foo"},
                {"type": "field", "children": [], "value": "bar"},
            ],
        }

    def test_parse_caches_result(self):
        parser = Parser()
        first = parser.parse("foo")
        second = parser.parse("foo")
        assert first is second

    def test_purge_clears_cache(self):
        parser = Parser()
        first = parser.parse("foo")
        Parser.purge()
        second = parser.parse("foo")
        assert first is not second

    def test_cache_max_size(self):
        parser = Parser()
        # Fill cache beyond max size; shouldn't grow unbounded.
        for i in range(Parser._MAX_SIZE + 10):
            parser.parse("field%d" % i)
        assert len(Parser._CACHE) <= Parser._MAX_SIZE


class TestParsedResult:
    def test_repr(self):
        result = Parser().parse("foo")
        assert repr(result) == repr(result.parsed)

    def test_search(self):
        result = Parser().parse("foo")
        assert result.search({"foo": "bar"}) == "bar"

    def test_search_with_missing_key(self):
        result = Parser().parse("foo")
        assert result.search({"other": "bar"}) is None

    def test_search_options(self):
        result = Parser().parse("foo")
        # options passed through to interpreter
        assert result.search({"foo": "bar"}, options=None) == "bar"

    def test_render_dot_file(self):
        result = Parser().parse("foo")
        contents = result._render_dot_file()
        assert isinstance(contents, str)
        assert "digraph" in contents

    def test_attributes(self):
        node = ast.field("foo")
        parsed = ParsedResult("foo", node)
        assert parsed.expression == "foo"
        assert parsed.parsed is node


class TestParserExpressions:
    def test_literal(self):
        result = Parser().parse("`true`")
        assert result.parsed == {"type": "literal", "children": [], "value": True}

    def test_current_node(self):
        result = Parser().parse("@")
        assert result.parsed == {"type": "current", "children": []}

    def test_index(self):
        result = Parser().parse("foo[0]")
        assert result.parsed["type"] == "index_expression"

    def test_slice(self):
        result = Parser().parse("foo[0:1:2]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "index_expression"

    def test_slice_values(self):
        result = Parser().parse("foo[1:2:3]")
        index_expr = result.parsed["children"][0]
        slice_node = index_expr["children"][1]
        assert slice_node == {
            "type": "slice",
            "children": [1, 2, 3],
        }

    def test_projection_star(self):
        result = Parser().parse("foo[*].bar")
        assert result.parsed["type"] == "projection"

    def test_value_projection(self):
        result = Parser().parse("foo.*")
        assert result.parsed["type"] == "value_projection"

    def test_filter_projection(self):
        result = Parser().parse("foo[?bar == `1`]")
        assert result.parsed["type"] == "filter_projection"

    def test_flatten(self):
        result = Parser().parse("foo[]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "flatten"

    def test_multi_select_list(self):
        result = Parser().parse("[foo, bar]")
        assert result.parsed["type"] == "multi_select_list"

    def test_multi_select_dict(self):
        result = Parser().parse("{a: foo, b: bar}")
        assert result.parsed["type"] == "multi_select_dict"

    def test_quoted_identifier(self):
        result = Parser().parse('"foo"')
        assert result.parsed["type"] == "field"
        assert result.parsed["value"] == "foo"

    def test_function_call(self):
        result = Parser().parse("length(@)")
        assert result.parsed["type"] == "function_expression"
        assert result.parsed["value"] == "length"

    def test_function_no_args(self):
        result = Parser().parse("now()")
        assert result.parsed == {
            "type": "function_expression",
            "children": [],
            "value": "now",
        }

    def test_function_multiple_args(self):
        result = Parser().parse("sort_by(foo, &bar)")
        assert result.parsed["type"] == "function_expression"
        assert len(result.parsed["children"]) == 2

    def test_pipe(self):
        result = Parser().parse("foo | bar")
        assert result.parsed["type"] == "pipe"

    def test_or(self):
        result = Parser().parse("foo || bar")
        assert result.parsed["type"] == "or_expression"

    def test_and(self):
        result = Parser().parse("foo && bar")
        assert result.parsed["type"] == "and_expression"

    def test_not(self):
        result = Parser().parse("!foo")
        assert result.parsed["type"] == "not_expression"

    def test_comparator_eq(self):
        result = Parser().parse("foo == bar")
        assert result.parsed["type"] == "comparator"
        assert result.parsed["value"] == "eq"

    def test_comparator_ne(self):
        result = Parser().parse("foo != bar")
        assert result.parsed["value"] == "ne"

    def test_comparator_gt(self):
        result = Parser().parse("foo > bar")
        assert result.parsed["value"] == "gt"

    def test_comparator_gte(self):
        result = Parser().parse("foo >= bar")
        assert result.parsed["value"] == "gte"

    def test_comparator_lt(self):
        result = Parser().parse("foo < bar")
        assert result.parsed["value"] == "lt"

    def test_comparator_lte(self):
        result = Parser().parse("foo <= bar")
        assert result.parsed["value"] == "lte"

    def test_expref(self):
        result = Parser().parse("sort_by(foo, &bar)")
        assert result.parsed["type"] == "function_expression"
        arg = result.parsed["children"][1]
        assert arg["type"] == "expref"

    def test_paren_expression(self):
        result = Parser().parse("(foo)")
        assert result.parsed == {"type": "field", "children": [], "value": "foo"}

    def test_nud_star(self):
        result = Parser().parse("*")
        assert result.parsed["type"] == "value_projection"
        assert result.parsed["children"][0] == {"type": "identity", "children": []}

    def test_nud_filter(self):
        result = Parser().parse("[?foo == `1`]")
        assert result.parsed["type"] == "filter_projection"

    def test_nud_flatten(self):
        result = Parser().parse("[]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "flatten"

    def test_lbracket_star(self):
        result = Parser().parse("[*].foo")
        assert result.parsed["type"] == "projection"

    def test_dot_star(self):
        result = Parser().parse("foo.bar.*")
        assert result.parsed["type"] == "value_projection"

    def test_nested_lbracket_index(self):
        result = Parser().parse("foo[0][1]")
        assert result.parsed["type"] == "index_expression"
        # Optimization adds children to the existing index_expression.
        assert len(result.parsed["children"]) == 3


class TestParserErrors:
    def test_empty_expression(self):
        # Empty input is caught by the lexer, not by the parser.
        with pytest.raises(exceptions.EmptyExpressionError):
            Parser().parse("")

    def test_parse_error_expression_attribute(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.")
        assert exc.value.expression == "foo."

    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo bar")

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("(`1`)(@)")

    def test_quoted_identifier_function(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('"foo"(@)')
        assert "Quoted identifier" in str(exc.value) or "Invalid function" in str(
            exc.value
        )

    def test_lexer_error(self):
        with pytest.raises(exceptions.LexerError) as exc:
            Parser().parse("'unterminated")
        assert exc.value.expression == "'unterminated"

    def test_missing_rbracket(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("foo[0")

    def test_invalid_slice(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo[0:1:2:3]")

    def test_parse_error_led_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("*[")

    def test_dot_rhs_error(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.bar.")
        assert exc.value.expression == "foo.bar."

    def test_parse_error_sets_expression_on_multi_select(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("{a: }")
        assert exc.value.expression == "{a: }"

    def test_incomplete_expression_via_eof(self):
        # A dangling dot at end-of-input produces a ParseError (not an
        # IncompleteExpressionError) because _parse_dot_rhs explicitly
        # raises "_raise_parse_error_for_token" for unexpected dotrhs tokens.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.")
        assert exc.value.expression == "foo."


class TestParserInternals:
    def test_lookahead_default(self):
        parser = Parser()
        assert parser._buffer_size == 2

    def test_custom_lookahead(self):
        parser = Parser(lookahead=3)
        assert parser._buffer_size == 3

    def test_initial_index(self):
        parser = Parser()
        assert parser._index == 0

    def test_binding_power_values(self):
        assert Parser.BINDING_POWER["pipe"] == 1
        assert Parser.BINDING_POWER["dot"] == 40
        assert Parser.BINDING_POWER["eof"] == 0

    def test_projection_stop(self):
        assert Parser._PROJECTION_STOP == 10

    def test_max_size(self):
        assert Parser._MAX_SIZE == 512

    def test_parse_resets_index(self):
        parser = Parser()
        parser.parse("foo")
        # After parsing, index points past the eof token.
        assert parser._tokens[parser._index]["type"] == "eof"
