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


class TestParserInit:
    def test_default_lookahead(self):
        p = Parser()
        assert p._buffer_size == 2
        assert p._tokens == [None, None]
        assert p._index == 0
        assert p.tokenizer is None

    def test_custom_lookahead(self):
        p = Parser(lookahead=4)
        assert p._buffer_size == 4
        assert len(p._tokens) == 4


class TestParserParse:
    def test_simple_field(self):
        result = Parser().parse("foo")
        assert isinstance(result, ParsedResult)
        assert result.expression == "foo"
        assert result.parsed == {"type": "field", "children": [], "value": "foo"}

    def test_caches_result(self):
        p = Parser()
        r1 = p.parse("foo")
        r2 = p.parse("foo")
        assert r1 is r2
        assert Parser._CACHE["foo"] is r1

    def test_cache_key_used(self):
        Parser().parse("foo.bar")
        assert "foo.bar" in Parser._CACHE

    def test_purge_clears_cache(self):
        Parser().parse("foo")
        assert len(Parser._CACHE) == 1
        Parser.purge()
        assert Parser._CACHE == {}

    def test_cache_max_size_eviction(self):
        original = Parser._MAX_SIZE
        Parser._MAX_SIZE = 2
        try:
            Parser.purge()
            p = Parser()
            p.parse("a")
            p.parse("b")
            assert len(Parser._CACHE) == 2
            p.parse("c")
            assert len(Parser._CACHE) == 2
            assert "a" not in Parser._CACHE
            assert "b" in Parser._CACHE
            assert "c" in Parser._CACHE
        finally:
            Parser._MAX_SIZE = original

    def test_parse_returns_parsed_result(self):
        result = Parser().parse("foo")
        assert isinstance(result, ParsedResult)
        assert result.search({"foo": "bar"}) == "bar"

    def test_lexer_error_attaches_expression(self):
        with pytest.raises(exceptions.LexerError) as exc:
            Parser().parse('"unterminated')
        assert exc.value.expression == '"unterminated'

    def test_parse_error_attaches_expression(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.[")
        assert exc.value.expression == "foo.["

    def test_incomplete_expression_error(self):
        # "foo." ends with eof after a dot, but _parse_dot_rhs raises a
        # ParseError (not IncompleteExpressionError) because the eof token
        # does not match the allowed types.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.")
        assert exc.value.expression == "foo."
        assert "Expecting:" in str(exc.value)


class TestNudTokens:
    def test_literal(self):
        result = Parser().parse("`1`")
        assert result.parsed == {"type": "literal", "children": [], "value": 1}

    def test_unquoted_identifier(self):
        result = Parser().parse("foo")
        assert result.parsed["type"] == "field"
        assert result.parsed["value"] == "foo"

    def test_quoted_identifier(self):
        result = Parser().parse('"foo"')
        assert result.parsed["type"] == "field"
        assert result.parsed["value"] == "foo"

    def test_quoted_identifier_function_name_error(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse('"foo"(@)')
        assert "Quoted identifier not allowed for function names" in str(exc.value)

    def test_star(self):
        result = Parser().parse("*")
        assert result.parsed["type"] == "value_projection"

    def test_star_on_array(self):
        result = Parser().parse("[*]")
        assert result.parsed["type"] == "projection"

    def test_filter(self):
        result = Parser().parse("[?a==`1`]")
        assert result.parsed["type"] == "filter_projection"

    def test_lbrace(self):
        result = Parser().parse("{foo: bar}")
        assert result.parsed["type"] == "multi_select_dict"
        assert result.parsed["children"][0]["type"] == "key_val_pair"

    def test_lparen(self):
        result = Parser().parse("(foo)")
        assert result.parsed == {"type": "field", "children": [], "value": "foo"}

    def test_lparen_missing_rparen(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("(foo")

    def test_flatten(self):
        result = Parser().parse("foo[]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "flatten"

    def test_not(self):
        result = Parser().parse("!foo")
        assert result.parsed["type"] == "not_expression"

    def test_lbracket_index(self):
        result = Parser().parse("foo[0]")
        assert result.parsed["type"] == "index_expression"

    def test_lbracket_slice(self):
        result = Parser().parse("foo[1:3]")
        # slice triggers projection
        assert result.parsed["type"] == "projection"

    def test_lbracket_multiselect(self):
        result = Parser().parse("[foo, bar]")
        assert result.parsed["type"] == "multi_select_list"

    def test_lbracket_star(self):
        result = Parser().parse("[*].foo")
        assert result.parsed["type"] == "projection"

    def test_current(self):
        result = Parser().parse("@")
        assert result.parsed["type"] == "current"

    def test_expref(self):
        result = Parser().parse("&foo")
        assert result.parsed["type"] == "expref"


class TestNudFlattenDetails:
    """Cover _token_nud_flatten (leading '[]'), which _token_led_flatten
    does not fully exercise."""

    def test_leading_flatten_builds_projection_of_flatten_identity(self):
        result = Parser().parse("[]")
        parsed = result.parsed
        assert parsed["type"] == "projection"
        left = parsed["children"][0]
        assert left["type"] == "flatten"
        assert left["children"][0]["type"] == "identity"
        assert parsed["children"][1]["type"] == "identity"

    def test_leading_flatten_with_rhs(self):
        result = Parser().parse("[].foo")
        parsed = result.parsed
        assert parsed["type"] == "projection"
        left = parsed["children"][0]
        assert left["type"] == "flatten"
        assert left["children"][0]["type"] == "identity"
        assert parsed["children"][1] == {
            "type": "field", "children": [], "value": "foo"}

    def test_leading_flatten_search(self):
        result = Parser().parse("[]")
        assert result.search([[1, 2], [3]]) == [1, 2, 3]

    def test_leading_flatten_with_dot_search(self):
        result = Parser().parse("[].a")
        assert result.search([{"a": 1}, {"a": 2}]) == [1, 2]


class TestLedTokens:
    def test_dot(self):
        result = Parser().parse("foo.bar")
        assert result.parsed["type"] == "subexpression"
        assert len(result.parsed["children"]) == 2

    def test_dot_extends_subexpression(self):
        result = Parser().parse("foo.bar.baz")
        assert result.parsed["type"] == "subexpression"
        assert len(result.parsed["children"]) == 3

    def test_dot_star(self):
        result = Parser().parse("foo.*.bar")
        assert result.parsed["type"] == "value_projection"

    def test_pipe(self):
        result = Parser().parse("foo | bar")
        assert result.parsed["type"] == "pipe"

    def test_or(self):
        result = Parser().parse("foo || bar")
        assert result.parsed["type"] == "or_expression"

    def test_and(self):
        result = Parser().parse("foo && bar")
        assert result.parsed["type"] == "and_expression"

    def test_function_call(self):
        result = Parser().parse("length(foo)")
        assert result.parsed["type"] == "function_expression"
        assert result.parsed["value"] == "length"
        assert len(result.parsed["children"]) == 1

    def test_function_multiple_args(self):
        result = Parser().parse("foo(a, b, c)")
        assert len(result.parsed["children"]) == 3

    def test_invalid_function_name(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("[0](a)")
        assert "Invalid function name" in str(exc.value)

    def test_filter_led(self):
        result = Parser().parse("foo[?a]")
        assert result.parsed["type"] == "filter_projection"

    def test_filter_then_flatten(self):
        # The trailing [] flatten after the filter causes the outer node
        # to become a projection wrapping the filter_projection.
        result = Parser().parse("foo[?a][]")
        assert result.parsed["type"] == "projection"
        assert result.parsed["children"][0]["type"] == "flatten"
        inner = result.parsed["children"][0]["children"][0]
        assert inner["type"] == "filter_projection"

    @pytest.mark.parametrize("expr,op", [
        ("a==b", "eq"),
        ("a!=b", "ne"),
        ("a>b", "gt"),
        ("a>=b", "gte"),
        ("a<b", "lt"),
        ("a<=b", "lte"),
    ])
    def test_comparators(self, expr, op):
        result = Parser().parse(expr)
        assert result.parsed["type"] == "comparator"
        assert result.parsed["value"] == op

    def test_flatten_led(self):
        result = Parser().parse("foo[]")
        assert result.parsed["type"] == "projection"

    def test_lbracket_led_index_extends(self):
        result = Parser().parse("foo[0][1]")
        assert result.parsed["type"] == "index_expression"
        # optimization merges both index nodes into the same index_expression
        children = result.parsed["children"]
        assert len(children) == 3
        assert children[0] == {"type": "field", "children": [], "value": "foo"}
        assert children[1] == {"type": "index", "value": 0, "children": []}
        assert children[2] == {"type": "index", "value": 1, "children": []}

    def test_lbracket_led_star(self):
        result = Parser().parse("foo[*].bar")
        assert result.parsed["type"] == "projection"


class TestSliceExpression:
    def test_slice_all_colons(self):
        result = Parser().parse("[::2]")
        assert result.parsed["type"] == "projection"

    def test_slice_too_many_colons(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("[:::]")

    def test_slice_invalid_token(self):
        with pytest.raises(exceptions.ParseError):
            Parser().parse("[1:foo]")

    def test_index_expression(self):
        result = Parser().parse("[1]")
        assert result.parsed["type"] == "index_expression"


class TestErrors:
    def test_unexpected_token(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo bar")
        assert "Unexpected token" in str(exc.value)

    def test_invalid_nud_token(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse(")")
        assert "invalid token" in str(exc.value)

    def test_eof_incomplete(self):
        # The dot right-hand-side with eof raises a ParseError, not
        # IncompleteExpressionError.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.")
        assert "Expecting:" in str(exc.value)

    def test_match_eof(self):
        with pytest.raises(exceptions.IncompleteExpressionError):
            Parser().parse("foo[")

    def test_dot_rhs_invalid(self):
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("foo.[?a]")
        assert "Expecting:" in str(exc.value)

    def test_error_led_token(self):
        # 'foo bar' -> after 'foo' the 'bar' token tries led dispatch
        with pytest.raises(exceptions.ParseError):
            Parser().parse("foo.bar baz")


class TestErrorLedTokenDetails:
    """Directly exercise _error_led_token and verify the exception fields
    are derived from the offending token (not None / not dropped).

    Note: ParseError normalizes token_type to uppercase in its msg/attribute,
    so we assert on the upper-cased form.
    """

    def test_error_led_token_raises_with_token_details(self):
        p = Parser()
        token = {"type": "rbracket", "value": "]", "start": 7}
        with pytest.raises(exceptions.ParseError) as exc:
            p._error_led_token(token)
        err = exc.value
        assert err.lex_position == 7
        assert err.token_value == "]"
        # ParseError upper-cases the token type.
        assert err.token_type == "RBRACKET"
        assert err.msg == "invalid token"

    def test_error_led_token_via_parse(self):
        # 'a {b: c}' -> 'lbrace' bound as led from 'a', raises invalid token.
        with pytest.raises(exceptions.ParseError) as exc:
            Parser().parse("a {b: c}")
        err = exc.value
        assert err.msg == "invalid token"
        # ParseError upper-cases the token type.
        assert err.token_type == "LBRACE"
        # lex position should match the '{' location, which is index 2.
        assert err.lex_position == 2


class TestMatchHelpers:
    def test_match_and_advance(self):
        p = Parser()
        p._tokens = [{"type": "foo", "value": None, "start": 0},
                     {"type": "eof", "value": None, "start": 0}]
        p._index = 0
        p._match("foo")
        assert p._index == 1

    def test_match_failure(self):
        p = Parser()
        p._tokens = [{"type": "foo", "value": None, "start": 0}]
        p._index = 0
        with pytest.raises(exceptions.ParseError):
            p._match("bar")

    def test_match_multiple_tokens_advances(self):
        p = Parser()
        p._tokens = [{"type": "unquoted_identifier", "value": None, "start": 0}]
        p._index = 0
        p._match_multiple_tokens(["quoted_identifier", "unquoted_identifier"])
        assert p._index == 1


class TestParsedResult:
    def test_search(self):
        result = Parser().parse("foo")
        assert result.search({"foo": "bar"}) == "bar"

    def test_search_missing(self):
        result = Parser().parse("foo")
        assert result.search({"other": "bar"}) is None

    def test_search_with_options(self):
        result = Parser().parse("b")
        from jmespath import visitor as v
        # Options is any object; the visitor uses options.visit if provided.
        class Opts(v.Options):
            pass
        assert result.search({"b": 1}, Opts()) == 1

    def test_repr(self):
        result = Parser().parse("foo")
        assert repr(result) == repr(result.parsed)

    def test_expression_attribute(self):
        result = Parser().parse("foo")
        assert result.expression == "foo"

    def test_search_literal(self):
        result = Parser().parse("`true`")
        assert result.search({}) is True

    def test_search_pipe(self):
        result = Parser().parse("foo | bar")
        assert result.search({"foo": {"bar": 42}}) == 42

    def test_render_dot_file(self):
        result = Parser().parse("foo")
        contents = result._render_dot_file()
        assert "digraph" in contents
        assert "foo" in contents

    def test_init_stores_expression_and_parsed(self):
        node = {"type": "raw", "children": [], "value": 1}
        pr = ParsedResult("expr-string", node)
        assert pr.expression == "expr-string"
        assert pr.parsed is node
