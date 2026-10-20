import pytest

from jmespath.visitor import (
    Options,
    Visitor,
    TreeInterpreter,
    GraphvizVisitor,
    _Expression,
    _equals,
    _is_special_number_case,
    _is_comparable,
    _is_actual_number,
)
from jmespath import functions


# ---------------------------------------------------------------------------
# Helper predicate functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_is_actual_number(self):
        assert _is_actual_number(1) is True
        assert _is_actual_number(1.5) is True
        assert _is_actual_number(True) is False
        assert _is_actual_number(False) is False
        assert _is_actual_number("1") is False
        assert _is_actual_number(None) is False

    def test_is_comparable(self):
        assert _is_comparable(1) is True
        assert _is_comparable("abc") is True
        assert _is_comparable(True) is False
        assert _is_comparable(None) is False
        assert _is_comparable([]) is False

    def test_is_special_number_case_true(self):
        assert _is_special_number_case(0, True) is True
        assert _is_special_number_case(1, False) is True
        assert _is_special_number_case(True, 0) is True
        assert _is_special_number_case(False, 1) is True

    def test_is_special_number_case_false(self):
        assert _is_special_number_case(2, True) is None
        assert _is_special_number_case(0, 0) is None
        assert _is_special_number_case("a", True) is None
        assert _is_special_number_case(True, True) is None

    def test_equals(self):
        assert _equals(1, 1) is True
        assert _equals(1, 2) is False
        # Special number case should be False even though Python says True
        assert _equals(0, False) is False
        assert _equals(1, True) is False
        assert _equals(True, 1) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults(self):
        opts = Options()
        assert opts.dict_cls is None
        assert opts.custom_functions is None

    def test_custom_values(self):
        sentinel_dict = object()
        sentinel_funcs = object()
        opts = Options(dict_cls=sentinel_dict, custom_functions=sentinel_funcs)
        assert opts.dict_cls is sentinel_dict
        assert opts.custom_functions is sentinel_funcs


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        calls = []

        class FakeInterpreter(object):
            def visit(self, node, *args, **kwargs):
                calls.append((node, args, kwargs))
                return "result"

        expr = _Expression("expr", FakeInterpreter())
        assert expr.expression == "expr"
        assert expr.visit("node", 1, kwarg="x") == "result"
        assert calls == [("node", (1,), {"kwarg": "x"})]


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError, match="default_visit"):
            v.visit({"type": "unknown"})

    def test_visit_dispatches_to_method(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, value):
                return ("foo", node, value)

        v = MyVisitor()
        node = {"type": "foo"}
        assert v.visit(node, 42) == ("foo", node, 42)

    def test_visit_caches_method(self):
        class MyVisitor(Visitor):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def visit_foo(self, node, value):
                self.calls += 1
                return value

        v = MyVisitor()
        v.visit({"type": "foo"}, 1)
        v.visit({"type": "foo"}, 2)
        assert v.calls == 2
        assert "foo" in v._method_cache


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

def _node(node_type, **kwargs):
    node = {"type": node_type}
    node.update(kwargs)
    return node


class TestTreeInterpreterInit:
    def test_defaults(self):
        t = TreeInterpreter()
        assert t._dict_cls is dict
        assert isinstance(t._functions, functions.Functions)

    def test_custom_dict_cls(self):
        from collections import OrderedDict
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert t._dict_cls is OrderedDict

    def test_custom_functions(self):
        sentinel = object()
        t = TreeInterpreter(Options(custom_functions=sentinel))
        assert t._functions is sentinel

    def test_default_visit_raises(self):
        t = TreeInterpreter()
        with pytest.raises(NotImplementedError, match="unknown"):
            t.visit({"type": "unknown"})


class TestTreeInterpreterVisit:
    def setup_method(self):
        self.t = TreeInterpreter()

    def test_visit_subexpression(self):
        node = _node("subexpression", children=[
            _node("identity"), _node("identity")
        ])
        assert self.t.visit(node, "v") == "v"

    def test_visit_field_dict(self):
        node = _node("field", value="a")
        assert self.t.visit(node, {"a": 1}) == 1
        assert self.t.visit(node, {"a": None}) is None

    def test_visit_field_missing(self):
        assert self.t.visit(_node("field", value="a"), {"b": 1}) is None

    def test_visit_field_non_dict(self):
        assert self.t.visit(_node("field", value="a"), "string") is None
        assert self.t.visit(_node("field", value="a"), None) is None

    def test_visit_current(self):
        assert self.t.visit(_node("current"), "value") == "value"

    def test_visit_identity(self):
        assert self.t.visit(_node("identity"), [1, 2]) == [1, 2]

    def test_visit_literal(self):
        assert self.t.visit(_node("literal", value=[1, 2]), "x") == [1, 2]

    def test_visit_index(self):
        node = _node("index", value=0)
        assert self.t.visit(node, [10, 20]) == 10
        assert self.t.visit(node, []) is None
        assert self.t.visit(node, "abc") is None

    def test_visit_index_expression(self):
        node = _node("index_expression", children=[
            _node("index", value=0),
            _node("index", value=1),
        ])
        assert self.t.visit(node, [[1, 2], [3, 4]]) == 2

    def test_visit_slice(self):
        node = _node("slice", children=[0, 2, None])
        assert self.t.visit(node, [1, 2, 3, 4]) == [1, 2]
        assert self.t.visit(node, "abcd") is None

    def test_visit_key_val_pair(self):
        node = _node("key_val_pair", children=[
            _node("literal", value="k")
        ])
        assert self.t.visit(node, "value") == "k"

    def test_visit_multi_select_dict(self):
        node = _node("multi_select_dict", children=[
            _node("field", value="a"),
            _node("field", value="b"),
        ])
        assert self.t.visit(node, {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_visit_multi_select_dict_none(self):
        node = _node("multi_select_dict", children=[])
        assert self.t.visit(node, None) is None

    def test_visit_multi_select_list(self):
        node = _node("multi_select_list", children=[
            _node("field", value="a"),
            _node("field", value="b"),
        ])
        assert self.t.visit(node, {"a": 1, "b": 2}) == [1, 2]

    def test_visit_multi_select_list_none(self):
        node = _node("multi_select_list", children=[])
        assert self.t.visit(node, None) is None

    def test_visit_expref(self):
        inner = _node("identity")
        node = _node("expref", children=[inner])
        result = self.t.visit(node, "value")
        assert isinstance(result, _Expression)
        assert result.expression is inner
        assert result.interpreter is self.t

    def test_visit_function_expression(self):
        node = _node("function_expression", value="length",
                     children=[_node("literal", value=[1, 2, 3])])
        assert self.t.visit(node, None) == 3

    def test_visit_function_expression_custom(self):
        class CustomFuncs(functions.Functions):
            def call_function(self, name, args):
                return ("custom", name, args)

        t = TreeInterpreter(Options(custom_functions=CustomFuncs()))
        node = _node("function_expression", value="f",
                     children=[_node("literal", value=1)])
        assert t.visit(node, None) == ("custom", "f", [1])

    def test_visit_comparator_eq(self):
        node = _node("comparator", value="eq", children=[
            _node("literal", value=1), _node("literal", value=1),
        ])
        assert self.t.visit(node, None) is True

    def test_visit_comparator_eq_special_bool(self):
        node = _node("comparator", value="eq", children=[
            _node("literal", value=1), _node("literal", value=True),
        ])
        assert self.t.visit(node, None) is False

    def test_visit_comparator_ne(self):
        node = _node("comparator", value="ne", children=[
            _node("literal", value=1), _node("literal", value=2),
        ])
        assert self.t.visit(node, None) is True

    def test_visit_comparator_ordering(self):
        node = _node("comparator", value="lt", children=[
            _node("literal", value=1), _node("literal", value=2),
        ])
        assert self.t.visit(node, None) is True

    def test_visit_comparator_ordering_strings(self):
        node = _node("comparator", value="lte", children=[
            _node("literal", value="a"), _node("literal", value="b"),
        ])
        assert self.t.visit(node, None) is True

    def test_visit_comparator_ordering_incomparable(self):
        node = _node("comparator", value="lt", children=[
            _node("literal", value=[1]), _node("literal", value=[2]),
        ])
        assert self.t.visit(node, None) is None

    def test_visit_filter_projection(self):
        base = [{"a": 1}, {"a": 2}, {"a": 3}]
        node = _node("filter_projection", children=[
            _node("identity"),
            _node("field", value="a"),
            _node("comparator", value="gt", children=[
                _node("field", value="a"),
                _node("literal", value=1),
            ]),
        ])
        assert self.t.visit(node, base) == [2, 3]

    def test_visit_filter_projection_non_list(self):
        node = _node("filter_projection", children=[
            _node("identity"),
            _node("identity"),
            _node("identity"),
        ])
        assert self.t.visit(node, {"a": 1}) is None

    def test_visit_flatten(self):
        node = _node("flatten", children=[_node("identity")])
        assert self.t.visit(node, [[1, 2], 3, [4]]) == [1, 2, 3, 4]

    def test_visit_flatten_non_list(self):
        node = _node("flatten", children=[_node("identity")])
        assert self.t.visit(node, "abc") is None

    def test_visit_or_expression_true(self):
        node = _node("or_expression", children=[
            _node("literal", value="first"),
            _node("literal", value="second"),
        ])
        assert self.t.visit(node, None) == "first"

    def test_visit_or_expression_false(self):
        node = _node("or_expression", children=[
            _node("literal", value=""),
            _node("literal", value="second"),
        ])
        assert self.t.visit(node, None) == "second"

    def test_visit_and_expression_true(self):
        node = _node("and_expression", children=[
            _node("literal", value="first"),
            _node("literal", value="second"),
        ])
        assert self.t.visit(node, None) == "second"

    def test_visit_and_expression_false(self):
        node = _node("and_expression", children=[
            _node("literal", value=""),
            _node("literal", value="second"),
        ])
        assert self.t.visit(node, None) == ""

    def test_visit_not_expression(self):
        node = _node("not_expression", children=[_node("literal", value=True)])
        assert self.t.visit(node, None) is False

    def test_visit_not_expression_zero(self):
        node = _node("not_expression", children=[_node("literal", value=0)])
        assert self.t.visit(node, None) is False

    def test_visit_not_expression_nonzero(self):
        node = _node("not_expression", children=[_node("literal", value=1)])
        assert self.t.visit(node, None) is False

    def test_visit_pipe(self):
        node = _node("pipe", children=[
            _node("identity"), _node("identity"),
        ])
        assert self.t.visit(node, "value") == "value"

    def test_visit_projection(self):
        node = _node("projection", children=[
            _node("identity"),
            _node("field", value="a"),
        ])
        base = [{"a": 1}, {"b": 2}, {"a": 3}]
        assert self.t.visit(node, base) == [1, 3]

    def test_visit_projection_non_list(self):
        node = _node("projection", children=[
            _node("identity"), _node("identity"),
        ])
        assert self.t.visit(node, {"a": 1}) is None

    def test_visit_value_projection(self):
        node = _node("value_projection", children=[
            _node("identity"),
            _node("field", value="a"),
        ])
        base = {"x": {"a": 1}, "y": {"b": 2}, "z": {"a": 3}}
        assert sorted(self.t.visit(node, base)) == [1, 3]

    def test_visit_value_projection_non_dict(self):
        node = _node("value_projection", children=[
            _node("identity"), _node("identity"),
        ])
        assert self.t.visit(node, [1, 2]) is None

    def test_is_false(self):
        assert self.t._is_false("") is True
        assert self.t._is_false([]) is True
        assert self.t._is_false({}) is True
        assert self.t._is_false(None) is True
        assert self.t._is_false(False) is True
        assert self.t._is_false(0) is False
        assert self.t._is_false("a") is False

    def test_is_true(self):
        assert self.t._is_true("a") is True
        assert self.t._is_true(0) is True
        assert self.t._is_true("") is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_simple_graph(self):
        g = GraphvizVisitor()
        node = {"type": "identity"}
        result = g.visit(node)
        lines = result.split("\n")
        assert lines[0] == "digraph AST {"
        assert lines[-1] == "}"
        assert 'identity1 [label="identity()"]' in lines

    def test_graph_with_children(self):
        g = GraphvizVisitor()
        node = {
            "type": "field",
            "value": "foo",
            "children": [
                {"type": "identity"},
            ],
        }
        result = g.visit(node)
        lines = result.split("\n")
        assert 'field1 [label="field(foo)"]' in lines
        assert "field1 -> identity2" in lines
        assert any('identity2 [label="identity()"]' in l for l in lines)

    def test_graph_counter_increments(self):
        g = GraphvizVisitor()
        node = {
            "type": "root",
            "children": [
                {"type": "child"},
                {"type": "child2"},
            ],
        }
        result = g.visit(node)
        # root1, child2, child3
        assert "root1 -> child2" in result
        assert "root1 -> child23" in result
        assert g._count == 4
