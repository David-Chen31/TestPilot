import pytest

from jmespath import functions
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


# ---------------------------------------------------------------------------
# Helper module-level functions
# ---------------------------------------------------------------------------

def test_is_actual_number_rejects_bool():
    assert _is_actual_number(True) is False
    assert _is_actual_number(False) is False


def test_is_actual_number_accepts_numbers():
    assert _is_actual_number(1) is True
    assert _is_actual_number(1.5) is True
    assert _is_actual_number(0) is True


def test_is_actual_number_rejects_non_numbers():
    assert _is_actual_number("1") is False
    assert _is_actual_number(None) is False
    assert _is_actual_number([1]) is False


def test_is_comparable_numbers_and_strings():
    assert _is_comparable(1) is True
    assert _is_comparable(1.5) is True
    assert _is_comparable("foo") is True
    assert _is_comparable(True) is False
    assert _is_comparable(None) is False
    assert _is_comparable([]) is False


def test_is_special_number_case_zero_one_vs_bool():
    # Number 0/1 vs bool -> special case True
    assert _is_special_number_case(0, True) is True
    assert _is_special_number_case(1, False) is True
    assert _is_special_number_case(True, 0) is True
    assert _is_special_number_case(False, 1) is True
    # Number 0/1 vs number -> no special case
    assert _is_special_number_case(0, 1) is None
    assert _is_special_number_case(1, 1) is None
    # Number other than 0/1 vs bool -> falls through (None)
    assert _is_special_number_case(2, True) is None
    # bool vs bool is not captured (bools are not actual numbers)
    assert _is_special_number_case(True, False) is None


def test_equals_basic():
    assert _equals(1, 1) is True
    assert _equals(1, 2) is False
    assert _equals("a", "a") is True
    assert _equals([1, 2], [1, 2]) is True
    assert _equals(None, None) is True


def test_equals_special_number_case_returns_false():
    # 0 == True normally in Python, but must be False in JMESPath.
    assert _equals(0, True) is False
    assert _equals(1, True) is False
    assert _equals(0, False) is False
    assert _equals(True, 0) is False
    assert _equals(False, 1) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def test_options_defaults():
    opts = Options()
    assert opts.dict_cls is None
    assert opts.custom_functions is None


def test_options_custom_values():
    opts = Options(dict_cls=dict, custom_functions="cf")
    assert opts.dict_cls is dict
    assert opts.custom_functions == "cf"


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class _RecordingInterpreter(object):
    def __init__(self):
        self.calls = []

    def visit(self, node, *args, **kwargs):
        self.calls.append((node, args, kwargs))
        return ("visited", node)


def test_expression_stores_attributes():
    interp = _RecordingInterpreter()
    expr = _Expression("expr", interp)
    assert expr.expression == "expr"
    assert expr.interpreter is interp


def test_expression_visit_delegates():
    interp = _RecordingInterpreter()
    expr = _Expression("expr", interp)
    result = expr.visit("node", 1, 2, key="val")
    assert result == ("visited", "node")
    assert interp.calls == [("node", (1, 2), {"key": "val"})]


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class _DummyVisitor(Visitor):
    def visit_foo(self, node, *args, **kwargs):
        return ("foo", node, args, kwargs)

    def default_visit(self, node, *args, **kwargs):
        return ("default", node, args, kwargs)


def test_visitor_dispatches_to_named_method():
    v = _DummyVisitor()
    node = {"type": "foo"}
    assert v.visit(node) == ("foo", node, (), {})


def test_visitor_dispatches_with_args():
    v = _DummyVisitor()
    node = {"type": "foo"}
    assert v.visit(node, 1, k=2) == ("foo", node, (1,), {"k": 2})


def test_visitor_falls_back_to_default():
    v = _DummyVisitor()
    node = {"type": "unknown"}
    assert v.visit(node) == ("default", node, (), {})


def test_visitor_caches_methods():
    v = _DummyVisitor()
    node = {"type": "foo"}
    v.visit(node)
    assert "foo" in v._method_cache


def test_base_visitor_default_visit_raises():
    v = Visitor()
    with pytest.raises(NotImplementedError):
        v.default_visit({"type": "x"})


def test_base_visitor_visit_unknown_raises():
    v = Visitor()
    with pytest.raises(NotImplementedError):
        v.visit({"type": "unknown"})


# ---------------------------------------------------------------------------
# TreeInterpreter: field / current / literal / identity
# ---------------------------------------------------------------------------

def test_visit_field_present():
    t = TreeInterpreter()
    assert t.visit({"type": "field", "value": "a"}, {"a": 1}) == 1


def test_visit_field_missing_returns_none():
    t = TreeInterpreter()
    assert t.visit({"type": "field", "value": "a"}, {"b": 1}) is None


def test_visit_field_non_dict_returns_none():
    t = TreeInterpreter()
    assert t.visit({"type": "field", "value": "a"}, [1, 2]) is None


def test_visit_current():
    t = TreeInterpreter()
    data = {"x": 1}
    assert t.visit({"type": "current"}, data) is data


def test_visit_identity():
    t = TreeInterpreter()
    data = [1, 2]
    assert t.visit({"type": "identity"}, data) is data


def test_visit_literal_returns_value():
    t = TreeInterpreter()
    assert t.visit({"type": "literal", "value": 42}, "ignored") == 42


def test_default_visit_raises_with_type():
    t = TreeInterpreter()
    with pytest.raises(NotImplementedError) as exc:
        t.visit({"type": "nonexistent"}, {})
    assert "nonexistent" in str(exc.value)


# ---------------------------------------------------------------------------
# TreeInterpreter: comparators
# ---------------------------------------------------------------------------

def _make_comparator(op, left_node, right_node):
    return {"type": "comparator", "value": op, "children": [left_node, right_node]}


def test_comparator_eq():
    t = TreeInterpreter()
    node = _make_comparator(
        "eq",
        {"type": "literal", "value": 1},
        {"type": "literal", "value": 1},
    )
    assert t.visit(node, None) is True


def test_comparator_eq_bool_number_special_case():
    t = TreeInterpreter()
    node = _make_comparator(
        "eq",
        {"type": "literal", "value": 1},
        {"type": "literal", "value": True},
    )
    assert t.visit(node, None) is False


def test_comparator_ne():
    t = TreeInterpreter()
    node = _make_comparator(
        "ne",
        {"type": "literal", "value": 1},
        {"type": "literal", "value": 2},
    )
    assert t.visit(node, None) is True


def test_comparator_lt_numbers():
    t = TreeInterpreter()
    node = _make_comparator(
        "lt",
        {"type": "literal", "value": 1},
        {"type": "literal", "value": 2},
    )
    assert t.visit(node, None) is True


def test_comparator_gt_numbers():
    t = TreeInterpreter()
    node = _make_comparator(
        "gt",
        {"type": "literal", "value": 3},
        {"type": "literal", "value": 2},
    )
    assert t.visit(node, None) is True


def test_comparator_lte():
    t = TreeInterpreter()
    node = _make_comparator(
        "lte",
        {"type": "literal", "value": 2},
        {"type": "literal", "value": 2},
    )
    assert t.visit(node, None) is True


def test_comparator_gte():
    t = TreeInterpreter()
    node = _make_comparator(
        "gte",
        {"type": "literal", "value": 2},
        {"type": "literal", "value": 3},
    )
    assert t.visit(node, None) is False


def test_comparator_ordering_strings():
    t = TreeInterpreter()
    node = _make_comparator(
        "lt",
        {"type": "literal", "value": "a"},
        {"type": "literal", "value": "b"},
    )
    assert t.visit(node, None) is True


def test_comparator_ordering_non_comparable_returns_none():
    t = TreeInterpreter()
    node = _make_comparator(
        "lt",
        {"type": "literal", "value": [1]},
        {"type": "literal", "value": [2]},
    )
    assert t.visit(node, None) is None


# ---------------------------------------------------------------------------
# TreeInterpreter: subexpression / pipe / index_expression
# ---------------------------------------------------------------------------

def test_visit_subexpression():
    t = TreeInterpreter()
    node = {
        "type": "subexpression",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "field", "value": "b"},
        ],
    }
    assert t.visit(node, {"a": {"b": 5}}) == 5


def test_visit_pipe():
    t = TreeInterpreter()
    node = {
        "type": "pipe",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "identity"},
        ],
    }
    assert t.visit(node, {"a": 10}) == 10


def test_visit_index_expression():
    t = TreeInterpreter()
    node = {
        "type": "index_expression",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "index", "value": 0},
        ],
    }
    assert t.visit(node, {"a": [7, 8]}) == 7


# ---------------------------------------------------------------------------
# TreeInterpreter: index / slice
# ---------------------------------------------------------------------------

def test_visit_index_valid():
    t = TreeInterpreter()
    assert t.visit({"type": "index", "value": 1}, [10, 20, 30]) == 20


def test_visit_index_negative():
    t = TreeInterpreter()
    assert t.visit({"type": "index", "value": -1}, [10, 20, 30]) == 30


def test_visit_index_out_of_range():
    t = TreeInterpreter()
    assert t.visit({"type": "index", "value": 5}, [1, 2]) is None


def test_visit_index_non_list():
    t = TreeInterpreter()
    assert t.visit({"type": "index", "value": 0}, "abc") is None


def test_visit_slice_basic():
    t = TreeInterpreter()
    node = {"type": "slice", "children": [1, 3, None]}
    assert t.visit(node, [0, 1, 2, 3, 4]) == [1, 2]


def test_visit_slice_with_step():
    t = TreeInterpreter()
    node = {"type": "slice", "children": [None, None, 2]}
    assert t.visit(node, [0, 1, 2, 3, 4]) == [0, 2, 4]


def test_visit_slice_non_list():
    t = TreeInterpreter()
    node = {"type": "slice", "children": [0, 1, None]}
    assert t.visit(node, "hello") is None


# ---------------------------------------------------------------------------
# TreeInterpreter: flatten
# ---------------------------------------------------------------------------

def test_visit_flatten_merges_nested_lists():
    t = TreeInterpreter()
    node = {"type": "flatten", "children": [{"type": "identity"}]}
    assert t.visit(node, [[1, 2], 3, [4]]) == [1, 2, 3, 4]


def test_visit_flatten_non_list_returns_none():
    t = TreeInterpreter()
    node = {"type": "flatten", "children": [{"type": "identity"}]}
    assert t.visit(node, {"a": 1}) is None


# ---------------------------------------------------------------------------
# TreeInterpreter: projections
# ---------------------------------------------------------------------------

def test_visit_projection():
    t = TreeInterpreter()
    node = {
        "type": "projection",
        "children": [
            {"type": "identity"},
            {"type": "field", "value": "a"},
        ],
    }
    result = t.visit(node, [{"a": 1}, {"a": 2}, {"b": 3}])
    assert result == [1, 2]


def test_visit_projection_non_list_returns_none():
    t = TreeInterpreter()
    node = {
        "type": "projection",
        "children": [
            {"type": "identity"},
            {"type": "identity"},
        ],
    }
    assert t.visit(node, {"a": 1}) is None


def test_visit_value_projection():
    t = TreeInterpreter()
    node = {
        "type": "value_projection",
        "children": [
            {"type": "identity"},
            {"type": "field", "value": "a"},
        ],
    }
    result = t.visit(node, {"x": {"a": 1}, "y": {"a": 2}})
    assert sorted(result) == [1, 2]


def test_visit_value_projection_non_dict_returns_none():
    t = TreeInterpreter()
    node = {
        "type": "value_projection",
        "children": [
            {"type": "identity"},
            {"type": "identity"},
        ],
    }
    assert t.visit(node, [1, 2]) is None


def test_visit_filter_projection():
    t = TreeInterpreter()
    node = {
        "type": "filter_projection",
        "children": [
            {"type": "identity"},
            {"type": "identity"},
            {"type": "identity"},
        ],
    }
    # Comparator identity => _is_true of each element; filters out falsey values.
    result = t.visit(node, [0, 1, "", "x", [], [1], None])
    assert result == [0, 1, "x", [1]]


def test_visit_filter_projection_non_list_returns_none():
    t = TreeInterpreter()
    node = {
        "type": "filter_projection",
        "children": [
            {"type": "identity"},
            {"type": "identity"},
            {"type": "identity"},
        ],
    }
    assert t.visit(node, {"a": 1}) is None


def test_visit_filter_projection_skips_none_current():
    t = TreeInterpreter()
    node = {
        "type": "filter_projection",
        "children": [
            {"type": "identity"},
            # field 'missing' -> None so should be skipped
            {"type": "field", "value": "missing"},
            {"type": "identity"},
        ],
    }
    result = t.visit(node, [{"a": 1}, {"b": 2}])
    assert result == []


# ---------------------------------------------------------------------------
# TreeInterpreter: multi-select
# ---------------------------------------------------------------------------

def test_visit_multi_select_dict():
    t = TreeInterpreter()
    node = {
        "type": "multi_select_dict",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "field", "value": "b"},
        ],
    }
    result = t.visit(node, {"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}


def test_visit_multi_select_dict_none_value():
    t = TreeInterpreter()
    node = {"type": "multi_select_dict", "children": []}
    assert t.visit(node, None) is None


def test_visit_multi_select_dict_custom_dict_cls():
    from collections import OrderedDict
    t = TreeInterpreter(Options(dict_cls=OrderedDict))
    node = {
        "type": "multi_select_dict",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "field", "value": "b"},
        ],
    }
    result = t.visit(node, {"a": 1, "b": 2})
    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == ["a", "b"]


def test_visit_multi_select_list():
    t = TreeInterpreter()
    node = {
        "type": "multi_select_list",
        "children": [
            {"type": "field", "value": "a"},
            {"type": "field", "value": "b"},
        ],
    }
    result = t.visit(node, {"a": 1, "b": 2})
    assert result == [1, 2]


def test_visit_multi_select_list_none_value():
    t = TreeInterpreter()
    node = {"type": "multi_select_list", "children": []}
    assert t.visit(node, None) is None


# ---------------------------------------------------------------------------
# TreeInterpreter: or / and / not
# ---------------------------------------------------------------------------

def test_visit_or_expression_left_truthy():
    t = TreeInterpreter()
    node = {
        "type": "or_expression",
        "children": [
            {"type": "literal", "value": "first"},
            {"type": "literal", "value": "second"},
        ],
    }
    assert t.visit(node, None) == "first"


def test_visit_or_expression_left_falsey():
    t = TreeInterpreter()
    node = {
        "type": "or_expression",
        "children": [
            {"type": "literal", "value": ""},
            {"type": "literal", "value": "second"},
        ],
    }
    assert t.visit(node, None) == "second"


def test_visit_and_expression_left_falsey_returns_it():
    t = TreeInterpreter()
    node = {
        "type": "and_expression",
        "children": [
            {"type": "literal", "value": []},
            {"type": "literal", "value": "second"},
        ],
    }
    assert t.visit(node, None) == []


def test_visit_and_expression_left_truthy_returns_right():
    t = TreeInterpreter()
    node = {
        "type": "and_expression",
        "children": [
            {"type": "literal", "value": "yes"},
            {"type": "literal", "value": "second"},
        ],
    }
    assert t.visit(node, None) == "second"


def test_visit_not_expression_zero_special_case():
    t = TreeInterpreter()
    node = {"type": "not_expression", "children": [{"type": "literal", "value": 0}]}
    assert t.visit(node, None) is False


def test_visit_not_expression_truthy():
    t = TreeInterpreter()
    node = {"type": "not_expression", "children": [{"type": "literal", "value": "x"}]}
    assert t.visit(node, None) is False


def test_visit_not_expression_falsey():
    t = TreeInterpreter()
    node = {"type": "not_expression", "children": [{"type": "literal", "value": ""}]}
    assert t.visit(node, None) is True


# ---------------------------------------------------------------------------
# TreeInterpreter: expref / function_expression
# ---------------------------------------------------------------------------

def test_visit_expref_returns_expression():
    t = TreeInterpreter()
    child = {"type": "identity"}
    node = {"type": "expref", "children": [child]}
    result = t.visit(node, None)
    assert isinstance(result, _Expression)
    assert result.expression is child
    assert result.interpreter is t


def test_visit_function_expression_calls_functions():
    t = TreeInterpreter()
    node = {
        "type": "function_expression",
        "value": "length",
        "children": [{"type": "literal", "value": [1, 2, 3]}],
    }
    assert t.visit(node, None) == 3


def test_visit_function_expression_custom_functions():
    class CustomFunctions(object):
        def __init__(self):
            self.called = False

        def call_function(self, name, args):
            self.called = True
            return (name, args)

    custom = CustomFunctions()
    t = TreeInterpreter(Options(custom_functions=custom))
    node = {
        "type": "function_expression",
        "value": "myfn",
        "children": [{"type": "literal", "value": 5}],
    }
    result = t.visit(node, None)
    assert result == ("myfn", [5])
    assert custom.called is True


# ---------------------------------------------------------------------------
# TreeInterpreter: key_val_pair
# ---------------------------------------------------------------------------

def test_visit_key_val_pair():
    t = TreeInterpreter()
    node = {
        "type": "key_val_pair",
        "value": "name",
        "children": [{"type": "field", "value": "a"}],
    }
    assert t.visit(node, {"a": 99}) == 99


# ---------------------------------------------------------------------------
# TreeInterpreter: internals
# ---------------------------------------------------------------------------

def test_interpreter_uses_default_functions():
    t = TreeInterpreter()
    assert isinstance(t._functions, functions.Functions)


def test_interpreter_uses_default_dict_cls():
    t = TreeInterpreter()
    assert t._dict_cls is dict


def test_interpreter_options_dict_cls():
    from collections import OrderedDict
    t = TreeInterpreter(Options(dict_cls=OrderedDict))
    assert t._dict_cls is OrderedDict


def test_interpreter_is_false_checks():
    t = TreeInterpreter()
    assert t._is_false("") is True
    assert t._is_false([]) is True
    assert t._is_false({}) is True
    assert t._is_false(None) is True
    assert t._is_false(False) is True
    assert t._is_false(0) is False  # 0 is not falsey in JMESPath
    assert t._is_false("x") is False


def test_interpreter_is_true_checks():
    t = TreeInterpreter()
    assert t._is_true(0) is True
    assert t._is_true("x") is True
    assert t._is_true("") is False
    assert t._is_true(None) is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

def test_graphviz_simple_identity():
    g = GraphvizVisitor()
    result = g.visit({"type": "identity"})
    assert isinstance(result, str)
    assert result.startswith("digraph AST {")
    assert result.endswith("}")
    assert "identity1" in result


def test_graphviz_with_children():
    g = GraphvizVisitor()
    node = {
        "type": "field",
        "value": "name",
        "children": [
            {"type": "identity"},
            {"type": "literal", "value": 5},
        ],
    }
    result = g.visit(node)
    assert "field1" in result
    assert "identity2" in result
    assert "literal3" in result
    assert "field1 -> identity2" in result
    assert "field1 -> literal3" in result
    assert 'label="field(name)"' in result


def test_graphviz_accumulates_lines():
    g = GraphvizVisitor()
    g.visit({"type": "identity"})
    lines_after_first = len(g._lines)
    g.visit({"type": "identity"})
    # Lines accumulate across calls (part of current behavior).
    assert len(g._lines) > lines_after_first


def test_graphviz_initial_state():
    g = GraphvizVisitor()
    assert g._lines == []
    assert g._count == 1
