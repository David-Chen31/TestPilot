import operator

import pytest

from jmespath.visitor import (
    Options,
    Visitor,
    TreeInterpreter,
    GraphvizVisitor,
    _Expression,
    _equals,
    _is_actual_number,
    _is_comparable,
    _is_special_number_case,
)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def test_equals_basic():
    assert _equals(1, 1) is True
    assert _equals(1, 2) is False
    assert _equals("a", "a") is True
    assert _equals([1, 2], [1, 2]) is True


def test_equals_special_number_case_bool_vs_int():
    # 0 == False and 1 == True in Python, but JMESPath treats them unequal.
    assert _equals(0, False) is False
    assert _equals(1, True) is False
    assert _equals(False, 0) is False
    assert _equals(True, 1) is False


def test_equals_bools_equal_themselves():
    assert _equals(True, True) is True
    assert _equals(False, False) is True
    assert _equals(True, False) is False


def test_equals_non_special_numbers():
    # Values other than 0/1 compare normally with bools (always unequal).
    assert _equals(2, True) is False
    assert _equals(-1, False) is False


def test_is_special_number_case():
    assert _is_special_number_case(0, False) is True
    assert _is_special_number_case(1, True) is True
    assert _is_special_number_case(False, 0) is True
    assert _is_special_number_case(True, 1) is True
    # numbers that are not 0/1 do not trigger the special case
    assert _is_special_number_case(2, True) is None
    assert _is_special_number_case(2, 3) is None
    assert _is_special_number_case("a", True) is None
    # 0/1 compared to non-bools: first branch hits isinstance(y, bool) -> False
    assert _is_special_number_case(0, 0) is False
    assert _is_special_number_case(1, 1) is False
    assert _is_special_number_case(0, 5) is False
    # A float equal to 0/1 is treated as an actual number too.
    assert _is_special_number_case(1, 1.0) is False
    assert _is_special_number_case(1.0, True) is True


def test_is_actual_number():
    assert _is_actual_number(1) is True
    assert _is_actual_number(1.5) is True
    assert _is_actual_number(True) is False
    assert _is_actual_number(False) is False
    assert _is_actual_number("1") is False
    assert _is_actual_number([1]) is False
    assert _is_actual_number(None) is False


def test_is_comparable():
    assert _is_comparable(1) is True
    assert _is_comparable(1.5) is True
    assert _is_comparable("abc") is True
    assert _is_comparable(True) is False
    assert _is_comparable([1]) is False
    assert _is_comparable(None) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def test_options_defaults():
    options = Options()
    assert options.dict_cls is None
    assert options.custom_functions is None


def test_options_values():
    from collections import OrderedDict

    def custom():  # pragma: no cover - not called
        pass

    options = Options(dict_cls=OrderedDict, custom_functions=custom)
    assert options.dict_cls is OrderedDict
    assert options.custom_functions is custom


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

def test_expression_visit_delegates():
    class FakeInterpreter:
        def visit(self, node, *args, **kwargs):
            return (node, args, kwargs)

    expr = _Expression({'type': 'x'}, FakeInterpreter())
    assert expr.expression == {'type': 'x'}
    assert expr.visit('node', 1, a=2) == ('node', (1,), {'a': 2})


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

def test_visitor_default_visit_raises():
    v = Visitor()
    with pytest.raises(NotImplementedError, match="default_visit"):
        v.visit({'type': 'unknown'}, 'value')


def test_visitor_dispatches_to_named_method():
    class MyVisitor(Visitor):
        def visit_foo(self, node, value):
            return ('foo', value)

    v = MyVisitor()
    assert v.visit({'type': 'foo'}, 42) == ('foo', 42)


def test_visitor_method_cache_populated():
    class MyVisitor(Visitor):
        def visit_foo(self, node, value):
            return value

    v = MyVisitor()
    v.visit({'type': 'foo'}, 1)
    assert 'foo' in v._method_cache


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

@pytest.fixture
def interp():
    return TreeInterpreter()


def test_tree_interpreter_default_dict_cls(interp):
    assert interp._dict_cls is dict


def test_tree_interpreter_options_dict_cls():
    from collections import OrderedDict

    interp = TreeInterpreter(Options(dict_cls=OrderedDict))
    assert interp._dict_cls is OrderedDict


def test_tree_interpreter_custom_functions():
    class FakeFunctions:
        def call_function(self, name, args):
            return (name, args)

    fake = FakeFunctions()
    interp = TreeInterpreter(Options(custom_functions=fake))
    assert interp._functions is fake


def test_tree_interpreter_default_visit_raises(interp):
    with pytest.raises(NotImplementedError, match="bogus"):
        interp.visit({'type': 'bogus'}, {})


def test_visit_subexpression(interp):
    node = {
        'type': 'subexpression',
        'children': [
            {'type': 'field', 'value': 'a'},
            {'type': 'field', 'value': 'b'},
        ],
    }
    assert interp.visit(node, {'a': {'b': 5}}) == 5


def test_visit_field(interp):
    assert interp.visit({'type': 'field', 'value': 'a'}, {'a': 1}) == 1
    assert interp.visit({'type': 'field', 'value': 'x'}, {'a': 1}) is None
    # Non-dict value returns None via AttributeError branch.
    assert interp.visit({'type': 'field', 'value': 'a'}, [1, 2]) is None
    assert interp.visit({'type': 'field', 'value': 'a'}, None) is None


def _cnode(op, left, right):
    return {
        'type': 'comparator',
        'value': op,
        'children': [left, right],
    }


def test_visit_comparator_eq(interp):
    left = {'type': 'literal', 'value': 1}
    right = {'type': 'literal', 'value': 1}
    assert interp.visit(_cnode('eq', left, right), {}) is True


def test_visit_comparator_ne(interp):
    left = {'type': 'literal', 'value': 1}
    right = {'type': 'literal', 'value': 2}
    assert interp.visit(_cnode('ne', left, right), {}) is True


def test_visit_comparator_ordering_numbers(interp):
    l = {'type': 'literal', 'value': 1}
    r = {'type': 'literal', 'value': 2}
    assert interp.visit(_cnode('lt', l, r), {}) is True
    assert interp.visit(_cnode('gt', l, r), {}) is False
    assert interp.visit(_cnode('lte', l, r), {}) is True
    assert interp.visit(_cnode('gte', l, r), {}) is False


def test_visit_comparator_ordering_strings(interp):
    l = {'type': 'literal', 'value': 'a'}
    r = {'type': 'literal', 'value': 'b'}
    assert interp.visit(_cnode('lt', l, r), {}) is True


def test_visit_comparator_ordering_incomparable_returns_none(interp):
    l = {'type': 'literal', 'value': [1]}
    r = {'type': 'literal', 'value': 2}
    assert interp.visit(_cnode('lt', l, r), {}) is None


def test_visit_current_and_identity(interp):
    assert interp.visit({'type': 'current'}, 'v') == 'v'
    assert interp.visit({'type': 'identity'}, 'v') == 'v'


def test_visit_expref(interp):
    child = {'type': 'field', 'value': 'a'}
    result = interp.visit({'type': 'expref', 'children': [child]}, {})
    assert isinstance(result, _Expression)
    assert result.expression is child
    assert result.interpreter is interp


def test_visit_function_expression(interp):
    node = {
        'type': 'function_expression',
        'value': 'length',
        'children': [{'type': 'literal', 'value': [1, 2, 3]}],
    }
    assert interp.visit(node, {}) == 3


def test_visit_filter_projection(interp):
    node = {
        'type': 'filter_projection',
        'children': [
            {'type': 'current'},
            {'type': 'identity'},
            _cnode('eq',
                   {'type': 'current'},
                   {'type': 'literal', 'value': 2}),
        ],
    }
    value = [1, 2, 3, 2]
    assert interp.visit(node, value) == [2, 2]


def test_visit_filter_projection_non_list(interp):
    node = {
        'type': 'filter_projection',
        'children': [
            {'type': 'current'},
            {'type': 'identity'},
            {'type': 'identity'},
        ],
    }
    assert interp.visit(node, {'a': 1}) is None


def test_visit_filter_projection_skips_none_results(interp):
    node = {
        'type': 'filter_projection',
        'children': [
            {'type': 'current'},
            {'type': 'field', 'value': 'b'},
            {'type': 'identity'},
        ],
    }
    value = [{'b': 1}, {'a': 2}, {'b': 3}]
    assert interp.visit(node, value) == [1, 3]


def test_visit_flatten(interp):
    node = {
        'type': 'flatten',
        'children': [{'type': 'current'}],
    }
    assert interp.visit(node, [1, [2, 3], 4]) == [1, 2, 3, 4]


def test_visit_flatten_non_list(interp):
    node = {
        'type': 'flatten',
        'children': [{'type': 'current'}],
    }
    assert interp.visit(node, {'a': 1}) is None


def test_visit_index(interp):
    assert interp.visit({'type': 'index', 'value': 0}, [10, 20]) == 10
    assert interp.visit({'type': 'index', 'value': -1}, [10, 20]) == 20
    assert interp.visit({'type': 'index', 'value': 5}, [10, 20]) is None
    assert interp.visit({'type': 'index', 'value': 0}, "abc") is None


def test_visit_index_expression(interp):
    node = {
        'type': 'index_expression',
        'children': [
            {'type': 'field', 'value': 'a'},
            {'type': 'index', 'value': 1},
        ],
    }
    assert interp.visit(node, {'a': [10, 20, 30]}) == 20


def test_visit_slice(interp):
    node = {'type': 'slice', 'children': [0, 4, 2]}
    assert interp.visit(node, [0, 1, 2, 3, 4, 5]) == [0, 2]
    assert interp.visit(node, "abcdef") is None


def test_visit_key_val_pair(interp):
    node = {
        'type': 'key_val_pair',
        'children': [{'type': 'literal', 'value': 'value'}],
    }
    assert interp.visit(node, 'anything') == 'value'


def test_visit_literal(interp):
    assert interp.visit({'type': 'literal', 'value': {'x': 1}}, 'ignored') == {'x': 1}


def test_visit_multi_select_dict(interp):
    node = {
        'type': 'multi_select_dict',
        'children': [
            {'type': 'field', 'value': 'a'},
            {'type': 'field', 'value': 'b'},
        ],
    }
    assert interp.visit(node, {'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def test_visit_multi_select_dict_none(interp):
    node = {'type': 'multi_select_dict', 'children': []}
    assert interp.visit(node, None) is None


def test_visit_multi_select_list(interp):
    node = {
        'type': 'multi_select_list',
        'children': [
            {'type': 'field', 'value': 'a'},
            {'type': 'field', 'value': 'b'},
        ],
    }
    assert interp.visit(node, {'a': 1, 'b': 2}) == [1, 2]


def test_visit_multi_select_list_none(interp):
    node = {'type': 'multi_select_list', 'children': []}
    assert interp.visit(node, None) is None


def test_visit_or_expression(interp):
    node = {
        'type': 'or_expression',
        'children': [
            {'type': 'literal', 'value': None},
            {'type': 'literal', 'value': 'default'},
        ],
    }
    assert interp.visit(node, {}) == 'default'


def test_visit_or_expression_first_truthy(interp):
    node = {
        'type': 'or_expression',
        'children': [
            {'type': 'literal', 'value': 'first'},
            {'type': 'literal', 'value': 'second'},
        ],
    }
    assert interp.visit(node, {}) == 'first'


def test_visit_and_expression(interp):
    node = {
        'type': 'and_expression',
        'children': [
            {'type': 'literal', 'value': 'first'},
            {'type': 'literal', 'value': 'second'},
        ],
    }
    assert interp.visit(node, {}) == 'second'


def test_visit_and_expression_false(interp):
    node = {
        'type': 'and_expression',
        'children': [
            {'type': 'literal', 'value': None},
            {'type': 'literal', 'value': 'second'},
        ],
    }
    assert interp.visit(node, {}) is None


def test_visit_not_expression(interp):
    assert interp.visit(
        {'type': 'not_expression', 'children': [{'type': 'literal', 'value': None}]},
        {}) is True
    assert interp.visit(
        {'type': 'not_expression', 'children': [{'type': 'literal', 'value': 'x'}]},
        {}) is False


def test_visit_not_expression_zero(interp):
    # Special case: !0 should be False, not True.
    node = {'type': 'not_expression', 'children': [{'type': 'literal', 'value': 0}]}
    assert interp.visit(node, {}) is False


def test_visit_pipe(interp):
    node = {
        'type': 'pipe',
        'children': [
            {'type': 'field', 'value': 'a'},
            {'type': 'field', 'value': 'b'},
        ],
    }
    assert interp.visit(node, {'a': {'b': 7}}) == 7


def test_visit_projection(interp):
    node = {
        'type': 'projection',
        'children': [
            {'type': 'current'},
            {'type': 'field', 'value': 'a'},
        ],
    }
    value = [{'a': 1}, {'b': 2}, {'a': 3}]
    assert interp.visit(node, value) == [1, 3]


def test_visit_projection_non_list(interp):
    node = {
        'type': 'projection',
        'children': [
            {'type': 'current'},
            {'type': 'field', 'value': 'a'},
        ],
    }
    assert interp.visit(node, {'a': 1}) is None


def test_visit_value_projection(interp):
    node = {
        'type': 'value_projection',
        'children': [
            {'type': 'current'},
            {'type': 'field', 'value': 'a'},
        ],
    }
    value = {'x': {'a': 1}, 'y': {'b': 2}, 'z': {'a': 3}}
    assert interp.visit(node, value) == [1, 3]


def test_visit_value_projection_non_dict(interp):
    node = {
        'type': 'value_projection',
        'children': [
            {'type': 'current'},
            {'type': 'field', 'value': 'a'},
        ],
    }
    assert interp.visit(node, [1, 2, 3]) is None


def test_is_false_and_is_true(interp):
    assert interp._is_false('') is True
    assert interp._is_false([]) is True
    assert interp._is_false({}) is True
    assert interp._is_false(None) is True
    assert interp._is_false(False) is True
    assert interp._is_false(0) is False
    assert interp._is_false('x') is False

    assert interp._is_true('') is False
    assert interp._is_true('x') is True
    assert interp._is_true(0) is True


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

def test_graphviz_visitor_simple():
    gv = GraphvizVisitor()
    node = {'type': 'identity'}
    output = gv.visit(node)
    lines = output.split('\n')
    assert lines[0] == 'digraph AST {'
    assert lines[-1] == '}'
    assert 'identity1 [label="identity()"]' in output


def test_graphviz_visitor_children():
    gv = GraphvizVisitor()
    node = {
        'type': 'field',
        'value': 'a',
        'children': [
            {'type': 'literal', 'value': 1},
        ],
    }
    output = gv.visit(node)
    assert 'field1 [label="field(a)"]' in output
    assert 'literal2 [label="literal(1)"]' in output
    assert 'field1 -> literal2' in output


def test_graphviz_visitor_resets_lines_start():
    gv = GraphvizVisitor()
    out = gv.visit({'type': 'identity'})
    assert out.startswith('digraph AST {')
