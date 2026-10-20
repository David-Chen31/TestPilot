import pytest
from collections import OrderedDict

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
# Helper node builders
# ---------------------------------------------------------------------------

def field(name):
    return {'type': 'field', 'value': name}


def literal(value):
    return {'type': 'literal', 'value': value}


def current():
    return {'type': 'current'}


def identity():
    return {'type': 'identity'}


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

class TestEquals:
    def test_equal_numbers(self):
        assert _equals(1, 1) is True
        assert _equals(1.5, 1.5) is True

    def test_not_equal(self):
        assert _equals(1, 2) is False

    def test_bool_vs_zero(self):
        assert _equals(0, False) is False
        assert _equals(False, 0) is False
        assert _equals(1, True) is False
        assert _equals(True, 1) is False

    def test_bool_vs_bool(self):
        assert _equals(True, True) is True
        assert _equals(False, False) is True
        assert _equals(True, False) is False

    def test_zero_vs_one(self):
        assert _equals(0, 0) is True
        assert _equals(0, 1) is False

    def test_strings(self):
        assert _equals("a", "a") is True
        assert _equals("a", "b") is False


class TestSpecialNumberCase:
    def test_zero_with_bool(self):
        assert _is_special_number_case(0, True) is True
        assert _is_special_number_case(0, False) is True
        assert _is_special_number_case(1, True) is True

    def test_bool_with_zero(self):
        assert _is_special_number_case(True, 0) is True
        assert _is_special_number_case(False, 0) is True
        assert _is_special_number_case(True, 1) is True

    def test_non_special(self):
        assert _is_special_number_case(2, True) is None
        assert _is_special_number_case(1, 2) is None
        assert _is_special_number_case("a", 0) is None


class TestIsActualNumber:
    def test_numbers(self):
        assert _is_actual_number(1) is True
        assert _is_actual_number(1.5) is True

    def test_bools(self):
        assert _is_actual_number(True) is False
        assert _is_actual_number(False) is False

    def test_strings(self):
        assert _is_actual_number("1") is False


class TestIsComparable:
    def test_numbers(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True

    def test_strings(self):
        assert _is_comparable("a") is True

    def test_bools_not_comparable(self):
        assert _is_comparable(True) is False

    def test_other(self):
        assert _is_comparable([]) is False
        assert _is_comparable(None) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults(self):
        opts = Options()
        assert opts.dict_cls is None
        assert opts.custom_functions is None

    def test_custom(self):
        opts = Options(dict_cls=OrderedDict, custom_functions=object())
        assert opts.dict_cls is OrderedDict
        assert opts.custom_functions is not None


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'unknown'})

    def test_caches_method(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, *args, **kwargs):
                return 'foo'

        v = MyVisitor()
        assert v.visit({'type': 'foo'}) == 'foo'
        # cache should contain the bound method
        cached = v._method_cache['foo']
        assert v.visit({'type': 'foo'}) == 'foo'
        assert v._method_cache['foo'] is cached

    def test_passes_args(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, value, extra=None):
                return (value, extra)

        v = MyVisitor()
        assert v.visit({'type': 'foo'}, 1, extra=2) == (1, 2)


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        interpreter = TreeInterpreter()
        expr = _Expression(literal(5), interpreter)
        assert expr.visit(literal(5)) == 5


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

class TestTreeInterpreter:
    def test_default_visit_raises(self):
        t = TreeInterpreter()
        with pytest.raises(NotImplementedError):
            t.visit({'type': 'unknown_node'})

    def test_options_none(self):
        t = TreeInterpreter()
        assert t._dict_cls is dict

    def test_options_dict_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert t._dict_cls is OrderedDict

    def test_options_custom_functions(self):
        marker = object()
        t = TreeInterpreter(Options(custom_functions=marker))
        assert t._functions is marker

    def test_default_functions(self):
        t = TreeInterpreter()
        from jmespath import functions
        assert isinstance(t._functions, functions.Functions)

    def test_visit_field(self):
        t = TreeInterpreter()
        node = field('a')
        assert t.visit(node, {'a': 1}) == 1
        assert t.visit(node, {}) is None
        # AttributeError path
        assert t.visit(node, [1, 2]) is None

    def test_visit_current(self):
        t = TreeInterpreter()
        assert t.visit(current(), {'a': 1}) == {'a': 1}

    def test_visit_identity(self):
        t = TreeInterpreter()
        assert t.visit(identity(), [1, 2]) == [1, 2]

    def test_visit_index(self):
        t = TreeInterpreter()
        node = {'type': 'index', 'value': 1}
        assert t.visit(node, [1, 2, 3]) == 2
        assert t.visit(node, [1]) is None
        assert t.visit(node, "abc") is None

    def test_visit_index_expression(self):
        t = TreeInterpreter()
        node = {
            'type': 'index_expression',
            'children': [
                field('a'),
                {'type': 'index', 'value': 1},
            ],
        }
        assert t.visit(node, {'a': [10, 20, 30]}) == 20

    def test_visit_slice(self):
        t = TreeInterpreter()
        node = {'type': 'slice', 'children': [None, None, 2]}
        assert t.visit(node, [1, 2, 3, 4]) == [1, 3]
        assert t.visit(node, "abc") is None

    def test_visit_literal(self):
        t = TreeInterpreter()
        assert t.visit(literal(42), {"a": 1}) == 42

    def test_visit_key_val_pair(self):
        t = TreeInterpreter()
        node = {
            'type': 'key_val_pair',
            'value': 'k',
            'children': [literal(7)],
        }
        assert t.visit(node, {}) == 7

    def test_visit_subexpression(self):
        t = TreeInterpreter()
        node = {
            'type': 'subexpression',
            'children': [field('a'), field('b')],
        }
        assert t.visit(node, {'a': {'b': 3}}) == 3
        assert t.visit(node, {'a': 1}) is None

    def test_visit_comparator_eq(self):
        t = TreeInterpreter()
        node = {
            'type': 'comparator',
            'value': 'eq',
            'children': [literal(1), literal(1)],
        }
        assert t.visit(node, {}) is True

    def test_visit_comparator_ne(self):
        t = TreeInterpreter()
        node = {
            'type': 'comparator',
            'value': 'ne',
            'children': [literal(1), literal(2)],
        }
        assert t.visit(node, {}) is True

    def test_visit_comparator_ordering(self):
        t = TreeInterpreter()
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [literal(1), literal(2)],
        }
        assert t.visit(node, {}) is True

    def test_visit_comparator_ordering_incomparable(self):
        t = TreeInterpreter()
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [literal([1]), literal(2)],
        }
        assert t.visit(node, {}) is None

    def test_visit_comparator_string_ordering(self):
        t = TreeInterpreter()
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [literal("a"), literal("b")],
        }
        assert t.visit(node, {}) is True

    def test_visit_expref(self):
        t = TreeInterpreter()
        node = {'type': 'expref', 'children': [current()]}
        result = t.visit(node, {})
        assert isinstance(result, _Expression)
        assert result.interpreter is t

    def test_visit_function_expression(self):
        t = TreeInterpreter()

        class FakeFns:
            def call_function(self, name, args):
                return (name, args)

        t._functions = FakeFns()
        node = {
            'type': 'function_expression',
            'value': 'foo',
            'children': [literal(1), literal(2)],
        }
        assert t.visit(node, {}) == ('foo', [1, 2])

    def test_visit_filter_projection(self):
        t = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [
                field('items'),
                field('name'),
                {'type': 'comparator', 'value': 'eq',
                 'children': [field('keep'), literal(True)]},
            ],
        }
        value = {'items': [
            {'keep': True, 'name': 'a'},
            {'keep': False, 'name': 'b'},
            {'keep': True, 'name': None},
        ]}
        assert t.visit(node, value) == ['a']

    def test_visit_filter_projection_not_list(self):
        t = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [field('items'), field('name'), current()],
        }
        assert t.visit(node, {'items': 'notalist'}) is None

    def test_visit_flatten(self):
        t = TreeInterpreter()
        node = {'type': 'flatten', 'children': [current()]}
        assert t.visit(node, [1, [2, 3], [4]]) == [1, 2, 3, 4]

    def test_visit_flatten_not_list(self):
        t = TreeInterpreter()
        node = {'type': 'flatten', 'children': [current()]}
        assert t.visit(node, "string") is None

    def test_visit_multi_select_dict(self):
        t = TreeInterpreter()
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('x')]},
                {'type': 'key_val_pair', 'value': 'b',
                 'children': [field('y')]},
            ],
        }
        assert t.visit(node, {'x': 1, 'y': 2}) == {'a': 1, 'b': 2}

    def test_visit_multi_select_dict_none(self):
        t = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': []}
        assert t.visit(node, None) is None

    def test_visit_multi_select_dict_uses_dict_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('x')]},
                {'type': 'key_val_pair', 'value': 'b',
                 'children': [field('y')]},
            ],
        }
        result = t.visit(node, {'x': 1, 'y': 2})
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ['a', 'b']

    def test_visit_multi_select_list(self):
        t = TreeInterpreter()
        node = {
            'type': 'multi_select_list',
            'children': [field('x'), field('y')],
        }
        assert t.visit(node, {'x': 1, 'y': 2}) == [1, 2]

    def test_visit_multi_select_list_none(self):
        t = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': []}
        assert t.visit(node, None) is None

    def test_visit_or_expression_true(self):
        t = TreeInterpreter()
        node = {
            'type': 'or_expression',
            'children': [literal('a'), literal('b')],
        }
        assert t.visit(node, {}) == 'a'

    def test_visit_or_expression_false(self):
        t = TreeInterpreter()
        node = {
            'type': 'or_expression',
            'children': [literal(''), literal('b')],
        }
        assert t.visit(node, {}) == 'b'

    def test_visit_and_expression_true(self):
        t = TreeInterpreter()
        node = {
            'type': 'and_expression',
            'children': [literal('a'), literal('b')],
        }
        assert t.visit(node, {}) == 'b'

    def test_visit_and_expression_false(self):
        t = TreeInterpreter()
        node = {
            'type': 'and_expression',
            'children': [literal(''), literal('b')],
        }
        assert t.visit(node, {}) == ''

    def test_visit_not_expression_zero(self):
        t = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(0)]}
        assert t.visit(node, {}) is False

    def test_visit_not_expression_nonzero(self):
        t = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(5)]}
        assert t.visit(node, {}) is False

    def test_visit_not_expression_true(self):
        t = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal('')]}
        assert t.visit(node, {}) is True

    def test_visit_pipe(self):
        t = TreeInterpreter()
        node = {
            'type': 'pipe',
            'children': [field('a'), field('b')],
        }
        assert t.visit(node, {'a': {'b': 9}}) == 9

    def test_visit_projection(self):
        t = TreeInterpreter()
        node = {
            'type': 'projection',
            'children': [field('items'), field('name')],
        }
        value = {'items': [{'name': 'a'}, {'name': None}, {'name': 'b'}]}
        assert t.visit(node, value) == ['a', 'b']

    def test_visit_projection_not_list(self):
        t = TreeInterpreter()
        node = {
            'type': 'projection',
            'children': [field('items'), field('name')],
        }
        assert t.visit(node, {'items': 5}) is None

    def test_visit_value_projection(self):
        t = TreeInterpreter()
        node = {
            'type': 'value_projection',
            'children': [field('obj'), field('x')],
        }
        value = {'obj': {'a': {'x': 1}, 'b': {'x': 2}}}
        result = t.visit(node, value)
        assert sorted(result) == [1, 2]

    def test_visit_value_projection_not_dict(self):
        t = TreeInterpreter()
        node = {
            'type': 'value_projection',
            'children': [field('obj'), field('x')],
        }
        assert t.visit(node, {'obj': [1, 2]}) is None

    def test_is_false(self):
        t = TreeInterpreter()
        assert t._is_false('') is True
        assert t._is_false([]) is True
        assert t._is_false({}) is True
        assert t._is_false(None) is True
        assert t._is_false(False) is True

    def test_is_false_truthy(self):
        t = TreeInterpreter()
        assert t._is_false(0) is False
        assert t._is_false('a') is False
        assert t._is_false([0]) is False

    def test_is_true(self):
        t = TreeInterpreter()
        assert t._is_true(1) is True
        assert t._is_true('') is False
        assert t._is_true(None) is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_empty_node(self):
        g = GraphvizVisitor()
        result = g.visit({'type': 'foo'})
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[1] == 'foo1 [label="foo()"]'
        assert lines[-1] == '}'

    def test_node_with_value(self):
        g = GraphvizVisitor()
        result = g.visit({'type': 'field', 'value': 'name'})
        assert 'field1 [label="field(name)"]' in result

    def test_node_with_children(self):
        g = GraphvizVisitor()
        node = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        result = g.visit(node)
        lines = result.split('\n')
        assert 'subexpression1 [label="subexpression()"]' in lines
        # children get sequential numbering
        assert '  subexpression1 -> field2' in lines
        assert '  subexpression1 -> field3' in lines
        assert 'field2 [label="field(a)"]' in lines
        assert 'field3 [label="field(b)"]' in lines

    def test_count_increments(self):
        g = GraphvizVisitor()
        g.visit({'type': 'a'})
        g.visit({'type': 'b'})
        assert g._count == 3

    def test_lines_accumulate(self):
        g = GraphvizVisitor()
        g.visit({'type': 'a'})
        first_len = len(g._lines)
        g.visit({'type': 'b'})
        assert len(g._lines) > first_len
