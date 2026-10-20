import pytest

from jmespath.visitor import (
    Options,
    Visitor,
    TreeInterpreter,
    GraphvizVisitor,
    _equals,
    _is_special_number_case,
    _is_comparable,
    _is_actual_number,
    _Expression,
)


# ---------------------------------------------------------------------------
# Helpers / module-level functions
# ---------------------------------------------------------------------------

class TestActualNumber:
    def test_int_is_number(self):
        assert _is_actual_number(5) is True

    def test_float_is_number(self):
        assert _is_actual_number(1.5) is True

    def test_bool_is_not_number(self):
        assert _is_actual_number(True) is False

    def test_false_is_not_number(self):
        assert _is_actual_number(False) is False

    def test_string_is_not_number(self):
        assert _is_actual_number("5") is False

    def test_none_is_not_number(self):
        assert _is_actual_number(None) is False


class TestSpecialNumberCase:
    @pytest.mark.parametrize("x,y", [(0, True), (1, True), (0, False), (1, False),
                                     (True, 0), (True, 1), (False, 0), (False, 1)])
    def test_bool_int_combos(self, x, y):
        assert _is_special_number_case(x, y) is True

    def test_both_numbers(self):
        assert _is_special_number_case(0, 1) is None

    def test_number_and_string(self):
        assert _is_special_number_case(0, "0") is False
        assert _is_special_number_case("1", 1) is False

    def test_two(self):
        assert _is_special_number_case(2, True) is None

    def test_booleans_only(self):
        assert _is_special_number_case(True, False) is None

    def test_bool_and_non_zero_int(self):
        # x is a number (but 2 not in (0,1)), y is bool but not actual number
        assert _is_special_number_case(2, True) is None
        assert _is_special_number_case(True, 2) is None


class TestEquals:
    def test_equal_numbers(self):
        assert _equals(1, 1) is True

    def test_unequal_numbers(self):
        assert _equals(1, 2) is False

    def test_zero_true_is_false(self):
        assert _equals(0, True) is False

    def test_one_true_is_false(self):
        assert _equals(1, True) is False

    def test_zero_false_is_false(self):
        assert _equals(0, False) is False

    def test_one_false_is_false(self):
        assert _equals(1, False) is False

    def test_true_zero_is_false(self):
        assert _equals(True, 0) is False

    def test_false_one_is_false(self):
        assert _equals(False, 1) is False

    def test_normal_strings(self):
        assert _equals("a", "a") is True
        assert _equals("a", "b") is False

    def test_none(self):
        assert _equals(None, None) is True
        assert _equals(None, 1) is False

    def test_lists(self):
        assert _equals([1, 2], [1, 2]) is True
        assert _equals([1, 2], [1, 3]) is False


class TestComparable:
    def test_number(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True

    def test_string(self):
        assert _is_comparable("a") is True

    def test_bool(self):
        assert _is_comparable(True) is False

    def test_list(self):
        assert _is_comparable([]) is False

    def test_none(self):
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
        from collections import OrderedDict
        fn = object()
        opts = Options(dict_cls=OrderedDict, custom_functions=fn)
        assert opts.dict_cls is OrderedDict
        assert opts.custom_functions is fn


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.default_visit({'type': 'foo'})

    def test_unknown_node_raises_default(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'unknown_type'})

    def test_method_cache_dispatch(self):
        v = Visitor()
        called = {}

        def visit_foo(node, *args, **kwargs):
            called['foo'] = (node, args, kwargs)
            return 'result'

        v.visit_foo = visit_foo
        result = v.visit({'type': 'foo'}, 1, 2, k=3)
        assert result == 'result'
        assert called['foo'][1] == (1, 2)
        assert called['foo'][2] == {'k': 3}

    def test_method_cache_reused(self):
        v = Visitor()
        count = {'n': 0}

        def visit_bar(node, *args, **kwargs):
            count['n'] += 1
            return 'r'

        v.visit_bar = visit_bar
        v.visit({'type': 'bar'})
        # After first call, cache populated
        assert 'bar' in v._method_cache
        v.visit({'type': 'bar'})
        v.visit({'type': 'bar'})
        assert count['n'] == 3


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        class Interp:
            def __init__(self):
                self.calls = []

            def visit(self, node, *args, **kwargs):
                self.calls.append((node, args, kwargs))
                return 'ok'

        interp = Interp()
        expr = _Expression('expr', interp)
        assert expr.expression == 'expr'
        assert expr.interpreter is interp
        result = expr.visit('node', 1, k=2)
        assert result == 'ok'
        assert interp.calls == [('node', (1,), {'k': 2})]


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

def field(name):
    return {'type': 'field', 'value': name}


def literal(value):
    return {'type': 'literal', 'value': value}


def identity():
    return {'type': 'identity'}


def comparator(op, left, right):
    return {'type': 'comparator', 'value': op, 'children': [left, right]}


class TestTreeInterpreterBasics:
    def test_init_defaults(self):
        ti = TreeInterpreter()
        assert ti._dict_cls is dict
        assert ti._options.dict_cls is None
        import jmespath.functions as fnmod
        assert isinstance(ti._functions, fnmod.Functions)

    def test_init_with_options_dict_cls(self):
        from collections import OrderedDict
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert ti._dict_cls is OrderedDict

    def test_init_with_custom_functions(self):
        custom = object()
        ti = TreeInterpreter(Options(custom_functions=custom))
        assert ti._functions is custom

    def test_default_visit_raises_with_node_type(self):
        ti = TreeInterpreter()
        with pytest.raises(NotImplementedError):
            ti.default_visit({'type': 'weird'})

    def test_visit_unknown_raises(self):
        ti = TreeInterpreter()
        with pytest.raises(NotImplementedError):
            ti.visit({'type': 'bogus'})


class TestTreeInterpreterFields:
    def test_field_found(self):
        ti = TreeInterpreter()
        assert ti.visit_field(field('a'), {'a': 1}) == 1

    def test_field_missing(self):
        ti = TreeInterpreter()
        assert ti.visit_field(field('a'), {'b': 1}) is None

    def test_field_on_non_dict(self):
        ti = TreeInterpreter()
        assert ti.visit_field(field('a'), [1, 2]) is None

    def test_field_on_none(self):
        ti = TreeInterpreter()
        assert ti.visit_field(field('a'), None) is None


class TestTreeInterpreterIdentityCurrent:
    def test_identity(self):
        ti = TreeInterpreter()
        assert ti.visit_identity(identity(), {'x': 1}) == {'x': 1}

    def test_current(self):
        ti = TreeInterpreter()
        assert ti.visit_current({}, 42) == 42

    def test_literal(self):
        ti = TreeInterpreter()
        assert ti.visit_literal(literal([1, 2]), None) == [1, 2]


class TestTreeInterpreterComparators:
    def test_eq(self):
        ti = TreeInterpreter()
        node = comparator('eq', literal(1), literal(1))
        assert ti.visit_comparator(node, None) is True

    def test_ne(self):
        ti = TreeInterpreter()
        node = comparator('ne', literal(1), literal(2))
        assert ti.visit_comparator(node, None) is True

    def test_eq_zero_true(self):
        ti = TreeInterpreter()
        node = comparator('eq', literal(0), literal(True))
        assert ti.visit_comparator(node, None) is False

    def test_lt(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal(1), literal(2))
        assert ti.visit_comparator(node, None) is True

    def test_gt(self):
        ti = TreeInterpreter()
        node = comparator('gt', literal(3), literal(2))
        assert ti.visit_comparator(node, None) is True

    def test_lte(self):
        ti = TreeInterpreter()
        node = comparator('lte', literal(2), literal(2))
        assert ti.visit_comparator(node, None) is True

    def test_gte(self):
        ti = TreeInterpreter()
        node = comparator('gte', literal(1), literal(2))
        assert ti.visit_comparator(node, None) is False

    def test_ordering_non_comparable_returns_none(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal([1]), literal([2]))
        assert ti.visit_comparator(node, None) is None

    def test_ordering_strings(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal('a'), literal('b'))
        assert ti.visit_comparator(node, None) is True

    def test_ordering_number_vs_string(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal(1), literal('b'))
        # both comparable, so comparator invoked -> TypeError in py3
        with pytest.raises(TypeError):
            ti.visit_comparator(node, None)


class TestTreeInterpreterSubexpression:
    def test_subexpression(self):
        ti = TreeInterpreter()
        node = {
            'type': 'subexpression',
            'children': [field('a'), field('b')],
        }
        assert ti.visit_subexpression(node, {'a': {'b': 5}}) == 5

    def test_subexpression_missing(self):
        ti = TreeInterpreter()
        node = {
            'type': 'subexpression',
            'children': [field('a'), field('b')],
        }
        assert ti.visit_subexpression(node, {'a': {}}) is None


class TestTreeInterpreterIndex:
    def test_index(self):
        ti = TreeInterpreter()
        assert ti.visit_index({'type': 'index', 'value': 1}, [10, 20, 30]) == 20

    def test_index_negative(self):
        ti = TreeInterpreter()
        assert ti.visit_index({'type': 'index', 'value': -1}, [10, 20, 30]) == 30

    def test_index_out_of_range(self):
        ti = TreeInterpreter()
        assert ti.visit_index({'type': 'index', 'value': 5}, [10]) is None

    def test_index_non_list(self):
        ti = TreeInterpreter()
        assert ti.visit_index({'type': 'index', 'value': 0}, 'abc') is None

    def test_index_expression(self):
        ti = TreeInterpreter()
        node = {
            'type': 'index_expression',
            'children': [field('a'), {'type': 'index', 'value': 0}],
        }
        assert ti.visit_index_expression(node, {'a': [1, 2]}) == 1


class TestTreeInterpreterSlice:
    def test_slice(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [1, 3, None]}
        assert ti.visit_slice(node, [0, 1, 2, 3, 4]) == [1, 2]

    def test_slice_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [0, 1, None]}
        assert ti.visit_slice(node, 'abc') is None


class TestTreeInterpreterMultiSelect:
    def test_multi_select_dict(self):
        ti = TreeInterpreter()
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('x')]},
                {'type': 'key_val_pair', 'value': 'b',
                 'children': [field('y')]},
            ],
        }
        assert ti.visit_multi_select_dict(node, {'x': 1, 'y': 2}) == {
            'a': 1, 'b': 2}

    def test_multi_select_dict_none(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': []}
        assert ti.visit_multi_select_dict(node, None) is None

    def test_multi_select_dict_custom_cls(self):
        from collections import OrderedDict
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('x')]},
            ],
        }
        result = ti.visit_multi_select_dict(node, {'x': 1})
        assert isinstance(result, OrderedDict)
        assert result == {'a': 1}

    def test_multi_select_list(self):
        ti = TreeInterpreter()
        node = {
            'type': 'multi_select_list',
            'children': [field('x'), field('y')],
        }
        assert ti.visit_multi_select_list(node, {'x': 1, 'y': 2}) == [1, 2]

    def test_multi_select_list_none(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': []}
        assert ti.visit_multi_select_list(node, None) is None


class TestTreeInterpreterLogic:
    def test_or_left_true(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [literal(1), literal(2)]}
        assert ti.visit_or_expression(node, None) == 1

    def test_or_left_false(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [literal(''), literal(2)]}
        assert ti.visit_or_expression(node, None) == 2

    def test_and_left_false(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [literal([]), literal(2)]}
        assert ti.visit_and_expression(node, None) == []

    def test_and_left_true(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [literal(1), literal(2)]}
        assert ti.visit_and_expression(node, None) == 2

    def test_not_zero(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(0)]}
        assert ti.visit_not_expression(node, None) is False

    def test_not_one(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(1)]}
        assert ti.visit_not_expression(node, None) is False

    def test_not_empty(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal('')]}
        assert ti.visit_not_expression(node, None) is True

    def test_not_none(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(None)]}
        assert ti.visit_not_expression(node, None) is True

    def test_is_false_values(self):
        ti = TreeInterpreter()
        assert ti._is_false('') is True
        assert ti._is_false([]) is True
        assert ti._is_false({}) is True
        assert ti._is_false(None) is True
        assert ti._is_false(False) is True

    def test_is_false_truthy(self):
        ti = TreeInterpreter()
        assert ti._is_false(0) is False
        assert ti._is_false(True) is False
        assert ti._is_false('a') is False

    def test_is_true(self):
        ti = TreeInterpreter()
        assert ti._is_true(1) is True
        assert ti._is_true('') is False


class TestTreeInterpreterPipe:
    def test_pipe_chains(self):
        ti = TreeInterpreter()
        node = {
            'type': 'pipe',
            'children': [field('a'), field('b')],
        }
        assert ti.visit_pipe(node, {'a': {'b': 5}}) == 5


class TestTreeInterpreterProjections:
    def test_projection(self):
        ti = TreeInterpreter()
        node = {
            'type': 'projection',
            'children': [field('items'), field('name')],
        }
        data = {'items': [{'name': 'a'}, {'name': 'b'}, {'other': 1}]}
        assert ti.visit_projection(node, data) == ['a', 'b']

    def test_projection_non_list(self):
        ti = TreeInterpreter()
        node = {
            'type': 'projection',
            'children': [field('items'), identity()],
        }
        assert ti.visit_projection(node, {'items': 'not-a-list'}) is None

    def test_value_projection(self):
        ti = TreeInterpreter()
        node = {
            'type': 'value_projection',
            'children': [field('obj'), field('x')],
        }
        data = {'obj': {'a': {'x': 1}, 'b': {'x': 2}, 'c': {}}}
        assert ti.visit_value_projection(node, data) == [1, 2]

    def test_value_projection_non_dict(self):
        ti = TreeInterpreter()
        node = {
            'type': 'value_projection',
            'children': [field('obj'), identity()],
        }
        assert ti.visit_value_projection(node, {'obj': [1, 2]}) is None

    def test_filter_projection(self):
        ti = TreeInterpreter()
        # children: [base, projection, comparator]
        node = {
            'type': 'filter_projection',
            'children': [
                field('items'),
                field('name'),
                comparator('eq', field('keep'), literal(True)),
            ],
        }
        data = {'items': [
            {'keep': True, 'name': 'a'},
            {'keep': False, 'name': 'b'},
            {'keep': True, 'name': 'c'},
        ]}
        assert ti.visit_filter_projection(node, data) == ['a', 'c']

    def test_filter_projection_non_list(self):
        ti = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [field('items'), identity(),
                         comparator('eq', identity(), literal(1))],
        }
        assert ti.visit_filter_projection(node, {'items': 'x'}) is None

    def test_flatten(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [identity()]}
        assert ti.visit_flatten(node, [1, [2, 3], 4, [5]]) == [1, 2, 3, 4, 5]

    def test_flatten_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [identity()]}
        assert ti.visit_flatten(node, 'abc') is None


class TestTreeInterpreterExprefAndFunctions:
    def test_expref(self):
        ti = TreeInterpreter()
        child = {'type': 'identity'}
        node = {'type': 'expref', 'children': [child]}
        expr = ti.visit_expref(node, None)
        assert isinstance(expr, _Expression)
        assert expr.expression is child
        assert expr.interpreter is ti

    def test_function_expression(self):
        ti = TreeInterpreter()
        node = {
            'type': 'function_expression',
            'value': 'length',
            'children': [literal('abcd')],
        }
        assert ti.visit_function_expression(node, None) == 4

    def test_function_expression_custom_functions(self):
        calls = []

        class CustomFns:
            def call_function(self, name, args):
                calls.append((name, args))
                return 'custom-result'

        ti = TreeInterpreter(Options(custom_functions=CustomFns()))
        node = {
            'type': 'function_expression',
            'value': 'foo',
            'children': [literal(1), literal(2)],
        }
        assert ti.visit_function_expression(node, None) == 'custom-result'
        assert calls == [('foo', [1, 2])]


class TestTreeInterpreterVisitDispatch:
    def test_visit_dispatches(self):
        ti = TreeInterpreter()
        assert ti.visit(literal(5), None) == 5
        assert ti.visit(identity(), 'x') == 'x'


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_simple_node(self):
        gv = GraphvizVisitor()
        result = gv.visit({'type': 'identity'})
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert 'identity1 [label="identity()"]' in result

    def test_node_with_value(self):
        gv = GraphvizVisitor()
        result = gv.visit({'type': 'field', 'value': 'foo'})
        assert 'field1 [label="field(foo)"]' in result

    def test_children(self):
        gv = GraphvizVisitor()
        node = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        result = gv.visit(node)
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert 'subexpression1 [label="subexpression()"]' in result
        assert '  subexpression1 -> field2' in result
        assert '  subexpression1 -> field3' in result
        assert 'field2 [label="field(a)"]' in result
        assert 'field3 [label="field(b)"]' in result

    def test_counter_increments_across_calls(self):
        gv = GraphvizVisitor()
        gv.visit({'type': 'identity'})
        assert gv._count == 2
        gv.visit({'type': 'identity'})
        assert gv._count == 3

    def test_returns_joined_lines(self):
        gv = GraphvizVisitor()
        result = gv.visit({'type': 'identity'})
        assert isinstance(result, str)
        assert '\n'.join(gv._lines) == result
