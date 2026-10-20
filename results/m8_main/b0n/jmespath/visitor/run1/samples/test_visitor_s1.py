import pytest
from collections import OrderedDict

from jmespath.visitor import (
    _equals,
    _is_special_number_case,
    _is_comparable,
    _is_actual_number,
    Options,
    _Expression,
    Visitor,
    TreeInterpreter,
    GraphvizVisitor,
)
from jmespath import functions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_interpreter(options=None):
    return TreeInterpreter(options)


def field(name):
    return {'type': 'field', 'value': name}


def literal(value):
    return {'type': 'literal', 'value': value}


def identity():
    return {'type': 'identity'}


def current():
    return {'type': 'current'}


def subexpression(*children):
    return {'type': 'subexpression', 'children': list(children)}


def comparator(op, left, right):
    return {'type': 'comparator', 'value': op, 'children': [left, right]}


def index_expression(*children):
    return {'type': 'index_expression', 'children': list(children)}


def index(value):
    return {'type': 'index', 'value': value}


def pipe(*children):
    return {'type': 'pipe', 'children': list(children)}


def function_expression(name, *children):
    return {'type': 'function_expression', 'value': name, 'children': list(children)}


def expref(child):
    return {'type': 'expref', 'children': [child]}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestEquals:
    def test_basic_equal(self):
        assert _equals(1, 1) is True
        assert _equals("a", "a") is True
        assert _equals([1, 2], [1, 2]) is True

    def test_basic_not_equal(self):
        assert _equals(1, 2) is False
        assert _equals("a", "b") is False

    def test_special_number_case_zero_false(self):
        assert _equals(0, False) is False
        assert _equals(False, 0) is False

    def test_special_number_case_one_true(self):
        assert _equals(1, True) is False
        assert _equals(True, 1) is False

    def test_zero_true(self):
        # 0 == True is normally False, but special case forces False anyway.
        assert _equals(0, True) is False

    def test_one_false(self):
        assert _equals(1, False) is False

    def test_bool_bool(self):
        assert _equals(True, True) is True
        assert _equals(False, False) is True
        assert _equals(True, False) is False

    def test_non_special_numbers(self):
        assert _equals(2, 2) is True
        assert _equals(2, True) is False


class TestSpecialNumberCase:
    def test_returns_false_for_non_special(self):
        assert _is_special_number_case(2, 2) is False
        assert _is_special_number_case("a", "a") is False

    def test_first_arg_number_zero_one_second_bool(self):
        assert _is_special_number_case(0, True) is True
        assert _is_special_number_case(1, False) is True

    def test_first_arg_number_non_bool(self):
        assert _is_special_number_case(0, 0) is False

    def test_second_arg_number_zero_one_first_bool(self):
        assert _is_special_number_case(True, 0) is True
        assert _is_special_number_case(False, 1) is True

    def test_second_arg_number_non_bool(self):
        assert _is_special_number_case(0, 0) is False

    def test_both_bools(self):
        # bools are not actual numbers, so falls through
        assert _is_special_number_case(True, False) is False


class TestIsComparable:
    def test_numbers(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True

    def test_strings(self):
        assert _is_comparable("x") is True

    def test_bools_not_comparable(self):
        assert _is_comparable(True) is False

    def test_others_not_comparable(self):
        assert _is_comparable(None) is False
        assert _is_comparable([]) is False
        assert _is_comparable({}) is False

    def test_complex_number(self):
        assert _is_comparable(1j) is True


class TestIsActualNumber:
    def test_int_float(self):
        assert _is_actual_number(1) is True
        assert _is_actual_number(1.5) is True

    def test_bool(self):
        assert _is_actual_number(True) is False
        assert _is_actual_number(False) is False

    def test_other(self):
        assert _is_actual_number("1") is False
        assert _is_actual_number(None) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults(self):
        o = Options()
        assert o.dict_cls is None
        assert o.custom_functions is None

    def test_custom(self):
        o = Options(dict_cls=OrderedDict, custom_functions="fn")
        assert o.dict_cls is OrderedDict
        assert o.custom_functions == "fn"


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        calls = []

        class Interp:
            def visit(self, node, *args, **kwargs):
                calls.append((node, args, kwargs))
                return "ok"

        expr = _Expression("expr", Interp())
        result = expr.visit("node", 1, 2, kw=3)
        assert result == "ok"
        assert calls == [("node", (1, 2), {"kw": 3})]

    def test_attributes(self):
        interp = object()
        expr = _Expression("expr", interp)
        assert expr.expression == "expr"
        assert expr.interpreter is interp


# ---------------------------------------------------------------------------
# Visitor base
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_dispatch_and_cache(self):
        v = Visitor()
        node = {'type': 'identity'}
        with pytest.raises(NotImplementedError):
            v.visit(node)
        # Method was cached; still default_visit
        cached = v._method_cache['identity']
        assert cached == v.default_visit

    def test_custom_method_dispatch(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, *args, **kwargs):
                return ('foo', args, kwargs)

        v = MyVisitor()
        assert v.visit({'type': 'foo'}, 1, kw=2) == ('foo', (1,), {'kw': 2})
        assert 'foo' in v._method_cache

    def test_unknown_type_uses_default(self):
        v = Visitor()
        with pytest.raises(NotImplementedError, match="default_visit"):
            v.visit({'type': 'unknown'})

    def test_missing_type_key(self):
        v = Visitor()
        with pytest.raises(KeyError):
            v.visit({})


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

class TestTreeInterpreterInit:
    def test_defaults(self):
        ti = TreeInterpreter()
        assert ti._dict_cls is dict
        assert isinstance(ti._functions, functions.Functions)

    def test_custom_dict_cls(self):
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert ti._dict_cls is OrderedDict

    def test_custom_functions(self):
        def custom(*args):
            return "custom"
        ti = TreeInterpreter(Options(custom_functions=custom))
        assert ti._functions is custom


class TestTreeInterpreterDefault:
    def test_default_visit_error_message(self):
        ti = TreeInterpreter()
        with pytest.raises(NotImplementedError, match="mystery"):
            ti.default_visit({'type': 'mystery'})
        with pytest.raises(NotImplementedError):
            ti.visit({'type': 'mystery'})


class TestFieldAndIdentity:
    def test_field_dict(self):
        ti = TreeInterpreter()
        assert ti.visit(field('a'), {'a': 1}) == 1

    def test_field_missing(self):
        ti = TreeInterpreter()
        assert ti.visit(field('a'), {'b': 1}) is None

    def test_field_non_dict(self):
        ti = TreeInterpreter()
        assert ti.visit(field('a'), [1, 2]) is None

    def test_identity(self):
        ti = TreeInterpreter()
        assert ti.visit(identity(), {'a': 1}) == {'a': 1}

    def test_current(self):
        ti = TreeInterpreter()
        assert ti.visit(current(), 42) == 42

    def test_subexpression(self):
        ti = TreeInterpreter()
        node = subexpression(field('a'), field('b'))
        assert ti.visit(node, {'a': {'b': 5}}) == 5


class TestComparator:
    def test_eq_true(self):
        ti = TreeInterpreter()
        node = comparator('eq', literal(1), literal(1))
        assert ti.visit(node, {}) is True

    def test_eq_special_number(self):
        ti = TreeInterpreter()
        node = comparator('eq', literal(0), literal(False))
        assert ti.visit(node, {}) is False

    def test_ne(self):
        ti = TreeInterpreter()
        node = comparator('ne', literal(0), literal(False))
        assert ti.visit(node, {}) is True

    def test_lt_numbers(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal(1), literal(2))
        assert ti.visit(node, {}) is True

    def test_gt_numbers(self):
        ti = TreeInterpreter()
        node = comparator('gt', literal(1), literal(2))
        assert ti.visit(node, {}) is False

    def test_lte(self):
        ti = TreeInterpreter()
        assert ti.visit(comparator('lte', literal(2), literal(2)), {}) is True

    def test_gte(self):
        ti = TreeInterpreter()
        assert ti.visit(comparator('gte', literal(2), literal(2)), {}) is True

    def test_lt_strings(self):
        ti = TreeInterpreter()
        assert ti.visit(comparator('lt', literal('a'), literal('b')), {}) is True

    def test_ordering_non_comparable(self):
        ti = TreeInterpreter()
        node = comparator('lt', literal(None), literal(1))
        assert ti.visit(node, {}) is None

    def test_ordering_bool_non_comparable(self):
        ti = TreeInterpreter()
        node = comparator('gt', literal(True), literal(False))
        assert ti.visit(node, {}) is None


class TestExpref:
    def test_expref_returns_expression(self):
        ti = TreeInterpreter()
        node = expref(field('a'))
        result = ti.visit(node, {})
        assert isinstance(result, _Expression)
        assert result.interpreter is ti
        assert result.expression == field('a')

    def test_expref_visit_evaluates(self):
        ti = TreeInterpreter()
        expr = ti.visit(expref(field('a')), {})
        assert expr.visit({'type': 'field', 'value': 'a'}, {'a': 10}) == 10


class TestFunctionExpression:
    def test_call_function(self):
        ti = TreeInterpreter()
        node = function_expression('length', literal('abc'))
        assert ti.visit(node, {}) == 3

    def test_custom_function(self):
        class Fns:
            def call_function(self, name, args):
                return (name, args)

        ti = TreeInterpreter(Options(custom_functions=Fns()))
        node = function_expression('foo', literal(1), literal(2))
        assert ti.visit(node, {}) == ('foo', [1, 2])


class TestFilterProjection:
    def test_basic_filter(self):
        ti = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [
                identity(),
                identity(),
                literal(5),
            ],
        }
        # comparator is literal 5 -> truthy for each element
        assert ti.visit(node, [1, 2, 3]) == [1, 2, 3]

    def test_filters_out_falsey(self):
        ti = TreeInterpreter()
        # filter compares element to 2
        node = {
            'type': 'filter_projection',
            'children': [
                identity(),
                identity(),
                comparator('eq', current(), literal(2)),
            ],
        }
        assert ti.visit(node, [1, 2, 3, 2]) == [2, 2]

    def test_non_list_base_returns_none(self):
        ti = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [identity(), identity(), literal(True)],
        }
        assert ti.visit(node, {'a': 1}) is None

    def test_skips_none_results(self):
        ti = TreeInterpreter()
        node = {
            'type': 'filter_projection',
            'children': [
                identity(),
                field('a'),
                literal(True),
            ],
        }
        result = ti.visit(node, [{'a': 1}, {'b': 2}, {'a': 3}])
        assert result == [1, 3]


class TestFlatten:
    def test_flatten_nested(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [identity()]}
        assert ti.visit(node, [[1, 2], [3], 4]) == [1, 2, 3, 4]

    def test_flatten_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [identity()]}
        assert ti.visit(node, {'a': 1}) is None


class TestIndex:
    def test_index_list(self):
        ti = TreeInterpreter()
        assert ti.visit(index(1), [10, 20, 30]) == 20

    def test_index_negative(self):
        ti = TreeInterpreter()
        assert ti.visit(index(-1), [10, 20, 30]) == 30

    def test_index_out_of_bounds(self):
        ti = TreeInterpreter()
        assert ti.visit(index(5), [1, 2]) is None

    def test_index_non_list(self):
        ti = TreeInterpreter()
        assert ti.visit(index(0), "abc") is None

    def test_index_expression(self):
        ti = TreeInterpreter()
        node = index_expression(field('a'), index(0))
        assert ti.visit(node, {'a': [7, 8]}) == 7


class TestSlice:
    def test_slice_list(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [0, 2]}
        assert ti.visit(node, [1, 2, 3, 4]) == [1, 2]

    def test_slice_with_step(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [None, None, 2]}
        assert ti.visit(node, [1, 2, 3, 4]) == [1, 3]

    def test_slice_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [0, 2]}
        assert ti.visit(node, "abcdef") is None


class TestLiteralAndKeyVal:
    def test_literal(self):
        ti = TreeInterpreter()
        assert ti.visit(literal({'a': 1}), "ignored") == {'a': 1}

    def test_key_val_pair(self):
        ti = TreeInterpreter()
        node = {'type': 'key_val_pair', 'children': [literal('x')]}
        assert ti.visit(node, {}) == 'x'


class TestMultiSelect:
    def test_multi_select_dict(self):
        ti = TreeInterpreter()
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('a')]},
                {'type': 'key_val_pair', 'value': 'b',
                 'children': [field('b')]},
            ],
        }
        assert ti.visit(node, {'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_multi_select_dict_none(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': []}
        assert ti.visit(node, None) is None

    def test_multi_select_dict_custom_cls(self):
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'b',
                 'children': [field('b')]},
                {'type': 'key_val_pair', 'value': 'a',
                 'children': [field('a')]},
            ],
        }
        result = ti.visit(node, {'a': 1, 'b': 2})
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ['b', 'a']

    def test_multi_select_list(self):
        ti = TreeInterpreter()
        node = {
            'type': 'multi_select_list',
            'children': [field('a'), field('b')],
        }
        assert ti.visit(node, {'a': 1, 'b': 2}) == [1, 2]

    def test_multi_select_list_none(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': []}
        assert ti.visit(node, None) is None


class TestOrAndNot:
    def test_or_first_truthy(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression', 'children': [literal(1), literal(2)]}
        assert ti.visit(node, {}) == 1

    def test_or_falsey_uses_second(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression', 'children': [literal(''), literal(2)]}
        assert ti.visit(node, {}) == 2

    def test_or_none(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression', 'children': [literal(None), literal(2)]}
        assert ti.visit(node, {}) == 2

    def test_or_empty_list(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression', 'children': [literal([]), literal(2)]}
        assert ti.visit(node, {}) == 2

    def test_or_empty_dict(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression', 'children': [literal({}), literal(2)]}
        assert ti.visit(node, {}) == 2

    def test_and_first_falsey(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression', 'children': [literal(''), literal(2)]}
        assert ti.visit(node, {}) == ''

    def test_and_first_false(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression', 'children': [literal(False), literal(2)]}
        assert ti.visit(node, {}) is False

    def test_and_second(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression', 'children': [literal(1), literal(2)]}
        assert ti.visit(node, {}) == 2

    def test_not_expression(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal('')]}
        assert ti.visit(node, {}) is True

    def test_not_truthy(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(1)]}
        assert ti.visit(node, {}) is False

    def test_not_zero_special(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [literal(0)]}
        # !0 should be False, not True
        assert ti.visit(node, {}) is False


class TestPipe:
    def test_pipe(self):
        ti = TreeInterpreter()
        node = pipe(field('a'), field('b'))
        assert ti.visit(node, {'a': {'b': 5}}) == 5


class TestProjection:
    def test_projection(self):
        ti = TreeInterpreter()
        node = {'type': 'projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, [{'a': 1}, {'a': 2}]) == [1, 2]

    def test_projection_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, {'a': 1}) is None

    def test_projection_skips_none(self):
        ti = TreeInterpreter()
        node = {'type': 'projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, [{'a': 1}, {'b': 2}]) == [1]


class TestValueProjection:
    def test_value_projection(self):
        ti = TreeInterpreter()
        node = {'type': 'value_projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, {'x': {'a': 1}, 'y': {'a': 2}}) == [1, 2]

    def test_value_projection_non_dict(self):
        ti = TreeInterpreter()
        node = {'type': 'value_projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, [1, 2]) is None

    def test_value_projection_skips_none(self):
        ti = TreeInterpreter()
        node = {'type': 'value_projection', 'children': [identity(), field('a')]}
        assert ti.visit(node, {'x': {'a': 1}, 'y': {'b': 2}}) == [1]


class TestIsFalseTrue:
    def test_is_false(self):
        ti = TreeInterpreter()
        for v in ['', [], {}, None, False]:
            assert ti._is_false(v) is True
        for v in [0, 1, 'x', [1], {'a': 1}, True]:
            assert ti._is_false(v) is False

    def test_is_true(self):
        ti = TreeInterpreter()
        assert ti._is_true(1) is True
        assert ti._is_true('') is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_single_node(self):
        gv = GraphvizVisitor()
        out = gv.visit({'type': 'identity'})
        lines = out.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert 'identity1 [label="identity()"]' in lines

    def test_with_value(self):
        gv = GraphvizVisitor()
        out = gv.visit({'type': 'field', 'value': 'foo'})
        assert 'field1 [label="field(foo)"]' in out

    def test_with_children(self):
        gv = GraphvizVisitor()
        node = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        out = gv.visit(node)
        lines = out.split('\n')
        assert 'subexpression1 [label="subexpression()"]' in lines
        assert '  subexpression1 -> field2' in lines
        assert '  subexpression1 -> field3' in lines
        assert 'field2 [label="field(a)"]' in lines
        assert 'field3 [label="field(b)"]' in lines

    def test_nested_children(self):
        gv = GraphvizVisitor()
        node = {
            'type': 'root',
            'children': [
                {
                    'type': 'mid',
                    'children': [{'type': 'leaf'}],
                },
            ],
        }
        out = gv.visit(node)
        assert 'root1 [label="root()"]' in out
        assert '  root1 -> mid2' in out
        assert 'mid2 [label="mid()"]' in out
        assert '  mid2 -> leaf3' in out
        assert 'leaf3 [label="leaf()"]' in out

    def test_multiple_visits_accumulate_lines(self):
        gv = GraphvizVisitor()
        gv.visit({'type': 'a'})
        out = gv.visit({'type': 'b'})
        assert out.count('digraph AST {') == 2
        # counter increments across calls
        assert 'a1' in out
        assert 'b2' in out

    def test_returns_string(self):
        gv = GraphvizVisitor()
        out = gv.visit({'type': 'x'})
        assert isinstance(out, str)
