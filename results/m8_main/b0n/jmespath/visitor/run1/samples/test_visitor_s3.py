import pytest
from collections import OrderedDict

from jmespath import visitor
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
# Module-level helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_equals_plain(self):
        assert _equals(1, 1) is True
        assert _equals("a", "a") is True
        assert _equals(1, 2) is False

    def test_equals_special_number_true(self):
        # True is not an actual number, 1 is; special case -> False
        assert _equals(1, True) is False
        assert _equals(True, 1) is False
        assert _equals(0, False) is False
        assert _equals(False, 0) is False

    def test_equals_special_only_zero_one(self):
        # 2 == True would normally be False anyway; special-case does not apply.
        assert _equals(2, True) is False
        assert _equals(True, 2) is False

    def test_equals_bools_themselves(self):
        assert _equals(True, True) is True
        assert _equals(False, False) is True

    def test_is_special_number_case(self):
        assert _is_special_number_case(0, True) is True
        assert _is_special_number_case(1, False) is True
        assert _is_special_number_case(True, 1) is True
        assert _is_special_number_case(False, 0) is True
        # Non-special
        assert _is_special_number_case(2, True) is None
        assert _is_special_number_case(1, 1) is None
        assert _is_special_number_case("a", 1) is None

    def test_is_comparable(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True
        assert _is_comparable("abc") is True
        assert _is_comparable(True) is False
        assert _is_comparable(None) is False
        assert _is_comparable([]) is False

    def test_is_actual_number(self):
        assert _is_actual_number(1) is True
        assert _is_actual_number(1.5) is True
        assert _is_actual_number(True) is False
        assert _is_actual_number(False) is False
        assert _is_actual_number("1") is False
        assert _is_actual_number(None) is False


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults(self):
        opts = Options()
        assert opts.dict_cls is None
        assert opts.custom_functions is None

    def test_custom(self):
        sentinel = object()
        opts = Options(dict_cls=OrderedDict, custom_functions=sentinel)
        assert opts.dict_cls is OrderedDict
        assert opts.custom_functions is sentinel


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.default_visit({'type': 'whatever'})

    def test_unknown_type_uses_default_visit(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'not_a_real_node'})

    def test_method_cache(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'foo'})
        assert 'foo' in v._method_cache
        assert v._method_cache['foo'] == v.default_visit

    def test_dispatch_to_visit_method(self):
        class MyVisitor(Visitor):
            def visit_node(self, node, *args, **kwargs):
                return (node, args, kwargs)

        v = MyVisitor()
        node = {'type': 'node'}
        assert v.visit(node, 1, 2, k=3) == (node, (1, 2), {'k': 3})
        # cached
        assert 'node' in v._method_cache


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        class FakeInterpreter:
            def __init__(self):
                self.calls = []

            def visit(self, node, *args, **kwargs):
                self.calls.append((node, args, kwargs))
                return 'result'

        interp = FakeInterpreter()
        expr = _Expression({'type': 'x'}, interp)
        assert expr.expression == {'type': 'x'}
        assert expr.interpreter is interp
        assert expr.visit('node', 'a', k=1) == 'result'
        assert interp.calls == [('node', ('a',), {'k': 1})]


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

def _field(name):
    return {'type': 'field', 'value': name}


def _literal(val):
    return {'type': 'literal', 'value': val}


def _identity():
    return {'type': 'identity'}


def _current():
    return {'type': 'current'}


def _comparator(op, left, right):
    return {'type': 'comparator', 'value': op, 'children': [left, right]}


class TestTreeInterpreterInit:
    def test_defaults(self):
        ti = TreeInterpreter()
        assert ti._dict_cls is dict
        assert isinstance(ti._functions, visitor.functions.Functions)

    def test_custom_dict_cls(self):
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert ti._dict_cls is OrderedDict

    def test_custom_functions(self):
        sentinel = object()
        ti = TreeInterpreter(Options(custom_functions=sentinel))
        assert ti._functions is sentinel

    def test_default_visit_raises_with_type(self):
        ti = TreeInterpreter()
        with pytest.raises(NotImplementedError) as excinfo:
            ti.visit({'type': 'bogus'})
        assert 'bogus' in str(excinfo.value)


class TestFieldAndIdentity:
    def test_visit_field_dict(self):
        ti = TreeInterpreter()
        assert ti.visit(_field('a'), {'a': 1, 'b': 2}) == 1

    def test_visit_field_missing(self):
        ti = TreeInterpreter()
        assert ti.visit(_field('a'), {'b': 2}) is None

    def test_visit_field_non_dict(self):
        ti = TreeInterpreter()
        assert ti.visit(_field('a'), [1, 2]) is None
        assert ti.visit(_field('a'), None) is None

    def test_visit_identity(self):
        ti = TreeInterpreter()
        assert ti.visit(_identity(), {'a': 1}) == {'a': 1}

    def test_visit_current(self):
        ti = TreeInterpreter()
        assert ti.visit(_current(), [1, 2]) == [1, 2]

    def test_visit_literal(self):
        ti = TreeInterpreter()
        assert ti.visit(_literal(42), {'ignored': True}) == 42


class TestSubexpression:
    def test_chain(self):
        ti = TreeInterpreter()
        node = {'type': 'subexpression', 'children': [_field('a'), _field('b')]}
        assert ti.visit(node, {'a': {'b': 10}}) == 10

    def test_empty_children_returns_input(self):
        ti = TreeInterpreter()
        node = {'type': 'subexpression', 'children': []}
        assert ti.visit(node, {'a': 1}) == {'a': 1}


class TestComparator:
    def test_eq(self):
        ti = TreeInterpreter()
        node = _comparator('eq', _literal(1), _literal(1))
        assert ti.visit(node, None) is True
        node = _comparator('eq', _literal(1), _literal(2))
        assert ti.visit(node, None) is False

    def test_eq_special(self):
        ti = TreeInterpreter()
        node = _comparator('eq', _literal(1), _literal(True))
        assert ti.visit(node, None) is False

    def test_ne(self):
        ti = TreeInterpreter()
        node = _comparator('ne', _literal(1), _literal(2))
        assert ti.visit(node, None) is True

    def test_lt_gt_lte_gte(self):
        ti = TreeInterpreter()
        assert ti.visit(_comparator('lt', _literal(1), _literal(2)), None) is True
        assert ti.visit(_comparator('gt', _literal(1), _literal(2)), None) is False
        assert ti.visit(_comparator('lte', _literal(2), _literal(2)), None) is True
        assert ti.visit(_comparator('gte', _literal(3), _literal(2)), None) is True

    def test_ordering_strings(self):
        ti = TreeInterpreter()
        assert ti.visit(_comparator('lt', _literal('a'), _literal('b')), None) is True

    def test_ordering_non_comparable_returns_none(self):
        ti = TreeInterpreter()
        assert ti.visit(_comparator('lt', _literal(1), _literal(True)), None) is None
        assert ti.visit(_comparator('gt', _literal(None), _literal(1)), None) is None
        assert ti.visit(_comparator('lte', _literal([]), _literal([])), None) is None


class TestIndexAndSlice:
    def test_index(self):
        ti = TreeInterpreter()
        node = {'type': 'index', 'value': 1}
        assert ti.visit(node, [10, 20, 30]) == 20

    def test_index_negative(self):
        ti = TreeInterpreter()
        node = {'type': 'index', 'value': -1}
        assert ti.visit(node, [10, 20, 30]) == 30

    def test_index_out_of_range(self):
        ti = TreeInterpreter()
        node = {'type': 'index', 'value': 5}
        assert ti.visit(node, [1, 2]) is None

    def test_index_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'index', 'value': 0}
        assert ti.visit(node, "string") is None
        assert ti.visit(node, {'a': 1}) is None

    def test_index_expression(self):
        ti = TreeInterpreter()
        node = {'type': 'index_expression', 'children': [
            _field('a'),
            {'type': 'index', 'value': 1}
        ]}
        assert ti.visit(node, {'a': [1, 2, 3]}) == 2

    def test_slice(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [1, 3, None]}
        assert ti.visit(node, [0, 1, 2, 3, 4]) == [1, 2]

    def test_slice_with_step(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [None, None, 2]}
        assert ti.visit(node, [0, 1, 2, 3]) == [0, 2]

    def test_slice_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'slice', 'children': [None, None, None]}
        assert ti.visit(node, "abc") is None


class TestFlatten:
    def test_flatten(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [_identity()]}
        assert ti.visit(node, [1, [2, 3], 4, [5]]) == [1, 2, 3, 4, 5]

    def test_flatten_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'flatten', 'children': [_identity()]}
        assert ti.visit(node, {'a': 1}) is None


class TestProjections:
    def test_projection(self):
        ti = TreeInterpreter()
        node = {'type': 'projection', 'children': [
            _identity(),
            _field('a'),
        ]}
        result = ti.visit(node, [{'a': 1}, {'a': 2}, {}])
        assert result == [1, 2]

    def test_projection_non_list(self):
        ti = TreeInterpreter()
        node = {'type': 'projection', 'children': [_identity(), _field('a')]}
        assert ti.visit(node, {'a': 1}) is None

    def test_value_projection(self):
        ti = TreeInterpreter()
        node = {'type': 'value_projection', 'children': [
            _identity(),
            _field('a'),
        ]}
        result = ti.visit(node, {'x': {'a': 1}, 'y': {'a': 2}})
        assert sorted(result) == [1, 2]

    def test_value_projection_non_dict(self):
        ti = TreeInterpreter()
        node = {'type': 'value_projection', 'children': [_identity(), _field('a')]}
        assert ti.visit(node, [1, 2]) is None

    def test_filter_projection(self):
        ti = TreeInterpreter()
        # Filter node uses children[2] as filter comparator.
        filter_node = _comparator('eq', _field('keep'), _literal(True))
        node = {
            'type': 'filter_projection',
            'children': [
                _identity(),
                _field('name'),
                filter_node,
            ],
        }
        value = [
            {'keep': True, 'name': 'a'},
            {'keep': False, 'name': 'b'},
            {'keep': True, 'name': 'c'},
        ]
        assert ti.visit(node, value) == ['a', 'c']

    def test_filter_projection_non_list(self):
        ti = TreeInterpreter()
        filter_node = _comparator('eq', _literal(1), _literal(1))
        node = {
            'type': 'filter_projection',
            'children': [_identity(), _field('name'), filter_node],
        }
        assert ti.visit(node, {'a': 1}) is None

    def test_filter_projection_skips_none(self):
        ti = TreeInterpreter()
        filter_node = _comparator('eq', _literal(1), _literal(1))
        node = {
            'type': 'filter_projection',
            'children': [_identity(), _field('missing'), filter_node],
        }
        value = [{'a': 1}, {'a': 2}]
        assert ti.visit(node, value) == []


class TestMultiSelect:
    def test_multi_select_dict(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': [
            {'type': 'key_val_pair',
             'children': [_literal('x'), _field('a')]},
            {'type': 'key_val_pair',
             'children': [_literal('y'), _field('b')]},
        ]}
        result = ti.visit(node, {'a': 1, 'b': 2})
        assert result == {'x': 1, 'y': 2}

    def test_multi_select_dict_custom_cls(self):
        ti = TreeInterpreter(Options(dict_cls=OrderedDict))
        node = {'type': 'multi_select_dict', 'children': [
            {'type': 'key_val_pair',
             'children': [_literal('x'), _field('a')]},
        ]}
        result = ti.visit(node, {'a': 1})
        assert isinstance(result, OrderedDict)
        assert result == OrderedDict([('x', 1)])

    def test_multi_select_dict_none_input(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': []}
        assert ti.visit(node, None) is None

    def test_multi_select_list(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': [
            _field('a'), _field('b'),
        ]}
        assert ti.visit(node, {'a': 1, 'b': 2}) == [1, 2]

    def test_multi_select_list_none_input(self):
        ti = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': []}
        assert ti.visit(node, None) is None


class TestLogic:
    def test_or_expression_first_truthy(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [_literal('first'), _literal('second')]}
        assert ti.visit(node, None) == 'first'

    def test_or_expression_falls_back(self):
        ti = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [_literal(''), _literal('second')]}
        assert ti.visit(node, None) == 'second'

    def test_and_expression_truthy(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [_literal('a'), _literal('b')]}
        assert ti.visit(node, None) == 'b'

    def test_and_expression_false(self):
        ti = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [_literal(''), _literal('b')]}
        assert ti.visit(node, None) == ''

    def test_not_expression(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal([])]}
        assert ti.visit(node, None) is True

    def test_not_expression_truthy(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal('x')]}
        assert ti.visit(node, None) is False

    def test_not_expression_zero(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal(0)]}
        assert ti.visit(node, None) is False

    def test_not_expression_float_zero(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal(0.0)]}
        assert ti.visit(node, None) is False

    def test_not_expression_false_bool(self):
        ti = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal(False)]}
        assert ti.visit(node, None) is True


class TestPipeAndFunction:
    def test_pipe(self):
        ti = TreeInterpreter()
        node = {'type': 'pipe', 'children': [_field('a'), _field('b')]}
        assert ti.visit(node, {'a': {'b': 5}}) == 5

    def test_function_expression(self):
        ti = TreeInterpreter()
        node = {'type': 'function_expression', 'value': 'length',
                'children': [_literal([1, 2, 3])]}
        assert ti.visit(node, None) == 3

    def test_expref(self):
        ti = TreeInterpreter()
        inner = _field('a')
        node = {'type': 'expref', 'children': [inner]}
        expr = ti.visit(node, {'a': 1})
        assert isinstance(expr, _Expression)
        assert expr.expression is inner
        assert expr.interpreter is ti
        assert expr.visit(_identity(), {'a': 1}) == {'a': 1}


class TestIsTrueFalse:
    def test_is_false(self):
        ti = TreeInterpreter()
        for val in ['', [], {}, None, False]:
            assert ti._is_false(val) is True

    def test_is_false_truthies(self):
        ti = TreeInterpreter()
        for val in ['a', [0], {'a': 1}, True, 0, 1, 0.0]:
            assert ti._is_false(val) is False

    def test_is_true(self):
        ti = TreeInterpreter()
        assert ti._is_true('a') is True
        assert ti._is_true('') is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_simple_output(self):
        gv = GraphvizVisitor()
        output = gv.visit({'type': 'root', 'value': 'v'})
        lines = output.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert 'root1 [label="root(v)"]' in lines

    def test_with_children(self):
        gv = GraphvizVisitor()
        node = {
            'type': 'root',
            'children': [
                {'type': 'child1'},
                {'type': 'child2', 'children': [{'type': 'grand'}]},
            ],
        }
        output = gv.visit(node)
        lines = output.split('\n')
        assert '  root1 -> child12' in lines
        assert '  root1 -> child23' in lines
        assert '  child23 -> grand4' in lines

    def test_missing_value_defaults_empty(self):
        gv = GraphvizVisitor()
        output = gv.visit({'type': 'x'})
        assert 'x1 [label="x()"]' in output.split('\n')

    def test_repeated_visits_accumulate_lines(self):
        gv = GraphvizVisitor()
        gv.visit({'type': 'a'})
        first_len = len(gv._lines)
        gv.visit({'type': 'b'})
        # _lines keeps growing between calls.
        assert len(gv._lines) > first_len
        # Count keeps increasing across calls.
        assert gv._count == 3
