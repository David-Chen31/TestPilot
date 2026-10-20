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
# Module-level helper functions
# ---------------------------------------------------------------------------

class TestIsActualNumber:
    def test_int(self):
        assert _is_actual_number(0) is True

    def test_float(self):
        assert _is_actual_number(1.5) is True

    def test_bool_true(self):
        assert _is_actual_number(True) is False

    def test_bool_false(self):
        assert _is_actual_number(False) is False

    def test_string(self):
        assert _is_actual_number('1') is False

    def test_none(self):
        assert _is_actual_number(None) is False

    def test_list(self):
        assert _is_actual_number([1]) is False


class TestIsComparable:
    def test_numbers(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True

    def test_string(self):
        assert _is_comparable('foo') is True

    def test_bool(self):
        assert _is_comparable(True) is False

    def test_none(self):
        assert _is_comparable(None) is False

    def test_list(self):
        assert _is_comparable([]) is False


class TestIsSpecialNumberCase:
    def test_zero_vs_false(self):
        assert _is_special_number_case(0, False) is True

    def test_zero_vs_true(self):
        assert _is_special_number_case(0, True) is True

    def test_one_vs_true(self):
        assert _is_special_number_case(1, True) is True

    def test_one_vs_false(self):
        assert _is_special_number_case(1, False) is True

    def test_swapped(self):
        assert _is_special_number_case(False, 0) is True
        assert _is_special_number_case(True, 1) is True

    def test_non_special_number_returns_none(self):
        assert _is_special_number_case(2, True) is None

    def test_number_vs_number_returns_none(self):
        assert _is_special_number_case(0, 1) is None

    def test_non_number_returns_none(self):
        assert _is_special_number_case('a', True) is None


class TestEquals:
    def test_zero_false_not_equal(self):
        assert _equals(0, False) is False

    def test_one_true_not_equal(self):
        assert _equals(1, True) is False

    def test_false_zero_not_equal(self):
        assert _equals(False, 0) is False

    def test_normal_numbers(self):
        assert _equals(1, 1) is True
        assert _equals(1, 2) is False

    def test_strings(self):
        assert _equals('a', 'a') is True
        assert _equals('a', 'b') is False

    def test_none(self):
        assert _equals(None, None) is True


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults(self):
        opts = Options()
        assert opts.dict_cls is None
        assert opts.custom_functions is None

    def test_explicit_values(self):
        funcs = object()
        opts = Options(dict_cls=dict, custom_functions=funcs)
        assert opts.dict_cls is dict
        assert opts.custom_functions is funcs


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class _SubVisitor(Visitor):
    def visit_foo(self, node, *args, **kwargs):
        return ('foo', args, kwargs)


class TestVisitor:
    def test_caches_method_lookup(self):
        v = _SubVisitor()
        node = {'type': 'foo'}
        result = v.visit(node, 1, key='x')
        assert result == ('foo', (1,), {'key': 'x'})
        # Second call uses cache but still returns same result.
        assert v.visit(node, 2) == ('foo', (2,), {})
        assert 'foo' in v._method_cache

    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'unknown'})

    def test_default_visit_directly(self):
        with pytest.raises(NotImplementedError):
            Visitor().default_visit({'type': 'x'})


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class _RecordingInterpreter:
    def __init__(self):
        self.calls = []

    def visit(self, node, *args, **kwargs):
        self.calls.append((node, args, kwargs))
        return 'visited'


class TestExpression:
    def test_delegates_to_interpreter(self):
        interp = _RecordingInterpreter()
        expr = _Expression('expr', interp)
        assert expr.expression == 'expr'
        assert expr.interpreter is interp
        result = expr.visit({'type': 'x'}, 1, k=2)
        assert result == 'visited'
        assert interp.calls == [({'type': 'x'}, (1,), {'k': 2})]


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

def _field(name):
    return {'type': 'field', 'value': name}


def _literal(v):
    return {'type': 'literal', 'value': v}


class TestTreeInterpreterInit:
    def test_defaults(self):
        interp = TreeInterpreter()
        assert interp._dict_cls is dict
        assert isinstance(interp._functions, functions.Functions)

    def test_custom_dict_cls(self):
        from collections import OrderedDict
        interp = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert interp._dict_cls is OrderedDict

    def test_custom_functions(self):
        funcs = object()
        interp = TreeInterpreter(Options(custom_functions=funcs))
        assert interp._functions is funcs

    def test_default_visit_raises_with_type_name(self):
        interp = TreeInterpreter()
        with pytest.raises(NotImplementedError) as excinfo:
            interp.default_visit({'type': 'weird'})
        assert 'weird' in str(excinfo.value)


class TestTreeInterpreterBasics:
    def test_identity(self):
        interp = TreeInterpreter()
        assert interp.visit_identity({}, 'v') == 'v'

    def test_current(self):
        interp = TreeInterpreter()
        assert interp.visit_current({}, {'a': 1}) == {'a': 1}

    def test_literal(self):
        interp = TreeInterpreter()
        assert interp.visit_literal({'value': [1, 2]}, 'ignored') == [1, 2]

    def test_field_on_dict(self):
        interp = TreeInterpreter()
        assert interp.visit_field(_field('a'), {'a': 1}) == 1

    def test_field_missing(self):
        interp = TreeInterpreter()
        assert interp.visit_field(_field('b'), {'a': 1}) is None

    def test_field_on_non_dict(self):
        interp = TreeInterpreter()
        assert interp.visit_field(_field('a'), [1, 2]) is None

    def test_subexpression(self):
        interp = TreeInterpreter()
        node = {'type': 'subexpression',
                'children': [_field('a'), _field('b')]}
        assert interp.visit_subexpression(node, {'a': {'b': 5}}) == 5

    def test_index_expression(self):
        interp = TreeInterpreter()
        node = {'type': 'index_expression',
                'children': [_field('a'), {'type': 'index', 'value': 0}]}
        assert interp.visit_index_expression(node, {'a': [10, 20]}) == 10


class TestTreeInterpreterComparators:
    def test_eq(self):
        interp = TreeInterpreter()
        node = {'type': 'comparator', 'value': 'eq',
                'children': [_literal(1), _literal(1)]}
        assert interp.visit_comparator(node, None) is True

    def test_ne(self):
        interp = TreeInterpreter()
        node = {'type': 'comparator', 'value': 'ne',
                'children': [_literal(1), _literal(2)]}
        assert interp.visit_comparator(node, None) is True

    def test_eq_zero_vs_false(self):
        interp = TreeInterpreter()
        node = {'type': 'comparator', 'value': 'eq',
                'children': [_literal(0), _literal(False)]}
        assert interp.visit_comparator(node, None) is False

    @pytest.mark.parametrize('op,left,right,expected', [
        ('lt', 1, 2, True),
        ('gt', 1, 2, False),
        ('lte', 2, 2, True),
        ('gte', 1, 2, False),
        ('lt', 'a', 'b', True),
    ])
    def test_ordering(self, op, left, right, expected):
        interp = TreeInterpreter()
        node = {'type': 'comparator', 'value': op,
                'children': [_literal(left), _literal(right)]}
        assert interp.visit_comparator(node, None) is expected

    def test_ordering_non_comparable_returns_none(self):
        interp = TreeInterpreter()
        node = {'type': 'comparator', 'value': 'lt',
                'children': [_literal(None), _literal(1)]}
        assert interp.visit_comparator(node, None) is None


class TestTreeInterpreterLogic:
    def test_or_expression_first_truthy(self):
        interp = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [_literal('a'), _literal('b')]}
        assert interp.visit_or_expression(node, None) == 'a'

    def test_or_expression_first_falsy(self):
        interp = TreeInterpreter()
        node = {'type': 'or_expression',
                'children': [_literal(''), _literal('b')]}
        assert interp.visit_or_expression(node, None) == 'b'

    def test_and_expression_first_falsy(self):
        interp = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [_literal(''), _literal('b')]}
        assert interp.visit_and_expression(node, None) == ''

    def test_and_expression_both_truthy(self):
        interp = TreeInterpreter()
        node = {'type': 'and_expression',
                'children': [_literal('a'), _literal('b')]}
        assert interp.visit_and_expression(node, None) == 'b'

    def test_not_expression(self):
        interp = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal('')]}
        assert interp.visit_not_expression(node, None) is True

    def test_not_expression_zero_is_false(self):
        interp = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal(0)]}
        assert interp.visit_not_expression(node, None) is False

    def test_not_expression_nonzero_number(self):
        interp = TreeInterpreter()
        node = {'type': 'not_expression', 'children': [_literal(5)]}
        assert interp.visit_not_expression(node, None) is False

    def test_is_false_values(self):
        interp = TreeInterpreter()
        for v in ['', [], {}, None, False]:
            assert interp._is_false(v) is True

    def test_is_false_non_values(self):
        interp = TreeInterpreter()
        for v in [0, 1, 'x', [1], {'a': 1}, True]:
            assert interp._is_false(v) is False

    def test_is_true(self):
        interp = TreeInterpreter()
        assert interp._is_true('x') is True
        assert interp._is_true('') is False


class TestTreeInterpreterProjections:
    def test_projection(self):
        interp = TreeInterpreter()
        node = {'type': 'projection',
                'children': [_literal([{'a': 1}, {'a': None}, {'a': 3}]),
                             _field('a')]}
        assert interp.visit_projection(node, None) == [1, 3]

    def test_projection_non_list(self):
        interp = TreeInterpreter()
        node = {'type': 'projection',
                'children': [_literal('notalist'), _field('a')]}
        assert interp.visit_projection(node, None) is None

    def test_value_projection(self):
        interp = TreeInterpreter()
        node = {'type': 'value_projection',
                'children': [_literal({'x': {'a': 1}, 'y': {'a': 2}}),
                             _field('a')]}
        assert sorted(interp.visit_value_projection(node, None)) == [1, 2]

    def test_value_projection_non_dict(self):
        interp = TreeInterpreter()
        node = {'type': 'value_projection',
                'children': [_literal([1, 2]), _field('a')]}
        assert interp.visit_value_projection(node, None) is None

    def test_filter_projection(self):
        interp = TreeInterpreter()
        comparator = {'type': 'comparator', 'value': 'eq',
                      'children': [_field('x'), _literal(1)]}
        node = {'type': 'filter_projection',
                'children': [_literal([{'x': 1, 'y': 10},
                                       {'x': 2, 'y': 20}]),
                             _field('y'),
                             comparator]}
        assert interp.visit_filter_projection(node, None) == [10]

    def test_filter_projection_non_list(self):
        interp = TreeInterpreter()
        comparator = {'type': 'comparator', 'value': 'eq',
                      'children': [_field('x'), _literal(1)]}
        node = {'type': 'filter_projection',
                'children': [_literal({'a': 1}), _field('y'), comparator]}
        assert interp.visit_filter_projection(node, None) is None


class TestTreeInterpreterFlatten:
    def test_flatten(self):
        interp = TreeInterpreter()
        node = {'type': 'flatten',
                'children': [_literal([1, [2, 3], [4], 5])]}
        assert interp.visit_flatten(node, None) == [1, 2, 3, 4, 5]

    def test_flatten_non_list(self):
        interp = TreeInterpreter()
        node = {'type': 'flatten', 'children': [_literal('abc')]}
        assert interp.visit_flatten(node, None) is None


class TestTreeInterpreterIndexSlice:
    def test_index(self):
        interp = TreeInterpreter()
        assert interp.visit_index({'value': 0}, [10, 20]) == 10

    def test_index_negative(self):
        interp = TreeInterpreter()
        assert interp.visit_index({'value': -1}, [10, 20]) == 20

    def test_index_out_of_range(self):
        interp = TreeInterpreter()
        assert interp.visit_index({'value': 5}, [10, 20]) is None

    def test_index_non_list(self):
        interp = TreeInterpreter()
        assert interp.visit_index({'value': 0}, 'ab') is None

    def test_slice(self):
        interp = TreeInterpreter()
        node = {'children': [0, 3, 1]}
        assert interp.visit_slice(node, [10, 20, 30, 40]) == [10, 20, 30]

    def test_slice_non_list(self):
        interp = TreeInterpreter()
        assert interp.visit_slice({'children': [0, 1, None]}, 'abc') is None


class TestTreeInterpreterMultiSelect:
    def test_multi_select_dict(self):
        interp = TreeInterpreter()
        node = {'type': 'multi_select_dict',
                'children': [_field('a'), _field('b')]}
        assert interp.visit_multi_select_dict(
            node, {'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_multi_select_dict_none(self):
        interp = TreeInterpreter()
        node = {'type': 'multi_select_dict', 'children': [_field('a')]}
        assert interp.visit_multi_select_dict(node, None) is None

    def test_multi_select_dict_uses_custom_cls(self):
        from collections import OrderedDict
        interp = TreeInterpreter(Options(dict_cls=OrderedDict))
        node = {'type': 'multi_select_dict',
                'children': [_field('b'), _field('a')]}
        result = interp.visit_multi_select_dict(node, {'a': 1, 'b': 2})
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ['b', 'a']

    def test_multi_select_list(self):
        interp = TreeInterpreter()
        node = {'type': 'multi_select_list',
                'children': [_field('a'), _field('b')]}
        assert interp.visit_multi_select_list(
            node, {'a': 1, 'b': 2}) == [1, 2]

    def test_multi_select_list_none(self):
        interp = TreeInterpreter()
        node = {'type': 'multi_select_list', 'children': [_field('a')]}
        assert interp.visit_multi_select_list(node, None) is None

    def test_key_val_pair(self):
        interp = TreeInterpreter()
        node = {'type': 'key_val_pair',
                'children': [_field('a'), _literal(1)]}
        assert interp.visit_key_val_pair(node, {'a': 5}) == 5


class TestTreeInterpreterMisc:
    def test_pipe(self):
        interp = TreeInterpreter()
        node = {'type': 'pipe', 'children': [_field('a'), _field('b')]}
        assert interp.visit_pipe(node, {'a': {'b': 7}}) == 7

    def test_expref(self):
        interp = TreeInterpreter()
        node = {'type': 'expref', 'children': [_literal(1)]}
        result = interp.visit_expref(node, None)
        assert isinstance(result, _Expression)
        assert result.interpreter is interp

    def test_function_expression(self):
        interp = TreeInterpreter()
        interp._functions = _FuncRecorder()
        node = {'type': 'function_expression',
                'value': 'myfunc',
                'children': [_literal(1), _literal(2)]}
        assert interp.visit_function_expression(node, None) == 'myfunc:1,2'
        assert interp._functions.calls == [('myfunc', [1, 2])]


class _FuncRecorder:
    def __init__(self):
        self.calls = []

    def call_function(self, name, args):
        self.calls.append((name, args))
        return '%s:%s' % (name, ','.join(str(a) for a in args))


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_single_node(self):
        gv = GraphvizVisitor()
        output = gv.visit({'type': 'field', 'value': 'foo'})
        lines = output.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[1] == 'field1 [label="field(foo)"]'
        assert lines[-1] == '}'

    def test_with_children(self):
        gv = GraphvizVisitor()
        node = {'type': 'subexpression',
                'children': [
                    {'type': 'field', 'value': 'a'},
                    {'type': 'field', 'value': 'b'},
                ]}
        output = gv.visit(node)
        lines = output.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[1] == 'subexpression1 [label="subexpression()"]'
        assert lines[2] == '  subexpression1 -> field2'
        assert lines[3] == 'field2 [label="field(a)"]'
        assert lines[4] == '  subexpression1 -> field3'
        assert lines[5] == 'field3 [label="field(b)"]'
        assert lines[-1] == '}'

    def test_counter_persists_between_calls(self):
        gv = GraphvizVisitor()
        gv.visit({'type': 'field', 'value': 'a'})
        output = gv.visit({'type': 'field', 'value': 'b'})
        # Second render uses field2 since _count increments.
        assert 'field2 [label="field(b)"]' in output
