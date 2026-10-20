import pytest
from collections import OrderedDict

from jmespath import visitor
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
# Helpers
# ---------------------------------------------------------------------------

def node(type_, **kwargs):
    d = {'type': type_}
    d.update(kwargs)
    return d


# ---------------------------------------------------------------------------
# _equals / _is_special_number_case / _is_comparable / _is_actual_number
# ---------------------------------------------------------------------------

class TestEquals:
    def test_basic_equal(self):
        assert _equals(1, 1) is True
        assert _equals('a', 'a') is True

    def test_basic_unequal(self):
        assert _equals(1, 2) is False
        assert _equals('a', 'b') is False

    def test_zero_vs_false(self):
        assert _equals(0, False) is False
        assert _equals(False, 0) is False

    def test_one_vs_true(self):
        assert _equals(1, True) is False
        assert _equals(True, 1) is False

    def test_zero_vs_true(self):
        assert _equals(0, True) is False
        assert _equals(True, 0) is False

    def test_one_vs_false(self):
        assert _equals(1, False) is False

    def test_non_special_numbers(self):
        # 2 vs True is not special-cased; plain equality applies.
        assert _equals(2, True) is False
        assert _equals(True, 2) is False

    def test_numbers_not_bools(self):
        assert _equals(0, 0) is True
        assert _equals(1, 1) is True


class TestIsSpecialNumberCase:
    def test_zero_bool(self):
        assert _is_special_number_case(0, False) is True
        assert _is_special_number_case(0, True) is True

    def test_one_bool(self):
        assert _is_special_number_case(1, True) is True
        assert _is_special_number_case(1, False) is True

    def test_reverse(self):
        assert _is_special_number_case(True, 0) is True
        assert _is_special_number_case(False, 1) is True

    def test_not_special(self):
        # Comparing number to number
        assert _is_special_number_case(0, 1) is None
        assert _is_special_number_case(0, 0) is None

    def test_number_nonzero_with_bool(self):
        # 2 is not in (0, 1) so no special case.
        assert _is_special_number_case(2, True) is None


class TestIsComparable:
    def test_numbers(self):
        assert _is_comparable(1) is True
        assert _is_comparable(1.5) is True

    def test_strings(self):
        assert _is_comparable('foo') is True

    def test_bools(self):
        assert _is_comparable(True) is False

    def test_others(self):
        assert _is_comparable(None) is False
        assert _is_comparable([1]) is False
        assert _is_comparable({}) is False


class TestIsActualNumber:
    def test_ints_floats(self):
        assert _is_actual_number(1) is True
        assert _is_actual_number(1.5) is True

    def test_bools(self):
        assert _is_actual_number(True) is False
        assert _is_actual_number(False) is False

    def test_others(self):
        assert _is_actual_number('1') is False
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
        o = Options(dict_cls=OrderedDict, custom_functions='cf')
        assert o.dict_cls is OrderedDict
        assert o.custom_functions == 'cf'


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError):
            v.visit(node('unknown'))

    def test_visit_dispatch_and_cache(self):
        class V(Visitor):
            def visit_foo(self, n, *args, **kwargs):
                return ('foo', n, args, kwargs)

        v = V()
        result = v.visit(node('foo', value=1), 'x', k='y')
        assert result[0] == 'foo'
        assert result[2] == ('x',)
        assert result[3] == {'k': 'y'}
        # cache should now contain 'foo'
        assert 'foo' in v._method_cache


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_visit_delegates(self):
        class FakeInterp:
            def __init__(self):
                self.calls = []

            def visit(self, n, *args, **kwargs):
                self.calls.append((n, args, kwargs))
                return 'ok'

        interp = FakeInterp()
        e = _Expression('expr', interp)
        assert e.expression == 'expr'
        assert e.visit('N', 1, a=2) == 'ok'
        assert interp.calls == [('N', (1,), {'a': 2})]


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

class TestTreeInterpreter:
    def setup_method(self):
        self.t = TreeInterpreter()
        self.data = {
            'foo': {'bar': 'baz'},
            'list': [1, 2, 3, None, 4],
            'nested': [[1, 2], [3, [4, 5]], 6],
            'objs': {'a': {'v': 1}, 'b': {'v': 2}, 'c': {'v': None}},
        }

    # -- constructor / options --

    def test_default_dict_cls(self):
        assert self.t._dict_cls is dict

    def test_custom_dict_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert t._dict_cls is OrderedDict

    def test_custom_functions(self):
        t = TreeInterpreter(Options(custom_functions='cf'))
        assert t._functions == 'cf'

    def test_default_functions_created(self):
        from jmespath import functions
        assert isinstance(self.t._functions, functions.Functions)

    def test_default_visit_raises_type(self):
        with pytest.raises(NotImplementedError) as exc:
            self.t.visit(node('nope'))
        assert 'nope' in str(exc.value)

    # -- subexpression / field / current / identity --

    def test_visit_subexpression(self):
        n = node('subexpression', children=[
            node('field', value='foo'),
            node('field', value='bar'),
        ])
        assert self.t.visit(n, self.data) == 'baz'

    def test_visit_field_present(self):
        assert self.t.visit(node('field', value='foo'), self.data) == {'bar': 'baz'}

    def test_visit_field_missing(self):
        assert self.t.visit(node('field', value='nope'), self.data) is None

    def test_visit_field_non_dict(self):
        assert self.t.visit(node('field', value='foo'), 5) is None

    def test_visit_current(self):
        assert self.t.visit(node('current'), 'val') == 'val'

    def test_visit_identity(self):
        assert self.t.visit(node('identity'), 'val') == 'val'

    # -- comparators --

    def test_visit_comparator_eq(self):
        n = node('comparator', value='eq', children=[
            node('literal', value=1), node('literal', value=1)])
        assert self.t.visit(n, {}) is True

    def test_visit_comparator_ne(self):
        n = node('comparator', value='ne', children=[
            node('literal', value=1), node('literal', value=2)])
        assert self.t.visit(n, {}) is True

    def test_visit_comparator_lt(self):
        n = node('comparator', value='lt', children=[
            node('literal', value=1), node('literal', value=2)])
        assert self.t.visit(n, {}) is True

    def test_visit_comparator_gt(self):
        n = node('comparator', value='gt', children=[
            node('literal', value=1), node('literal', value=2)])
        assert self.t.visit(n, {}) is False

    def test_visit_comparator_lte(self):
        n = node('comparator', value='lte', children=[
            node('literal', value=2), node('literal', value=2)])
        assert self.t.visit(n, {}) is True

    def test_visit_comparator_gte(self):
        n = node('comparator', value='gte', children=[
            node('literal', value=3), node('literal', value=2)])
        assert self.t.visit(n, {}) is True

    def test_visit_comparator_ordering_non_comparable(self):
        n = node('comparator', value='lt', children=[
            node('literal', value=[1]), node('literal', value=2)])
        assert self.t.visit(n, {}) is None

    def test_visit_comparator_ordering_strings(self):
        n = node('comparator', value='lt', children=[
            node('literal', value='a'), node('literal', value='b')])
        assert self.t.visit(n, {}) is True

    # -- expref --

    def test_visit_expref(self):
        child = node('identity')
        n = node('expref', children=[child])
        result = self.t.visit(n, {})
        assert isinstance(result, _Expression)
        assert result.interpreter is self.t
        assert result.expression is child

    # -- function expression --

    def test_visit_function_expression(self):
        class FakeFuncs:
            def __init__(self):
                self.calls = []

            def call_function(self, name, args):
                self.calls.append((name, args))
                return 'ret'

        t = TreeInterpreter(Options(custom_functions=FakeFuncs()))
        n = node('function_expression', value='myfn', children=[
            node('literal', value=1), node('literal', value=2)])
        assert t.visit(n, {}) == 'ret'
        assert t._functions.calls == [('myfn', [1, 2])]

    # -- filter projection --

    def test_visit_filter_projection(self):
        # filter list where element > 2
        cmp_node = node('comparator', value='gt', children=[
            node('current'), node('literal', value=2)])
        n = node('filter_projection', children=[
            node('field', value='list'),
            node('current'),
            cmp_node,
        ])
        data = {'list': [1, 2, 3, 4]}
        assert self.t.visit(n, data) == [3, 4]

    def test_visit_filter_projection_drops_none(self):
        cmp_node = node('literal', value=True)
        n = node('filter_projection', children=[
            node('field', value='list'),
            node('current'),
            cmp_node,
        ])
        data = {'list': [1, None, 2]}
        assert self.t.visit(n, data) == [1, 2]

    def test_visit_filter_projection_non_list(self):
        n = node('filter_projection', children=[
            node('field', value='foo'),
            node('current'),
            node('literal', value=True),
        ])
        assert self.t.visit(n, {'foo': 'x'}) is None

    # -- flatten --

    def test_visit_flatten(self):
        n = node('flatten', children=[node('field', value='nested')])
        data = {'nested': [[1, 2], [3, [4, 5]], 6]}
        assert self.t.visit(n, data) == [1, 2, 3, [4, 5], 6]

    def test_visit_flatten_non_list(self):
        n = node('flatten', children=[node('field', value='foo')])
        assert self.t.visit(n, {'foo': 'x'}) is None

    # -- index --

    def test_visit_index(self):
        n = node('index', value=1)
        assert self.t.visit(n, [10, 20, 30]) == 20

    def test_visit_index_negative(self):
        n = node('index', value=-1)
        assert self.t.visit(n, [10, 20, 30]) == 30

    def test_visit_index_out_of_range(self):
        n = node('index', value=5)
        assert self.t.visit(n, [10, 20]) is None

    def test_visit_index_non_list(self):
        n = node('index', value=0)
        assert self.t.visit(n, 'string') is None

    def test_visit_index_expression(self):
        n = node('index_expression', children=[
            node('field', value='list'), node('index', value=0)])
        assert self.t.visit(n, {'list': [7, 8]}) == 7

    # -- slice --

    def test_visit_slice(self):
        n = node('slice', children=[0, 2, None])
        assert self.t.visit(n, [1, 2, 3, 4]) == [1, 2]

    def test_visit_slice_step(self):
        n = node('slice', children=[None, None, 2])
        assert self.t.visit(n, [0, 1, 2, 3, 4]) == [0, 2, 4]

    def test_visit_slice_non_list(self):
        n = node('slice', children=[0, 1, None])
        assert self.t.visit(n, 'string') is None

    # -- key_val_pair / literal --

    def test_visit_key_val_pair(self):
        n = node('key_val_pair', children=[node('literal', value='k')])
        assert self.t.visit(n, {}) == 'k'

    def test_visit_literal(self):
        assert self.t.visit(node('literal', value=42), {}) == 42

    # -- multi_select --

    def test_visit_multi_select_dict(self):
        n = node('multi_select_dict', children=[
            node('field', value='foo'),
            node('field', value='bar'),
        ])
        # Use a field-based children node with value attr
        n = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'field', 'value': 'foo'},
                {'type': 'field', 'value': 'bar'},
            ],
        }
        data = {'foo': 1, 'bar': 2}
        assert self.t.visit(n, data) == {'foo': 1, 'bar': 2}

    def test_visit_multi_select_dict_none(self):
        n = {'type': 'multi_select_dict', 'children': [
            {'type': 'field', 'value': 'foo'}]}
        assert self.t.visit(n, None) is None

    def test_visit_multi_select_dict_custom_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        n = {'type': 'multi_select_dict', 'children': [
            {'type': 'field', 'value': 'foo'},
            {'type': 'field', 'value': 'bar'},
        ]}
        result = t.visit(n, {'foo': 1, 'bar': 2})
        assert isinstance(result, OrderedDict)
        assert result == OrderedDict([('foo', 1), ('bar', 2)])

    def test_visit_multi_select_list(self):
        n = {'type': 'multi_select_list', 'children': [
            {'type': 'field', 'value': 'foo'},
            {'type': 'field', 'value': 'bar'},
        ]}
        assert self.t.visit(n, {'foo': 1, 'bar': 2}) == [1, 2]

    def test_visit_multi_select_list_none(self):
        n = {'type': 'multi_select_list', 'children': [
            {'type': 'field', 'value': 'foo'}]}
        assert self.t.visit(n, None) is None

    # -- or / and / not --

    def test_visit_or_expression_first_truthy(self):
        n = node('or_expression', children=[
            node('literal', value='first'),
            node('literal', value='second')])
        assert self.t.visit(n, {}) == 'first'

    def test_visit_or_expression_first_falsy(self):
        n = node('or_expression', children=[
            node('literal', value=None),
            node('literal', value='second')])
        assert self.t.visit(n, {}) == 'second'

    def test_visit_and_expression_first_falsy(self):
        n = node('and_expression', children=[
            node('literal', value=None),
            node('literal', value='second')])
        assert self.t.visit(n, {}) is None

    def test_visit_and_expression_first_truthy(self):
        n = node('and_expression', children=[
            node('literal', value='a'),
            node('literal', value='b')])
        assert self.t.visit(n, {}) == 'b'

    def test_visit_not_expression_true(self):
        n = node('not_expression', children=[node('literal', value=None)])
        assert self.t.visit(n, {}) is True

    def test_visit_not_expression_false(self):
        n = node('not_expression', children=[node('literal', value='x')])
        assert self.t.visit(n, {}) is False

    def test_visit_not_expression_zero(self):
        n = node('not_expression', children=[node('literal', value=0)])
        assert self.t.visit(n, {}) is False

    def test_visit_not_expression_one(self):
        n = node('not_expression', children=[node('literal', value=1)])
        assert self.t.visit(n, {}) is False

    def test_visit_not_expression_empty_list(self):
        n = node('not_expression', children=[node('literal', value=[])])
        assert self.t.visit(n, {}) is True

    # -- pipe --

    def test_visit_pipe(self):
        n = node('pipe', children=[
            node('field', value='foo'),
            node('field', value='bar'),
        ])
        assert self.t.visit(n, {'foo': {'bar': 'baz'}}) == 'baz'

    # -- projection --

    def test_visit_projection(self):
        n = node('projection', children=[
            node('field', value='items'),
            node('field', value='name'),
        ])
        data = {'items': [{'name': 'a'}, {'name': 'b'}, {'other': 1}]}
        assert self.t.visit(n, data) == ['a', 'b']

    def test_visit_projection_non_list(self):
        n = node('projection', children=[
            node('field', value='items'),
            node('field', value='name'),
        ])
        assert self.t.visit(n, {'items': 'x'}) is None

    def test_visit_value_projection(self):
        n = node('value_projection', children=[
            node('field', value='objs'),
            node('field', value='v'),
        ])
        assert self.t.visit(n, self.data) == [1, 2]

    def test_visit_value_projection_non_dict(self):
        n = node('value_projection', children=[
            node('field', value='foo'),
            node('field', value='v'),
        ])
        assert self.t.visit(n, {'foo': 'string'}) is None

    # -- _is_false / _is_true --

    def test_is_false(self):
        assert self.t._is_false('') is True
        assert self.t._is_false([]) is True
        assert self.t._is_false({}) is True
        assert self.t._is_false(None) is True
        assert self.t._is_false(False) is True
        assert self.t._is_false(0) is False
        assert self.t._is_false('a') is False
        assert self.t._is_false([0]) is False

    def test_is_true(self):
        assert self.t._is_true(0) is True
        assert self.t._is_true('') is False
        assert self.t._is_true(False) is False
        assert self.t._is_true(None) is False
        assert self.t._is_true([1]) is True


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_simple(self):
        g = GraphvizVisitor()
        result = g.visit(node('field', value='foo'))
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert 'field1 [label="field(foo)"]' in lines

    def test_with_children(self):
        g = GraphvizVisitor()
        n = node('subexpression', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        result = g.visit(n)
        assert 'subexpression1 [label="subexpression()"]' in result
        assert 'field2 [label="field(a)"]' in result
        assert 'field3 [label="field(b)"]' in result
        assert '  subexpression1 -> field2' in result
        assert '  subexpression1 -> field3' in result

    def test_count_increments_across_calls(self):
        g = GraphvizVisitor()
        r1 = g.visit(node('field', value='x'))
        r2 = g.visit(node('field', value='y'))
        assert 'field1' in r1
        assert 'field2' in r2

    def test_node_without_value(self):
        g = GraphvizVisitor()
        result = g.visit(node('identity'))
        assert 'identity1 [label="identity()"]' in result
