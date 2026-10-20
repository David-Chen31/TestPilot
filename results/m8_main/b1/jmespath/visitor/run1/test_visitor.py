import collections
import pytest

from jmespath import visitor
from jmespath import functions


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_defaults_are_none(self):
        opts = visitor.Options()
        assert opts.dict_cls is None
        assert opts.custom_functions is None

    def test_custom_values_stored(self):
        custom = functions.Functions()
        opts = visitor.Options(dict_cls=collections.OrderedDict,
                               custom_functions=custom)
        assert opts.dict_cls is collections.OrderedDict
        assert opts.custom_functions is custom


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = visitor.Visitor()
        with pytest.raises(NotImplementedError) as exc:
            v.default_visit({'type': 'foo'})
        assert str(exc.value) == 'default_visit'

    def test_visit_dispatches_to_method(self):
        v = visitor.Visitor()
        # Monkey-patch a visit method on the instance class via subclass.
        class Sub(visitor.Visitor):
            def visit_mynode(self, node, *args, **kwargs):
                return ('called', node['type'], args, kwargs)

        s = Sub()
        result = s.visit({'type': 'mynode'}, 1, 2, k=3)
        assert result == ('called', 'mynode', (1, 2), {'k': 3})

    def test_visit_caches_method(self):
        v = visitor.Visitor()
        with pytest.raises(NotImplementedError):
            v.visit({'type': 'unknown_type'})
        # The default_visit bound method was cached.
        assert 'unknown_type' in v._method_cache
        cached = v._method_cache['unknown_type']
        assert cached == v.default_visit

    def test_visit_missing_node_type_key(self):
        v = visitor.Visitor()
        with pytest.raises(KeyError):
            v.visit({}, None)


# ---------------------------------------------------------------------------
# TreeInterpreter basics and initialization
# ---------------------------------------------------------------------------

class TestTreeInterpreterInit:
    def test_default_options(self):
        interp = visitor.TreeInterpreter()
        assert isinstance(interp._options, visitor.Options)
        assert interp._dict_cls is dict
        assert isinstance(interp._functions, functions.Functions)

    def test_custom_dict_cls(self):
        interp = visitor.TreeInterpreter(
            visitor.Options(dict_cls=collections.OrderedDict))
        assert interp._dict_cls is collections.OrderedDict

    def test_custom_functions(self):
        custom = functions.Functions()
        interp = visitor.TreeInterpreter(
            visitor.Options(custom_functions=custom))
        assert interp._functions is custom

    def test_default_visit_raises_with_node_type(self):
        interp = visitor.TreeInterpreter()
        with pytest.raises(NotImplementedError) as exc:
            interp.default_visit({'type': 'bogus'})
        assert str(exc.value) == 'bogus'


# ---------------------------------------------------------------------------
# Helper functions used by TreeInterpreter
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_equals_bool_int_special_case(self):
        # 0 == False normally, but special-cased
        assert visitor._equals(0, False) is False
        assert visitor._equals(0, True) is False
        assert visitor._equals(1, True) is False
        assert visitor._equals(1, False) is False
        assert visitor._equals(False, 0) is False
        assert visitor._equals(True, 1) is False

    def test_equals_regular(self):
        assert visitor._equals(2, 2) is True
        assert visitor._equals('a', 'a') is True
        assert visitor._equals([1], [1]) is True
        assert visitor._equals(2, 3) is False

    def test_equals_non_special_numbers(self):
        assert visitor._equals(2, True) is False
        assert visitor._equals(0, 0) is True
        assert visitor._equals(1, 1) is True

    def test_is_special_number_case_returns_none_when_not_special(self):
        assert visitor._is_special_number_case(2, 'x') is None
        assert visitor._is_special_number_case('x', 2) is None
        assert visitor._is_special_number_case(True, 2) is None

    def test_is_comparable(self):
        assert visitor._is_comparable(1) is True
        assert visitor._is_comparable(1.5) is True
        assert visitor._is_comparable('a') is True
        assert visitor._is_comparable(True) is False
        assert visitor._is_comparable(None) is False
        assert visitor._is_comparable([]) is False

    def test_is_actual_number(self):
        assert visitor._is_actual_number(1) is True
        assert visitor._is_actual_number(1.5) is True
        assert visitor._is_actual_number(True) is False
        assert visitor._is_actual_number(False) is False
        assert visitor._is_actual_number('1') is False


# ---------------------------------------------------------------------------
# _Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_attributes_and_visit_delegates(self):
        interp = visitor.TreeInterpreter()
        expr = visitor._Expression('n', interp)
        assert expr.expression == 'n'
        assert expr.interpreter is interp
        result = expr.visit({'type': 'literal', 'value': 5}, None)
        assert result == 5


# ---------------------------------------------------------------------------
# TreeInterpreter visit_* methods
# ---------------------------------------------------------------------------

@pytest.fixture
def interp():
    return visitor.TreeInterpreter()


def _lit(value):
    return {'type': 'literal', 'value': value}


class TestTreeInterpreterVisits:
    def test_visit_subexpression(self, interp):
        node = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        assert interp.visit(node, {'a': {'b': 42}}) == 42

    def test_visit_field(self, interp):
        node = {'type': 'field', 'value': 'foo'}
        assert interp.visit(node, {'foo': 'bar'}) == 'bar'
        assert interp.visit(node, {'baz': 'bar'}) is None
        assert interp.visit(node, None) is None
        assert interp.visit(node, 'string') is None

    def test_visit_current_and_identity(self, interp):
        assert interp.visit({'type': 'current'}, 'val') == 'val'
        assert interp.visit({'type': 'identity'}, 'val') == 'val'

    def test_visit_literal(self, interp):
        assert interp.visit(_lit([1, 2]), None) == [1, 2]
        assert interp.visit(_lit({'a': 1}), None) == {'a': 1}

    def test_visit_expref(self, interp):
        child = {'type': 'identity'}
        node = {'type': 'expref', 'children': [child]}
        expr = interp.visit(node, None)
        assert isinstance(expr, visitor._Expression)
        assert expr.expression is child
        assert expr.interpreter is interp

    def test_visit_function_expression(self, interp):
        node = {
            'type': 'function_expression',
            'value': 'length',
            'children': [_lit([1, 2, 3])],
        }
        assert interp.visit(node, None) == 3

    def test_visit_function_expression_resolves_args_in_order(self, interp):
        # contains(string, search)
        node = {
            'type': 'function_expression',
            'value': 'contains',
            'children': [_lit('foobar'), _lit('oba')],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_eq_true(self, interp):
        node = {
            'type': 'comparator',
            'value': 'eq',
            'children': [_lit(1), _lit(1)],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_eq_bool_int_special(self, interp):
        node = {
            'type': 'comparator',
            'value': 'eq',
            'children': [_lit(1), _lit(True)],
        }
        assert interp.visit(node, None) is False

    def test_visit_comparator_ne(self, interp):
        node = {
            'type': 'comparator',
            'value': 'ne',
            'children': [_lit(1), _lit(2)],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_ordering(self, interp):
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [_lit(1), _lit(2)],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_ordering_non_comparable(self, interp):
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [_lit([1]), _lit(2)],
        }
        assert interp.visit(node, None) is None

    def test_visit_comparator_ordering_strings(self, interp):
        node = {
            'type': 'comparator',
            'value': 'lt',
            'children': [_lit('a'), _lit('b')],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_gte(self, interp):
        node = {
            'type': 'comparator',
            'value': 'gte',
            'children': [_lit(2), _lit(2)],
        }
        assert interp.visit(node, None) is True

    def test_visit_comparator_gt_and_lte(self, interp):
        gt = {'type': 'comparator', 'value': 'gt',
              'children': [_lit(3), _lit(2)]}
        lte = {'type': 'comparator', 'value': 'lte',
               'children': [_lit(2), _lit(3)]}
        assert interp.visit(gt, None) is True
        assert interp.visit(lte, None) is True

    def test_visit_filter_projection(self, interp):
        # foo[?bar > `1`].baz  (loosely)
        node = {
            'type': 'filter_projection',
            'children': [
                {'type': 'field', 'value': 'items'},
                {'type': 'field', 'value': 'baz'},
                {'type': 'comparator', 'value': 'gt', 'children': [
                    {'type': 'field', 'value': 'bar'},
                    _lit(1),
                ]},
            ],
        }
        data = {'items': [
            {'bar': 2, 'baz': 'a'},
            {'bar': 0, 'baz': 'b'},
            {'bar': 5, 'baz': 'c'},
        ]}
        assert interp.visit(node, data) == ['a', 'c']

    def test_visit_filter_projection_non_list(self, interp):
        node = {
            'type': 'filter_projection',
            'children': [
                {'type': 'field', 'value': 'items'},
                {'type': 'identity'},
                {'type': 'identity'},
            ],
        }
        assert interp.visit(node, {'items': 'not-a-list'}) is None

    def test_visit_flatten(self, interp):
        node = {'type': 'flatten', 'children': [_lit([1, [2, 3], 4])]}
        assert interp.visit(node, None) == [1, 2, 3, 4]

    def test_visit_flatten_non_list(self, interp):
        node = {'type': 'flatten', 'children': [_lit('abc')]}
        assert interp.visit(node, None) is None

    def test_visit_index(self, interp):
        node = {'type': 'index', 'value': 1}
        assert interp.visit(node, ['a', 'b', 'c']) == 'b'

    def test_visit_index_out_of_range(self, interp):
        node = {'type': 'index', 'value': 5}
        assert interp.visit(node, ['a']) is None

    def test_visit_index_negative(self, interp):
        node = {'type': 'index', 'value': -1}
        assert interp.visit(node, [1, 2, 3]) == 3

    def test_visit_index_non_list(self, interp):
        node = {'type': 'index', 'value': 0}
        assert interp.visit(node, 'string') is None

    def test_visit_index_expression(self, interp):
        node = {
            'type': 'index_expression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'index', 'value': 0},
            ],
        }
        assert interp.visit(node, {'a': [10, 20]}) == 10

    def test_visit_slice(self, interp):
        node = {'type': 'slice', 'children': [1, 4, None]}
        assert interp.visit(node, [0, 1, 2, 3, 4, 5]) == [1, 2, 3]

    def test_visit_slice_with_step(self, interp):
        node = {'type': 'slice', 'children': [None, None, 2]}
        assert interp.visit(node, [0, 1, 2, 3, 4]) == [0, 2, 4]

    def test_visit_slice_non_list(self, interp):
        node = {'type': 'slice', 'children': [0, 2, None]}
        assert interp.visit(node, 'abcdef') is None

    def test_visit_key_val_pair(self, interp):
        node = {'type': 'key_val_pair',
                'children': [_lit('value')]}
        assert interp.visit(node, None) == 'value'

    def test_visit_multi_select_dict(self, interp):
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'x',
                 'children': [_lit(1)]},
                {'type': 'key_val_pair', 'value': 'y',
                 'children': [_lit(2)]},
            ],
        }
        assert interp.visit(node, {}) == {'x': 1, 'y': 2}

    def test_visit_multi_select_dict_none(self, interp):
        node = {'type': 'multi_select_dict', 'children': []}
        assert interp.visit(node, None) is None

    def test_visit_multi_select_dict_uses_custom_dict(self):
        interp = visitor.TreeInterpreter(
            visitor.Options(dict_cls=collections.OrderedDict))
        node = {
            'type': 'multi_select_dict',
            'children': [
                {'type': 'key_val_pair', 'value': 'y',
                 'children': [_lit(2)]},
                {'type': 'key_val_pair', 'value': 'x',
                 'children': [_lit(1)]},
            ],
        }
        result = interp.visit(node, {})
        assert isinstance(result, collections.OrderedDict)
        assert list(result.keys()) == ['y', 'x']

    def test_visit_multi_select_list(self, interp):
        node = {'type': 'multi_select_list',
                'children': [_lit(1), _lit(2)]}
        assert interp.visit(node, {}) == [1, 2]

    def test_visit_multi_select_list_none(self, interp):
        node = {'type': 'multi_select_list', 'children': []}
        assert interp.visit(node, None) is None

    def test_visit_or_expression_false_left(self, interp):
        node = {'type': 'or_expression',
                'children': [_lit(None), _lit('fallback')]}
        assert interp.visit(node, None) == 'fallback'

    def test_visit_or_expression_true_left(self, interp):
        node = {'type': 'or_expression',
                'children': [_lit('value'), _lit('fallback')]}
        assert interp.visit(node, None) == 'value'

    def test_visit_and_expression_false_left(self, interp):
        node = {'type': 'and_expression',
                'children': [_lit(None), _lit('value')]}
        assert interp.visit(node, None) is None

    def test_visit_and_expression_true_left(self, interp):
        node = {'type': 'and_expression',
                'children': [_lit('a'), _lit('b')]}
        assert interp.visit(node, None) == 'b'

    def test_visit_not_expression_zero(self, interp):
        node = {'type': 'not_expression', 'children': [_lit(0)]}
        assert interp.visit(node, None) is False

    def test_visit_not_expression_nonzero(self, interp):
        node = {'type': 'not_expression', 'children': [_lit(5)]}
        assert interp.visit(node, None) is False

    def test_visit_not_expression_empty(self, interp):
        node = {'type': 'not_expression', 'children': [_lit('')]}
        assert interp.visit(node, None) is True

    def test_visit_not_expression_none(self, interp):
        node = {'type': 'not_expression', 'children': [_lit(None)]}
        assert interp.visit(node, None) is True

    def test_visit_pipe(self, interp):
        node = {
            'type': 'pipe',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        assert interp.visit(node, {'a': {'b': 7}}) == 7

    def test_visit_projection(self, interp):
        node = {
            'type': 'projection',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        assert interp.visit(node, {'a': [{'b': 1}, {'b': 2}]}) == [1, 2]

    def test_visit_projection_filters_none(self, interp):
        node = {
            'type': 'projection',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        assert interp.visit(node, {'a': [{'b': 1}, {'c': 2}]}) == [1]

    def test_visit_projection_non_list(self, interp):
        node = {
            'type': 'projection',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'identity'},
            ],
        }
        assert interp.visit(node, {'a': 'oops'}) is None

    def test_visit_value_projection(self, interp):
        node = {
            'type': 'value_projection',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        data = {'a': {'x': {'b': 1}, 'y': {'b': 2}}}
        assert interp.visit(node, data) == [1, 2]

    def test_visit_value_projection_non_dict(self, interp):
        node = {
            'type': 'value_projection',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'identity'},
            ],
        }
        assert interp.visit(node, {'a': [1, 2]}) is None

    def test_is_false_rules(self, interp):
        assert interp._is_false('') is True
        assert interp._is_false([]) is True
        assert interp._is_false({}) is True
        assert interp._is_false(None) is True
        assert interp._is_false(False) is True
        assert interp._is_false(0) is False
        assert interp._is_false('a') is False
        assert interp._is_false([0]) is False

    def test_is_true_rules(self, interp):
        assert interp._is_true('') is False
        assert interp._is_true(0) is True
        assert interp._is_true('a') is True


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_simple_visit(self):
        gv = visitor.GraphvizVisitor()
        node = {'type': 'identity'}
        result = gv.visit(node)
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'
        assert any('identity1' in ln for ln in lines)

    def test_label_with_value(self):
        gv = visitor.GraphvizVisitor()
        node = {'type': 'field', 'value': 'foo'}
        result = gv.visit(node)
        assert 'field1 [label="field(foo)"]' in result

    def test_label_without_value(self):
        gv = visitor.GraphvizVisitor()
        node = {'type': 'identity'}
        result = gv.visit(node)
        assert 'identity1 [label="identity()"]' in result

    def test_children_and_edges(self):
        gv = visitor.GraphvizVisitor()
        node = {
            'type': 'field',
            'value': 'a',
            'children': [
                {'type': 'identity'},
                {'type': 'literal', 'value': 5},
            ],
        }
        result = gv.visit(node)
        lines = result.split('\n')
        # root gets field1, children then identity2 and literal3
        assert 'field1 [label="field(a)"]' in lines
        assert '  field1 -> identity2' in lines
        assert '  field1 -> literal3' in lines
        assert 'identity2 [label="identity()"]' in lines
        assert 'literal3 [label="literal(5)"]' in lines

    def test_visit_increments_count(self):
        gv = visitor.GraphvizVisitor()
        assert gv._count == 1
        gv.visit({'type': 'identity'})
        assert gv._count == 2

    def test_recursive_nesting(self):
        gv = visitor.GraphvizVisitor()
        node = {
            'type': 'identity',
            'children': [
                {'type': 'identity',
                 'children': [{'type': 'identity'}]},
            ],
        }
        result = gv.visit(node)
        assert 'identity1' in result
        assert 'identity2' in result
        assert 'identity3' in result
        assert '  identity1 -> identity2' in result
        assert '  identity2 -> identity3' in result
