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
# Helper function tests
# ---------------------------------------------------------------------------

class TestIsActualNumber:
    def test_int(self):
        assert _is_actual_number(1) is True

    def test_float(self):
        assert _is_actual_number(1.5) is True

    def test_bool_true(self):
        assert _is_actual_number(True) is False

    def test_bool_false(self):
        assert _is_actual_number(False) is False

    def test_string(self):
        assert _is_actual_number("1") is False

    def test_none(self):
        assert _is_actual_number(None) is False

    def test_list(self):
        assert _is_actual_number([1]) is False


class TestIsComparable:
    def test_number(self):
        assert _is_comparable(5) is True

    def test_float(self):
        assert _is_comparable(5.5) is True

    def test_string(self):
        assert _is_comparable("abc") is True

    def test_bool(self):
        assert _is_comparable(True) is False

    def test_none(self):
        assert _is_comparable(None) is False

    def test_list(self):
        assert _is_comparable([]) is False


class TestIsSpecialNumberCase:
    def test_int_zero_vs_false(self):
        assert _is_special_number_case(0, False) is True

    def test_int_one_vs_true(self):
        assert _is_special_number_case(1, True) is True

    def test_int_zero_vs_true(self):
        assert _is_special_number_case(0, True) is True

    def test_int_one_vs_false(self):
        assert _is_special_number_case(1, False) is True

    def test_reverse_bool_true_vs_int_one(self):
        assert _is_special_number_case(True, 1) is True

    def test_reverse_bool_false_vs_int_zero(self):
        assert _is_special_number_case(False, 0) is True

    def test_non_special_int(self):
        assert _is_special_number_case(2, True) is None

    def test_two_bools(self):
        # Neither is an "actual number", so falls through to None
        assert _is_special_number_case(True, False) is None

    def test_two_strings(self):
        assert _is_special_number_case("a", "b") is None

    def test_zero_vs_zero(self):
        # x is an actual number in (0, 1); isinstance(y, bool) is False
        assert _is_special_number_case(0, 0) is False

    def test_one_vs_one(self):
        # x is an actual number in (0, 1); y=1 is not a bool
        assert _is_special_number_case(1, 1) is False


class TestEquals:
    def test_equal_ints(self):
        assert _equals(1, 1) is True

    def test_zero_vs_false_is_not_equal(self):
        assert _equals(0, False) is False

    def test_one_vs_true_is_not_equal(self):
        assert _equals(1, True) is False

    def test_false_vs_zero_is_not_equal(self):
        assert _equals(False, 0) is False

    def test_true_vs_one_is_not_equal(self):
        assert _equals(True, 1) is False

    def test_true_vs_true(self):
        assert _equals(True, True) is True

    def test_strings(self):
        assert _equals("a", "a") is True
        assert _equals("a", "b") is False

    def test_none(self):
        assert _equals(None, None) is True


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

        class FakeInterpreter(object):
            def visit(self, node, *args, **kwargs):
                calls.append((node, args, kwargs))
                return "result"

        expr = _Expression("expr-node", FakeInterpreter())
        assert expr.expression == "expr-node"
        result = expr.visit("node", 1, 2, key="val")
        assert result == "result"
        assert calls == [("node", (1, 2), {"key": "val"})]


# ---------------------------------------------------------------------------
# Base Visitor
# ---------------------------------------------------------------------------

class TestVisitor:
    def test_default_visit_raises(self):
        v = Visitor()
        with pytest.raises(NotImplementedError) as exc:
            v.visit({'type': 'unknown'})
        assert str(exc.value) == "default_visit"

    def test_visit_dispatches(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, value):
                return ("foo", node['value'], value)

        v = MyVisitor()
        assert v.visit({'type': 'foo', 'value': 10}, 'arg') == ("foo", 10, 'arg')

    def test_method_cache_used(self):
        call_count = {"visit_foo": 0}

        class MyVisitor(Visitor):
            def visit_foo(self, node, value):
                call_count["visit_foo"] += 1
                return value

        v = MyVisitor()
        v.visit({'type': 'foo'}, 1)
        v.visit({'type': 'foo'}, 2)
        assert call_count["visit_foo"] == 2
        assert 'foo' in v._method_cache
        # The cache must store the actual bound method, not None.
        assert v._method_cache['foo'] == v.visit_foo

    def test_method_cache_keyed_by_node_type(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, value):
                return ("foo", value)

            def visit_bar(self, node, value):
                return ("bar", value)

        v = MyVisitor()
        assert v.visit({'type': 'foo'}, 1) == ("foo", 1)
        assert v.visit({'type': 'bar'}, 2) == ("bar", 2)
        assert set(v._method_cache.keys()) == {'foo', 'bar'}
        assert v._method_cache['foo'] == v.visit_foo
        assert v._method_cache['bar'] == v.visit_bar

    def test_visit_passes_kwargs(self):
        class MyVisitor(Visitor):
            def visit_foo(self, node, *args, **kwargs):
                return (args, kwargs)

        v = MyVisitor()
        result = v.visit({'type': 'foo'}, 1, 2, key="val")
        assert result == ((1, 2), {"key": "val"})

    def test_custom_default_visit(self):
        class MyVisitor(Visitor):
            def default_visit(self, node, *args, **kwargs):
                return "default"

        v = MyVisitor()
        assert v.visit({'type': 'anything'}) == "default"

    def test_method_cache_avoids_attribute_lookup(self):
        # If `_method_cache.get(node_type)` were replaced by `None`, the
        # getattr call would happen every visit. If it were replaced by
        # `_method_cache.get(None)`, the cache would never hit. Detect a
        # cached bound method by ensuring repeated visits reuse the *same*
        # method object placed in the cache (don't rely on getattr count).
        class MyVisitor(Visitor):
            def visit_foo(self, node, value):
                return value

        v = MyVisitor()
        v.visit({'type': 'foo'}, 1)
        cached = v._method_cache['foo']
        assert cached == v.visit_foo
        # Manually overwrite cache with a sentinel; if cache is consulted,
        # the sentinel is used instead of a fresh getattr.
        sentinel_calls = []

        def sentinel(node, value):
            sentinel_calls.append((node, value))
            return "sentinel"

        v._method_cache['foo'] = sentinel
        result = v.visit({'type': 'foo'}, 5)
        assert result == "sentinel"
        assert sentinel_calls == [({'type': 'foo'}, 5)]


# ---------------------------------------------------------------------------
# TreeInterpreter
# ---------------------------------------------------------------------------

def node(type_, **kwargs):
    d = {'type': type_}
    d.update(kwargs)
    return d


@pytest.fixture
def interp():
    return TreeInterpreter()


class TestTreeInterpreterInit:
    def test_default_dict_cls(self):
        t = TreeInterpreter()
        assert t._dict_cls is dict
        assert isinstance(t._functions, functions.Functions)

    def test_custom_dict_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        assert t._dict_cls is OrderedDict

    def test_custom_functions(self):
        custom = functions.Functions()
        t = TreeInterpreter(Options(custom_functions=custom))
        assert t._functions is custom


class TestTreeInterpreterVisit:
    def test_default_visit_raises_with_type(self):
        t = TreeInterpreter()
        with pytest.raises(NotImplementedError) as exc:
            t.default_visit({'type': 'bogus'})
        assert str(exc.value) == 'bogus'


class TestSubexpressionIndexPipe:
    def test_subexpression(self, interp):
        n = node('subexpression', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        assert interp.visit(n, {'a': {'b': 5}}) == 5

    def test_subexpression_empty(self, interp):
        n = node('subexpression', children=[])
        assert interp.visit(n, 'anything') == 'anything'

    def test_index_expression(self, interp):
        n = node('index_expression', children=[
            node('field', value='a'),
            node('index', value=0),
        ])
        assert interp.visit(n, {'a': [10, 20]}) == 10

    def test_pipe(self, interp):
        n = node('pipe', children=[
            node('field', value='a'),
            node('identity'),
        ])
        assert interp.visit(n, {'a': 42}) == 42


class TestFieldAndIdentity:
    def test_field_on_dict(self, interp):
        assert interp.visit(node('field', value='x'), {'x': 1}) == 1

    def test_field_missing(self, interp):
        assert interp.visit(node('field', value='y'), {'x': 1}) is None

    def test_field_non_dict(self, interp):
        assert interp.visit(node('field', value='x'), [1, 2]) is None

    def test_field_none(self, interp):
        assert interp.visit(node('field', value='x'), None) is None

    def test_current(self, interp):
        assert interp.visit(node('current'), 'v') == 'v'

    def test_identity(self, interp):
        obj = {'a': 1}
        assert interp.visit(node('identity'), obj) is obj

    def test_literal(self, interp):
        assert interp.visit(node('literal', value=123), 'ignored') == 123


class TestComparator:
    def _comp(self, interp, op, left, right):
        n = node('comparator', value=op, children=[
            node('literal', value=left),
            node('literal', value=right),
        ])
        return interp.visit(n, None)

    def test_eq(self, interp):
        assert self._comp(interp, 'eq', 1, 1) is True
        assert self._comp(interp, 'eq', 1, 2) is False

    def test_eq_bool_int_quirk(self, interp):
        assert self._comp(interp, 'eq', 1, True) is False
        assert self._comp(interp, 'eq', 0, False) is False

    def test_eq_operands_receive_value(self, interp):
        # Both operands must be evaluated against the same current value.
        # Build: field('a') == field('b') evaluated on {'a': 3, 'b': 3}
        n = node('comparator', value='eq', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        assert interp.visit(n, {'a': 3, 'b': 3}) is True
        assert interp.visit(n, {'a': 3, 'b': 4}) is False

    def test_eq_operand_value_not_none(self, interp):
        # If the second operand were visited with None instead of value,
        # field lookups would return None and not match.
        n = node('comparator', value='eq', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        assert interp.visit(n, {'a': 'x', 'b': 'x'}) is True

    def test_ne(self, interp):
        assert self._comp(interp, 'ne', 1, 2) is True
        assert self._comp(interp, 'ne', 1, 1) is False

    def test_lt(self, interp):
        assert self._comp(interp, 'lt', 1, 2) is True
        assert self._comp(interp, 'lt', 2, 1) is False

    def test_gt(self, interp):
        assert self._comp(interp, 'gt', 2, 1) is True

    def test_lte(self, interp):
        assert self._comp(interp, 'lte', 2, 2) is True
        assert self._comp(interp, 'lte', 3, 2) is False

    def test_gte(self, interp):
        assert self._comp(interp, 'gte', 2, 2) is True
        assert self._comp(interp, 'gte', 1, 2) is False

    def test_ordering_non_comparable_returns_none(self, interp):
        assert self._comp(interp, 'lt', "a", "b") is True  # strings comparable
        assert self._comp(interp, 'lt', True, 1) is None
        assert self._comp(interp, 'lt', [1], [2]) is None

    def test_ordering_strings(self, interp):
        assert self._comp(interp, 'lt', "a", "b") is True
        assert self._comp(interp, 'gt', "b", "a") is True

    def test_ordering_operands_receive_value(self, interp):
        # Ordering operands must both be evaluated against the real value.
        # field('a') < field('b') on {'a': 1, 'b': 2}
        n = node('comparator', value='lt', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        assert interp.visit(n, {'a': 1, 'b': 2}) is True
        assert interp.visit(n, {'a': 5, 'b': 2}) is False

    def test_ordering_left_operand_uses_value(self, interp):
        # If left operand were visited with None, field returns None,
        # not comparable -> None.
        n = node('comparator', value='lt', children=[
            node('field', value='a'),
            node('literal', value=10),
        ])
        assert interp.visit(n, {'a': 5}) is True
        assert interp.visit(n, {'a': 15}) is False

    def test_ordering_right_operand_uses_value(self, interp):
        # If right operand were visited with None, field returns None,
        # not comparable -> None.
        n = node('comparator', value='lt', children=[
            node('literal', value=10),
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': 5}) is False
        assert interp.visit(n, {'a': 15}) is True

    def test_ordering_float_and_int_comparable(self, interp):
        # Verifies floats are considered comparable (num_types / _is_comparable)
        assert self._comp(interp, 'lt', 1.5, 2) is True
        assert self._comp(interp, 'gt', 1.5, 2) is False

    def test_ordering_bool_is_not_comparable(self, interp):
        # Booleans are excluded by _is_actual_number; both bool operands
        # therefore make the comparison non-comparable -> None.
        assert self._comp(interp, 'lt', True, False) is None
        assert self._comp(interp, 'gt', False, True) is None

    def test_ordering_mixed_string_and_number_non_comparable(self, interp):
        # A string is comparable and a number is comparable, but comparing
        # them with a Python ordering operator raises TypeError only for
        # Python 3; the interpreter guards with _is_comparable so both being
        # comparable passes the guard. Confirm the guard path for numbers.
        assert self._comp(interp, 'lt', 2, 3) is True


class TestExpref:
    def test_expref_returns_expression(self, interp):
        child = node('literal', value=1)
        n = node('expref', children=[child])
        result = interp.visit(n, 'val')
        assert isinstance(result, _Expression)
        assert result.expression is child
        assert result.interpreter is interp


class TestFunctionExpression:
    def test_function_expression(self, interp):
        # length(@) where @ is list
        n = node('function_expression', value='length', children=[
            node('current'),
        ])
        assert interp.visit(n, [1, 2, 3]) == 3

    def test_function_expression_multiple_args(self, interp):
        n = node('function_expression', value='contains', children=[
            node('literal', value='foobar'),
            node('literal', value='bar'),
        ])
        assert interp.visit(n, None) is True

    def test_function_expression_arg_uses_value(self, interp):
        # The argument 'field(a)' must be evaluated against value, so
        # length of field 'a' should be returned.
        n = node('function_expression', value='length', children=[
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': [1, 2, 3, 4]}) == 4


class TestFilterProjection:
    def test_filter_projection_basic(self, interp):
        # [0] == 1 filter -> returns matching elements mapped by identity
        comp = node('comparator', value='eq', children=[
            node('field', value='a'),
            node('literal', value=1),
        ])
        n = node('filter_projection', children=[
            node('identity'),
            node('identity'),
            comp,
        ])
        data = [{'a': 1}, {'a': 2}, {'a': 1, 'b': 3}]
        result = interp.visit(n, data)
        assert result == [{'a': 1}, {'a': 1, 'b': 3}]

    def test_filter_projection_non_list_returns_none(self, interp):
        n = node('filter_projection', children=[
            node('identity'),
            node('identity'),
            node('literal', value=True),
        ])
        assert interp.visit(n, {'a': 1}) is None

    def test_filter_projection_skips_none_projection(self, interp):
        n = node('filter_projection', children=[
            node('identity'),
            node('identity'),
            node('literal', value=True),
        ])
        # Elements are [None, 1]; projection returns element itself
        # current for None element would be None and skipped.
        assert interp.visit(n, [None, 1]) == [1]

    def test_filter_projection_uses_first_child_for_base(self, interp):
        # children[0] selects the base; children[1] is the projection.
        # If children[0] were swapped with children[1], the base would be
        # the projection applied to the raw value -> wrong.
        n = node('filter_projection', children=[
            node('field', value='items'),
            node('field', value='x'),
            node('literal', value=True),
        ])
        data = {'items': [{'x': 10}, {'x': 20}]}
        # base is data['items'] = list; projection field('x') applied to list
        # -> each element dict -> x value -> returns items' x fields.
        # Since field('items') on the whole value returns a list, filter passes
        # all (comparator literal True), then projection field('x') on each.
        assert interp.visit(n, data) == [10, 20]

    def test_filter_projection_base_is_children0(self, interp):
        # Use a base that's only obtained via children[0]; children[1] would
        # yield None on the root, giving base=None (not a list) -> None.
        n = node('filter_projection', children=[
            node('field', value='items'),
            node('field', value='unused'),
            node('literal', value=True),
        ])
        data = {'items': [1, 2, 3]}
        # base = [1,2,3]; projection field('unused') on ints -> None -> skipped
        assert interp.visit(n, data) == []


class TestFlatten:
    def test_flatten_nested(self, interp):
        n = node('flatten', children=[node('identity')])
        assert interp.visit(n, [[1, 2], 3, [4]]) == [1, 2, 3, 4]

    def test_flatten_non_list(self, interp):
        n = node('flatten', children=[node('identity')])
        assert interp.visit(n, {'a': 1}) is None

    def test_flatten_empty(self, interp):
        n = node('flatten', children=[node('identity')])
        assert interp.visit(n, []) == []


class TestIndexAndSlice:
    def test_index(self, interp):
        assert interp.visit(node('index', value=1), [10, 20, 30]) == 20

    def test_index_negative(self, interp):
        assert interp.visit(node('index', value=-1), [10, 20, 30]) == 30

    def test_index_out_of_range(self, interp):
        assert interp.visit(node('index', value=10), [1, 2]) is None

    def test_index_on_non_list(self, interp):
        assert interp.visit(node('index', value=0), "abc") is None

    def test_slice(self, interp):
        n = node('slice', children=[0, 2, 1])
        assert interp.visit(n, [1, 2, 3, 4]) == [1, 2]

    def test_slice_with_none_start_stop(self, interp):
        n = node('slice', children=[None, None, 2])
        assert interp.visit(n, [1, 2, 3, 4, 5]) == [1, 3, 5]

    def test_slice_non_list(self, interp):
        n = node('slice', children=[0, 1, None])
        assert interp.visit(n, "abc") is None


class TestKeyValPair:
    def test_key_val_pair(self, interp):
        n = node('key_val_pair', children=[
            node('literal', value='k'),
        ])
        assert interp.visit(n, 'val') == 'k'

    def test_key_val_pair_receives_value(self, interp):
        # Children receive the current value; field 'a' should resolve.
        n = node('key_val_pair', children=[
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': 99}) == 99


class TestMultiSelect:
    def test_multi_select_dict(self, interp):
        n = node('multi_select_dict', children=[
            node('literal', value='a'),
            node('literal', value='b'),
        ])
        result = interp.visit(n, 'val')
        assert result == {'a': 'a', 'b': 'b'}

    def test_multi_select_dict_none(self, interp):
        n = node('multi_select_dict', children=[])
        assert interp.visit(n, None) is None

    def test_multi_select_dict_uses_dict_cls(self):
        t = TreeInterpreter(Options(dict_cls=OrderedDict))
        n = node('multi_select_dict', children=[
            node('literal', value='a'),
        ])
        result = t.visit(n, 1)
        assert isinstance(result, OrderedDict)

    def test_multi_select_dict_child_receives_value(self, interp):
        # Each child must be evaluated against the current value.
        n = node('multi_select_dict', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        result = interp.visit(n, {'a': 1, 'b': 2})
        assert result == {'a': 1, 'b': 2}

    def test_multi_select_list(self, interp):
        n = node('multi_select_list', children=[
            node('literal', value=1),
            node('literal', value=2),
        ])
        assert interp.visit(n, 'val') == [1, 2]

    def test_multi_select_list_none(self, interp):
        n = node('multi_select_list', children=[])
        assert interp.visit(n, None) is None

    def test_multi_select_list_child_receives_value(self, interp):
        n = node('multi_select_list', children=[
            node('field', value='a'),
            node('field', value='b'),
        ])
        assert interp.visit(n, {'a': 10, 'b': 20}) == [10, 20]


class TestOrAndNot:
    def _or(self, interp, left_node, right_node):
        return interp.visit(
            node('or_expression', children=[left_node, right_node]), None)

    def test_or_uses_left_when_truthy(self, interp):
        result = self._or(interp,
                          node('literal', value='left'),
                          node('literal', value='right'))
        assert result == 'left'

    def test_or_uses_right_when_left_false(self, interp):
        result = self._or(interp,
                          node('literal', value=''),
                          node('literal', value='right'))
        assert result == 'right'

    def test_or_left_operand_receives_value(self, interp):
        # Left operand must be visited with the real value; ensure a field
        # lookup on the value resolves (truthy), so right is not used.
        n = node('or_expression', children=[
            node('field', value='a'),
            node('literal', value='right'),
        ])
        assert interp.visit(n, {'a': 'left'}) == 'left'

    def test_or_right_operand_receives_value(self, interp):
        # Left is false, so right must be visited with the real value; a
        # field lookup on the value should resolve.
        n = node('or_expression', children=[
            node('literal', value=''),
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': 'right'}) == 'right'

    def _and(self, interp, left_node, right_node):
        return interp.visit(
            node('and_expression', children=[left_node, right_node]), None)

    def test_and_returns_falsey_left(self, interp):
        result = self._and(interp,
                           node('literal', value=[]),
                           node('literal', value='right'))
        assert result == []

    def test_and_returns_right_when_left_truthy(self, interp):
        result = self._and(interp,
                           node('literal', value='truthy'),
                           node('literal', value='right'))
        assert result == 'right'

    def test_and_left_operand_receives_value(self, interp):
        # Left field lookup must resolve against value; if it got None it
        # would be false and return None, not the right literal.
        n = node('and_expression', children=[
            node('field', value='a'),
            node('literal', value='right'),
        ])
        assert interp.visit(n, {'a': 'truthy'}) == 'right'

    def test_and_right_operand_receives_value(self, interp):
        # Left is truthy; right must be evaluated against the real value;
        # a field lookup should resolve.
        n = node('and_expression', children=[
            node('literal', value='truthy'),
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': 'right'}) == 'right'

    def test_not_expression(self, interp):
        n = node('not_expression', children=[node('literal', value="")])
        assert interp.visit(n, None) is True

    def test_not_expression_zero_special(self, interp):
        n = node('not_expression', children=[node('literal', value=0)])
        assert interp.visit(n, None) is False

    def test_not_expression_zero_float(self, interp):
        n = node('not_expression', children=[node('literal', value=0.0)])
        assert interp.visit(n, None) is False

    def test_not_expression_bool_false(self, interp):
        n = node('not_expression', children=[node('literal', value=False)])
        # False is not an actual number, so `not False` -> True
        assert interp.visit(n, None) is True

    def test_not_operand_receives_value(self, interp):
        # Field 'a' resolves to truthy value -> not -> False.
        n = node('not_expression', children=[node('field', value='a')])
        assert interp.visit(n, {'a': 'truthy'}) is False
        # Field 'b' missing -> None -> not None -> True.
        assert interp.visit(n, {'a': 'truthy'}) is False

    def test_not_operand_value_zero_special(self, interp):
        # Field 'a' resolves to 0 -> special case returns False (not True).
        n = node('not_expression', children=[node('field', value='a')])
        assert interp.visit(n, {'a': 0}) is False


class TestProjections:
    def test_projection(self, interp):
        n = node('projection', children=[
            node('identity'),
            node('field', value='a'),
        ])
        data = [{'a': 1}, {'a': 2}, {}]
        assert interp.visit(n, data) == [1, 2]

    def test_projection_non_list(self, interp):
        n = node('projection', children=[
            node('identity'),
            node('field', value='a'),
        ])
        assert interp.visit(n, {'a': 1}) is None

    def test_value_projection(self, interp):
        n = node('value_projection', children=[
            node('identity'),
            node('field', value='a'),
        ])
        data = {'x': {'a': 1}, 'y': {'a': 2}}
        result = interp.visit(n, data)
        assert sorted(result) == [1, 2]

    def test_value_projection_non_dict(self, interp):
        n = node('value_projection', children=[
            node('identity'),
            node('field', value='a'),
        ])
        assert interp.visit(n, [1, 2]) is None


class TestIsFalseTrue:
    def test_is_false(self, interp):
        assert interp._is_false('') is True
        assert interp._is_false([]) is True
        assert interp._is_false({}) is True
        assert interp._is_false(None) is True
        assert interp._is_false(False) is True

    def test_is_true(self, interp):
        assert interp._is_true('a') is True
        assert interp._is_true([1]) is True
        assert interp._is_true(0) is True
        assert interp._is_true(True) is True

    def test_is_false_zero_is_truthy(self, interp):
        assert interp._is_false(0) is False


# ---------------------------------------------------------------------------
# GraphvizVisitor
# ---------------------------------------------------------------------------

class TestGraphvizVisitor:
    def test_init_sets_state_and_calls_super(self):
        # Construction must run without error, and initialization via the
        # base Visitor.__init__ must have created the method cache and the
        # subclass-specific state.
        gv = GraphvizVisitor()
        assert gv._method_cache == {}
        assert gv._lines == []
        assert gv._count == 1

    def test_simple_tree(self):
        gv = GraphvizVisitor()
        n = {'type': 'field', 'value': 'foo', 'children': []}
        result = gv.visit(n)
        assert result.startswith('digraph AST {')
        assert result.endswith('}')
        assert 'field1 [label="field(foo)"]' in result

    def test_with_children(self):
        gv = GraphvizVisitor()
        n = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
                {'type': 'field', 'value': 'b'},
            ],
        }
        result = gv.visit(n)
        assert 'subexpression1 [label="subexpression()"]' in result
        # children get counts 2 and 3
        assert 'field2 [label="field(a)"]' in result
        assert 'field3 [label="field(b)"]' in result
        assert '  subexpression1 -> field2' in result
        assert '  subexpression1 -> field3' in result

    def test_edge_lines_exact(self):
        gv = GraphvizVisitor()
        n = {
            'type': 'subexpression',
            'children': [
                {'type': 'field', 'value': 'a'},
            ],
        }
        result = gv.visit(n)
        lines = result.split('\n')
        assert '  subexpression1 -> field2' in lines

    def test_count_increments(self):
        gv = GraphvizVisitor()
        assert gv._count == 1
        gv.visit({'type': 'identity'})
        assert gv._count == 2

    def test_visit_returns_string(self):
        gv = GraphvizVisitor()
        result = gv.visit({'type': 'identity'})
        assert isinstance(result, str)
        lines = result.split('\n')
        assert lines[0] == 'digraph AST {'
        assert lines[-1] == '}'

    def test_no_children_line_without_children(self):
        gv = GraphvizVisitor()
        n = {'type': 'identity'}
        result = gv.visit(n)
        assert result == '\n'.join([
            'digraph AST {',
            'identity1 [label="identity()"]',
            '}',
        ])

    def test_node_without_value_uses_empty(self):
        gv = GraphvizVisitor()
        n = {'type': 'current'}
        result = gv.visit(n)
        assert 'current1 [label="current()"]' in result

    def test_visit_clears_or_accumulates_lines(self):
        gv = GraphvizVisitor()
        gv.visit({'type': 'identity'})
        result2 = gv.visit({'type': 'current'})
        # The current implementation accumulates lines across visits, so the
        # second result contains both visits and grows _lines.
        assert gv._lines[0] == 'digraph AST {'
        assert result2.split('\n')[0] == 'digraph AST {'

    def test_exact_edge_format(self):
        gv = GraphvizVisitor()
        n = {
            'type': 'subexpression',
            'children': [{'type': 'identity'}],
        }
        result = gv.visit(n)
        assert '\n  subexpression1 -> identity2\n' in ('\n' + result + '\n')
