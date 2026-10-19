import pytest
from collections import OrderedDict, defaultdict
from collections.abc import Mapping

from toolz.dicttoolz import (
    merge,
    merge_with,
    valmap,
    keymap,
    itemmap,
    valfilter,
    keyfilter,
    itemfilter,
    assoc,
    dissoc,
    assoc_in,
    update_in,
    get_in,
    _get_factory,
)


# ---------- merge ----------

def test_merge_empty():
    assert merge() == {}


def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_non_mapping_iterable():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping_not_unpacked():
    # A single mapping is not treated as an iterable of dicts.
    assert merge({1: 2}) == {1: 2}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, 2), (3, 4)]


def test_merge_defaultdict_factory():
    import collections
    result = merge({1: 2}, factory=lambda: collections.defaultdict(list))
    assert isinstance(result, collections.defaultdict)


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad=3)


def test_merge_unexpected_kwarg_message():
    with pytest.raises(TypeError) as exc:
        merge({1: 2}, bad=3)
    assert "merge" in str(exc.value)
    assert "bad" in str(exc.value)


def test_merge_unexpected_kwarg_message_exact():
    with pytest.raises(TypeError) as exc:
        merge({1: 2}, bad=3)
    assert str(exc.value) == "merge() got an unexpected keyword argument 'bad'"


def test_merge_factory_and_bad_kwarg_raises_on_bad():
    # factory kwarg is consumed, remaining kwarg reported.
    with pytest.raises(TypeError) as exc:
        merge({1: 2}, factory=OrderedDict, extra=1)
    assert str(exc.value) == "merge() got an unexpected keyword argument 'extra'"


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------- _get_factory ----------

def test_get_factory_default():
    kwargs = {}
    assert _get_factory(merge, kwargs) is dict
    assert kwargs == {}


def test_get_factory_explicit():
    kwargs = {'factory': OrderedDict}
    assert _get_factory(merge, kwargs) is OrderedDict
    assert kwargs == {}


def test_get_factory_unexpected_message():
    kwargs = {'nope': 1}
    with pytest.raises(TypeError) as exc:
        _get_factory(merge, kwargs)
    assert str(exc.value) == "merge() got an unexpected keyword argument 'nope'"


# ---------- merge_with ----------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 2, 3: 4}) == {1: 2, 3: 4}


def test_merge_with_single_non_mapping_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_receives_list():
    seen = {}

    def f(vals):
        seen['vals'] = list(vals)
        return sum(vals)

    assert merge_with(f, {1: 5}, {1: 6}) == {1: 11}
    assert seen['vals'] == [5, 6]


def test_merge_with_preserves_insertion_order():
    # Dicts ordered: key 'a' first seen in first dict
    result = merge_with(sum, {'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    assert list(result.keys()) == ['a', 'b', 'c']


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=3)


def test_merge_with_unexpected_kwarg_message_exact():
    with pytest.raises(TypeError) as exc:
        merge_with(sum, {1: 1}, bad=3)
    assert str(exc.value) == "merge_with() got an unexpected keyword argument 'bad'"


def test_merge_with_empty():
    assert merge_with(sum) == {}


# ---------- valmap ----------

def test_valmap_basic():
    assert valmap(lambda x: x + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}


def test_valmap_preserves_keys():
    d = {'a': 1, 'b': 2}
    assert valmap(str, d) == {'a': '1', 'b': '2'}


def test_valmap_factory():
    result = valmap(lambda x: x, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_does_not_mutate():
    d = {'a': 1}
    valmap(lambda x: x + 1, d)
    assert d == {'a': 1}


# ---------- keymap ----------

def test_keymap_basic():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_factory():
    result = keymap(str.upper, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_collision_later_wins():
    # keys reduced to same value; later one wins
    assert keymap(str.lower, {'A': 1, 'a': 2}) == {'a': 2}


def test_keymap_does_not_mutate():
    d = {'a': 1}
    keymap(str.upper, d)
    assert d == {'a': 1}


# ---------- itemmap ----------

def test_itemmap_basic():
    result = itemmap(lambda kv: (kv[0], kv[1] * 2), {'a': 1, 'b': 2})
    assert result == {'a': 2, 'b': 4}


def test_itemmap_reverse():
    result = itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2})
    assert result == {1: 'a', 2: 'b'}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


# ---------- valfilter ----------

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    assert valfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    result = valfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_valfilter_no_match():
    assert valfilter(lambda x: False, {1: 2}) == {}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda x: x > 2, d)
    assert d == {1: 2, 2: 3}


# ---------- keyfilter ----------

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    assert keyfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda x: x == 2, d)
    assert d == {1: 2, 2: 3}


# ---------- itemfilter ----------

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    assert itemfilter(isvalid, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda item: item[0] == 2, d)
    assert d == {1: 2, 2: 3}


# ---------- assoc ----------

def test_assoc_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


# ---------- dissoc ----------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1}) == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_factory_many_keys():
    # Exercise the "else" branch (many keys relative to dict size)
    d = {'a': 1, 'b': 2}
    result = dissoc(d, 'a', 'b', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_unexpected_kwarg_message_exact():
    with pytest.raises(TypeError) as exc:
        dissoc({'x': 1}, 'x', bad=1)
    assert str(exc.value) == "dissoc() got an unexpected keyword argument 'bad'"


def test_dissoc_empty_dict():
    assert dissoc({}, 'x') == {}


# Boundary tests around the `len(keys) < len(d) * .6` branch selection.
# Both branches produce the same set of remaining keys, BUT they differ in
# key ORDER of the returned dict:
#   - "update" branch (few keys): iterates over d's order, deletes some ->
#     survivors appear in d's insertion order.
#   - "else" branch (many keys): iterates over a `set(d)` and builds fresh ->
#     survivors appear in set-iteration order (ascending for small ints).
# By inserting keys in a non-monotonic order, insertion order and set
# iteration order diverge, so the two branches are distinguishable.

def test_dissoc_boundary_equal_threshold_few_keys_order():
    # len(d) == 5, .6 * 5 == 3.0; len(keys) == 3 -> 3 < 3.0 is False -> else.
    # Survivors {2, 0} appear in set-iteration order [0, 2], NOT insertion
    # order [2, 0]. A `<=` mutant would flip to the update branch, yielding
    # [2, 0].
    d = OrderedDict([(3, 3), (1, 1), (4, 4), (2, 2), (0, 0)])
    result = dissoc(d, 3, 1, 4, factory=OrderedDict)
    assert list(result.keys()) == [0, 2]


def test_dissoc_threshold_midway_many_keys_uses_set_order():
    # len(d) == 10, .6*10 == 6.0; len(keys) == 7 -> 7 < 6.0 is False -> else.
    # The `< len(d)/.6` (~16.7) and `< len(d)*1.6` (==16.0) mutants would both
    # flip this case to the update branch. Survivors {2,0,1} appear in
    # set-iteration order [0,1,2], NOT insertion order [2,0,1].
    d = OrderedDict([
        (2, 2), (9, 9), (0, 0), (8, 8), (1, 1),
        (7, 7), (6, 6), (5, 5), (4, 4), (3, 3),
    ])
    result = dissoc(d, 9, 8, 7, 6, 5, 4, 3, factory=OrderedDict)
    assert list(result.keys()) == [0, 1, 2]


def test_dissoc_branch_selection_observable_via_order():
    # len(d) == 25, .6*25 == 15.0. Use len(keys) == 15 to hit the boundary.
    # With `<` this is False -> else branch (set iteration order not guaranteed
    # insertion order). With `<=` it becomes True -> update branch (preserves
    # insertion order). We assert insertion order of surviving keys.
    n = 25
    k = 15
    d = OrderedDict((i, i) for i in range(n))
    remove = list(range(k))  # remove first 15 keys; survivors 15..24
    result = dissoc(d, *remove, factory=OrderedDict)
    assert list(result.keys()) == list(range(k, n))


def test_dissoc_branch_few_keys_order_preserved():
    # update branch: keys < threshold, must preserve insertion order.
    d = OrderedDict([('z', 1), ('a', 2), ('m', 3), ('b', 4), ('y', 5)])
    # len(d)=5, .6*5=3.0, len(keys)=1 -> update branch
    result = dissoc(d, 'a', factory=OrderedDict)
    assert list(result.keys()) == ['z', 'm', 'b', 'y']


# ---------- assoc_in ----------

def test_assoc_in_nested_existing():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'


def test_assoc_in_does_not_mutate():
    purchase = {'order': {'costs': [1.0]}}
    assoc_in(purchase, ['order', 'costs'], [2.0])
    assert purchase == {'order': {'costs': [1.0]}}


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 5) == {'a': {'b': {'c': 5}}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


def test_assoc_in_replaces_existing_value_with_value_not_none():
    d = {'a': {'x': 1}}
    result = assoc_in(d, ['a', 'b'], 42)
    assert result == {'a': {'x': 1, 'b': 42}}


def test_assoc_in_missing_parent_uses_value_as_default():
    d = {'a': {'x': 1}}
    result = assoc_in(d, ['a', 'y'], 99)
    assert result['a']['y'] == 99


def test_assoc_in_default_distinct_from_none_multi_level():
    d = {'x': {'y': {'z': 'old'}}}
    result = assoc_in(d, ['x', 'y', 'w'], 'new')
    assert result == {'x': {'y': {'z': 'old', 'w': 'new'}}}
    assert d == {'x': {'y': {'z': 'old'}}}


def test_assoc_in_value_replaces_and_default_is_value():
    d = {'a': {}}
    result = assoc_in(d, ['a', 'b'], 'VAL')
    assert result == {'a': {'b': 'VAL'}}
    assert result['a']['b'] == 'VAL'


# ---------- update_in ----------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_default_used():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_default_with_func():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    transaction = {'purchase': {'costs': [0.50, 1.25]}}
    update_in(transaction, ['purchase', 'costs'], sum)
    assert transaction == {'purchase': {'costs': [0.50, 1.25]}}


def test_update_in_nested_partial_exists():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 10)
    assert result == {'a': {'b': 11}}
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------- get_in ----------

def test_get_in_simple():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    d = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], d) is None


def test_get_in_out_of_range():
    d = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 10], d) is None


def test_get_in_wrong_type_index():
    d = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 'apple'], d) is None


def test_get_in_custom_default():
    d = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], d, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in(['items', 5], {'items': [1]}, no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 'b'], {'a': 5}, no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_nested_list_index():
    d = {'a': [{'b': 7}]}
    assert get_in(['a', 0, 'b'], d) == 7
