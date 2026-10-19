import collections
import pytest

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
)
from toolz.dicttoolz import _get_factory


# -------------------------------------------------------------------------
# merge
# -------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 1}) == {1: 1}


def test_merge_single_iterable_of_dicts():
    # single non-Mapping argument gets unpacked as iterable of dicts
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}
    assert merge([{1: 1}, {1: 2}]) == {1: 2}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 1}, {2: 2})) == {1: 1, 2: 2}


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}


def test_merge_factory():
    result = merge({1: 1}, {2: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1, 2: 2}


def test_merge_bad_keyword():
    with pytest.raises(TypeError):
        merge({1: 1}, badarg=1)


# -------------------------------------------------------------------------
# merge_with
# -------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_values():
    result = merge_with(list, {1: 'a', 2: 'b'}, {1: 'c'}, {1: 'd'})
    assert result == {1: ['a', 'c', 'd'], 2: ['b']}


def test_merge_with_single_value():
    # even keys that appear only once get passed as a list
    assert merge_with(tuple, {1: 1}) == {1: (1,)}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_iterable():
    result = merge_with(sum, [{1: 1}, {1: 2}])
    assert result == {1: 3}


def test_merge_with_preserves_order_of_appearance():
    result = merge_with(list, {1: 'a'}, {2: 'b'}, {1: 'c'}, {2: 'd'})
    assert result == {1: ['a', 'c'], 2: ['b', 'd']}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_keyword():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, badarg=1)


# -------------------------------------------------------------------------
# valmap
# -------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 10, 2: 20}) == {1: 20, 2: 40}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 1}
    valmap(lambda x: x + 1, d)
    assert d == {1: 1}


def test_valmap_keys_preserved():
    d = {'a': 1, 'b': 2}
    result = valmap(lambda v: v * 100, d)
    assert set(result.keys()) == {'a', 'b'}
    assert result == {'a': 100, 'b': 200}


# -------------------------------------------------------------------------
# keymap
# -------------------------------------------------------------------------

def test_keymap():
    assert keymap(lambda x: x + 1, {1: 'a', 2: 'b'}) == {2: 'a', 3: 'b'}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 'a'}


def test_keymap_value_preserved():
    d = {1: object()}
    result = keymap(lambda k: k * 2, d)
    assert result[2] is d[1]


def test_keymap_does_not_mutate():
    d = {1: 'a'}
    keymap(str, d)
    assert d == {1: 'a'}


# -------------------------------------------------------------------------
# itemmap
# -------------------------------------------------------------------------

def test_itemmap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {1: 'a', 2: 'b'}) == {'a': 1, 'b': 2}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 'a'},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 'a'}


def test_itemmap_single_arg_function():
    def swap(item):
        k, v = item
        return v, k
    assert itemmap(swap, {1: 'a'}) == {'a': 1}


def test_itemmap_does_not_mutate():
    d = {1: 'a'}
    itemmap(lambda kv: kv, d)
    assert d == {1: 'a'}


# -------------------------------------------------------------------------
# valfilter
# -------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_none_match():
    assert valfilter(lambda v: False, {1: 1, 2: 2}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


def test_valfilter_no_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    assert d == {1: 2, 2: 3}


# -------------------------------------------------------------------------
# keyfilter
# -------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: False, {1: 1}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


def test_keyfilter_no_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k == 1, d)
    assert d == {1: 2, 2: 3}


# -------------------------------------------------------------------------
# itemfilter
# -------------------------------------------------------------------------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_none_match():
    assert itemfilter(lambda item: False, {1: 1}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 1},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


def test_itemfilter_no_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda item: item[0] == 1, d)
    assert d == {1: 2, 2: 3}


# -------------------------------------------------------------------------
# assoc
# -------------------------------------------------------------------------

def test_assoc_update_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 5)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# -------------------------------------------------------------------------
# dissoc
# -------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_none():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_empty():
    assert dissoc({}, 'a') == {}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_large_number_of_keys_branch():
    # Exercise the branch where len(keys) >= len(d) * 0.6
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(7))
    assert result == {7: 7, 8: 8, 9: 9}


def test_dissoc_small_number_of_keys_branch():
    d = {i: i for i in range(10)}
    result = dissoc(d, 1)
    assert result == {i: i for i in range(10) if i != 1}


def test_dissoc_threshold_boundary():
    # len(d) == 10 -> threshold len(d) * .6 == 6.0
    # with exactly 6 keys, len(keys) < threshold is False (6 < 6.0 is False),
    # so the "else" branch (set difference) is used.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(6))
    assert result == {6: 6, 7: 7, 8: 8, 9: 9}


def test_dissoc_threshold_just_below():
    # 5 keys: 5 < 6.0 is True -> first branch.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(5))
    assert result == {i: i for i in range(5, 10)}


def test_dissoc_threshold_five_of_ten():
    # 5/10 = 0.5 < 0.6 -> first branch (true).
    d = {i: i for i in range(10)}
    assert dissoc(d, 0, 1, 2, 3, 4) == {5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


def test_dissoc_threshold_six_of_ten():
    # 6/10 = 0.6, condition len(keys) < len(d)*.6 is False -> else branch.
    d = {i: i for i in range(10)}
    assert dissoc(d, 0, 1, 2, 3, 4, 5) == {6: 6, 7: 7, 8: 8, 9: 9}


def test_dissoc_distinguishes_multiplicand_by_factory_call_count():
    calls = []

    class CountingDict(dict):
        def update(self, *args, **kwargs):
            calls.append(('update', args, kwargs))
            return super().update(*args, **kwargs)

        def __setitem__(self, k, v):
            calls.append(('setitem', k, v))
            return super().__setitem__(k, v)

    d = {i: i for i in range(10)}
    # 5 keys -> first branch: uses update + del
    calls.clear()
    dissoc(d, 0, 1, 2, 3, 4, factory=CountingDict)
    assert any(c[0] == 'update' for c in calls)

    # 6 keys -> else branch: uses setitem
    calls.clear()
    dissoc(d, 0, 1, 2, 3, 4, 5, factory=CountingDict)
    assert any(c[0] == 'setitem' for c in calls)


def test_dissoc_threshold_requires_strict_less_than():
    class BranchSpy(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.used_update = False
            self.used_setitem = False

        def update(self, *args, **kwargs):
            self.used_update = True
            return super().update(*args, **kwargs)

        def __setitem__(self, k, v):
            self.used_setitem = True
            return super().__setitem__(k, v)

    d = {i: i for i in range(10)}

    captured = {}

    def factory():
        obj = BranchSpy()
        captured['obj'] = obj
        return obj

    dissoc(d, 0, 1, 2, 3, 4, factory=factory)
    assert captured['obj'].used_update is True
    assert captured['obj'].used_setitem is False

    captured.clear()
    dissoc(d, 0, 1, 2, 3, 4, 5, factory=factory)
    assert captured['obj'].used_update is False
    assert captured['obj'].used_setitem is True


def test_dissoc_threshold_multiplicand_is_point_six():
    class BranchSpy(dict):
        def __init__(self):
            super().__init__()
            self.used_update = False
            self.used_setitem = False

        def update(self, *args, **kwargs):
            self.used_update = True
            return super().update(*args, **kwargs)

        def __setitem__(self, k, v):
            self.used_setitem = True
            return super().__setitem__(k, v)

    captured = {}

    def factory():
        obj = BranchSpy()
        captured['obj'] = obj
        return obj

    d = {i: i for i in range(10)}

    # 6 keys, first-branch expected False (else branch).
    dissoc(d, 0, 1, 2, 3, 4, 5, factory=factory)
    assert captured['obj'].used_update is False
    assert captured['obj'].used_setitem is True

    # 9 keys -> also first-branch False
    captured.clear()
    dissoc(d, 0, 1, 2, 3, 4, 5, 6, 7, factory=factory)
    assert captured['obj'].used_update is False
    assert captured['obj'].used_setitem is True


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_keyword():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', badarg=1)


# -------------------------------------------------------------------------
# assoc_in
# -------------------------------------------------------------------------

def test_assoc_in_existing():
    d = {'order': {'costs': [0.5, 1.25]}}
    result = assoc_in(d, ['order', 'costs'], [0.25, 1.0])
    assert result == {'order': {'costs': [0.25, 1.0]}}


def test_assoc_in_does_not_mutate():
    d = {'order': {'costs': [0.5]}}
    assoc_in(d, ['order', 'costs'], [1.0])
    assert d == {'order': {'costs': [0.5]}}


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_top_level():
    assert assoc_in({'a': 1}, ['a'], 5) == {'a': 5}


def test_assoc_in_function_uppercases_default():
    # func is lambda x: value; default is now `value`, so when key missing
    # func(default) == func(value) == value.
    assert assoc_in({}, ['a', 'b'], 42) == {'a': {'b': 42}}


def test_assoc_in_missing_key_with_callable_value():
    # The default passed to update_in is `value` itself; func ignores input,
    # so this must produce exactly `value` at the target location.
    sentinel = object()
    result = assoc_in({'x': 1}, ['a', 'b', 'c'], sentinel)
    assert result['a']['b']['c'] is sentinel


def test_assoc_in_replaces_existing_nested_value_with_identity():
    # Verify the value (not a copy) is stored even when overwriting.
    sentinel = object()
    d = {'a': {'b': 'old'}}
    result = assoc_in(d, ['a', 'b'], sentinel)
    assert result['a']['b'] is sentinel
    assert d['a']['b'] == 'old'


def test_assoc_in_overwrites_only_target():
    d = {'a': {'b': 1, 'c': 2}, 'd': 3}
    result = assoc_in(d, ['a', 'b'], 99)
    assert result == {'a': {'b': 99, 'c': 2}, 'd': 3}


def test_assoc_in_partial_existing_path():
    # a exists but b doesn't
    d = {'a': {'x': 1}}
    result = assoc_in(d, ['a', 'b'], 5)
    assert result == {'a': {'x': 1, 'b': 5}}


def test_assoc_in_default_actually_passes_value_to_func():
    assert assoc_in({}, ['k'], 7) == {'k': 7}


def test_assoc_in_single_key_existing():
    assert assoc_in({'a': 1, 'b': 2}, ['a'], 10) == {'a': 10, 'b': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# --- Additional assoc_in tests exercising the `default` argument independence.
# The mutant at line 242 changes the `default` passed to update_in from
# `value` to `None`. Because assoc_in's `func` is `lambda x: value` (ignores
# its argument), this is an equivalent mutant -- the value placed at the
# target slot is `value` regardless of what `default` is. Verified here in
# multiple scenarios: missing top-level, missing nested, and existing leaves.

def test_assoc_in_missing_leaf_value_independent_of_default():
    # Missing leaf: update_in calls func(default). Result is `value` either way.
    for value in (0, None, 'x', object(), [1, 2], {'k': 'v'}):
        result = assoc_in({'a': {}}, ['a', 'b'], value)
        assert result == {'a': {'b': value}}


def test_assoc_in_missing_leaf_none_value():
    # value=None: result must be exactly None at the leaf.
    result = assoc_in({}, ['a', 'b', 'c'], None)
    assert result == {'a': {'b': {'c': None}}}
    assert result['a']['b']['c'] is None


def test_assoc_in_existing_leaf_value_kept():
    # Existing leaf: update_in calls func(d[k]); result is `value`.
    for value in (0, None, 'x', object()):
        result = assoc_in({'a': {'b': 'old'}}, ['a', 'b'], value)
        assert result['a']['b'] is value


def test_assoc_in_new_deep_path_none_default_value():
    # New deep path: creates nested dicts then func(default) at bottom.
    result = assoc_in({}, ['a', 'b', 'c', 'd'], 'leaf')
    assert result == {'a': {'b': {'c': {'d': 'leaf'}}}}


# -------------------------------------------------------------------------
# update_in
# -------------------------------------------------------------------------

def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': 1}}
    assert update_in(d, ['a', 'b'], inc) == {'a': {'b': 2}}


def test_update_in_missing_uses_func_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=5,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 5}}


def test_update_in_top_level_missing():
    result = update_in({}, ['k'], lambda x: x * 2, default=3)
    assert result == {'k': 6}


def test_update_in_single_key_existing():
    assert update_in({'a': 1}, ['a'], lambda x: x * 10) == {'a': 10}


def test_update_in_single_key_missing_default():
    assert update_in({}, ['a'], lambda x: x + 5, default=1) == {'a': 6}


def test_update_in_shallow_copy_only_changed_path():
    # Verify only the direct path is copied; siblings shared.
    inner = {'b': 1, 'c': 2}
    d = {'a': inner}
    result = update_in(d, ['a', 'b'], lambda x: x + 10)
    assert result['a']['c'] == 2
    assert result['a']['b'] == 11
    assert d['a']['b'] == 1


# -------------------------------------------------------------------------
# get_in
# -------------------------------------------------------------------------

def test_get_in_basic():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_index_error_returns_default():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction, 'x') == 'x'


def test_get_in_type_error_returns_default():
    assert get_in(['a', 'b'], {'a': 5}) is None
    assert get_in(['a', 'b'], {'a': 5}, 'def') == 'def'


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 'b'], {'a': 5}, no_default=True)


def test_get_in_empty_keys():
    # reduce over empty keys returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_single_key():
    assert get_in(['a'], {'a': 42}) == 42


def test_get_in_nested_lists():
    coll = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], coll) == 3


def test_get_in_tuple_keys_supported():
    # reduce takes any iterable of keys, including tuples.
    coll = {'a': {'b': 1}}
    assert get_in(('a', 'b'), coll) == 1


# -------------------------------------------------------------------------
# _get_factory helper (public-ish)
# -------------------------------------------------------------------------

def test_get_factory_default():
    kwargs = {}
    assert _get_factory(merge, kwargs) is dict
    assert kwargs == {}


def test_get_factory_custom():
    kwargs = {'factory': collections.OrderedDict}
    assert _get_factory(merge, kwargs) is collections.OrderedDict
    assert kwargs == {}


def test_get_factory_unexpected():
    kwargs = {'badarg': 1}
    with pytest.raises(TypeError):
        _get_factory(merge, kwargs)


def test_get_factory_unexpected_message_content():
    with pytest.raises(TypeError) as excinfo:
        _get_factory(merge, {'badarg': 1})
    msg = str(excinfo.value)
    assert 'merge' in msg
    assert 'badarg' in msg
    assert 'unexpected keyword argument' in msg
    assert msg == "merge() got an unexpected keyword argument 'badarg'"


def test_get_factory_unexpected_message_valmap():
    with pytest.raises(TypeError) as excinfo:
        _get_factory(valmap, {'weird_kw': 1})
    assert str(excinfo.value) == (
        "valmap() got an unexpected keyword argument 'weird_kw'"
    )


def test_get_factory_first_extra_kwarg_used():
    with pytest.raises(TypeError) as excinfo:
        _get_factory(dissoc, {'first': 1})
    assert 'first' in str(excinfo.value)


def test_merge_unexpected_message():
    with pytest.raises(TypeError) as excinfo:
        merge({1: 1}, nope=2)
    assert str(excinfo.value) == (
        "merge() got an unexpected keyword argument 'nope'"
    )


def test_dissoc_unexpected_message():
    with pytest.raises(TypeError) as excinfo:
        dissoc({'x': 1}, 'x', wrong=1)
    assert str(excinfo.value) == (
        "dissoc() got an unexpected keyword argument 'wrong'"
    )


def test_merge_with_unexpected_message():
    with pytest.raises(TypeError) as excinfo:
        merge_with(sum, {1: 1}, extra=9)
    assert str(excinfo.value) == (
        "merge_with() got an unexpected keyword argument 'extra'"
    )


# -------------------------------------------------------------------------
# Cross-API integration tests
# -------------------------------------------------------------------------

def test_merge_then_valmap():
    merged = merge({'a': 1}, {'b': 2})
    assert valmap(lambda v: v * 10, merged) == {'a': 10, 'b': 20}


def test_assoc_in_roundtrip_get_in():
    d = assoc_in({}, ['a', 'b', 'c'], 7)
    assert get_in(['a', 'b', 'c'], d) == 7


def test_update_in_roundtrip_with_get_in():
    d = update_in({'x': {'y': 1}}, ['x', 'y'], lambda v: v + 1)
    assert get_in(['x', 'y'], d) == 2
