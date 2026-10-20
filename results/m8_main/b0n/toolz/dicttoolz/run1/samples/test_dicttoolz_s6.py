import collections
import operator

import pytest

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
)
from toolz.dicttoolz import _get_factory


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_sequence():
    assert merge([{1: 'a'}, {2: 'b'}]) == {1: 'a', 2: 'b'}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping_arg():
    # A single Mapping is treated as a single dict, not unpacked
    assert merge({1: 2}) == {1: 2}


def test_merge_does_not_mutate():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, {2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 2: 3}
    assert list(result) == [1, 2]


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_nonoverlapping():
    result = merge_with(sum, {1: 1}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 20, 3: 30}


def test_merge_with_single_dict():
    # Each key maps to its sole value list; func receives the list
    result = merge_with(lambda vals: vals, {1: 1, 2: 2})
    assert result == {1: [1], 2: [2]}


def test_merge_with_sequence_of_dicts():
    result = merge_with(sum, [{1: 1}, {1: 2}, {1: 3}])
    assert result == {1: 6}


def test_merge_with_lists():
    result = merge_with(list, {1: 'a'}, {1: 'b', 2: 'c'})
    assert result == {1: ['a', 'b'], 2: ['c']}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=2)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_factory():
    result = keymap(str.upper, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'A': 1}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(reversed, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_filters_empty():
    assert valfilter(bool, {}) == {}
    assert keyfilter(bool, {}) == {}
    assert itemfilter(bool, {}) == {}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    keyfilter(lambda k: k > 1, d)
    itemfilter(lambda kv: False, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    result = valfilter(lambda v: v > 1, {1: 1, 2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 1, {1: 1, 2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] > 1, {1: 1, 2: 3},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_many():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_large_key_threshold_path():
    # Deleting >= 60% of keys exercises the branch using set difference
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(6))
    assert result == {6: 6, 7: 7, 8: 8, 9: 9}


def test_dissoc_small_key_threshold_path():
    # Deleting < 60% of keys exercises the update-then-delete branch
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1)
    assert result == {i: i for i in range(2, 10)}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_simple():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}


def test_assoc_in_creates_nested():
    result = assoc_in({}, ['a', 'b', 'c'], 1)
    assert result == {'a': {'b': {'c': 1}}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict) if isinstance(result['a'], dict) else True
    assert result == {'a': 1}


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested_existing():
    d = {'a': {'b': 1}}
    assert update_in(d, ['a', 'b'], lambda x: x + 10) == {'a': {'b': 11}}
    # original untouched
    assert d == {'a': {'b': 1}}


def test_update_in_missing_key_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_key_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_single_key():
    assert update_in({}, ['a'], lambda x: x, default=5) == {'a': 5}


def test_update_in_preserves_siblings():
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    result = update_in(d, ['b', 'c'], lambda x: x * 100)
    assert result == {'a': 1, 'b': {'c': 200, 'd': 3}}
    assert d == {'a': 1, 'b': {'c': 2, 'd': 3}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def setup_transaction():
    return {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }


def test_get_in_nested():
    transaction = setup_transaction()
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single():
    transaction = setup_transaction()
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default_none():
    transaction = setup_transaction()
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_wrong_index_type_returns_default():
    transaction = setup_transaction()
    # string index into a list raises TypeError
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_out_of_range_returns_default():
    transaction = setup_transaction()
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_explicit_default():
    transaction = setup_transaction()
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_success():
    assert get_in(['a'], {'a': 1}, no_default=True) == 1


def test_get_in_empty_keys():
    # reduce with empty keys returns the collection itself
    coll = {'a': 1}
    assert get_in([], coll) == coll


# ---------------------------------------------------------------------------
# _get_factory helper (internal but exercised by public API)
# ---------------------------------------------------------------------------

def test_get_factory_default():
    kwargs = {}
    assert _get_factory(merge, kwargs) is dict
    assert kwargs == {}


def test_get_factory_custom():
    kwargs = {'factory': collections.OrderedDict}
    assert _get_factory(merge, kwargs) is collections.OrderedDict


def test_get_factory_unexpected():
    kwargs = {'bad': 1}
    with pytest.raises(TypeError):
        _get_factory(merge, kwargs)
