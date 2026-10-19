import pytest
from collections import OrderedDict
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
)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_single_mapping():
    # A single mapping is not unpacked further
    assert merge({1: 2, 3: 4}) == {1: 2, 3: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, 2), (3, 4)]


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


def test_merge_does_not_mutate():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_partial_keys():
    assert merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 22, 3: 30}


def test_merge_with_list():
    result = merge_with(list, {1: 1}, {1: 2}, {1: 3})
    assert result == {1: [1, 2, 3]}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=True)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_factory():
    result = keymap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


def test_keymap_does_not_mutate():
    d = {1: 2}
    keymap(lambda x: x + 1, d)
    assert d == {1: 2}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


def test_itemmap_does_not_mutate():
    d = {1: 2}
    itemmap(lambda kv: (kv[0] + 1, kv[1]), d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    result = valfilter(lambda v: v % 2 == 0, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v % 2 == 0, d)
    assert d == {1: 2, 2: 3}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k % 2 == 0, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 3}


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k % 2 == 0, d)
    assert d == {1: 2, 2: 3}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == 1, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda kv: kv[0] == 1, d)
    assert d == {1: 2, 2: 3}


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
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_large_removal_path():
    # Trigger the "else" branch (>= 60% of keys removed)
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    result = dissoc(d, 'a', 'b', 'c', 'd')
    assert result == {'e': 5}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', bad=True)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_nested():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
    }


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': 1.75},
    }


def test_update_in_creates_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_existing():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], str, default="x", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

@pytest.fixture
def transaction():
    return {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }


def test_get_in_works(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_bad_type_returns_default(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_indexerror():
    with pytest.raises(IndexError):
        get_in([10], [1, 2, 3], no_default=True)


def test_get_in_no_default_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 0], {'a': 5}, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) is transaction


def test_get_in_nested_lists():
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
