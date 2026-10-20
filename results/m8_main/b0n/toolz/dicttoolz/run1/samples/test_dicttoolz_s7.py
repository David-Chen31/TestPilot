import pytest
from collections import OrderedDict, defaultdict
from collections.abc import Mapping

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
)


# ---------- merge ----------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_non_mapping_iterable():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping_is_treated_as_one_dict():
    # A Mapping is not unpacked as a collection of dicts
    assert merge({1: 2, 3: 4}) == {1: 2, 3: 4}


def test_merge_factory():
    rv = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert list(rv.items()) == [(1, 2), (3, 4)]


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


def test_merge_does_not_mutate():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------- merge_with ----------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_partial_keys():
    assert merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 22, 3: 30}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_non_mapping_iterable():
    assert merge_with(sum, [{1: 2}, {1: 3}]) == {1: 5}


def test_merge_with_factory():
    rv = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=1)


# ---------- valmap ----------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    rv = valmap(sum, {"a": [1, 2]}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {"a": 3}


# ---------- keymap ----------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_factory():
    rv = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {"A": 1}


# ---------- itemmap ----------

def test_itemmap_basic():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    rv = itemmap(lambda kv: (kv[0].upper(), kv[1]), {"a": 1}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {"A": 1}


# ---------- valfilter ----------

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_factory():
    rv = valfilter(lambda v: v > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {2: 2}


# ---------- keyfilter ----------

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_factory():
    rv = keyfilter(lambda k: k > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {2: 2}


# ---------- itemfilter ----------

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    rv = itemfilter(lambda item: item[1] > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {2: 2}


# ---------- assoc ----------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    rv = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'x': 1, 'y': 2}


# ---------- dissoc ----------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    rv = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_both_branches():
    # Few keys removed -> update path
    assert dissoc({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, 'a') == \
        {'b': 2, 'c': 3, 'd': 4, 'e': 5}
    # Many keys removed -> set-difference path
    assert dissoc({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
                  'a', 'b', 'c', 'd') == {'e': 5}


# ---------- assoc_in ----------

def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # original unchanged
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_missing():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_factory():
    rv = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert isinstance(rv['a'], OrderedDict)
    assert rv == {'a': {'b': 1}}


# ---------- update_in ----------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    # original unchanged
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_creates_nested_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_existing_sibling():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    rv = update_in({}, ['a', 'b'], lambda x: x + 1, 0, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert isinstance(rv['a'], OrderedDict)
    assert rv == {'a': {'b': 1}}


# ---------- get_in ----------

def test_get_in_basic():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['credit card'], transaction) == '5555-1234-1234-1234'


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)


def test_get_in_empty_keys():
    # reduce with empty iterable returns the initial collection
    assert get_in([], {'a': 1}) == {'a': 1}
