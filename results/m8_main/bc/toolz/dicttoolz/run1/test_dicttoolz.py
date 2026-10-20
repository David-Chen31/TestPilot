import pytest
from collections import OrderedDict
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


def test_merge_single_dict():
    assert merge({1: 2}) == {1: 2}


def test_merge_iterable_of_dicts():
    # single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------- merge_with ----------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_missing_keys():
    result = merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 22, 3: 30}


def test_merge_with_collects_all_values():
    captured = {}

    def capture(values):
        captured['values'] = list(values)
        return tuple(values)

    result = merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: ('a', 'b', 'c')}
    assert captured['values'] == ['a', 'b', 'c']


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=1)


# ---------- valmap ----------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------- keymap ----------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(lambda k: k + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ---------- itemmap ----------

def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------- valfilter ----------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty_result():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x > 100, d) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------- keyfilter ----------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty_result():
    d = {1: 2, 2: 3}
    assert keyfilter(lambda x: False, d) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------- itemfilter ----------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty_result():
    d = {1: 2}
    assert itemfilter(lambda it: False, d) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda it: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------- assoc ----------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 3, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 3}


# ---------- dissoc ----------

def test_dissoc():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
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


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad_kwarg=1)


def test_dissoc_missing_key_small_fraction():
    # When len(keys) < len(d) * 0.6, the implementation takes the update path
    # and checks ``if key in d2`` for each key.  With a key that is not in the
    # dict, the branch is skipped (key not present) - assert result unchanged.
    d = {i: i for i in range(10)}
    result = dissoc(d, 'missing')
    assert result == d
    # and with one present key it is removed
    result = dissoc(d, 3)
    expected = dict(d)
    del expected[3]
    assert result == expected


def test_dissoc_present_and_missing_keys_small_fraction():
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    # len(keys)=2 < 5*0.6=3 -> update path; 'b' present, 'z' missing
    result = dissoc(d, 'b', 'z')
    assert result == {'a': 1, 'c': 3, 'd': 4, 'e': 5}


def test_dissoc_all_keys_large_fraction():
    # len(keys) >= len(d)*0.6 -> remaining-set path
    d = {'a': 1, 'b': 2}
    assert dissoc(d, 'a', 'b') == {}
    assert dissoc(d, 'a') == {'b': 2}


# ---------- assoc_in ----------

def test_assoc_in_basic():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {'credit card': '5555-1234-1234-1234',
                      'name': 'Alice',
                      'order': {'costs': [0.25, 1.00],
                                'items': ['Apple', 'Orange']}}


def test_assoc_in_does_not_mutate():
    purchase = {'order': {'costs': [0.50]}}
    assoc_in(purchase, ['order', 'costs'], [0.25])
    assert purchase == {'order': {'costs': [0.50]}}


def test_assoc_in_creates_missing_path():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


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
    assert result['name'] == 'Alice'


def test_update_in_missing_key_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_top_level():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({'a': 0}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 1}


# ---------- get_in ----------

def test_get_in_basic():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_key_in_list():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_out_of_range():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_default():
    transaction = {'purchase': {'items': ['Apple']}}
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
    d = {'a': 1}
    assert get_in([], d) == d
