import pytest
from collections import OrderedDict, defaultdict

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
)


# ---------------------------------------------------------------- merge

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_from_list_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_from_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_from_generator():
    assert merge(d for d in [{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_does_not_mutate():
    a = {1: 2}
    merge(a, {3: 4})
    assert a == {1: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ---------------------------------------------------------------- merge_with

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict():
    # values should be passed as a list
    assert merge_with(lambda vs: vs, {1: 1, 2: 2}) == {1: [1], 2: [2]}


def test_merge_with_from_list():
    assert merge_with(sum, [{1: 1}, {1: 10}]) == {1: 11}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, other=1)


# ---------------------------------------------------------------- valmap

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 1, 2: 2}) == {1: 2, 2: 3}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1, 2: 2}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 1}
    valmap(lambda x: x + 1, d)
    assert d == {1: 1}


# ---------------------------------------------------------------- keymap

def test_keymap():
    assert keymap(lambda x: x + 1, {1: 'a', 2: 'b'}) == {2: 'a', 3: 'b'}


def test_keymap_factory():
    result = keymap(lambda x: x, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_does_not_mutate():
    d = {1: 'a'}
    keymap(lambda x: x + 1, d)
    assert d == {1: 'a'}


# ---------------------------------------------------------------- itemmap

def test_itemmap():
    assert itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 'a', 2: 'b'}) == {
        2: 'a', 3: 'b'}


def test_itemmap_swap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2}) == {
        1: 'a', 2: 'b'}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


# ---------------------------------------------------------------- valfilter

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    assert valfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {1: 2, 3: 4}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valfilter_none_match():
    assert valfilter(lambda v: False, {1: 2}) == {}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v == 2, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------- keyfilter

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    assert keyfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {
        2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: False, {1: 2}) == {}


# ---------------------------------------------------------------- itemfilter

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_itemfilter_none_match():
    assert itemfilter(lambda item: False, {1: 2}) == {}


# ---------------------------------------------------------------- assoc

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 3, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 2)
    assert d == {'x': 1}


# ---------------------------------------------------------------- dissoc

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_all():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_factory_path_many_keys():
    # exercise the "many keys" branch (len(keys) >= len(d) * .6)
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    assert dissoc(d, 1, 2, 3) == {4: 'd', 5: 'e'}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', other=1)


# ---------------------------------------------------------------- assoc_in

def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    # original untouched
    assert d == {'a': {'b': 1}}


def test_assoc_in_nested_existing():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'],
                  'costs': [0.50, 1.25]},
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'],
                  'costs': [0.25, 1.00]},
    }
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, [1, 2, 3], 'foo') == {1: {2: {3: 'foo'}}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 1}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


# ---------------------------------------------------------------- update_in

def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': 1.75},
    }


def test_update_in_missing_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}}


def test_update_in_missing_partial():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_factory():
    inc = lambda x: x + 1
    result = update_in({}, ['a'], inc, default=0, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 1}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    inc = lambda x: x + 1
    update_in(d, ['a', 'b'], inc)
    assert d == {'a': {'b': 1}}


def test_update_in_single_key_deep_copy_top():
    d = {'a': {'b': 1}}
    inc = lambda x: x + 1
    result = update_in(d, ['a'], inc)
    assert result['a'] == 1


# ---------------------------------------------------------------- get_in

def test_get_in_dict():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_simple_key():
    assert get_in(['name'], {'name': 'Alice'}) == 'Alice'


def test_get_in_missing_key_returns_default():
    assert get_in(['purchase', 'total'], {'purchase': {}}) is None


def test_get_in_missing_returns_custom_default():
    assert get_in(['purchase', 'total'], {'purchase': {}}, 0) == 0


def test_get_in_index_error_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error_returns_default():
    # attempting to index into an int -> TypeError
    assert get_in(['a', 'b'], {'a': 1}) is None


def test_get_in_string_index_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 'b'], {'a': 1}, no_default=True)


def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    assert get_in([], coll) is coll
