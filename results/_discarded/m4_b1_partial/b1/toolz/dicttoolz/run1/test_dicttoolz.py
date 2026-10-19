import pytest
from collections import OrderedDict, defaultdict
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


# ---------- merge ----------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_sequence_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_factory_with_sequence():
    result = merge([{1: 2}], factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad=3)


# ---------- merge_with ----------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_list():
    assert merge_with(lambda vals: vals, {1: 1}, {1: 2}, {1: 3}) == {1: [1, 2, 3]}


def test_merge_with_missing_keys():
    result = merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 22, 3: 30}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_sequence_input():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad=3)


# ---------- valmap ----------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 2, 3: 4}) == {1: 4, 3: 8}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


# ---------- keymap ----------

def test_keymap():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(lambda k: k + 1, {1: 'a'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 'a'}


# ---------- itemmap ----------

def test_itemmap():
    assert itemmap(lambda kv: (kv[0] + 1, kv[1] * 2), {1: 2, 3: 4}) == {
        2: 4, 4: 8}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_reverse():
    assert itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2}) == {
        1: 'a', 2: 'b'}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0], kv[1] + 1), {1: 2},
                     factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


# ---------- valfilter ----------

def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_none_match():
    assert valfilter(lambda x: False, {1: 2}) == {}


def test_valfilter_factory():
    result = valfilter(lambda x: x > 0, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


# ---------- keyfilter ----------

def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_none_match():
    assert keyfilter(lambda x: False, {1: 2}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda x: x > 0, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


# ---------- itemfilter ----------

def test_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda x: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda x: x[0] > 0, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


# ---------- assoc ----------

def test_assoc_update_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_no_mutation():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------- dissoc ----------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_mutation():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}
    assert result == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_many_keys_branch():
    # When len(keys) >= len(d) * 0.6, uses the "remaining set" branch
    d = {i: i for i in range(5)}
    result = dissoc(d, 1, 2, 3)
    assert result == {0: 0, 4: 4}


def test_dissoc_few_keys_branch():
    # When len(keys) < len(d) * 0.6, uses the "update then delete" branch
    d = {i: i for i in range(10)}
    result = dissoc(d, 1)
    expected = {i: i for i in range(10) if i != 1}
    assert result == expected


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_factory_many_keys():
    d = {i: i for i in range(5)}
    result = dissoc(d, 1, 2, 3, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {0: 0, 4: 4}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=2)


# ---------- assoc_in ----------

def test_assoc_in_existing_nested():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_assoc_in_new_nested():
    result = assoc_in({}, ['a', 'b', 'c'], 5)
    assert result == {'a': {'b': {'c': 5}}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_new_top_level():
    assert assoc_in({'a': 1}, ['b'], 2) == {'a': 1, 'b': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------- update_in ----------

def test_update_in_existing():
    assert update_in({'a': 0}, ['a'], lambda x: x + 1) == {'a': 1}


def test_update_in_nested_existing():
    d = {'a': {'b': 2}}
    result = update_in(d, ['a', 'b'], lambda x: x * 10)
    assert result == {'a': {'b': 20}}
    assert d == {'a': {'b': 2}}


def test_update_in_missing_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default():
    assert update_in({1: 'foo'}, [2, 3, 4], lambda x: x + 1, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_single_key_missing():
    assert update_in({}, ['a'], lambda x: x + 1, 0) == {'a': 1}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, 5, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 5}}


def test_update_in_does_not_mutate_nested():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


# ---------- get_in ----------

def test_get_in_transaction():
    transaction = {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    transaction = {
        'purchase': {'items': ['Apple', 'Orange']},
    }
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default():
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
        get_in(['a'], [1, 2], no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_wrapped_in_try():
    # non-missing key with no_default True should return the value
    assert get_in(['a'], {'a': 1}, no_default=True) == 1
