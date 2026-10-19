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


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_mapping():
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_iterable_of_dicts():
    assert merge([{1: 'one'}, {2: 'two'}]) == {1: 'one', 2: 'two'}


def test_merge_iterable_of_dicts_precedence():
    assert merge([{1: 1}, {1: 2}]) == {1: 2}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 'one'}, {2: 'two'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'one', 2: 'two'}


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_lists():
    result = merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(str, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': '1'}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_factory():
    result = keymap(str.upper, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'A': 1}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    assert itemmap(reversed, {'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(reversed, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_factory():
    d = {1: 2, 2: 3}
    result = valfilter(lambda v: v == 2, d, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valfilter_none_match():
    assert valfilter(lambda v: False, {1: 2, 2: 3}) == {}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_keyfilter_factory():
    d = {1: 2, 2: 3}
    result = keyfilter(lambda k: k == 1, d, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: False, {1: 2, 2: 3}) == {}


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(bool, {}) == {}


def test_itemfilter_factory():
    d = {1: 2, 2: 3}
    result = itemfilter(lambda kv: kv[0] == 2, d, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result == {'x': 2}
    assert result is not d


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


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


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}
    assert result == {'x': 1}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad_kwarg=True)


def test_dissoc_many_keys_branch():
    # len(keys) >= len(d) * .6 triggers the "remaining" branch
    d = {i: i for i in range(10)}
    keys = list(range(6))
    result = dissoc(d, *keys)
    assert result == {6: 6, 7: 7, 8: 8, 9: 9}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
        'credit card': '5555-1234-1234-1234',
    }
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_nested_path():
    result = assoc_in({}, ['a', 'b', 'c'], 1)
    assert result == {'a': {'b': {'c': 1}}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_update_in_simple():
    assert update_in({'a': 0}, ['a'], lambda x: x + 1) == {'a': 1}


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
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_path_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_path_inc():
    assert update_in({1: 'foo'}, [2, 3, 4], lambda x: x + 1, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}
    assert result is not d


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 2}


def test_assoc_in_factory():
    result = assoc_in({'a': 1}, ['a'], 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 2}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_basic():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_default_none():
    transaction = {
        'purchase': {'items': ['Apple', 'Orange']},
    }
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_missing_custom_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
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


def test_get_in_no_default_ok():
    assert get_in(['a', 'b'], {'a': {'b': 1}}, no_default=True) == 1


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_lists():
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
