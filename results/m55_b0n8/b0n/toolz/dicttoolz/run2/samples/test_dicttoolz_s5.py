import pytest
from collections import OrderedDict, defaultdict

from toolz.dicttoolz import (
    merge, merge_with,
    valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
)


# ---------------- merge ----------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_non_mapping_sequence():
    # A single non-Mapping argument is treated as an iterable of mappings.
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_no_args():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError) as exc:
        merge({1: 2}, foo=1)
    assert "unexpected keyword argument" in str(exc.value)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------------- merge_with ----------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_extra_keys():
    assert merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 22, 3: 30}


def test_merge_with_single_sequence():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=2)


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


# ---------------- valmap / keymap / itemmap ----------------

def test_valmap():
    d = {'a': 1, 'b': 2}
    assert valmap(lambda x: x * 10, d) == {'a': 10, 'b': 20}


def test_valmap_factory():
    result = valmap(lambda x: x, {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 1}


def test_keymap():
    d = {'Alice': 1, 'Bob': 2}
    assert keymap(str.lower, d) == {'alice': 1, 'bob': 2}


def test_keymap_factory():
    result = keymap(str.lower, {'A': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemmap():
    d = {'Alice': 10, 'Bob': 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), d) == {10: 'Alice', 20: 'Bob'}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------- filters ----------------

def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    result = valfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------- assoc ----------------

def test_assoc_overwrite():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_no_mutation():
    d = {'x': 1}
    assoc(d, 'y', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------- dissoc ----------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_many():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_mutation():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_no_keys():
    assert dissoc({'x': 1}) == {'x': 1}


def test_dissoc_many_keys_branch():
    # Many keys triggers the alternative (set-based) branch.
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, 5, 6, 7)
    assert result == {8: 8, 9: 9}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=2)


# ---------------- assoc_in / update_in ----------------

def test_assoc_in():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # original not mutated
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    result = assoc_in({}, ['a', 'b', 'c'], 1)
    assert result == {'a': {'b': {'c': 1}}}


def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_new_keys_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_existing():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------- get_in ----------------

def test_get_in_basic():
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


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_missing_index():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_wrong_type_returns_default():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)
