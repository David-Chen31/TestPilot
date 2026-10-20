import operator
from collections import OrderedDict
from collections.abc import Mapping

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


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_no_args():
    assert merge() == {}


def test_merge_single_mapping_argument():
    assert merge({1: 2, 3: 4}) == {1: 2, 3: 4}


def test_merge_single_iterable_of_mappings():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_generator_argument():
    assert merge({i: i} for i in range(3)) == {0: 0, 1: 1, 2: 2}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_values_as_list():
    assert merge_with(list, {1: 1}, {1: 2}, {1: 3}) == {1: [1, 2, 3]}


def test_merge_with_first():
    assert merge_with(lambda x: x[0], {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: 1, 2: 2, 3: 30
    }


def test_merge_with_no_args():
    assert merge_with(sum) == {}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=2)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 10, 2: 20}) == {1: 20, 2: 40}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_keys_preserved():
    assert valmap(str, {'a': 1, 'b': 2}) == {'a': '1', 'b': '2'}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(str.lower, {'Alice': 1, 'Bob': 2}) == {'alice': 1, 'bob': 2}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_keymap_does_not_mutate():
    d = {'A': 1}
    keymap(str.lower, d)
    assert d == {'A': 1}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_reversed():
    assert itemmap(reversed, {'Alice': 10, 'Bob': 20}) == {10: 'Alice', 20: 'Bob'}


def test_itemmap():
    assert itemmap(lambda kv: (kv[0] + 1, kv[1] * 2), {1: 10}) == {2: 20}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_all_filtered():
    assert valfilter(lambda x: False, {1: 2, 3: 4}) == {}


def test_valfilter_factory():
    result = valfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda x: x > 1, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


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
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    d = {'x': 1, 'y': 2}
    assert dissoc(d, 'y') == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_ignores_missing():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    d = {'x': 1, 'y': 2}
    result = dissoc(d)
    assert result == d
    assert result is not d


def test_dissoc_many_keys_uses_set_path():
    # Ensure the set-difference branch is exercised (len(keys) >= len(d) * .6)
    d = {i: i for i in range(10)}
    keys = list(range(6))
    result = dissoc(d, *keys)
    assert result == {i: i for i in range(6, 10)}
    assert d == {i: i for i in range(10)}


def test_dissoc_few_keys_uses_update_path():
    d = {i: i for i in range(10)}
    result = dissoc(d, 0)
    assert result == {i: i for i in range(1, 10)}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'x', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'y': 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_nested_structure():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], int)


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_does_not_mutate():
    d = {'a': 0}
    update_in(d, ['a'], lambda x: x + 1)
    assert d == {'a': 0}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_keys_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default_value():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


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


def test_get_in_list_index(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single_key(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_index_type(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction


def test_get_in_no_default_key_error():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([0], [], no_default=True)


def test_get_in_no_default_type_error():
    with pytest.raises(TypeError):
        get_in(['x'], 5, no_default=True)
