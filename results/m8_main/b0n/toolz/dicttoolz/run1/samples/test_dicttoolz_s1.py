import collections
from collections import OrderedDict, defaultdict

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


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_dict_as_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_list_of_dicts():
    assert merge([{1: 2}, {2: 3}]) == {1: 2, 2: 3}


def test_merge_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_generator_of_dicts():
    assert merge({i: i} for i in range(3)) == {0: 0, 1: 1, 2: 2}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_factory_list_of_dicts():
    result = merge([{1: 2}], factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


def test_merge_does_not_mutate_inputs():
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


def test_merge_with_first():
    assert merge_with(lambda vals: vals[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_list_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, nope=1)


def test_merge_with_values_are_list():
    seen = {}

    def capture(vals):
        seen['vals'] = vals
        return len(vals)

    merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert seen['vals'] == ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 10, 2: 20}) == {1: 11, 2: 21}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: '2'}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30],
        "bob": [10, 35],
    }


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'1': 2}


def test_keymap_collision_last_wins():
    # Both keys map to same value -> later item wins
    assert keymap(lambda k: 0, {1: 'a', 2: 'b'}) == {0: 'b'}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda item: item, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_itemmap_transform():
    assert itemmap(lambda kv: (kv[0] + 1, kv[1] * 2), {1: 10}) == {2: 20}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_none_match():
    assert valfilter(lambda v: False, {1: 2, 3: 4}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: v > 1, {1: 2, 2: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: False, {1: 2}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 1, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 3}


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


def test_itemfilter_none_match():
    assert itemfilter(lambda item: False, {1: 2}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda it: it[0] == 1, {1: 2, 2: 3},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result == {'x': 2}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_empty_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'x')
    assert d == {'x': 1, 'y': 2}
    assert result == {'y': 2}
    assert result is not d


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'x', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'y': 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_many_keys_vs_small():
    # Exercise both branches (many keys -> set path)
    d = dict.fromkeys(range(10), 0)
    result = dissoc(d, *range(9))
    assert result == {9: 0}
    # few keys -> copy/delete path
    result = dissoc(d, 0)
    assert result == dict.fromkeys(range(1, 10), 0)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    purchase = {
        'name': 'Alice',
        'order': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # Original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_top_level():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    # Original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=5,
                       factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result['a']['b'] == 5


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


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_top_level(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_wrong_type_index(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_success(transaction):
    assert get_in(['name'], transaction, no_default=True) == 'Alice'


def test_get_in_nested_lists():
    data = {'a': [{'b': [10, 20]}]}
    assert get_in(['a', 0, 'b', 1], data) == 20
