import collections

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


def test_merge_single_mapping():
    d = {1: 2}
    assert merge(d) == {1: 2}
    assert merge(d) is not d


def test_merge_iterable_of_dicts():
    dicts = [{1: 2}, {3: 4}]
    assert merge(dicts) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, badkwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    assert merge_with(lambda x: x[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_single_mapping():
    assert merge_with(sum, {1: 5, 2: 6}) == {1: 5, 2: 6}


def test_merge_with_iterable():
    dicts = [{1: 1}, {1: 2, 2: 3}]
    assert merge_with(sum, dicts) == {1: 3, 2: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_values_are_lists():
    captured = {}

    def capture(values):
        captured['values'] = list(values)
        return values[0]

    merge_with(capture, {1: 'a'}, {1: 'b'})
    assert captured['values'] == ['a', 'b']


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, badkwarg=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 10, 2: 20}) == {1: 11, 2: 21}


def test_valmap_does_not_mutate():
    d = {1: 10}
    valmap(lambda x: x + 1, d)
    assert d == {1: 10}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: '2'}


def test_keymap():
    assert keymap(lambda k: k * 2, {1: 'a', 2: 'b'}) == {2: 'a', 4: 'b'}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 2}


def test_itemmap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_itemmap_reversed():
    assert itemmap(reversed, {'a': 1}) == {1: 'a'}


def test_itemmap_factory():
    result = itemmap(reversed, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    even = lambda x: x % 2 == 0
    assert valfilter(even, {1: 2, 2: 3, 3: 4, 4: 5}) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: False, {1: 2}) == {}


def test_valfilter_factory():
    result = valfilter(lambda x: x > 1, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


def test_keyfilter():
    even = lambda x: x % 2 == 0
    assert keyfilter(even, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda x: True, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
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

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    d = {'x': 1, 'y': 2}
    result = dissoc(d)
    assert result == d
    assert result is not d


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'x')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_large_keys_branch():
    # Many keys removed -> goes through the "remaining" set path
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(9))
    assert result == {9: 9}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', badkwarg=1)


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'],
                  'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_nested():
    result = assoc_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result['a']['b'] == 1


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_uses_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default_value():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate_original():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=0,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

@pytest.fixture
def transaction():
    return {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }


def test_get_in_list_index(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_list_index(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([5], [1, 2], no_default=True)


def test_get_in_type_error_returns_default():
    assert get_in(['a'], [1, 2]) is None


def test_get_in_type_error_no_default_raises():
    with pytest.raises(TypeError):
        get_in(['a'], [1, 2], no_default=True)
