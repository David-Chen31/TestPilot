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


def test_merge_no_args():
    assert merge() == {}


def test_merge_single_iterable_of_dicts():
    dicts = [{1: 'one'}, {2: 'two'}]
    assert merge(dicts) == {1: 'one', 2: 'two'}


def test_merge_single_mapping_is_mapping():
    # A single mapping is not unpacked as an iterable of items.
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_factory():
    result = merge({1: 'one'}, {2: 'two'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'one', 2: 'two'}
    assert list(result.keys()) == [1, 2]


def test_merge_does_not_mutate_inputs():
    a = {1: 'one'}
    b = {2: 'two'}
    merge(a, b)
    assert a == {1: 'one'}
    assert b == {2: 'two'}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_receives_list():
    seen = {}

    def capture(values):
        seen['values'] = list(values)
        return values[0]

    result = merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert seen['values'] == ['a', 'b', 'c']
    assert result == {1: 'a'}


def test_merge_with_preserves_order():
    result = merge_with(tuple, {1: 'a'}, {1: 'b'})
    assert result == {1: ('a', 'b')}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
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
    result = keymap(str, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 'a'}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_tuple_transform():
    def swap(item):
        k, v = item
        return (v, k)

    assert itemmap(swap, {1: 'a', 2: 'b'}) == {'a': 1, 'b': 2}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    def swap(item):
        k, v = item
        return (v, k)

    result = itemmap(swap, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    iseven = lambda x: x % 2 == 0
    result = valfilter(iseven, {1: 2, 2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


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
    iseven = lambda x: x % 2 == 0
    result = keyfilter(iseven, {1: 2, 2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
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


def test_itemfilter_factory():
    def isvalid(item):
        k, v = item
        return k % 2 == 0

    result = itemfilter(isvalid, {1: 2, 2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
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
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one_key():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_all_keys():
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
    assert result is not d


def test_dissoc_many_keys_branch():
    # Removing more than 60% of keys exercises the "remaining set" branch.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(8))
    assert result == {8: 8, 9: 9}
    assert d == {i: i for i in range(10)}


def test_dissoc_few_keys_branch():
    # Removing fewer than 60% of keys exercises the "update then delete" branch.
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1)
    assert result == {i: i for i in range(2, 10)}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', bad_kwarg=1)


def test_dissoc_factory_branch_with_remaining():
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(8), factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {8: 8, 9: 9}


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
    # Original not mutated
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_single_key():
    assert assoc_in({}, ['a'], 1) == {'a': 1}


def test_assoc_in_partial_existing():
    assert assoc_in({1: 'foo'}, [2, 3, 4], 1) == {1: 'foo', 2: {3: {4: 1}}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

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
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_default_create():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo',
        2: {3: {4: 1}},
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 10)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x + 1, 0,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': {'b': 1}}


def test_update_in_returns_new_dict():
    d = {'a': 1}
    result = update_in(d, ['a'], lambda x: x + 1)
    assert result is not d


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


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_index_type(transaction):
    # 'apple' used as a list index raises TypeError, swallowed.
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], None, no_default=True)
