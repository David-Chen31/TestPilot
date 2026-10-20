import pytest
from collections import OrderedDict, defaultdict, Counter

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


def test_merge_later_wins():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_dict():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_generator_of_dicts():
    gen = ({i: i} for i in range(3))
    assert merge(gen) == {0: 0, 1: 1, 2: 2}


def test_merge_factory():
    rv = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert list(rv.items()) == [(1, 2), (3, 4)]


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, badarg=1)


def test_merge_does_not_mutate():
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


def test_merge_with_lists():
    assert merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: [1], 2: [2, 20], 3: [30]
    }


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    rv = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, badarg=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


def test_valmap_factory():
    rv = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 3}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30], "bob": [10, 35]
    }


def test_keymap_factory():
    rv = keymap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {2: 2}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {
        10: "Alice", 20: "Bob"
    }


def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    rv = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {2: 2}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda x: x == 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    rv = valfilter(lambda x: x == 2, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 2}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    rv = keyfilter(lambda x: x == 1, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    rv = itemfilter(lambda item: item[0] == 1, {1: 2}, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {1: 2}


# ---------------------------------------------------------------------------
# assoc / dissoc
# ---------------------------------------------------------------------------

def test_assoc_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 3)
    assert d == {'x': 1}


def test_assoc_factory():
    rv = assoc({'x': 1}, 'y', 3, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'x': 1, 'y': 3}


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


def test_dissoc_many_path():
    # Trigger the branch where len(keys) >= len(d) * .6
    d = {'a': 1, 'b': 2, 'c': 3}
    assert dissoc(d, 'a', 'b') == {'c': 3}


def test_dissoc_factory():
    rv = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', badarg=1)


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_update_in_existing():
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


def test_update_in_creates_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_creates_nested_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    rv = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'a': 2}


def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    assert assoc_in(d, ['a', 'b'], 5) == {'a': {'b': 5}}


def test_assoc_in_creates():
    assert assoc_in({}, ['a', 'b'], 5) == {'a': {'b': 5}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 5)
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    rv = assoc_in({}, ['a'], 1, factory=OrderedDict)
    assert isinstance(rv, OrderedDict)
    assert rv == {'a': 1}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

@pytest.fixture
def transaction():
    return {'name': 'Alice',
            'purchase': {'items': ['Apple', 'Orange'],
                         'costs': [0.50, 1.25]},
            'credit card': '5555-1234-1234-1234'}


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_simple(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_index(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_bad_type(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_default(transaction):
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


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction
