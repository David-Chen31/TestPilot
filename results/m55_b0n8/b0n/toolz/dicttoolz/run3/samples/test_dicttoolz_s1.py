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
import collections


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_simple():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_non_mapping_iterable():
    # a single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 'one'}, {2: 'two'}]) == {1: 'one', 2: 'two'}


def test_merge_single_mapping():
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 'a'}, {2: 'b'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'a', 2: 'b'}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, foo=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_non_mapping():
    assert merge_with(sum, [{1: 1}, {1: 10}]) == {1: 11}


def test_merge_with_first():
    first = lambda seq: seq[0]
    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: 1, 2: 2, 3: 30
    }


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, foo=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: '2'}


def test_keymap():
    bills = {"Alice": [20, 15], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 'a'}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    result = itemmap(lambda kv: (kv[1], kv[0]), accountids)
    assert result == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[1], kv[0]), {1: 2},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 1}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_factory():
    result = valfilter(bool, {1: 0, 2: 1},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 1}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(bool, {0: 'a', 1: 'b'},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'b'}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == 1, {1: 'a', 2: 'b'},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# assoc / dissoc
# ---------------------------------------------------------------------------

def test_assoc_replaces():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert result == {'x': 2}
    assert d == {'x': 1}  # original unchanged


def test_assoc_adds():
    d = {'x': 1}
    result = assoc(d, 'y', 3)
    assert result == {'x': 1, 'y': 3}
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


def test_dissoc_single():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert result == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_multiple():
    d = {'x': 1, 'y': 2}
    assert dissoc(d, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y',
                    factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', foo=1)


def test_dissoc_many_keys_branch():
    # triggers the branch where len(keys) >= len(d) * 0.6
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, 5, 6)
    assert result == {7: 7, 8: 8, 9: 9}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'],
                  'costs': [0.25, 1.00]},
    }
    # original unchanged
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_new_keys():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}}
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({'a': 0}, ['a'], lambda x: x + 1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


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


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_index_error(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error(transaction):
    # indexing a list with a string raises TypeError, which is caught
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default(transaction):
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)
    with pytest.raises(TypeError):
        get_in(['purchase', 'items', 'apple'], transaction, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction
