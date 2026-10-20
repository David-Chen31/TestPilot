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


# ----------------------------- merge -----------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_non_mapping_iterable():
    # A single non-Mapping arg is treated as an iterable of dicts
    assert merge([{1: 'one'}, {2: 'two'}]) == {1: 'one', 2: 'two'}


def test_merge_single_mapping():
    # A single Mapping arg is treated as one dict to merge
    d = {1: 'one'}
    assert merge(d) == {1: 'one'}
    assert merge(d) is not d


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError) as exc:
        merge({1: 2}, bogus=1)
    assert 'bogus' in str(exc.value)


# --------------------------- merge_with ---------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_lists():
    seen = {}

    def capture(vals):
        return list(vals)

    result = merge_with(capture, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_order_preserved():
    # Values should be passed in the order from dicts arguments
    order = []

    def rec(vals):
        order.append(list(vals))
        return sum(vals)

    merge_with(rec, {'a': 1}, {'a': 2}, {'a': 3})
    assert order == [[1, 2, 3]]


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bogus=1)


# ----------------------------- valmap -----------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: '2'}


# ----------------------------- keymap -----------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 'a'}


# ---------------------------- itemmap -----------------------------

def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0], kv[1]), {1: 2},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


# ---------------------------- valfilter ---------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: v > 1, {1: 1, 2: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


# ---------------------------- keyfilter ---------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 1, {1: 1, 2: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


# ---------------------------- itemfilter --------------------------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == 2, {1: 1, 2: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


# ------------------------------ assoc -----------------------------

def test_assoc_existing_key():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert result == {'x': 2}
    assert d == {'x': 1}  # original unchanged


def test_assoc_new_key():
    d = {'x': 1}
    assert assoc(d, 'y', 3) == {'x': 1, 'y': 3}
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({}, 'a', 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ------------------------------ dissoc ----------------------------

def test_dissoc_single_key():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert result == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_multiple_keys():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    d = {'x': 1}
    assert dissoc(d, 'y') == {'x': 1}


def test_dissoc_no_keys():
    d = {1: 2, 3: 4}
    assert dissoc(d) == {1: 2, 3: 4}
    assert dissoc(d) is not d


def test_dissoc_many_keys_branch():
    # Many keys triggers the "remaining" set branch
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(8))
    assert result == {8: 8, 9: 9}
    assert d == {i: i for i in range(10)}


def test_dissoc_few_keys_branch():
    d = {i: i for i in range(10)}
    result = dissoc(d, 0)
    expected = {i: i for i in range(1, 10)}
    assert result == expected


def test_dissoc_factory():
    result = dissoc({1: 2, 3: 4}, 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {3: 4}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({1: 2}, 1, bogus=1)


# ----------------------------- assoc_in ---------------------------

def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    # original unchanged
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nesting():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------- update_in ---------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': {'c': 1}}}
    result = update_in(d, ['a', 'b', 'c'], inc)
    assert result == {'a': {'b': {'c': 2}}}
    assert d == {'a': {'b': {'c': 1}}}


def test_update_in_missing_creates_nesting():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default_func():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_factory():
    inc = lambda x: x + 1
    result = update_in({}, ['a'], inc, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


# ------------------------------ get_in ----------------------------

def test_get_in_basic():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)


def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    assert get_in([], coll) == coll
