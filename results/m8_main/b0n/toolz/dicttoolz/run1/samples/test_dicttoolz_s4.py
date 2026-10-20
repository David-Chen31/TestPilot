import collections
import collections.abc
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


def test_merge_single_seq_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_factory():
    result = merge({1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        merge({1: 2}, bad_kwarg=1)


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


def test_merge_with_first():
    assert merge_with(lambda vals: vals[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_single_non_mapping():
    assert merge_with(sum, [{1: 2}, {1: 3}]) == {1: 5}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        merge_with(sum, {1: 1}, bad=2)


def test_merge_with_empty():
    assert merge_with(sum) == {}


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30],
                                        "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(lambda k: k + 1, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 'a'}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {
        10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 'a'},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 'a'}


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
    result = valfilter(lambda v: v > 1, {1: 1, 2: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


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
    result = keyfilter(lambda k: k > 1, {1: 1, 2: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


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
    result = itemfilter(lambda item: item[0] == 1, {1: 1, 2: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_update_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_no_mutation():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'x', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one_key():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple_keys():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_mutation():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_empty():
    assert dissoc({}, 'x') == {}


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_simple():
    assert assoc_in({'a': 0}, ['a'], 1) == {'a': 1}


def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {'name': 'Alice',
                      'order': {'items': ['Apple', 'Orange'],
                                'costs': [0.25, 1.00]}}


def test_assoc_in_creates_nested():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_no_mutation():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {'name': 'Alice',
                      'purchase': {'items': ['Apple', 'Orange'],
                                   'costs': 1.75}}


def test_update_in_missing_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == \
        {1: {2: {3: 'bar'}}}


def test_update_in_missing_with_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_no_mutation():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 2}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

TRANSACTION = {'name': 'Alice',
               'purchase': {'items': ['Apple', 'Orange'],
                            'costs': [0.50, 1.25]},
               'credit card': '5555-1234-1234-1234'}


def test_get_in_nested():
    assert get_in(['purchase', 'items', 0], TRANSACTION) == 'Apple'


def test_get_in_single():
    assert get_in(['name'], TRANSACTION) == 'Alice'


def test_get_in_missing_key_default():
    assert get_in(['purchase', 'total'], TRANSACTION) is None


def test_get_in_missing_bad_index_type():
    assert get_in(['purchase', 'items', 'apple'], TRANSACTION) is None


def test_get_in_index_out_of_range():
    assert get_in(['purchase', 'items', 10], TRANSACTION) is None


def test_get_in_custom_default():
    assert get_in(['purchase', 'total'], TRANSACTION, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_index():
    with pytest.raises(IndexError):
        get_in(['items', 10], {'items': [1, 2]}, no_default=True)


def test_get_in_empty_keys():
    # reduce with empty keys returns the collection itself
    assert get_in([], TRANSACTION) is TRANSACTION
