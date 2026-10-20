import pytest
from collections import OrderedDict, defaultdict

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
)


# ---------------------------------------------------------------- merge

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_sequence_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping():
    # A single Mapping stays wrapped, not unpacked as an iterable of keys
    assert merge({1: 2}) == {1: 2}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ------------------------------------------------------------ merge_with

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict_values_lists():
    assert merge_with(list, {1: 2}) == {1: [2]}


def test_merge_with_first_item():
    def first(seq):
        return seq[0]
    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == \
        {1: 1, 2: 2, 3: 30}


def test_merge_with_sequence_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, nope=1)


# ---------------------------------------------------------------- valmap

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 2, 3: 4}) == {1: 4, 3: 8}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------- keymap

def test_keymap():
    bills = {"Alice": [20], "Bob": [10]}
    assert keymap(str.lower, bills) == {"alice": [20], "bob": [10]}


def test_keymap_factory():
    result = keymap(lambda k: k + 1, {1: 'a'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 'a'}


# --------------------------------------------------------------- itemmap

def test_itemmap():
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == \
        {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ------------------------------------------------------------- valfilter

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_all_removed():
    assert valfilter(lambda v: False, {1: 2, 3: 4}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ------------------------------------------------------------- keyfilter

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 1, {1: 2, 3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ------------------------------------------------------------ itemfilter

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------- assoc

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# --------------------------------------------------------------- dissoc

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}
    assert result is not d


def test_dissoc_many_keys_branch():
    # Triggers the else-branch (many keys relative to dict size)
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    assert dissoc(d, 1, 2, 3, 4, 5) == {}


def test_dissoc_more_keys_than_dict():
    assert dissoc({1: 'a'}, 1, 2, 3) == {}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', nope=1)


# ------------------------------------------------------------- assoc_in

def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {'name': 'Alice',
                      'order': {'items': ['Apple', 'Orange'],
                                'costs': [0.25, 1.00]}}


def test_assoc_in_does_not_mutate():
    purchase = {'order': {'costs': [0.5]}}
    assoc_in(purchase, ['order', 'costs'], [1.0])
    assert purchase == {'order': {'costs': [0.5]}}


def test_assoc_in_new_nested_path():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


# ------------------------------------------------------------ update_in

def test_update_in_simple():
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


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == \
        {1: {2: {3: 'bar'}}}


def test_update_in_existing_plus_new():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 2}


# --------------------------------------------------------------- get_in

@pytest.fixture
def transaction():
    return {'name': 'Alice',
            'purchase': {'items': ['Apple', 'Orange'],
                         'costs': [0.50, 1.25]},
            'credit card': '5555-1234-1234-1234'}


def test_get_in_nested_list(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_simple(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_with_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_type_error_returns_default(transaction):
    # 'apple' index into a list raises TypeError
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_error_returns_default(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_success(transaction):
    assert get_in(['name'], transaction, no_default=True) == 'Alice'
