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


def test_merge_single_sequence_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_no_args():
    assert merge() == {}


def test_merge_factory():
    rv = merge({1: 2}, factory=dict)
    assert rv == {1: 2}
    assert type(rv) is dict


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ----------------------------- merge_with -----------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: 1, 2: 2, 3: 30
    }


def test_merge_with_single_sequence():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    rv = merge_with(sum, {1: 1}, factory=dict)
    assert rv == {1: 1}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=True)


# ----------------------------- valmap -----------------------------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_factory():
    rv = valmap(sum, {"a": [1]}, factory=dict)
    assert rv == {"a": 1}


# ----------------------------- keymap -----------------------------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30], "bob": [10, 35]
    }


def test_keymap_empty():
    assert keymap(str.lower, {}) == {}


def test_keymap_factory():
    rv = keymap(str.lower, {"A": 1})
    assert rv == {"a": 1}


# ----------------------------- itemmap -----------------------------

def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_identity():
    d = {"a": 1, "b": 2}
    assert itemmap(lambda kv: kv, d) == d


def test_itemmap_empty():
    assert itemmap(reversed, {}) == {}


# ----------------------------- valfilter -----------------------------

def test_valfilter_iseven():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_all_false():
    assert valfilter(lambda x: False, {1: 2, 3: 4}) == {}


# ----------------------------- keyfilter -----------------------------

def test_keyfilter_iseven():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


# ----------------------------- itemfilter -----------------------------

def test_itemfilter_isvalid():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda x: True, {}) == {}


# ----------------------------- assoc -----------------------------

def test_assoc_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 3)
    assert d == {'x': 1}


def test_assoc_factory():
    rv = assoc({'x': 1}, 'y', 2, factory=dict)
    assert rv == {'x': 1, 'y': 2}


# ----------------------------- dissoc -----------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_many():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad_kwarg=True)


def test_dissoc_factory():
    rv = dissoc({'x': 1, 'y': 2}, 'y', factory=dict)
    assert rv == {'x': 1}


def test_dissoc_many_keys_takes_other_branch():
    # len(keys) >= len(d) * .6 exercises the difference_update branch
    d = {1: 'a', 2: 'b', 3: 'c'}
    assert dissoc(d, 1, 2) == {3: 'c'}


def test_dissoc_branch_with_present_and_absent_keys():
    # len(keys) < len(d) * 0.6 exercises the update-then-delete branch.
    # Include both a key that is present (takes `if key in d2` true branch)
    # and a key that is absent (takes the false branch, skipped).
    d = {1: 'a', 2: 'b', 3: 'c'}
    # 1 key is < 3 * 0.6 == 1.8, so the first branch is used.
    assert dissoc(d, 1) == {2: 'b', 3: 'c'}
    assert dissoc(d, 99) == {1: 'a', 2: 'b', 3: 'c'}
    # Also assert that the returned dict is a fresh copy.
    rv = dissoc(d, 99)
    assert rv == d
    assert rv is not d


def test_dissoc_first_branch_deletes_present_key():
    # Directly target `if key in d2:` being True on the first branch.
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    # 2 keys < 5 * 0.6 == 3.0, so the first branch is taken.
    assert dissoc(d, 2, 4) == {1: 'a', 3: 'c', 5: 'e'}


# ----------------------------- assoc_in -----------------------------

def test_assoc_in_nested_new_key():
    d = {'a': {'b': 1}}
    assert assoc_in(d, ['a', 'c'], 2) == {'a': {'b': 1, 'c': 2}}


def test_assoc_in_overwrite():
    d = {'a': {'b': 1}}
    assert assoc_in(d, ['a', 'b'], 5) == {'a': {'b': 5}}


def test_assoc_in_creates_nesting():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 5)
    assert d == {'a': {'b': 1}}


# ----------------------------- update_in -----------------------------

def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': 1}}
    assert update_in(d, ['a', 'b'], inc) == {'a': {'b': 2}}


def test_update_in_missing_key_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}
    }


def test_update_in_partial_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    inc = lambda x: x + 1
    rv = update_in({'a': 0}, ['a'], inc, factory=dict)
    assert rv == {'a': 1}


# ----------------------------- get_in -----------------------------

def test_get_in_nested():
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


def test_get_in_missing_key_default_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_index_default_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error_branch():
    # indexing a string with a string raises TypeError
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_default_value():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([0], [], no_default=True)
