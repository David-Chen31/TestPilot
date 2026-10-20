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


# -----------------------------------------------------------------------------
# merge
# -----------------------------------------------------------------------------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_dict():
    assert merge({1: 2}) == {1: 2}


def test_merge_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=list)
    assert isinstance(result, list)


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


def test_merge_factory_and_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, factory=list, bad=1)


# -----------------------------------------------------------------------------
# merge_with
# -----------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]
    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(len, {1: 'a'}, factory=list)
    assert isinstance(result, list)


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=1)


def test_merge_with_does_not_mutate():
    a = {1: 1}
    b = {1: 2}
    merge_with(sum, a, b)
    assert a == {1: 1}
    assert b == {1: 2}


# -----------------------------------------------------------------------------
# valmap
# -----------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 2, 3: 4}) == {1: 4, 3: 8}


def test_valmap_bills():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# -----------------------------------------------------------------------------
# keymap
# -----------------------------------------------------------------------------

def test_keymap():
    assert keymap(lambda x: x + 1, {1: 'a', 3: 'b'}) == {2: 'a', 4: 'b'}


def test_keymap_lower():
    bills = {"Alice": [20], "Bob": [10]}
    assert keymap(str.lower, bills) == {"alice": [20], "bob": [10]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_keymap_does_not_mutate():
    d = {1: 2}
    keymap(lambda x: x + 1, d)
    assert d == {1: 2}


# -----------------------------------------------------------------------------
# itemmap
# -----------------------------------------------------------------------------

def test_itemmap_reversed():
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == {10: "Alice", 20: "Bob"}


def test_itemmap_identity():
    d = {1: 2, 3: 4}
    assert itemmap(lambda kv: kv, d) == d


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_itemmap_does_not_mutate():
    d = {1: 2}
    itemmap(reversed, d)
    assert d == {1: 2}


# -----------------------------------------------------------------------------
# valfilter
# -----------------------------------------------------------------------------

def test_valfilter_iseven():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_none_match():
    assert valfilter(lambda v: v > 100, {1: 2, 3: 4}) == {}


def test_valfilter_factory():
    result = valfilter(bool, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_valfilter_does_not_mutate():
    d = {1: 2, 3: 4}
    valfilter(lambda v: v > 2, d)
    assert d == {1: 2, 3: 4}


# -----------------------------------------------------------------------------
# keyfilter
# -----------------------------------------------------------------------------

def test_keyfilter_iseven():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: k > 100, {1: 2}) == {}


def test_keyfilter_factory():
    result = keyfilter(bool, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_keyfilter_does_not_mutate():
    d = {1: 2, 3: 4}
    keyfilter(lambda k: k > 1, d)
    assert d == {1: 2, 3: 4}


# -----------------------------------------------------------------------------
# itemfilter
# -----------------------------------------------------------------------------

def test_itemfilter_isvalid():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(bool, {}) == {}


def test_itemfilter_none_match():
    assert itemfilter(lambda kv: False, {1: 2}) == {}


def test_itemfilter_factory():
    result = itemfilter(bool, {1: 2}, factory=list)
    assert isinstance(result, list)


def test_itemfilter_does_not_mutate():
    d = {1: 2, 3: 4}
    itemfilter(lambda kv: kv[0] > 1, d)
    assert d == {1: 2, 3: 4}


# -----------------------------------------------------------------------------
# assoc
# -----------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'x', 2)
    assert d == {'x': 1}


def test_assoc_returns_new_dict():
    d = {'x': 1}
    d2 = assoc(d, 'x', 2)
    assert d2 is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=list)
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# dissoc
# -----------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_empty():
    assert dissoc({}, 'x') == {}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=list)
    assert isinstance(result, list)


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_many_keys_branch():
    # Exercise the else branch (len(keys) >= len(d) * .6)
    d = {'a': 1, 'b': 2, 'c': 3}
    assert dissoc(d, 'a', 'b') == {'c': 3}


# -----------------------------------------------------------------------------
# assoc_in
# -----------------------------------------------------------------------------

def test_assoc_in_nested():
    purchase = {'order': {'items': ['Apple']}}
    result = assoc_in(purchase, ['order', 'items'], ['Orange'])
    assert result == {'order': {'items': ['Orange']}}


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    result = assoc_in({'a': 1}, ['b'], 2, factory=list)
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# update_in
# -----------------------------------------------------------------------------

def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_nested_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_nested_existing():
    inc = lambda x: x + 1
    assert update_in({'a': {'b': 1}}, ['a', 'b'], inc) == {'a': {'b': 2}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_single_key():
    assert update_in({'a': 1}, ['a'], lambda x: x * 10) == {'a': 10}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=list)
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# get_in
# -----------------------------------------------------------------------------

TRANSACTION = {
    'name': 'Alice',
    'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    'credit card': '5555-1234-1234-1234',
}


def test_get_in_nested():
    assert get_in(['purchase', 'items', 0], TRANSACTION) == 'Apple'


def test_get_in_single():
    assert get_in(['name'], TRANSACTION) == 'Alice'


def test_get_in_missing_default_none():
    assert get_in(['purchase', 'total'], TRANSACTION) is None


def test_get_in_missing_key_returns_default():
    assert get_in(['purchase', 'total'], TRANSACTION, 0) == 0


def test_get_in_index_error_default():
    assert get_in(['purchase', 'items', 10], TRANSACTION) is None


def test_get_in_type_error_default():
    assert get_in(['purchase', 'items', 'apple'], TRANSACTION) is None


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_indexerror():
    with pytest.raises(IndexError):
        get_in([10], [1, 2, 3], no_default=True)


def test_get_in_no_default_typeerror():
    with pytest.raises(TypeError):
        get_in(['apple'], ['a', 'b'], no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_nonmapping():
    assert get_in([0], [10, 20]) == 10
