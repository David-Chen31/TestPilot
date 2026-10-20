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


def test_merge_single_mapping():
    d = {1: 2}
    result = merge(d)
    assert result == {1: 2}
    assert result is not d


def test_merge_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_iterable_of_dicts_with_factory():
    result = merge([{1: 2}], factory=dict)
    assert result == {1: 2}


def test_merge_factory():
    result = merge({1: 2}, factory=dict)
    assert result == {1: 2}
    assert type(result) is dict


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bogus=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first_like():
    assert merge_with(lambda vals: vals[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_single_values():
    # Function receives a list even for keys appearing once
    seen = {}

    def capture(vals):
        seen['last'] = list(vals)
        return sum(vals)

    result = merge_with(capture, {1: 5}, {2: 7})
    assert result == {1: 5, 2: 7}
    assert list(seen['last']) in ([5], [7])


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=dict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bogus=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = valmap(lambda x: x * 10, d)
    assert result == {1: 20, 3: 40}
    assert d == {1: 2, 3: 4}


def test_valmap_factory():
    d = {1: 2}
    result = valmap(lambda x: x + 1, d, factory=dict)
    assert result == {1: 3}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_does_not_mutate():
    d = {1: 'a', 2: 'b'}
    result = keymap(lambda k: k + 10, d)
    assert result == {11: 'a', 12: 'b'}
    assert d == {1: 'a', 2: 'b'}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = itemmap(lambda kv: (kv[0] + 1, kv[1] * 2), d)
    assert result == {2: 4, 4: 8}
    assert d == {1: 2, 3: 4}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    assert valfilter(lambda v: v > 2, d) == {2: 3}
    assert keyfilter(lambda k: k > 1, d) == {2: 3}
    assert itemfilter(lambda kv: kv[0] == 1, d) == {1: 2}
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert result == {'x': 1, 'y': 2}
    assert d == {'x': 1}
    assert result is not d


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'x')
    assert result == {'y': 2}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_many_keys_branch():
    # Many keys -> exercises the set-difference branch
    d = {i: i for i in range(100)}
    result = dissoc(d, *range(90))
    assert result == {i: i for i in range(90, 100)}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'x', factory=dict)
    assert result == {'y': 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bogus=1)


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]
    assert result is not purchase


def test_assoc_in_new_path():
    result = assoc_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': {'c': 1}}}
    result = update_in(d, ['a', 'b', 'c'], inc)
    assert result == {'a': {'b': {'c': 2}}}
    assert d == {'a': {'b': {'c': 1}}}


def test_update_in_missing_keys_creates_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_existing():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_no_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=dict)
    assert result == {'a': 2}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_basic():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple'], 'costs': [0.5]}}
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_index_error():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error():
    transaction = {'purchase': {'items': ['Apple']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 0], {'a': 5}, no_default=True)
