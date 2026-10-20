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


def test_merge_single_sequence_of_dicts():
    dicts = [{1: 'one'}, {2: 'two'}]
    assert merge(dicts) == {1: 'one', 2: 'two'}


def test_merge_single_dict_is_mapping():
    # A single dict is a Mapping, so it's treated as one dict, not iterated.
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 1}, {2: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [(1, 1), (2, 2)]


def test_merge_factory_sequence():
    dicts = [{1: 1}, {2: 2}]
    result = merge(dicts, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1, 2: 2}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 1}, badkwarg=1)


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    result = merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}
    assert result == {1: 1, 2: 2}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_partial_overlap():
    result = merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_single_sequence():
    dicts = [{1: 1}, {1: 2}]
    assert merge_with(sum, dicts) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, notakwarg=1)


def test_merge_with_order_of_values():
    result = merge_with(list, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: ['a', 'b', 'c']}


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    result = valmap(str.upper, {'a': 'x'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 'X'}


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
    result = keymap(str.upper, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'A': 1}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_addition():
    assert itemmap(lambda kv: (kv[0], kv[1] + 1), {'a': 1}) == {'a': 2}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_none_pass():
    assert valfilter(lambda v: False, {1: 2, 3: 4}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: v > 1, {'a': 1, 'b': 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'b': 2}


def test_valfilter_no_mutation():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v == 2, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_none_pass():
    assert keyfilter(lambda k: False, {1: 2}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 'a', {'a': 1, 'b': 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_none_pass():
    assert itemfilter(lambda item: False, {1: 2}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: item[0] == 'a', {'a': 1, 'b': 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result is not d


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 3, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 3}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_basic():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}
    assert result is not d


def test_dissoc_many_keys_branch():
    # Exercise the branch where len(keys) >= len(d) * 0.6
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert dissoc(d, 'a', 'b', 'c') == {'d': 4}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', notakwarg=1)


def test_dissoc_missing_key_in_update_branch():
    # With a large dict and few keys, dissoc takes the update() branch. In
    # that branch, a key that is not present must be skipped via `if key in d2`.
    d = {i: i for i in range(10)}
    result = dissoc(d, 'missing')
    assert result == d
    assert result is not d


def test_dissoc_missing_and_present_keys_in_update_branch():
    # Mix of missing and present keys exercises both the `if key in d2`
    # branch and the `del` branch, in the update() code path.
    d = {i: i for i in range(10)}
    result = dissoc(d, 'missing', 3, 7)
    expected = {0: 0, 1: 1, 2: 2, 4: 4, 5: 5, 6: 6, 8: 8, 9: 9}
    assert result == expected
    # Original is not mutated.
    assert d == {i: i for i in range(10)}


def test_dissoc_all_missing_keys_in_update_branch():
    d = {i: i for i in range(10)}
    result = dissoc(d, 'a', 'b')
    assert result == d
    assert result is not d


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': 1}}
    assert update_in(d, ['a', 'b'], inc) == {'a': {'b': 2}}


def test_update_in_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_default_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo',
        2: {3: {4: 1}},
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 2}


def test_update_in_single_key_missing():
    assert update_in({}, ['a'], lambda x: x + 1, default=0) == {'a': 1}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 5, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 5}}


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


def test_get_in_basic(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_index_type(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_type_error():
    with pytest.raises(TypeError):
        get_in([0], 5, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction


def test_get_in_with_lists():
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
