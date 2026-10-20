import collections
import pytest

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter,
    assoc, dissoc, assoc_in, update_in, get_in,
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


def test_merge_single_dict():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_iterable_of_dicts():
    # When a single non-Mapping argument is given, it is treated as an
    # iterable of dicts.
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_iterable_generator():
    assert merge(iter([{1: 2}, {3: 4}])) == {1: 2, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 3: 4}
    assert list(result) == [1, 3]


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, notakwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_values_lists():
    captured = {}

    def capture(vals):
        captured['vals'] = vals
        return len(vals)

    result = merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'}, {2: 'x'})
    assert result == {1: 3, 2: 1}
    assert captured['vals'] == ['a', 'b', 'c'] or captured['vals'] == ['x']


def test_merge_with_single_value():
    assert merge_with(sum, {1: 5, 2: 6}) == {1: 5, 2: 6}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, notakwarg=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = valmap(lambda x: x * 2, d)
    assert result == {1: 4, 3: 8}
    assert d == {1: 2, 3: 4}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: '2'}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_does_not_mutate():
    d = {1: 'a', 2: 'b'}
    result = keymap(lambda x: x * 10, d)
    assert result == {10: 'a', 20: 'b'}
    assert d == {1: 'a', 2: 'b'}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 2}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    result = itemmap(lambda kv: (kv[1], kv[0]), accountids)
    assert result == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_does_not_mutate():
    d = {1: 2}
    result = itemmap(lambda kv: (kv[0] + 1, kv[1] + 1), d)
    assert result == {2: 3}
    assert d == {1: 2}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_none_match():
    assert valfilter(lambda v: False, {1: 1, 2: 2}) == {}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2},
                       factory=collections.OrderedDict)
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
    assert keyfilter(bool, {}) == {}


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k > 1, d)
    assert d == {1: 2, 2: 3}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


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
    assert itemfilter(bool, {}) == {}


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda item: item[0] == 2, d)
    assert d == {1: 2, 2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_overwrite():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 3)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_no_keys():
    d = {'x': 1, 'y': 2}
    result = dissoc(d)
    assert result == {'x': 1, 'y': 2}
    assert result is not d


def test_dissoc_few_keys_path():
    # few keys relative to size -> update-then-delete branch
    d = {i: i for i in range(10)}
    result = dissoc(d, 1, 2)
    expected = {i: i for i in range(10) if i not in (1, 2)}
    assert result == expected


def test_dissoc_many_keys_path():
    # many keys -> set-difference branch
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(8))
    assert result == {8: 8, 9: 9}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y',
                    factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', notakwarg=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_assoc_in_new_nested():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_partially_existing():
    assert assoc_in({1: 'foo'}, [2, 3, 4], 1) == {1: 'foo', 2: {3: {4: 1}}}


def test_assoc_in_does_not_mutate():
    purchase = {'order': {'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {'order': {'costs': [0.25, 1.00]}}
    assert purchase == {'order': {'costs': [0.50, 1.25]}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    trans = {'purchase': {'costs': [0.50, 1.25]}}
    assert update_in(trans, ['purchase', 'costs'], sum) == {
        'purchase': {'costs': 1.75}}


def test_update_in_missing_creates_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}}


def test_update_in_missing_with_default_func():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x + 1, 0,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


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


def test_get_in_single_key(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_returns_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_missing_index(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_bad_type(transaction):
    # indexing list with a string -> TypeError
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(['purchase', 'items', 'apple'], transaction, no_default=True)


def test_get_in_nested_default_used_only_on_error():
    assert get_in(['a', 'b'], {'a': {'b': 0}}, 5) == 0
