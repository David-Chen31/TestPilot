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


# ---------------------------------------------------------------- merge

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_wins():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_iterable_of_dicts():
    dicts = [{1: 'one'}, {2: 'two'}]
    assert merge(dicts) == {1: 'one', 2: 'two'}


def test_merge_single_mapping_is_not_unpacked():
    # A single Mapping arg is treated as a dict itself (not iterated).
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_no_args():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, factory=collections.OrderedDict)
    assert result == {1: 2}
    assert isinstance(result, collections.OrderedDict)


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, foo=3)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {2: 3}
    merge(a, b)
    assert a == {1: 2}
    assert b == {2: 3}


# ---------------------------------------------------------------- merge_with

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == \
        {1: 1, 2: 2, 3: 30}


def test_merge_with_single_iterable():
    dicts = [{1: 1}, {1: 2}, {1: 3}]
    assert merge_with(sum, dicts) == {1: 6}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2},
                        factory=collections.OrderedDict)
    assert result == {1: 3}
    assert isinstance(result, collections.OrderedDict)


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, nope=1)


def test_merge_with_single_value_passed_as_list():
    captured = {}

    def capture(seq):
        captured['seq'] = seq
        return seq[0]

    merge_with(capture, {1: 5})
    assert captured['seq'] == [5]


# ---------------------------------------------------------------- valmap

def test_valmap():
    assert valmap(lambda x: x + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}


def test_valmap_factory():
    result = valmap(str, {1: 10}, factory=collections.OrderedDict)
    assert result == {1: '10'}
    assert isinstance(result, collections.OrderedDict)


def test_valmap_empty():
    assert valmap(str, {}) == {}


# ---------------------------------------------------------------- keymap

def test_keymap():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_factory():
    result = keymap(str, {1: 'x'}, factory=collections.OrderedDict)
    assert result == {'1': 'x'}
    assert isinstance(result, collections.OrderedDict)


def test_keymap_empty():
    assert keymap(str, {}) == {}


# ---------------------------------------------------------------- itemmap

def test_itemmap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2}) == \
        {1: 'a', 2: 'b'}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=collections.OrderedDict)
    assert result == {1: 2}
    assert isinstance(result, collections.OrderedDict)


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


# ---------------------------------------------------------------- valfilter

def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: v, {1: 0, 2: 1},
                       factory=collections.OrderedDict)
    assert result == {2: 1}
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------- keyfilter

def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 1, {1: 'a', 2: 'b'},
                       factory=collections.OrderedDict)
    assert result == {1: 'a'}
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------- itemfilter

def test_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == 1, {1: 'a', 2: 'b'},
                        factory=collections.OrderedDict)
    assert result == {1: 'a'}
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------- assoc

def test_assoc_overwrite():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------- dissoc

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
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y',
                    factory=collections.OrderedDict)
    assert result == {'x': 1}
    assert isinstance(result, collections.OrderedDict)


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_few_keys_branch_and_many_keys_branch():
    # Few keys relative to dict size -> update/delete branch
    d = {i: i for i in range(10)}
    assert dissoc(d, 0) == {i: i for i in range(1, 10)}
    # Many keys -> rebuild branch
    d2 = {i: i for i in range(4)}
    assert dissoc(d2, 0, 1, 2) == {3: 3}


# ---------------------------------------------------------------- assoc_in

def test_assoc_in_nested():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # original not mutated
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_new_top_key():
    assert assoc_in({'x': 1}, ['y'], 2) == {'x': 1, 'y': 2}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert result == {'a': {'b': 1}}
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)


# ---------------------------------------------------------------- update_in

def test_update_in_basic():
    assert update_in({'a': 0}, ['a'], lambda x: x + 1) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == \
        {1: {2: {3: 'bar'}}}


def test_update_in_preserves_other_keys():
    assert update_in({1: 'foo'}, [2, 3, 4], lambda x: x + 1, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x + 1, 0,
                       factory=collections.OrderedDict)
    assert result == {'a': {'b': 1}}
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


# ---------------------------------------------------------------- get_in

def test_get_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_index_returns_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_type_error_caught():
    # Indexing an int with a key raises TypeError -> default
    assert get_in(['a', 'b'], {'a': 5}, default='dflt') == 'dflt'


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in(['a', 5], {'a': [1, 2]}, no_default=True)


def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    assert get_in([], coll) is coll
