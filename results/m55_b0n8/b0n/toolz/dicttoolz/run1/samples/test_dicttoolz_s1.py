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


# ---------- merge ----------

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_mapping():
    # A single dict should be treated as itself, not iterated as a list of dicts
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_single_sequence_of_dicts():
    assert merge([{1: 'one'}, {2: 'two'}]) == {1: 'one', 2: 'two'}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    import collections
    result = merge({1: 'a'}, {2: 'b'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'a', 2: 'b'}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, badarg=True)


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}


# ---------- merge_with ----------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_single_mapping():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_sequence_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    import collections
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, badarg=True)


def test_merge_with_receives_list():
    collected = {}

    def capture(vals):
        collected['vals'] = list(vals)
        return len(vals)

    result = merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: 3}
    assert collected['vals'] == ['a', 'b', 'c']


# ---------- valmap ----------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 10, 2: 20}) == {1: 20, 2: 40}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    import collections
    result = valmap(lambda x: x + 1, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 2}


def test_valmap_no_mutation():
    d = {1: 1}
    valmap(lambda x: x + 1, d)
    assert d == {1: 1}


# ---------- keymap ----------

def test_keymap():
    assert keymap(str.lower, {'A': 1, 'B': 2}) == {'a': 1, 'b': 2}


def test_keymap_empty():
    assert keymap(str.lower, {}) == {}


def test_keymap_factory():
    import collections
    result = keymap(str.lower, {'A': 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------- itemmap ----------

def test_itemmap():
    assert itemmap(lambda kv: (kv[0] * 10, kv[1]), {1: 'a', 2: 'b'}) == {
        10: 'a', 20: 'b'
    }


def test_itemmap_reverse():
    assert itemmap(reversed, {'Alice': 10, 'Bob': 20}) == {10: 'Alice', 20: 'Bob'}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    import collections
    result = itemmap(lambda kv: (kv[0], kv[1] + 1), {'a': 1},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 2}


# ---------- valfilter ----------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    assert valfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {1: 2, 3: 4}


def test_valfilter_none_left():
    assert valfilter(lambda v: False, {1: 1, 2: 2}) == {}


def test_valfilter_factory():
    import collections
    result = valfilter(lambda v: v > 0, {1: 1},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


# ---------- keyfilter ----------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    assert keyfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3, 4: 5}


def test_keyfilter_none_left():
    assert keyfilter(lambda k: False, {1: 1, 2: 2}) == {}


def test_keyfilter_factory():
    import collections
    result = keyfilter(lambda k: k > 0, {1: 1},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


# ---------- itemfilter ----------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3}


def test_itemfilter_none_left():
    assert itemfilter(lambda item: False, {1: 1}) == {}


def test_itemfilter_factory():
    import collections
    result = itemfilter(lambda item: item[0] == 1, {1: 1},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


# ---------- assoc ----------

def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}


def test_assoc_factory():
    import collections
    result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------- dissoc ----------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    import collections
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', badarg=True)


def test_dissoc_both_branches():
    # Few keys -> update-then-delete branch
    d_small = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    assert dissoc(d_small, 'a') == {'b': 2, 'c': 3, 'd': 4, 'e': 5}

    # Many keys -> remaining-set branch
    assert dissoc({'a': 1, 'b': 2}, 'a', 'b') == {}


# ---------- assoc_in ----------

def test_assoc_in_existing_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_partial():
    assert assoc_in({'a': {'x': 1}}, ['a', 'b'], 2) == {'a': {'x': 1, 'b': 2}}


def test_assoc_in_no_mutation():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


# ---------- update_in ----------

def test_update_in_existing():
    assert update_in({'a': 0}, ['a'], lambda x: x + 1) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'


def test_update_in_missing_keys_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_keys_default_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_no_mutation():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    import collections
    result = update_in({'a': {'b': 1}}, ['a', 'b'], lambda x: x + 1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 2}}


# ---------- get_in ----------

def test_get_in_existing():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'costs'], transaction) == [0.50, 1.25]


def test_get_in_missing_default_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_default_value():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_bad_list_index():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_key_error():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([0, 5], [1, 2, 3], no_default=True)


def test_get_in_no_default_success():
    assert get_in(['a', 'b'], {'a': {'b': 42}}, no_default=True) == 42


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2]) == [1, 2]


def test_get_in_type_error_no_default():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)
