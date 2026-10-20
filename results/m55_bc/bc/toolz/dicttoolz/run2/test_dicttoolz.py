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


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_iterable_of_dicts():
    # len==1 and element is not a Mapping -> treated as an iterable
    result = merge([{1: 2}, {3: 4}])
    assert result == {1: 2, 3: 4}


def test_merge_no_dicts():
    assert merge() == {}


def test_merge_factory():
    from collections import OrderedDict
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, foo=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_key():
    assert merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: [1], 2: [2, 20], 3: [30]
    }


def test_merge_with_iterable_of_dicts():
    result = merge_with(sum, [{1: 1}, {1: 2}])
    assert result == {1: 3}


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    from collections import OrderedDict
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, foo=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_does_not_mutate():
    d = {1: 2, 3: 4}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2, 3: 4}


def test_valmap_factory():
    from collections import OrderedDict
    result = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30], "bob": [10, 35]
    }


def test_keymap_factory():
    from collections import OrderedDict
    result = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"A": 1}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    from collections import OrderedDict
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: "a"}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: "a"}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    from collections import OrderedDict
    result = valfilter(lambda v: v > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    from collections import OrderedDict
    result = keyfilter(lambda k: k > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
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


def test_itemfilter_factory():
    from collections import OrderedDict
    result = itemfilter(lambda item: item[0] > 1, {1: 1, 2: 2},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert result == {'x': 2}
    assert d == {'x': 1}


def test_assoc_new_key():
    d = {'x': 1}
    result = assoc(d, 'y', 3)
    assert result == {'x': 1, 'y': 3}
    assert d == {'x': 1}


def test_assoc_factory():
    from collections import OrderedDict
    result = assoc({1: 1}, 2, 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1, 2: 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    d = {'x': 1, 'y': 2}
    assert dissoc(d, 'y') == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2, 'z': 3}
    dissoc(d, 'x')
    assert d == {'x': 1, 'y': 2, 'z': 3}


def test_dissoc_large_key_set_branch():
    # len(keys) >= len(d) * 0.6 triggers the else branch
    d = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    keys = (1, 2, 3, 4, 5, 6, 7)
    result = dissoc(d, *keys)
    expected = {8: 8, 9: 9, 10: 10}
    assert result == expected
    assert d == {i: i for i in range(1, 11)}


def test_dissoc_threshold_branch():
    # len(keys) < len(d) * .6 triggers the update/delete branch
    d = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    keys = (1, 2)
    result = dissoc(d, *keys)
    assert result == {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}


def test_dissoc_update_branch_missing_key():
    # Hits the update/delete branch where ``key in d2`` is False, i.e. the
    # branch at line 220 is not taken.
    d = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    # 1 key out of 10 -> len(keys)=1 < 6.0 -> update/delete branch; key 99
    # is not present so ``if key in d2`` is False.
    result = dissoc(d, 99)
    assert result == {i: i for i in range(1, 11)}
    assert d == {i: i for i in range(1, 11)}


def test_dissoc_update_branch_mixed_present_and_missing():
    # Exercises the update/delete loop where some keys are present and some
    # are missing: present keys are deleted, missing keys are skipped.
    d = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    result = dissoc(d, 1, 42, 2)
    assert result == {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
    assert d == {i: i for i in range(1, 11)}


def test_dissoc_factory():
    from collections import OrderedDict
    result = dissoc({1: 1, 2: 2}, 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({1: 2}, 1, foo=1)


def test_dissoc_empty_dict():
    assert dissoc({}, 'x') == {}


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_nested():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
        'credit card': '5555-1234-1234-1234',
    }
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': 1.75},
        'credit card': '5555-1234-1234-1234',
    }
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_key_creates_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_existing():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    from collections import OrderedDict
    result = update_in({}, [1, 2], lambda x: x + 1, default=0,
                       factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result[1], OrderedDict)
    assert result == {1: {2: 1}}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_type_error_no_default():
    with pytest.raises(TypeError):
        get_in(['x', 'y'], {'x': 1}, no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}
