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


def test_merge_single_non_mapping_iterable():
    # A single non-Mapping argument is treated as an iterable of dicts.
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}
    assert merge(({1: 1}, {1: 2})) == {1: 2}


def test_merge_no_args():
    assert merge() == {}


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    result = merge(a, b)
    assert result == {1: 1, 2: 2}
    assert a == {1: 1}
    assert b == {2: 2}


def test_merge_factory():
    from collections import OrderedDict
    result = merge({1: 1}, {2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, 1), (2, 2)]


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 1}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(xs):
        return xs[0]
    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_list_values():
    result = merge_with(list, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: ['a', 'b', 'c']}


def test_merge_with_single_non_mapping_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_no_args():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    from collections import OrderedDict
    result = merge_with(sum, {1: 1}, {2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {1: 1, 2: 2}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=True)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_factory():
    from collections import OrderedDict
    result = valmap(str, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {1: '1', 2: '2'}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str.lower, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict
    result = keymap(str, {1: 'a'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {'1': 'a'}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(reversed, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict
    result = itemmap(lambda kv: (kv[0], kv[1] * 2), {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {1: 4}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    from collections import OrderedDict
    result = valfilter(lambda v: v > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {2: 2}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict
    result = keyfilter(lambda k: k == 2, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {2: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict
    result = itemfilter(lambda kv: kv[0] == 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {1: 1}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_no_mutation():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}
    assert result is not d


def test_assoc_factory():
    from collections import OrderedDict
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one_key():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple_keys():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_no_mutation():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}
    assert result == {'x': 1}


def test_dissoc_few_keys_branch():
    # Few keys relative to dict size -> update-then-delete branch.
    d = {i: i for i in range(10)}
    assert dissoc(d, 0) == {i: i for i in range(1, 10)}


def test_dissoc_many_keys_branch():
    # Many keys relative to dict size -> rebuild branch.
    d = {1: 1, 2: 2, 3: 3}
    assert dissoc(d, 1, 2) == {3: 3}


def test_dissoc_factory():
    from collections import OrderedDict
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert dict(result) == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=True)


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
    assert result['order']['costs'] == [0.25, 1.00]
    # Other data untouched
    assert result['name'] == 'Alice'
    assert result['order']['items'] == ['Apple', 'Orange']
    # Original not mutated
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


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_update_in_missing_uses_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_nested_existing():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_factory():
    from collections import OrderedDict
    result = update_in({}, ['a'], lambda x: x + 1, 0, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert dict(result) == {'a': 1}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_existing():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_default():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([0], [], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in((), [1, 2]) == [1, 2]


def test_get_in_empty_keys_no_default():
    # No lookup error occurs, so default is irrelevant
    assert get_in([], {'a': 1}, default=5) == {'a': 1}
