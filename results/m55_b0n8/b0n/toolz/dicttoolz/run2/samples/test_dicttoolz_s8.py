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


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 'a'}) == {1: 'a'}


def test_merge_single_iterable_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 'a'}, {2: 'b'}]) == {1: 'a', 2: 'b'}


def test_merge_single_iterable_generator():
    gen = ({1: 'a'}, {2: 'b'})
    assert merge(gen) == {1: 'a', 2: 'b'}


def test_merge_does_not_mutate_inputs():
    d1 = {1: 'a'}
    d2 = {2: 'b'}
    merge(d1, d2)
    assert d1 == {1: 'a'}
    assert d2 == {2: 'b'}


def test_merge_factory():
    from collections import OrderedDict
    result = merge({1: 'a'}, {2: 'b'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'a', 2: 'b'}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 'a'}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_list_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_values_are_lists():
    captured = []

    def func(values):
        captured.append(list(values))
        return sum(values)

    result = merge_with(func, {1: 1, 2: 2}, {1: 10})
    assert result == {1: 11, 2: 2}
    # values passed in dict iteration order: key 1 has [1, 10]
    assert [1, 10] in captured


def test_merge_with_factory():
    from collections import OrderedDict
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_factory():
    from collections import OrderedDict
    result = valmap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: '2'}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": 1, "Bob": 2}
    assert keymap(str.lower, bills) == {"alice": 1, "bob": 2}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict
    result = keymap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'1': 2}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_tuple_func():
    assert itemmap(lambda kv: (kv[1], kv[0]), {1: 2, 3: 4}) == {2: 1, 4: 3}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict
    result = itemmap(reversed, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 1}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_factory():
    from collections import OrderedDict
    result = valfilter(bool, {1: 0, 2: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 1}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict
    result = keyfilter(bool, {0: 1, 1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(bool, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict
    result = itemfilter(lambda kv: kv[1] > 0, {1: 0, 2: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 1}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(bool, d)
    keyfilter(bool, d)
    itemfilter(bool, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}


def test_assoc_factory():
    from collections import OrderedDict
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_many():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_many_keys_branch():
    # When len(keys) >= len(d) * 0.6, the "remaining set" branch is used
    d = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert dissoc(d, 1, 2, 3, 4) == {5: 5}


def test_dissoc_few_keys_branch():
    d = {i: i for i in range(10)}
    assert dissoc(d, 0) == {i: i for i in range(1, 10)}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    from collections import OrderedDict
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', bad_kwarg=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_mixed():
    d = {'a': {'x': 1}}
    assert assoc_in(d, ['a', 'y', 'z'], 5) == {'a': {'x': 1, 'y': {'z': 5}}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x * 10)
    assert result == {'a': {'b': 10}}
    assert d == {'a': {'b': 1}}


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_creates_partial_nested():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_single_key_missing():
    assert update_in({}, ['a'], lambda x: x + 1, 0) == {'a': 1}


def test_update_in_factory():
    from collections import OrderedDict
    result = update_in(OrderedDict(), ['a'], len, "abc", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 3}


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


def test_get_in_nested_list_index(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_simple(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_wrong_index_type(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_missing_with_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_missing_key():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_type_error():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) is transaction
