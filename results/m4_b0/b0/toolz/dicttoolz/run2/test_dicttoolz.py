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


def test_merge_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_factory():
    rv = merge({1: 2}, {3: 4}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 2, 3: 4}


def test_merge_does_not_mutate():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------
def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_dict_values_are_lists():
    assert merge_with(list, {1: 2}, {1: 3}) == {1: [2, 3]}


def test_merge_with_defaultdict_handling():
    # a key present only in one dict still gets a list of one element
    assert merge_with(tuple, {1: 'a'}, {2: 'b'}) == {1: ('a',), 2: ('b',)}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    rv = merge_with(sum, {1: 1, 2: 2}, {1: 10},
                    factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 11, 2: 2}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------
def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x * 10, d)
    assert d == {1: 2}


def test_valmap_factory():
    rv = valmap(lambda x: x + 1, {1: 1, 2: 2},
                factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 2, 2: 3}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30],
                                        "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    rv = keymap(lambda k: k + 1, {1: 'a'},
                factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {2: 'a'}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    rv = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 'a'},
                 factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {2: 'a'}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------
def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v == 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    rv = valfilter(lambda v: v > 1, {1: 1, 2: 2},
                   factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {2: 2}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_factory():
    rv = keyfilter(lambda k: k > 1, {1: 'a', 2: 'b'},
                   factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {2: 'b'}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    rv = itemfilter(lambda item: item[0] == 2, {1: 'a', 2: 'b'},
                    factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {2: 'b'}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------
def test_assoc_update_existing():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assoc(d, 'y', 2)
    assert d == {'x': 1}


def test_assoc_factory():
    rv = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------
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


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_factory():
    rv = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'x': 1}


def test_dissoc_many_keys_path():
    # Removing most keys exercises the "else" branch
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    assert dissoc(d, 1, 2, 3, 4) == {5: 'e'}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', bad=1)


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------
def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_update_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    # items unchanged
    assert result['order']['items'] == ['Apple', 'Orange']
    # original not mutated
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_keys_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_partial_missing():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    inc = lambda x: x + 1
    rv = update_in({'a': 0}, ['a'], inc, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'a': 1}


def test_assoc_in_factory():
    rv = assoc_in({}, ['a'], 1, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'a': 1}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------
@pytest.fixture
def transaction():
    return {'name': 'Alice',
            'purchase': {'items': ['Apple', 'Orange'],
                         'costs': [0.50, 1.25]},
            'credit card': '5555-1234-1234-1234'}


def test_get_in_basic(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_missing_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_bad_key_type(transaction):
    # IndexError/TypeError fall back to default
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_raises_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(['name', 0], transaction, no_default=True)
