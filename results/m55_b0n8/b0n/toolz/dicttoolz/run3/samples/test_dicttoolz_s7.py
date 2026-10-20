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

def test_merge_variadic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_sequence_of_dicts():
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 1}, {2: 2})) == {1: 1, 2: 2}


def test_merge_no_args():
    assert merge() == {}


def test_merge_with_factory():
    rv = merge({1: 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 1}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 1}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    assert merge_with(lambda x: x[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_single_sequence():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    rv = merge_with(sum, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 1}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=1)


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_factory():
    rv = valmap(lambda x: x + 1, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'a': 2}


def test_valmap_does_not_mutate():
    d = {'a': 1}
    valmap(lambda x: x + 1, d)
    assert d == {'a': 1}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_factory():
    rv = keymap(str.upper, {'a': 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'A': 1}


def test_itemmap():
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    rv = itemmap(lambda kv: (kv[1], kv[0]), {'a': 1},
                 factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    rv = valfilter(lambda x: x > 0, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 1}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    rv = keyfilter(lambda k: k > 0, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {1: 1}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    rv = itemfilter(lambda item: item[0] == 'a', {'a': 1},
                    factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'a': 1}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    keyfilter(lambda k: k > 1, d)
    itemfilter(lambda item: item[0] > 1, d)
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
    rv = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert rv is not d


def test_assoc_factory():
    rv = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_many():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_keys():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'x')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_else_branch():
    # Many keys triggers the set-based branch (len(keys) >= len(d) * 0.6)
    d = {'a': 1, 'b': 2, 'c': 3}
    assert dissoc(d, 'a', 'c') == {'b': 2}


def test_dissoc_factory():
    rv = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad_kwarg=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_path():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_factory():
    rv = assoc_in({}, ['a'], 1, factory=collections.OrderedDict)
    assert isinstance(rv, collections.OrderedDict)
    assert rv == {'a': 1}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_path_creates_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_path_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    rv = update_in({}, ['a'], lambda x: x + 1, default=0,
                   factory=collections.OrderedDict)
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


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_index(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_raises_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(['purchase', 'items', 'apple'], transaction, no_default=True)
