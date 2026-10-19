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


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping():
    # a single mapping is not treated as an iterable of dicts
    assert merge({1: 2}) == {1: 2}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, badkwarg=True)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_values():
    received = {}

    def capture(vals):
        received['vals'] = list(vals)
        return vals[0]

    result = merge_with(capture, {1: 'a', 2: 'b'}, {2: 'c', 3: 'd'})
    assert result == {1: 'a', 2: 'b', 3: 'd'}
    assert received['vals'] == ['b', 'c']


def test_merge_with_single_iterable':
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, badkwarg=True)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 2, 3: 4}) == {1: 3, 3: 5}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: '2'}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x * 10, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(lambda x: x + 1, {1: 'a', 2: 'b'}) == {2: 'a', 3: 'b'}


def test_keymap_factory():
    result = keymap(str, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'1': 'a'}


def test_keymap_does_not_mutate():
    d = {1: 'a'}
    keymap(lambda x: x + 1, d)
    assert d == {1: 'a'}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {1: 'a', 2: 'b'}) == {
        'a': 1, 'b': 2}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 'a'}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 'a'}


def test_itemmap_does_not_mutate():
    d = {1: 'a'}
    itemmap(lambda kv: kv, d)
    assert d == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    d = {1: 2, 2: 3}
    result = valfilter(lambda x: x % 2 == 0, d,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    d = {1: 2, 2: 3}
    result = keyfilter(lambda x: x % 2 == 0, d,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    result = itemfilter(lambda kv: kv[0] % 2 == 0 and kv[1] < 4, d)
    assert result == {2: 3}


def test_itemfilter_factory():
    d = {1: 2, 2: 3}
    result = itemfilter(lambda kv: kv[0] == 2, d,
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

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
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y',
                    factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', badkwarg=True)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_top_level_new_key():
    assert assoc_in({}, ['a'], 1) == {'a': 1}


def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'credit card': '5555-1234-1234-1234',
        'name': 'Alice',
        'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']},
    }
    # original not mutated
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 1}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'credit card': '5555-1234-1234-1234',
        'name': 'Alice',
        'purchase': {'costs': 1.75, 'items': ['Apple', 'Orange']},
    }
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}}}


def test_update_in_creates_nested_inc():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a'], str, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'a': 'None'}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_bad_index_returns_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_bad_index_default():
    transaction = {'purchase': {'items': ['Apple', 'Orange']}}
    assert get_in(['purchase', 'items', 10], transaction, 'nope') == 'nope'


def test_get_in_type_error_returns_default():
    # indexing a dict with a bad key is a TypeError; get_in catches it
    transaction = {'a': 1}
    assert get_in(['a', 'b'], transaction) is None
    assert get_in(['a', 'b'], transaction, 'x') == 'x'


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([10], [1, 2], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 'b'], {'a': 1}, no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}


def test_get_in_nested_lists():
    coll = {'a': [{'b': 'found'}]}
    assert get_in(['a', 0, 'b'], coll) == 'found'
