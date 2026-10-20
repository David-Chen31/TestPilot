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


def test_merge_single_iterable():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping_not_unpacked():
    # A single Mapping should be treated as one dict, not iterated as a sequence
    assert merge({1: 2}) == {1: 2}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, foo=1)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_passes_list():
    seen = {}

    def capture(vals):
        seen['vals'] = vals
        return vals

    merge_with(capture, {1: 'a'}, {1: 'b'})
    assert seen['vals'] == ['a', 'b']


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 1}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, foo=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 2, 3: 4}) == {1: 3, 3: 5}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x * 10, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_swap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_factory():
    d = {1: 2, 2: 3}
    result = valfilter(lambda v: v > 2, d, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_keyfilter_factory():
    d = {1: 2, 2: 3}
    result = keyfilter(lambda k: k > 1, d, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


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
    assert itemfilter(lambda kv: True, {}) == {}


def test_itemfilter_factory():
    d = {1: 2, 2: 3}
    result = itemfilter(lambda kv: kv[0] == 2, d, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_add():
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


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'y', foo=1)


def test_dissoc_many_keys_uses_set_branch():
    # Trigger the branch where len(keys) >= len(d) * 0.6
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    assert dissoc(d, 1, 2, 3, 4) == {5: 'e'}


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    assert assoc_in(d, ['a', 'b'], 2) == {'a': {'b': 2}}


def test_assoc_in_new_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    result = assoc_in({'a': {}}, ['a', 'b'], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_inc():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    d = {'a': {'b': 1}}
    assert update_in(d, ['a', 'b'], lambda x: x * 2) == {'a': {'b': 2}}


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_branch_with_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 10)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=5,
                       factory=collections.OrderedDict)
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


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_index_default(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_missing_with_custom_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_bad_index_type(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    assert get_in([], coll) is coll
