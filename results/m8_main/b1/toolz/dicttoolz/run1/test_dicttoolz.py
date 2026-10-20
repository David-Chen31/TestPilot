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


def test_merge_no_args():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 2}, {2: 3})) == {1: 2, 2: 3}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    result = merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}
    assert result is not a
    assert result is not b


def test_merge_factory():
    result = merge({1: 2}, {2: 3}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2, 2: 3}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


def test_merge_custom_mapping_subclass():
    class MyMapping(collections.abc.Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, k):
            return self._data[k]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    result = merge(MyMapping({1: 2}), {3: 4})
    assert result == {1: 2, 3: 4}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_values():
    seen = {}

    def f(values):
        seen['values'] = list(values)
        return values

    result = merge_with(f, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: ['a', 'b', 'c']}
    assert seen['values'] == ['a', 'b', 'c']


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=True)


def test_merge_with_does_not_mutate():
    a = {1: 1}
    b = {1: 2}
    merge_with(sum, a, b)
    assert a == {1: 1}
    assert b == {1: 2}


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30],
        "bob": [10, 35],
    }


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(lambda x: x + 1, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


def test_keymap_does_not_mutate():
    d = {1: 2}
    keymap(lambda x: x + 1, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_basic():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 2},
                     factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {2: 2}


def test_itemmap_does_not_mutate():
    d = {1: 2}
    itemmap(reversed, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: v == 2, {1: 2},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v == 2, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 1, {1: 2, 2: 3},
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k == 1, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: item[0] == 1, {1: 2, 2: 3},
                        factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda item: item[0] == 1, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_update():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result is not d


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


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_empty_dict():
    assert dissoc({}, 'x') == {}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'x')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_many_keys_branch():
    # Force the "else" branch by removing a large fraction of keys.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(9))
    assert result == {9: 9}


def test_dissoc_few_keys_branch():
    d = {i: i for i in range(10)}
    result = dissoc(d, 0)
    assert result == {i: i for i in range(1, 10)}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=True)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_basic():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'credit card': '5555-1234-1234-1234',
        'name': 'Alice',
        'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']},
    }
    # original unchanged
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_single_key():
    assert assoc_in({}, ['a'], 1) == {'a': 1}


def test_assoc_in_factory():
    result = assoc_in({'a': {}}, ['a', 'b'], 1,
                      factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25],
        },
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'credit card': '5555-1234-1234-1234',
        'name': 'Alice',
        'purchase': {'costs': 1.75, 'items': ['Apple', 'Orange']},
    }
    # original unchanged
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_missing_creates_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: 'bar'}},
    }


def test_update_in_missing_inc_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}},
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_update_in_factory():
    result = update_in({'a': {'b': 1}}, ['a', 'b'], lambda x: x + 1,
                       factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result['a'], collections.OrderedDict)
    assert result == {'a': {'b': 2}}


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


def test_get_in_list_index(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_index_into_list(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_custom_default(transaction):
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


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction


def test_get_in_index_into_string():
    # Indexing a string by 0 returns its first character; this does not
    # raise, so the default is not used.
    assert get_in([0], "not a list of dicts", default='fallback') == 'n'


def test_get_in_typeerror_uses_default():
    # A non-integer key index into a string raises TypeError, which uses
    # the supplied default.
    assert get_in(['x'], "not a list of dicts", default='fallback') == 'fallback'
