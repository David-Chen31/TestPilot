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
    d = {1: 2}
    assert merge(d) == {1: 2}
    assert merge(d) is not d


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_iterable_generator():
    def gen():
        yield {1: 2}
        yield {2: 3}

    assert merge(gen()) == {1: 2, 2: 3}


def test_merge_factory():
    from collections import OrderedDict

    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, foo=1)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {2: 3}
    merge(a, b)
    assert a == {1: 2}
    assert b == {2: 3}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    result = merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 2, 3: 30}


def test_merge_with_no_args():
    assert merge_with(list) == {}


def test_merge_with_single_iterable_of_dicts():
    result = merge_with(sum, [{1: 1}, {1: 2}, {1: 3}])
    assert result == {1: 6}


def test_merge_with_passes_list():
    captured = {}

    def func(vals):
        captured['vals'] = list(vals)
        return len(vals)

    result = merge_with(func, {1: 'a'}, {1: 'b'})
    assert result == {1: 2}
    assert captured['vals'] == ['a', 'b']


def test_merge_with_factory():
    from collections import OrderedDict

    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bar=2)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = valmap(lambda x: x * 10, d)
    assert d == {1: 2, 3: 4}
    assert result == {1: 20, 3: 40}


def test_valmap_factory():
    from collections import OrderedDict

    result = valmap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict

    result = keymap(lambda x: x + 1, {1: 'a'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 'a'}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_basic():
    accountids = {"Alice": 10, "Bob": 20}
    result = itemmap(lambda item: (item[1], item[0]), accountids)
    assert result == {10: "Alice", 20: "Bob"}


def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    result = itemmap(reversed, accountids)
    assert result == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict

    result = itemmap(lambda item: (item[1], item[0]), {'a': 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter_basic():
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
    from collections import OrderedDict

    result = valfilter(lambda v: v > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


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
    from collections import OrderedDict

    result = keyfilter(lambda k: k == 1, {1: 'a', 2: 'b'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'a'}


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
    assert itemfilter(lambda x: True, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict

    result = itemfilter(lambda item: item[0] == 1, {1: 'a', 2: 'b'},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result == {'x': 2}
    assert result is not d


def test_assoc_factory():
    from collections import OrderedDict

    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2, 'z': 3}
    result = dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2, 'z': 3}
    assert result == {'x': 1, 'z': 3}


def test_dissoc_many_keys_branch():
    # Many keys relative to dict size triggers the set-difference branch.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(7))
    assert result == {7: 7, 8: 8, 9: 9}


def test_dissoc_factory():
    from collections import OrderedDict

    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', foo=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_new_path():
    assert assoc_in({}, ['a', 'b'], 5) == {'a': {'b': 5}}


def test_assoc_in_single_key():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_does_not_mutate_nested():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_assoc_in_factory():
    from collections import OrderedDict

    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], inc)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_update_in_missing_with_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_single_key_missing():
    inc = lambda x: x + 1
    assert update_in({}, ['a'], inc, default=10) == {'a': 11}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_factory():
    from collections import OrderedDict

    result = update_in({}, ['a', 'b'], lambda x: x, default=1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 1}}


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


def test_get_in_missing_key_returns_default(transaction):
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_bad_index(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_type_error(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(['a'], 5, no_default=True)


def test_get_in_empty_keys_returns_coll(transaction):
    assert get_in([], transaction) is transaction


def test_get_in_default_with_no_default_false(transaction):
    # no_default=False is explicit equivalent to default behaviour
    assert get_in(['missing'], transaction, default='x') == 'x'
