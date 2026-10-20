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


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 'one'}) == {1: 'one'}


def test_merge_iterable_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 'one'}, {2: 'two'}]) == {1: 'one', 2: 'two'}


def test_merge_generator_of_dicts():
    gens = ({i: i} for i in range(3))
    assert merge(gens) == {0: 0, 1: 1, 2: 2}


def test_merge_factory():
    from collections import OrderedDict
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_does_not_mutate_input():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, wrong=1)


def test_merge_returns_new_dict():
    a = {1: 2}
    result = merge(a)
    assert result == a
    assert result is not a


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_no_overlap():
    result = merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 22, 3: 30}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_dict():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable():
    assert merge_with(sum, [{1: 1}, {1: 10}]) == {1: 11}


def test_merge_with_receives_list():
    captured = {}

    def capture(vals):
        return list(vals)

    result = merge_with(capture, {1: 'a'}, {1: 'b'}, {1: 'c'})
    assert result == {1: ['a', 'b', 'c']}


def test_merge_with_factory():
    from collections import OrderedDict
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, wrong=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_does_not_mutate():
    d = {"a": 1}
    valmap(lambda x: x + 1, d)
    assert d == {"a": 1}


def test_valmap_factory():
    from collections import OrderedDict
    result = valmap(lambda x: x * 2, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_does_not_mutate():
    d = {"A": 1}
    keymap(str.lower, d)
    assert d == {"A": 1}


def test_keymap_factory():
    from collections import OrderedDict
    result = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"A": 1}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_does_not_mutate():
    d = {"a": 1}
    itemmap(lambda kv: (kv[0], kv[1] + 1), d)
    assert d == {"a": 1}


def test_itemmap_factory():
    from collections import OrderedDict
    result = itemmap(lambda kv: kv, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_none_match():
    d = {1: 1, 2: 3}
    assert valfilter(lambda v: v % 2 == 0, d) == {}


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

def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k == 1, d)
    assert d == {1: 2, 2: 3}


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


def test_itemfilter_empty():
    assert itemfilter(lambda kv: True, {}) == {}


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda kv: kv[0] == 1, d)
    assert d == {1: 2, 2: 3}


def test_itemfilter_factory():
    from collections import OrderedDict
    result = itemfilter(lambda kv: kv[1] > 1, {1: 1, 2: 2},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


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
    assert result is not d


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


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


def test_dissoc_empty_dict():
    assert dissoc({}, 'a') == {}


def test_dissoc_branch_many_keys():
    # triggers the "else" branch (len(keys) >= len(d) * .6)
    d = {1: 1, 2: 2, 3: 3}
    assert dissoc(d, 1, 2) == {3: 3}


def test_dissoc_branch_few_keys():
    # triggers the "if" branch (len(keys) < len(d) * .6)
    d = {i: i for i in range(10)}
    assert dissoc(d, 0) == {i: i for i in range(1, 10)}


def test_dissoc_factory():
    from collections import OrderedDict
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', wrong=1)


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
    assert result['order']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}


def test_assoc_in_single_key():
    assert assoc_in({}, ['a'], 1) == {'a': 1}


def test_assoc_in_overwrites():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


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
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    # original untouched
    assert transaction['purchase']['costs'] == [0.50, 1.25]


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_existing_top_key():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}
    }


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}
    assert result == {'a': {'b': 2}}


def test_update_in_default_none():
    result = update_in({}, ['a'], lambda x: x)
    assert result == {'a': None}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

@pytest.fixture
def transaction():
    return {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }


def test_get_in_nested(transaction):
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_top_level(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_bad_index_returns_none(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


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


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction


def test_get_in_empty_keys_no_default(transaction):
    assert get_in([], transaction, no_default=True) == transaction
