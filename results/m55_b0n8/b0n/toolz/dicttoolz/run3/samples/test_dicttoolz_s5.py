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


def test_merge_no_args():
    assert merge() == {}


def test_merge_single_mapping_argument():
    # A single Mapping is not treated as a collection of dicts.
    assert merge({1: 2, 3: 4}) == {1: 2, 3: 4}


def test_merge_single_collection_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4}, {1: 5})) == {1: 5, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=list)
    assert result == [1, 3]


def test_merge_unexpected_kwarg_raises():
    with pytest.raises(TypeError):
        merge({1: 2}, bad=3)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first_semantics():
    assert merge_with(lambda vs: vs[0],
                      {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_single_mapping():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_single_collection():
    assert merge_with(sum, [{1: 1}, {1: 10}]) == {1: 11}


def test_merge_with_passes_list():
    seen = []

    def capture(vals):
        seen.append(list(vals))
        return vals

    merge_with(capture, {1: 'a'}, {1: 'b'})
    assert seen == [['a', 'b']]


def test_merge_with_factory():
    assert merge_with(sum, {1: 1}, {1: 2}, factory=list) == [1]


def test_merge_with_unexpected_kwarg_raises():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=3)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = valmap(lambda x: x * 10, d)
    assert result == {1: 20, 3: 40}
    assert d == {1: 2, 3: 4}


def test_valmap_factory():
    assert valmap(lambda x: x * 2, {1: 2, 3: 4}, factory=list) == [1, 3]


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30],
                                        "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_does_not_mutate():
    d = {'a': 1, 'b': 2}
    result = keymap(str.upper, d)
    assert result == {'A': 1, 'B': 2}
    assert d == {'a': 1, 'b': 2}


def test_keymap_factory():
    result = keymap(str.upper, {'a': 1, 'b': 2}, factory=list)
    assert result == ['A', 'B']


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_basic():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda item: (item[1], item[0]), accountids) == {
        10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda item: item, {}) == {}


def test_itemmap_does_not_mutate():
    d = {1: 2, 3: 4}
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), d)
    assert result == {2: 2, 4: 4}
    assert d == {1: 2, 3: 4}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2, 3: 4}, factory=list)
    assert result == [(1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    assert valfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_none_match():
    assert valfilter(lambda v: v > 100, {1: 2, 3: 4}) == {}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v % 2 == 0, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    result = valfilter(lambda v: v % 2 == 0, {1: 2, 2: 3}, factory=list)
    assert result == [1]


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    assert keyfilter(iseven, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: k > 100, {1: 2}) == {}


def test_keyfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    keyfilter(lambda k: k == 1, d)
    assert d == {1: 2, 2: 3}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 2, {1: 2, 2: 3}, factory=list)
    assert result == [2]


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, {1: 2, 2: 3, 3: 4, 4: 5}) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_none_match():
    assert itemfilter(lambda item: item[0] > 100, {1: 2}) == {}


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda kv: kv[0] == 1, d)
    assert d == {1: 2, 2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == 2, {1: 2, 2: 3}, factory=list)
    assert result == [(2, 3)]


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 3)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 3}


def test_assoc_empty():
    assert assoc({}, 'a', 1) == {'a': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=list)
    assert result == ['x', 'y']


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
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'a': 1, 'b': 2}, 'b', factory=list)
    assert sorted(result) == ['a']


def test_dissoc_unexpected_kwarg_raises():
    with pytest.raises(TypeError):
        dissoc({'a': 1}, 'a', bad=3)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {'credit card': '5555-1234-1234-1234',
                      'name': 'Alice',
                      'order': {'costs': [0.25, 1.00],
                                'items': ['Apple', 'Orange']}}


def test_assoc_in_does_not_mutate():
    purchase = {'order': {'costs': [0.50]}}
    assoc_in(purchase, ['order', 'costs'], [0.25])
    assert purchase == {'order': {'costs': [0.50]}}


def test_assoc_in_creates_nested():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_top_level():
    assert assoc_in({'a': 1}, ['b'], 2) == {'a': 1, 'b': 2}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {'credit card': '5555-1234-1234-1234',
                      'name': 'Alice',
                      'purchase': {'costs': 1.75,
                                   'items': ['Apple', 'Orange']}}


def test_update_in_does_not_mutate():
    transaction = {'purchase': {'costs': [0.50, 1.25]}}
    update_in(transaction, ['purchase', 'costs'], sum)
    assert transaction == {'purchase': {'costs': [0.50, 1.25]}}


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_keeps_existing_siblings():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_update_in_factory():
    result = update_in({'a': {'b': 1}}, ['a', 'b'], lambda x: x + 1,
                       factory=dict)
    assert result == {'a': {'b': 2}}


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
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_top(transaction):
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default_none(transaction):
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_string_index(transaction):
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_explicit_default(transaction):
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_raises_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(['name', 'x'], transaction, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction
