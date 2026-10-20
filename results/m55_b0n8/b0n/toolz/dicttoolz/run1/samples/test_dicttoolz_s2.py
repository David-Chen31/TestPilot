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
    assert merge({1: 2}) == {1: 2}


def test_merge_sequence_of_dicts():
    # Single non-Mapping arg is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_generator_of_dicts():
    assert merge({k: v} for k, v in [(1, 2), (3, 4)]) == {1: 2, 3: 4}


def test_merge_factory():
    from collections import OrderedDict
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bogus=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_custom_func():
    result = merge_with(lambda vals: sorted(vals), {1: 3, 2: 1}, {1: 1, 3: 2})
    assert result == {1: [1, 3], 2: [1], 3: [2]}


def test_merge_with_values_are_lists():
    seen = {}

    def capture(vals):
        seen.update(vals if isinstance(vals, dict) else {})
        return list(vals)

    result = merge_with(capture, {1: 'a'}, {1: 'b'})
    assert result == {1: ['a', 'b']}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_sequence():
    assert merge_with(sum, [{1: 2}, {1: 3}]) == {1: 5}


def test_merge_with_factory():
    from collections import OrderedDict
    result = merge_with(sum, {1: 2}, {1: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 5}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bogus=1)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 2, 3: 4}) == {1: 3, 3: 5}


def test_valmap_sum():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    from collections import OrderedDict
    result = valmap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x * 10, d)
    assert d == {1: 2}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(lambda k: k + 1, {1: 2, 3: 4}) == {2: 2, 4: 4}


def test_keymap_lower():
    bills = {"Alice": [20, 15], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(lambda k: k, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict
    result = keymap(lambda k: k, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_reverse():
    assert itemmap(lambda kv: (kv[1], kv[0]), {1: 2, 3: 4}) == {2: 1, 4: 3}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_all_removed():
    assert valfilter(lambda v: False, {1: 2}) == {}


def test_valfilter_empty():
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_factory():
    from collections import OrderedDict
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter():
    iseven = lambda k: k % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict
    result = keyfilter(lambda k: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


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
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'y', 2)
    assert d == {'x': 1}
    assert result == {'x': 1, 'y': 2}
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


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}


def test_dissoc_many_keys_branch():
    # When len(keys) >= len(d) * .6, the alternative branch is used
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    assert dissoc(d, 'a', 'b', 'c', 'd') == {'e': 5}


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
        dissoc({'x': 1}, 'x', bogus=1)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}


def test_assoc_in_new_nested():
    result = assoc_in({}, ['a', 'b', 'c'], 5)
    assert result == {'a': {'b': {'c': 5}}}


def test_assoc_in_top_level():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}, 'c': 2}
    assoc_in(d, ['a', 'b'], 99)
    assert d == {'a': {'b': 1}, 'c': 2}


def test_assoc_in_partial_nested():
    d = {'a': {'x': 1}}
    result = assoc_in(d, ['a', 'b', 'c'], 2)
    assert result == {'a': {'x': 1, 'b': {'c': 2}}}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    d = {'purchase': {'costs': [0.5, 1.25]}}
    result = update_in(d, ['purchase', 'costs'], sum)
    assert result == {'purchase': {'costs': 1.75}}
    assert d == {'purchase': {'costs': [0.5, 1.25]}}


def test_update_in_missing_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_nested_default():
    inc = lambda x: x + 1
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_identity_function():
    d = {'a': 1}
    result = update_in(d, ['a'], lambda x: x)
    assert result == {'a': 1}
    assert result is not d


def test_update_in_factory():
    from collections import OrderedDict
    result = update_in({}, ['a', 'b'], str, default="x", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)
    assert result == {'a': {'b': 'x'}}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

transaction = {
    'name': 'Alice',
    'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    'credit card': '5555-1234-1234-1234',
}


def test_get_in_nested_list():
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_simple():
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_returns_none():
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_missing_list_index():
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_wrong_type_in_list():
    assert get_in(['purchase', 'items', 'apple'], transaction) is None


def test_get_in_custom_default():
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_key_error():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_index_error():
    with pytest.raises(IndexError):
        get_in(['purchase', 'items', 10], transaction, no_default=True)


def test_get_in_no_default_raises_type_error():
    with pytest.raises(TypeError):
        get_in(['purchase', 'items', 'apple'], transaction, no_default=True)


def test_get_in_empty_keys_returns_coll():
    d = {'a': 1}
    assert get_in([], d) == {'a': 1}


def test_get_in_top_level():
    assert get_in(['a'], {'a': 42}) == 42
