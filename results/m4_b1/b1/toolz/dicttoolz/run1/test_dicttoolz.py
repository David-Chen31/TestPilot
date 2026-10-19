import collections
from collections import OrderedDict, defaultdict

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
from toolz.dicttoolz import _get_factory


# ---------------------------------------------------------------------------
# _get_factory
# ---------------------------------------------------------------------------
def test_get_factory_default():
    kwargs = {}
    assert _get_factory(merge, kwargs) is dict
    assert kwargs == {}


def test_get_factory_custom():
    kwargs = {"factory": OrderedDict}
    assert _get_factory(merge, kwargs) is OrderedDict
    assert kwargs == {}


def test_get_factory_unexpected_kwarg():
    kwargs = {"nope": 1}
    with pytest.raises(TypeError) as excinfo:
        _get_factory(merge, kwargs)
    assert "'nope'" in str(excinfo.value)


def test_get_factory_unexpected_kwarg_name_in_message():
    kwargs = {"factory": dict, "bad": 1}
    with pytest.raises(TypeError) as excinfo:
        _get_factory(merge, kwargs)
    assert "merge()" in str(excinfo.value)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def test_merge_basic():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_generator():
    assert merge({i: i for i in range(2)} for _ in range(1)) == {0: 0, 1: 1}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_kwarg_error():
    with pytest.raises(TypeError):
        merge({}, bad=1)


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    result = merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}
    assert result is not a and result is not b


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------
def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first_picks_first():
    first = lambda seq: seq[0]
    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_values_as_list():
    captured = []

    def collect(vals):
        captured.append(list(vals))
        return len(vals)

    result = merge_with(collect, {1: "a"}, {1: "b"}, {1: "c"})
    assert result == {1: 3}
    assert captured == [["a", "b", "c"]]


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_iterable():
    result = merge_with(sum, [{1: 1}, {1: 2}])
    assert result == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_kwarg_error():
    with pytest.raises(TypeError):
        merge_with(sum, {}, bad=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------
def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    res = valmap(lambda x: x * 10, d)
    assert d == {1: 2}
    assert res == {1: 20}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_factory():
    result = keymap(lambda x: x + 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0] + 1, kv[1]), {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------
def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k == 1, {1: 2, 2: 3}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v > 2, d)
    keyfilter(lambda k: k > 1, d)
    itemfilter(lambda kv: kv[0] == 1, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc / dissoc
# ---------------------------------------------------------------------------
def test_assoc_update_existing():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_add_new():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_mutate():
    d = {"x": 1}
    result = assoc(d, "x", 2)
    assert d == {"x": 1}
    assert result == {"x": 2}


def test_assoc_factory():
    result = assoc({"x": 1}, "y", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1, "y": 2}


def test_dissoc_one():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    assert dissoc({"x": 1}) == {"x": 1}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    result = dissoc(d, "y")
    assert d == {"x": 1, "y": 2}
    assert result == {"x": 1}


def test_dissoc_factory():
    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_kwarg_error():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "y", bad=1)


def test_dissoc_large_keys_branch():
    # many keys -> uses set difference branch
    d = {i: i for i in range(10)}
    keys = tuple(range(8))
    result = dissoc(d, *keys)
    assert result == {8: 8, 9: 9}


def test_dissoc_few_keys_branch():
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1)
    expected = {i: i for i in range(2, 10)}
    assert result == expected


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------
def test_assoc_in_existing():
    purchase = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result["order"]["costs"] == [0.25, 1.00]
    assert result["order"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    # original untouched
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_new_nested():
    result = assoc_in({}, ["a", "b", "c"], 1)
    assert result == {"a": {"b": {"c": 1}}}


def test_assoc_in_factory():
    result = assoc_in({}, ["a"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], int)


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_nested():
    transaction = {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result["purchase"]["costs"] == 1.75
    assert result["purchase"]["items"] == ["Apple", "Orange"]
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_missing_keys_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_update_in_missing_keys_with_default_inc():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}
    result = update_in(d, ["a", "b"], lambda x: x + 1)
    assert d == {"a": {"b": 1}}
    assert result == {"a": {"b": 2}}


def test_update_in_factory():
    result = update_in({}, ["a"], lambda x: 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------
transaction = {
    "name": "Alice",
    "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    "credit card": "5555-1234-1234-1234",
}


def test_get_in_list_index():
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_top_level():
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_key_returns_none():
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_bad_index_returns_none():
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_out_of_range_returns_none():
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default():
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises_key_error():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_raises_index_error():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {"a": 1}) == {"a": 1}


def test_get_in_nested_lists():
    coll = [[1, 2], [3, 4]]
    assert get_in([1, 0], coll) == 3
