import collections
from collections import OrderedDict

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
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_dict():
    d = {1: 2}
    assert merge(d) == d


def test_merge_list_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


def test_merge_does_not_modify_input():
    d1 = {1: 2}
    d2 = {2: 3}
    merge(d1, d2)
    assert d1 == {1: 2}
    assert d2 == {2: 3}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_values_grouped_in_order():
    seen = {}

    def capture(seq):
        return list(seq)

    result = merge_with(capture, {1: "a"}, {1: "b"}, {1: "c"})
    assert result == {1: ["a", "b", "c"]}


def test_merge_with_list_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=2)


def test_merge_with_empty():
    assert merge_with(sum) == {}


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


def test_valmap_empty():
    assert valmap(str, {}) == {}


def test_valmap_does_not_modify():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


def test_keymap():
    bills = {"Alice": [1], "Bob": [2]}
    assert keymap(str.lower, bills) == {"alice": [1], "bob": [2]}


def test_keymap_factory():
    result = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"A": 1}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[1], kv[0]), {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: "a"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_factory():
    result = valfilter(bool, {1: 1, 2: 0}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 1, {1: "a", 2: "b"}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: "b"}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda kv: kv[0] == kv[1], {1: 1, 2: 3},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1}


def test_itemfilter_empty():
    assert itemfilter(bool, {}) == {}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_new_key():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_modify():
    d = {"x": 1}
    assoc(d, "y", 2)
    assert d == {"x": 1}


def test_assoc_factory():
    result = assoc({"x": 1}, "y", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_does_not_modify():
    d = {"x": 1, "y": 2}
    dissoc(d, "y")
    assert d == {"x": 1, "y": 2}


def test_dissoc_no_keys():
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_dissoc_factory():
    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad_kwarg=1)


def test_dissoc_many_keys_branch():
    # Trigger the branch where many keys are removed (>= 60% of keys)
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    assert dissoc(d, "a", "b", "c", "d") == {"e": 5}


def test_dissoc_few_keys_branch():
    # Few keys -> the copy-and-delete branch
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    assert dissoc(d, "a") == {"b": 2, "c": 3, "d": 4, "e": 5}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing():
    purchase = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result["order"]["costs"] == [0.25, 1.00]
    assert result["order"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    # original untouched
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_new_nested():
    assert assoc_in({}, ["a", "b"], 1) == {"a": {"b": 1}}


def test_assoc_in_does_not_modify():
    d = {"a": {"b": 1}}
    assoc_in(d, ["a", "b"], 2)
    assert d == {"a": {"b": 1}}


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_nested():
    transaction = {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    }
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result["purchase"]["costs"] == 1.75
    assert result["purchase"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_missing_key_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_update_in_missing_key_partial():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_factory():
    result = update_in({"a": 1}, ["a"], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 2}


def test_update_in_does_not_modify():
    d = {"a": {"b": 1}}
    update_in(d, ["a", "b"], lambda x: x + 1)
    assert d == {"a": {"b": 1}}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

@pytest.fixture
def transaction():
    return {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }


def test_get_in_nested_item(transaction):
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_simple(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_returns_default(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_bad_type_index(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(["a", "b"], {"a": 1}, no_default=True)


def test_get_in_empty_keys_returns_coll(transaction):
    assert get_in([], transaction) is transaction


def test_get_in_works_with_lists():
    data = [[1, 2], [3, 4]]
    assert get_in([1, 0], data) == 3


# ---------------------------------------------------------------------------
# _get_factory error branch via merge_with/dissoc covered above; test message
# ---------------------------------------------------------------------------

def test_merge_unexpected_kwarg_message():
    with pytest.raises(TypeError) as excinfo:
        merge({}, foo=1)
    assert "foo" in str(excinfo.value)


def test_dissoc_unexpected_kwarg_message():
    with pytest.raises(TypeError) as excinfo:
        dissoc({}, bar=1)
    assert "bar" in str(excinfo.value)
