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


def test_merge_later_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_iterable_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts.
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_iterable_tuple():
    assert merge(({1: 2}, {2: 3})) == {1: 2, 2: 3}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    result = merge(a, b)
    assert result == {1: 2, 3: 4}
    assert a == {1: 2}
    assert b == {3: 4}
    assert result is not a


def test_merge_factory():
    from collections import OrderedDict

    result = merge({3: 4}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == [3, 1]


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}, {1: 3}]) == {1: 6}


def test_merge_with_passes_list_of_values():
    seen = {}

    def collect(vals):
        seen["last"] = list(vals)
        return tuple(vals)

    result = merge_with(collect, {1: "a"}, {1: "b"}, {2: "c"})
    assert seen["last"] == ["c"]
    assert result == {1: ("a", "b"), 2: ("c",)}


def test_merge_with_factory():
    from collections import OrderedDict

    result = merge_with(sum, {2: 2}, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == [2, 1]


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=True)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_does_not_mutate():
    d = {1: 2}
    result = valmap(lambda x: x + 1, d)
    assert result == {1: 3}
    assert d == {1: 2}


def test_valmap_factory():
    from collections import OrderedDict

    result = valmap(lambda x: x, {2: 2, 1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict

    result = keymap(lambda k: k, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(reversed, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict

    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    from collections import OrderedDict

    result = valfilter(lambda v: True, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict

    result = keyfilter(lambda k: True, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


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

    result = itemfilter(lambda item: True, {1: 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_existing_key():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_new_key():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_mutate():
    d = {"x": 1}
    result = assoc(d, "y", 2)
    assert result == {"x": 1, "y": 2}
    assert d == {"x": 1}


def test_assoc_factory():
    from collections import OrderedDict

    result = assoc({"a": 1}, "b", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    assert dissoc({"x": 1}) == {"x": 1}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    result = dissoc(d, "y")
    assert result == {"x": 1}
    assert d == {"x": 1, "y": 2}


def test_dissoc_factory():
    from collections import OrderedDict

    # Use enough keys to hit the "remaining" branch.
    result = dissoc({"a": 1, "b": 2, "c": 3}, "a", "b", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"c": 3}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad=True)


def test_dissoc_both_branches():
    # Few keys -> update branch
    d = {i: i for i in range(10)}
    assert dissoc(d, 0, 1) == {i: i for i in range(2, 10)}
    # Many keys -> remaining branch
    assert dissoc(d, *range(8)) == {8: 8, 9: 9}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_nested():
    transaction = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result["purchase"]["costs"] == 1.75
    assert result["purchase"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    # does not mutate original
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_create_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_update_in_create_nested_with_existing():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }


def test_update_in_default_used_when_missing():
    assert update_in({}, ["a"], lambda x: x + 1, default=10) == {"a": 11}


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}
    result = update_in(d, ["a", "b"], lambda x: x + 1)
    assert result == {"a": {"b": 2}}
    assert d == {"a": {"b": 1}}


def test_update_in_factory():
    from collections import OrderedDict

    result = update_in({"a": 1}, ["a"], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_assoc_in():
    purchase = {
        "name": "Alice",
        "order": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result["order"]["costs"] == [0.25, 1.00]
    assert result["order"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    # original untouched
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ["a", "b"], 5) == {"a": {"b": 5}}


def test_assoc_in_factory():
    from collections import OrderedDict

    result = assoc_in({}, ["a"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

TRANSACTION = {
    "name": "Alice",
    "purchase": {
        "items": ["Apple", "Orange"],
        "costs": [0.50, 1.25],
    },
    "credit card": "5555-1234-1234-1234",
}


def test_get_in_nested():
    assert get_in(["purchase", "items", 0], TRANSACTION) == "Apple"


def test_get_in_simple():
    assert get_in(["name"], TRANSACTION) == "Alice"


def test_get_in_missing_key_returns_none():
    assert get_in(["purchase", "total"], TRANSACTION) is None


def test_get_in_bad_index_returns_none():
    assert get_in(["purchase", "items", 10], TRANSACTION) is None


def test_get_in_bad_type_returns_none():
    assert get_in(["purchase", "items", "apple"], TRANSACTION) is None


def test_get_in_default_value():
    assert get_in(["purchase", "total"], TRANSACTION, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_typeerror():
    with pytest.raises(TypeError):
        get_in(["a"], 5, no_default=True)


def test_get_in_empty_keys():
    assert get_in([], TRANSACTION) is TRANSACTION
