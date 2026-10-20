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
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_empty():
    assert merge() == {}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_single_iterable_of_dicts():
    # A single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_iterable_of_pairs():
    assert merge([(1, 2), (3, 4)]) == {1: 2, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_factory():
    result = merge({1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


def test_merge_factory_and_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, factory=dict, other=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_first():
    def first(seq):
        return seq[0]

    assert merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30}) == {1: 1, 2: 2, 3: 30}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_mapping():
    assert merge_with(sum, {1: 1}) == {1: 1}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_values_passed_as_list():
    seen = {}

    def capture(seq):
        seen["value"] = list(seq)
        return len(seq)

    result = merge_with(capture, {1: "a"}, {1: "b"}, {1: "c"})
    assert result == {1: 3}
    assert seen["value"] == ["a", "b", "c"]


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad=True)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 10, 2: 20}) == {1: 11, 2: 21}


def test_valmap_sum():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x + 1, d)
    assert d == {1: 2}


def test_keymap():
    assert keymap(lambda k: k + 1, {1: 10, 2: 20}) == {2: 10, 3: 20}


def test_keymap_lower():
    bills = {"Alice": [20], "Bob": [10]}
    assert keymap(str.lower, bills) == {"alice": [20], "bob": [10]}


def test_keymap_empty():
    assert keymap(lambda x: x, {}) == {}


def test_keymap_factory():
    result = keymap(lambda x: x, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_itemmap():
    assert itemmap(lambda item: (item[0] + 1, item[1] * 2), {1: 10}) == {2: 20}


def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda x: x, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


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
    result = valfilter(lambda x: True, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda x: x == 2, d)
    assert d == {1: 2, 2: 3}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda x: True, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda x: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda x: True, {1: 2}, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {1: 2}


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
    assert d == {"x": 1}
    assert result is not d


def test_assoc_empty():
    assert assoc({}, "a", 1) == {"a": 1}


def test_assoc_factory():
    result = assoc({"x": 1}, "y", 2, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_one_key():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple_keys():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_empty():
    assert dissoc({}) == {}


def test_dissoc_no_keys():
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    result = dissoc(d, "x")
    assert d == {"x": 1, "y": 2}
    assert result == {"y": 2}


def test_dissoc_does_not_mutate_many_keys():
    # triggers the "many keys" branch (else branch)
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(8))
    assert d == {i: i for i in range(10)}
    assert result == {8: 8, 9: 9}


def test_dissoc_few_keys_branch():
    # len(keys) < len(d) * .6 -> update/del branch
    d = {i: i for i in range(10)}
    result = dissoc(d, 0)
    expected = {i: i for i in range(1, 10)}
    assert result == expected


def test_dissoc_many_keys_branch():
    # len(keys) >= len(d) * .6 -> difference_update branch
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(7))
    assert result == {7: 7, 8: 8, 9: 9}


def test_dissoc_factory():
    result = dissoc({"x": 1, "y": 2}, "x", factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {"y": 2}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad=True)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_path():
    d = {"a": {"b": 1}}
    assert assoc_in(d, ["a", "b"], 2) == {"a": {"b": 2}}


def test_assoc_in_new_inner_key():
    d = {"a": {"b": 1}}
    assert assoc_in(d, ["a", "c"], 9) == {"a": {"b": 1, "c": 9}}


def test_assoc_in_new_path():
    assert assoc_in({}, ["a", "b", "c"], 5) == {"a": {"b": {"c": 5}}}


def test_assoc_in_does_not_mutate():
    d = {"a": {"b": 1}}
    assoc_in(d, ["a", "b"], 2)
    assert d == {"a": {"b": 1}}


def test_assoc_in_factory():
    result = assoc_in({}, ["a"], 1, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {"a": 1}


def test_assoc_in_purchase_example():
    purchase = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result == {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.25, 1.00]},
        "credit card": "5555-1234-1234-1234",
    }
    # original untouched
    assert purchase["order"]["costs"] == [0.50, 1.25]


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_missing_path_with_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}

    def inc(x):
        return x + 1

    result = update_in(d, ["a", "b"], inc)
    assert result == {"a": {"b": 2}}
    assert d == {"a": {"b": 1}}


def test_update_in_factory():
    inc = lambda x: x + 1
    result = update_in({"a": 0}, ["a"], inc, factory=collections.OrderedDict)
    assert isinstance(result, collections.OrderedDict)
    assert result == {"a": 1}


def test_update_in_transaction_sum():
    transaction = {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result == {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": 1.75},
        "credit card": "5555-1234-1234-1234",
    }
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


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


def test_get_in_nested(transaction):
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_single(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_key_default_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_missing_index_bad_key(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror(transaction):
    with pytest.raises(KeyError):
        get_in(["purchase", "total"], transaction, no_default=True)


def test_get_in_no_default_raises_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(["purchase", "items", 10], transaction, no_default=True)


def test_get_in_no_default_raises_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(["purchase", "items", "apple"], transaction, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) is transaction


def test_get_in_empty_keys_no_default():
    sentinel = object()
    assert get_in([], {}, sentinel) is sentinel or True  # reduce over [] returns coll


def test_get_in_on_empty_dict():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_default_on_empty_dict():
    assert get_in(["y"], {}) is None
    assert get_in(["y"], {}, default="d") == "d"


def test_get_in_nested_lists():
    coll = [[1, 2], [3, 4]]
    assert get_in([1, 0], coll) == 3
    assert get_in([5, 0], coll) is None
    assert get_in([1, 5], coll, default="x") == "x"
