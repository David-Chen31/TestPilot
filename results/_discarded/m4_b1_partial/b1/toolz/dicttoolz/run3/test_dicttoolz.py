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


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping_not_expanded():
    # A single Mapping argument is treated as one dict, not iterated.
    assert merge({1: 2}) == {1: 2}


def test_merge_factory():
    from collections import OrderedDict

    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=1)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------
def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_one_value():
    assert merge_with(lambda vals: vals, {1: 1}) == {1: [1]}


def test_merge_with_multiple_values():
    result = merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 2}, {1: 3}]) == {1: 5}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    from collections import OrderedDict

    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------
def test_valmap():
    d = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, d) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_does_not_mutate():
    d = {"a": 1}
    result = valmap(lambda x: x + 1, d)
    assert result == {"a": 2}
    assert d == {"a": 1}


def test_valmap_factory():
    from collections import OrderedDict

    result = valmap(lambda x: x, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keymap():
    d = {"Alice": 1, "Bob": 2}
    assert keymap(str.lower, d) == {"alice": 1, "bob": 2}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict

    result = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemmap():
    d = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, d) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(lambda x: x, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict

    result = itemmap(lambda kv: kv, {"a": 1}, factory=OrderedDict)
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

    result = valfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict

    result = keyfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_empty():
    assert itemfilter(lambda x: True, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict

    result = itemfilter(lambda x: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    itemfilter(lambda kv: kv[0] == 1, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------
def test_assoc_existing_key():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_new_key():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_mutate():
    d = {"x": 1}
    result = assoc(d, "x", 2)
    assert result == {"x": 2}
    assert d == {"x": 1}


def test_assoc_factory():
    from collections import OrderedDict

    result = assoc({"x": 1}, "y", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------
def test_dissoc_one_key():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple_keys():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    d = {"x": 1, "y": 2}
    assert dissoc(d) == {"x": 1, "y": 2}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    result = dissoc(d, "y")
    assert result == {"x": 1}
    assert d == {"x": 1, "y": 2}


def test_dissoc_factory():
    from collections import OrderedDict

    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad=1)


def test_dissoc_large_key_count_branch():
    # Exercises the "remaining set" branch when most keys are removed.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(9))
    assert result == {9: 9}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------
def test_assoc_in_existing_path():
    purchase = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result["order"]["costs"] == [0.25, 1.00]
    # Original untouched
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_new_path():
    result = assoc_in({}, ["a", "b"], 1)
    assert result == {"a": {"b": 1}}


def test_update_in_existing():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_new_key_with_default():
    inc = lambda x: x + 1
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}
    result = update_in(d, ["a", "b"], lambda x: x + 1)
    assert result == {"a": {"b": 2}}
    assert d == {"a": {"b": 1}}


def test_update_in_factory():
    from collections import OrderedDict

    result = update_in({"a": 0}, ["a"], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


def test_assoc_in_factory():
    from collections import OrderedDict

    result = assoc_in({}, ["a", "b"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": {"b": 1}}


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
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_returns_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_index_error_returns_default(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_indexerror(transaction):
    with pytest.raises(IndexError):
        get_in(["purchase", "items", 10], transaction, no_default=True)


def test_get_in_no_default_typeerror(transaction):
    with pytest.raises(TypeError):
        get_in(["purchase", "items", "apple"], transaction, no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) == transaction
