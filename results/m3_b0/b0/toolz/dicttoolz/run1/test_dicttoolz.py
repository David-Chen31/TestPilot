import pytest
from collections import OrderedDict

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


class MyDict(dict):
    pass


def inc(x):
    return x + 1


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_empty_cases():
    assert merge() == {}
    assert merge([]) == {}
    assert merge(()) == {}
    assert merge(iter([])) == {}


def test_merge_multiple_dicts_later_has_precedence():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_mapping_is_not_treated_as_iterable_of_dicts():
    assert merge({1: 2}) == {1: 2}
    assert merge(OrderedDict([("a", 1), ("b", 2)])) == {"a": 1, "b": 2}


def test_merge_single_iterable_of_mappings():
    dicts = [{1: 1}, {2: 2}, {1: 3}]
    assert merge(dicts) == {1: 3, 2: 2}
    assert merge(iter(dicts)) == {1: 3, 2: 2}


def test_merge_factory():
    rv = merge(
        OrderedDict([("b", 2)]),
        OrderedDict([("a", 1)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2), ("a", 1)]

    assert type(merge({"a": 1}, factory=MyDict)) is MyDict
    assert type(merge(factory=OrderedDict)) is OrderedDict


def test_merge_unexpected_keyword_argument():
    with pytest.raises(TypeError) as excinfo:
        merge({"a": 1}, bad=True)
    assert str(excinfo.value) == (
        "merge() got an unexpected keyword argument 'bad'"
    )


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_combines_values():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(max, {"a": 1, "b": 10}, {"a": 5, "b": 2}) == {
        "a": 5,
        "b": 10,
    }


def test_merge_with_passes_list_of_values_to_function():
    result = merge_with(lambda values: values, {"a": 1}, {"a": 2, "b": 3})
    assert result == {"a": [1, 2], "b": [3]}
    assert type(result["a"]) is list


def test_merge_with_single_iterable_of_mappings():
    assert merge_with(sum, [{1: 1}, {1: 2}, {2: 3}]) == {1: 3, 2: 3}
    assert merge_with(sum, []) == {}
    assert merge_with(sum, {1: 1}) == {1: 1}


def test_merge_with_factory():
    rv = merge_with(
        sum,
        OrderedDict([("b", 1)]),
        OrderedDict([("a", 2)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 1), ("a", 2)]

    assert type(merge_with(sum, [], factory=MyDict)) is MyDict


def test_merge_with_unexpected_keyword_argument():
    with pytest.raises(TypeError) as excinfo:
        merge_with(sum, {"a": 1}, bad=True)
    assert str(excinfo.value) == (
        "merge_with() got an unexpected keyword argument 'bad'"
    )


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap_transforms_values_without_mutating_input():
    d = {"a": 1, "b": 2}
    assert valmap(lambda x: x * 10, d) == {"a": 10, "b": 20}
    assert d == {"a": 1, "b": 2}
    assert valmap(str, {}) == {}


def test_valmap_factory():
    rv = valmap(
        lambda x: x + 1,
        OrderedDict([("b", 1), ("a", 2)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2), ("a", 3)]


def test_valmap_propagates_errors():
    with pytest.raises(ZeroDivisionError):
        valmap(lambda v: 1 / v, {"a": 1, "b": 0})


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap_transforms_keys_without_mutating_input():
    d = {"a": 1, "b": 2}
    assert keymap(str.upper, d) == {"A": 1, "B": 2}
    assert d == {"a": 1, "b": 2}
    assert keymap(str, {}) == {}


def test_keymap_duplicate_keys_last_value_wins():
    assert keymap(lambda k: "x", {"a": 1, "b": 2}) == {"x": 2}


def test_keymap_factory():
    rv = keymap(
        lambda k: k.upper(),
        OrderedDict([("b", 1), ("a", 2)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("B", 1), ("A", 2)]


def test_keymap_propagates_errors():
    with pytest.raises(TypeError):
        keymap(lambda k: k + 1, {"a": 1})


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap_transforms_items_without_mutating_input():
    d = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda item: (item[1], item[0]), d) == {
        10: "Alice",
        20: "Bob",
    }
    assert d == {"Alice": 10, "Bob": 20}
    assert itemmap(lambda item: item, {}) == {}


def test_itemmap_duplicate_result_keys_last_value_wins():
    assert itemmap(lambda item: ("x", item[1]), {"a": 1, "b": 2}) == {
        "x": 2
    }


def test_itemmap_factory():
    rv = itemmap(
        lambda item: (item[1], item[0]),
        OrderedDict([("b", 2), ("a", 1)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [(2, "b"), (1, "a")]


def test_itemmap_invalid_return_raises():
    with pytest.raises(ValueError):
        itemmap(lambda item: (item[0],), {"a": 1})


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter_filters_by_value_without_mutating_input():
    is_even = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(is_even, d) == {1: 2, 3: 4}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda v: True, {}) == {}


def test_valfilter_factory():
    rv = valfilter(
        lambda v: v % 2 == 0,
        OrderedDict([("b", 2), ("a", 3), ("c", 4)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2), ("c", 4)]


def test_valfilter_propagates_predicate_errors():
    with pytest.raises(ZeroDivisionError):
        valfilter(lambda v: (1 / v) > 0, {1: 0})


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter_filters_by_key_without_mutating_input():
    is_even = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(is_even, d) == {2: 3, 4: 5}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda k: True, {}) == {}


def test_keyfilter_factory():
    rv = keyfilter(
        lambda k: k != "a",
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2), ("c", 3)]


def test_keyfilter_propagates_predicate_errors():
    with pytest.raises(ZeroDivisionError):
        keyfilter(lambda k: (1 / k) > 0, {0: "zero"})


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter_filters_by_item_without_mutating_input():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    rv = itemfilter(
        lambda item: item[1] > 1,
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        factory=OrderedDict,
    )
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2), ("c", 3)]


def test_itemfilter_propagates_predicate_errors():
    with pytest.raises(ZeroDivisionError):
        itemfilter(lambda item: (1 / item[0]) > 0, {0: "zero"})


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_returns_new_dict_without_mutating_input():
    d = {"x": 1}

    rv = assoc(d, "x", 2)
    assert rv == {"x": 2}
    assert d == {"x": 1}
    assert rv is not d

    rv2 = assoc(d, "y", 3)
    assert rv2 == {"x": 1, "y": 3}
    assert d == {"x": 1}


def test_assoc_factory():
    rv = assoc({"a": 1}, "b", 2, factory=MyDict)
    assert type(rv) is MyDict
    assert rv == {"a": 1, "b": 2}


def test_assoc_unhashable_key_raises():
    with pytest.raises(TypeError):
        assoc({}, [], 1)


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_removes_keys_without_mutating_input():
    d = {"x": 1, "y": 2}
    rv = dissoc(d, "y")
    assert rv == {"x": 1}
    assert d == {"x": 1, "y": 2}
    assert rv is not d

    assert dissoc(d, "y", "x") == {}
    assert dissoc(d, "missing") == d


def test_dissoc_missing_keys_are_ignored_in_both_branches():
    d = {i: i for i in range(5)}
    assert dissoc(d, 10) == d
    assert dissoc({"a": 1}, "missing") == {"a": 1}
    assert dissoc({}, "missing") == {}


def test_dissoc_no_keys_returns_shallow_copy():
    d = {"a": 1, "b": 2}
    rv = dissoc(d)
    assert rv == d
    assert rv is not d


def test_dissoc_else_branch_with_multiple_keys():
    d = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    rv = dissoc(d, 1, 2, 3)
    assert rv == {0: 0, 4: 4}
    assert d == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_dissoc_factory():
    rv = dissoc({"a": 1, "b": 2}, "a", factory=OrderedDict)
    assert type(rv) is OrderedDict
    assert list(rv.items()) == [("b", 2)]

    rv2 = dissoc({i: i for i in range(5)}, 1, 2, 3, factory=MyDict)
    assert type(rv2) is MyDict
    assert rv2 == {0: 0, 4: 4}


def test_dissoc_unexpected_keyword_argument():
    with pytest.raises(TypeError) as excinfo:
        dissoc({"a": 1}, "a", bad=True)
    assert str(excinfo.value) == (
        "dissoc() got an unexpected keyword argument 'bad'"
    )


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_updates_nested_value_without_mutating_input():
    d = {
        "name": "Alice",
        "order": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
    }
    original = {
        "name": "Alice",
        "order": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
    }

    rv = assoc_in(d, ["order", "costs"], [0.25, 1.00])

    assert rv == {
        "name": "Alice",
        "order": {
            "items": ["Apple", "Orange"],
            "costs": [0.25, 1.00],
        },
    }
    assert d == original
    assert rv is not d
    assert rv["order"] is not d["order"]


def test_assoc_in_creates_nested_keys():
    assert assoc_in({}, [1, 2, 3], "bar") == {1: {2: {3: "bar"}}}
    assert assoc_in({"a": 1}, ["b", "c"], 5) == {
        "a": 1,
        "b": {"c": 5},
    }


def test_assoc_in_existing_top_level_key():
    d = {"a": 1}
    assert assoc_in(d, ["a"], 2) == {"a": 2}
    assert d == {"a": 1}


def test_assoc_in_factory():
    rv = assoc_in({}, [1, 2], "x", factory=MyDict)
    assert type(rv) is MyDict
    assert type(rv[1]) is MyDict
    assert rv == {1: {2: "x"}}


def test_assoc_in_empty_keys_raises_stop_iteration():
    with pytest.raises(StopIteration):
        assoc_in({}, [], 1)


def test_assoc_in_non_mapping_intermediate_raises():
    with pytest.raises(TypeError):
        assoc_in({"a": 1}, ["a", "b"], 2)


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_existing_value():
    d = {"a": 0}
    rv = update_in(d, ["a"], inc)
    assert rv == {"a": 1}
    assert d == {"a": 0}


def test_update_in_missing_key_uses_default():
    assert update_in({}, ["a"], inc, default=10) == {"a": 11}
    assert update_in({}, ["a"], lambda x: "created") == {"a": "created"}


def test_update_in_existing_none_value_is_updated():
    d = {"a": None}
    rv = update_in(d, ["a"], lambda x: 5 if x is None else x)
    assert rv == {"a": 5}
    assert d == {"a": None}


def test_update_in_nested_existing_value_preserves_original():
    d = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }
    original = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }

    rv = update_in(d, ["purchase", "costs"], sum)

    assert rv == {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": 1.75,
        },
        "credit card": "5555-1234-1234-1234",
    }
    assert d == original
    assert rv["purchase"] is not d["purchase"]


def test_update_in_creates_nested_dicts():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: "bar"}}
    }
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }
    assert update_in(
        {"a": {"b": 1}}, ["a", "c", "d"], inc, default=4
    ) == {"a": {"b": 1, "c": {"d": 5}}}


def test_update_in_factory():
    rv = update_in({}, [1, 2], lambda x: x, default=0, factory=MyDict)
    assert type(rv) is MyDict
    assert type(rv[1]) is MyDict
    assert rv == {1: {2: 0}}

    rv2 = update_in(
        {"a": {"b": 1}},
        ["a", "b"],
        inc,
        factory=MyDict,
    )
    assert type(rv2) is MyDict
    assert type(rv2["a"]) is MyDict
    assert rv2 == {"a": {"b": 2}}


def test_update_in_does_not_mutate_original_for_new_path():
    d = {"a": 1}
    rv = update_in(d, ["b", "c"], inc, default=0)
    assert rv == {"a": 1, "b": {"c": 1}}
    assert d == {"a": 1}


def test_update_in_accepts_iterator_of_keys():
    def keygen():
        yield "a"
        yield "b"

    assert update_in({}, keygen(), inc, default=1) == {"a": {"b": 2}}


def test_update_in_empty_keys_raises_stop_iteration():
    with pytest.raises(StopIteration):
        update_in({}, [], inc)


def test_update_in_non_mapping_intermediate_raises():
    with pytest.raises(TypeError):
        update_in({"a": 1}, ["a", "b"], inc)


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

def test_get_in_nested_dict_and_list_values():
    transaction = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }

    assert get_in(["purchase", "items", 0], transaction) == "Apple"
    assert get_in(("purchase", "items", 1), transaction) == "Orange"
    assert get_in(["name"], transaction) == "Alice"
    assert get_in([], transaction) is transaction


def test_get_in_missing_values_return_default():
    transaction = {
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        }
    }

    assert get_in(["purchase", "total"], transaction) is None
    assert get_in(["purchase", "total"], transaction, default=0) == 0
    assert get_in(["purchase", "items", "apple"], transaction) is None
    assert get_in(["purchase", "items", 10], transaction) is None
    assert get_in(["missing"], {}, default="d") == "d"
    assert get_in(
        ["purchase", "missing", "x"], transaction, default="fallback"
    ) == "fallback"


def test_get_in_missing_with_no_default_raises_original_exception():
    with pytest.raises(KeyError) as excinfo:
        get_in(["missing"], {}, no_default=True)
    assert excinfo.value.args == ("missing",)

    with pytest.raises(IndexError):
        get_in([2], [1], no_default=True)

    with pytest.raises(TypeError):
        get_in(["bad"], [], no_default=True)


def test_get_in_existing_none_is_not_replaced_by_default():
    assert get_in(["a"], {"a": None}, default=5) is None


def test_get_in_no_default_with_empty_keys_returns_collection():
    coll = {"a": 1}
    assert get_in([], coll, no_default=True) is coll
