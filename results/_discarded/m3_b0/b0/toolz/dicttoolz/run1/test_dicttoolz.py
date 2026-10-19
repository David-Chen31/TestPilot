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


def test_merge_multiple_dicts_later_values_win():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 20, "c": 30}

    result = merge(d1, d2)

    assert result == {"a": 1, "b": 20, "c": 30}
    assert d1 == {"a": 1, "b": 2}
    assert d2 == {"b": 20, "c": 30}
    assert result is not d1
    assert result is not d2


def test_merge_no_args_and_single_iterable():
    assert merge() == {}
    assert merge([]) == {}
    assert merge([{"a": 1}, {"b": 2}]) == {"a": 1, "b": 2}
    assert merge(({"a": 1}, {"a": 2})) == {"a": 2}
    assert merge(iter([{"a": 1}, {"b": 2}])) == {"a": 1, "b": 2}


def test_merge_single_mapping_is_not_treated_as_an_iterable_of_dicts():
    d = {"a": 1}
    result = merge(d)

    assert result == {"a": 1}
    assert result is not d


def test_merge_factory():
    result = merge({"b": 2}, {"a": 1}, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("b", 2), ("a", 1)]

    empty = merge(factory=OrderedDict)
    assert isinstance(empty, OrderedDict)
    assert empty == {}


def test_merge_rejects_unexpected_keyword_argument():
    with pytest.raises(TypeError) as excinfo:
        merge({"a": 1}, bad=True)

    assert str(excinfo.value) == "merge() got an unexpected keyword argument 'bad'"


def test_merge_with_collects_values_by_key_in_order():
    result = merge_with(
        lambda values: list(values),
        {"a": 1, "b": 2},
        {"a": 3},
        {"b": 4, "a": 5},
    )

    assert result == {"a": [1, 3, 5], "b": [2, 4]}


def test_merge_with_applies_function_to_grouped_values():
    result = merge_with(sum, {"a": 1, "b": 2}, {"a": 10, "c": 3}, {"a": 100})

    assert result == {"a": 111, "b": 2, "c": 3}


def test_merge_with_no_args_single_iterable_and_single_mapping():
    assert merge_with(sum) == {}
    assert merge_with(sum, [{"a": 1}, {"a": 2}]) == {"a": 3}
    assert merge_with(lambda values: list(values), {"a": 1}) == {"a": [1]}


def test_merge_with_factory():
    result = merge_with(
        lambda values: list(values),
        {"b": 2},
        {"a": 1},
        factory=OrderedDict,
    )

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("b", [2]), ("a", [1])]


def test_merge_with_rejects_unexpected_keyword_argument():
    with pytest.raises(TypeError) as excinfo:
        merge_with(sum, {"a": 1}, bad=True)

    assert (
        str(excinfo.value)
        == "merge_with() got an unexpected keyword argument 'bad'"
    )


def test_valmap_transforms_values_without_mutating_input():
    d = {"a": 1, "b": 2}

    result = valmap(lambda value: value * 10, d)

    assert result == {"a": 10, "b": 20}
    assert d == {"a": 1, "b": 2}
    assert valmap(lambda value: value, {}) == {}


def test_valmap_factory_and_order():
    d = OrderedDict([("b", 2), ("a", 1)])

    result = valmap(str, d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("b", "2"), ("a", "1")]


def test_keymap_transforms_keys_without_mutating_input():
    d = {"a": 1, "b": 2}

    result = keymap(str.upper, d)

    assert result == {"A": 1, "B": 2}
    assert d == {"a": 1, "b": 2}
    assert keymap(lambda key: key, {}) == {}


def test_keymap_collisions_last_value_wins():
    result = keymap(lambda key: key % 2, {1: "a", 2: "b", 3: "c"})

    assert result == {0: "b", 1: "c"}


def test_keymap_factory_and_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    result = keymap(str.upper, d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("A", 1), ("B", 2)]


def test_itemmap_transforms_items_without_mutating_input():
    d = {"a": 1, "b": 2}

    result = itemmap(lambda item: (item[1], item[0]), d)

    assert result == {1: "a", 2: "b"}
    assert d == {"a": 1, "b": 2}
    assert itemmap(lambda item: item, {}) == {}


def test_itemmap_collisions_last_value_wins():
    d = OrderedDict([(1, "a"), (2, "b"), (3, "c")])

    result = itemmap(lambda item: (item[0] % 2, item[1]), d)

    assert result == {0: "b", 1: "c"}


def test_itemmap_factory_and_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    result = itemmap(lambda item: (item[1], item[0]), d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, "a"), (2, "b")]


def test_valfilter_filters_by_value():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    result = valfilter(lambda value: value % 2 == 0, d)

    assert result == {1: 2, 3: 4}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda value: False, d) == {}


def test_keyfilter_filters_by_key():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    result = keyfilter(lambda key: key % 2 == 0, d)

    assert result == {2: 3, 4: 5}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda key: False, d) == {}


def test_itemfilter_filters_by_item():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        key, value = item
        return key % 2 == 0 and value < 4

    result = itemfilter(isvalid, d)

    assert result == {2: 3}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(lambda item: False, d) == {}


def test_filters_use_factory():
    d = OrderedDict([(1, 2), (2, 3), (3, 4)])

    value_result = valfilter(lambda value: value % 2 == 0, d, factory=OrderedDict)
    key_result = keyfilter(lambda key: key % 2 == 0, d, factory=OrderedDict)
    item_result = itemfilter(
        lambda item: item[0] % 2 == 1 and item[1] % 2 == 0,
        d,
        factory=OrderedDict,
    )

    assert isinstance(value_result, OrderedDict)
    assert list(value_result.items()) == [(1, 2), (3, 4)]

    assert isinstance(key_result, OrderedDict)
    assert list(key_result.items()) == [(2, 3)]

    assert isinstance(item_result, OrderedDict)
    assert list(item_result.items()) == [(1, 2), (3, 4)]


def test_assoc_returns_new_dict_with_change():
    d = {"a": 1}

    result_new_key = assoc(d, "b", 2)
    result_existing_key = assoc(d, "a", 9)

    assert result_new_key == {"a": 1, "b": 2}
    assert result_existing_key == {"a": 9}
    assert d == {"a": 1}
    assert result_new_key is not d
    assert result_existing_key is not d


def test_assoc_factory_preserves_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    existing = assoc(d, "a", 9, factory=OrderedDict)
    new = assoc(d, "c", 3, factory=OrderedDict)

    assert isinstance(existing, OrderedDict)
    assert list(existing.items()) == [("a", 9), ("b", 2)]

    assert isinstance(new, OrderedDict)
    assert list(new.items()) == [("a", 1), ("b", 2), ("c", 3)]


def test_dissoc_removes_keys_without_mutating_input():
    d = {"a": 1, "b": 2, "c": 3}

    result = dissoc(d, "b")
    missing = dissoc(d, "missing")

    assert result == {"a": 1, "c": 3}
    assert missing == d
    assert missing is not d
    assert d == {"a": 1, "b": 2, "c": 3}


def test_dissoc_multiple_keys_and_empty_cases():
    assert dissoc({"a": 1, "b": 2, "c": 3}, "a", "c") == {"b": 2}
    assert dissoc({"a": 1}, "a") == {}
    assert dissoc({"a": 1}) == {"a": 1}
    assert dissoc({}, "a") == {}


def test_dissoc_both_internal_strategy_branches_ignore_missing_keys():
    d5 = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    d3 = {"a": 1, "b": 2, "c": 3}

    # len(keys)=2 < len(d5)*0.6, uses update/delete branch
    assert dissoc(d5, "b", "missing") == {"a": 1, "c": 3, "d": 4, "e": 5}

    # len(keys)=2 is not < len(d3)*0.6, uses set-difference branch
    assert dissoc(d3, "a", "missing") == {"b": 2, "c": 3}

    assert dissoc(d3, "a", "b", "c") == {}


def test_dissoc_factory_and_keyword_validation():
    d = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

    result = dissoc(d, "b", factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1), ("c", 3)]

    with pytest.raises(TypeError) as excinfo:
        dissoc(d, "b", bad=True)

    assert str(excinfo.value) == "dissoc() got an unexpected keyword argument 'bad'"


def test_update_in_updates_existing_value_and_returns_copy():
    d = {"a": 1, "b": 2}

    result = update_in(d, ["a"], lambda value: value + 10)

    assert result == {"a": 11, "b": 2}
    assert d == {"a": 1, "b": 2}
    assert result is not d


def test_update_in_missing_keys_creates_nested_dicts_with_default():
    result_nested = update_in({}, ["a", "b"], lambda value: value + 1, default=10)
    result_top = update_in({}, ["a"], lambda value: value * 2, default=3)
    result_partial = update_in({"a": {}}, ["a", "b"], lambda value: value + 5, default=1)
    result_new_top = update_in({"a": 1}, ["b"], lambda value: value + 5, default=2)

    assert result_nested == {"a": {"b": 11}}
    assert result_top == {"a": 6}
    assert result_partial == {"a": {"b": 6}}
    assert result_new_top == {"a": 1, "b": 7}


def test_update_in_deep_update_copies_containers_on_path_only():
    d = {"a": {"b": {"c": 1}}, "x": {"y": 2}, "z": 3}

    result = update_in(d, ["a", "b", "c"], lambda value: value + 1)

    assert result == {"a": {"b": {"c": 2}}, "x": {"y": 2}, "z": 3}
    assert d == {"a": {"b": {"c": 1}}, "x": {"y": 2}, "z": 3}

    assert result is not d
    assert result["a"] is not d["a"]
    assert result["a"]["b"] is not d["a"]["b"]
    assert result["a"]["b"]["c"] == 2
    assert result["x"] is d["x"]


def test_update_in_factory_is_used_for_copied_and_new_levels():
    d = OrderedDict([("a", OrderedDict([("b", 1)])), ("c", 2)])

    existing = update_in(d, ["a", "b"], lambda value: value + 1, factory=OrderedDict)
    created = update_in({}, ["a", "b"], lambda value: value + 1, default=0, factory=OrderedDict)

    assert isinstance(existing, OrderedDict)
    assert isinstance(existing["a"], OrderedDict)
    assert list(existing.items()) == [
        ("a", OrderedDict([("b", 2)])),
        ("c", 2),
    ]

    assert isinstance(created, OrderedDict)
    assert isinstance(created["a"], OrderedDict)
    assert list(created.items()) == [("a", OrderedDict([("b", 1)]))]


def test_update_in_empty_keys_raises_stopiteration():
    with pytest.raises(StopIteration):
        update_in({"a": 1}, [], lambda value: value)


def test_update_in_non_mapping_intermediate_raises_type_error():
    with pytest.raises(TypeError):
        update_in({"a": 1}, ["a", "b"], lambda value: value)


def test_assoc_in_replaces_existing_nested_value_without_mutating_input():
    d = {
        "name": "Alice",
        "order": {"items": ["Apple"], "costs": [0.5]},
    }

    result = assoc_in(d, ["order", "costs"], [0.25])

    assert result == {
        "name": "Alice",
        "order": {"items": ["Apple"], "costs": [0.25]},
    }
    assert d == {
        "name": "Alice",
        "order": {"items": ["Apple"], "costs": [0.5]},
    }
    assert result is not d
    assert result["order"] is not d["order"]


def test_assoc_in_creates_missing_paths_with_value():
    assert assoc_in({}, ["a", "b", "c"], 7) == {"a": {"b": {"c": 7}}}
    assert assoc_in({"a": {}}, ["a", "b"], 9) == {"a": {"b": 9}}
    assert assoc_in({"x": 1}, ["y"], 2) == {"x": 1, "y": 2}
    assert assoc_in({}, (key for key in ["a", "b"]), 3) == {"a": {"b": 3}}


def test_assoc_in_uses_factory_for_all_new_levels():
    created = assoc_in({}, ["a", "b"], 5, factory=OrderedDict)

    assert isinstance(created, OrderedDict)
    assert isinstance(created["a"], OrderedDict)
    assert list(created.items()) == [("a", OrderedDict([("b", 5)]))]

    d = OrderedDict([("a", OrderedDict([("b", 1)]))])
    existing = assoc_in(d, ["a", "b"], 2, factory=OrderedDict)

    assert isinstance(existing, OrderedDict)
    assert isinstance(existing["a"], OrderedDict)
    assert list(existing["a"].items()) == [("b", 2)]


def test_assoc_in_empty_keys_raises_stopiteration():
    with pytest.raises(StopIteration):
        assoc_in({"a": 1}, [], 2)


def test_get_in_nested_structures():
    transaction = {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.5, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }

    assert get_in(["name"], transaction) == "Alice"
    assert get_in(["purchase", "items", 0], transaction) == "Apple"
    assert get_in(["purchase", "costs", 1], transaction) == 1.25
    assert get_in([0], [10, 20]) == 10
    assert get_in([-1], [10, 20]) == 20


def test_get_in_returns_default_for_missing_or_invalid_access():
    d = {"a": {"b": 1}, "lst": [1]}
    sentinel = object()

    assert get_in(["a", "x"], d, default=sentinel) is sentinel
    assert get_in(["lst", 5], d, default="missing-index") == "missing-index"
    assert get_in(["lst", "bad"], d, default="type-error") == "type-error"
    assert get_in(["a", "b"], {"a": 1}, default="not-mapping") == "not-mapping"
    assert get_in(["a"], None, default="none") == "none"
    assert get_in(["missing"], {}, default=None) is None


def test_get_in_empty_keys_returns_collection_itself():
    d = {"a": 1}

    assert get_in([], d) is d
    assert get_in((), d) is d


def test_get_in_no_default_raises_original_exceptions():
    with pytest.raises(KeyError) as excinfo_key:
        get_in(["missing"], {}, no_default=True)

    assert excinfo_key.value.args == ("missing",)

    with pytest.raises(IndexError):
        get_in([10], [1], no_default=True)

    with pytest.raises(TypeError):
        get_in(["a"], [1], no_default=True)

    with pytest.raises(KeyError):
        get_in(["missing"], {}, default="ignored", no_default=True)
