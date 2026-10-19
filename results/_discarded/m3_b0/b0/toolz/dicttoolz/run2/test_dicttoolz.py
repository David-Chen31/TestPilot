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


def test_merge_basic_and_precedence():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    assert merge() == {}


def test_merge_single_mapping_vs_iterable_of_mappings():
    assert merge({1: "a"}) == {1: "a"}
    assert merge([{1: "a"}, {2: "b"}, {1: "c"}]) == {1: "c", 2: "b"}
    assert merge(d for d in ({"a": 1}, {"b": 2})) == {"a": 1, "b": 2}


def test_merge_does_not_mutate_inputs():
    left = {"a": 1}
    right = {"a": 2, "b": 3}

    result = merge(left, right)

    assert result == {"a": 2, "b": 3}
    assert result is not left
    assert result is not right
    assert left == {"a": 1}
    assert right == {"a": 2, "b": 3}


def test_merge_factory_and_unexpected_kwarg():
    result = merge({"a": 1}, {"b": 2}, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1), ("b", 2)]

    with pytest.raises(
        TypeError,
        match=r"merge\(\) got an unexpected keyword argument 'bad'",
    ):
        merge({}, bad=1)


def test_merge_with_combines_values():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(sum, {1: 1}, {2: 2}) == {1: 1, 2: 2}
    assert merge_with(list, {1: "a", 2: "b"}, {1: "c"}) == {
        1: ["a", "c"],
        2: ["b"],
    }


def test_merge_with_single_mapping_and_iterable_of_mappings():
    assert merge_with(sum, {1: 2}) == {1: 2}
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}
    assert merge_with(sum) == {}


def test_merge_with_receives_lists_and_does_not_mutate():
    def capture(values):
        assert isinstance(values, list)
        return sum(values)

    left = {1: 1, 2: 2}
    right = {1: 10}

    result = merge_with(capture, left, right)

    assert result == {1: 11, 2: 2}
    assert left == {1: 1, 2: 2}
    assert right == {1: 10}


def test_merge_with_factory_and_unexpected_kwarg():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, 3)]

    with pytest.raises(
        TypeError,
        match=r"merge_with\(\) got an unexpected keyword argument 'bad'",
    ):
        merge_with(sum, {}, bad=1)


def test_valmap_applies_to_values_and_does_not_mutate():
    d = {"a": 1, "b": 2}

    result = valmap(lambda x: x * 10, d)

    assert result == {"a": 10, "b": 20}
    assert d == {"a": 1, "b": 2}
    assert valmap(str, {}) == {}


def test_valmap_factory_preserves_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    result = valmap(str, d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", "1"), ("b", "2")]


def test_keymap_applies_to_keys_and_handles_duplicates():
    d = {"Alice": 1, "Bob": 2}

    result = keymap(str.lower, d)

    assert result == {"alice": 1, "bob": 2}
    assert d == {"Alice": 1, "Bob": 2}
    assert keymap(lambda k: "same", {"a": 1, "b": 2}) == {"same": 2}
    assert keymap(str, {}) == {}


def test_keymap_factory_preserves_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    result = keymap(str.upper, d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("A", 1), ("B", 2)]


def test_itemmap_applies_to_items_and_handles_duplicates():
    d = {"a": 1, "b": 2}

    result = itemmap(lambda item: (item[0].upper(), item[1] * 10), d)

    assert result == {"A": 10, "B": 20}
    assert d == {"a": 1, "b": 2}
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == {
        10: "Alice",
        20: "Bob",
    }
    assert itemmap(lambda kv: ("same", kv[1]), {"a": 1, "b": 2}) == {
        "same": 2,
    }
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory_preserves_order():
    d = OrderedDict([("a", 1), ("b", 2)])

    result = itemmap(lambda kv: (kv[0], kv[1] + 1), d, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 2), ("b", 3)]


def test_valfilter_filters_by_value_and_supports_factory():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    is_even = lambda x: x % 2 == 0

    result = valfilter(is_even, d)

    assert result == {1: 2, 3: 4}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda v: False, d) == {}

    ordered = OrderedDict([(1, 2), (2, 3), (3, 4)])
    ordered_result = valfilter(is_even, ordered, factory=OrderedDict)

    assert isinstance(ordered_result, OrderedDict)
    assert list(ordered_result.items()) == [(1, 2), (3, 4)]


def test_keyfilter_filters_by_key_and_supports_factory():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    is_even = lambda x: x % 2 == 0

    result = keyfilter(is_even, d)

    assert result == {2: 3, 4: 5}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda k: False, d) == {}

    ordered = OrderedDict([(1, 2), (2, 3), (3, 4)])
    ordered_result = keyfilter(is_even, ordered, factory=OrderedDict)

    assert isinstance(ordered_result, OrderedDict)
    assert list(ordered_result.items()) == [(2, 3)]


def test_itemfilter_filters_by_item_and_supports_factory():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def is_valid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    result = itemfilter(is_valid, d)

    assert result == {2: 3}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(lambda item: False, d) == {}

    ordered = OrderedDict([(1, 2), (3, 4), (4, 5)])
    ordered_result = itemfilter(
        lambda item: item[0] >= 3,
        ordered,
        factory=OrderedDict,
    )

    assert isinstance(ordered_result, OrderedDict)
    assert list(ordered_result.items()) == [(3, 4), (4, 5)]


def test_assoc_returns_new_mapping_without_mutating_input():
    d = {"x": 1}

    result = assoc(d, "x", 2)

    assert result == {"x": 2}
    assert result is not d
    assert d == {"x": 1}

    result = assoc(d, "y", 3)

    assert result == {"x": 1, "y": 3}
    assert d == {"x": 1}


def test_assoc_factory_preserves_order():
    d = OrderedDict([("a", 1)])

    result = assoc(d, "b", 2, factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1), ("b", 2)]


def test_dissoc_copy_branch_removes_keys_and_ignores_missing():
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    result = dissoc(d, "b")

    assert result == {"a": 1, "c": 3, "d": 4, "e": 5}
    assert result is not d
    assert d == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    result = dissoc(d, "b", "missing")

    assert result == {"a": 1, "c": 3, "d": 4, "e": 5}
    assert d == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def test_dissoc_remaining_branch():
    d = {"a": 1, "b": 2, "c": 3}

    result = dissoc(d, "a", "b")

    assert result == {"c": 3}
    assert d == {"a": 1, "b": 2, "c": 3}

    assert dissoc({"a": 1}, "missing") == {"a": 1}
    assert dissoc({"a": 1, "b": 2}, "a", "b") == {}
    assert dissoc({}) == {}


def test_dissoc_factory_and_unexpected_kwarg():
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    result = dissoc(d, "b", factory=OrderedDict)

    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1), ("c", 3), ("d", 4), ("e", 5)]

    with pytest.raises(
        TypeError,
        match=r"dissoc\(\) got an unexpected keyword argument 'bad'",
    ):
        dissoc({"a": 1}, bad=1)


def test_assoc_in_nested_update_and_non_mutation():
    original = {
        "name": "Alice",
        "order": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }

    result = assoc_in(original, ["order", "costs"], [0.25, 1.00])

    assert result["order"]["costs"] == [0.25, 1.00]
    assert result["order"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    assert result is not original
    assert result["order"] is not original["order"]
    assert original["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_creates_missing_paths_and_supports_factory():
    source = {}

    result = assoc_in(source, ["a", "b"], 5)

    assert result == {"a": {"b": 5}}
    assert source == {}

    assert assoc_in({"a": 1}, ["a"], 2) == {"a": 2}

    ordered = assoc_in({}, ["a", "b"], 1, factory=OrderedDict)

    assert isinstance(ordered, OrderedDict)
    assert isinstance(ordered["a"], OrderedDict)
    assert dict(ordered["a"]) == {"b": 1}

    with pytest.raises(StopIteration):
        assoc_in({}, [], 1)


def test_update_in_updates_existing_value_without_mutating():
    d = {"a": 0}

    result = update_in(d, ["a"], lambda x: x + 1)

    assert result == {"a": 1}
    assert result is not d
    assert d == {"a": 0}


def test_update_in_nested_existing_path():
    d = {
        "purchase": {
            "costs": [0.50, 1.25],
            "items": ["Apple", "Orange"],
        },
        "name": "Alice",
    }

    result = update_in(d, ["purchase", "costs"], sum)

    assert result["purchase"]["costs"] == 1.75
    assert result["purchase"]["items"] == ["Apple", "Orange"]
    assert result["name"] == "Alice"
    assert result["purchase"] is not d["purchase"]
    assert d["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_missing_paths_and_defaults():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: "bar"}},
    }
    assert update_in({1: "foo"}, [2, 3, 4], lambda x: x + 1, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }

    d = {"a": 10}

    assert update_in(d, ["a"], lambda x: x * 2, default=100) == {"a": 20}
    assert update_in(d, ["b"], lambda x: x + 5, default=100) == {
        "a": 10,
        "b": 105,
    }
    assert d == {"a": 10}


def test_update_in_distinguishes_missing_keys_from_none_values():
    assert update_in(
        {"a": None},
        ["a"],
        lambda x: f"got-{x}",
        default="default",
    ) == {"a": "got-None"}

    assert update_in(
        {},
        ["a"],
        lambda x: f"got-{x}",
        default="default",
    ) == {"a": "got-default"}


def test_update_in_factory_and_errors():
    result = update_in(
        {},
        ["a", "b"],
        lambda x: x + 1,
        default=0,
        factory=OrderedDict,
    )

    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], OrderedDict)
    assert dict(result["a"]) == {"b": 1}

    with pytest.raises(StopIteration):
        update_in({}, [], lambda x: x)

    with pytest.raises(TypeError):
        update_in({"a": 1}, ["a", "b"], lambda x: x)


def test_get_in_nested_access():
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


def test_get_in_missing_keys_and_indexes_use_default():
    coll = {"a": {"b": [10, 20]}}
    sentinel = object()

    assert get_in(["a", "b", 0], coll) == 10
    assert get_in(["a", "missing", "x"], coll) is None
    assert get_in(["a", "missing", "x"], coll, default="x") == "x"
    assert get_in(["a", "b", 5], coll) is None
    assert get_in(["a", "b", "bad"], coll) is None
    assert get_in(["x"], coll, sentinel) is sentinel
    assert get_in(["a"], 1) is None


def test_get_in_no_default_raises_original_exception():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)

    with pytest.raises(KeyError):
        get_in(["y"], {}, default=42, no_default=True)

    with pytest.raises(IndexError):
        get_in(["a", 0], {"a": []}, no_default=True)

    with pytest.raises(TypeError):
        get_in(["a", "b"], {"a": 1}, no_default=True)
