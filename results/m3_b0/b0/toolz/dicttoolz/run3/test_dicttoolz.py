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


def _as_list(values):
    return list(values)


def test_merge_basic_and_precedence():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    assert merge() == {}
    assert merge({"a": 1}) == {"a": 1}


def test_merge_single_non_mapping_iterable():
    assert merge([{"a": 1}, {"b": 2, "a": 3}]) == {"a": 3, "b": 2}
    assert merge(d for d in ({"a": 1}, {"b": 2})) == {"a": 1, "b": 2}


def test_merge_does_not_mutate_inputs():
    d1 = {"a": 1}
    d2 = {"b": 2}

    result = merge(d1, d2)
    result["a"] = 99

    assert d1 == {"a": 1}
    assert d2 == {"b": 2}


def test_merge_factory_and_unexpected_keyword():
    result = merge(
        OrderedDict([("b", 2)]),
        OrderedDict([("a", 1)]),
        factory=OrderedDict,
    )
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("b", 2), ("a", 1)]

    empty = merge(factory=OrderedDict)
    assert isinstance(empty, OrderedDict)
    assert empty == {}

    with pytest.raises(
        TypeError,
        match=r"merge\(\) got an unexpected keyword argument 'bad'",
    ):
        merge({}, bad=True)


def test_merge_with_basic_and_value_order():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(
        _as_list,
        {"a": 1, "b": 2},
        {"a": 3, "c": 4},
    ) == {
        "a": [1, 3],
        "b": [2],
        "c": [4],
    }


def test_merge_with_single_non_mapping_and_empty():
    assert merge_with(sum, [{"a": 1}, {"a": 2, "b": 3}]) == {"a": 3, "b": 3}
    assert merge_with(sum) == {}


def test_merge_with_factory_and_unexpected_keyword():
    result = merge_with(sum, {"a": 1}, {"a": 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 3)]

    empty = merge_with(sum, factory=OrderedDict)
    assert isinstance(empty, OrderedDict)
    assert empty == {}

    with pytest.raises(
        TypeError,
        match=r"merge_with\(\) got an unexpected keyword argument 'bad'",
    ):
        merge_with(sum, {}, bad=True)


def test_merge_with_does_not_mutate_inputs():
    d1 = {"a": 1}
    d2 = {"a": 2}

    result = merge_with(_as_list, d1, d2)
    result["a"].append(3)

    assert d1 == {"a": 1}
    assert d2 == {"a": 2}


def test_valmap_basic_factory_and_no_mutation():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}

    result = valmap(sum, bills)
    assert result == {"Alice": 65, "Bob": 45}
    assert result is not bills
    assert bills == {"Alice": [20, 15, 30], "Bob": [10, 35]}

    ordered = valmap(
        lambda value: value + 1,
        OrderedDict([("b", 2), ("a", 1)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("b", 3), ("a", 2)]

    assert valmap(str, {}) == {}


def test_keymap_basic_collision_factory_and_no_mutation():
    original = {"a": 1, "b": 2}
    assert keymap(str.upper, original) == {"A": 1, "B": 2}
    assert original == {"a": 1, "b": 2}

    assert keymap(lambda key: "same", {"a": 1, "b": 2}) == {"same": 2}

    ordered = keymap(
        str.upper,
        OrderedDict([("b", 2), ("a", 1)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("B", 2), ("A", 1)]

    assert keymap(str.upper, {}) == {}


def test_itemmap_basic_collision_factory_and_no_mutation():
    original = {"Alice": 10, "Bob": 20}
    result = itemmap(lambda item: (item[1], item[0]), original)
    assert result == {10: "Alice", 20: "Bob"}
    assert original == {"Alice": 10, "Bob": 20}

    assert itemmap(lambda item: ("x", item[1]), {"a": 1, "b": 2}) == {"x": 2}

    ordered = itemmap(
        lambda item: (item[0].upper(), item[1] + 1),
        OrderedDict([("a", 1), ("b", 2)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("A", 2), ("B", 3)]

    assert itemmap(lambda item: item, {}) == {}


def test_valfilter_basic_factory_and_predicate_receives_values():
    data = {1: 2, 2: 3, 3: 4, 4: 5}
    seen = []

    def is_even(value):
        seen.append(value)
        return value % 2 == 0

    result = valfilter(is_even, data)
    assert result == {1: 2, 3: 4}
    assert seen == [2, 3, 4, 5]
    assert data == {1: 2, 2: 3, 3: 4, 4: 5}

    ordered = valfilter(
        lambda value: value > 1,
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("b", 2), ("c", 3)]

    assert valfilter(lambda value: True, {}) == {}


def test_keyfilter_basic_factory_and_predicate_receives_keys():
    data = {1: 2, 2: 3, 3: 4, 4: 5}
    seen = []

    def is_even(key):
        seen.append(key)
        return key % 2 == 0

    result = keyfilter(is_even, data)
    assert result == {2: 3, 4: 5}
    assert seen == [1, 2, 3, 4]
    assert data == {1: 2, 2: 3, 3: 4, 4: 5}

    ordered = keyfilter(
        lambda key: key != "b",
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("a", 1), ("c", 3)]

    assert keyfilter(lambda key: True, {}) == {}


def test_itemfilter_basic_factory_and_predicate_receives_items():
    data = {1: 2, 2: 3, 3: 4, 4: 5}
    seen = []

    def is_valid(item):
        seen.append(item)
        key, value = item
        return key % 2 == 0 and value < 4

    result = itemfilter(is_valid, data)
    assert result == {2: 3}
    assert seen == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert data == {1: 2, 2: 3, 3: 4, 4: 5}

    ordered = itemfilter(
        lambda item: item[1] > 1,
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("b", 2), ("c", 3)]

    assert itemfilter(lambda item: True, {}) == {}


def test_assoc_replaces_adds_and_does_not_mutate():
    original = {"x": 1}

    replaced = assoc(original, "x", 2)
    assert replaced == {"x": 2}
    assert replaced is not original
    assert original == {"x": 1}

    added = assoc(original, "y", 3)
    assert added == {"x": 1, "y": 3}
    assert original == {"x": 1}

    ordered = assoc(
        OrderedDict([("a", 1)]),
        "b",
        2,
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("a", 1), ("b", 2)]


def test_dissoc_update_branch_and_missing_keys():
    data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    original = dict(data)

    result = dissoc(data, "a", "b")
    assert result == {"c": 3, "d": 4, "e": 5}
    assert result is not data
    assert data == original

    result_missing = dissoc(data, "a", "missing")
    assert result_missing == {"b": 2, "c": 3, "d": 4, "e": 5}
    assert data == original


def test_dissoc_remaining_set_branch_and_missing_keys():
    data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    original = dict(data)

    result = dissoc(data, "a", "b", "c")
    assert result == {"d": 4, "e": 5}
    assert data == original

    result_missing = dissoc(data, "a", "b", "missing")
    assert result_missing == {"c": 3, "d": 4, "e": 5}
    assert data == original


def test_dissoc_empty_no_keys_factory_and_unexpected_keyword():
    data = {"a": 1, "b": 2}

    no_keys = dissoc(data)
    assert no_keys == data
    assert no_keys is not data

    assert dissoc({}) == {}
    assert dissoc({}, "x") == {}
    assert dissoc({"x": 1}, "y") == {"x": 1}

    ordered = dissoc(
        OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        "b",
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("a", 1), ("c", 3)]

    ordered_set_branch = dissoc(
        OrderedDict([("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]),
        "a",
        "b",
        "c",
        factory=OrderedDict,
    )
    assert isinstance(ordered_set_branch, OrderedDict)
    assert ordered_set_branch == {"d": 4, "e": 5}

    with pytest.raises(
        TypeError,
        match=r"dissoc\(\) got an unexpected keyword argument 'bad'",
    ):
        dissoc({"a": 1}, "a", bad=True)


def test_assoc_in_existing_and_missing_paths():
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
    assert result["order"] is not purchase["order"]
    assert result["order"]["items"] is purchase["order"]["items"]
    assert purchase["order"]["costs"] == [0.50, 1.25]

    assert assoc_in({}, ["a", "b"], 5) == {"a": {"b": 5}}
    assert assoc_in({"a": {"b": 1, "c": 2}}, ["a", "b"], 10) == {
        "a": {"b": 10, "c": 2}
    }


def test_assoc_in_factory_and_empty_keys():
    result = assoc_in({}, ["a", "b"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], OrderedDict)
    assert result == {"a": {"b": 1}}

    with pytest.raises(StopIteration):
        assoc_in({}, [], 1)


def test_update_in_existing_value():
    original = {"a": 0, "b": 5}

    result = update_in(original, ["a"], lambda value: value + 1)
    assert result == {"a": 1, "b": 5}
    assert result is not original
    assert original == {"a": 0, "b": 5}


def test_update_in_nested_existing_path_and_shallow_copying():
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
    assert result["purchase"] is not transaction["purchase"]
    assert result["purchase"]["items"] is transaction["purchase"]["items"]
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_missing_paths_and_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: "bar"}}
    }

    assert update_in({1: "foo"}, [2, 3, 4], lambda value: value + 1, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }

    assert update_in({}, ["a"], lambda value: value + 1, default=10) == {
        "a": 11
    }
    assert update_in({"a": 5}, ["a"], lambda value: value + 1, default=100) == {
        "a": 6
    }


def test_update_in_factory():
    original = OrderedDict([("a", OrderedDict([("b", 1)]))])

    result = update_in(
        original,
        ["a", "b"],
        lambda value: value + 1,
        factory=OrderedDict,
    )

    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], OrderedDict)
    assert result == {"a": {"b": 2}}
    assert original == {"a": {"b": 1}}

    missing = update_in(
        {},
        ["a", "b"],
        lambda value: value,
        default=7,
        factory=OrderedDict,
    )
    assert isinstance(missing, OrderedDict)
    assert isinstance(missing["a"], OrderedDict)
    assert missing == {"a": {"b": 7}}


def test_update_in_empty_keys_and_non_mapping_intermediate():
    with pytest.raises(StopIteration):
        update_in({}, [], lambda value: value)

    with pytest.raises(TypeError):
        update_in({"a": 1}, ["a", "b"], lambda value: value)


def test_get_in_successful_lookup():
    transaction = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
        "credit card": "5555-1234-1234-1234",
    }

    assert get_in(["purchase", "items", 0], transaction) == "Apple"
    assert get_in(["purchase", "costs", 1], transaction) == 1.25
    assert get_in(["name"], transaction) == "Alice"
    assert get_in([], transaction) is transaction
    assert get_in(["name"], transaction, default="fallback") == "Alice"
    assert get_in(["name"], transaction, no_default=True) == "Alice"


def test_get_in_missing_values_return_default():
    transaction = {
        "name": "Alice",
        "purchase": {
            "items": ["Apple", "Orange"],
            "costs": [0.50, 1.25],
        },
    }

    assert get_in(["purchase", "total"], transaction) is None
    assert get_in(["purchase", "total"], transaction, default=0) == 0
    assert get_in(["purchase", "items", "apple"], transaction) is None
    assert get_in(["purchase", "items", 10], transaction) is None
    assert get_in(["missing"], transaction, default="fallback") == "fallback"


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)

    with pytest.raises(IndexError):
        get_in([3], [1, 2], no_default=True)

    with pytest.raises(TypeError):
        get_in(["a", "b"], {"a": 1}, no_default=True)
