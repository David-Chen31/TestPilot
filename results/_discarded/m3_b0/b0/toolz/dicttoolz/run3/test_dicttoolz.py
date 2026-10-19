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


def test_merge_basic_and_iterable_inputs():
    assert merge() == {}
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    assert merge({1: 1}) == {1: 1}
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}
    assert merge(({1: 1}, {1: 2}, {3: 3})) == {1: 2, 3: 3}
    assert merge(iter([{"a": 1}, {"b": 2}])) == {"a": 1, "b": 2}


def test_merge_factory_and_bad_kwargs():
    result = merge({"a": 1}, {"b": 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1), ("b", 2)]

    with pytest.raises(
        TypeError, match=r"merge\(\) got an unexpected keyword argument 'bogus'"
    ):
        merge({}, bogus=True)


def test_merge_with_basic_and_iterable_inputs():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(
        sum, {"a": 1, "b": 2}, {"b": 3, "c": 4}
    ) == {"a": 1, "b": 5, "c": 4}
    assert merge_with(sum) == {}
    assert merge_with(sum, {}) == {}
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_function_receives_value_lists_and_factory():
    calls = []

    def func(values):
        calls.append(values)
        return len(values)

    result = merge_with(func, {"a": 1, "b": 2}, {"a": 3})
    assert result == {"a": 2, "b": 1}
    assert calls == [[1, 3], [2]]

    result = merge_with(sum, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 1)]

    with pytest.raises(
        TypeError,
        match=r"merge_with\(\) got an unexpected keyword argument 'bogus'",
    ):
        merge_with(sum, {}, bogus=True)


def test_valmap():
    original = {"a": 1, "b": 2}
    result = valmap(lambda x: x + 1, original)
    assert result == {"a": 2, "b": 3}
    assert original == {"a": 1, "b": 2}
    assert valmap(lambda x: x * 10, {}) == {}

    ordered = valmap(lambda x: x * 2, {"a": 1, "b": 2}, factory=OrderedDict)
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("a", 2), ("b", 4)]


def test_keymap():
    original = {"a": 1, "b": 2}
    assert keymap(str.upper, original) == {"A": 1, "B": 2}
    assert original == {"a": 1, "b": 2}
    assert keymap(lambda k: k % 2, {1: "one", 3: "three"}) == {1: "three"}
    assert keymap(str.upper, {}) == {}

    ordered = keymap(str.upper, {"a": 1, "b": 2}, factory=OrderedDict)
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("A", 1), ("B", 2)]


def test_itemmap():
    original = {"a": 1, "b": 2}
    assert itemmap(lambda item: (item[1], item[0]), original) == {
        1: "a",
        2: "b",
    }
    assert original == {"a": 1, "b": 2}
    assert itemmap(reversed, {"a": 1, "b": 2}) == {1: "a", 2: "b"}
    assert itemmap(lambda item: (item[0] % 2, item[1]), {1: "a", 3: "b"}) == {
        1: "b"
    }
    assert itemmap(lambda item: item, {}) == {}

    ordered = itemmap(
        lambda item: (item[0].upper(), item[1]),
        {"a": 1, "b": 2},
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("A", 1), ("B", 2)]


def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    result = valfilter(lambda v: v % 2 == 0, d)
    assert result == {1: 2, 3: 4}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda v: True, {}) == {}

    ordered = valfilter(lambda v: v > 1, {"a": 1, "b": 2}, factory=OrderedDict)
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("b", 2)]


def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda k: k % 2 == 0, d) == {2: 3, 4: 5}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda k: True, {}) == {}

    ordered = keyfilter(lambda k: k == "b", {"a": 1, "b": 2}, factory=OrderedDict)
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("b", 2)]


def test_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, d) == {2: 3}
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(lambda item: item[1] == 2, {"a": 1, "b": 2}) == {
        "b": 2
    }
    assert itemfilter(lambda item: True, {}) == {}

    ordered = itemfilter(
        lambda item: item[0] % 2 == 0,
        {1: 10, 2: 20, 3: 30},
        factory=OrderedDict,
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [(2, 20)]


def test_assoc():
    d = {"x": 1, "y": 2}
    result = assoc(d, "x", 10)
    assert result == {"x": 10, "y": 2}
    assert d == {"x": 1, "y": 2}
    assert assoc({"x": 1}, "y", 2) == {"x": 1, "y": 2}

    ordered = assoc(
        OrderedDict([("a", 1), ("b", 2)]), "a", 3, factory=OrderedDict
    )
    assert isinstance(ordered, OrderedDict)
    assert list(ordered.items()) == [("a", 3), ("b", 2)]


def test_dissoc_basic_copy_and_missing_keys():
    d = {"x": 1, "y": 2, "z": 3}
    result = dissoc(d, "y")
    assert result == {"x": 1, "z": 3}
    assert d == {"x": 1, "y": 2, "z": 3}
    assert dissoc({"x": 1}, "y") == {"x": 1}
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}
    assert dissoc({}) == {}


def test_dissoc_remaining_set_branch():
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    result = dissoc(d, "a", "c", "e")
    assert result == {"b": 2, "d": 4}
    assert d == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    assert dissoc({"a": 1, "b": 2, "c": 3}, "a", "x", "y") == {
        "b": 2,
        "c": 3,
    }


def test_dissoc_factory_and_bad_kwargs():
    result = dissoc({"a": 1, "b": 2, "c": 3}, "a", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("b", 2), ("c", 3)]

    with pytest.raises(
        TypeError, match=r"dissoc\(\) got an unexpected keyword argument 'bogus'"
    ):
        dissoc({"a": 1}, "a", bogus=True)


def test_assoc_in_updates_existing_and_creates_missing_paths():
    original = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(original, ["order", "costs"], [0.25, 1.00])
    expected = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.25, 1.00]},
        "credit card": "5555-1234-1234-1234",
    }
    assert result == expected
    assert original["order"]["costs"] == [0.50, 1.25]
    assert result["order"] is not original["order"]

    assert assoc_in({}, ["a", "b", "c"], 1) == {"a": {"b": {"c": 1}}}
    assert assoc_in({"a": 1}, ["b", "c"], 2) == {"a": 1, "b": {"c": 2}}


def test_assoc_in_factory_and_empty_keys():
    result = assoc_in(
        OrderedDict([("a", 1)]), ["b", "c"], 2, factory=OrderedDict
    )
    assert isinstance(result, OrderedDict)
    assert isinstance(result["b"], OrderedDict)
    assert result == {"a": 1, "b": {"c": 2}}
    assert list(result.keys()) == ["a", "b"]

    with pytest.raises(StopIteration):
        assoc_in({}, [], 1)


def test_update_in_existing_nested_value_is_non_mutating():
    original = {"a": {"b": 1}, "x": 10}
    result = update_in(original, ["a", "b"], lambda x: x + 1)

    assert result == {"a": {"b": 2}, "x": 10}
    assert original == {"a": {"b": 1}, "x": 10}
    assert result is not original
    assert result["a"] is not original["a"]


def test_update_in_single_key_and_defaults():
    assert update_in({"a": 1}, ["a"], lambda x: x * 2) == {"a": 2}
    assert update_in({}, ["a"], lambda x: x + 1, default=5) == {"a": 6}
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: "bar"}}
    }
    assert update_in({1: "foo"}, [2, 3, 4], lambda x: x + 1, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }


def test_update_in_mixed_existing_and_missing_paths():
    original = {"a": {"b": 1}}
    calls = []

    def inc(x):
        calls.append(x)
        return x + 1

    result = update_in(original, ["a", "c"], inc, default=10)

    assert result == {"a": {"b": 1, "c": 11}}
    assert original == {"a": {"b": 1}}
    assert calls == [10]
    assert result["a"] is not original["a"]


def test_update_in_factory_and_empty_keys():
    result = update_in(
        OrderedDict([("a", 1)]),
        ["b", "c"],
        lambda x: x + 1,
        default=0,
        factory=OrderedDict,
    )
    assert isinstance(result, OrderedDict)
    assert isinstance(result["b"], OrderedDict)
    assert result == {"a": 1, "b": {"c": 1}}

    with pytest.raises(StopIteration):
        update_in({}, [], lambda x: x)


def test_get_in_nested_lookup_and_empty_keys():
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


def test_get_in_defaults_and_no_default_errors():
    transaction = {"purchase": {"items": ["Apple", "Orange"]}}
    sentinel = object()

    assert get_in(["purchase", "total"], transaction) is None
    assert get_in(["purchase", "items", "apple"], transaction) is None
    assert get_in(["purchase", "items", 10], transaction) is None
    assert get_in(["missing"], transaction, sentinel) is sentinel
    assert get_in(["purchase", "total"], transaction, 0) == 0

    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)

    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)

    with pytest.raises(TypeError):
        get_in(["a", "b"], {"a": 1}, no_default=True)
