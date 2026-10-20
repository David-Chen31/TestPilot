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


def test_merge_single_iterable_of_dicts():
    dicts = [{1: 2}, {3: 4}, {5: 6}]
    assert merge(dicts) == {1: 2, 3: 4, 5: 6}


def test_merge_no_dicts():
    assert merge() == {}


def test_merge_single_mapping_unwrapped():
    # single dict is treated as one dict, not iterable of dicts
    assert merge({1: 2}) == {1: 2}


def test_merge_factory():
    from collections import OrderedDict

    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}
    assert list(result) == [1, 3]


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError) as excinfo:
        merge({1: 2}, foo=1)
    assert "foo" in str(excinfo.value)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_values_are_lists():
    seen = {}

    def capture(values):
        seen["values"] = list(values)
        return sum(values)

    result = merge_with(capture, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 22, 3: 30}
    assert seen["values"] == [2, 20]


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}, {1: 3}]) == {1: 6}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single_value_single_dict():
    result = merge_with(sum, {1: 5})
    assert result == {1: 5}


def test_merge_with_factory():
    from collections import OrderedDict

    result = merge_with(sum, {1: 1}, {2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1, 2: 2}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad=1)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_preserves_keys():
    d = {"a": 1, "b": 2}
    result = valmap(lambda v: v * 10, d)
    assert result == {"a": 10, "b": 20}
    assert d == {"a": 1, "b": 2}  # unchanged


def test_valmap_factory():
    from collections import OrderedDict

    result = valmap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: "2"}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30],
        "bob": [10, 35],
    }


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_does_not_modify():
    d = {"a": 1}
    assert keymap(str.upper, d) == {"A": 1}
    assert d == {"a": 1}


def test_keymap_factory():
    from collections import OrderedDict

    result = keymap(str.upper, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"A": 1}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda item: (item[1], item[0]), accountids) == {
        10: "Alice",
        20: "Bob",
    }


def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_empty():
    assert itemmap(reversed, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict

    result = itemmap(reversed, {"a": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: "a"}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_none_match():
    assert valfilter(lambda x: x > 100, {1: 2}) == {}


def test_valfilter_empty():
    assert valfilter(bool, {}) == {}


def test_valfilter_factory():
    from collections import OrderedDict

    result = valfilter(lambda v: v > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_none_match():
    assert keyfilter(lambda k: k > 100, {1: 2}) == {}


def test_keyfilter_empty():
    assert keyfilter(bool, {}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict

    result = keyfilter(lambda k: k > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_none_match():
    assert itemfilter(lambda item: False, {1: 2}) == {}


def test_itemfilter_empty():
    assert itemfilter(bool, {}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict

    result = itemfilter(lambda item: item[0] > 1, {1: 1, 2: 2},
                        factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


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
    assert d == {"x": 1}
    assert result == {"x": 2}
    assert result is not d


def test_assoc_empty():
    assert assoc({}, "a", 1) == {"a": 1}


def test_assoc_factory():
    from collections import OrderedDict

    result = assoc({"a": 1}, "b", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single_key():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple_keys():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2, "z": 3, "w": 4}
    result = dissoc(d, "x")
    assert d == {"x": 1, "y": 2, "z": 3, "w": 4}
    assert result == {"y": 2, "z": 3, "w": 4}


def test_dissoc_alternative_branch():
    # Many keys triggers the "remaining set" code path.
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(1, 10))
    assert result == {0: 0}


def test_dissoc_factory():
    from collections import OrderedDict

    result = dissoc({"a": 1, "b": 2}, "a", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"b": 2}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({"a": 1}, "a", bogus=1)


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
    # original unchanged
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    result = assoc_in({}, ["a", "b", "c"], 5)
    assert result == {"a": {"b": {"c": 5}}}


def test_assoc_in_single_key():
    assert assoc_in({"a": 1}, ["a"], 2) == {"a": 2}


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
    assert update_in({}, [1, 2, 3], str, default="bar") == {
        1: {2: {3: "bar"}}
    }


def test_update_in_missing_keys_with_inc_default():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}
    result = update_in(d, ["a", "b"], lambda x: x + 1)
    assert d == {"a": {"b": 1}}
    assert result == {"a": {"b": 2}}


def test_update_in_factory():
    from collections import OrderedDict

    result = update_in({}, ["a", "b"], str, default="x", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], OrderedDict)
    assert result == {"a": {"b": "x"}}


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
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }


def test_get_in_nested_list(transaction):
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_single(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_key_default_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_bad_index(transaction):
    # indexing a list with a string -> TypeError, returned as default
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_out_of_range(transaction):
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises_key_error(transaction):
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_raises_index_error(transaction):
    with pytest.raises(IndexError):
        get_in([10], [1, 2, 3], no_default=True)


def test_get_in_no_default_raises_type_error():
    with pytest.raises(TypeError):
        get_in(["a"], {1: 2}, no_default=True)


def test_get_in_empty_keys_returns_coll():
    coll = {"a": 1}
    assert get_in([], coll) is coll
