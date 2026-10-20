import pytest
from collections import OrderedDict, defaultdict

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


def test_merge_sequence_of_dicts():
    # single non-Mapping argument is treated as an iterable of dicts
    assert merge([{1: 2}, {2: 3}]) == {1: 2, 2: 3}


def test_merge_tuple_of_dicts():
    assert merge(({1: 2}, {3: 4})) == {1: 2, 3: 4}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_values_lists():
    result = merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_sequence_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x + 1, {1: 2, 3: 4}) == {1: 3, 3: 5}


def test_valmap_does_not_mutate():
    d = {1: 2}
    result = valmap(lambda x: x * 10, d)
    assert d == {1: 2}
    assert result == {1: 20}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keymap():
    assert keymap(lambda x: x + 1, {1: 2, 3: 4}) == {2: 2, 4: 4}


def test_keymap_str_lower():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_factory():
    result = keymap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemmap():
    assert itemmap(reversed, {"Alice": 10, "Bob": 20}) == {10: "Alice", 20: "Bob"}


def test_itemmap_identity():
    d = {1: 2, 3: 4}
    assert itemmap(lambda item: item, d) == d


def test_itemmap_factory():
    result = itemmap(lambda item: item, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

def test_valfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: v == 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_keyfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


# ---------------------------------------------------------------------------
# assoc / dissoc
# ---------------------------------------------------------------------------

def test_assoc_update_existing():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_add_new():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_mutate():
    d = {"x": 1}
    result = assoc(d, "y", 3)
    assert d == {"x": 1}
    assert result == {"x": 1, "y": 3}
    assert result is not d


def test_assoc_factory():
    result = assoc({"x": 1}, "y", 3, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_dissoc_one():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    dissoc(d, "y")
    assert d == {"x": 1, "y": 2}


def test_dissoc_factory():
    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "y", bad_kwarg=True)


def test_dissoc_many_keys_branch():
    # Exercise the branch where len(keys) >= len(d) * .6
    d = {i: i for i in range(10)}
    result = dissoc(d, *range(10))
    assert result == {}


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------

def test_assoc_in_nested_existing():
    d = {"order": {"costs": [0.50, 1.25], "items": ["Apple"]}}
    result = assoc_in(d, ["order", "costs"], [0.25, 1.00])
    assert result == {"order": {"costs": [0.25, 1.00], "items": ["Apple"]}}
    # original unchanged
    assert d == {"order": {"costs": [0.50, 1.25], "items": ["Apple"]}}


def test_assoc_in_creates_nested():
    result = assoc_in({}, [1, 2, 3], "bar")
    assert result == {1: {2: {3: "bar"}}}


def test_assoc_in_factory():
    result = assoc_in({}, ["a", "b"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    transaction = {"purchase": {"costs": [0.50, 1.25]}}
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result == {"purchase": {"costs": 1.75}}
    # original not mutated
    assert transaction == {"purchase": {"costs": [0.50, 1.25]}}


def test_update_in_missing_uses_default():
    result = update_in({}, [1, 2, 3], str, default="bar")
    assert result == {1: {2: {3: "bar"}}}


def test_update_in_missing_with_numeric_default():
    inc = lambda x: x + 1
    result = update_in({1: "foo"}, [2, 3, 4], inc, 0)
    assert result == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_does_not_mutate():
    d = {"a": {"b": 1}}
    update_in(d, ["a", "b"], lambda x: x + 1)
    assert d == {"a": {"b": 1}}


def test_update_in_factory():
    result = update_in({}, ["a", "b"], lambda x: x, default=1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


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


def test_get_in_top_level(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_key_default_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_bad_index_type(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_index_error():
    with pytest.raises(IndexError):
        get_in([0], [], no_default=True)


def test_get_in_no_default_found(transaction):
    assert get_in(["name"], transaction, no_default=True) == "Alice"


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) is transaction
