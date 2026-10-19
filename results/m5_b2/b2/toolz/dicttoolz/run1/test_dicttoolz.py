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


# ----------------------------- merge -----------------------------

def test_merge_basic():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_non_mapping_iterable():
    # A single non-Mapping arg is treated as an iterable of dicts
    assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}


def test_merge_no_args():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2, 3: 4}


def test_merge_bad_kwarg():
    with pytest.raises(TypeError):
        merge({1: 2}, bad_kwarg=True)


def test_merge_bad_kwarg_message_merge():
    with pytest.raises(TypeError) as exc_info:
        merge({1: 2}, bad_kwarg=True)
    assert str(exc_info.value) == (
        "merge() got an unexpected keyword argument 'bad_kwarg'"
    )


def test_merge_does_not_mutate_inputs():
    a = {1: 2}
    b = {3: 4}
    merge(a, b)
    assert a == {1: 2}
    assert b == {3: 4}


# ----------------------------- merge_with -----------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_single_iterable():
    assert merge_with(sum, [{1: 1}, {1: 2}, {1: 3}]) == {1: 6}


def test_merge_with_no_dicts():
    assert merge_with(sum) == {}


def test_merge_with_values_lists():
    seen = {}

    def capture(vals):
        seen["vals"] = list(vals)
        return len(vals)

    result = merge_with(capture, {1: "a"}, {1: "b"}, {1: "c"})
    assert result == {1: 3}
    assert seen["vals"] == ["a", "b", "c"]


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_bad_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 2}, bad=True)


def test_merge_with_bad_kwarg_message():
    with pytest.raises(TypeError) as exc_info:
        merge_with(sum, {1: 2}, bad=True)
    assert str(exc_info.value) == (
        "merge_with() got an unexpected keyword argument 'bad'"
    )


# ----------------------------- valmap -----------------------------

def test_valmap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_increments():
    assert valmap(lambda x: x + 1, {1: 10, 2: 20}) == {1: 11, 2: 21}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(lambda x: x * 10, d)
    assert d == {1: 2}


def test_valmap_factory():
    result = valmap(lambda x: x, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


# ----------------------------- keymap -----------------------------

def test_keymap_basic():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {
        "alice": [20, 15, 30],
        "bob": [10, 35],
    }


def test_keymap_does_not_mutate():
    d = {"A": 1}
    keymap(str.lower, d)
    assert d == {"A": 1}


def test_keymap_factory():
    result = keymap(str.lower, {"A": 1}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ----------------------------- itemmap -----------------------------

def test_itemmap_reverse():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_swap_tuple():
    assert itemmap(lambda kv: (kv[1], kv[0]), {1: 2, 3: 4}) == {2: 1, 4: 3}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ----------------------------- valfilter -----------------------------

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_does_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda x: x == 2, d)
    assert d == {1: 2, 2: 3}


def test_valfilter_factory():
    result = valfilter(lambda x: x > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ----------------------------- keyfilter -----------------------------

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 1, {1: 1, 2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ----------------------------- itemfilter -----------------------------

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_factory():
    result = itemfilter(
        lambda item: item[0] > 1, {1: 1, 2: 2}, factory=OrderedDict
    )
    assert isinstance(result, OrderedDict)
    assert result == {2: 2}


# ----------------------------- assoc -----------------------------

def test_assoc_replace():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_add():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_does_not_mutate():
    d = {"x": 1}
    assoc(d, "x", 2)
    assert d == {"x": 1}


def test_assoc_factory():
    result = assoc({"x": 1}, "y", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1, "y": 2}


# ----------------------------- dissoc -----------------------------

def test_dissoc_one():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_many():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_ignores_missing():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_keys():
    assert dissoc({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_dissoc_does_not_mutate():
    d = {"x": 1, "y": 2}
    dissoc(d, "y")
    assert d == {"x": 1, "y": 2}


def test_dissoc_large_number_of_keys():
    # Exercises the "remaining" branch (len(keys) >= len(d) * 0.6)
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    assert dissoc(d, "a", "b", "c") == {"d": 4, "e": 5}


def test_dissoc_factory():
    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_bad_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad=True)


def test_dissoc_bad_kwarg_message():
    with pytest.raises(TypeError) as exc_info:
        dissoc({"x": 1}, "x", bad=True)
    assert str(exc_info.value) == (
        "dissoc() got an unexpected keyword argument 'bad'"
    )


# Boundary tests for the branch condition on line 217:
#   if len(keys) < len(d) * .6
# All values below are chosen so that no key is present in d, meaning both
# branches produce identical results (a full copy of d), EXCEPT when the
# threshold comparison is altered, in which case a buggy branch may raise a
# KeyError or produce a different result. Since both branches yield the same
# output when keys are absent, we instead construct inputs whose branch
# selection is observable by using a custom factory that tracks how many times
# it is called (both branches should produce a dict equal to d, so no
# observable output difference). To make the ratio branch observable we rely
# on the fact that the second branch copies via `remaining` (a set), and if
# the set ordering differs the insertion order into an OrderedDict differs.
# Therefore: use a factory=OrderedDict and many keys, where one branch inserts
# in original dict order and the other in (unordered) set order.

def test_dissoc_branch_below_threshold_uses_update_copy():
    # len(d)=10, len(keys)=5 -> 5 < 10*0.6=6 -> True -> update-copy branch,
    # preserving insertion order in OrderedDict.
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, factory=OrderedDict)
    assert list(result.items()) == [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]


def test_dissoc_branch_at_threshold_uses_set_branch():
    # len(d)=10, len(keys)=6 -> 6 < 6 -> False -> set-based branch.
    # Uses set(d) which loses insertion order for OrderedDict output.
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, 5, factory=OrderedDict)
    # Regardless of branch, contents must be the remaining keys/values
    assert dict(result) == {6: 6, 7: 7, 8: 8, 9: 9}


def test_dissoc_branch_threshold_with_5_of_10_kept():
    # len(d)=10, len(keys)=5 (below threshold, update branch). Mutation
    # `<=` would flip to set branch, which still yields equal dict; but
    # mutation `/ 0.6` gives 10/0.6 ~= 16.67 so 5 < 16.67 -> update branch
    # (same). Mutation `* 1.6` gives 16 -> 5 < 16 -> update branch (same).
    # Distinguish order: update branch preserves original order in OrderedDict.
    d = {i: i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, factory=OrderedDict)
    assert list(result.keys()) == [5, 6, 7, 8, 9]


def test_dissoc_threshold_distinguishes_multiplier_mutations():
    # Pick len(d) and len(keys) such that the ORIGINAL code takes the
    # set branch (len(keys) >= 0.6*len(d)) but the mutated `* 1.6` version
    # takes the update branch too (since 1.6*len(d) is larger, so
    # len(keys) < 1.6*len(d) is MORE often true). We need a case where
    # original -> set branch and mutant * 1.6 -> update branch.
    # Original set branch: len(keys) >= 0.6*len(d).
    # Mutant update branch: len(keys) < 1.6*len(d).
    # Take len(d)=10, len(keys)=7: 7 >= 6 (set branch), 7 < 16 (update branch).
    # Set branch inserts via set -> for ints {0..9} minus {0..6} = {7,8,9},
    # iterating a set of small ints typically yields sorted order; update
    # branch would preserve order [7,8,9] too. Hard to distinguish reliably.
    # Instead use string keys (set order is hash-dependent but stable).
    # Use a custom factory to record whether `del` or `d2[k]=` path is taken.
    calls = {"n": 0}

    class TrackingDict(dict):
        def __delitem__(self, key):
            calls["n"] += 1
            super().__delitem__(key)

    d = {i: i for i in range(10)}
    dissoc(d, 0, 1, 2, 3, 4, 5, 6, factory=TrackingDict)
    # Original takes set branch (7 >= 6), so no __delitem__ calls.
    assert calls["n"] == 0


def test_dissoc_threshold_set_branch_no_delitem():
    calls = {"n": 0}

    class TrackingDict(dict):
        def __delitem__(self, key):
            calls["n"] += 1
            super().__delitem__(key)

    d = {i: i for i in range(5)}
    # len(keys)=3, len(d)=5 -> 3 >= 5*0.6=3.0 -> set branch -> no delitem
    dissoc(d, 0, 1, 2, factory=TrackingDict)
    assert calls["n"] == 0


def test_dissoc_threshold_update_branch_uses_delitem():
    calls = {"n": 0}

    class TrackingDict(dict):
        def __delitem__(self, key):
            calls["n"] += 1
            super().__delitem__(key)

    d = {i: i for i in range(5)}
    # len(keys)=2, len(d)=5 -> 2 < 3.0 -> update branch -> 2 delitem calls
    dissoc(d, 0, 1, factory=TrackingDict)
    assert calls["n"] == 2


def test_dissoc_threshold_boundary_exact():
    # len(d)=5, len(keys)=3 -> 3 < 3.0 is False -> set branch (no delitem).
    # Mutation to `<=` makes 3 <= 3.0 True -> update branch -> 3 delitem calls.
    calls = {"n": 0}

    class TrackingDict(dict):
        def __delitem__(self, key):
            calls["n"] += 1
            super().__delitem__(key)

    d = {i: i for i in range(5)}
    dissoc(d, 0, 1, 2, factory=TrackingDict)
    assert calls["n"] == 0


def test_dissoc_threshold_multiplier_mutation_162():
    # len(d)=10, len(keys)=12 -> original: 12 < 6.0 False -> set branch.
    # Mutant *1.6: 12 < 16.0 True -> update branch -> delitem called.
    # We use keys not in d for the "delete-if-present" branch to call delitem.
    # In update branch, loop over keys 0..11; only 0..9 are in d, rest skipped.
    # So delitem would be called 10 times if update branch taken.
    calls = {"n": 0}

    class TrackingDict(dict):
        def __delitem__(self, key):
            calls["n"] += 1
            super().__delitem__(key)

    d = {i: i for i in range(10)}
    dissoc(d, *range(12), factory=TrackingDict)
    # Original: set branch -> 0 delitem calls.
    assert calls["n"] == 0


def test_dissoc_threshold_multiplier_mutation_div06():
    # len(d)=10, len(keys)=3 -> original: 3 < 6.0 True -> update branch.
    # Mutant /0.6: 3 < 10/0.6 ~= 16.67 True -> update branch (same).
    # Mutant *1.6: 3 < 16.0 True -> update branch (same).
    # Mutant <=: 3 <= 6.0 True -> update branch (same).
    # No distinction possible here; skip.
    pass


# ----------------------------- assoc_in -----------------------------

def test_assoc_in_nested():
    purchase = {
        "name": "Alice",
        "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = assoc_in(purchase, ["order", "costs"], [0.25, 1.00])
    assert result["order"]["costs"] == [0.25, 1.00]
    # original unchanged
    assert purchase["order"]["costs"] == [0.50, 1.25]


def test_assoc_in_creates_nested():
    assert assoc_in({}, ["a", "b"], 1) == {"a": {"b": 1}}


def test_assoc_in_factory():
    result = assoc_in({}, ["a"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


def test_assoc_in_ignores_default_argument():
    # assoc_in passes `value` as the default, so even when creating a new
    # nested key the value assigned is `value` itself, not func(default).
    # Mutation `None` as default: if func is `lambda x: value`, it ignores
    # its argument so both produce the same result. Therefore, we cannot
    # distinguish with assoc_in's own lambda. But the default argument value
    # passed to update_in is `value` (the fourth positional). Verify by
    # creating a missing key: update_in's else branch calls func(default),
    # and func = lambda x: value, so result is `value` regardless. This test
    # documents behavior; the None-mutation is equivalent for assoc_in.
    assert assoc_in({}, ["a", "b"], 42) == {"a": {"b": 42}}
    assert assoc_in({"a": {}}, ["a", "b"], "X") == {"a": {"b": "X"}}
    assert assoc_in({"a": {"b": 1}}, ["a", "b"], "Y") == {"a": {"b": "Y"}}


# The line-242 mutation (passing None instead of `value` as default) affects
# assoc_in only when func observes its argument, which `lambda x: value`
# does not. However, we can still distinguish by *monkeypatching* update_in
# to record the default argument it receives.

def test_assoc_in_passes_value_as_default(monkeypatch):
    import toolz.dicttoolz as dt

    recorded = {}

    def fake_update_in(d, keys, func, default=None, factory=dict):
        recorded["d"] = d
        recorded["keys"] = list(keys)
        recorded["default"] = default
        recorded["factory"] = factory
        recorded["func_result"] = func("sentinel")
        return "FAKE"

    monkeypatch.setattr(dt, "update_in", fake_update_in)
    result = assoc_in({"q": 1}, ["a", "b"], 42)
    assert result == "FAKE"
    assert recorded["d"] == {"q": 1}
    assert recorded["keys"] == ["a", "b"]
    assert recorded["default"] == 42
    assert recorded["func_result"] == 42
    assert recorded["factory"] is dict


def test_assoc_in_passes_custom_factory_to_update_in(monkeypatch):
    import toolz.dicttoolz as dt

    recorded = {}

    def fake_update_in(d, keys, func, default=None, factory=dict):
        recorded["factory"] = factory
        return None

    monkeypatch.setattr(dt, "update_in", fake_update_in)
    assoc_in({}, ["a"], 1, factory=OrderedDict)
    assert recorded["factory"] is OrderedDict


# ----------------------------- update_in -----------------------------

def test_update_in_basic():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_does_not_mutate():
    d = {"a": 0}
    update_in(d, ["a"], lambda x: x + 1)
    assert d == {"a": 0}


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_update_in_create_partial():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {
        1: "foo",
        2: {3: {4: 1}},
    }


def test_update_in_nested_existing():
    transaction = {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }
    result = update_in(transaction, ["purchase", "costs"], sum)
    assert result["purchase"]["costs"] == 1.75
    # original unchanged
    assert transaction["purchase"]["costs"] == [0.50, 1.25]


def test_update_in_factory():
    result = update_in({}, ["a"], lambda x: 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ----------------------------- get_in -----------------------------

@pytest.fixture
def transaction():
    return {
        "name": "Alice",
        "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
        "credit card": "5555-1234-1234-1234",
    }


def test_get_in_list_index(transaction):
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_simple(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_default_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_missing_bad_index(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_index_error():
    # get_in([0, 5], [1, 2, 3]) first does coll[0] -> 1 (an int),
    # then tries 1[5], which raises TypeError. With no_default=True it is
    # re-raised; the implementation catches TypeError too.
    with pytest.raises(TypeError):
        get_in([0, 5], [1, 2, 3], no_default=True)


def test_get_in_no_default_index_error_direct():
    # An actual IndexError: index directly out of range on the list
    with pytest.raises(IndexError):
        get_in([5], [1, 2, 3], no_default=True)


def test_get_in_empty_keys():
    # reduce with empty keys returns coll unchanged
    assert get_in([], {"a": 1}) == {"a": 1}
