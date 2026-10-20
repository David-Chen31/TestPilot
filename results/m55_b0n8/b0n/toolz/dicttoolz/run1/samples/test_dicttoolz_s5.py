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

def test_merge_empty():
    assert merge() == {}


def test_merge_single():
    assert merge({1: "one"}) == {1: "one"}


def test_merge_multiple():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_list_of_dicts():
    assert merge([{1: "one"}, {2: "two"}]) == {1: "one", 2: "two"}


def test_merge_does_not_mutate_inputs():
    a = {1: 1}
    b = {2: 2}
    merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}


def test_merge_factory():
    from collections import OrderedDict

    result = merge({1: 1}, {2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1, 2: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge({1: 1}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_single():
    assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_with_list_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}


def test_merge_with_passes_list():
    seen = {}

    def capture(values):
        seen["values"] = list(values)
        return sum(values)

    result = merge_with(capture, {1: 1}, {1: 2, 2: 3})
    assert result == {1: 3, 2: 3}
    assert seen["values"] in ([1, 2], [2, 1])


def test_merge_with_factory():
    from collections import OrderedDict

    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bad_kwarg=True)


# ---------------------------------------------------------------------------
# valmap
# ---------------------------------------------------------------------------

def test_valmap():
    assert valmap(lambda x: x * 2, {1: 2, 3: 4}) == {1: 4, 3: 8}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    from collections import OrderedDict

    result = valmap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: "2"}


# ---------------------------------------------------------------------------
# keymap
# ---------------------------------------------------------------------------

def test_keymap():
    assert keymap(lambda k: k + 1, {1: "a", 2: "b"}) == {2: "a", 3: "b"}


def test_keymap_empty():
    assert keymap(lambda k: k, {}) == {}


def test_keymap_factory():
    from collections import OrderedDict

    result = keymap(str, {1: "a"}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"1": "a"}


# ---------------------------------------------------------------------------
# itemmap
# ---------------------------------------------------------------------------

def test_itemmap():
    assert itemmap(lambda kv: (kv[0], kv[1] * 10), {1: 2, 3: 4}) == {1: 20, 3: 40}


def test_itemmap_swap():
    assert itemmap(lambda kv: (kv[1], kv[0]), {"a": 1, "b": 2}) == {1: "a", 2: "b"}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    from collections import OrderedDict

    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# valfilter
# ---------------------------------------------------------------------------

def test_valfilter_even():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_none():
    d = {1: 2, 2: 3}
    assert valfilter(lambda x: False, d) == {}


def test_valfilter_factory():
    from collections import OrderedDict

    result = valfilter(lambda v: v == 2, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# keyfilter
# ---------------------------------------------------------------------------

def test_keyfilter_even():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda k: k % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_none():
    assert keyfilter(lambda k: False, {1: 2}) == {}


def test_keyfilter_factory():
    from collections import OrderedDict

    result = keyfilter(lambda k: k == 1, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# itemfilter
# ---------------------------------------------------------------------------

def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_none():
    assert itemfilter(lambda item: False, {1: 2}) == {}


def test_itemfilter_factory():
    from collections import OrderedDict

    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

def test_assoc_replace():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}


def test_assoc_add():
    assert assoc({"x": 1}, "y", 3) == {"x": 1, "y": 3}


def test_assoc_no_mutation():
    d = {"x": 1}
    assoc(d, "y", 2)
    assert d == {"x": 1}


def test_assoc_factory():
    from collections import OrderedDict

    result = assoc({"x": 1}, "y", 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

def test_dissoc_single():
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}


def test_dissoc_multiple():
    assert dissoc({"x": 1, "y": 2}, "y", "x") == {}


def test_dissoc_missing_key():
    assert dissoc({"x": 1}, "y") == {"x": 1}


def test_dissoc_no_mutation():
    d = {"x": 1, "y": 2}
    dissoc(d, "y")
    assert d == {"x": 1, "y": 2}


def test_dissoc_factory():
    from collections import OrderedDict

    result = dissoc({"x": 1, "y": 2}, "y", factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"x": 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({"x": 1}, "x", bad_kwarg=True)


# ---------------------------------------------------------------------------
# assoc_in
# ---------------------------------------------------------------------------

def test_assoc_in_existing_nested():
    d = {"a": {"b": 1}}
    result = assoc_in(d, ["a", "b"], 2)
    assert result == {"a": {"b": 2}}
    # original untouched
    assert d == {"a": {"b": 1}}


def test_assoc_in_new_nested():
    result = assoc_in({}, ["a", "b", "c"], 5)
    assert result == {"a": {"b": {"c": 5}}}


def test_assoc_in_add_key():
    d = {"a": {"b": 1}}
    result = assoc_in(d, ["a", "c"], 2)
    assert result == {"a": {"b": 1, "c": 2}}
    assert d == {"a": {"b": 1}}


def test_assoc_in_factory():
    from collections import OrderedDict

    result = assoc_in({}, ["a"], 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {"a": 1}


# ---------------------------------------------------------------------------
# update_in
# ---------------------------------------------------------------------------

def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}


def test_update_in_nested():
    inc = lambda x: x + 1
    d = {"a": {"b": {"c": 1}}}
    result = update_in(d, ["a", "b", "c"], inc)
    assert result == {"a": {"b": {"c": 2}}}
    assert d == {"a": {"b": {"c": 1}}}


def test_update_in_missing_uses_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_update_in_partial_existing():
    inc = lambda x: x + 1
    assert update_in({1: "foo"}, [2, 3, 4], inc, 0) == {1: "foo", 2: {3: {4: 1}}}


def test_update_in_no_mutation():
    d = {"a": {"b": 1}}
    update_in(d, ["a", "b"], lambda x: x + 1)
    assert d == {"a": {"b": 1}}


def test_update_in_factory():
    from collections import OrderedDict

    result = update_in({}, ["a", "b"], lambda x: x + 1, 0, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert isinstance(result["a"], OrderedDict)
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


def test_get_in_list_index(transaction):
    assert get_in(["purchase", "items", 0], transaction) == "Apple"


def test_get_in_simple(transaction):
    assert get_in(["name"], transaction) == "Alice"


def test_get_in_missing_key_default_none(transaction):
    assert get_in(["purchase", "total"], transaction) is None


def test_get_in_bad_index_string(transaction):
    assert get_in(["purchase", "items", "apple"], transaction) is None


def test_get_in_index_out_of_range(transaction):
    assert get_in(["purchase", "items", 10], transaction) is None


def test_get_in_custom_default(transaction):
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([0], [], no_default=True)


def test_get_in_no_default_typeerror():
    with pytest.raises(TypeError):
        get_in(["a"], "string", no_default=True)


def test_get_in_empty_keys(transaction):
    assert get_in([], transaction) is transaction
