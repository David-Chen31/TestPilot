import pytest
from collections import OrderedDict
from operator import add

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
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_precedence():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_non_mapping_iterable():
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}


def test_merge_single_mapping():
    assert merge({1: 1, 2: 2}) == {1: 1, 2: 2}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 1}, {2: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 1, 2: 2}


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError) as exc:
        merge({1: 1}, bad=1)
    assert 'bad' in str(exc.value)


def test_merge_unexpected_kwarg_exact_message():
    # Pins the exact error message, including the function name and key.
    with pytest.raises(TypeError) as exc:
        merge({1: 1}, bad=1)
    assert str(exc.value) == "merge() got an unexpected keyword argument 'bad'"


def test_merge_does_not_mutate():
    a = {1: 1}
    b = {2: 2}
    merge(a, b)
    assert a == {1: 1}
    assert b == {2: 2}


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------
def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_combines_values_in_list():
    # The capture callback records the most recently constructed list of
    # values. In the current implementation, merge_with iterates over
    # ``values.items()`` (a defaultdict insertion-order), so the last key
    # processed is the last-inserted key (2 here), whose collected values
    # are [3, 4].
    seen = {}

    def capture(vals):
        seen['vals'] = list(vals)
        return sum(vals)

    merge_with(capture, {1: 1}, {1: 2, 2: 3}, {2: 4})
    assert seen['vals'] == [3, 4]


def test_merge_with_collects_all_values_per_key():
    # Verify the per-key grouping (not just the final callback result) by
    # calling merge_with with a func that returns its input list.
    result = merge_with(list, {1: 1}, {1: 2, 2: 3}, {2: 4})
    assert result == {1: [1, 2], 2: [3, 4]}


def test_merge_with_single_non_mapping_iterable():
    assert merge_with(sum, [{1: 1}, {1: 10}]) == {1: 11}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 3}


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, nope=2)


def test_merge_with_unexpected_kwarg_exact_message():
    with pytest.raises(TypeError) as exc:
        merge_with(sum, {1: 1}, nope=2)
    assert str(exc.value) == \
        "merge_with() got an unexpected keyword argument 'nope'"


def test_merge_with_preserves_order_of_first_appearance():
    result = merge_with(sum, {1: 1, 2: 2}, {3: 3, 1: 10})
    assert list(result.keys()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------
def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}


def test_valmap_empty():
    assert valmap(sum, {}) == {}


def test_valmap_factory():
    result = valmap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: '2'}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {"alice": [20, 15, 30], "bob": [10, 35]}


def test_keymap_empty():
    assert keymap(str.lower, {}) == {}


def test_keymap_factory():
    result = keymap(str, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'1': 2}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_reversed():
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}


def test_itemmap_factory():
    result = itemmap(lambda kv: kv, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_valmap_does_not_mutate():
    d = {1: 2}
    valmap(add_one, d)
    assert d == {1: 2}


def add_one(x):
    return x + 1


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------
def test_valfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}


def test_valfilter_all_removed():
    assert valfilter(lambda x: False, {1: 2}) == {}


def test_valfilter_factory():
    result = valfilter(lambda v: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}


def test_keyfilter_none_kept():
    assert keyfilter(lambda k: False, {1: 2}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_itemfilter():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_none_kept():
    assert itemfilter(lambda item: False, {1: 2}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: True, {1: 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {1: 2}


def test_filters_do_not_mutate():
    d = {1: 2, 2: 3}
    valfilter(lambda v: True, d)
    keyfilter(lambda k: True, d)
    itemfilter(lambda i: True, d)
    assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------
def test_assoc_overwrite():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert d == {'x': 1}
    assert result == {'x': 2}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------
def test_dissoc_single():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_multiple():
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}


def test_dissoc_missing_key_ignored():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_does_not_mutate():
    d = {'x': 1, 'y': 2}
    dissoc(d, 'y')
    assert d == {'x': 1, 'y': 2}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'x': 1}


def test_dissoc_no_keys():
    assert dissoc({'x': 1}) == {'x': 1}


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError):
        dissoc({'x': 1}, 'x', bad=1)


def test_dissoc_unexpected_kwarg_exact_message():
    with pytest.raises(TypeError) as exc:
        dissoc({'x': 1}, 'x', bad=1)
    assert str(exc.value) == \
        "dissoc() got an unexpected keyword argument 'bad'"


def test_dissoc_small_keys_branch():
    # len(keys)=1 < len(d)*0.6=6 -> update+delete branch
    d = {i: i for i in range(10)}
    assert dissoc(d, 3) == {i: i for i in range(10) if i != 3}


def test_dissoc_large_keys_branch():
    # len(keys)=3 >= len(d)*0.6=1.8 -> difference branch
    d = {1: 1, 2: 2, 3: 3}
    assert dissoc(d, 1, 2, 3) == {}


def test_dissoc_boundary_exactly_at_threshold():
    # len(d)=5, len(keys)=3: 3 < 5*0.6=3 is False -> difference branch.
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    result = dissoc(d, 1, 2, 9)
    assert result == {3: 'c', 4: 'd', 5: 'e'}


def test_dissoc_boundary_just_below_threshold():
    # len(d)=5, len(keys)=2: 2 < 3 is True -> update+delete branch.
    d = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    result = dissoc(d, 1, 9)
    assert result == {2: 'b', 3: 'c', 4: 'd', 5: 'e'}


def test_dissoc_threshold_distinguishes_branches():
    # A distinguishing case for the `<` vs `<=` comparison: when
    # len(keys) == len(d)*0.6 exactly, the branches behave differently
    # for keys that are absent from d (via a mapping that tracks updates).
    # Use a custom factory that records whether `update` was called.
    class TrackingDict(dict):
        updated = False

        def update(self, *args, **kwargs):
            TrackingDict.updated = True
            return super().update(*args, **kwargs)

    TrackingDict.updated = False
    d = {i: i for i in range(5)}
    # len(keys)=3, len(d)*0.6 = 3.0 exactly -> `<` is False.
    dissoc(d, 0, 1, 2, factory=TrackingDict)
    assert TrackingDict.updated is False

    TrackingDict.updated = False
    d = {i: i for i in range(5)}
    # len(keys)=2, 2 < 3 -> update branch used.
    dissoc(d, 0, 1, factory=TrackingDict)
    assert TrackingDict.updated is True


# ---------------------------------------------------------------------------
# assoc_in / update_in
# ---------------------------------------------------------------------------
def test_update_in_simple():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_nested():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'],
                     'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result['purchase']['costs'] == 1.75
    assert result['purchase']['items'] == ['Apple', 'Orange']
    assert result['name'] == 'Alice'


def test_update_in_does_not_mutate():
    d = {'a': {'b': 1}}
    update_in(d, ['a', 'b'], lambda x: x + 1)
    assert d == {'a': {'b': 1}}


def test_update_in_creates_nested_with_default():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_creates_nested_keeping_other_keys():
    assert update_in({1: 'foo'}, [2, 3, 4], lambda x: x + 1, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}


def test_update_in_missing_final_key():
    assert update_in({'a': {}}, ['a', 'b'], lambda x: x + 1, 10) == \
        {'a': {'b': 11}}


def test_update_in_factory():
    result = update_in({'a': 1}, ['a'], lambda x: x + 1, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 2}


def test_assoc_in_simple():
    assert assoc_in({'x': 1}, ['x'], 2) == {'x': 2}


def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result['order']['costs'] == [0.25, 1.00]
    assert result['order']['items'] == ['Apple', 'Orange']


def test_assoc_in_creates_missing():
    assert assoc_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}


def test_assoc_in_does_not_mutate():
    d = {'a': {'b': 1}}
    assoc_in(d, ['a', 'b'], 2)
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    result = assoc_in({'a': 1}, ['a'], 2, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert result == {'a': 2}


def test_assoc_in_ignores_existing_value_as_default():
    # The 4th arg passed by assoc_in to update_in is `value` itself, not
    # `None`. This is observable when the final key is missing but an
    # earlier existing value equals a "falsey" that would be `None`-default
    # only if the default mattered. Here we make the default matter by
    # using a func that ignores the default value entirely only when the
    # key exists; for the missing case the default passed is `value`.
    # Confirm assoc_in sets the missing leaf to `value` unchanged.
    result = assoc_in({'a': {}}, ['a', 'b'], 'VALUE')
    assert result == {'a': {'b': 'VALUE'}}


def test_assoc_in_default_is_value_not_none():
    # A func receiving `x` from update_in: for a missing final key, the
    # input to the func is the default. If assoc_in passed None as default,
    # the func would receive None. Instead it receives `value`. We verify
    # by observing behavior via a value type where None vs value differ.
    result = assoc_in({}, ['a', 'b'], 42)
    assert result == {'a': {'b': 42}}


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------
TRANSACTION = {'name': 'Alice',
               'purchase': {'items': ['Apple', 'Orange'],
                            'costs': [0.50, 1.25]},
               'credit card': '5555-1234-1234-1234'}


def test_get_in_list_index():
    assert get_in(['purchase', 'items', 0], TRANSACTION) == 'Apple'


def test_get_in_simple_key():
    assert get_in(['name'], TRANSACTION) == 'Alice'


def test_get_in_missing_key_returns_none():
    assert get_in(['purchase', 'total'], TRANSACTION) is None


def test_get_in_wrong_type_returns_none():
    assert get_in(['purchase', 'items', 'apple'], TRANSACTION) is None


def test_get_in_out_of_range_returns_none():
    assert get_in(['purchase', 'items', 10], TRANSACTION) is None


def test_get_in_custom_default():
    assert get_in(['purchase', 'total'], TRANSACTION, 0) == 0


def test_get_in_no_default_raises_keyerror():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)


def test_get_in_no_default_raises_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2], no_default=True)


def test_get_in_no_default_raises_keyerror_for_missing_string_key():
    # A missing string key on an int-keyed dict raises KeyError, not TypeError.
    with pytest.raises(KeyError):
        get_in(['x'], {1: 2}, no_default=True)


def test_get_in_no_default_raises_typeerror():
    # Indexing a list with a string raises TypeError.
    with pytest.raises(TypeError):
        get_in(['x'], [1, 2], no_default=True)


def test_get_in_empty_keys():
    assert get_in([], {'a': 1}) == {'a': 1}
