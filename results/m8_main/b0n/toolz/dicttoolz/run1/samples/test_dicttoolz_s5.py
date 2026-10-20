import collections
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
# _get_factory / error handling shared by merge, merge_with and dissoc
# ---------------------------------------------------------------------------

class TestFactoryErrors:
    def test_merge_unexpected_keyword(self):
        with pytest.raises(TypeError) as exc:
            merge({1: 2}, bad=1)
        assert "unexpected keyword argument 'bad'" in str(exc.value)

    def test_merge_with_unexpected_keyword(self):
        with pytest.raises(TypeError) as exc:
            merge_with(sum, {1: 2}, bad=1)
        assert "merge_with() got an unexpected keyword argument 'bad'" in str(exc.value)

    def test_dissoc_unexpected_keyword(self):
        with pytest.raises(TypeError) as exc:
            dissoc({1: 2}, 1, bad=1)
        assert "dissoc() got an unexpected keyword argument 'bad'" in str(exc.value)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_two(self):
        assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}

    def test_merge_precedence(self):
        assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}

    def test_merge_no_args(self):
        assert merge() == {}

    def test_merge_single_mapping(self):
        assert merge({1: 2, 3: 4}) == {1: 2, 3: 4}

    def test_merge_single_non_mapping(self):
        # list of dicts is iterated over
        assert merge([{1: 2}, {3: 4}]) == {1: 2, 3: 4}

    def test_merge_does_not_mutate_inputs(self):
        a = {1: 2}
        b = {3: 4}
        merge(a, b)
        assert a == {1: 2}
        assert b == {3: 4}

    def test_merge_factory(self):
        result = merge({1: 2}, collections.OrderedDict, **(dict(factory=collections.OrderedDict)) if False else {'factory': collections.OrderedDict}) if False else merge({1: 2}, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {1: 2}

    def test_merge_returns_new_dict(self):
        a = {1: 2}
        result = merge(a)
        assert result == a
        assert result is not a


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------

class TestMergeWith:
    def test_merge_with_sum(self):
        assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}

    def test_merge_with_single_mapping(self):
        assert merge_with(sum, {1: 1, 2: 2}) == {1: 1, 2: 2}

    def test_merge_with_list_of_dicts(self):
        assert merge_with(sum, [{1: 1}, {1: 2}, {1: 3}]) == {1: 6}

    def test_merge_with_gets_list(self):
        seen = {}

        def collect(vals):
            seen['vals'] = list(vals)
            return tuple(vals)

        result = merge_with(collect, {1: 'a'}, {1: 'b', 2: 'c'})
        assert result == {1: ('a', 'b'), 2: ('c',)}
        assert seen['vals'] == ['a', 'b']

    def test_merge_with_no_dicts(self):
        assert merge_with(sum) == {}

    def test_merge_with_factory(self):
        result = merge_with(sum, {1: 1}, {1: 2}, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {1: 3}


# ---------------------------------------------------------------------------
# valmap / keymap / itemmap
# ---------------------------------------------------------------------------

class TestMaps:
    def test_valmap(self):
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        assert valmap(sum, bills) == {"Alice": 65, "Bob": 45}

    def test_valmap_empty(self):
        assert valmap(sum, {}) == {}

    def test_valmap_does_not_mutate(self):
        d = {'a': 1}
        valmap(lambda x: x + 1, d)
        assert d == {'a': 1}

    def test_valmap_factory(self):
        result = valmap(lambda x: x + 1, {'a': 1}, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'a': 2}

    def test_keymap(self):
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        assert keymap(str.lower, bills) == {
            "alice": [20, 15, 30],
            "bob": [10, 35],
        }

    def test_keymap_empty(self):
        assert keymap(str.lower, {}) == {}

    def test_keymap_factory(self):
        result = keymap(str.upper, {'a': 1}, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'A': 1}

    def test_itemmap(self):
        accountids = {"Alice": 10, "Bob": 20}
        assert itemmap(reversed, accountids) == {10: "Alice", 20: "Bob"}

    def test_itemmap_empty(self):
        assert itemmap(reversed, {}) == {}

    def test_itemmap_factory(self):
        result = itemmap(reversed, {'a': 1}, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {1: 'a'}


# ---------------------------------------------------------------------------
# valfilter / keyfilter / itemfilter
# ---------------------------------------------------------------------------

class TestFilters:
    def test_valfilter(self):
        iseven = lambda x: x % 2 == 0
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        assert valfilter(iseven, d) == {1: 2, 3: 4}

    def test_valfilter_empty(self):
        assert valfilter(bool, {}) == {}

    def test_valfilter_all_dropped(self):
        assert valfilter(lambda x: False, {1: 2, 3: 4}) == {}

    def test_valfilter_factory(self):
        result = valfilter(lambda x: x > 1, {'a': 1, 'b': 2},
                           factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'b': 2}

    def test_keyfilter(self):
        iseven = lambda x: x % 2 == 0
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        assert keyfilter(iseven, d) == {2: 3, 4: 5}

    def test_keyfilter_empty(self):
        assert keyfilter(bool, {}) == {}

    def test_keyfilter_factory(self):
        result = keyfilter(lambda k: k == 'a', {'a': 1, 'b': 2},
                           factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'a': 1}

    def test_itemfilter(self):
        def isvalid(item):
            k, v = item
            return k % 2 == 0 and v < 4

        d = {1: 2, 2: 3, 3: 4, 4: 5}
        assert itemfilter(isvalid, d) == {2: 3}

    def test_itemfilter_empty(self):
        assert itemfilter(bool, {}) == {}

    def test_itemfilter_factory(self):
        result = itemfilter(lambda item: item[0] == 'a', {'a': 1, 'b': 2},
                            factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'a': 1}

    def test_filters_do_not_mutate(self):
        d = {1: 2, 2: 3}
        valfilter(lambda v: v > 2, d)
        keyfilter(lambda k: k > 1, d)
        itemfilter(lambda item: item[0] > 1, d)
        assert d == {1: 2, 2: 3}


# ---------------------------------------------------------------------------
# assoc
# ---------------------------------------------------------------------------

class TestAssoc:
    def test_assoc_replace(self):
        assert assoc({'x': 1}, 'x', 2) == {'x': 2}

    def test_assoc_add(self):
        assert assoc({'x': 1}, 'y', 3) == {'x': 1, 'y': 3}

    def test_assoc_does_not_mutate(self):
        d = {'x': 1}
        result = assoc(d, 'x', 2)
        assert d == {'x': 1}
        assert result == {'x': 2}

    def test_assoc_factory(self):
        result = assoc({'x': 1}, 'y', 2, factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'x': 1, 'y': 2}


# ---------------------------------------------------------------------------
# dissoc
# ---------------------------------------------------------------------------

class TestDissoc:
    def test_dissoc_one(self):
        assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}

    def test_dissoc_multiple(self):
        assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}

    def test_dissoc_missing_key_ignored(self):
        assert dissoc({'x': 1}, 'y') == {'x': 1}

    def test_dissoc_no_keys(self):
        assert dissoc({'x': 1, 'y': 2}) == {'x': 1, 'y': 2}

    def test_dissoc_does_not_mutate(self):
        d = {'x': 1, 'y': 2}
        dissoc(d, 'y')
        assert d == {'x': 1, 'y': 2}

    def test_dissoc_many_keys(self):
        # exercise the alternate branch (remaining-based)
        d = {i: i for i in range(10)}
        assert dissoc(d, 0, 1, 2, 3) == {i: i for i in range(4, 10)}

    def test_dissoc_factory(self):
        result = dissoc({'x': 1, 'y': 2}, 'y', factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'x': 1}


# ---------------------------------------------------------------------------
# update_in / assoc_in
# ---------------------------------------------------------------------------

class TestUpdateIn:
    def test_update_in_simple(self):
        inc = lambda x: x + 1
        assert update_in({'a': 0}, ['a'], inc) == {'a': 1}

    def test_update_in_nested(self):
        transaction = {'name': 'Alice',
                       'purchase': {'items': ['Apple', 'Orange'],
                                    'costs': [0.50, 1.25]}}
        result = update_in(transaction, ['purchase', 'costs'], sum)
        assert result == {'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': 1.75}}

    def test_update_in_does_not_mutate(self):
        transaction = {'name': 'Alice',
                       'purchase': {'items': ['Apple', 'Orange'],
                                    'costs': [0.50, 1.25]}}
        update_in(transaction, ['purchase', 'costs'], sum)
        assert transaction['purchase']['costs'] == [0.50, 1.25]

    def test_update_in_create_nested_with_default(self):
        assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}

    def test_update_in_existing_partial(self):
        inc = lambda x: x + 1
        assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {1: 'foo', 2: {3: {4: 1}}}

    def test_update_in_factory(self):
        inc = lambda x: x + 1
        result = update_in({'a': 0}, ['a'], inc,
                           factory=collections.OrderedDict)
        assert isinstance(result, collections.OrderedDict)
        assert result == {'a': 1}


class TestAssocIn:
    def test_assoc_in_nested(self):
        purchase = {'name': 'Alice',
                    'order': {'items': ['Apple', 'Orange'],
                              'costs': [0.50, 1.25]},
                    'credit card': '5555-1234-1234-1234'}
        result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
        assert result == {'name': 'Alice',
                          'order': {'items': ['Apple', 'Orange'],
                                    'costs': [0.25, 1.00]},
                          'credit card': '5555-1234-1234-1234'}

    def test_assoc_in_create(self):
        assert assoc_in({}, ['a', 'b'], 5) == {'a': {'b': 5}}

    def test_assoc_in_does_not_mutate(self):
        purchase = {'name': 'Alice',
                    'order': {'items': ['Apple', 'Orange'],
                              'costs': [0.50, 1.25]}}
        assoc_in(purchase, ['order', 'costs'], [0.25])
        assert purchase['order']['costs'] == [0.50, 1.25]


# ---------------------------------------------------------------------------
# get_in
# ---------------------------------------------------------------------------

class TestGetIn:
    def setup_method(self):
        self.transaction = {'name': 'Alice',
                            'purchase': {'items': ['Apple', 'Orange'],
                                         'costs': [0.50, 1.25]},
                            'credit card': '5555-1234-1234-1234'}

    def test_get_in_list_index(self):
        assert get_in(['purchase', 'items', 0], self.transaction) == 'Apple'

    def test_get_in_simple(self):
        assert get_in(['name'], self.transaction) == 'Alice'

    def test_get_in_missing_returns_none(self):
        assert get_in(['purchase', 'total'], self.transaction) is None

    def test_get_in_missing_from_list_returns_none(self):
        assert get_in(['purchase', 'items', 'apple'], self.transaction) is None

    def test_get_in_index_out_of_range_returns_none(self):
        assert get_in(['purchase', 'items', 10], self.transaction) is None

    def test_get_in_custom_default(self):
        assert get_in(['purchase', 'total'], self.transaction, 0) == 0

    def test_get_in_no_default_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_in(['y'], {}, no_default=True)

    def test_get_in_no_default_raises_indexerror(self):
        with pytest.raises(IndexError):
            get_in([5], [1, 2, 3], no_default=True)

    def test_get_in_no_default_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_in(['a', 'b'], {'a': 1}, no_default=True)

    def test_get_in_empty_keys(self):
        assert get_in([], self.transaction) is self.transaction
