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


# merge

def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_single_mapping():
    assert merge({1: 2}) == {1: 2}
    od = OrderedDict([(1, 2)])
    result = merge(od)
    assert result == {1: 2}
    assert type(result) is dict


def test_merge_single_iterable_of_dicts():
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}
    assert merge(({1: 1}, {2: 2})) == {1: 1, 2: 2}
    gen = ({i: i} for i in range(2))
    assert merge(gen) == {0: 0, 1: 1}
    assert merge([]) == {}


def test_merge_empty():
    assert merge() == {}


def test_merge_factory():
    result = merge({1: 2}, {3: 4}, factory=OrderedDict)
    assert result == OrderedDict([(1, 2), (3, 4)])
    assert isinstance(result, OrderedDict)


def test_merge_unexpected_kwarg():
    with pytest.raises(TypeError, match=r"merge\(\) got an unexpected keyword argument 'foo'"):
        merge({1: 2}, foo=3)


def test_merge_does_not_modify_input():
    d1 = {1: 1}
    d2 = {2: 2}
    result = merge(d1, d2)
    assert result == {1: 1, 2: 2}
    assert result is not d1
    d1[3] = 3
    assert result == {1: 1, 2: 2}
    d2[4] = 4
    assert result == {1: 1, 2: 2}


# merge_with

def test_merge_with_basic():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_lists():
    result = merge_with(lambda xs: xs, {'a': 1}, {'a': 2, 'b': 3})
    assert result == {'a': [1, 2], 'b': [3]}


def test_merge_with_single_occurrence():
    assert merge_with(sum, {'a': 1}) == {'a': 1}
    assert merge_with(lambda xs: xs, {'a': 1}) == {'a': [1]}


def test_merge_with_single_mapping():
    assert merge_with(sum, {1: 2}) == {1: 2}


def test_merge_with_single_iterable_of_dicts():
    assert merge_with(sum, [{1: 1}, {1: 2}]) == {1: 3}
    assert merge_with(sum, []) == {}


def test_merge_with_empty():
    assert merge_with(sum) == {}


def test_merge_with_factory():
    result = merge_with(sum, {1: 1}, {1: 2}, factory=OrderedDict)
    assert result == OrderedDict([(1, 3)])
    assert isinstance(result, OrderedDict)


def test_merge_with_unexpected_kwarg():
    with pytest.raises(TypeError, match=r"merge_with\(\) got an unexpected keyword argument 'foo'"):
        merge_with(sum, {1: 1}, foo=2)


# valmap

def test_valmap_basic():
    assert valmap(lambda x: x * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}


def test_valmap_empty():
    assert valmap(lambda x: x, {}) == {}


def test_valmap_factory():
    result = valmap(lambda x: x + 1, {'a': 1}, factory=OrderedDict)
    assert result == OrderedDict([('a', 2)])
    assert isinstance(result, OrderedDict)


def test_valmap_does_not_modify_input():
    d = {'a': 1}
    result = valmap(lambda x: x + 1, d)
    assert result == {'a': 2}
    assert result is not d
    assert d == {'a': 1}


# keymap

def test_keymap_basic():
    assert keymap(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_keymap_collision():
    assert keymap(lambda k: 'x', {'a': 1, 'b': 2}) == {'x': 2}


def test_keymap_empty():
    assert keymap(str.upper, {}) == {}


def test_keymap_factory():
    result = keymap(str.upper, {'a': 1}, factory=OrderedDict)
    assert result == OrderedDict([('A', 1)])
    assert isinstance(result, OrderedDict)


def test_keymap_does_not_modify_input():
    d = {'a': 1}
    result = keymap(str.upper, d)
    assert result == {'A': 1}
    assert result is not d
    assert d == {'a': 1}


# itemmap

def test_itemmap_basic():
    assert itemmap(lambda kv: (kv[0], kv[1] * 2), {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}


def test_itemmap_reversed():
    assert itemmap(reversed, {'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_itemmap_empty():
    assert itemmap(lambda kv: kv, {}) == {}


def test_itemmap_factory():
    result = itemmap(lambda kv: (kv[0].upper(), kv[1]), {'a': 1}, factory=OrderedDict)
    assert result == OrderedDict([('A', 1)])
    assert isinstance(result, OrderedDict)


def test_itemmap_does_not_modify_input():
    d = {'a': 1}
    result = itemmap(lambda kv: (kv[0], kv[1] + 1), d)
    assert result == {'a': 2}
    assert result is not d
    assert d == {'a': 1}


# valfilter

def test_valfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}


def test_valfilter_all_filtered():
    assert valfilter(lambda x: False, {1: 1, 2: 2}) == {}


def test_valfilter_empty():
    assert valfilter(lambda x: True, {}) == {}


def test_valfilter_factory():
    result = valfilter(lambda x: x > 0, {'a': 1, 'b': -1}, factory=OrderedDict)
    assert result == OrderedDict([('a', 1)])
    assert isinstance(result, OrderedDict)


def test_valfilter_does_not_modify_input():
    d = {1: 2, 2: 3}
    result = valfilter(lambda x: x == 2, d)
    assert result == {1: 2}
    assert result is not d
    assert d == {1: 2, 2: 3}


# keyfilter

def test_keyfilter_basic():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_keyfilter_all_filtered():
    assert keyfilter(lambda x: False, {1: 1, 2: 2}) == {}


def test_keyfilter_empty():
    assert keyfilter(lambda x: True, {}) == {}


def test_keyfilter_factory():
    result = keyfilter(lambda k: k > 0, {1: 'a', -1: 'b'}, factory=OrderedDict)
    assert result == OrderedDict([(1, 'a')])
    assert isinstance(result, OrderedDict)


def test_keyfilter_does_not_modify_input():
    d = {1: 2, 2: 3}
    result = keyfilter(lambda k: k == 1, d)
    assert result == {1: 2}
    assert result is not d
    assert d == {1: 2, 2: 3}


# itemfilter

def test_itemfilter_basic():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}


def test_itemfilter_all_filtered():
    assert itemfilter(lambda item: False, {1: 1, 2: 2}) == {}


def test_itemfilter_empty():
    assert itemfilter(lambda item: True, {}) == {}


def test_itemfilter_factory():
    result = itemfilter(lambda item: item[1] > 0, {'a': 1, 'b': -1}, factory=OrderedDict)
    assert result == OrderedDict([('a', 1)])
    assert isinstance(result, OrderedDict)


def test_itemfilter_does_not_modify_input():
    d = {1: 2, 2: 3}
    result = itemfilter(lambda item: item[0] == 1, d)
    assert result == {1: 2}
    assert result is not d
    assert d == {1: 2, 2: 3}


# assoc

def test_assoc_new_key():
    assert assoc({'x': 1}, 'y', 2) == {'x': 1, 'y': 2}


def test_assoc_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}


def test_assoc_empty():
    assert assoc({}, 'x', 1) == {'x': 1}


def test_assoc_factory():
    result = assoc({'x': 1}, 'y', 2, factory=OrderedDict)
    assert result == OrderedDict([('x', 1), ('y', 2)])
    assert isinstance(result, OrderedDict)


def test_assoc_does_not_modify_input():
    d = {'x': 1}
    result = assoc(d, 'x', 2)
    assert result == {'x': 2}
    assert result is not d
    assert d == {'x': 1}


# dissoc

def test_dissoc_remove_one():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}


def test_dissoc_remove_multiple():
    assert dissoc({'x': 1, 'y': 2, 'z': 3}, 'y', 'z') == {'x': 1}


def test_dissoc_remove_missing_key():
    assert dissoc({'x': 1}, 'y') == {'x': 1}


def test_dissoc_remove_all():
    assert dissoc({'x': 1, 'y': 2}, 'x', 'y') == {}


def test_dissoc_no_keys():
    d = {'x': 1, 'y': 2}
    assert dissoc(d) == {'x': 1, 'y': 2}
    assert dissoc({}) == {}


def test_dissoc_branch_update_delete():
    d = {i: i for i in range(5)}
    result = dissoc(d, 1, 3)
    assert result == {0: 0, 2: 2, 4: 4}


def test_dissoc_branch_set_difference():
    d = {i: i for i in range(5)}
    result = dissoc(d, 0, 2, 4)
    assert result == {1: 1, 3: 3}


def test_dissoc_remove_missing_key_branch1():
    d = {i: i for i in range(5)}
    result = dissoc(d, 'y')
    assert result == d


def test_dissoc_empty_dict():
    assert dissoc({}, 'a') == {}
    assert dissoc({}, 'a', 'b') == {}


def test_dissoc_factory():
    result = dissoc({'x': 1, 'y': 2}, 'y', factory=OrderedDict)
    assert result == OrderedDict([('x', 1)])
    assert isinstance(result, OrderedDict)


def test_dissoc_unexpected_kwarg():
    with pytest.raises(TypeError, match=r"dissoc\(\) got an unexpected keyword argument 'foo'"):
        dissoc({'x': 1}, 'x', foo=2)


def test_dissoc_does_not_modify_input():
    d = {'x': 1, 'y': 2}
    result = dissoc(d, 'y')
    assert result == {'x': 1}
    assert result is not d
    assert d == {'x': 1, 'y': 2}


# assoc_in

def test_assoc_in_existing():
    assert assoc_in({'a': 1}, ['a'], 2) == {'a': 2}


def test_assoc_in_nested_existing():
    d = {'a': {'b': 1}}
    assert assoc_in(d, ['a', 'b'], 2) == {'a': {'b': 2}}


def test_assoc_in_nested_missing():
    assert assoc_in({}, ['a', 'b'], 2) == {'a': {'b': 2}}


def test_assoc_in_deep_missing():
    assert assoc_in({}, [1, 2, 3], 'bar') == {1: {2: {3: 'bar'}}}


def test_assoc_in_does_not_modify_input():
    d = {'a': {'b': 1}}
    result = assoc_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert result is not d
    assert result['a'] is not d['a']
    assert d == {'a': {'b': 1}}


def test_assoc_in_factory():
    result = assoc_in({}, ['a', 'b'], 1, factory=OrderedDict)
    assert result == OrderedDict([('a', OrderedDict([('b', 1)]))])
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)


def test_assoc_in_empty_keys():
    with pytest.raises(StopIteration):
        assoc_in({}, [], 1)


def test_assoc_in_tuple_keys():
    assert assoc_in({}, ('a', 'b'), 1) == {'a': {'b': 1}}


def test_assoc_in_generator_keys():
    gen = (k for k in ['a', 'b'])
    assert assoc_in({}, gen, 1) == {'a': {'b': 1}}


# update_in

def test_update_in_existing():
    assert update_in({'a': 0}, ['a'], lambda x: x + 1) == {'a': 1}


def test_update_in_nested_existing():
    d = {'a': {'b': 0}}
    assert update_in(d, ['a', 'b'], lambda x: x + 1) == {'a': {'b': 1}}


def test_update_in_missing_key():
    assert update_in({}, ['a'], lambda x: x + 1, default=0) == {'a': 1}


def test_update_in_missing_nested():
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}


def test_update_in_missing_intermediate():
    assert update_in({'a': {}}, ['a', 'b'], lambda x: x + 1, default=5) == {'a': {'b': 6}}


def test_update_in_missing_final_key_with_default():
    assert update_in({'a': {}}, ['a', 'b'], lambda x: x * 2, default=10) == {'a': {'b': 20}}


def test_update_in_default_not_used_for_existing_none():
    result = update_in({'a': None}, ['a'], lambda x: 1, default=5)
    assert result == {'a': 1}


def test_update_in_empty_keys():
    with pytest.raises(StopIteration):
        update_in({'a': 1}, [], lambda x: x)


def test_update_in_factory():
    result = update_in({}, ['a', 'b'], lambda x: x, default=1, factory=OrderedDict)
    assert result == OrderedDict([('a', OrderedDict([('b', 1)]))])
    assert isinstance(result, OrderedDict)
    assert isinstance(result['a'], OrderedDict)


def test_update_in_does_not_modify_input():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 1)
    assert result == {'a': {'b': 2}}
    assert result is not d
    assert result['a'] is not d['a']
    assert d == {'a': {'b': 1}}


def test_update_in_tuple_keys():
    assert update_in({'a': {'b': 1}}, ('a', 'b'), lambda x: x + 1) == {'a': {'b': 2}}


def test_update_in_generator_keys():
    gen = (k for k in ['a', 'b'])
    assert update_in({'a': {'b': 1}}, gen, lambda x: x + 1) == {'a': {'b': 2}}


def test_update_in_non_dict_intermediate_raises():
    with pytest.raises(TypeError):
        update_in({'a': 1}, ['a', 'b'], lambda x: x)


# get_in

def test_get_in_dict():
    d = {'a': {'b': 1}}
    assert get_in(['a', 'b'], d) == 1


def test_get_in_list():
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2


def test_get_in_nested_mixed():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_default():
    d = {'a': {'b': 1}}
    assert get_in(['a', 'c'], d) is None
    assert get_in(['a', 'c'], d, default=0) == 0


def test_get_in_missing_index_default():
    d = {'a': [1, 2]}
    assert get_in(['a', 5], d) is None
    assert get_in(['a', 5], d, default='missing') == 'missing'


def test_get_in_type_error_default():
    d = {'a': 1}
    assert get_in(['a', 'b'], d) is None
    assert get_in(['a', 'b'], d, default='bad') == 'bad'


def test_get_in_empty_keys():
    d = {'a': 1}
    assert get_in([], d) is d


def test_get_in_no_default_keyerror():
    with pytest.raises(KeyError):
        get_in(['a', 'b'], {'a': {}}, no_default=True)


def test_get_in_no_default_indexerror():
    with pytest.raises(IndexError):
        get_in([5], [1, 2], no_default=True)


def test_get_in_no_default_typeerror():
    with pytest.raises(TypeError):
        get_in(['a', 'b'], {'a': 1}, no_default=True)


def test_get_in_no_default_success():
    assert get_in(['a'], {'a': 1}, no_default=True) == 1


def test_get_in_no_default_ignores_default():
    with pytest.raises(KeyError):
        get_in(['a'], {}, default=42, no_default=True)


def test_get_in_empty_keys_no_default():
    d = {'a': 1}
    assert get_in([], d, no_default=True) is d


def test_get_in_none_coll():
    assert get_in(['a'], None) is None
    assert get_in(['a'], None, default=0) == 0


def test_get_in_tuple_keys():
    assert get_in(('a', 'b'), {'a': {'b': 1}}) == 1


def test_get_in_iterator_keys():
    keys = iter(['a', 'b'])
    assert get_in(keys, {'a': {'b': 1}}) == 1
