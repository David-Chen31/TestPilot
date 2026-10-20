import pytest
from collections import OrderedDict, Counter, defaultdict

from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact, is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke,
    where, pluck, pluck_attr, invoke,
)


# ---------------------------------------------------------------- empty

def test_empty_list():
    assert empty([1, 2, 3]) == []

def test_empty_dict():
    assert empty({'a': 1}) == {}

def test_empty_set():
    assert empty({1, 2}) == set()
    assert isinstance(empty({1, 2}), set)

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_str():
    assert empty("abc") == ""
    assert isinstance(empty("abc"), str)

def test_empty_bytes():
    assert empty(b"abc") == b""
    assert isinstance(empty(b"abc"), bytes)

def test_empty_iterator():
    e = empty(iter([1, 2]))
    assert list(e) == []

def test_empty_counter():
    e = empty(Counter(a=1))
    assert isinstance(e, Counter)
    assert e == Counter()

def test_empty_defaultdict():
    d = defaultdict(list)
    d['a'].append(1)
    e = empty(d)
    assert isinstance(e, defaultdict)
    assert len(e) == 0


# ------------------------------------------------------------ iteritems/itervalues

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_non_dict():
    data = [('a', 1), ('b', 2)]
    assert iteritems(data) is data

def test_itervalues_dict():
    d = {'a': 1, 'b': 2}
    assert list(itervalues(d)) == [1, 2]

def test_itervalues_non_dict():
    data = [1, 2, 3]
    assert itervalues(data) is data


# --------------------------------------------------------------- join/merge

def test_join_empty():
    assert join([]) is None

def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'

def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'

def test_join_dicts():
    result = join([{'a': 1}, {'b': 2}, {'a': 3}])
    assert result == {'a': 3, 'b': 2}

def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}

def test_join_lists():
    assert join([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

def test_join_tuples():
    assert join([(1, 2), (3,)]) == (1, 2, 3)

def test_join_iterators():
    result = join([iter([1, 2]), iter([3])])
    assert list(result) == [1, 2, 3]

def test_join_ranges():
    result = join([range(2), range(2, 4)])
    assert list(result) == [0, 1, 2, 3]

def test_join_unsupported():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge():
    assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge([1], [2]) == [1, 2]


# ------------------------------------------------------------ join_with/merge_with

def test_join_with_no_dicts():
    assert join_with(sum, []) == {}

def test_join_with_single_dict_non_strict():
    d = {'a': 1}
    result = join_with(sum, [d])
    assert result is d

def test_join_with_single_dict_strict():
    assert join_with(sum, [{'a': 1}], strict=True) == {'a': 1}

def test_join_with_sum():
    result = join_with(sum, [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}])
    assert result == {'a': 4, 'b': 2, 'c': 4}

def test_join_with_list_default():
    # When f is list, values are left as lists
    result = join_with(list, [{'a': 1}, {'a': 2}])
    assert result == {'a': [1, 2]}

def test_join_with_list_of_pairs():
    result = join_with(sum, [[('a', 1)], [('a', 2), ('b', 3)]])
    assert result == {'a': 3, 'b': 3}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}


# ----------------------------------------------------------------- walk family

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict():
    assert walk(lambda kv: (kv[0], kv[1] + 1), {'a': 1}) == {'a': 2}

def test_walk_set():
    # Sets iterate over elements; walk applies to each element.
    result = walk(lambda x: x + 1, {1, 2, 3})
    assert result == {2, 3, 4}

def test_walk_string():
    assert walk(lambda c: c.upper(), "abc") == "ABC"

def test_walk_iterator():
    result = walk(lambda x: x + 1, iter([1, 2]))
    assert list(result) == [2, 3]

def test_walk_keys_dict():
    assert walk_keys(lambda k: k.upper(), {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_keys_str():
    # make_func turns string into a regex finder
    result = walk_keys('a', {'aa': 1, 'b': 2})
    # finder returns match object for 'aa', None for 'b'
    assert 'aa' in result
    assert result['aa'] == 1

def test_walk_values_dict():
    assert walk_values(lambda v: v * 10, {'a': 1, 'b': 2}) == {'a': 10, 'b': 20}

def test_walk_values_list_of_pairs():
    result = walk_values(lambda v: v + 1, [('a', 1), ('b', 2)])
    assert result == [('a', 2), ('b', 3)]

def test_walk_values_counter(mocker=None):
    c = Counter(a=1, b=2)
    result = walk_values(lambda v: v * 2, c)
    assert isinstance(result, Counter)
    assert result == Counter(a=2, b=4)

def test_walk_values_defaultdict():
    d = defaultdict(list)
    d['a'] = [1]
    result = walk_values(len, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 1


# --------------------------------------------------------- select family

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_select_dict():
    assert select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_with_int():
    # make_pred with int => itemgetter, used as predicate on element
    # For list of tuples works
    result = select(0, [('a',), ('b',)])
    # itemgetter(0) returns truthy string 'a' and 'b'
    assert result == [('a',), ('b',)]

def test_select_keys():
    assert select_keys(lambda k: k.startswith('a'), {'ab': 1, 'ba': 2}) == {'ab': 1}

def test_select_keys_with_set():
    assert select_keys({'a', 'b'}, {'a': 1, 'b': 2, 'c': 3}) == {'a': 1, 'b': 2}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_values_with_dict_pred():
    assert select_values({'x': 1}, {'a': 'x', 'b': 'y'}) == {'a': 'x'}


# --------------------------------------------------------------- split_keys

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'ab': 1, 'ba': 2, 'ac': 3})
    assert yes == {'ab': 1, 'ac': 3}
    assert no == {'ba': 2}

def test_split_keys_preserves_type():
    yes, no = split_keys(lambda k: k == 'a', {'a': 1, 'b': 2})
    assert isinstance(yes, dict)
    assert isinstance(no, dict)


# ----------------------------------------------------------------- compact

def test_compact_list():
    assert compact([0, 1, 2, None, 3, False, '']) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 1, 'b': 0, 'c': 2, 'd': ''}) == {'a': 1, 'c': 2}

def test_compact_set():
    assert compact({0, 1, 2}) == {1, 2}


# ------------------------------------------------------------- is_distinct

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 2]) is False

def test_is_distinct_empty():
    assert is_distinct([]) is True

def test_is_distinct_key():
    data = ['a', 'A', 'b']
    assert is_distinct(data, key=str.lower) is False

def test_is_distinct_key_distinct():
    data = ['a', 'b', 'c']
    assert is_distinct(data, key=str.lower) is True


# ------------------------------------------------------------- all/any/none/one/some

def test_all_with_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 1, [1, 2, 3]) is False

def test_all_single_arg():
    assert all([True, True]) is True
    assert all([True, False]) is False

def test_any_with_seq():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False

def test_any_single_arg():
    assert any([False, True]) is True
    assert any([False, False]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 2, [1, 2, 3]) is False

def test_none_single_arg():
    assert none([0, 0]) is True
    assert none([0, 1]) is False

def test_one():
    assert one(lambda x: x == 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 5, [1, 2, 3]) is False

def test_one_single_arg():
    assert one([0, 1, 0]) is True
    assert one([0, 0]) is False

def test_some():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_single_arg():
    assert some([0, '', 'yes', 'no']) == 'yes'
    assert some([0, '', None]) is None


# ------------------------------------------------------ zipdict/flip/project/omit

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_uneven():
    assert zipdict(['a', 'b'], [1]) == {'a': 1}

def test_flip():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c', 'z']) == {'a': 1, 'c': 3}

def test_project_preserves_type():
    result = project(OrderedDict([('a', 1), ('b', 2)]), ['b'])
    assert isinstance(result, OrderedDict)
    assert result == OrderedDict([('b', 2)])

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'b': 2}

def test_omit_preserves_type():
    result = omit(OrderedDict([('a', 1), ('b', 2)]), ['a'])
    assert isinstance(result, OrderedDict)


# --------------------------------------------------------- zip_values/zip_dicts

def test_zip_values():
    result = list(zip_values({'a': 1, 'b': 2}, {'a': 10, 'c': 30}))
    assert result == [(1, 10)]

def test_zip_values_multiple():
    result = list(zip_values({'a': 1}, {'a': 2}, {'a': 3}))
    assert result == [(1, 2, 3)]

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = list(zip_dicts({'a': 1, 'b': 2}, {'a': 10, 'c': 30}))
    assert result == [('a', (1, 10))]

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# --------------------------------------------------------- get_in/get_lax

def test_get_in():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(data, ['a', 'b', 'c']) == 1

def test_get_in_default():
    data = {'a': {'b': 1}}
    assert get_in(data, ['a', 'x']) is None
    assert get_in(data, ['a', 'x'], 'dflt') == 'dflt'

def test_get_in_list():
    assert get_in([[1, 2], [3, 4]], [1, 0]) == 3

def test_get_in_index_error():
    assert get_in([1, 2], [5], 'def') == 'def'

def test_get_in_type_error_raises():
    with pytest.raises(TypeError):
        get_in(5, [0])

def test_get_lax():
    data = {'a': {'b': 1}}
    assert get_lax(data, ['a', 'b']) == 1

def test_get_lax_wrong_type():
    assert get_lax(5, [0]) is None

def test_get_lax_key_error():
    assert get_lax({'a': 1}, ['b'], 'x') == 'x'


# --------------------------------------------------------- set_in/update_in

def test_set_in_dict():
    data = {'a': {'b': 1}}
    result = set_in(data, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    # original untouched
    assert data == {'a': {'b': 1}}

def test_set_in_list():
    data = [[1, 2], [3, 4]]
    result = set_in(data, [0, 1], 99)
    assert result == [[1, 99], [3, 4]]
    assert data == [[1, 2], [3, 4]]

def test_set_in_creates_nested():
    result = set_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}

def test_set_in_empty_path():
    assert set_in({'a': 1}, [], 'x') == 'x'

def test_update_in():
    data = {'a': {'b': 1}}
    result = update_in(data, ['a', 'b'], lambda v: v + 10)
    assert result == {'a': {'b': 11}}

def test_update_in_default():
    result = update_in({}, ['a'], lambda v: (v or 0) + 1)
    assert result == {'a': 1}

def test_update_in_empty_path():
    assert update_in(5, [], lambda v: v * 2) == 10


# ----------------------------------------------------------------- del_in

def test_del_in_dict():
    data = {'a': {'b': 1, 'c': 2}}
    result = del_in(data, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert data == {'a': {'b': 1, 'c': 2}}

def test_del_in_list():
    data = [1, 2, 3]
    result = del_in(data, [1])
    assert result == [1, 3]
    assert data == [1, 2, 3]

def test_del_in_missing_key():
    data = {'a': 1}
    assert del_in(data, ['x']) == data
    assert del_in(data, ['x']) is data

def test_del_in_missing_nested():
    data = {'a': {'b': 1}}
    assert del_in(data, ['a', 'x']) is data

def test_del_in_empty_path():
    data = {'a': 1}
    assert del_in(data, []) is data


# -------------------------------------------------------------- has_path

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True

def test_has_path_false():
    assert has_path({'a': 1}, ['a', 'b']) is False

def test_has_path_list():
    assert has_path([[1, 2]], [0, 1]) is True
    assert has_path([[1, 2]], [0, 5]) is False

def test_has_path_empty():
    assert has_path({'a': 1}, []) is True


# --------------------------------------------------------- lwhere/where

class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

def test_where():
    data = [
        {'a': 1, 'b': 2},
        {'a': 1, 'b': 3},
        {'a': 2, 'b': 2},
    ]
    result = list(where(data, a=1))
    assert result == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

def test_where_no_cond():
    data = [{'a': 1}, {'b': 2}]
    assert list(where(data)) == data

def test_lwhere():
    data = [{'a': 1}, {'a': 2}, {'b': 3}]
    result = lwhere(data, a=2)
    assert result == [{'a': 2}]
    assert isinstance(result, list)


# --------------------------------------------------------- pluck/pluck_attr

def test_pluck():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', data)) == [1, 2]

def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    result = lpluck('a', data)
    assert result == [1, 2]
    assert isinstance(result, list)

def test_pluck_attr():
    objs = [Obj(x=1), Obj(x=2)]
    assert list(pluck_attr('x', objs)) == [1, 2]

def test_lpluck_attr():
    objs = [Obj(x=1), Obj(x=2)]
    result = lpluck_attr('x', objs)
    assert result == [1, 2]
    assert isinstance(result, list)


# -------------------------------------------------------- invoke/linvoke

class Adder:
    def add(self, a, b=0):
        return a + b

def test_invoke():
    objs = [Adder(), Adder()]
    assert list(invoke(objs, 'add', 1, b=2)) == [3, 3]

def test_linvoke():
    objs = [Adder(), Adder()]
    result = linvoke(objs, 'add', 5)
    assert result == [5, 5]
    assert isinstance(result, list)

def test_invoke_method_args():
    objs = [Adder()]
    assert list(invoke(objs, 'add', 1)) == [1]
