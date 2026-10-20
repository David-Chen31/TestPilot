import pytest
from collections import Counter, defaultdict, OrderedDict

import funcy.colls as colls
from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact, is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke, where, pluck, pluck_attr, invoke,
)


# ---------- empty ----------

def test_empty_dict():
    assert empty({'a': 1}) == {}
    assert type(empty({'a': 1})) is dict

def test_empty_list():
    assert empty([1, 2, 3]) == []

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_str():
    assert empty('abc') == ''

def test_empty_bytes():
    assert empty(b'abc') == b''

def test_empty_iterator():
    it = iter([1, 2, 3])
    result = empty(it)
    assert list(result) == []

def test_empty_counter():
    c = Counter(a=1, b=2)
    e = empty(c)
    assert isinstance(e, Counter)
    assert e == Counter()

def test_empty_defaultdict():
    d = defaultdict(list, a=[1])
    e = empty(d)
    assert isinstance(e, defaultdict)
    assert e.default_factory is list
    assert e == {}

def test_empty_dict_keys():
    assert empty({}.keys()) == []
    assert empty({}.values()) == []
    assert empty({}.items()) == []


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_no_items():
    assert iteritems([1, 2, 3]) == [1, 2, 3]

def test_itervalues_dict():
    assert sorted(itervalues({'a': 1, 'b': 2})) == [1, 2]

def test_itervalues_no_values():
    assert itervalues([1, 2, 3]) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'

def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'

def test_join_dicts():
    assert join([{'a': 1}, {'b': 2}, {'a': 3}]) == {'a': 3, 'b': 2}

def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}

def test_join_lists():
    assert join([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

def test_join_tuples():
    assert join([(1, 2), (3,)]) == (1, 2, 3)

def test_join_iterators():
    result = join([iter([1, 2]), iter([3])])
    assert list(result) == [1, 2, 3]

def test_join_range():
    result = join([range(2), range(2, 4)])
    assert list(result) == [0, 1, 2, 3]

def test_join_empty_returns_none():
    assert join([]) is None

def test_join_unsupported():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge():
    assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge([1], [2, 3]) == [1, 2, 3]


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(list, []) == {}

def test_join_with_single():
    d = {'a': 1}
    assert join_with(list, [d]) is d

def test_join_with_single_strict():
    assert join_with(list, [{'a': 1}], strict=True) == {'a': [1]}

def test_join_with_list():
    result = join_with(list, [{'a': 1, 'b': 2}, {'a': 3}])
    assert result == {'a': [1, 3], 'b': [2]}

def test_join_with_sum():
    result = join_with(sum, [{'a': 1}, {'a': 2, 'b': 3}])
    assert result == {'a': 3, 'b': 3}

def test_join_with_non_dict_iterable():
    result = join_with(list, [[('a', 1)], [('a', 2)]])
    assert result == {'a': [1, 2]}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2}) == {'a': 3}


# ---------- walk / walk_keys / walk_values ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict():
    assert walk(lambda p: (p[0], p[1] * 2), {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_tuple():
    assert walk(lambda x: x + 1, (1, 2)) == (2, 3)

def test_walk_str():
    assert walk(str.upper, 'abc') == 'ABC'

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_keys_int_key():
    assert walk_keys(0, {'a': 1}) == {'a': 1}

def test_walk_values():
    assert walk_values(lambda v: v + 10, {'a': 1, 'b': 2}) == {'a': 11, 'b': 12}

def test_walk_values_counter():
    c = Counter(a=2, b=1)
    result = walk_values(lambda v: v * 2, c)
    assert isinstance(result, Counter)
    assert result == Counter(a=4, b=2)

def test_walk_values_defaultdict():
    d = defaultdict(list)
    d['a'] = [1]
    result = walk_values(len, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 1


# ---------- select / select_keys / select_values ----------

def test_select_list():
    assert select(lambda x: x > 2, [1, 2, 3, 4]) == [3, 4]

def test_select_dict():
    assert select(lambda p: p[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_str():
    assert select(lambda c: c in 'aeiou', 'hello') == 'eo'

def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2, 'c': 3}) == {'b': 2, 'c': 3}

def test_select_with_set_pred():
    assert select({1, 2}, [0, 1, 2, 3]) == [1, 2]

def test_select_keys_with_regex():
    assert select_keys(r'^a', {'ab': 1, 'b': 2, 'ac': 3}) == {'ab': 1, 'ac': 3}


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'ab': 1, 'b': 2, 'ac': 3})
    assert yes == {'ab': 1, 'ac': 3}
    assert no == {'b': 2}

def test_split_keys_with_regex():
    yes, no = split_keys(r'^a', {'ab': 1, 'bc': 2})
    assert yes == {'ab': 1}
    assert no == {'bc': 2}


# ---------- compact ----------

def test_compact_list():
    assert compact([0, 1, None, 2, '', 3]) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 0, 'b': 1, 'c': None}) == {'b': 1}

def test_compact_str():
    assert compact('a0b') == 'ab'


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 1, 2]) is False

def test_is_distinct_with_key():
    assert is_distinct(['a', 'A', 'b'], key=str.lower) is False
    assert is_distinct(['a', 'b'], key=str.lower) is True

def test_is_distinct_empty():
    assert is_distinct([]) is True


# ---------- all / any / none / one / some ----------

def test_all_with_pred():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2]) is False

def test_all_no_seq():
    assert all([1, 2, 3]) is True
    assert all([1, 0, 3]) is False

def test_any_with_pred():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2]) is False

def test_any_no_seq():
    assert any([0, '', 1]) is True
    assert any([0, '', None]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2]) is True
    assert none(lambda x: x > 1, [1, 2]) is False

def test_none_no_seq():
    assert none([0, '', None]) is True
    assert none([0, 1]) is False

def test_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False

def test_one_no_seq():
    assert one([0, 1, 0]) is True
    assert one([0, 0]) is False

def test_some():
    assert some(lambda x: x > 2, [1, 2, 3, 4]) == 3
    assert some(lambda x: x > 10, [1, 2]) is None

def test_some_no_seq():
    assert some([0, '', 5]) == 5
    assert some([0, '']) is None


# ---------- zipdict / flip ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_truncates():
    assert zipdict(['a', 'b'], [1]) == {'a': 1}

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}


# ---------- project / omit ----------

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}

def test_project_missing():
    assert project({'a': 1}, ['a', 'z']) == {'a': 1}

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = sorted(zip_values({'a': 1, 'b': 2}, {'a': 3, 'c': 4}))
    assert result == [(1, 3)]

def test_zip_values_empty_raises():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = sorted(zip_dicts({'a': 1, 'b': 2}, {'a': 3}))
    assert result == [('a', (1, 3))]

def test_zip_dicts_empty_raises():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax ----------

def test_get_in():
    d = {'a': {'b': [1, 2, 3]}}
    assert get_in(d, ['a', 'b', 1]) == 2

def test_get_in_default():
    assert get_in({'a': 1}, ['b'], default='x') == 'x'

def test_get_in_none_default():
    assert get_in({'a': 1}, ['b']) is None

def test_get_in_index_error():
    assert get_in([1, 2], [5], default='d') == 'd'

def test_get_lax_type_error():
    assert get_lax([1, 2], ['x'], default=0) == 0

def test_get_lax_success():
    assert get_lax([{'a': 5}], [0, 'a']) == 5


# ---------- set_in / update_in / del_in / has_path ----------

def test_set_in_dict():
    d = {'a': {'b': 1}}
    result = set_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}  # original unchanged

def test_set_in_creates():
    assert set_in({}, ['a', 'b'], 1) == {'a': {'b': 1}}

def test_set_in_list():
    assert set_in([[1, 2], [3, 4]], [1, 0], 9) == [[1, 2], [9, 4]]

def test_update_in():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda v: v + 10)
    assert result == {'a': {'b': 11}}

def test_update_in_empty_path():
    assert update_in(5, [], lambda v: v * 2) == 10

def test_update_in_default():
    assert update_in({}, ['a'], lambda v: v + 1, default=0) == {'a': 1}

def test_del_in():
    d = {'a': {'b': 1, 'c': 2}}
    result = del_in(d, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert d == {'a': {'b': 1, 'c': 2}}

def test_del_in_list():
    assert del_in([[1, 2], [3]], [0, 1]) == [[1], [3]]

def test_del_in_missing():
    d = {'a': 1}
    assert del_in(d, ['b']) == {'a': 1}

def test_del_in_empty_path():
    d = {'a': 1}
    assert del_in(d, []) is d

def test_has_path():
    d = {'a': {'b': [1, 2]}}
    assert has_path(d, ['a', 'b', 1]) is True
    assert has_path(d, ['a', 'c']) is False
    assert has_path(d, []) is True


# ---------- where / lwhere / pluck / lpluck / pluck_attr / lpluck_attr / invoke / linvoke ----------

def test_where():
    users = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}]
    result = list(where(users, age=30))
    assert result == [{'name': 'b', 'age': 30}]

def test_where_no_cond():
    users = [{'a': 1}, {'b': 2}]
    assert list(where(users)) == users

def test_lwhere():
    users = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}]
    assert lwhere(users, age=20) == [{'name': 'a', 'age': 20}]

def test_pluck():
    users = [{'name': 'a'}, {'name': 'b'}]
    assert list(pluck('name', users)) == ['a', 'b']

def test_lpluck():
    users = [{'name': 'a'}, {'name': 'b'}]
    assert lpluck('name', users) == ['a', 'b']

def test_pluck_attr():
    class Obj:
        def __init__(self, x):
            self.x = x
    objs = [Obj(1), Obj(2)]
    assert list(pluck_attr('x', objs)) == [1, 2]

def test_lpluck_attr():
    class Obj:
        def __init__(self, x):
            self.x = x
    objs = [Obj(1), Obj(2)]
    assert lpluck_attr('x', objs) == [1, 2]

def test_invoke():
    class Obj:
        def add(self, a, b=0):
            return a + b
    objs = [Obj(), Obj()]
    assert list(invoke(objs, 'add', 1, b=2)) == [3, 3]

def test_linvoke():
    class Obj:
        def mul(self, a):
            return a * 2
    objs = [Obj(), Obj()]
    assert linvoke(objs, 'mul', 3) == [6, 6]
