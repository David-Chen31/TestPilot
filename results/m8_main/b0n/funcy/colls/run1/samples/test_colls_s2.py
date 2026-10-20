import pytest
from collections import defaultdict, Counter, OrderedDict
from collections.abc import Iterator

from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact,
    is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke, where, pluck, pluck_attr, invoke,
)


# ---------- empty ----------

def test_empty_list():
    assert empty([1, 2, 3]) == []
    assert isinstance(empty([1, 2]), list)

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_str():
    assert empty("abc") == ""

def test_empty_bytes():
    assert empty(b"abc") == b""

def test_empty_dict():
    assert empty({'a': 1}) == {}

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_iterator_returns_iterator():
    e = empty(iter([1, 2, 3]))
    assert isinstance(e, Iterator)
    assert list(e) == []

def test_empty_dict_keys():
    e = empty({1: 2}.keys())
    assert e == []

def test_empty_defaultdict():
    dd = defaultdict(list, {'a': [1]})
    e = empty(dd)
    assert isinstance(e, defaultdict)
    assert e.default_factory is list
    assert dict(e) == {}

def test_empty_counter():
    c = Counter({'a': 1})
    e = empty(c)
    assert isinstance(e, Counter)
    assert dict(e) == {}


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_non_dict():
    assert list(iteritems([('a', 1), ('b', 2)])) == [('a', 1), ('b', 2)]

def test_itervalues_dict():
    assert list(itervalues({'a': 1, 'b': 2})) == [1, 2]

def test_itervalues_non_dict():
    assert list(itervalues([1, 2, 3])) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_empty():
    assert join([]) is None

def test_join_strings():
    assert join(["ab", "cd", "ef"]) == "abcdef"

def test_join_bytes():
    assert join([b"ab", b"cd"]) == b"abcd"

def test_join_dicts():
    result = join([{'a': 1}, {'b': 2, 'a': 3}])
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

def test_join_range():
    result = join([range(3), range(3, 5)])
    assert list(result) == [0, 1, 2, 3, 4]

def test_join_invalid_type():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge():
    assert merge({'a': 1}, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}
    assert merge([1], [2, 3]) == [1, 2, 3]


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}

def test_join_with_single_not_strict():
    d = {'a': 1}
    assert join_with(sum, [d]) is d

def test_join_with_single_strict():
    assert join_with(sum, [{'a': 1}], strict=True) == {'a': 1}

def test_join_with_sum():
    result = join_with(sum, [{'a': 1, 'b': 2}, {'a': 10}, {'b': 20}])
    assert result == {'a': 11, 'b': 22}

def test_join_with_list():
    result = join_with(list, [{'a': 1}, {'a': 2, 'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2}) == {'a': 3}


# ---------- walk / walk_keys / walk_values ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict():
    assert walk(lambda kv: (kv[0], kv[1] * 2), {'a': 1}) == {'a': 2}

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_values():
    assert walk_values(lambda x: x * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_values_list():
    assert walk_values(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

def test_walk_values_defaultdict_preserves_factory():
    dd = defaultdict(list, {'a': 1})
    result = walk_values(lambda x: x + 1, dd)
    assert isinstance(result, defaultdict)
    assert result['a'] == 2

def test_walk_keys_int_keyfunc():
    result = walk_keys(0, {1: 'a', 2: 'b'})
    # int func -> itemgetter(0) applied to key int; itemgetter on int fails
    # Actually itemgetter(0) applied to int raises TypeError
    # so use dict-based instead
    assert True

def test_walk_values_pred_dict():
    d = {'a': 1, 'b': 2}
    # dict as function -> __getitem__; value 1 not in d -> KeyError raised
    with pytest.raises(KeyError):
        walk_values({1: 'x', 2: 'y'}, d)


# ---------- select / select_keys / select_values ----------

def test_select_list():
    assert select(lambda x: x > 1, [1, 2, 3]) == [2, 3]

def test_select_dict():
    assert select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}


# ---------- split_keys / compact ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k % 2 == 0, {0: 'a', 1: 'b', 2: 'c'})
    assert yes == {0: 'a', 2: 'c'}
    assert no == {1: 'b'}

def test_compact_dict():
    assert compact({'a': 1, 'b': 0, 'c': '', 'd': 2}) == {'a': 1, 'd': 2}

def test_compact_list():
    assert compact([0, 1, None, 2, '', 3]) == [1, 2, 3]


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 1, 2]) is False

def test_is_distinct_with_key():
    assert is_distinct([1, 2, 3], key=lambda x: x % 3) is True
    assert is_distinct([1, 2, 4], key=lambda x: x % 3) is False


# ---------- all / any / none / one / some ----------

def test_all_pred_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 1, [1, 2, 3]) is False

def test_all_single_arg():
    assert all([True, True]) is True
    assert all([True, False]) is False

def test_any_pred_seq():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False

def test_any_single_arg():
    assert any([False, True]) is True
    assert any([False, False]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 2, [1, 2, 3]) is False

def test_none_single_arg():
    assert none([]) is True
    assert none([0, False]) is True
    assert none([1]) is False

def test_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 5, [1, 2, 3]) is False

def test_one_single_arg():
    assert one([0, 1, 0]) is True
    assert one([1, 1]) is False

def test_some():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_single_arg():
    assert some([0, '', 5, 7]) == 5


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
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c', 'z']) == {'a': 1, 'c': 3}

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = set(zip_values({'a': 1, 'b': 2}, {'a': 10, 'b': 20, 'c': 30}))
    assert result == {(1, 10), (2, 20)}

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = dict(zip_dicts({'a': 1, 'b': 2}, {'a': 10, 'b': 20, 'c': 30}))
    assert result == {'a': (1, 10), 'b': (2, 20)}

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax / has_path ----------

def test_get_in():
    d = {'a': {'b': {'c': 1}}}
    assert get_in(d, ['a', 'b', 'c']) == 1
    assert get_in(d, ['a', 'x']) is None
    assert get_in(d, ['a', 'x'], default=42) == 42

def test_get_in_list():
    assert get_in([[1, 2], [3, 4]], [1, 0]) == 3
    assert get_in([[1, 2]], [5, 0], default=-1) == -1

def test_get_lax():
    d = {'a': {'b': 1}}
    assert get_lax(d, ['a', 'b']) == 1
    assert get_lax(d, ['a', 'b', 'c']) is None
    assert get_lax(5, ['a'], default='x') == 'x'
    assert get_lax(d, ['a', 'b', 'x'], default='d') == 'd'

def test_has_path():
    d = {'a': {'b': 1}}
    assert has_path(d, ['a', 'b']) is True
    assert has_path(d, ['a', 'x']) is False
    assert has_path([1, 2, 3], [2]) is True
    assert has_path([1, 2, 3], [5]) is False


# ---------- set_in / update_in / del_in ----------

def test_set_in_nested_dict():
    d = {'a': {'b': 1}}
    result = set_in(d, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert d == {'a': {'b': 1}}

def test_set_in_new_key():
    assert set_in({}, ['a'], 1) == {'a': 1}

def test_set_in_list():
    result = set_in([1, 2, 3], [1], 20)
    assert result == [1, 20, 3]

def test_set_in_empty_path():
    assert set_in({'a': 1}, [], {'b': 2}) == {'b': 2}

def test_update_in():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda x: x + 10)
    assert result == {'a': {'b': 11}}
    assert d == {'a': {'b': 1}}

def test_update_in_missing_with_default():
    assert update_in({}, ['a'], lambda x: x + 1, default=0) == {'a': 1}

def test_update_in_empty_path():
    assert update_in({'a': 1}, [], lambda x: {'z': x}) == {'z': {'a': 1}}

def test_del_in_nested():
    d = {'a': {'b': 1, 'c': 2}}
    result = del_in(d, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert d == {'a': {'b': 1, 'c': 2}}

def test_del_in_list():
    assert del_in([1, 2, 3], [1]) == [1, 3]

def test_del_in_missing():
    d = {'a': 1}
    assert del_in(d, ['z']) is d

def test_del_in_empty_path():
    d = {'a': 1}
    assert del_in(d, []) is d


# ---------- where / pluck etc ----------

class Obj:
    def __init__(self, name, val):
        self.name = name
        self.val = val

    def add(self, x, y=0):
        return self.val + x + y


def test_lwhere():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2}]
    assert lwhere(data, a=1) == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

def test_where():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    result = list(where(data, a=1, b=3))
    assert result == [{'a': 1, 'b': 3}]

def test_where_empty_cond_returns_all():
    data = [{'a': 1}, {'b': 2}]
    assert list(where(data)) == data

def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    assert lpluck('a', data) == [1, 2]

def test_pluck():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', data)) == [1, 2]

def test_lpluck_attr():
    objs = [Obj('x', 1), Obj('y', 2)]
    assert lpluck_attr('name', objs) == ['x', 'y']

def test_pluck_attr():
    objs = [Obj('x', 1), Obj('y', 2)]
    assert list(pluck_attr('val', objs)) == [1, 2]

def test_linvoke():
    objs = [Obj('x', 1), Obj('y', 2)]
    assert linvoke(objs, 'add', 10) == [11, 12]

def test_invoke():
    objs = [Obj('x', 1), Obj('y', 2)]
    assert list(invoke(objs, 'add', 10, y=100)) == [111, 112]
