import pytest
from collections import OrderedDict, defaultdict, Counter

import funcy.colls as colls
from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact, is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke,
    where, pluck, pluck_attr, invoke,
)


# ---------- empty ----------

def test_empty_list():
    assert empty([1, 2, 3]) == []

def test_empty_dict():
    assert empty({'a': 1}) == {}

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_str():
    assert empty("abc") == ""

def test_empty_bytes():
    assert empty(b"abc") == b""

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_iterator_returns_iterator():
    r = empty(iter([1, 2]))
    assert iter(r) is r
    assert list(r) == []

def test_empty_defaultdict():
    dd = defaultdict(list, {'a': [1]})
    result = empty(dd)
    assert isinstance(result, defaultdict)
    assert result.default_factory is list
    assert dict(result) == {}

def test_empty_counter():
    c = Counter(a=1)
    result = empty(c)
    assert isinstance(result, Counter)
    assert dict(result) == {}

def test_empty_dict_keys_returns_list():
    assert empty({1: 2}.keys()) == []

def test_empty_dict_values_returns_list():
    assert empty({1: 2}.values()) == []

def test_empty_dict_items_returns_list():
    assert empty({1: 2}.items()) == []


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_list():
    data = [(1, 2), (3, 4)]
    assert iteritems(data) is data

def test_itervalues_dict():
    assert sorted(itervalues({'a': 1, 'b': 2})) == [1, 2]

def test_itervalues_list():
    data = [1, 2, 3]
    assert itervalues(data) is data


# ---------- join / merge ----------

def test_join_empty_returns_none():
    assert join([]) is None

def test_join_strings():
    assert join(["ab", "cd"]) == "abcd"

def test_join_bytes():
    assert join([b"ab", b"cd"]) == b"abcd"

def test_join_dicts():
    result = join([{'a': 1}, {'b': 2}, {'a': 3}])
    assert result == {'a': 3, 'b': 2}

def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}

def test_join_lists():
    assert join([[1], [2, 3]]) == [1, 2, 3]

def test_join_tuples():
    assert join([(1,), (2, 3)]) == (1, 2, 3)

def test_join_iterators_returns_chain():
    it = iter([iter([1, 2]), iter([3])])
    result = join(it)
    assert list(result) == [1, 2, 3]

def test_join_range():
    result = join([range(3), range(3, 5)])
    assert list(result) == [0, 1, 2, 3, 4]

def test_join_unsupported_type():
    with pytest.raises(TypeError):
        join([1, 2])

def test_merge_dicts():
    assert merge({'a': 1}, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}

def test_merge_lists():
    assert merge([1], [2, 3]) == [1, 2, 3]


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}

def test_join_with_single_non_strict_returns_original():
    d = {'a': 1}
    assert join_with(sum, [d]) is d

def test_join_with_single_strict():
    result = join_with(sum, [{'a': 1}], strict=True)
    assert result == {'a': 1}

def test_join_with_function():
    result = join_with(sum, [{'a': 1}, {'a': 2}])
    assert result == {'a': 3}

def test_join_with_list():
    result = join_with(list, [{'a': 1}, {'a': 2}, {'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2, 'b': 5}) == {'a': 3, 'b': 5}


# ---------- walk / walk_keys / walk_values ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict():
    assert walk(lambda x: x + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}

def test_walk_set():
    assert walk(lambda x: x + 1, {1, 2}) == {2, 3}

def test_walk_string():
    assert walk(lambda s: s.upper(), "ab") == "AB"

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_values():
    assert walk_values(lambda v: v * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_values_string_key_arg():
    # using a key/slice mapper via make_func
    assert walk_values(0, {'a': (1, 2), 'b': (3, 4)}) == {'a': 1, 'b': 3}

def test_walk_values_keeps_counter_type():
    c = Counter(a=1, b=2)
    result = walk_values(lambda v: v * 10, c)
    assert isinstance(result, Counter)
    assert result == Counter(a=10, b=20)


# ---------- select / select_keys / select_values ----------

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_select_dict():
    assert select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_set():
    assert select(lambda x: x > 1, {1, 2, 3}) == {2, 3}

def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_values_dict_arg_as_pred():
    assert select_values({'a', 'b'}, {'a': 'a', 'b': 'c'}) == {'a': 'a'}


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k % 2 == 0, {1: 'a', 2: 'b', 3: 'c'})
    assert yes == {2: 'b'}
    assert no == {1: 'a', 3: 'c'}

def test_split_keys_ordered():
    d = OrderedDict([('a', 1), ('b', 2)])
    yes, no = split_keys(lambda k: k == 'a', d)
    assert isinstance(yes, OrderedDict)
    assert yes == OrderedDict([('a', 1)])
    assert no == OrderedDict([('b', 2)])


# ---------- compact ----------

def test_compact_list():
    assert compact([0, 1, None, 2, "", 3]) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 0, 'b': 1, 'c': None}) == {'b': 1}

def test_compact_set():
    assert compact({0, 1, 2}) == {1, 2}


# ---------- content tests ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False

def test_is_distinct_with_key():
    assert is_distinct([1, 2, 3], key=lambda x: x % 3) is True
    assert is_distinct([1, 2, 4], key=lambda x: x % 3) is False

def test_all_explicit():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2]) is False

def test_all_single_truthy_check():
    assert all([1, 2, 3]) is True
    assert all([1, 0]) is False

def test_any_explicit():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2]) is False

def test_any_single_truthy_check():
    assert any([0, 0, 1]) is True
    assert any([0, 0]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 1, [1, 2, 3]) is False

def test_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 10, [1, 2, 3]) is False

def test_one_single():
    assert one([0, 0, 1, 0]) is True
    assert one([1, 1]) is False

def test_some():
    assert some(lambda x: x > 2, [1, 2, 3, 4]) == 3
    assert some(lambda x: x > 10, [1, 2, 3]) is None

def test_some_single():
    assert some([0, 0, 5, 6]) == 5
    assert some([0, 0]) is None


# ---------- zipdict / flip ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_short():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}


# ---------- project / omit ----------

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c', 'd']) == {'a': 1, 'c': 3}

def test_project_keeps_type():
    d = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    result = project(d, ['a', 'c'])
    assert isinstance(result, OrderedDict)
    assert result == OrderedDict([('a', 1), ('c', 3)])

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'b': 2}

def test_omit_keeps_type():
    d = OrderedDict([('a', 1), ('b', 2)])
    result = omit(d, ['a'])
    assert isinstance(result, OrderedDict)
    assert result == OrderedDict([('b', 2)])


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = set(zip_values({'a': 1, 'b': 2}, {'a': 3, 'c': 4}))
    assert result == {(1, 3)}

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = dict(zip_dicts({'a': 1, 'b': 2}, {'a': 3, 'c': 4}))
    assert result == {'a': (1, 3)}

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax ----------

def test_get_in():
    data = {'a': {'b': [10, 20]}}
    assert get_in(data, ['a', 'b', 1]) == 20

def test_get_in_missing_key():
    assert get_in({'a': 1}, ['b']) is None
    assert get_in({'a': 1}, ['b'], default='x') == 'x'

def test_get_in_missing_index():
    assert get_in([1, 2], [5]) is None

def test_get_in_empty_path():
    assert get_in({'a': 1}, []) == {'a': 1}

def test_get_in_raises_on_type_error():
    with pytest.raises(TypeError):
        get_in([1, 2], ['a'])

def test_get_lax():
    data = {'a': {'b': 1}}
    assert get_lax(data, ['a', 'b']) == 1

def test_get_lax_type_error():
    assert get_lax([1, 2], ['a']) is None
    assert get_lax([1, 2], ['a'], default=0) == 0

def test_get_lax_missing_key():
    assert get_lax({'a': 1}, ['b']) is None


# ---------- set_in / update_in ----------

def test_set_in_simple():
    d = {'a': 1}
    result = set_in(d, ['a'], 2)
    assert result == {'a': 2}
    assert d == {'a': 1}

def test_set_in_nested():
    d = {'a': {'b': 1}}
    result = set_in(d, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}
    assert d == {'a': {'b': 1}}

def test_set_in_new_path():
    result = set_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}

def test_set_in_list():
    data = [1, [2, 3]]
    result = set_in(data, [1, 0], 9)
    assert result == [1, [9, 3]]
    assert data == [1, [2, 3]]

def test_update_in_function():
    d = {'a': {'b': 1}}
    result = update_in(d, ['a', 'b'], lambda v: v + 10)
    assert result == {'a': {'b': 11}}

def test_update_in_empty_path():
    result = update_in(5, [], lambda v: v * 2)
    assert result == 10

def test_update_in_default():
    result = update_in({}, ['a'], lambda v: v + 1, default=0)
    assert result == {'a': 1}


# ---------- del_in ----------

def test_del_in_simple():
    d = {'a': 1, 'b': 2}
    result = del_in(d, ['a'])
    assert result == {'b': 2}
    assert d == {'a': 1, 'b': 2}

def test_del_in_nested():
    d = {'a': {'b': 1, 'c': 2}}
    result = del_in(d, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert d == {'a': {'b': 1, 'c': 2}}

def test_del_in_missing():
    d = {'a': 1}
    assert del_in(d, ['b']) == {'a': 1}

def test_del_in_empty_path():
    d = {'a': 1}
    assert del_in(d, []) == {'a': 1}

def test_del_in_list():
    data = [1, 2, 3]
    result = del_in(data, [1])
    assert result == [1, 3]
    assert data == [1, 2, 3]


# ---------- has_path ----------

def test_has_path_true():
    assert has_path({'a': {'b': [1]}}, ['a', 'b', 0]) is True

def test_has_path_false():
    assert has_path({'a': 1}, ['a', 'b']) is False

def test_has_path_empty():
    assert has_path({'a': 1}, []) is True

def test_has_path_bad_index():
    assert has_path([1, 2], [5]) is False


# ---------- where / pluck / pluck_attr / invoke ----------

def test_where():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2}]
    result = list(where(data, a=1, b=2))
    assert result == [{'a': 1, 'b': 2}]

def test_lwhere():
    data = [{'a': 1}, {'a': 2}]
    result = lwhere(data, a=2)
    assert result == [{'a': 2}]
    assert isinstance(result, list)

def test_pluck():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', data)) == [1, 2]

def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    result = lpluck('a', data)
    assert result == [1, 2]
    assert isinstance(result, list)


class _Obj:
    def __init__(self, val):
        self.val = val
    def add(self, x, y=0):
        return self.val + x + y


def test_pluck_attr():
    objs = [_Obj(1), _Obj(2)]
    assert list(pluck_attr('val', objs)) == [1, 2]

def test_lpluck_attr():
    objs = [_Obj(1), _Obj(2)]
    result = lpluck_attr('val', objs)
    assert result == [1, 2]
    assert isinstance(result, list)

def test_invoke():
    objs = [_Obj(1), _Obj(2)]
    assert list(invoke(objs, 'add', 10)) == [11, 12]

def test_invoke_kwargs():
    objs = [_Obj(1)]
    assert list(invoke(objs, 'add', 10, y=5)) == [16]

def test_linvoke():
    objs = [_Obj(1), _Obj(2)]
    result = linvoke(objs, 'add', 1)
    assert result == [2, 3]
    assert isinstance(result, list)


# ---------- iteritems / itervalues are exposed on module ----------

def test_module_exports():
    expected = {'empty', 'iteritems', 'itervalues', 'join', 'merge',
                'join_with', 'merge_with', 'walk', 'walk_keys', 'walk_values',
                'select', 'select_keys', 'select_values', 'split_keys', 'compact',
                'is_distinct', 'all', 'any', 'none', 'one', 'some', 'zipdict',
                'flip', 'project', 'omit', 'zip_values', 'zip_dicts', 'where',
                'pluck', 'pluck_attr', 'invoke', 'lwhere', 'lpluck',
                'lpluck_attr', 'linvoke', 'get_in', 'get_lax', 'set_in',
                'update_in', 'del_in', 'has_path'}
    for name in expected:
        assert hasattr(colls, name)
