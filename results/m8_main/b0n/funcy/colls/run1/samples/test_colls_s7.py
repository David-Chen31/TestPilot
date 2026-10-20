import pytest
from collections import defaultdict, Counter, OrderedDict, namedtuple

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

def test_empty_dict():
    assert empty({'a': 1}) == {}
    assert type(empty({'a': 1})) is dict


def test_empty_list():
    assert empty([1, 2]) == []
    assert type(empty([1, 2])) is list


def test_empty_tuple():
    assert empty((1, 2)) == ()
    assert isinstance(empty((1, 2)), tuple)


def test_empty_set():
    assert empty({1, 2}) == set()
    assert isinstance(empty({1, 2}), set)


def test_empty_str():
    assert empty('abc') == ''


def test_empty_bytes():
    assert empty(b'abc') == b''


def test_empty_iterator():
    it = iter([1, 2, 3])
    result = empty(it)
    assert list(result) == []


def test_empty_dict_keys():
    assert empty({}.keys()) == []
    assert isinstance(empty({}.keys()), list)


def test_empty_dict_values():
    assert empty({}.values()) == []


def test_empty_dict_items():
    assert empty({}.items()) == []


def test_empty_counter():
    c = Counter('aab')
    e = empty(c)
    assert isinstance(e, Counter)
    assert len(e) == 0


def test_empty_defaultdict():
    dd = defaultdict(list, {'a': [1]})
    e = empty(dd)
    assert isinstance(e, defaultdict)
    assert len(e) == 0


def test_empty_object_dict():
    class A:
        pass
    a = A()
    e = empty(a.__dict__)
    assert e == {}
    assert type(e) is dict


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert dict(iteritems({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}


def test_iteritems_non_dict():
    pairs = [(1, 2), (3, 4)]
    assert list(iteritems(pairs)) == pairs


def test_itervalues_dict():
    assert list(itervalues({'a': 1, 'b': 2})) == [1, 2]


def test_itervalues_non_dict():
    assert list(itervalues([1, 2, 3])) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_empty():
    assert join([]) is None


def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'


def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'


def test_join_dicts():
    assert join([{'a': 1}, {'b': 2}, {'a': 3}]) == {'a': 3, 'b': 2}


def test_join_dicts_preserves_type():
    result = join([OrderedDict(a=1), OrderedDict(b=2)])
    assert isinstance(result, OrderedDict)


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


def test_join_unsupported():
    with pytest.raises(TypeError):
        join([1, 2])


def test_merge():
    assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge([1], [2]) == [1, 2]


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}


def test_join_with_single_dict_not_strict():
    d = {'a': 1}
    assert join_with(sum, [d]) is d


def test_join_with_single_dict_strict():
    assert join_with(sum, [{'a': 1}], strict=True) == {'a': 1}


def test_join_with_sum():
    assert join_with(sum, [{'a': 1, 'b': 2}, {'a': 10}]) == {'a': 11, 'b': 2}


def test_join_with_list_default():
    result = join_with(list, [{'a': 1}, {'a': 2, 'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}


def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2}) == {'a': 3}


# ---------- walk ----------

def test_walk_dict():
    assert walk(lambda x: x + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}


def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_walk_set():
    assert walk(str, {1, 2}) == {'1', '2'}


def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1}) == {'A': 1}


def test_walk_keys_int_key():
    assert walk_keys(0, {(1, 2): 'x'}) == {1: 'x'}


def test_walk_values():
    assert walk_values(lambda x: x + 1, {'a': 1}) == {'a': 2}


def test_walk_values_list():
    assert walk_values(str, [1, 2]) == ['1', '2']


def test_walk_values_defaultdict_preserves():
    dd = defaultdict(int, {'a': 1})
    result = walk_values(lambda x: x + 1, dd)
    assert isinstance(result, defaultdict)
    assert result == {'a': 2}
    assert result.default_factory is int


# ---------- select ----------

def test_select_dict():
    assert select(lambda p: p[1] > 1, {'a': 1, 'b': 2, 'c': 3}) == {'b': 2, 'c': 3}


def test_select_list():
    assert select(lambda x: x % 2, [1, 2, 3, 4]) == [1, 3]


def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}


def test_select_keys_int():
    # make_pred with int -> itemgetter
    coll = {0: 'x', 1: 'y'}
    assert select_keys(0, coll) == {}


def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k % 2 == 0, {0: 'a', 1: 'b', 2: 'c'})
    assert yes == {0: 'a', 2: 'c'}
    assert no == {1: 'b'}


def test_split_keys_empty():
    yes, no = split_keys(bool, {})
    assert yes == {}
    assert no == {}


# ---------- compact ----------

def test_compact_mapping():
    assert compact({'a': 1, 'b': 0, 'c': 2, 'd': None}) == {'a': 1, 'c': 2}


def test_compact_sequence():
    assert compact([0, 1, 2, 0, 3]) == [1, 2, 3]


def test_compact_empty():
    assert compact([]) == []


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True


def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False


def test_is_distinct_with_key():
    assert is_distinct(['a', 'A'], key=str.lower) is False
    assert is_distinct(['a', 'b'], key=str.lower) is True


# ---------- all / any / none / one / some ----------

def test_all():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -1]) is False
    assert all([1, 2, 3]) is True
    assert all([1, 0]) is False


def test_any():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2]) is False
    assert any([0, 1]) is True
    assert any([0, 0]) is False


def test_none():
    assert none(lambda x: x > 5, [1, 2]) is True
    assert none(lambda x: x > 1, [1, 2]) is False
    assert none([0, 0]) is True


def test_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 10, [1, 2]) is False
    assert one([0, 1, 0]) is True
    assert one([1, 2, 3]) is False


def test_some():
    assert some(lambda x: x > 2, [1, 2, 3, 4]) == 3
    assert some(lambda x: x > 10, [1, 2]) is None
    assert some([0, '', 5]) == 5


# ---------- zipdict ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}


def test_zipdict_uneven():
    assert zipdict(['a', 'b', 'c'], [1]) == {'a': 1}


# ---------- flip ----------

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}


# ---------- project / omit ----------

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}


def test_project_missing_keys():
    assert project({'a': 1}, ['a', 'z']) == {'a': 1}


def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}


def test_omit_preserves_type():
    result = omit(OrderedDict([('a', 1), ('b', 2)]), ['a'])
    assert isinstance(result, OrderedDict)
    assert result == {'b': 2}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = set(zip_values({'a': 1, 'b': 2}, {'a': 10, 'c': 20}))
    assert result == {(1, 10)}


def test_zip_values_single():
    result = set(zip_values({'a': 1, 'b': 2}))
    assert result == {(1,), (2,)}


def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())


def test_zip_dicts():
    result = set(zip_dicts({'a': 1, 'b': 2}, {'a': 10, 'c': 20}))
    assert result == {('a', (1, 10))}


def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax ----------

def test_get_in():
    coll = {'a': {'b': [10, 20]}}
    assert get_in(coll, ['a', 'b', 1]) == 20


def test_get_in_missing_key():
    assert get_in({'a': {}}, ['a', 'b'], default=42) == 42


def test_get_in_index_error():
    assert get_in([1, 2], [5], default='x') == 'x'


def test_get_lax_type_error():
    # get_in raises TypeError here, get_lax returns default
    assert get_lax([1, 2], ['a'], default='dft') == 'dft'


def test_get_lax_missing():
    assert get_lax({}, ['x'], default=1) == 1


def test_get_in_non_indexable_raises():
    with pytest.raises(TypeError):
        get_in([1, 2], ['a'])


# ---------- set_in / update_in ----------

def test_set_in():
    coll = {'a': {'b': 1}}
    result = set_in(coll, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}
    # original not mutated
    assert coll == {'a': {'b': 1}}


def test_set_in_list():
    coll = [[1, 2], [3, 4]]
    result = set_in(coll, [1, 0], 99)
    assert result == [[1, 2], [99, 4]]
    assert coll == [[1, 2], [3, 4]]


def test_set_in_empty_path():
    assert set_in({'a': 1}, [], 'val') == 'val'


def test_update_in():
    coll = {'a': {'b': 1}}
    result = update_in(coll, ['a', 'b'], lambda x: x + 1)
    assert result == {'a': {'b': 2}}


def test_update_in_creates_path():
    coll = {}
    result = update_in(coll, ['a', 'b'], lambda x: x, default=0)
    assert result == {'a': {'b': 0}}


def test_update_in_missing_with_default():
    result = update_in({}, ['a'], lambda x: x + 1, default=10)
    assert result == {'a': 11}


# ---------- del_in ----------

def test_del_in():
    coll = {'a': {'b': 1, 'c': 2}}
    result = del_in(coll, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert coll == {'a': {'b': 1, 'c': 2}}


def test_del_in_list():
    coll = [1, 2, 3]
    result = del_in(coll, [1])
    assert result == [1, 3]
    assert coll == [1, 2, 3]


def test_del_in_empty_path():
    coll = {'a': 1}
    assert del_in(coll, []) == coll


def test_del_in_missing_path():
    coll = {'a': 1}
    assert del_in(coll, ['b']) == {'a': 1}
    assert del_in(coll, ['a', 'b']) == coll


# ---------- has_path ----------

def test_has_path_true():
    assert has_path({'a': {'b': [1]}}, ['a', 'b', 0]) is True


def test_has_path_false():
    assert has_path({'a': {}}, ['a', 'b']) is False
    assert has_path([1], [5]) is False


# ---------- where / lwhere ----------

def test_where():
    items = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2}]
    result = list(where(items, a=1))
    assert result == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]


def test_where_empty_cond():
    items = [{'a': 1}, {'b': 2}]
    assert list(where(items)) == items


def test_lwhere():
    items = [{'a': 1}, {'a': 2}]
    assert lwhere(items, a=2) == [{'a': 2}]


# ---------- pluck / lpluck / pluck_attr / lpluck_attr ----------

def test_pluck():
    items = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', items)) == [1, 2]


def test_lpluck():
    assert lpluck('a', [{'a': 1}, {'a': 2}]) == [1, 2]


def test_pluck_attr():
    P = namedtuple('P', 'x y')
    items = [P(1, 2), P(3, 4)]
    assert list(pluck_attr('x', items)) == [1, 3]


def test_lpluck_attr():
    P = namedtuple('P', 'x')
    assert lpluck_attr('x', [P(1), P(2)]) == [1, 2]


# ---------- invoke / linvoke ----------

def test_invoke():
    result = list(invoke(['ab', 'cd'], 'upper'))
    assert result == ['AB', 'CD']


def test_invoke_args():
    result = list(invoke(['a-b', 'c-d'], 'split', '-'))
    assert result == [['a', 'b'], ['c', 'd']]


def test_linvoke():
    assert linvoke(['ab', 'cd'], 'upper') == ['AB', 'CD']


def test_linvoke_kwargs():
    assert linvoke(['a b', 'c d'], 'split', sep=' ') == [['a', 'b'], ['c', 'd']]
