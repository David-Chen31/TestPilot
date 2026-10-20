import pytest
from collections import defaultdict, Counter, OrderedDict, namedtuple
from types import MappingProxyType

from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact,
    is_distinct, all, any, none, one, some,
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

def test_empty_iterator():
    result = empty(iter([1, 2, 3]))
    assert list(result) == []

def test_empty_dict_keys():
    assert empty({1: 2}.keys()) == []

def test_empty_dict_values():
    assert empty({1: 2}.values()) == []

def test_empty_dict_items():
    assert empty({1: 2}.items()) == []

def test_empty_defaultdict():
    d = defaultdict(list, {'a': [1]})
    result = empty(d)
    assert isinstance(result, defaultdict)
    assert result == {}
    assert result.default_factory is list

def test_empty_counter():
    c = Counter("aab")
    result = empty(c)
    assert isinstance(result, Counter)
    assert result == Counter()


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert dict(iteritems({'a': 1})) == {'a': 1}

def test_iteritems_non_dict():
    assert list(iteritems([('a', 1), ('b', 2)])) == [('a', 1), ('b', 2)]

def test_itervalues_dict():
    assert sorted(itervalues({'a': 1, 'b': 2})) == [1, 2]

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

def test_join_unsupported():
    with pytest.raises(TypeError):
        join([1, 2])

def test_merge_lists():
    assert merge([1], [2], [3]) == [1, 2, 3]

def test_merge_dicts_later_wins():
    assert merge({'a': 1}, {'a': 2}) == {'a': 2}

def test_merge_empty():
    assert merge() is None


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}

def test_join_with_single_non_strict():
    d = {'a': 1}
    assert join_with(sum, [d]) is d

def test_join_with_single_strict():
    assert join_with(sum, [{'a': 1}], strict=True) == {'a': 1}

def test_join_with_sum():
    assert join_with(sum, [{'a': 1, 'b': 2}, {'a': 3}]) == {'a': 4, 'b': 2}

def test_join_with_list_identity():
    result = join_with(list, [{'a': 1}, {'a': 2}])
    assert result == {'a': [1, 2]}

def test_merge_with():
    assert merge_with(max, {'a': 1}, {'a': 5}) == {'a': 5}


# ---------- walk family ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict():
    # walk over items
    assert walk(lambda kv: (kv[0], kv[1] + 1), {'a': 1}) == {'a': 2}

def test_walk_str():
    assert walk(str.upper, "abc") == "ABC"

def test_walk_set():
    assert walk(lambda x: x + 1, {1, 2}) == {2, 3}

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_keys_callable_spec():
    # make_func with itemgetter for int key
    assert walk_keys(0, [(1, 'x'), (2, 'y')]) == {1: 'x', 2: 'y'}

def test_walk_values():
    assert walk_values(lambda v: v * 10, {'a': 1, 'b': 2}) == {'a': 10, 'b': 20}

def test_walk_values_list():
    assert walk_values(lambda v: v + 1, [1, 2]) == [2, 3]

def test_walk_values_defaultdict_preserves_factory():
    d = defaultdict(list, {'a': 1})
    result = walk_values(lambda v: [v], d)
    assert isinstance(result, defaultdict)
    assert result == {'a': [1]}
    assert result['missing'] == []


# ---------- select family ----------

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_select_dict():
    assert select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_str():
    assert select(lambda ch: ch.isalpha(), "a1b2") == "ab"

def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}

def test_select_keys_from_set_spec():
    assert select_keys({'a', 'b'}, {'a': 1, 'b': 2, 'c': 3}) == {'a': 1, 'b': 2}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'a1': 1, 'b': 2, 'a2': 3})
    assert yes == {'a1': 1, 'a2': 3}
    assert no == {'b': 2}

def test_split_keys_empty():
    yes, no = split_keys(bool, {})
    assert yes == {}
    assert no == {}


# ---------- compact ----------

def test_compact_list():
    assert compact([0, 1, 2, None, '', 3]) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 0, 'b': 1, 'c': None}) == {'b': 1}


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False

def test_is_distinct_key():
    assert is_distinct(['a', 'A', 'b'], key=str.lower) is False
    assert is_distinct(['a', 'b', 'c'], key=str.lower) is True


# ---------- all/any/none/one/some ----------

def test_all_simple():
    assert all(bool, [1, 2, 3]) is True
    assert all(bool, [1, 0, 3]) is False

def test_all_no_seq():
    assert all([1, 2, 3]) is True
    assert all([1, 0]) is False

def test_any_simple():
    assert any(bool, [0, 0, 1]) is True
    assert any(bool, [0, 0, 0]) is False

def test_any_no_seq():
    assert any([0, 0, 1]) is True
    assert any([]) is False

def test_none_simple():
    assert none(bool, [0, 0, 0]) is True
    assert none(bool, [0, 1]) is False

def test_none_no_seq():
    assert none([]) is True
    assert none([0, 0]) is True
    assert none([0, 1]) is False

def test_one():
    assert one(lambda x: x > 0, [0, 1, 0]) is True
    assert one(lambda x: x > 0, [1, 2, 0]) is False
    assert one(lambda x: x > 0, [0, 0]) is False

def test_one_no_seq():
    assert one([0, 1, 0]) is True
    assert one([1, 1]) is False

def test_some():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_no_seq():
    assert some([0, 0, 2]) == 2
    assert some([0, 0]) is None


# ---------- zipdict / flip / project / omit ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_short():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c', 'z']) == {'a': 1, 'c': 3}

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}

def test_omit_used_by_decorators():
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    assert omit(kwargs, ['b', 'c']) == {'a': 1}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = list(zip_values({'a': 1, 'b': 2}, {'a': 3, 'c': 4}))
    assert set(result) == {(1, 3)}

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = list(zip_dicts({'a': 1, 'b': 2}, {'a': 3, 'b': 4}))
    assert set(result) == {('a', (1, 3)), ('b', (2, 4))}

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax / has_path ----------

def test_get_in_ok():
    coll = {'a': {'b': [10, 20]}}
    assert get_in(coll, ['a', 'b', 1]) == 20

def test_get_in_default():
    coll = {'a': {'b': [10, 20]}}
    assert get_in(coll, ['a', 'x']) is None
    assert get_in(coll, ['a', 'x'], 'dflt') == 'dflt'

def test_get_lax_ok():
    assert get_lax({'a': {'b': 1}}, ['a', 'b']) == 1

def test_get_lax_does_not_raise_on_type_error():
    assert get_lax({'a': 1}, ['a', 'b'], 'dflt') == 'dflt'

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True

def test_has_path_false():
    assert has_path({'a': {'b': 1}}, ['a', 'c']) is False


# ---------- set_in / update_in / del_in ----------

def test_set_in_dict():
    coll = {'a': {'b': 1}}
    result = set_in(coll, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    # original unchanged
    assert coll == {'a': {'b': 1}}

def test_set_in_new_key():
    result = set_in({}, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}

def test_set_in_list():
    coll = {'a': [1, 2, 3]}
    result = set_in(coll, ['a', 1], 99)
    assert result == {'a': [1, 99, 3]}
    assert coll == {'a': [1, 2, 3]}

def test_update_in():
    result = update_in({'a': 1}, ['a'], lambda v: v + 1)
    assert result == {'a': 2}

def test_update_in_empty_path():
    result = update_in({'a': 1}, [], lambda c: dict(c, b=2))
    assert result == {'a': 1, 'b': 2}

def test_update_in_with_default():
    result = update_in({}, ['a'], lambda v: v + 1, default=10)
    assert result == {'a': 11}

def test_del_in():
    coll = {'a': {'b': 1, 'c': 2}}
    result = del_in(coll, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert coll == {'a': {'b': 1, 'c': 2}}

def test_del_in_missing_key():
    coll = {'a': 1}
    assert del_in(coll, ['x']) == {'a': 1}

def test_del_in_empty_path():
    coll = {'a': 1}
    assert del_in(coll, []) == {'a': 1}

def test_del_in_list():
    coll = {'a': [1, 2, 3]}
    result = del_in(coll, ['a', 1])
    assert result == {'a': [1, 3]}


# ---------- where / lwhere ----------

def test_where():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2}]
    result = list(where(data, a=1))
    assert result == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

def test_where_multiple():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    assert list(where(data, a=1, b=3)) == [{'a': 1, 'b': 3}]

def test_where_empty_cond():
    data = [{'a': 1}, {'b': 2}]
    assert list(where(data)) == data

def test_lwhere():
    data = [{'a': 1}, {'a': 2}]
    assert lwhere(data, a=2) == [{'a': 2}]


# ---------- pluck / lpluck ----------

def test_pluck():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', data)) == [1, 2]

def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    assert lpluck('a', data) == [1, 2]


# ---------- pluck_attr / lpluck_attr ----------

class Obj:
    def __init__(self, v):
        self.v = v

def test_pluck_attr():
    objs = [Obj(1), Obj(2)]
    assert list(pluck_attr('v', objs)) == [1, 2]

def test_lpluck_attr():
    objs = [Obj(1), Obj(2)]
    assert lpluck_attr('v', objs) == [1, 2]


# ---------- invoke / linvoke ----------

class Caller:
    def __init__(self, base):
        self.base = base
    def add(self, x):
        return self.base + x
    def mul(self, x, y=1):
        return self.base * x * y

def test_invoke():
    objs = [Caller(1), Caller(2)]
    assert list(invoke(objs, 'add', 10)) == [11, 12]

def test_invoke_with_kwargs():
    objs = [Caller(1), Caller(2)]
    assert list(invoke(objs, 'mul', 3, y=2)) == [6, 12]

def test_linvoke():
    objs = [Caller(1), Caller(2)]
    assert linvoke(objs, 'add', 10) == [11, 12]


# ---------- Namedtuple & ordered dict sanity ----------

def test_walk_namedtuple():
    P = namedtuple('P', 'x y')
    p = P(1, 2)
    result = walk(lambda pair: (pair[0], pair[1] * 10), p)
    assert result == P(1, 20)

def test_join_ordered_dict():
    od1 = OrderedDict([('a', 1)])
    od2 = OrderedDict([('b', 2)])
    result = join([od1, od2])
    assert result == OrderedDict([('a', 1), ('b', 2)])
    assert isinstance(result, OrderedDict)
