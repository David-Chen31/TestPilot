import pytest
from collections import defaultdict, Counter, OrderedDict

from funcy.colls import (
    empty, iteritems, itervalues, join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact, is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke,
    where, pluck, pluck_attr, invoke,
)


# --- empty ------------------------------------------------------------------

def test_empty_list():
    assert empty([1, 2, 3]) == []
    assert isinstance(empty([1, 2, 3]), list)

def test_empty_tuple():
    assert empty((1, 2)) == ()
    assert isinstance(empty((1, 2)), tuple)

def test_empty_dict():
    assert empty({'a': 1}) == {}
    assert isinstance(empty({'a': 1}), dict)

def test_empty_set():
    e = empty({1, 2})
    assert e == set()
    assert isinstance(e, set)

def test_empty_str():
    assert empty('abc') == ''

def test_empty_bytes():
    assert empty(b'abc') == b''

def test_empty_iterator():
    e = empty(iter([1, 2, 3]))
    assert list(e) == []
    assert hasattr(e, '__next__')

def test_empty_counter():
    c = Counter(a=1, b=2)
    e = empty(c)
    assert e == Counter()
    assert isinstance(e, Counter)

def test_empty_defaultdict():
    d = defaultdict(list, {'a': [1]})
    e = empty(d)
    assert e == {}
    assert isinstance(e, defaultdict)
    assert e.default_factory is list

def test_empty_dict_keys():
    e = empty({1: 2}.keys())
    assert e == []
    assert isinstance(e, list)

def test_empty_dict_values():
    e = empty({1: 2}.values())
    assert e == []
    assert isinstance(e, list)

def test_empty_dict_items():
    e = empty({1: 2}.items())
    assert e == []
    assert isinstance(e, list)


# --- iteritems / itervalues --------------------------------------------------

def test_iteritems_dict():
    d = {'a': 1, 'b': 2}
    assert list(iteritems(d)) == list(d.items())

def test_iteritems_non_dict():
    assert list(iteritems([(1, 2), (3, 4)])) == [(1, 2), (3, 4)]

def test_itervalues_dict():
    d = {'a': 1, 'b': 2}
    assert list(itervalues(d)) == [1, 2]

def test_itervalues_non_dict():
    assert list(itervalues([1, 2, 3])) == [1, 2, 3]


# --- join / merge ------------------------------------------------------------

def test_join_empty_returns_none():
    assert join([]) is None

def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'

def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'

def test_join_dicts():
    result = join([{'a': 1}, {'b': 2}, {'a': 3}])
    assert result == {'a': 3, 'b': 2}

def test_join_sets():
    result = join([{1, 2}, {2, 3}, {4}])
    assert result == {1, 2, 3, 4}

def test_join_lists():
    assert join([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

def test_join_tuples():
    assert join([(1, 2), (3,)]) == (1, 2, 3)

def test_join_iterators():
    result = join([iter([1, 2]), iter([3, 4])])
    assert list(result) == [1, 2, 3, 4]

def test_join_range():
    result = join([range(2), range(2, 4)])
    assert list(result) == [0, 1, 2, 3]

def test_join_unsupported_type():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge_dicts():
    assert merge({'a': 1}, {'b': 2}, {'a': 3}) == {'a': 3, 'b': 2}

def test_merge_sets():
    assert merge({1, 2}, {3}) == {1, 2, 3}

def test_merge_lists():
    assert merge([1], [2, 3]) == [1, 2, 3]

def test_merge_empty():
    assert merge() is None


# --- join_with / merge_with -------------------------------------------------

def test_join_with_empty_list():
    assert join_with(sum, []) == {}

def test_join_with_single_dict():
    d = {'a': 1}
    assert join_with(sum, [d]) is d

def test_join_with_single_dict_strict():
    d = {'a': 1}
    result = join_with(sum, [d], strict=True)
    assert result == {'a': 1}
    assert result is not d

def test_join_with_combines():
    result = join_with(sum, [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}])
    assert result == {'a': 4, 'b': 2, 'c': 4}

def test_join_with_identity():
    result = join_with(list, [{'a': 1, 'b': 2}, {'a': 3}])
    assert result == {'a': [1, 3], 'b': [2]}

def test_join_with_non_list_func():
    result = join_with(tuple, [{'a': [1, 2]}, {'a': [3]}])
    assert result == {'a': (1, 2, 3)} or result == {'a': ([1, 2], [3])}

def test_merge_with():
    result = merge_with(sum, {'a': 1}, {'a': 2}, {'a': 3})
    assert result == {'a': 6}

def test_merge_with_list():
    result = merge_with(list, {'a': 1}, {'a': 2})
    assert result == {'a': [1, 2]}


# --- walk -------------------------------------------------------------------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

def test_walk_dict_items():
    result = walk(lambda kv: (kv[0], kv[1] * 2), {'a': 1, 'b': 2})
    assert result == {'a': 2, 'b': 4}

def test_walk_set():
    assert walk(lambda x: x + 1, {1, 2, 3}) == {2, 3, 4}

def test_walk_tuple():
    assert walk(lambda x: x + 1, (1, 2)) == (2, 3)

def test_walk_string():
    assert walk(str.upper, 'abc') == 'ABC'


# --- walk_keys / walk_values ------------------------------------------------

def test_walk_keys():
    result = walk_keys(str.upper, {'a': 1, 'b': 2})
    assert result == {'A': 1, 'B': 2}

def test_walk_keys_with_int():
    result = walk_keys(0, {('x', 'y'): 1})
    # int key means itemgetter -> for dict it's items keyed? Actually walk_keys on dict uses items
    # Here f is itemgetter(0) applied to key
    assert result == {'x': 1}

def test_walk_values():
    result = walk_values(lambda x: x * 2, {'a': 1, 'b': 2})
    assert result == {'a': 2, 'b': 4}

def test_walk_values_list():
    assert walk_values(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

def test_walk_values_with_string():
    result = walk_values(str.upper, {'a': 'x', 'b': 'y'})
    assert result == {'a': 'X', 'b': 'Y'}


# --- select / select_keys / select_values -----------------------------------

def test_select_list():
    assert select(lambda x: x > 1, [1, 2, 3]) == [2, 3]

def test_select_dict():
    result = select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2, 'c': 3})
    assert result == {'b': 2, 'c': 3}

def test_select_set():
    assert select(lambda x: x % 2 == 0, {1, 2, 3, 4}) == {2, 4}

def test_select_keys():
    result = select_keys(lambda k: k.startswith('a'), {'a1': 1, 'b': 2, 'a2': 3})
    assert result == {'a1': 1, 'a2': 3}

def test_select_values():
    result = select_values(lambda v: v > 1, {'a': 1, 'b': 2, 'c': 3})
    assert result == {'b': 2, 'c': 3}

def test_select_keys_with_containment():
    result = select_keys({'a', 'b'}, {'a': 1, 'b': 2, 'c': 3})
    assert result == {'a': 1, 'b': 2}


# --- split_keys -------------------------------------------------------------

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'a1': 1, 'b': 2, 'a2': 3})
    assert yes == {'a1': 1, 'a2': 3}
    assert no == {'b': 2}

def test_split_keys_all_pass():
    yes, no = split_keys(lambda k: True, {'a': 1, 'b': 2})
    assert yes == {'a': 1, 'b': 2}
    assert no == {}

def test_split_keys_none_pass():
    yes, no = split_keys(lambda k: False, {'a': 1})
    assert yes == {}
    assert no == {'a': 1}


# --- compact ----------------------------------------------------------------

def test_compact_list():
    assert compact([0, 1, None, 2, '', 3, False]) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 1, 'b': 0, 'c': None, 'd': 2}) == {'a': 1, 'd': 2}

def test_compact_set():
    assert compact({0, 1, 2, None}) == {1, 2}

def test_compact_empty():
    assert compact([]) == []


# --- is_distinct ------------------------------------------------------------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 2]) is False

def test_is_distinct_with_key():
    assert is_distinct([1, 2, 3], key=lambda x: x % 3) is True
    assert is_distinct([1, 2, 4], key=lambda x: x % 3) is False

def test_is_distinct_empty():
    assert is_distinct([]) is True

def test_is_distinct_string_key():
    data = ['apple', 'banana', 'cherry']
    assert is_distinct(data, key=0) is True


# --- all / any / none / one / some ------------------------------------------

def test_all_without_seq():
    assert all([True, True]) is True
    assert all([True, False]) is False

def test_all_with_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2, 3]) is False

def test_any_without_seq():
    assert any([0, 1]) is True
    assert any([0, 0]) is False

def test_any_with_seq():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False

def test_none_with_seq():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 2, [1, 2, 3]) is False

def test_none_without_seq():
    assert none([0, 0]) is True
    assert none([0, 1]) is False

def test_one_exactly_one():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 5, [1, 2, 3]) is False

def test_one_without_seq():
    assert one([0, 1]) is True
    assert one([0, 0]) is False
    assert one([1, 1]) is False

def test_some_with_seq():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_without_seq():
    assert some([0, 2, 3]) == 2
    assert some([0, 0]) is None


# --- zipdict / flip ---------------------------------------------------------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_shorter():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}

def test_flip():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {1: 'a', 2: 'b'}


# --- project / omit ---------------------------------------------------------

def test_project():
    result = project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    assert result == {'a': 1, 'c': 3}

def test_project_missing_keys():
    result = project({'a': 1}, ['a', 'missing'])
    assert result == {'a': 1}

def test_omit():
    result = omit({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    assert result == {'b': 2}

def test_omit_missing_keys():
    result = omit({'a': 1}, ['z'])
    assert result == {'a': 1}


# --- zip_values / zip_dicts -------------------------------------------------

def test_zip_values():
    result = list(zip_values({'a': 1, 'b': 2}, {'a': 3, 'c': 4}))
    assert result == [(1, 3)]

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_values_single():
    result = list(zip_values({'a': 1, 'b': 2}))
    assert set(result) == {(1,), (2,)}

def test_zip_dicts():
    result = dict(zip_dicts({'a': 1, 'b': 2}, {'a': 3, 'b': 4}))
    assert result == {'a': (1, 3), 'b': (2, 4)}

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# --- get_in / get_lax -------------------------------------------------------

def test_get_in_nested():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(data, ['a', 'b', 'c']) == 1

def test_get_in_missing():
    data = {'a': {'b': 1}}
    assert get_in(data, ['a', 'x'], default='D') == 'D'

def test_get_in_list():
    data = [[1, 2], [3, 4]]
    assert get_in(data, [1, 0]) == 3

def test_get_in_index_out_of_range():
    assert get_in([[1]], [0, 5], default=None) is None

def test_get_in_empty_path():
    data = {'a': 1}
    assert get_in(data, []) == data

def test_get_lax_wrong_type():
    data = {'a': 1}
    assert get_lax(data, ['a', 'b'], default='D') == 'D'

def test_get_lax_ok():
    data = {'a': {'b': 1}}
    assert get_lax(data, ['a', 'b']) == 1

def test_get_in_wrong_type_raises():
    with pytest.raises(TypeError):
        get_in({'a': 1}, ['a', 'b'])


# --- set_in / update_in / del_in / has_path --------------------------------

def test_set_in_simple():
    data = {'a': 1}
    result = set_in(data, ['a'], 2)
    assert result == {'a': 2}
    assert data == {'a': 1}

def test_set_in_nested():
    data = {'a': {'b': 1}}
    result = set_in(data, ['a', 'b'], 2)
    assert result == {'a': {'b': 2}}
    assert data == {'a': {'b': 1}}

def test_set_in_new_path():
    data = {}
    result = set_in(data, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}

def test_set_in_list():
    data = [[1, 2], [3, 4]]
    result = set_in(data, [0, 1], 5)
    assert result == [[1, 5], [3, 4]]
    assert data == [[1, 2], [3, 4]]

def test_set_in_empty_path():
    result = set_in({'a': 1}, [], {'b': 2})
    assert result == {'b': 2}

def test_update_in():
    data = {'a': {'b': 1}}
    result = update_in(data, ['a', 'b'], lambda x: x + 10)
    assert result == {'a': {'b': 11}}

def test_update_in_default():
    data = {}
    result = update_in(data, ['a', 'b'], lambda x: x + 1, default=0)
    assert result == {'a': {'b': 1}}

def test_del_in():
    data = {'a': 1, 'b': 2}
    result = del_in(data, ['a'])
    assert result == {'b': 2}
    assert data == {'a': 1, 'b': 2}

def test_del_in_nested():
    data = {'a': {'b': 1, 'c': 2}}
    result = del_in(data, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert data == {'a': {'b': 1, 'c': 2}}

def test_del_in_list():
    data = [1, 2, 3]
    result = del_in(data, [1])
    assert result == [1, 3]
    assert data == [1, 2, 3]

def test_del_in_missing_returns_same():
    data = {'a': 1}
    assert del_in(data, ['z']) is data

def test_del_in_empty_path():
    data = {'a': 1}
    assert del_in(data, []) is data

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True

def test_has_path_false_key():
    assert has_path({'a': {'b': 1}}, ['a', 'z']) is False

def test_has_path_false_index():
    assert has_path([[1]], [0, 5]) is False

def test_has_path_empty():
    assert has_path({'a': 1}, []) is True


# --- where / pluck / pluck_attr / invoke -----------------------------------

def test_where():
    mappings = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 2}]
    result = list(where(mappings, a=1))
    assert result == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

def test_where_multiple_conditions():
    mappings = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 2}]
    result = list(where(mappings, a=1, b=2))
    assert result == [{'a': 1, 'b': 2}]

def test_where_missing_key():
    mappings = [{'a': 1}, {'b': 2}]
    result = list(where(mappings, a=1))
    assert result == [{'a': 1}]

def test_lwhere():
    mappings = [{'a': 1}, {'a': 2}]
    result = lwhere(mappings, a=1)
    assert result == [{'a': 1}]
    assert isinstance(result, list)

def test_pluck():
    mappings = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', mappings)) == [1, 2]

def test_lpluck():
    mappings = [{'a': 1}, {'a': 2}]
    result = lpluck('a', mappings)
    assert result == [1, 2]
    assert isinstance(result, list)

def test_pluck_attr():
    class Obj:
        def __init__(self, v):
            self.v = v

    objs = [Obj(1), Obj(2)]
    assert list(pluck_attr('v', objs)) == [1, 2]

def test_lpluck_attr():
    class Obj:
        def __init__(self, v):
            self.v = v

    objs = [Obj(1), Obj(2)]
    result = lpluck_attr('v', objs)
    assert result == [1, 2]
    assert isinstance(result, list)

def test_invoke():
    class Obj:
        def foo(self, x, y=0):
            return x + y

    objs = [Obj(), Obj()]
    assert list(invoke(objs, 'foo', 1, y=2)) == [3, 3]

def test_linvoke():
    class Obj:
        def foo(self, x):
            return x * 2

    objs = [Obj(), Obj()]
    result = linvoke(objs, 'foo', 3)
    assert result == [6, 6]
    assert isinstance(result, list)
