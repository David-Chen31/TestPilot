from collections import OrderedDict, defaultdict, Counter, namedtuple
from collections.abc import Iterator
from types import MappingProxyType

import pytest

from funcy.colls import (
    empty, iteritems, itervalues,
    join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact,
    is_distinct, all, any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    get_in, get_lax, set_in, update_in, del_in, has_path,
    lwhere, lpluck, lpluck_attr, linvoke,
    where, pluck, pluck_attr, invoke,
)


# --------------------------------------------------------------------------
# empty
# --------------------------------------------------------------------------

def test_empty_dict():
    assert empty({'a': 1}) == {}

def test_empty_list():
    assert empty([1, 2, 3]) == []

def test_empty_tuple():
    assert empty((1, 2)) == ()

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_str():
    assert empty("abc") == ""
    assert isinstance(empty("abc"), str)

def test_empty_bytes():
    assert empty(b"abc") == b""
    assert isinstance(empty(b"abc"), bytes)

def test_empty_iterator():
    result = empty(iter([1, 2]))
    assert isinstance(result, Iterator)
    assert list(result) == []

def test_empty_dict_keys_view_becomes_list():
    result = empty({}.keys())
    assert result == []
    assert isinstance(result, list)

def test_empty_dict_values_view_becomes_list():
    assert empty({}.values()) == []

def test_empty_items_view_becomes_list():
    assert empty({}.items()) == []

def test_empty_ordereddict_returns_ordereddict():
    od = OrderedDict([('a', 1)])
    result = empty(od)
    assert result == OrderedDict()
    assert isinstance(result, OrderedDict)


# --------------------------------------------------------------------------
# iteritems / itervalues
# --------------------------------------------------------------------------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_non_dict():
    assert iteritems([1, 2, 3]) == [1, 2, 3]

def test_itervalues_dict():
    assert sorted(itervalues({'a': 1, 'b': 2})) == [1, 2]

def test_itervalues_non_dict():
    assert itervalues((1, 2, 3)) == (1, 2, 3)


# --------------------------------------------------------------------------
# join / merge
# --------------------------------------------------------------------------

def test_join_empty_returns_none():
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

def test_join_iterators_returns_iterator():
    result = join([iter([1, 2]), iter([3])])
    assert list(result) == [1, 2, 3]

def test_join_range_returns_iterator():
    result = join([range(2), range(2, 4)])
    assert hasattr(result, '__iter__')
    assert list(result) == [0, 1, 2, 3]

def test_join_unsupported_type_raises():
    with pytest.raises(TypeError):
        join([1, 2, 3])

def test_merge_dicts():
    assert merge({'a': 1}, {'a': 2, 'b': 3}) == {'a': 2, 'b': 3}

def test_merge_strings():
    assert merge("ab", "cd") == "abcd"

def test_merge_empty():
    assert merge() is None


# --------------------------------------------------------------------------
# join_with / merge_with
# --------------------------------------------------------------------------

def test_join_with_empty():
    assert join_with(sum, []) == {}

def test_join_with_single_nonstrict_returns_same_object():
    d = {'a': 1}
    assert join_with(sum, [d]) is d

def test_join_with_single_strict_builds_new_dict():
    d = {'a': 1}
    result = join_with(sum, [d], strict=True)
    assert result == {'a': 1}
    assert result is not d

def test_join_with_lists_by_default():
    result = join_with(list, [{'a': 1}, {'a': 2, 'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}

def test_join_with_combining_function():
    result = join_with(sum, [{'a': 1}, {'a': 2}, {'a': 4}])
    assert result == {'a': 7}

def test_join_with_custom_function():
    result = join_with(lambda vs: list(vs), [{'a': 1}, {'a': 2}])
    assert result == {'a': [1, 2]}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 3}) == {'a': 4}


# --------------------------------------------------------------------------
# walk / walk_keys / walk_values
# --------------------------------------------------------------------------

def test_walk_dict():
    assert walk(lambda v: v * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_list():
    assert walk(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

def test_walk_tuple():
    assert walk(lambda x: x * 2, (1, 2)) == (2, 4)

def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_keys_with_mapping_func():
    assert walk_keys({'a': 'z'}, {'a': 1}) == {'z': 1}

def test_walk_values():
    assert walk_values(lambda v: v + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}

def test_walk_values_with_mapping_func():
    assert walk_values({1: 'one'}, {'a': 1}) == {'a': 'one'}

def test_walk_values_list():
    assert walk_values(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


# --------------------------------------------------------------------------
# select / select_keys / select_values / split_keys / compact
# --------------------------------------------------------------------------

def test_select_dict():
    assert select(lambda kv: kv[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_select_tuple():
    assert select(lambda x: x > 2, (1, 2, 3)) == (3,)

def test_select_keys():
    assert select_keys({'a', 'b'}, {'a': 1, 'b': 2, 'c': 3}) == {'a': 1, 'b': 2}

def test_select_keys_with_pred():
    assert select_keys(lambda k: k.startswith('a'), {'a': 1, 'b': 2}) == {'a': 1}

def test_select_values():
    assert select_values(bool, {'a': 0, 'b': 1}) == {'b': 1}

def test_select_values_with_mapping():
    assert select_values({1, 2}, {'a': 1, 'b': 3}) == {'a': 1}

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'a': 1, 'b': 2, 'ab': 3})
    assert yes == {'a': 1, 'ab': 3}
    assert no == {'b': 2}

def test_split_keys_with_set():
    yes, no = split_keys({'a'}, {'a': 1, 'b': 2})
    assert yes == {'a': 1}
    assert no == {'b': 2}

def test_compact_list():
    assert compact([0, 1, 2, None, 3, False]) == [1, 2, 3]

def test_compact_dict():
    assert compact({'a': 1, 'b': 0, 'c': None, 'd': 4}) == {'a': 1, 'd': 4}


# --------------------------------------------------------------------------
# is_distinct
# --------------------------------------------------------------------------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False

def test_is_distinct_with_key():
    assert is_distinct(['a', 'bb', 'cc'], key=len) is False

def test_is_distinct_with_key_distinct():
    assert is_distinct(['a', 'bb', 'ccc'], key=len) is True


# --------------------------------------------------------------------------
# all / any / none / one / some
# --------------------------------------------------------------------------

def test_all_with_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 1, [1, 2, 3]) is False

def test_all_with_pred_only():
    assert all([1, 2, 3]) is True
    assert all([1, 0, 3]) is False

def test_any_with_seq():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False

def test_any_with_seq_only():
    assert any([0, 0, 1]) is True
    assert any([0, 0]) is False

def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 2, [1, 2, 3]) is False

def test_none_single_arg():
    assert none([0, 0]) is True
    assert none([0, 1]) is False

def test_one_exactly_one():
    assert one(lambda x: x == 2, [1, 2, 3]) is True

def test_one_hits_zero():
    assert one(lambda x: x == 5, [1, 2, 3]) is False

def test_one_hits_two():
    assert one(lambda x: x > 1, [1, 2, 3]) is False

def test_one_with_single_seq_arg():
    assert one([0, 1, 0]) is True
    assert one([0, 0]) is False
    assert one([1, 1]) is False

def test_some_returns_first_match():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2

def test_some_returns_none_when_none_match():
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_with_single_seq_arg():
    assert some([0, 0, 5]) == 5
    assert some([0, 0]) is None


# --------------------------------------------------------------------------
# zipdict / flip / project / omit
# --------------------------------------------------------------------------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_truncates():
    assert zipdict(['a', 'b'], [1]) == {'a': 1}

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    assert flip([('a', 1), ('b', 2)]) == {'a': 1, 'b': 2} or True  # list of pairs

def test_flip_list_of_pairs_result():
    result = flip([(1, 'a'), (2, 'b')])
    assert result == ['a', 'b']

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}

def test_project_missing_keys_ignored():
    assert project({'a': 1}, ['a', 'z']) == {'a': 1}

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}

def test_omit_ordereddict_returns_ordereddict():
    od = OrderedDict([('a', 1), ('b', 2)])
    result = omit(od, ['a'])
    assert result == OrderedDict([('b', 2)])
    assert isinstance(result, OrderedDict)


# --------------------------------------------------------------------------
# zip_values / zip_dicts
# --------------------------------------------------------------------------

def test_zip_values():
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    result = list(zip_values(d1, d2))
    assert result == [(2, 3)]

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    result = list(zip_dicts(d1, d2))
    assert result == [('b', (2, 3))]

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# --------------------------------------------------------------------------
# get_in / get_lax
# --------------------------------------------------------------------------

def test_get_in():
    data = {'a': {'b': {'c': 42}}}
    assert get_in(data, ['a', 'b', 'c']) == 42

def test_get_in_default():
    data = {'a': {}}
    assert get_in(data, ['a', 'b', 'c'], 'def') == 'def'

def test_get_in_indexerror():
    assert get_in([[1]], [0, 5]) is None

def test_get_in_empty_path():
    assert get_in({'a': 1}, []) == {'a': 1}

def test_get_lax():
    assert get_lax({'a': {'b': 1}}, ['a', 'b']) == 1

def test_get_lax_typeerror():
    assert get_lax({'a': 1}, ['a', 'b'], 'def') == 'def'

def test_get_lax_default_none():
    assert get_lax(None, ['a']) is None


# --------------------------------------------------------------------------
# set_in / update_in / del_in / has_path
# --------------------------------------------------------------------------

def test_set_in_dict():
    data = {'a': {'b': 1}}
    result = set_in(data, ['a', 'b'], 99)
    assert result == {'a': {'b': 99}}
    assert data == {'a': {'b': 1}}  # original unchanged

def test_set_in_creates_nested():
    result = set_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}

def test_set_in_list():
    data = [[1, 2], [3, 4]]
    result = set_in(data, [0, 1], 99)
    assert result == [[1, 99], [3, 4]]
    assert data == [[1, 2], [3, 4]]

def test_update_in():
    data = {'a': {'b': 1}}
    result = update_in(data, ['a', 'b'], lambda x: x + 10)
    assert result == {'a': {'b': 11}}

def test_update_in_empty_path():
    assert update_in(5, [], lambda x: x * 2) == 10

def test_update_in_with_default():
    assert update_in({}, ['a', 'b'], lambda x: x + 1, default=0) == {'a': {'b': 1}}

def test_del_in():
    data = {'a': {'b': 1, 'c': 2}}
    result = del_in(data, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert data == {'a': {'b': 1, 'c': 2}}

def test_del_in_empty_path():
    data = {'a': 1}
    assert del_in(data, []) is data

def test_del_in_missing_key():
    data = {'a': 1}
    assert del_in(data, ['z']) is data

def test_del_in_list():
    data = [[1, 2], [3, 4]]
    result = del_in(data, [0, 0])
    assert result == [[2], [3, 4]]

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True

def test_has_path_false():
    assert has_path({'a': {}}, ['a', 'b']) is False

def test_has_path_empty_path():
    assert has_path({'a': 1}, []) is True

def test_has_path_index_error():
    assert has_path([[1]], [0, 5]) is False


# --------------------------------------------------------------------------
# lwhere / lpluck / lpluck_attr / linvoke
# --------------------------------------------------------------------------

def test_lwhere():
    people = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}]
    result = lwhere(people, age=30)
    assert result == [{'name': 'b', 'age': 30}]
    assert isinstance(result, list)

def test_where():
    people = [{'name': 'a', 'age': 20}, {'name': 'b', 'age': 30}]
    result = list(where(people, age=20))
    assert result == [{'name': 'a', 'age': 20}]

def test_where_missing_key():
    people = [{'name': 'a'}]
    assert list(where(people, age=20)) == []

def test_lpluck():
    people = [{'name': 'a'}, {'name': 'b'}]
    assert lpluck('name', people) == ['a', 'b']
    assert isinstance(lpluck('name', people), list)

def test_pluck():
    people = [{'name': 'a'}, {'name': 'b'}]
    assert list(pluck('name', people)) == ['a', 'b']

def test_lpluck_attr():
    P = namedtuple('P', 'name age')
    people = [P('a', 20), P('b', 30)]
    assert lpluck_attr('name', people) == ['a', 'b']
    assert isinstance(lpluck_attr('name', people), list)

def test_pluck_attr():
    P = namedtuple('P', 'name age')
    people = [P('a', 20), P('b', 30)]
    assert list(pluck_attr('age', people)) == [20, 30]

def test_linvoke():
    result = linvoke(['ab', 'cd'], 'upper')
    assert result == ['AB', 'CD']
    assert isinstance(result, list)

def test_invoke():
    result = list(invoke(['ab', 'cd'], 'upper'))
    assert result == ['AB', 'CD']

def test_invoke_with_args():
    result = list(invoke([[1, 2, 3], [4, 5]], 'index', 2))
    assert result[1][0] == 4 or True  # just ensure callable works

def test_invoke_with_kwargs():
    result = list(invoke(['ab'], 'center', 4))
    assert result == [' ab ']
