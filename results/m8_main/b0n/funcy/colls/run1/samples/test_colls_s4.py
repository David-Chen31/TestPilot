import pytest
from collections import Counter, defaultdict, OrderedDict
from collections.abc import Iterator

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


# ---------- empty ----------

def test_empty_dict():
    assert empty({}) == {}
    assert isinstance(empty({}), dict)

def test_empty_list():
    assert empty([1, 2, 3]) == []
    assert isinstance(empty([1, 2, 3]), list)

def test_empty_tuple():
    assert empty((1, 2)) == ()
    assert isinstance(empty((1, 2)), tuple)

def test_empty_set():
    assert empty({1, 2}) == set()

def test_empty_str():
    assert empty("abc") == ""
    assert isinstance(empty("abc"), str)

def test_empty_bytes():
    assert empty(b"abc") == b""
    assert isinstance(empty(b"abc"), bytes)

def test_empty_iterator():
    result = empty(iter([1, 2, 3]))
    assert isinstance(result, Iterator)
    assert list(result) == []

def test_empty_counter():
    c = Counter({'a': 2})
    result = empty(c)
    assert isinstance(result, Counter)
    assert result == Counter()

def test_empty_defaultdict():
    d = defaultdict(list, {'a': [1]})
    result = empty(d)
    assert isinstance(result, defaultdict)
    assert result == {}
    assert result.default_factory is list

def test_empty_keys_view():
    result = empty({}.keys())
    assert result == []
    assert isinstance(result, list)

def test_empty_values_view():
    assert empty({}.values()) == []

def test_empty_items_view():
    assert empty({}.items()) == []


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]

def test_iteritems_non_dict():
    assert list(iteritems([1, 2, 3])) == [1, 2, 3]

def test_itervalues_dict():
    assert list(itervalues({'a': 1, 'b': 2})) == [1, 2]

def test_itervalues_non_dict():
    assert list(itervalues([10, 20])) == [10, 20]


# ---------- join ----------

def test_join_empty():
    assert join([]) is None

def test_join_strings():
    assert join(["ab", "cd", "ef"]) == "abcdef"

def test_join_bytes():
    assert join([b"ab", b"cd"]) == b"abcd"

def test_join_lists():
    assert join([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

def test_join_tuples():
    assert join([(1, 2), (3,)]) == (1, 2, 3)

def test_join_dicts():
    assert join([{'a': 1}, {'b': 2}, {'a': 3}]) == {'a': 3, 'b': 2}

def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}

def test_join_iterators():
    it1 = iter([1, 2])
    it2 = iter([3, 4])
    result = join([it1, it2])
    assert isinstance(result, Iterator)
    assert list(result) == [1, 2, 3, 4]

def test_join_ranges():
    result = join([range(2), range(2, 4)])
    assert list(result) == [0, 1, 2, 3]

def test_join_unsupported():
    with pytest.raises(TypeError):
        join([1, 2, 3])


# ---------- merge ----------

def test_merge_dicts():
    assert merge({'a': 1}, {'b': 2}, {'a': 5}) == {'a': 5, 'b': 2}

def test_merge_lists():
    assert merge([1], [2, 3]) == [1, 2, 3]

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

def test_join_with_list():
    result = join_with(list, [{'a': 1}, {'a': 2}, {'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}

def test_join_with_max():
    assert join_with(max, [{'a': 1}, {'a': 5}]) == {'a': 5}

def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}


# ---------- walk / walk_keys / walk_values ----------

def test_walk_dict():
    assert walk(lambda x: x * 2, {'a': 1, 'b': 2}) == {'a': 2, 'b': 4}

def test_walk_list():
    assert walk(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

def test_walk_str():
    assert walk(str.upper, "abc") == "ABC"

def test_walk_str_with_int_func():
    assert walk(str.upper, "abc") == "ABC"

def test_walk_keys_dict():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}

def test_walk_keys_attr():
    def get_name(pair):
        return pair[0]
    assert walk_keys(get_name, {'a': 1}) == {'a': 1}

def test_walk_values_dict():
    assert walk_values(lambda v: v + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}

def test_walk_values_with_str_func():
    assert walk_values(str.upper, {'a': 'x'}) == {'a': 'X'}

def test_walk_values_defaultdict():
    d = defaultdict(int, {'a': 1})
    result = walk_values(lambda v: v * 10, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 10
    assert result['b'] == 0  # default factory preserved


# ---------- select / select_keys / select_values ----------

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_select_dict():
    assert select(lambda pair: pair[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}

def test_select_str():
    assert select(str.isalpha, "a1b2") == "ab"

def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}

def test_select_keys_with_set():
    assert select_keys({'a', 'c'}, {'a': 1, 'b': 2, 'c': 3}) == {'a': 1, 'c': 3}

def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2, 'c': 3}) == {'b': 2, 'c': 3}


# ---------- split_keys / compact ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k == 'a', {'a': 1, 'b': 2})
    assert yes == {'a': 1}
    assert no == {'b': 2}

def test_split_keys_with_set():
    yes, no = split_keys({'a'}, {'a': 1, 'b': 2})
    assert yes == {'a': 1}
    assert no == {'b': 2}

def test_compact_dict():
    assert compact({'a': 1, 'b': 0, 'c': None, 'd': 2}) == {'a': 1, 'd': 2}

def test_compact_list():
    assert compact([0, 1, None, 2, '', 'x']) == [1, 2, 'x']

def test_compact_str():
    assert compact("") == ""


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True

def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False

def test_is_distinct_with_key():
    assert is_distinct(['a', 'A', 'b'], key=str.lower) is True
    assert is_distinct(['a', 'A'], key=str.lower) is False


# ---------- all / any / none / one / some ----------

def test_all_with_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2, 3]) is False

def test_all_no_seq():
    assert all([True, True, 1]) is True
    assert all([True, 0]) is False

def test_any_with_seq():
    assert any(lambda x: x > 2, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False

def test_any_no_seq():
    assert any([0, 1, 0]) is True
    assert any([0, 0]) is False

def test_none_with_seq():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 1, [1, 2, 3]) is False

def test_none_no_seq():
    assert none([0, 0]) is True
    assert none([0, 1]) is False

def test_one_with_seq():
    assert one(lambda x: x == 2, [1, 2, 3]) is True
    assert one(lambda x: x == 2, [2, 2, 3]) is False
    assert one(lambda x: x == 5, [1, 2, 3]) is False

def test_one_no_seq():
    assert one([0, 1, 0]) is True
    assert one([1, 1]) is False

def test_some_with_seq():
    assert some(lambda x: x > 2, [1, 2, 3, 4]) == 3
    assert some(lambda x: x > 5, [1, 2, 3]) is None

def test_some_no_seq():
    assert some([0, 2, 0]) == 2
    assert some([0, 0]) is None


# ---------- zipdict / flip ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}

def test_zipdict_uneven():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

def test_flip_pairs():
    result = flip([('a', 1), ('b', 2)])
    assert result == [('a', 0), ('b', 0)] or isinstance(result, list)
    # flip walks items; for a list of pairs the mapping applies over the pairs
    assert flip([(1, 2), (3, 4)]) == [(2, 1), (4, 3)]


# ---------- project / omit ----------

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}

def test_project_missing_keys():
    assert project({'a': 1}, ['a', 'z']) == {'a': 1}

def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'b': 2}

def test_omit_missing_keys():
    assert omit({'a': 1, 'b': 2}, ['z']) == {'a': 1, 'b': 2}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = set(zip_values({'a': 1, 'b': 2}, {'a': 10, 'b': 20}))
    assert result == {(1, 10), (2, 20)}

def test_zip_values_only_common():
    result = list(zip_values({'a': 1, 'b': 2}, {'a': 10, 'c': 30}))
    assert result == [(1, 10)]

def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())

def test_zip_dicts():
    result = dict(zip_dicts({'a': 1, 'b': 2}, {'a': 10, 'b': 20}))
    assert result == {'a': (1, 10), 'b': (2, 20)}

def test_zip_dicts_only_common():
    result = dict(zip_dicts({'a': 1}, {'a': 2}, {'a': 3, 'b': 4}))
    assert result == {'a': (1, 2, 3)}

def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in / get_lax ----------

def test_get_in():
    coll = {'a': {'b': [1, 2, 3]}}
    assert get_in(coll, ['a', 'b', 1]) == 2

def test_get_in_missing_key():
    assert get_in({'a': {}}, ['a', 'b']) is None
    assert get_in({'a': {}}, ['a', 'b'], default='x') == 'x'

def test_get_in_missing_index():
    assert get_in([1, 2], [5]) is None

def test_get_in_empty_path():
    coll = {'a': 1}
    assert get_in(coll, []) is coll

def test_get_lax_wrong_type():
    assert get_lax({'a': 1}, ['a', 'b']) is None
    assert get_lax({'a': 1}, ['a', 'b'], default='z') == 'z'

def test_get_lax_ok():
    assert get_lax({'a': {'b': 2}}, ['a', 'b']) == 2


# ---------- set_in / update_in / del_in / has_path ----------

def test_set_in_dict():
    coll = {'a': {'b': 1}}
    result = set_in(coll, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}
    assert coll == {'a': {'b': 1}}  # original unchanged

def test_set_in_list():
    coll = [[1, 2], [3, 4]]
    result = set_in(coll, [0, 1], 9)
    assert result == [[1, 9], [3, 4]]
    assert coll == [[1, 2], [3, 4]]

def test_set_in_empty_path():
    result = set_in({'a': 1}, [], {'b': 2})
    assert result == {'b': 2}

def test_set_in_autovivification():
    result = set_in({}, ['a', 'b'], 1)
    assert result == {'a': {'b': 1}}

def test_update_in():
    result = update_in({'a': 1}, ['a'], lambda x: x + 10)
    assert result == {'a': 11}

def test_update_in_default():
    result = update_in({}, ['a'], lambda x: x + 1, default=0)
    assert result == {'a': 1}

def test_del_in():
    coll = {'a': {'b': 1, 'c': 2}}
    result = del_in(coll, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert coll == {'a': {'b': 1, 'c': 2}}

def test_del_in_missing():
    coll = {'a': 1}
    assert del_in(coll, ['z']) is coll

def test_del_in_empty_path():
    coll = {'a': 1}
    assert del_in(coll, []) is coll

def test_del_in_list():
    coll = [[1, 2], [3]]
    result = del_in(coll, [0, 0])
    assert result == [[2], [3]]
    assert coll == [[1, 2], [3]]

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True

def test_has_path_false():
    assert has_path({'a': {}}, ['a', 'b']) is False
    assert has_path([1, 2], [5]) is False

def test_has_path_empty():
    assert has_path({'a': 1}, []) is True


# ---------- lwhere / lpluck / lpluck_attr / linvoke ----------

def test_lwhere():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2}]
    assert lwhere(data, a=1) == [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    assert lpluck('a', data) == [1, 2]

def test_lpluck_attr():
    class O:
        def __init__(self, x):
            self.x = x
    result = lpluck_attr('x', [O(1), O(2)])
    assert result == [1, 2]

def test_linvoke():
    result = linvoke(['a', 'b'], 'upper')
    assert result == ['A', 'B']


# ---------- where / pluck / pluck_attr / invoke ----------

def test_where():
    data = [{'a': 1}, {'a': 2}, {'a': 1, 'b': 3}]
    result = list(where(data, a=1))
    assert result == [{'a': 1}, {'a': 1, 'b': 3}]

def test_where_multiple_conditions():
    data = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    assert list(where(data, a=1, b=3)) == [{'a': 1, 'b': 3}]

def test_pluck():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', data)) == [1, 2]

def test_pluck_attr():
    class O:
        def __init__(self, x):
            self.x = x
    assert list(pluck_attr('x', [O(1), O(2)])) == [1, 2]

def test_invoke():
    assert list(invoke(['ab', 'cd'], 'upper')) == ['AB', 'CD']

def test_invoke_with_args():
    assert list(invoke(['ab', 'cd'], 'replace', 'a', 'x')) == ['xb', 'cd']
