import pytest
from collections import Counter, defaultdict, OrderedDict

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
    result = empty(iter([1, 2, 3]))
    assert list(result) == []


def test_empty_dict_keys_view():
    assert empty({1: 2}.keys()) == []


def test_empty_dict_values_view():
    assert empty({1: 2}.values()) == []


def test_empty_dict_items_view():
    assert empty({1: 2}.items()) == []


def test_empty_defaultdict():
    d = defaultdict(list, {'a': [1]})
    result = empty(d)
    assert isinstance(result, defaultdict)
    assert result == {}
    assert result.default_factory is list


def test_empty_counter():
    c = Counter('aab')
    result = empty(c)
    assert isinstance(result, Counter)
    assert result == Counter()


def test_empty_generator_returns_iterator():
    # empty() on a generic Iterator returns iter([]), not the original type
    gen = (x for x in [1, 2, 3])
    result = empty(gen)
    assert iter(result) is result
    assert list(result) == []


# ---------- iteritems / itervalues ----------

def test_iteritems_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]


def test_iteritems_no_items():
    assert list(iteritems([('a', 1), ('b', 2)])) == [('a', 1), ('b', 2)]


def test_itervalues_dict():
    assert list(itervalues({'a': 1, 'b': 2})) == [1, 2]


def test_itervalues_no_values():
    assert list(itervalues([1, 2, 3])) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_empty():
    assert join([]) is None


def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'


def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'


def test_join_lists():
    assert join([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_join_tuples():
    assert join([(1, 2), (3,)]) == (1, 2, 3)


def test_join_dicts():
    assert join([{'a': 1}, {'b': 2}, {'a': 3}]) == {'a': 3, 'b': 2}


def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}


def test_join_iterators():
    result = join([iter([1, 2]), iter([3, 4])])
    assert list(result) == [1, 2, 3, 4]


def test_join_range():
    result = join([range(3), range(3, 5)])
    assert list(result) == [0, 1, 2, 3, 4]


def test_join_unsupported():
    class Foo:
        def __init__(self, v):
            self.v = v
    with pytest.raises(TypeError):
        join([Foo(1), Foo(2)])


def test_merge():
    assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge([1, 2], [3]) == [1, 2, 3]
    assert merge('ab', 'cd') == 'abcd'


# ---------- join_with / merge_with ----------

def test_join_with_empty():
    assert join_with(sum, []) == {}


def test_join_with_single_non_strict():
    d = {'a': 1}
    assert join_with(sum, [d]) is d


def test_join_with_single_strict():
    assert join_with(sum, [{'a': 1}], strict=True) == {'a': 1}


def test_join_with_sum():
    result = join_with(sum, [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}])
    assert result == {'a': 4, 'b': 2, 'c': 4}


def test_join_with_list_identity():
    result = join_with(list, [{'a': 1}, {'a': 2, 'b': 3}])
    assert result == {'a': [1, 2], 'b': [3]}


def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}


# ---------- walk ----------

def test_walk_dict():
    # walk applies f to (key, value) pairs; f must return a pair
    result = walk(lambda p: (p[0], p[1] * 2), {'a': 1, 'b': 2})
    assert result == {'a': 2, 'b': 4}


def test_walk_list_pairs():
    # A list has no .items(), so iteritems returns the list itself.
    # walk therefore applies f to each element individually, and the
    # factory (list) collects the results.
    result = walk(lambda x: x * 2, [10, 20])
    assert result == [20, 40]


def test_walk_string_regex():
    # For a list of strings, factory is list; walk treats each string as its
    # own iterable of items via iteritems (no .items() -> coll itself).
    # Applying a regex finder on the whole string gives first match.
    result = walk(r'\d', ['a1', 'b2'])
    assert result == ['1', '2']


def test_walk_str_factory_join():
    # walk on a str uses str factory (coll_type().join) over the mapped items
    result = walk(str.upper, 'abc')
    assert result == 'ABC'


def test_walk_bytes_factory_ints():
    # walk on bytes uses bytes_factory; iteritems(bytes) yields ints, mapper
    # returns ints, so bytes_factory(items) rebuilds from ints.
    result = walk(lambda x: x + 1, b'AB')
    assert result == b'BC'


def test_walk_bytes_factory_bytes():
    # mapper returns bytes objects; bytes_factory joins them
    result = walk(lambda x: bytes([x]), b'AB')
    assert result == b'AB'


def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_walk_keys_funcmaker():
    # int as mapper is itemgetter; each key is a tuple like ('k1', 'v1')
    # walking a list preserves list factory (with pair_f).
    result = walk_keys(0, [(('k1', 'v1'), 1), (('k2', 'v2'), 2)])
    assert result == [('k1', 1), ('k2', 2)]


def test_walk_keys_tuple_keys():
    # walk_keys maps each key with f. Using a dict of tuple keys where each
    # key is length 2 would unpack into two args to str.upper -> TypeError.
    # Instead, use a callable f on tuple keys, e.g. reversed.
    result = walk_keys(lambda k: tuple(reversed(k)), {('a', 'b'): 1})
    assert result == {('b', 'a'): 1}


def test_walk_values():
    assert walk_values(lambda x: x + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}


def test_walk_values_on_bytes_raises():
    # walk_values calls pair_f on each item of iteritems(coll). For bytes,
    # iteritems returns the bytes object, whose iteration yields ints.
    # pair_f tries to unpack each int into (k, v) -> TypeError.
    with pytest.raises(TypeError):
        walk_values(lambda x: x + 1, b'AB')


def test_walk_values_on_str_raises():
    # For str, iteritems returns the str; iterating yields 1-char strings.
    # pair_f unpacks each char into (k, v); a 1-char string cannot be
    # unpacked into two names -> ValueError.
    with pytest.raises(ValueError):
        walk_values(str.upper, 'abc')


def test_walk_values_str_factory():
    # walk_values on a dict with str values
    assert walk_values(str.upper, {'a': 'x', 'b': 'y'}) == {'a': 'X', 'b': 'Y'}


def test_walk_values_defaultdict():
    # walk_values builds a defaultdict whose default_factory is
    # compose(len, list). Existing values are put in as-is (dict.update
    # does not invoke the factory), so the mapping equals the input with
    # values mapped through len.
    d = defaultdict(list, {'a': [1], 'b': [2]})
    result = walk_values(len, d)
    assert isinstance(result, defaultdict)
    assert result == {'a': 1, 'b': 1}
    # The default factory is composed with the mapper, so missing keys
    # produce a length of an empty list.
    assert result['missing'] == 0


# ---------- select ----------

def test_select_dict():
    assert select(lambda p: p[1] > 1, {'a': 1, 'b': 2, 'c': 3}) == {'b': 2, 'c': 3}


def test_select_list():
    assert select(lambda p: p[1] > 1, [(1, 1), (2, 2), (3, 3)]) == [(2, 2), (3, 3)]


def test_select_string_factory():
    # list of strings, factory is list
    assert select(lambda p: p[0].startswith('a'), ['ab', 'bc', 'ad']) == ['ab', 'ad']


def test_select_keys():
    assert select_keys(lambda k: k.startswith('a'), {'a': 1, 'b': 2, 'aa': 3}) == {'a': 1, 'aa': 3}


def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2, 'c': 3}) == {'b': 2, 'c': 3}


def test_select_values_funcmaker():
    # string as pred is regex tester
    assert select_values(r'\d', {'a': '1', 'b': 'x'}) == {'a': '1'}


def test_select_str_factory():
    # select on a str uses the str factory (coll_type().join). iteritems(str)
    # yields characters, each of which is a 1-char "pair". select with a pred
    # over characters keeps matching ones; joined back.
    result = select(lambda p: p in 'aeiou', 'hello')
    # each element 'p' is a single char; truthy test -> keeps 'e','o'
    assert result == 'eo'


def test_select_bytes_factory():
    # select on bytes uses bytes_factory (no mapper). iteritems(bytes) yields
    # ints. Filtering keeps ints, rebuilt into bytes.
    result = select(lambda x: x > 65, b'ABC')
    assert result == b'BC'


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k.startswith('a'), {'a': 1, 'b': 2, 'ab': 3})
    assert yes == {'a': 1, 'ab': 3}
    assert no == {'b': 2}


def test_split_keys_preserves_type():
    d = defaultdict(int, {'a': 1, 'b': 2})
    yes, no = split_keys(lambda k: k == 'a', d)
    assert isinstance(yes, defaultdict)
    assert isinstance(no, defaultdict)


# ---------- compact ----------

def test_compact_mapping():
    assert compact({'a': 1, 'b': 0, 'c': 2, 'd': None}) == {'a': 1, 'c': 2}


def test_compact_list():
    assert compact([0, 1, 2, None, 3, '']) == [1, 2, 3]


def test_compact_list_pairs():
    # items are tuples; all non-empty tuples are truthy
    assert compact([(0, 1), (1, 0), (2, 3)]) == [(0, 1), (1, 0), (2, 3)]
    assert compact([(0, 1), (), (2, 3)]) == [(0, 1), (2, 3)]


# ---------- is_distinct ----------

def test_is_distinct_true():
    assert is_distinct([1, 2, 3]) is True


def test_is_distinct_false():
    assert is_distinct([1, 2, 1]) is False


def test_is_distinct_with_key():
    # key maps each element with xmap (no funcmaker semantics issue here since
    # callable), so use equal case-insensitive values
    assert is_distinct(['a', 'B', 'c'], key=str.lower) is True
    assert is_distinct(['a', 'A', 'b'], key=str.lower) is False


# ---------- all / any / none / one / some ----------

def test_all_with_seq():
    assert all(lambda x: x > 0, [1, 2, 3]) is True
    assert all(lambda x: x > 0, [1, -2, 3]) is False


def test_all_no_seq():
    assert all([True, True]) is True
    assert all([True, False]) is False


def test_any_with_seq():
    assert any(lambda x: x > 1, [1, 2, 3]) is True
    assert any(lambda x: x > 5, [1, 2, 3]) is False


def test_any_no_seq():
    assert any([False, True]) is True
    assert any([False, False]) is False


def test_none():
    assert none(lambda x: x > 5, [1, 2, 3]) is True
    assert none(lambda x: x > 1, [1, 2, 3]) is False


def test_none_no_seq():
    assert none([False, False]) is True
    assert none([True, False]) is False


def test_one_with_seq():
    assert one(lambda x: x > 2, [1, 2, 3]) is True
    assert one(lambda x: x > 1, [1, 2, 3]) is False
    assert one(lambda x: x > 5, [1, 2, 3]) is False


def test_one_no_seq():
    assert one([False, True, False]) is True
    assert one([True, True]) is False


def test_one_early_exit():
    # Ideally won't consume whole generator; here just verify value
    assert one(lambda x: x == 1, [1, 2, 3, 4, 5]) is True


def test_some_with_seq():
    assert some(lambda x: x > 1, [1, 2, 3]) == 2
    assert some(lambda x: x > 5, [1, 2, 3]) is None


def test_some_no_seq():
    assert some([0, 0, 5, 0]) == 5
    assert some([0, 0, 0]) is None


# ---------- zipdict ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}


def test_zipdict_short():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}


# ---------- flip ----------

def test_flip_dict():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_flip_pairs_list():
    # flip on a list of pairs walks each pair; result is a list of reversed pairs
    result = flip([('a', 1), ('b', 2)])
    assert result == [(1, 'a'), (2, 'b')]


# ---------- project / omit ----------

def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}


def test_project_missing_key():
    assert project({'a': 1}, ['a', 'b']) == {'a': 1}


def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'b': 2}


def test_omit_preserves_type():
    d = OrderedDict([('a', 1), ('b', 2)])
    result = omit(d, ['a'])
    assert isinstance(result, OrderedDict)


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = list(zip_values({'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}))
    assert sorted(result) == [(1, 3), (2, 4)]


def test_zip_values_no_args():
    with pytest.raises(TypeError):
        list(zip_values())


def test_zip_dicts():
    result = list(zip_dicts({'a': 1, 'b': 2}, {'a': 3, 'b': 4}))
    assert sorted(result) == [('a', (1, 3)), ('b', (2, 4))]


def test_zip_dicts_no_args():
    with pytest.raises(TypeError):
        list(zip_dicts())


# ---------- get_in ----------

def test_get_in():
    d = {'a': {'b': [10, 20]}}
    assert get_in(d, ['a', 'b', 1]) == 20


def test_get_in_default():
    d = {'a': {'b': 1}}
    assert get_in(d, ['a', 'c']) is None
    assert get_in(d, ['a', 'c'], default='X') == 'X'


def test_get_in_empty_path():
    d = {'a': 1}
    assert get_in(d, []) is d


def test_get_lax_on_type_error():
    assert get_lax({'a': 1}, ['a', 'b'], default='X') == 'X'


def test_get_lax_success():
    assert get_lax({'a': {'b': 2}}, ['a', 'b']) == 2


def test_get_lax_index_error():
    assert get_lax([1, 2], [5], default='Y') == 'Y'


# ---------- set_in / update_in ----------

def test_set_in():
    d = {'a': {'b': 1}}
    result = set_in(d, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}
    assert d == {'a': {'b': 1}}  # original unchanged


def test_set_in_autovivify():
    d = {}
    result = set_in(d, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}


def test_set_in_list():
    lst = [[1, 2], [3, 4]]
    result = set_in(lst, [0, 1], 9)
    assert result == [[1, 9], [3, 4]]
    assert lst == [[1, 2], [3, 4]]


def test_update_in():
    d = {'a': 1}
    assert update_in(d, ['a'], lambda x: x + 1) == {'a': 2}


def test_update_in_empty_path():
    assert update_in(5, [], lambda x: x * 2) == 10


def test_update_in_with_default():
    d = {}
    result = update_in(d, ['a'], lambda x: x + 1, default=0)
    assert result == {'a': 1}


# ---------- del_in ----------

def test_del_in():
    d = {'a': {'b': 1, 'c': 2}}
    result = del_in(d, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert d == {'a': {'b': 1, 'c': 2}}


def test_del_in_empty_path():
    d = {'a': 1}
    assert del_in(d, []) == d


def test_del_in_missing_key():
    d = {'a': 1}
    assert del_in(d, ['b']) == d


def test_del_in_list():
    lst = [[1, 2], [3, 4]]
    result = del_in(lst, [0, 0])
    assert result == [[2], [3, 4]]
    assert lst == [[1, 2], [3, 4]]


def test_del_in_list_index_error():
    lst = [1, 2]
    assert del_in(lst, [5]) == lst


# ---------- has_path ----------

def test_has_path_true():
    assert has_path({'a': {'b': 1}}, ['a', 'b']) is True


def test_has_path_false():
    assert has_path({'a': {'b': 1}}, ['a', 'c']) is False


def test_has_path_list_index():
    assert has_path([1, [10, 20]], [1, 0]) is True
    assert has_path([1, [10, 20]], [1, 5]) is False


def test_has_path_empty():
    assert has_path({'a': 1}, []) is True


# ---------- where / lwhere ----------

def test_where():
    items = [
        {'name': 'a', 'age': 1},
        {'name': 'b', 'age': 2},
        {'name': 'c', 'age': 1},
    ]
    result = list(where(items, age=1))
    assert result == [{'name': 'a', 'age': 1}, {'name': 'c', 'age': 1}]


def test_lwhere():
    items = [{'x': 1}, {'x': 2}, {'x': 1}]
    assert lwhere(items, x=1) == [{'x': 1}, {'x': 1}]


# ---------- pluck / lpluck ----------

def test_pluck():
    items = [{'a': 1}, {'a': 2}]
    assert list(pluck('a', items)) == [1, 2]


def test_lpluck():
    items = [{'a': 1}, {'a': 2}]
    assert lpluck('a', items) == [1, 2]


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

class WithMethods:
    def __init__(self, base):
        self.base = base

    def add(self, x, y=0):
        return self.base + x + y


def test_invoke():
    objs = [WithMethods(1), WithMethods(10)]
    assert list(invoke(objs, 'add', 2)) == [3, 12]


def test_invoke_kwargs():
    objs = [WithMethods(1)]
    assert list(invoke(objs, 'add', 2, y=5)) == [8]


def test_linvoke():
    objs = [WithMethods(1), WithMethods(10)]
    assert linvoke(objs, 'add', 2) == [3, 12]
