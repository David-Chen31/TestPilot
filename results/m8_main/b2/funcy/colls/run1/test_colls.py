from collections import Counter, defaultdict, OrderedDict
from itertools import islice

import pytest

from funcy.colls import (
    empty, iteritems, itervalues,
    join, merge, join_with, merge_with,
    walk, walk_keys, walk_values, select, select_keys, select_values,
    split_keys, compact,
    is_distinct, all as colls_all, any as colls_any, none, one, some,
    zipdict, flip, project, omit, zip_values, zip_dicts,
    where, pluck, pluck_attr, invoke, lwhere, lpluck, lpluck_attr, linvoke,
    get_in, get_lax, set_in, update_in, del_in, has_path,
)


# ---------- empty ----------

def test_empty_preserves_type():
    assert empty([1, 2, 3]) == []
    assert empty((1, 2, 3)) == ()
    assert empty({1, 2, 3}) == set()
    assert empty({'a': 1}) == {}
    assert empty('abc') == ''
    assert empty(b'abc') == b''


def test_empty_iterator_returns_iterator():
    it = empty(iter([1, 2, 3]))
    assert iter(it) is it
    assert list(it) == []


def test_empty_orderdict_preserves_type():
    result = empty(OrderedDict([('a', 1)]))
    assert isinstance(result, OrderedDict)
    assert len(result) == 0


def test_empty_counter():
    result = empty(Counter([1, 1, 2]))
    assert isinstance(result, Counter)
    assert len(result) == 0


def test_empty_defaultdict():
    dd = defaultdict(list)
    result = empty(dd)
    assert isinstance(result, defaultdict)


def test_empty_dict_keys_views():
    assert empty({}.keys()) == []
    assert empty({}.values()) == []
    assert empty({}.items()) == []


# ---------- iteritems / itervalues ----------

def test_iteritems_with_dict():
    assert list(iteritems({'a': 1, 'b': 2})) == [('a', 1), ('b', 2)]


def test_iteritems_without_items():
    assert list(iteritems([1, 2, 3])) == [1, 2, 3]


def test_itervalues_with_dict():
    assert sorted(itervalues({'a': 1, 'b': 2})) == [1, 2]


def test_itervalues_without_values():
    assert list(itervalues([1, 2, 3])) == [1, 2, 3]


# ---------- join / merge ----------

def test_join_none_when_empty():
    assert join([]) is None


def test_join_strings():
    assert join(['ab', 'cd', 'ef']) == 'abcdef'


def test_join_bytes():
    assert join([b'ab', b'cd']) == b'abcd'


def test_join_dicts():
    assert join([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]) == {'a': 1, 'b': 3, 'c': 4}


def test_join_dicts_does_not_mutate_first():
    first = {'a': 1}
    join([first, {'b': 2}])
    assert first == {'a': 1}


def test_join_sets():
    assert join([{1, 2}, {2, 3}]) == {1, 2, 3}


def test_join_sets_subclass_returns_plain_set():
    # set.union() on a subclass instance returns a plain set (unless overridden)
    class MySet(set):
        pass
    result = join([MySet([1, 2]), MySet([3])])
    assert result == {1, 2, 3}
    assert type(result) is set


def test_join_iterators():
    result = join([iter([1, 2]), iter([3, 4])])
    assert list(result) == [1, 2, 3, 4]


def test_join_ranges():
    result = join([range(3), range(3, 5)])
    assert list(result) == [0, 1, 2, 3, 4]


def test_join_lists():
    assert join([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_join_tuples():
    assert join([(1, 2), (3, 4)]) == (1, 2, 3, 4)


def test_join_unsupported_type_raises():
    class Foo(object):
        pass
    with pytest.raises(TypeError):
        join([Foo(), Foo()])


def test_join_unsupported_type_message():
    class Foo(object):
        pass
    with pytest.raises(TypeError) as exc:
        join([Foo(), Foo()])
    assert str(exc.value) == "Don't know how to join Foo"


def test_join_unsupported_type_message_lowercase_check():
    # Ensure case-sensitive exact message
    class Bar(object):
        pass
    with pytest.raises(TypeError) as exc:
        join([Bar()])
    msg = str(exc.value)
    assert msg.startswith("Don't")
    assert "Bar" in msg


def test_merge_basic():
    assert merge([1, 2], [3]) == [1, 2, 3]
    assert merge({'a': 1}, {'a': 2}) == {'a': 2}
    assert merge('ab', 'cd') == 'abcd'


# ---------- join_with / merge_with ----------

def test_join_with_empty_returns_empty_dict():
    assert join_with(sum, []) == {}


def test_join_with_single_returns_as_is():
    d = {'a': 1}
    assert join_with(sum, [d]) is d


def test_join_with_single_strict():
    d = {'a': 1}
    result = join_with(sum, [d], strict=True)
    assert result == {'a': 1}
    assert result is not d


def test_join_with_function():
    dicts = [{'a': 1, 'b': 2}, {'a': 3}, {'b': 4, 'c': 5}]
    assert join_with(sum, dicts) == {'a': 4, 'b': 6, 'c': 5}


def test_join_with_list_identity():
    dicts = [{'a': 1}, {'a': 2, 'b': 3}]
    result = join_with(list, dicts)
    assert result == {'a': [1, 2], 'b': [3]}


def test_merge_with():
    assert merge_with(sum, {'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}


# ---------- walk / walk_keys / walk_values ----------

def test_walk_list():
    assert walk(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_walk_dict():
    result = walk(lambda pair: (pair[0], pair[1] * 2), {'a': 1, 'b': 2})
    assert result == {'a': 2, 'b': 4}


def test_walk_set():
    assert walk(lambda x: x + 1, {1, 2}) == {2, 3}


def test_walk_string():
    assert walk(str.upper, 'abc') == 'ABC'


def test_walk_keys():
    assert walk_keys(str.upper, {'a': 1, 'b': 2}) == {'A': 1, 'B': 2}


def test_walk_keys_non_callable():
    # make_func with an int produces itemgetter
    assert walk_keys(0, {('x', 1): 'a', ('y', 2): 'b'}) == {'x': 'a', 'y': 'b'}


def test_walk_values():
    assert walk_values(lambda v: v + 1, {'a': 1, 'b': 2}) == {'a': 2, 'b': 3}


def test_walk_values_preserves_defaultdict():
    d = defaultdict(list, {'a': 1})
    result = walk_values(lambda v: v * 10 if isinstance(v, int) else v, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 10


def test_walk_values_defaultdict_with_mapper_and_factory():
    # The mapper must be composed with the factory: when a *new* missing key
    # is accessed in the result, the factory value must pass through the mapper.
    d = defaultdict(int, {'a': 1})
    result = walk_values(lambda v: v + 100, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 101
    # Trigger the factory path: new missing key -> factory() then mapper applied
    assert result['brand_new'] == 100   # int() == 0, mapped +100


def test_walk_values_defaultdict_without_factory():
    d = defaultdict(None, {'a': 1})
    result = walk_values(lambda v: v + 1, d)
    assert isinstance(result, defaultdict)
    assert result['a'] == 2
    # No factory -> accessing missing key raises KeyError
    with pytest.raises(KeyError):
        result['missing']


def test_walk_values_defaultdict_factory_type_preserved():
    d = defaultdict(list, {'a': [1]})
    result = walk_values(lambda v: v, d)
    assert isinstance(result, defaultdict)
    # default_factory preserved
    assert result['new'] == []


def test_walk_values_counter():
    c = Counter({'a': 2, 'b': 1})
    result = walk_values(lambda v: v * 2, c)
    assert isinstance(result, Counter)
    assert result == Counter({'a': 4, 'b': 2})


# ---------- select / select_keys / select_values ----------

def test_select_list():
    assert select(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]


def test_select_dict():
    assert select(lambda pair: pair[1] > 1, {'a': 1, 'b': 2}) == {'b': 2}


def test_select_set():
    assert select(lambda x: x > 1, {1, 2, 3}) == {2, 3}


def test_select_keys():
    assert select_keys(lambda k: k == 'a', {'a': 1, 'b': 2}) == {'a': 1}


def test_select_keys_with_set():
    assert select_keys({'a'}, {'a': 1, 'b': 2}) == {'a': 1}


def test_select_values():
    assert select_values(lambda v: v > 1, {'a': 1, 'b': 2}) == {'b': 2}


def test_select_values_with_non_callable():
    assert select_values(None, {'a': 0, 'b': 1}) == {'b': 1}


# ---------- split_keys ----------

def test_split_keys():
    yes, no = split_keys(lambda k: k == 'a', {'a': 1, 'b': 2, 'c': 3})
    assert yes == {'a': 1}
    assert no == {'b': 2, 'c': 3}


def test_split_keys_non_callable():
    yes, no = split_keys('a', {'a': 1, 'b': 2})
    assert yes == {'a': 1}
    assert no == {'b': 2}


# ---------- compact ----------
# NOTE: select_values(bool, coll) and select_values(None, coll) are equivalent
# because make_pred(None) -> bool. The surviving mutants on lines 200/202 are
# therefore equivalent; we still assert real behavior on truthy/falsy values.

def test_compact_list():
    assert compact([0, 1, 2, '', 'a', None]) == [1, 2, 'a']


def test_compact_mapping():
    assert compact({'a': 0, 'b': 1, 'c': None}) == {'b': 1}


def test_compact_list_falsy_types():
    # 0, '', None, [], {}, set() all falsy; only truthy kept
    assert compact([0, '', None, [], {}, set(), 1, 'x', [1]]) == [1, 'x', [1]]


def test_compact_mapping_with_falsy_variants():
    assert compact({'a': 0, 'b': '', 'c': None, 'd': [], 'e': 1}) == {'e': 1}


# ---------- content tests ----------

def test_is_distinct():
    assert is_distinct([1, 2, 3]) is True
    assert is_distinct([1, 1, 2]) is False


def test_is_distinct_with_key():
    assert is_distinct(['a', 'A', 'b'], key=str.lower) is False
    assert is_distinct(['a', 'b', 'c'], key=str.lower) is True


def test_colls_all():
    assert colls_all(lambda x: x > 0, [1, 2, 3]) is True
    assert colls_all(lambda x: x > 0, [1, -2, 3]) is False


def test_colls_all_no_seq_truthy():
    assert colls_all([1, 2, 3]) is True
    assert colls_all([1, 0, 3]) is False


def test_colls_any():
    assert colls_any(lambda x: x > 0, [-1, 2]) is True
    assert colls_any(lambda x: x > 0, [-1, -2]) is False


def test_colls_any_no_seq():
    assert colls_any([0, 1]) is True
    assert colls_any([0, 0]) is False


def test_none():
    assert none(lambda x: x > 0, [-1, -2]) is True
    assert none(lambda x: x > 0, [1]) is False


def test_none_no_seq():
    assert none([0, 0]) is True


def test_one():
    assert one(lambda x: x > 0, [-1, 2, -3]) is True
    assert one(lambda x: x > 0, [1, 2]) is False
    assert one(lambda x: x > 0, [-1, -2]) is False


def test_one_no_seq():
    # one(bool, pred) and one(None, pred) are equivalent since make_pred(None)->bool,
    # so those mutants are equivalent; still test the semantics rigorously.
    assert one([0, 1, 0]) is True
    assert one([1, 1]) is False
    assert one([]) is False
    assert one([0, 0, 0]) is False
    assert one([1, 0, 0]) is True


def test_some():
    assert some(lambda x: x > 0, [-1, 2, 3]) == 2
    assert some(lambda x: x > 0, [-1, -2]) is None


def test_some_no_seq():
    # some(bool, pred) and some(None, pred) are equivalent since make_pred(None)->bool
    assert some([0, 2, 3]) == 2
    assert some([0, 0]) is None
    assert some([0, 0, 1]) == 1


# ---------- zipdict / flip / project / omit ----------

def test_zipdict():
    assert zipdict(['a', 'b'], [1, 2]) == {'a': 1, 'b': 2}


def test_zipdict_truncates():
    assert zipdict(['a', 'b', 'c'], [1, 2]) == {'a': 1, 'b': 2}


def test_flip():
    assert flip({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_flip_pairs():
    # flip on a list uses the list factory, so it returns a list of flipped pairs
    assert flip([('a', 1), ('b', 2)]) == [(1, 'a'), (2, 'b')]


def test_project():
    assert project({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']) == {'a': 1, 'c': 3}


def test_project_ignores_missing():
    assert project({'a': 1}, ['a', 'z']) == {'a': 1}


def test_omit():
    assert omit({'a': 1, 'b': 2, 'c': 3}, ['b']) == {'a': 1, 'c': 3}


def test_omit_none_missing():
    assert omit({'a': 1}, ['z']) == {'a': 1}


# ---------- zip_values / zip_dicts ----------

def test_zip_values():
    result = sorted(zip_values({'a': 1, 'b': 2}, {'a': 10, 'c': 30}))
    assert result == [(1, 10)]


def test_zip_values_no_args_raises():
    with pytest.raises(TypeError) as exc:
        list(zip_values())
    assert str(exc.value) == 'zip_values expects at least one argument'


def test_zip_values_single_arg():
    # Single dict is allowed: should yield one-element tuples, not raise.
    result = sorted(zip_values({'a': 1, 'b': 2}))
    assert result == [(1,), (2,)]


def test_zip_dicts():
    result = sorted(zip_dicts({'a': 1, 'b': 2}, {'a': 10, 'c': 30}))
    assert result == [('a', (1, 10))]


def test_zip_dicts_no_args_raises():
    with pytest.raises(TypeError) as exc:
        list(zip_dicts())
    assert str(exc.value) == 'zip_dicts expects at least one argument'


def test_zip_dicts_single_arg():
    # Single dict is allowed: should yield (key, (val,)) tuples.
    result = sorted(zip_dicts({'a': 1, 'b': 2}))
    assert result == [('a', (1,)), ('b', (2,))]


# ---------- get_in / get_lax ----------

def test_get_in():
    coll = {'a': {'b': [1, 2, 3]}}
    assert get_in(coll, ['a', 'b', 1]) == 2


def test_get_in_default():
    coll = {'a': {'b': 1}}
    assert get_in(coll, ['a', 'z'], default='x') == 'x'


def test_get_in_type_error_propagates():
    # get_in only catches KeyError/IndexError; a wrong type raises TypeError
    coll = {'a': {'b': 1}}
    with pytest.raises(TypeError):
        get_in(coll, ['a', 'b', 'c'], default=None)


def test_get_in_empty_path():
    coll = {'a': 1}
    assert get_in(coll, []) is coll


def test_get_lax_type_error():
    # get_in would raise TypeError, get_lax returns default
    assert get_lax(5, ['a'], default='d') == 'd'


def test_get_lax_normal():
    assert get_lax({'a': [1, 2]}, ['a', 0]) == 1


def test_get_lax_default():
    assert get_lax({'a': 1}, ['b'], default=42) == 42


# ---------- set_in / update_in / del_in / has_path ----------

def test_set_in_nested():
    coll = {'a': {'b': 1}}
    result = set_in(coll, ['a', 'b'], 5)
    assert result == {'a': {'b': 5}}
    # original unchanged
    assert coll == {'a': {'b': 1}}


def test_set_in_new_key():
    result = set_in({}, ['a'], 1)
    assert result == {'a': 1}


def test_set_in_autovivify():
    result = set_in({}, ['a', 'b', 'c'], 1)
    assert result == {'a': {'b': {'c': 1}}}


def test_set_in_list():
    result = set_in([1, 2, 3], [1], 9)
    assert result == [1, 9, 3]


def test_set_in_list_nested():
    # Nested list update must read *current* value at path[0],
    # then recurse into subsequent path elements.
    coll = [[1, 2], [3, 4]]
    result = set_in(coll, [1, 0], 99)
    assert result == [[1, 2], [99, 4]]
    # original unchanged
    assert coll == [[1, 2], [3, 4]]


def test_set_in_list_deep_nested():
    coll = [[{'x': 1}]]
    result = set_in(coll, [0, 0, 'x'], 7)
    assert result == [[{'x': 7}]]
    assert coll == [[{'x': 1}]]


def test_update_in_empty_path():
    assert update_in(5, [], lambda x: x + 1) == 6


def test_update_in_nested():
    coll = {'a': {'b': 1}}
    result = update_in(coll, ['a', 'b'], lambda x: x + 10)
    assert result == {'a': {'b': 11}}
    assert coll == {'a': {'b': 1}}


def test_update_in_with_default():
    result = update_in({}, ['a', 'b'], lambda x: x + 1, default=0)
    assert result == {'a': {'b': 1}}


def test_update_in_list():
    coll = [1, 2, 3]
    result = update_in(coll, [1], lambda x: x * 10)
    assert result == [1, 20, 3]
    assert coll == [1, 2, 3]


def test_del_in_top_level():
    coll = {'a': 1, 'b': 2}
    result = del_in(coll, ['a'])
    assert result == {'b': 2}
    assert coll == {'a': 1, 'b': 2}


def test_del_in_nested():
    coll = {'a': {'b': 1, 'c': 2}}
    result = del_in(coll, ['a', 'b'])
    assert result == {'a': {'c': 2}}
    assert coll == {'a': {'b': 1, 'c': 2}}


def test_del_in_missing():
    coll = {'a': 1}
    assert del_in(coll, ['missing']) is coll


def test_del_in_empty_path():
    coll = {'a': 1}
    assert del_in(coll, []) is coll


def test_del_in_list():
    coll = [1, 2, 3]
    result = del_in(coll, [1])
    assert result == [1, 3]
    assert coll == [1, 2, 3]


def test_has_path():
    coll = {'a': {'b': [1, 2]}}
    assert has_path(coll, ['a', 'b', 0]) is True
    assert has_path(coll, ['a', 'z']) is False
    assert has_path(coll, ['a', 'b', 5]) is False


def test_has_path_empty():
    assert has_path({'a': 1}, []) is True


# ---------- where / pluck / pluck_attr / invoke ----------

def test_where():
    data = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y': 2}]
    result = list(where(data, x=1))
    assert result == [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}]


def test_where_no_cond():
    data = [{'a': 1}, {'b': 2}]
    assert list(where(data)) == data


def test_lwhere():
    data = [{'x': 1}, {'x': 2}]
    assert lwhere(data, x=2) == [{'x': 2}]


def test_pluck():
    data = [{'a': 1}, {'a': 2}, {'a': 3}]
    assert list(pluck('a', data)) == [1, 2, 3]


def test_lpluck():
    data = [{'a': 1}, {'a': 2}]
    assert lpluck('a', data) == [1, 2]


def test_pluck_attr():
    class Obj(object):
        def __init__(self, v):
            self.v = v
    objs = [Obj(1), Obj(2)]
    assert list(pluck_attr('v', objs)) == [1, 2]


def test_lpluck_attr():
    class Obj(object):
        def __init__(self, v):
            self.v = v
    objs = [Obj(1), Obj(2)]
    assert lpluck_attr('v', objs) == [1, 2]


def test_invoke():
    objs = ['foo', 'bar']
    assert list(invoke(objs, 'upper')) == ['FOO', 'BAR']


def test_invoke_with_args():
    objs = ['a,b', 'c,d']
    assert list(invoke(objs, 'split', ',')) == [['a', 'b'], ['c', 'd']]


def test_linvoke():
    objs = ['foo', 'bar']
    assert linvoke(objs, 'upper') == ['FOO', 'BAR']


# ---------- bytes/str factory edge cases in walk ----------

def test_walk_bytes_returns_bytes():
    assert walk(lambda b: b + 1, b'abc') == b'bcd'


def test_walk_str_join():
    assert walk(str.upper, 'ab') == 'AB'


# ---------- extra: bytes factory edge cases ----------

def test_walk_bytes_empty():
    # empty bytes input: bytes factory with empty items -> returns b''
    assert walk(lambda b: b, b'') == b''


def test_walk_bytes_mapper_returns_bytes_chunks():
    # Mapper returns bytes objects (not ints): factory uses join path
    result = walk(lambda b: bytes([b]), b'ab')
    assert result == b'ab'


def test_walk_bytes_mapper_returns_multibyte():
    result = walk(lambda b: bytes([b, b]), b'ab')
    assert result == b'aabb'
