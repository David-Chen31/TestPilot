import copy
import pickle

import pytest

from boltons.dictutils import (
    OrderedMultiDict,
    OMD,
    MultiDict,
    FastIterOrderedMultiDict,
    OneToOne,
    ManyToMany,
    subdict,
    FrozenDict,
    FrozenHashError,
)


# ---------------------------------------------------------------------------
# OrderedMultiDict basics
# ---------------------------------------------------------------------------

def test_omd_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


def test_omd_empty():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items() == []
    assert omd.items(multi=True) == []
    assert omd.keys() == []
    assert omd.keys(multi=True) == []
    assert omd.values() == []
    assert omd.values(multi=True) == []
    assert repr(omd) == 'OrderedMultiDict([])'


def test_omd_init_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict({}, {})


def test_omd_basic_docstring_example():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert list(reversed(omd)) == ['b', 'a']
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"
    assert omd.poplast('a') == 3
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2)])"
    assert omd.pop('a') == 1
    assert repr(omd) == "OrderedMultiDict([('b', 2)])"


def test_omd_init_from_pairs():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd['a'] == 3
    assert omd['b'] == 2
    assert omd.getlist('a') == [1, 3]
    assert list(omd) == ['a', 'b']
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.itervalues()) == [3, 2]
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]


def test_omd_init_with_kwargs():
    omd = OrderedMultiDict([('a', 1)], b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_add():
    omd = OrderedMultiDict()
    omd.add('k', 1)
    omd.add('k', 2)
    assert omd.getlist('k') == [1, 2]
    assert omd['k'] == 2


def test_omd_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]
    assert repr(omd) == "OrderedMultiDict([('a', -1), ('a', 0), ('a', 1), ('a', 2)])"


def test_omd_addlist_empty_iterable():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', [])
    assert omd.getlist('a') == [-1]
    # key not even created if new
    omd.addlist('b', [])
    assert 'b' not in omd


def test_omd_addlist_one_shot_iterator():
    omd = OrderedMultiDict()
    omd.addlist('k', iter([1, 2, 3]))
    assert omd.getlist('k') == [1, 2, 3]


def test_omd_get_default():
    omd = OrderedMultiDict()
    assert omd.get('missing') is None
    assert omd.get('missing', 42) == 42


def test_omd_getlist_default():
    omd = OrderedMultiDict()
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', 'x') == 'x'


def test_omd_getlist_returns_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    lst = omd.getlist('a')
    lst.append(99)
    assert omd.getlist('a') == [1, 2]


def test_omd_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('a', 5) is None
    assert omd.setdefault('b', 5) == 5
    assert omd['b'] == 5


def test_omd_setitem_overwrites():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd['a'] = 10
    assert omd.getlist('a') == [10]
    assert omd['b'] == 2


def test_omd_delitem():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd['b'] == 3
    with pytest.raises(KeyError):
        del omd['a']


def test_omd_contains():
    omd = OrderedMultiDict([('a', 1)])
    assert 'a' in omd
    assert 'x' not in omd


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.clear()
    assert omd.items(multi=True) == []
    assert len(omd) == 0


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    cp = omd.copy()
    assert isinstance(cp, OrderedMultiDict)
    assert cp is not omd
    assert cp.items(multi=True) == omd.items(multi=True)
    cp.add('c', 4)
    assert 'c' not in omd


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd.items() == [('a', None), ('b', None)]
    omd = OrderedMultiDict.fromkeys(['a', 'b'], default=7)
    assert omd['a'] == 7
    assert omd['b'] == 7


def test_omd_update_from_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update({'a': 10, 'c': 30})
    assert omd.items(multi=True) == [('b', 2), ('a', 10), ('c', 30)]


def test_omd_update_from_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 10), ('a', 20)])
    omd.update(other)
    assert omd.getlist('a') == [10, 20]
    assert omd['b'] == 3


def test_omd_update_self_noop():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update(omd)
    assert omd.getlist('a') == [1, 2]


def test_omd_update_iterable_pairs_with_dupes():
    omd = OrderedMultiDict([('a', 1)])
    omd.update([('a', 2), ('a', 3)])
    # first occurrence triggers deletion of old 'a', then both added
    assert omd.getlist('a') == [2, 3]


def test_omd_update_kwargs():
    omd = OrderedMultiDict([('a', 1)])
    omd.update({'b': 2}, c=3)
    assert omd['b'] == 2
    assert omd['c'] == 3


def test_omd_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('b', 3)])
    assert omd.getlist('a') == [1, 2]
    assert omd['b'] == 3


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]
    assert omd.getlist('b') == [2, 2]


def test_omd_update_extend_from_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 10), ('a', 20)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 10, 20]


def test_omd_update_extend_from_dictlike():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_pop():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.pop('a') == 2
    assert 'a' not in omd
    assert omd['b'] == 3


def test_omd_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('missing', 'd') == 'd'
    with pytest.raises(KeyError):
        omd.pop('missing')


def test_omd_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd


def test_omd_popall_default():
    omd = OrderedMultiDict()
    assert omd.popall('x', []) == []
    with pytest.raises(KeyError):
        omd.popall('x')


def test_omd_poplast():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.poplast('a') == 2
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd
    assert omd['b'] == 3


def test_omd_poplast_default():
    omd = OrderedMultiDict()
    assert omd.poplast('x', 'd') == 'd'
    with pytest.raises(KeyError):
        omd.poplast('x')


def test_omd_poplast_no_arg():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.poplast() == 3
    assert omd.poplast() == 2
    assert omd.poplast() == 1
    with pytest.raises(KeyError):
        omd.poplast()


def test_omd_poplast_empty_with_default():
    omd = OrderedMultiDict()
    assert omd.poplast(default='d') == 'd'


def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    d = omd.todict(multi=True)
    assert d == {'a': [1, 3], 'b': [2]}
    # copy safe
    d['a'].append(99)
    assert omd.getlist('a') == [1, 3]


def test_omd_eq():
    a = OrderedMultiDict([('a', 1), ('a', 2)])
    b = OrderedMultiDict([('a', 1), ('a', 2)])
    assert a == b
    assert not (a != b)

    c = OrderedMultiDict([('a', 2), ('a', 1)])
    assert a != c

    assert a != OrderedMultiDict()
    assert a != "not a dict"

    # compares against plain dict
    assert OrderedMultiDict([('x', 1)]) == {'x': 1}
    assert OrderedMultiDict([('x', 1)]) != {'x': 2}
    assert OrderedMultiDict([('x', 1)]) != {'y': 1}


def test_omd_eq_self():
    omd = OrderedMultiDict([('a', 1)])
    assert omd == omd


def test_omd_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'a': 2, 'b': 3}
    assert omd['a'] == 2
    assert omd['b'] == 3


def test_omd_iteritems_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]


def test_omd_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    srt = omd.sorted(reverse=True)
    assert isinstance(srt, OrderedMultiDict)
    assert srt.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_omd_sorted_by_key():
    omd = OrderedMultiDict(zip('hello', 'world'))
    srt = omd.sorted(key=lambda i: i[1])
    assert srt.items(multi=True) == [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]


def test_omd_sortedvalues():
    omd = OrderedMultiDict()
    omd.addlist('even', [6, 2])
    omd.addlist('odd', [1, 5])
    omd.add('even', 4)
    omd.add('odd', 3)
    somd = omd.sortedvalues()
    assert somd.getlist('even') == [2, 4, 6]
    assert somd.getlist('odd') == [1, 3, 5]
    assert somd.keys(multi=True) == omd.keys(multi=True)
    assert omd != somd


def test_omd_inverted():
    omd = OrderedMultiDict([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]
    assert omd.inverted().inverted() == omd


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts['a'] == 2
    assert counts['b'] == 1


def test_omd_keys_values_items():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'a', 'b']
    assert omd.values() == [2, 3]
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.items() == [('a', 2), ('b', 3)]
    assert omd.items(multi=True) == [('a', 1), ('a', 2), ('b', 3)]


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_views():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert set(omd.viewkeys()) == {'a', 'b'}
    assert list(omd.viewvalues()) == [2, 3]
    assert set(omd.viewitems()) == {('a', 2), ('b', 3)}


def test_omd_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    restored = pickle.loads(data)
    assert isinstance(restored, OrderedMultiDict)
    assert restored.items(multi=True) == omd.items(multi=True)


def test_omd_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('a', 2)]
    other = OrderedMultiDict()
    other.__setstate__(state)
    assert other.items(multi=True) == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def test_fast_omd_basic():
    fomd = FastIterOrderedMultiDict()
    fomd.add('a', 1)
    fomd.add('b', 2)
    fomd.add('a', 3)
    assert fomd.getlist('a') == [1, 3]
    assert fomd['a'] == 3
    assert list(fomd.iterkeys()) == ['a', 'b']
    assert list(fomd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(fomd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]


def test_fast_omd_setitem():
    fomd = FastIterOrderedMultiDict([('a', 1)])
    fomd['a'] = 5
    assert fomd.getlist('a') == [5]


def test_fast_omd_poplast():
    fomd = FastIterOrderedMultiDict([('a', 1), ('a', 2)])
    assert fomd.poplast('a') == 2
    assert fomd.getlist('a') == [1]


def test_fast_omd_del():
    fomd = FastIterOrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    del fomd['a']
    assert 'a' not in fomd
    assert fomd['b'] == 3


def test_fast_omd_reversed():
    fomd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(fomd)) == ['b', 'a']


def test_fast_omd_remove_all_then_iter():
    fomd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del fomd['a']
    assert list(fomd.iterkeys(multi=True)) == ['b']
    fomd.add('c', 9)
    assert list(fomd.iteritems(multi=True)) == [('b', 2), ('c', 9)]


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_onetoone_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2


def test_onetoone_overwrite_both_directions():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_onetoone_setitem_replaces_value():
    oto = OneToOne({'a': 1})
    oto['a'] = 2
    assert oto['a'] == 2
    assert 1 not in oto.inv
    assert oto.inv[2] == 'a'


def test_onetoone_setitem_steals_value():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['c'] = 1
    assert oto['c'] == 1
    assert 'a' not in oto
    assert oto.inv[1] == 'c'
    assert len(oto) == 2


def test_onetoone_setitem_unhashable():
    oto = OneToOne({'a': 1})
    with pytest.raises(TypeError):
        oto['b'] = [1, 2]


def test_onetoone_dupe_values_collapse():
    oto = OneToOne({'a': 1, 'b': 1})
    # Kept only one key mapping to 1
    assert len(oto) == 1
    assert 1 in oto.inv


def test_onetoone_unique_raises():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_onetoone_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto['b'] == 2
    assert oto.inv[1] == 'a'


def test_onetoone_unique_across_inputs():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)


def test_onetoone_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert oto.inv[2] == 'b'


def test_onetoone_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    cp = oto.copy()
    assert isinstance(cp, OneToOne)
    assert cp is not oto
    assert cp == {'a': 1, 'b': 2}
    assert cp.inv[1] == 'a'
    cp['c'] = 3
    assert 'c' not in oto


def test_onetoone_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert oto.pop('missing', 'default') == 'default'
    with pytest.raises(KeyError):
        oto.pop('missing')


def test_onetoone_popitem():
    oto = OneToOne({'a': 1})
    k, v = oto.popitem()
    assert (k, v) == ('a', 1)
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.inv[1] == 'a'
    assert oto.setdefault('a', 2) == 1
    assert oto.inv.get(2) is None


def test_onetoone_update_dict():
    oto = OneToOne({'a': 1})
    oto.update({'b': 2})
    assert oto['b'] == 2
    assert oto.inv[2] == 'b'


def test_onetoone_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1), ('b', 2)])
    assert oto == {'a': 1, 'b': 2}
    assert oto.inv[1] == 'a'


def test_onetoone_update_kwargs():
    oto = OneToOne()
    oto.update({'a': 1}, b=2)
    assert oto['a'] == 1
    assert oto['b'] == 2
    assert oto.inv[2] == 'b'


def test_onetoone_update_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto.update({'a': [1, 2]})


def test_onetoone_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto) == "OneToOne({'a': 1})"


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_manytomany_basic():
    m2m = ManyToMany([('a', 1)])
    assert m2m.get('a') == frozenset([1])
    assert m2m['a'] == frozenset([1])
    assert m2m.get('missing') == frozenset()
    assert m2m.get('missing', None) is None


def test_manytomany_getitem_missing():
    m2m = ManyToMany()
    with pytest.raises(KeyError):
        m2m['missing']


def test_manytomany_add():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['a'])
    assert m2m.inv[2] == frozenset(['a'])


def test_manytomany_setitem():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset([2, 3])
    assert 1 not in m2m.inv
    assert m2m.inv[3] == frozenset(['a'])


def test_manytomany_delitem():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    del m2m['a']
    assert 'a' not in m2m
    assert 1 not in m2m.inv
    assert 2 not in m2m.inv


def test_manytomany_remove():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m.remove('a', 1)
    assert m2m['a'] == frozenset([2])
    assert 1 not in m2m.inv


def test_manytomany_remove_last_cleans():
    m2m = ManyToMany([('a', 1)])
    m2m.remove('a', 1)
    assert 'a' not in m2m
    assert 1 not in m2m.inv


def test_manytomany_replace():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m.replace('a', 'b')
    assert 'a' not in m2m
    assert m2m['b'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['b'])


def test_manytomany_replace_missing():
    m2m = ManyToMany()
    m2m.replace('a', 'b')
    assert 'b' not in m2m


def test_manytomany_update_iterable():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('a', 2)])
    assert m2m['a'] == frozenset([1, 2])


def test_manytomany_update_dictlike():
    m2m = ManyToMany()
    m2m.update({'a': 1, 'b': 2})
    assert m2m['a'] == frozenset([1])
    assert m2m['b'] == frozenset([2])


def test_manytomany_update_manytomany():
    m2m = ManyToMany([('a', 1)])
    other = ManyToMany([('a', 2), ('b', 3)])
    m2m.update(other)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([3])
    assert m2m.inv[2] == frozenset(['a'])
    assert m2m.inv[3] == frozenset(['b'])


def test_manytomany_iteritems():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    items = set(m2m.iteritems())
    assert items == {('a', 1), ('a', 2)}


def test_manytomany_keys_contains_iter_len():
    m2m = ManyToMany([('a', 1), ('b', 2)])
    assert set(m2m.keys()) == {'a', 'b'}
    assert 'a' in m2m
    assert 'z' not in m2m
    assert set(iter(m2m)) == {'a', 'b'}
    assert len(m2m) == 2


def test_manytomany_eq():
    m2m1 = ManyToMany([('a', 1)])
    m2m2 = ManyToMany([('a', 1)])
    assert m2m1 == m2m2
    m2m3 = ManyToMany([('a', 2)])
    assert m2m1 != m2m3
    assert m2m1 != "not manytomany"


def test_manytomany_repr():
    m2m = ManyToMany([('a', 1)])
    r = repr(m2m)
    assert r.startswith('ManyToMany(')
    assert r.endswith(')')


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_default():
    assert subdict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def test_subdict_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_missing_key():
    assert subdict({'a': 1}, keep=['a', 'b']) == {'a': 1}


def test_subdict_keep_and_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_returns_type():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    result = subdict(omd, keep=['a'])
    assert isinstance(result, OrderedMultiDict)
    assert result['a'] == 1


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert len(fd) == 2
    assert repr(fd) == "FrozenDict({'a': 1, 'b': 2})" or repr(fd) == "FrozenDict({'b': 2, 'a': 1})"


def test_frozendict_hash():
    fd1 = FrozenDict({'a': 1, 'b': 2})
    fd2 = FrozenDict({'b': 2, 'a': 1})
    assert hash(fd1) == hash(fd2)
    assert {fd1, fd2} == {fd1}
    d = {fd1: 'value'}
    assert d[fd2] == 'value'


def test_frozendict_hash_cached():
    fd = FrozenDict({'a': 1})
    h1 = hash(fd)
    h2 = hash(fd)
    assert h1 == h2


def test_frozendict_unhashable_raises():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)
    # repeated also raises
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_immutable_setitem():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd['b'] = 2


def test_frozendict_immutable_delitem():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        del fd['a']


def test_frozendict_immutable_update():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.update({'b': 2})


def test_frozendict_immutable_setdefault():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.setdefault('b', 2)


def test_frozendict_immutable_pop():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.pop('a')


def test_frozendict_immutable_popitem():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.popitem()


def test_frozendict_immutable_clear():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.clear()


def test_frozendict_immutable_ior():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd |= {'b': 2}


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    new = fd.updated({'b': 2})
    assert isinstance(new, FrozenDict)
    assert new == {'a': 1, 'b': 2}
    assert fd == {'a': 1}


def test_frozendict_updated_kwargs():
    fd = FrozenDict({'a': 1})
    new = fd.updated(b=2)
    assert new == {'a': 1, 'b': 2}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 5)
    assert isinstance(fd, FrozenDict)
    assert fd == {'a': 5, 'b': 5}


def test_frozendict_copy_returns_self():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    restored = pickle.loads(pickle.dumps(fd))
    assert isinstance(restored, FrozenDict)
    assert restored == fd


def test_frozendict_as_key():
    fd = FrozenDict({'a': 1})
    d = {fd: 'x'}
    assert d[FrozenDict({'a': 1})] == 'x'
