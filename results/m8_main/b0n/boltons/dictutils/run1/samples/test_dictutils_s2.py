# test_dictutils.py
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

def test_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


def test_empty_omd():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items(multi=True) == []
    assert repr(omd) == 'OrderedMultiDict([])'


def test_init_with_args_and_kwargs():
    omd = OrderedMultiDict([('a', 1), ('b', 2)], c=3)
    assert omd['a'] == 1
    assert omd['b'] == 2
    assert omd['c'] == 3


def test_init_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict([('a', 1)], [('b', 2)])


def test_getitem_and_setitem():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    assert omd['a'] == 1
    assert omd['b'] == 2
    with pytest.raises(KeyError):
        omd['c']


def test_setitem_replaces_all_values():
    omd = OrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    omd.add('b', 3)
    omd['a'] = 9
    assert omd.getlist('a') == [9]
    assert omd['a'] == 9
    assert omd.items(multi=True) == [('b', 3), ('a', 9)]


def test_add_and_getlist():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 3)
    assert omd['a'] == 3
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]


def test_getlist_copy_is_safe():
    omd = OrderedMultiDict([('a', 1)])
    lst = omd.getlist('a')
    lst.append(99)
    assert omd.getlist('a') == [1]


def test_getlist_default():
    omd = OrderedMultiDict()
    assert omd.getlist('nope') == []
    assert omd.getlist('nope', default=None) is None
    sentinel = object()
    assert omd.getlist('nope', default=sentinel) is sentinel


def test_get_default():
    omd = OrderedMultiDict()
    assert omd.get('x') is None
    assert omd.get('x', 5) == 5


def test_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]
    assert omd.items(multi=True) == [('a', -1), ('a', 0), ('a', 1), ('a', 2)]


def test_addlist_with_iterator():
    omd = OrderedMultiDict()
    omd.addlist('a', iter([1, 2, 3]))
    assert omd.getlist('a') == [1, 2, 3]


def test_addlist_empty():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', [])
    assert omd.getlist('a') == [-1]


def test_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []
    omd['x'] = 1
    assert omd['x'] == 1


def test_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('a', 5) is None
    assert omd.setdefault('b', 7) == 7


def test_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd2 = omd.copy()
    assert omd2 == omd
    omd2.add('a', 4)
    assert omd.getlist('a') == [1, 3]
    assert omd2.getlist('a') == [1, 3, 4]


def test_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b', 'c'])
    assert omd.items(multi=True) == [('a', None), ('b', None), ('c', None)]
    omd2 = OrderedMultiDict.fromkeys(['a', 'b'], default=5)
    assert omd2.items(multi=True) == [('a', 5), ('b', 5)]


def test_update_with_omd():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    other = OrderedMultiDict([('a', 3), ('c', 4)])
    omd.update(other)
    assert omd.getlist('a') == [3]
    assert omd['b'] == 2
    assert omd['c'] == 4


def test_update_with_dict():
    omd = OrderedMultiDict([('a', 1)])
    omd.update({'a': 2, 'b': 3})
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


def test_update_with_iterable():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update([('a', 10), ('a', 11), ('c', 12)])
    assert omd.getlist('a') == [10, 11]
    assert omd['b'] == 2
    assert omd['c'] == 12


def test_update_with_self_is_noop():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.getlist('a') == [1]


def test_update_with_kwargs():
    omd = OrderedMultiDict([('a', 1)])
    omd.update({'b': 2}, c=3)
    assert omd['a'] == 1
    assert omd['b'] == 2
    assert omd['c'] == 3


def test_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('a', 3)])
    assert omd.getlist('a') == [1, 2, 3]


def test_update_extend_with_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_update_extend_with_omd():
    omd = OrderedMultiDict()
    other = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2]


def test_update_extend_with_dict_like():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_ior_operator():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'a': 2, 'b': 3}
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


def test_delitem():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del omd['a']
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        del omd['a']


def test_pop():
    omd = OrderedMultiDict([('a', 1), ('a', 3), ('b', 2)])
    assert omd.pop('a') == 3
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        omd.pop('a')
    assert omd.pop('a', 'default') == 'default'


def test_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 3), ('b', 2)])
    assert omd.popall('a') == [1, 3]
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        omd.popall('a')
    assert omd.popall('a', []) == []


def test_poplast():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]
    assert omd.poplast('a') == 1
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        omd.poplast('a')
    assert omd.poplast('a', 'd') == 'd'


def test_poplast_no_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast() == 3
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]
    assert omd.poplast() == 2
    assert omd.poplast() == 1
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='x') == 'x'


def test_iteritems():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]


def test_iterkeys():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.iterkeys()) == ['a', 'b']


def test_itervalues():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]
    assert list(omd.itervalues()) == [3, 2]


def test_keys_values_items():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.keys() == ['a', 'b']
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.values() == [3, 2]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items() == [('a', 3), ('b', 2)]


def test_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    d = omd.todict(multi=True)
    assert d == {'a': [1, 3], 'b': [2]}
    d['a'].append(99)
    assert omd.getlist('a') == [1, 3]


def test_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    res = omd.sorted(reverse=True)
    assert res.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_sorted_with_key():
    omd = OrderedMultiDict([('a', 3), ('b', 1), ('c', 2)])
    res = omd.sorted(key=lambda i: i[1])
    assert res.items(multi=True) == [('b', 1), ('c', 2), ('a', 3)]


def test_sortedvalues():
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


def test_sortedvalues_reverse():
    omd = OrderedMultiDict()
    omd.addlist('a', [1, 3, 2])
    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('a') == [3, 2, 1]


def test_inverted():
    omd = OMD([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted() == omd


def test_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert counts.getlist('a') == [2]
    assert counts.getlist('b') == [1]


def test_iter_dunder():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd) == ['a', 'b']


def test_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_repr():
    omd = OrderedMultiDict([('a', 1)])
    assert repr(omd) == "OrderedMultiDict([('a', 1)])"


def test_eq_omd():
    omd1 = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd2 = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd3 = OrderedMultiDict([('a', 3), ('b', 2), ('a', 1)])
    assert omd1 == omd1
    assert omd1 == omd2
    assert omd1 != omd3


def test_eq_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd == {'a': 3, 'b': 2}
    assert omd != {'a': 1, 'b': 2}
    assert omd != {'a': 3}


def test_eq_other_types():
    omd = OrderedMultiDict([('a', 1)])
    assert (omd == [('a', 1)]) is False
    assert (omd != [('a', 1)]) is True
    assert (omd == 5) is False


def test_eq_omd_length_mismatch():
    omd1 = OrderedMultiDict([('a', 1), ('b', 2)])
    omd2 = OrderedMultiDict([('a', 1)])
    assert omd1 != omd2


def test_views():
    from collections.abc import KeysView, ValuesView, ItemsView
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert isinstance(omd.viewkeys(), KeysView)
    assert isinstance(omd.viewvalues(), ValuesView)
    assert isinstance(omd.viewitems(), ItemsView)
    assert set(omd.viewkeys()) == {'a', 'b'}
    assert set(omd.viewvalues()) == {1, 2}
    assert set(omd.viewitems()) == {('a', 1), ('b', 2)}


def test_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    omd2 = pickle.loads(data)
    assert omd2 == omd
    assert omd2.items(multi=True) == omd.items(multi=True)


def test_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('a', 2)]
    omd2 = OrderedMultiDict()
    omd2.__setstate__(state)
    assert omd2.items(multi=True) == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def test_fiomd_basic():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.getlist('a') == [1, 3]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items() == [('a', 3), ('b', 2)]


def test_fiomd_iterkeys():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.iterkeys()) == ['b', 'a']


def test_fiomd_reversed():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('c', 3)
    assert list(reversed(omd)) == ['c', 'b', 'a']


def test_fiomd_reversed_with_dupes():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    result = list(reversed(omd))
    assert set(result) == {'a', 'b'}


def test_fiomd_remove():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.pop('a') == 3
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_fiomd_remove_all():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    del omd['a']
    assert omd.items() == [('b', 2)]
    assert omd['b'] == 2


def test_fiomd_clear_and_reuse():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.clear()
    assert len(omd) == 0
    omd.add('x', 1)
    assert omd['x'] == 1


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_onetoone_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2


def test_onetoone_overwrite():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_onetoone_setitem():
    oto = OneToOne()
    oto['a'] = 1
    assert oto.inv[1] == 'a'
    oto['b'] = 1
    assert 'a' not in oto
    assert oto.inv[1] == 'b'


def test_onetoone_setitem_pushes_out_reverse():
    oto = OneToOne()
    oto['a'] = 1
    oto['a'] = 2
    assert 1 not in oto.inv
    assert oto.inv[2] == 'a'


def test_onetoone_setitem_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = []


def test_onetoone_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv
    with pytest.raises(KeyError):
        del oto['a']


def test_onetoone_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    oto2 = oto.copy()
    assert oto2 == oto
    assert oto2.inv == oto.inv


def test_onetoone_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 1 not in oto.inv
    assert oto.pop('a', 'default') == 'default'
    with pytest.raises(KeyError):
        oto.pop('a')


def test_onetoone_popitem():
    oto = OneToOne({'a': 1})
    k, v = oto.popitem()
    assert k == 'a'
    assert v == 1
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.inv[1] == 'a'
    assert oto.setdefault('a', 2) == 1


def test_onetoone_update_dict():
    oto = OneToOne()
    oto.update({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_onetoone_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1), ('b', 2)])
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'


def test_onetoone_update_kwargs():
    oto = OneToOne()
    oto.update({'a': 1}, b=2)
    assert oto['a'] == 1
    assert oto['b'] == 2


def test_onetoone_update_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto.update({'a': []})


def test_onetoone_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto) == "OneToOne({'a': 1})"


def test_onetoone_default_dupe_collapses():
    # Default constructor with duplicate values collapses to last
    oto = OneToOne({'a': 1, 'b': 1})
    assert len(oto) == 1
    assert oto['b'] == 1
    assert oto.inv[1] == 'b'


def test_onetoone_unique_success():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'


def test_onetoone_unique_raises():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_onetoone_unique_across_inputs():
    a_dict = {'a': 2}
    with pytest.raises(ValueError):
        OneToOne.unique(a_dict, b=2)


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_manytomany_empty():
    m = ManyToMany()
    assert len(m) == 0
    assert list(m) == []


def test_manytomany_add():
    m = ManyToMany()
    m.add('a', 1)
    m.add('a', 2)
    m.add('b', 2)
    assert m['a'] == frozenset([1, 2])
    assert m['b'] == frozenset([2])
    assert m.inv[2] == frozenset(['a', 'b'])


def test_manytomany_get():
    m = ManyToMany()
    m.add('a', 1)
    assert m.get('a') == frozenset([1])
    assert m.get('x') == frozenset()
    assert m.get('x', default='d') == 'd'


def test_manytomany_getitem_missing():
    m = ManyToMany()
    with pytest.raises(KeyError):
        m['missing']


def test_manytomany_setitem():
    m = ManyToMany()
    m['a'] = [1, 2]
    assert m['a'] == frozenset([1, 2])
    m['a'] = [2, 3]
    assert m['a'] == frozenset([2, 3])
    assert 1 not in m.inv


def test_manytomany_delitem():
    m = ManyToMany()
    m.add('a', 1)
    m.add('a', 2)
    m.add('b', 2)
    del m['a']
    assert 'a' not in m
    assert 1 not in m.inv
    assert m.inv[2] == frozenset(['b'])


def test_manytomany_update_pairs():
    m = ManyToMany()
    m.update([('a', 1), ('a', 2), ('b', 2)])
    assert m['a'] == frozenset([1, 2])
    assert m['b'] == frozenset([2])


def test_manytomany_update_dict_like():
    m = ManyToMany()
    m.update({'a': 1, 'b': 2})
    assert m['a'] == frozenset([1])
    assert m['b'] == frozenset([2])


def test_manytomany_update_with_other():
    m1 = ManyToMany([('a', 1), ('b', 2)])
    m2 = ManyToMany([('a', 3), ('c', 4)])
    m1.update(m2)
    assert m1['a'] == frozenset([1, 3])
    assert m1['b'] == frozenset([2])
    assert m1['c'] == frozenset([4])
    assert 3 in m1.inv


def test_manytomany_init_with_items():
    m = ManyToMany([('a', 1), ('b', 2)])
    assert m['a'] == frozenset([1])
    assert m['b'] == frozenset([2])
    assert len(m) == 2


def test_manytomany_remove():
    m = ManyToMany([('a', 1), ('a', 2)])
    m.remove('a', 1)
    assert m['a'] == frozenset([2])
    assert 1 not in m.inv
    m.remove('a', 2)
    assert 'a' not in m


def test_manytomany_replace():
    m = ManyToMany([('a', 1), ('a', 2), ('b', 3)])
    m.replace('a', 'c')
    assert 'a' not in m
    assert m['c'] == frozenset([1, 2])
    assert m.inv[1] == frozenset(['c'])
    assert m.inv[2] == frozenset(['c'])


def test_manytomany_replace_missing():
    m = ManyToMany([('a', 1)])
    m.replace('z', 'c')
    assert 'z' not in m
    assert 'c' not in m


def test_manytomany_iteritems():
    m = ManyToMany()
    m.add('a', 1)
    assert list(m.iteritems()) == [('a', 1)]


def test_manytomany_keys_contains_iter_len():
    m = ManyToMany([('a', 1), ('b', 2)])
    assert set(m.keys()) == {'a', 'b'}
    assert 'a' in m
    assert 'x' not in m
    assert set(iter(m)) == {'a', 'b'}
    assert len(m) == 2


def test_manytomany_eq():
    m1 = ManyToMany([('a', 1)])
    m2 = ManyToMany([('a', 1)])
    m3 = ManyToMany([('a', 2)])
    assert m1 == m2
    assert m1 != m3
    assert (m1 == {'a': frozenset([1])}) is False


def test_manytomany_repr():
    m = ManyToMany()
    m.add('a', 1)
    assert repr(m) == "ManyToMany([('a', 1)])"


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_no_args():
    assert subdict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def test_subdict_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_and_drop():
    result = subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'b'], drop=['b'])
    assert result == {'a': 1}


def test_subdict_returns_same_type():
    result = subdict(OrderedMultiDict([('a', 1), ('b', 2)]), keep=['a'])
    assert isinstance(result, OrderedMultiDict)
    assert result.items(multi=True) == [('a', 1)]


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert len(fd) == 2


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert hash(fd) == h
    assert isinstance(h, int)


def test_frozendict_as_key():
    fd = FrozenDict({'a': 1})
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'


def test_frozendict_hash_unhashable_values():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert fd2 == {'a': 1, 'b': 2}
    assert fd == {'a': 1}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'])
    assert fd == {'a': None, 'b': None}


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_copy():
    fd = FrozenDict({'a': 1})
    fd2 = copy.copy(fd)
    assert fd2 is fd


def test_frozendict_immutable():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd['b'] = 2
    with pytest.raises(TypeError):
        del fd['a']
    with pytest.raises(TypeError):
        fd.update({'b': 2})
    with pytest.raises(TypeError):
        fd.setdefault('b', 2)
    with pytest.raises(TypeError):
        fd.pop('a')
    with pytest.raises(TypeError):
        fd.clear()


def test_frozendict_pickle_roundtrip():
    fd = FrozenDict({'a': 1, 'b': 2})
    fd2 = pickle.loads(pickle.dumps(fd))
    assert fd2 == fd
    assert isinstance(fd2, FrozenDict)


def test_frozenhash_error_is_typeerror():
    assert issubclass(FrozenHashError, TypeError)
