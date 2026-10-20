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
# Aliases
# ---------------------------------------------------------------------------

def test_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


# ---------------------------------------------------------------------------
# OrderedMultiDict basics
# ---------------------------------------------------------------------------

def test_omd_empty():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items() == []
    assert omd.keys() == []
    assert omd.values() == []
    assert repr(omd) == 'OrderedMultiDict([])'


def test_omd_setitem_getitem():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    assert omd['a'] == 1
    assert omd['b'] == 2
    assert len(omd) == 2
    assert list(omd) == ['a', 'b']


def test_omd_setitem_overwrites():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 2)
    assert omd.getlist('a') == [1, 2]
    omd['a'] = 3
    assert omd.getlist('a') == [3]
    assert omd['a'] == 3


def test_omd_add_preserves():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]


def test_omd_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]
    assert omd.items(multi=True) == [('a', -1), ('a', 0), ('a', 1), ('a', 2)]


def test_omd_addlist_empty():
    omd = OrderedMultiDict()
    omd.addlist('a', [])
    assert 'a' not in omd
    assert omd.getlist('a') == []


def test_omd_addlist_iterator():
    omd = OrderedMultiDict()
    omd.addlist('a', iter([1, 2, 3]))
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_get_default():
    omd = OrderedMultiDict()
    assert omd.get('x') is None
    assert omd.get('x', 5) == 5
    omd['x'] = 1
    assert omd.get('x') == 1


def test_omd_getlist_default():
    omd = OrderedMultiDict()
    assert omd.getlist('x') == []
    assert omd.getlist('x', None) is None
    assert omd.getlist('x', 42) == 42


def test_omd_getlist_returns_copy():
    omd = OrderedMultiDict([('a', 1)])
    lst = omd.getlist('a')
    lst.append(99)
    assert omd.getlist('a') == [1]


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.clear()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items(multi=True) == []


def test_omd_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('a', 5) is None
    assert omd.setdefault('b', 7) == 7
    assert omd['b'] == 7


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    c = omd.copy()
    assert isinstance(c, OrderedMultiDict)
    assert c == omd
    c.add('a', 4)
    assert omd.getlist('a') == [1, 3]


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b', 'c'])
    assert omd.items() == [('a', None), ('b', None), ('c', None)]
    omd2 = OrderedMultiDict.fromkeys(['a', 'b'], 0)
    assert omd2.items() == [('a', 0), ('b', 0)]


def test_omd_init_toomany_args():
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])


def test_omd_init_kwargs():
    omd = OrderedMultiDict(a=1, b=2)
    assert omd.items() == [('a', 1), ('b', 2)]


def test_omd_init_iterable():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert omd.items() == [('a', 1), ('b', 2)]


# ---------------------------------------------------------------------------
# update / update_extend
# ---------------------------------------------------------------------------

def test_omd_update_from_dict():
    omd = OrderedMultiDict([('a', 1)])
    omd.update({'a': 2, 'b': 3})
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


def test_omd_update_from_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    other = OrderedMultiDict([('a', 10), ('a', 20), ('b', 3)])
    omd.update(other)
    assert omd.getlist('a') == [10, 20]
    assert omd['b'] == 3


def test_omd_update_iterable():
    omd = OrderedMultiDict([('a', 1)])
    omd.update([('a', 2), ('b', 3)])
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


def test_omd_update_self_noop():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.getlist('a') == [1]


def test_omd_update_kwargs():
    omd = OrderedMultiDict()
    omd.update({'a': 1}, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('b', 3)])
    assert omd.getlist('a') == [1, 2]
    assert omd['b'] == 3


def test_omd_update_extend_from_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 2), ('a', 3)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_update_extend_from_dict():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd.items() == [('a', 1), ('b', 2)]


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_omd_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'a': 2, 'b': 3}
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


# ---------------------------------------------------------------------------
# pop / popall / poplast / del
# ---------------------------------------------------------------------------

def test_omd_delitem():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd['b'] == 3
    with pytest.raises(KeyError):
        del omd['a']


def test_omd_pop():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.pop('a') == 2
    assert 'a' not in omd
    with pytest.raises(KeyError):
        omd.pop('a')
    assert omd.pop('a', 'dflt') == 'dflt'


def test_omd_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('x', None) is None
    assert omd.pop('x', 5) == 5


def test_omd_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd
    with pytest.raises(KeyError):
        omd.popall('a')
    assert omd.popall('a', 'dflt') == 'dflt'


def test_omd_popall_default_value():
    omd = OrderedMultiDict()
    assert omd.popall('x', []) == []


def test_omd_poplast_with_key():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.poplast('a') == 2
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_omd_poplast_no_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast() == 3
    assert omd.getlist('a') == [1]
    assert omd.poplast() == 2


def test_omd_poplast_empty():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='dflt') == 'dflt'


def test_omd_poplast_missing_key():
    omd = OrderedMultiDict([('a', 1)])
    with pytest.raises(KeyError):
        omd.poplast('x')
    assert omd.poplast('x', default=7) == 7


# ---------------------------------------------------------------------------
# iteration
# ---------------------------------------------------------------------------

def test_omd_iteritems_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items(multi=False) == [('a', 3), ('b', 2)]


def test_omd_iterkeys_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.keys(multi=False) == ['a', 'b']
    assert list(omd) == ['a', 'b']


def test_omd_itervalues_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.values(multi=False) == [3, 2]


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_reversed_multiple():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
    assert list(reversed(omd)) == ['c', 'b', 'a']


def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    d = omd.todict(multi=True)
    assert d == {'a': [1, 3], 'b': [2]}
    d['a'].append(99)
    assert omd.getlist('a') == [1, 3]


# ---------------------------------------------------------------------------
# sorting / inverting / counting
# ---------------------------------------------------------------------------

def test_omd_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert s.items() == [(2, 2), (1, 1), (0, 0)]


def test_omd_sorted_with_key():
    omd = OrderedMultiDict(zip('hello', 'world'))
    s = omd.sorted(key=lambda i: i[1])
    assert s.items() == [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]


def test_omd_sortedvalues():
    omd = OrderedMultiDict()
    omd.addlist('even', [6, 2])
    omd.addlist('odd', [1, 5])
    omd.add('even', 4)
    omd.add('odd', 3)
    somd = omd.sortedvalues()
    assert somd.getlist('even') == [2, 4, 6]
    assert somd.keys(multi=True) == omd.keys(multi=True)
    assert omd != somd


def test_omd_sortedvalues_reverse():
    omd = OrderedMultiDict()
    omd.addlist('a', [3, 1, 2])
    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('a') == [3, 2, 1]


def test_omd_inverted():
    omd = OMD([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted() == omd


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    c = omd.counts()
    assert isinstance(c, OrderedMultiDict)
    assert c['a'] == 2
    assert c['b'] == 1


# ---------------------------------------------------------------------------
# equality
# ---------------------------------------------------------------------------

def test_omd_eq_identity():
    omd = OrderedMultiDict()
    assert omd == omd


def test_omd_eq_omd():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('a', 1), ('b', 2)])
    assert a == b
    assert not (a != b)
    c = OrderedMultiDict([('a', 1), ('b', 3)])
    assert a != c


def test_omd_eq_omd_order_matters_multi():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('b', 2), ('a', 1)])
    assert a != b


def test_omd_eq_dict():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    assert a == {'a': 1, 'b': 2}
    assert a != {'a': 1}
    assert a != {'a': 1, 'b': 3}


def test_omd_eq_nonmapping():
    omd = OrderedMultiDict([('a', 1)])
    assert omd != 5
    assert omd != [('a', 1)]


def test_omd_eq_other_no_len():
    omd = OrderedMultiDict()
    assert omd != object()


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def test_omd_viewkeys():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    vk = omd.viewkeys()
    assert set(vk) == {'a', 'b'}


def test_omd_viewvalues():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    vv = omd.viewvalues()
    assert set(vv) == {3, 2}


def test_omd_viewitems():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    vi = omd.viewitems()
    assert set(vi) == {('a', 3), ('b', 2)}


# ---------------------------------------------------------------------------
# pickle / reduce
# ---------------------------------------------------------------------------

def test_omd_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    data = pickle.dumps(omd)
    omd2 = pickle.loads(data)
    assert omd2.items(multi=True) == omd.items(multi=True)


def test_omd_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('a', 2)]
    omd2 = OrderedMultiDict()
    omd2.__setstate__(state)
    assert omd2.items(multi=True) == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def test_fiod_basic():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.getlist('a') == [1, 3]
    assert omd['a'] == 3
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.keys(multi=False) == ['a', 'b']
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_fiod_remove_all_on_set():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    omd['a'] = 5
    assert omd.getlist('a') == [5]


def test_fiod_pop_reverse():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('c', 3)
    assert list(reversed(omd)) == ['c', 'b', 'a']


def test_fiod_poplast():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    assert omd.poplast() == 2
    assert omd.poplast() == 1
    assert len(omd) == 0


def test_fiod_update():
    omd = FastIterOrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 2)
    omd.add('b', 3)
    assert omd.getlist('a') == [1, 2]
    assert 'b' in omd


def test_fiod_iteration_after_removes():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    omd.add('b', 3)
    omd.pop('a')
    assert omd.keys(multi=True) == ['b']
    assert omd.items(multi=True) == [('b', 3)]


def test_fiod_update_extend():
    omd = FastIterOrderedMultiDict()
    omd.update_extend([('a', 1), ('a', 2), ('b', 3)])
    assert omd.getlist('a') == [1, 2]
    assert omd['b'] == 3


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_oto_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2


def test_oto_overwrite_via_inv():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_oto_overwrite_via_setitem():
    oto = OneToOne({'a': 1})
    oto['b'] = 1
    assert 'a' not in oto
    assert oto['b'] == 1
    assert oto.inv[1] == 'b'


def test_oto_overwrite_existing_key():
    oto = OneToOne({'a': 1})
    oto['a'] = 2
    assert oto['a'] == 2
    assert 1 not in oto.inv
    assert oto.inv[2] == 'a'


def test_oto_dupe_values():
    oto = OneToOne({'a': 1, 'b': 1})
    # non-unique: last value wins
    assert len(oto) == 1


def test_oto_unique_dupe():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_oto_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_oto_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = []


def test_oto_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert len(oto.inv) == 1


def test_oto_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_oto_copy():
    oto = OneToOne({'a': 1})
    c = oto.copy()
    assert isinstance(c, OneToOne)
    assert c == {'a': 1}
    assert c.inv[1] == 'a'


def test_oto_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_oto_pop_default():
    oto = OneToOne()
    assert oto.pop('x', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        oto.pop('x')


def test_oto_popitem():
    oto = OneToOne({'a': 1})
    k, v = oto.popitem()
    assert (k, v) == ('a', 1)
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_oto_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.inv[1] == 'a'
    assert oto.setdefault('a', 99) == 1


def test_oto_update():
    oto = OneToOne({'a': 1})
    oto.update({'b': 2})
    assert oto['b'] == 2
    assert oto.inv[2] == 'b'


def test_oto_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1), ('b', 2)])
    assert oto == {'a': 1, 'b': 2}


def test_oto_update_kwargs():
    oto = OneToOne()
    oto.update({}, a=1)
    assert oto['a'] == 1


def test_oto_update_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto.update({'a': []})


def test_oto_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto) == "OneToOne({'a': 1})"


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_m2m_basic():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    assert m2m['a'] == frozenset({1, 2})
    assert m2m.inv[1] == frozenset({'a'})
    assert 'a' in m2m
    assert len(m2m) == 1


def test_m2m_get():
    m2m = ManyToMany()
    assert m2m.get('x') == frozenset()
    assert m2m.get('x', None) is None
    m2m.add('a', 1)
    assert m2m.get('a') == frozenset({1})


def test_m2m_setitem():
    m2m = ManyToMany()
    m2m['a'] = [1, 2, 3]
    assert m2m['a'] == frozenset({1, 2, 3})
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset({2, 3})
    assert 1 not in m2m.inv


def test_m2m_delitem():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    del m2m['a']
    assert 'a' not in m2m
    assert 1 not in m2m.inv
    assert 2 not in m2m.inv


def test_m2m_remove():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.remove('a', 1)
    assert m2m['a'] == frozenset({2})
    m2m.remove('a', 2)
    assert 'a' not in m2m
    assert 2 not in m2m.inv


def test_m2m_replace():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.replace('a', 'b')
    assert 'a' not in m2m
    assert m2m['b'] == frozenset({1, 2})
    assert m2m.inv[1] == frozenset({'b'})


def test_m2m_replace_missing():
    m2m = ManyToMany()
    m2m.replace('x', 'y')
    assert 'x' not in m2m


def test_m2m_update_iterable():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('a', 2), ('b', 3)])
    assert m2m['a'] == frozenset({1, 2})
    assert m2m['b'] == frozenset({3})


def test_m2m_update_m2m():
    a = ManyToMany()
    a.add('a', 1)
    b = ManyToMany()
    b.add('a', 2)
    b.add('b', 3)
    a.update(b)
    assert a['a'] == frozenset({1, 2})
    assert a['b'] == frozenset({3})


def test_m2m_update_mapping():
    m2m = ManyToMany()
    m2m.update({'a': 1})
    assert m2m['a'] == frozenset({1})


def test_m2m_iterkeys_len():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('b', 2)
    assert set(m2m.keys()) == {'a', 'b'}
    assert set(m2m) == {'a', 'b'}
    assert sorted(m2m.iteritems()) == [('a', 1), ('b', 2)]


def test_m2m_eq():
    a = ManyToMany()
    a.add('a', 1)
    b = ManyToMany()
    b.add('a', 1)
    assert a == b
    c = ManyToMany()
    assert a != c
    assert a != {}


def test_m2m_repr():
    m2m = ManyToMany()
    m2m.add('a', 1)
    assert repr(m2m) == "ManyToMany([('a', 1)])"


def test_m2m_init_items():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    assert m2m['a'] == frozenset({1, 2})


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_default():
    d = {'a': 1, 'b': 2}
    assert subdict(d) == d


def test_subdict_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_and_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_keep_missing():
    d = {'a': 1}
    assert subdict(d, keep=['a', 'z']) == {'a': 1}


def test_subdict_returns_type():
    omd = OrderedMultiDict([('a', 1)])
    result = subdict(omd, keep=['a'])
    assert isinstance(result, OrderedMultiDict)


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert fd == {'a': 1, 'b': 2}
    assert len(fd) == 2


def test_frozendict_hash():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert isinstance(h, int)
    assert h == hash(fd)


def test_frozendict_hash_unhashable():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)


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
        fd.popitem()
    with pytest.raises(TypeError):
        fd.clear()
    with pytest.raises(TypeError):
        fd |= {'b': 2}


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert isinstance(fd2, FrozenDict)
    assert fd2 == {'a': 1, 'b': 2}
    assert fd == {'a': 1}


def test_frozendict_updated_kwargs():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated(b=2)
    assert fd2 == {'a': 1, 'b': 2}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 0)
    assert isinstance(fd, FrozenDict)
    assert fd == {'a': 0, 'b': 0}


def test_frozendict_copy():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_reduce():
    fd = FrozenDict({'a': 1})
    data = pickle.dumps(fd)
    fd2 = pickle.loads(data)
    assert fd2 == {'a': 1}
    assert isinstance(fd2, FrozenDict)


def test_frozendict_as_key():
    fd = FrozenDict({'a': 1})
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'
