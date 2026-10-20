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


# --------------------------------------------------------------------------- #
# OrderedMultiDict basics
# --------------------------------------------------------------------------- #

def test_omd_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


def test_omd_empty():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items() == []
    assert omd.keys() == []
    assert omd.values() == []
    assert omd.items(multi=True) == []
    assert repr(omd) == "OrderedMultiDict([])"


def test_omd_construction_and_precedence():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd['a'] == 3
    assert omd['b'] == 2
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', 'deflt') == 'deflt'
    assert omd.get('missing') is None
    assert omd.get('missing', 'd') == 'd'
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]


def test_omd_init_errors():
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])


def test_omd_init_kwargs():
    omd = OrderedMultiDict(a=1, b=2)
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_omd_add_and_addlist():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 2)
    omd.add('b', 3)
    assert omd.items(multi=True) == [('a', 1), ('a', 2), ('b', 3)]
    omd.addlist('c', [10, 11])
    assert omd.getlist('c') == [10, 11]
    omd.addlist('d', [])
    assert 'd' not in omd
    # generator is materialized
    omd.addlist('e', (x for x in range(3)))
    assert omd.getlist('e') == [0, 1, 2]


def test_omd_getitem_setitem():
    omd = OrderedMultiDict()
    omd['x'] = 1
    assert omd['x'] == 1
    omd['x'] = 2  # overwrite: removes previous values
    assert omd.getlist('x') == [2]
    assert omd.items(multi=True) == [('x', 2)]
    with pytest.raises(KeyError):
        omd['nope']


def test_omd_getlist_copy_is_safe():
    omd = OrderedMultiDict([('a', 1)])
    lst = omd.getlist('a')
    lst.append(99)
    assert omd.getlist('a') == [1]


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.clear()
    assert omd.items(multi=True) == []
    assert len(omd) == 0
    assert list(reversed(omd)) == []


def test_omd_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('b', 5) == 5
    assert omd.setdefault('b', 999) == 5
    assert omd.getlist('b') == [5]


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    cp = omd.copy()
    assert cp == omd
    assert cp is not omd
    assert cp.items(multi=True) == [('a', 1), ('a', 2)]
    cp.add('a', 3)
    assert omd.getlist('a') == [1, 2]


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd.items(multi=True) == [('a', None), ('b', None)]
    omd2 = OrderedMultiDict.fromkeys(['a', 'b'], 0)
    assert omd2.items(multi=True) == [('a', 0), ('b', 0)]


def test_omd_eq_and_ne():
    omd1 = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd2 = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd3 = OrderedMultiDict([('a', 3), ('b', 2)])
    omd4 = OrderedMultiDict([('a', 1), ('a', 3), ('b', 2)])
    assert omd1 == omd2
    assert not (omd1 != omd2)
    assert omd1 != omd3
    assert omd1 != omd4  # order of items matters
    # Note: comparison to plain dict compares only per-key latest values,
    # not total multi-item length, so omd1 == {'a': 3, 'b': 2}.
    assert omd1 == {'a': 3, 'b': 2}
    assert omd3 == {'a': 3, 'b': 2}
    assert omd3 == {'b': 2, 'a': 3}
    assert omd3 != {'a': 3, 'b': 99}
    assert omd3 != {'a': 3, 'c': 2}
    # non-mapping returns False
    assert omd1 != [1, 2, 3]
    assert omd1.__eq__(5) is False
    same = OrderedMultiDict([('a', 1)])
    assert same.__eq__(same) is True


def test_omd_ior():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd |= [('a', 3), ('b', 4)]
    assert omd.items(multi=True) == [('a', 3), ('b', 4)]


def test_omd_pop_and_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd
    assert omd.pop('b') == 3
    assert omd.items(multi=True) == []
    with pytest.raises(KeyError):
        omd.pop('nope')
    assert omd.pop('nope', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        omd.popall('nope')
    assert omd.popall('nope', 'dflt') == 'dflt'


def test_omd_poplast():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast() == 3  # most recently inserted key is 'a'
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]
    assert omd.poplast('a') == 1
    # 'a' now removed from OMD
    assert 'a' not in omd
    assert omd.poplast() == 2
    assert list(omd.items(multi=True)) == []
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='dflt') == 'dflt'
    with pytest.raises(KeyError):
        omd.poplast('missing')
    assert omd.poplast('missing', 'dflt') == 'dflt'


def test_omd_poplast_partial():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    assert omd.poplast('a') == 2
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_omd_iteritems_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iterkeys()) == ['a', 'b']
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.itervalues()) == [3, 2]
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']
    assert list(reversed(OrderedMultiDict())) == []


def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    td = omd.todict(multi=True)
    assert td == {'a': [1, 3], 'b': [2]}
    # safe to mutate
    td['a'].append(99)
    assert omd.getlist('a') == [1, 3]


def test_omd_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    assert omd.sorted(reverse=True).items(multi=True) == [(2, 2), (1, 1), (0, 0)]
    omd2 = OrderedMultiDict(zip('hello', 'world'))
    result = omd2.sorted(key=lambda i: i[1])
    assert result.items(multi=True) == [
        ('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')
    ]


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
    omd = OMD([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted() == omd


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts.items(multi=True) == [('a', 2), ('b', 1)]


def test_omd_views():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    vk = omd.viewkeys()
    vv = omd.viewvalues()
    vi = omd.viewitems()
    assert set(vk) == {'a', 'b'}
    assert set(vv) == {1, 2}
    assert set(vi) == {('a', 1), ('b', 2)}


def test_omd_update_from_mapping():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    omd.update({'a': 10, 'c': 4})
    # 'a' key is deleted and re-inserted at end of order
    assert omd.items(multi=True) == [('b', 3), ('a', 10), ('c', 4)]


def test_omd_update_from_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 10), ('a', 11), ('c', 4)])
    omd.update(other)
    assert omd.items(multi=True) == [('b', 3), ('a', 10), ('a', 11), ('c', 4)]


def test_omd_update_from_iterable():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update([('a', 3), ('a', 4), ('b', 5)])
    assert omd.items(multi=True) == [('a', 3), ('a', 4), ('b', 5)]


def test_omd_update_self_is_noop():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update(omd)
    assert omd.items(multi=True) == [('a', 1), ('a', 2)]


def test_omd_update_with_kwargs():
    omd = OrderedMultiDict()
    omd.update({'a': 1}, b=2)
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_omd_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('b', 3)])
    assert omd.items(multi=True) == [('a', 1), ('a', 2), ('b', 3)]


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update_extend(omd)
    # iterating over self.items() while mutating; actual result on this
    # implementation yields the partial interleaving below.
    assert omd.items(multi=True) == [('a', 1), ('a', 2), ('a', 2)]


def test_omd_update_extend_from_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 2), ('a', 3)])
    omd.update_extend(other)
    assert omd.items(multi=True) == [('a', 1), ('a', 2), ('a', 3)]


def test_omd_update_extend_from_mapping():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_omd_delitem():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        del omd['a']


def test_omd_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    omd2 = pickle.loads(data)
    assert omd2.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd2.__class__ is OrderedMultiDict


def test_omd_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('a', 2)]
    omd2 = OrderedMultiDict()
    omd2.__setstate__(state)
    assert omd2.items(multi=True) == [('a', 1), ('a', 2)]


def test_omd_reduce():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    cls, args, state = omd.__reduce__()
    assert cls is OrderedMultiDict
    assert args == ()
    assert state == [('a', 1), ('a', 2)]


# --------------------------------------------------------------------------- #
# FastIterOrderedMultiDict
# --------------------------------------------------------------------------- #

def test_fiomd_basic():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd['a'] == 3  # most recent
    assert omd.getlist('a') == [1, 3]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    # non-multi iteration for FIOMD yields the first-inserted per key
    assert omd.items() == [('a', 1), ('b', 2)]
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    # values() mirrors iteritems(), so non-multi yields first value per key
    assert omd.values() == [1, 2]
    assert omd.values(multi=True) == [1, 2, 3]


def test_fiomd_add_remove():
    omd = FastIterOrderedMultiDict()
    omd.add('x', 1)
    omd.add('x', 2)
    omd.add('y', 3)
    assert omd.items(multi=True) == [('x', 1), ('x', 2), ('y', 3)]
    omd.pop('x')
    assert 'x' not in omd
    assert omd.items(multi=True) == [('y', 3)]


def test_fiomd_poplast():
    omd = FastIterOrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.poplast('a') == 2
    assert omd.getlist('a') == [1]
    assert omd.poplast() == 3
    assert omd.items(multi=True) == [('a', 1)]


def test_fiomd_update_and_invert():
    omd = FastIterOrderedMultiDict([('a', 1)])
    omd.update({'b': 2})
    assert omd.items() == [('a', 1), ('b', 2)]
    inv = omd.inverted()
    assert isinstance(inv, FastIterOrderedMultiDict)
    assert inv.getlist(1) == ['a']
    assert inv.getlist(2) == ['b']


def test_fiomd_reversed():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_fiomd_clear():
    omd = FastIterOrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    omd.clear()
    assert omd.items(multi=True) == []


# --------------------------------------------------------------------------- #
# OneToOne
# --------------------------------------------------------------------------- #

def test_onetoone_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'
    assert len(oto) == 2
    assert len(oto.inv) == 2


def test_onetoone_overwrite():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto == {'c': 1, 'b': 2}
    assert oto.inv == {1: 'c', 2: 'b'}


def test_onetoone_setitem_overwrites_both_directions():
    oto = OneToOne()
    oto['a'] = 1
    oto['b'] = 2
    oto['a'] = 3  # old val 1 removed from inv
    assert oto == {'a': 3, 'b': 2}
    assert oto.inv == {3: 'a', 2: 'b'}
    oto['c'] = 2  # old key 'b' removed
    assert oto == {'a': 3, 'c': 2}
    assert oto.inv == {3: 'a', 2: 'c'}


def test_onetoone_duplicate_values_last_wins():
    oto = OneToOne({'a': 1, 'b': 1})
    # behavior: last value inserted wins
    assert dict(oto.inv) == {1: 'b'}
    assert len(oto) == 1
    assert oto == {'b': 1}


def test_onetoone_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto == {'a': 1, 'b': 2}


def test_onetoone_unique_dupe_raises():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_onetoone_unique_cross_inputs_raises():
    a_dict = {'a': 2}
    with pytest.raises(ValueError):
        OneToOne.unique(a_dict, b=2)


def test_onetoone_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert oto == {'b': 2}
    assert oto.inv == {2: 'b'}
    with pytest.raises(KeyError):
        del oto['a']


def test_onetoone_clear_and_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    cp = oto.copy()
    assert cp == oto
    assert cp.inv == oto.inv
    cp['c'] = 3
    assert 'c' not in oto
    oto.clear()
    assert dict(oto) == {}
    assert dict(oto.inv) == {}


def test_onetoone_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert oto.inv == {2: 'b'}
    assert oto.pop('zzz', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        oto.pop('zzz')


def test_onetoone_popitem():
    oto = OneToOne({'a': 1})
    k, v = oto.popitem()
    assert (k, v) == ('a', 1)
    assert dict(oto) == {}
    assert dict(oto.inv) == {}


def test_onetoone_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.setdefault('a', 2) == 1
    assert oto.inv[1] == 'a'


def test_onetoone_update():
    oto = OneToOne()
    oto.update({'a': 1, 'b': 2})
    assert oto == {'a': 1, 'b': 2}
    oto.update([('c', 3)])
    assert oto == {'a': 1, 'b': 2, 'c': 3}
    assert oto.inv == {1: 'a', 2: 'b', 3: 'c'}


def test_onetoone_update_kwargs():
    oto = OneToOne()
    oto.update({'a': 1}, b=2)
    assert oto == {'a': 1, 'b': 2}
    assert oto.inv == {1: 'a', 2: 'b'}


def test_onetoone_setitem_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = []


def test_onetoone_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto) == "OneToOne({'a': 1})"


# --------------------------------------------------------------------------- #
# ManyToMany
# --------------------------------------------------------------------------- #

def test_manytomany_basic():
    m2m = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([1])
    assert m2m.inv[1] == frozenset(['a', 'b'])
    assert m2m.inv[2] == frozenset(['a'])
    assert len(m2m) == 2
    assert 'a' in m2m
    assert 'z' not in m2m


def test_manytomany_get():
    m2m = ManyToMany([('a', 1)])
    assert m2m.get('a') == frozenset([1])
    assert m2m.get('missing') == frozenset()
    assert m2m.get('missing', 5) == 5


def test_manytomany_setitem():
    m2m = ManyToMany()
    m2m['a'] = [1, 2]
    assert m2m['a'] == frozenset([1, 2])
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset([2, 3])
    # val 1 was fully removed, so it's no longer a key in inv
    assert 1 not in m2m.inv


def test_manytomany_delitem():
    m2m = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
    del m2m['a']
    assert 'a' not in m2m
    assert m2m.inv[1] == frozenset(['b'])
    assert 2 not in m2m.inv
    with pytest.raises(KeyError):
        del m2m['a']


def test_manytomany_add_remove():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.add('b', 1)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['a', 'b'])
    m2m.remove('a', 1)
    assert m2m['a'] == frozenset([2])
    assert m2m.inv[1] == frozenset(['b'])
    m2m.remove('a', 2)
    assert 'a' not in m2m


def test_manytomany_replace():
    m2m = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
    m2m.replace('a', 'c')
    assert 'a' not in m2m
    assert m2m['c'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['c', 'b'])
    assert m2m.inv[2] == frozenset(['c'])
    # no-op
    m2m.replace('not-there', 'x')
    assert 'x' not in m2m


def test_manytomany_update_with_same_type():
    m2m = ManyToMany([('a', 1)])
    other = ManyToMany([('a', 2), ('b', 3)])
    m2m.update(other)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([3])
    assert m2m.inv[2] == frozenset(['a'])


def test_manytomany_update_with_mapping():
    m2m = ManyToMany()
    m2m.update({'a': 1, 'b': 2})
    assert m2m['a'] == frozenset([1])
    assert m2m['b'] == frozenset([2])
    assert m2m.inv[1] == frozenset(['a'])


def test_manytomany_update_iterable():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('a', 2)])
    assert m2m['a'] == frozenset([1, 2])


def test_manytomany_iter_and_repr():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    assert set(m2m) == {'a'}
    items = sorted(m2m.iteritems())
    assert items == [('a', 1), ('a', 2)]
    assert set(m2m.keys()) == {'a'}
    r = repr(m2m)
    assert r.startswith('ManyToMany(')


def test_manytomany_eq():
    m2m1 = ManyToMany([('a', 1)])
    m2m2 = ManyToMany([('a', 1)])
    m2m3 = ManyToMany([('a', 2)])
    assert m2m1 == m2m2
    assert m2m1 != m2m3
    assert m2m1 != {'a': frozenset([1])}


# --------------------------------------------------------------------------- #
# subdict
# --------------------------------------------------------------------------- #

def test_subdict_no_args():
    d = {'a': 1, 'b': 2}
    result = subdict(d)
    assert result == {'a': 1, 'b': 2}
    assert result is not d


def test_subdict_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_and_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_keep_missing_keys():
    d = {'a': 1}
    assert subdict(d, keep=['a', 'missing']) == {'a': 1}


def test_subdict_preserves_type():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    result = subdict(omd, keep=['a'])
    assert isinstance(result, OrderedMultiDict)


# --------------------------------------------------------------------------- #
# FrozenDict
# --------------------------------------------------------------------------- #

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert fd['b'] == 2
    assert len(fd) == 2
    assert dict(fd) == {'a': 1, 'b': 2}


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert isinstance(h, int)
    # stable between calls
    assert hash(fd) == h
    # usable as dict key / set element
    s = {fd}
    assert fd in s
    d = {fd: 'x'}
    assert d[FrozenDict({'a': 1})] == 'x'


def test_frozendict_hash_equality():
    fd1 = FrozenDict({'a': 1, 'b': 2})
    fd2 = FrozenDict({'b': 2, 'a': 1})
    assert fd1 == fd2
    assert hash(fd1) == hash(fd2)


def test_frozendict_unhashable_values_raises():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)
    # second attempt also raises (cached error)
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_immutable_operations():
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
    # original unchanged
    assert dict(fd) == {'a': 1}


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert isinstance(fd2, FrozenDict)
    assert dict(fd2) == {'a': 1, 'b': 2}
    assert dict(fd) == {'a': 1}
    fd3 = fd.updated([('c', 3)])
    assert dict(fd3) == {'a': 1, 'c': 3}
    fd4 = fd.updated(d=4)
    assert dict(fd4) == {'a': 1, 'd': 4}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'])
    assert isinstance(fd, FrozenDict)
    assert dict(fd) == {'a': None, 'b': None}
    fd2 = FrozenDict.fromkeys(['a', 'b'], 0)
    assert dict(fd2) == {'a': 0, 'b': 0}


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_copy():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    fd2 = pickle.loads(pickle.dumps(fd))
    assert isinstance(fd2, FrozenDict)
    assert dict(fd2) == {'a': 1, 'b': 2}
    assert hash(fd2) == hash(fd)
