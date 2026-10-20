import copy
import pickle

import pytest

from boltons.dictutils import (
    OrderedMultiDict,
    FastIterOrderedMultiDict,
    OneToOne,
    ManyToMany,
    subdict,
    FrozenHashError,
    FrozenDict,
    MultiDict,
    OMD,
)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

def test_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


# ---------------------------------------------------------------------------
# OrderedMultiDict construction and basic protocol
# ---------------------------------------------------------------------------

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
    assert bool(omd) is False


def test_omd_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])


def test_omd_from_items():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert len(omd) == 2
    assert omd['a'] == 3
    assert omd['b'] == 2
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]
    assert omd.getlist('c') == []
    assert omd.getlist('c', default=None) is None
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items(multi=False) == [('a', 3), ('b', 2)]


def test_omd_from_kwargs():
    omd = OrderedMultiDict(a=1, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_from_dict():
    omd = OrderedMultiDict({'a': 1, 'b': 2})
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_get():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    assert omd.get('a') == 2
    assert omd.get('missing') is None
    assert omd.get('missing', 'default') == 'default'


def test_omd_setitem_replaces_all():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    omd['a'] = 10
    assert omd.getlist('a') == [10]
    assert omd.items(multi=True) == [('b', 3), ('a', 10)]


def test_omd_setitem_new():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    assert omd['a'] == 1
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_omd_delitem():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 3)]
    with pytest.raises(KeyError):
        del omd['a']


def test_omd_contains():
    omd = OrderedMultiDict([('a', 1)])
    assert 'a' in omd
    assert 'b' not in omd


def test_omd_add():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 2)
    omd.add('b', 3)
    assert omd.getlist('a') == [1, 2]
    assert omd.getlist('b') == [3]
    assert omd['a'] == 2


def test_omd_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]


def test_omd_addlist_iterator():
    omd = OrderedMultiDict()
    omd.addlist('a', iter([1, 2, 3]))
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_addlist_empty():
    omd = OrderedMultiDict()
    omd.addlist('a', [])
    assert 'a' not in omd


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []
    # Still usable after clear.
    omd.add('c', 5)
    assert omd.items(multi=True) == [('c', 5)]


def test_omd_setdefault():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.setdefault('a', 99) == 1
    assert omd.getlist('a') == [1]
    assert omd.setdefault('b') is None
    assert omd.getlist('b') == [None]
    assert omd.setdefault('c', 5) == 5
    assert omd.getlist('c') == [5]


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    cp = omd.copy()
    assert isinstance(cp, OrderedMultiDict)
    assert cp.items(multi=True) == omd.items(multi=True)
    assert cp is not omd
    cp.add('a', 4)
    assert omd.getlist('a') == [1, 3]


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd.items(multi=True) == [('a', None), ('b', None)]
    omd2 = OrderedMultiDict.fromkeys(['a', 'b'], 5)
    assert omd2.getlist('a') == [5]
    assert omd2.getlist('b') == [5]


def test_omd_update_from_omd():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    other = OrderedMultiDict([('a', 10), ('a', 11), ('c', 3)])
    omd.update(other)
    assert omd.getlist('a') == [10, 11]
    assert omd.getlist('c') == [3]
    assert omd.getlist('b') == [2]


def test_omd_update_from_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update({'a': 10, 'c': 3})
    assert omd.getlist('a') == [10]
    assert omd.getlist('c') == [3]


def test_omd_update_from_iterable():
    omd = OrderedMultiDict([('a', 1)])
    omd.update([('a', 10), ('a', 11), ('b', 2)])
    assert omd.getlist('a') == [10, 11]
    assert omd.getlist('b') == [2]


def test_omd_update_with_kwargs():
    omd = OrderedMultiDict()
    omd.update({'a': 1}, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_omd_update_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.getlist('a') == [1]


def test_omd_update_extend_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 2), ('a', 3)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_update_extend_dict():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend({'a': 2, 'b': 3})
    assert omd.getlist('a') == [1, 2]
    assert omd.getlist('b') == [3]


def test_omd_update_extend_iterable():
    omd = OrderedMultiDict()
    omd.update_extend([('a', 1), ('a', 2)])
    assert omd.getlist('a') == [1, 2]


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_omd_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'a': 5, 'b': 2}
    assert omd.getlist('a') == [5]
    assert omd['b'] == 2


# ---------------------------------------------------------------------------
# pop / popall / poplast
# ---------------------------------------------------------------------------

def test_omd_pop():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.pop('a') == 2
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 3)]


def test_omd_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('missing', 'x') == 'x'
    with pytest.raises(KeyError):
        omd.pop('missing')


def test_omd_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd


def test_omd_popall_default():
    omd = OrderedMultiDict()
    assert omd.popall('missing', 'x') == 'x'
    with pytest.raises(KeyError):
        omd.popall('missing')


def test_omd_poplast_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_omd_poplast_removes_key_when_empty():
    omd = OrderedMultiDict([('a', 1)])
    omd.poplast('a')
    assert 'a' not in omd
    assert len(omd) == 0


def test_omd_poplast_no_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast() == 3
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]
    assert omd.poplast() == 2
    assert omd.poplast() == 1
    assert len(omd) == 0


def test_omd_poplast_empty():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='d') == 'd'


def test_omd_poplast_default():
    omd = OrderedMultiDict()
    assert omd.poplast('missing', default='d') == 'd'
    with pytest.raises(KeyError):
        omd.poplast('missing')


# ---------------------------------------------------------------------------
# Iteration order and views
# ---------------------------------------------------------------------------

def test_omd_iteritems_order():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iteritems(multi=False)) == [('a', 3), ('b', 2)]


def test_omd_iterkeys_order():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.iterkeys(multi=False)) == ['a', 'b']
    assert list(iter(omd)) == ['a', 'b']


def test_omd_itervalues_order():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]
    assert list(omd.itervalues(multi=False)) == [3, 2]


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_reversed_empty():
    omd = OrderedMultiDict()
    assert list(reversed(omd)) == []


def test_omd_views():
    from collections.abc import KeysView, ValuesView, ItemsView
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert isinstance(omd.viewkeys(), KeysView)
    assert isinstance(omd.viewvalues(), ValuesView)
    assert isinstance(omd.viewitems(), ItemsView)
    assert set(omd.viewkeys()) == {'a', 'b'}
    assert set(omd.viewvalues()) == {1, 2}
    assert set(omd.viewitems()) == {('a', 1), ('b', 2)}


# ---------------------------------------------------------------------------
# todict
# ---------------------------------------------------------------------------

def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    multi = omd.todict(multi=True)
    assert multi == {'a': [1, 3], 'b': [2]}
    # Ensure copies are returned.
    multi['a'].append(99)
    assert omd.getlist('a') == [1, 3]


# ---------------------------------------------------------------------------
# sorted / sortedvalues / inverted / counts
# ---------------------------------------------------------------------------

def test_omd_sorted():
    omd = OrderedMultiDict([('b', 4), ('a', 1), ('c', 3)])
    s = omd.sorted()
    assert isinstance(s, OrderedMultiDict)
    assert s.items(multi=True) == [('a', 1), ('b', 4), ('c', 3)]


def test_omd_sorted_reverse():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_omd_sorted_key():
    omd = OrderedMultiDict([('c', 3), ('a', 1), ('b', 2)])
    s = omd.sorted(key=lambda kv: kv[1])
    assert s.items(multi=True) == [('a', 1), ('b', 2), ('c', 3)]


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


def test_omd_sortedvalues_reverse():
    omd = OrderedMultiDict()
    omd.addlist('e', [1, 3, 2])
    s = omd.sortedvalues(reverse=True)
    assert s.getlist('e') == [3, 2, 1]


def test_omd_inverted():
    omd = OrderedMultiDict([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted().items(multi=True) == [(0, 2), (1, 2)]


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts['a'] == 2
    assert counts['b'] == 1


# ---------------------------------------------------------------------------
# eq / ne
# ---------------------------------------------------------------------------

def test_omd_eq_same_object():
    omd = OrderedMultiDict([('a', 1)])
    assert omd == omd


def test_omd_eq_omd():
    a = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    b = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert a == b
    c = OrderedMultiDict([('a', 3), ('b', 2), ('a', 1)])
    assert a != c


def test_omd_eq_length_mismatch():
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 1), ('b', 2)])
    assert a != b


def test_omd_eq_dict():
    a = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert a == {'a': 3, 'b': 2}
    assert a != {'a': 1, 'b': 2}
    assert a != {'a': 3}


def test_omd_eq_non_mapping():
    a = OrderedMultiDict([('a', 1)])
    assert a != 5
    assert (a == 5) is False


def test_omd_ne():
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 2)])
    assert a != b
    assert not (a != a)


# ---------------------------------------------------------------------------
# repr / pickle / copy
# ---------------------------------------------------------------------------

def test_omd_repr():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"


def test_omd_repr_empty():
    omd = OrderedMultiDict()
    assert repr(omd) == "OrderedMultiDict([])"


def test_omd_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('b', 2), ('a', 3)]
    new = OrderedMultiDict()
    new.__setstate__(state)
    assert new.items(multi=True) == omd.items(multi=True)


def test_omd_pickle():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    restored = pickle.loads(data)
    assert restored.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_omd_reduce():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    cls, args, state = omd.__reduce__()
    assert cls is OrderedMultiDict
    assert args == ()
    assert state == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def test_fiomd_basic_add_iter():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items(multi=False) == [('a', 3), ('b', 2)]
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.keys(multi=False) == ['a', 'b']
    assert omd.values(multi=True) == [1, 2, 3]


def test_fiomd_remove():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert omd.popall('a') == [1, 3]
    assert omd.items(multi=True) == [('b', 2)]


def test_fiomd_remove_single():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    assert omd.poplast('a') == 2
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_fiomd_reversed():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    assert list(reversed(omd)) == ['b', 'a']


def test_fiomd_setitem_and_delitem():
    omd = FastIterOrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.getlist('a') == [1, 3]
    del omd['a']
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]


def test_fiomd_clear():
    omd = FastIterOrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.clear()
    assert len(omd) == 0
    omd.add('c', 3)
    assert omd.items(multi=True) == [('c', 3)]


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_oto_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'
    assert len(oto) == 2
    assert len(oto.inv) == 2


def test_oto_overwrite_key():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['a'] = 3
    assert oto['a'] == 3
    assert 1 not in oto.inv
    assert oto.inv[3] == 'a'


def test_oto_overwrite_from_inv():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_oto_duplicate_values_collapsed():
    oto = OneToOne({'a': 1, 'b': 1})
    # Last write wins, with only one entry in the reverse map.
    assert len(oto) == 1
    assert len(oto.inv) == 1


def test_oto_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_oto_unique_dupe():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_oto_unique_across_inputs():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)


def test_oto_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_oto_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_oto_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    cp = oto.copy()
    assert isinstance(cp, OneToOne)
    assert cp == oto
    assert cp.inv[1] == 'a'


def test_oto_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 1 not in oto.inv
    assert oto.pop('missing', 'd') == 'd'
    with pytest.raises(KeyError):
        oto.pop('missing')


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
    assert oto.setdefault('a', 2) == 1


def test_oto_update_dict():
    oto = OneToOne()
    oto.update({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_oto_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1), ('b', 2)])
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_oto_update_kwargs():
    oto = OneToOne()
    oto.update({}, b=2)
    assert oto['b'] == 2


def test_oto_setitem_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = []


def test_oto_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto).startswith('OneToOne(')


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_m2m_basic():
    m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
    assert m['a'] == frozenset([1, 2])
    assert m['b'] == frozenset([2])
    assert m.inv[2] == frozenset(['a', 'b'])
    assert m.inv[1] == frozenset(['a'])
    assert 'a' in m
    assert 'c' not in m
    assert set(iter(m)) == {'a', 'b'}
    assert len(m) == 2


def test_m2m_get():
    m = ManyToMany([('a', 1)])
    assert m.get('a') == frozenset([1])
    assert m.get('missing') == frozenset()
    assert m.get('missing', None) is None


def test_m2m_setitem():
    m = ManyToMany()
    m['a'] = [1, 2]
    assert m['a'] == frozenset([1, 2])
    m['a'] = [2, 3]
    assert m['a'] == frozenset([2, 3])
    assert m.inv[1] == frozenset()


def test_m2m_delitem():
    m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
    del m['a']
    assert 'a' not in m
    assert m.inv[1] == frozenset()
    assert m.inv[2] == frozenset(['b'])


def test_m2m_update_from_m2m():
    m = ManyToMany([('a', 1)])
    m2 = ManyToMany([('a', 2), ('b', 3)])
    m.update(m2)
    assert m['a'] == frozenset([1, 2])
    assert m['b'] == frozenset([3])
    assert m.inv[2] == frozenset(['a'])


def test_m2m_update_from_mapping():
    m = ManyToMany()
    m.update({'a': 1, 'b': 2})
    assert m['a'] == frozenset([1])
    assert m['b'] == frozenset([2])


def test_m2m_add():
    m = ManyToMany()
    m.add('a', 1)
    m.add('a', 2)
    m.add('b', 1)
    assert m['a'] == frozenset([1, 2])
    assert m.inv[1] == frozenset(['a', 'b'])


def test_m2m_remove():
    m = ManyToMany([('a', 1), ('a', 2)])
    m.remove('a', 1)
    assert m['a'] == frozenset([2])
    m.remove('a', 2)
    assert 'a' not in m


def test_m2m_replace():
    m = ManyToMany([('a', 1), ('a', 2)])
    m.replace('a', 'b')
    assert 'a' not in m
    assert m['b'] == frozenset([1, 2])
    assert m.inv[1] == frozenset(['b'])


def test_m2m_replace_missing():
    m = ManyToMany([('a', 1)])
    m.replace('missing', 'b')
    assert 'missing' not in m
    assert 'b' not in m


def test_m2m_iteritems():
    m = ManyToMany([('a', 1), ('a', 2)])
    items = sorted(m.iteritems())
    assert items == [('a', 1), ('a', 2)]


def test_m2m_keys():
    m = ManyToMany([('a', 1), ('b', 2)])
    assert set(m.keys()) == {'a', 'b'}


def test_m2m_eq():
    m1 = ManyToMany([('a', 1)])
    m2 = ManyToMany([('a', 1)])
    assert m1 == m2
    m3 = ManyToMany([('a', 2)])
    assert m1 != m3
    assert m1 != {'a': {1}}


def test_m2m_repr():
    m = ManyToMany([('a', 1)])
    assert 'ManyToMany' in repr(m)


def test_m2m_empty():
    m = ManyToMany()
    assert len(m) == 0
    assert list(m.iteritems()) == []
    assert m.get('anything') == frozenset()
    assert m.data == {}
    assert m.inv.data == {}


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_no_args():
    d = {'a': 1, 'b': 2}
    result = subdict(d)
    assert result == {'a': 1, 'b': 2}
    assert result is not d


def test_subdict_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_and_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_empty():
    assert subdict({}) == {}
    assert subdict({'a': 1}, keep=[]) == {}


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert dict(fd) == {'a': 1, 'b': 2}
    assert len(fd) == 2


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1, 'b': 2})
    fd2 = FrozenDict({'a': 1, 'b': 2})
    assert hash(fd) == hash(fd2)
    assert {fd, fd2} == {fd}


def test_frozendict_used_as_key():
    fd = FrozenDict({'a': 1})
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'


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


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert isinstance(fd2, FrozenDict)
    assert dict(fd2) == {'a': 1, 'b': 2}
    assert dict(fd) == {'a': 1}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 5)
    assert dict(fd) == {'a': 5, 'b': 5}


def test_frozendict_copy():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    restored = pickle.loads(pickle.dumps(fd))
    assert isinstance(restored, FrozenDict)
    assert dict(restored) == {'a': 1, 'b': 2}


def test_frozendict_unhashable_value():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_hash_cached():
    fd = FrozenDict({'a': 1})
    h1 = hash(fd)
    h2 = hash(fd)
    assert h1 == h2


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd).startswith('FrozenDict(')


def test_frozendict_eq_dict():
    fd = FrozenDict({'a': 1})
    assert fd == {'a': 1}


def test_frozen_hash_error_subclass():
    assert issubclass(FrozenHashError, TypeError)
