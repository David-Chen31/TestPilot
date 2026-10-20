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


def test_omd_init_empty():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert omd.items(multi=True) == []
    assert omd.keys() == []
    assert omd.values() == []


def test_omd_init_dict_and_kwargs():
    omd = OrderedMultiDict({'a': 1}, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_omd_init_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict({}, {})


def test_omd_add_and_get():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert omd['a'] == 3


def test_omd_get_default():
    omd = OrderedMultiDict()
    assert omd.get('missing') is None
    assert omd.get('missing', 42) == 42


def test_omd_getlist_default():
    omd = OrderedMultiDict()
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', ['x']) == ['x']


def test_omd_getlist_returns_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    lst = omd.getlist('a')
    lst.append(3)
    assert omd.getlist('a') == [1, 2]


def test_omd_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.items(multi=True) == [('a', -1), ('a', 0), ('a', 1), ('a', 2)]


def test_omd_addlist_empty():
    omd = OrderedMultiDict()
    omd.addlist('a', [])
    assert 'a' not in omd


def test_omd_addlist_iterator():
    omd = OrderedMultiDict()
    omd.addlist('a', iter([1, 2, 3]))
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('a', 5) is None
    assert omd.setdefault('b', 7) == 7
    assert omd['b'] == 7


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []
    omd.add('c', 3)
    assert omd.items(multi=True) == [('c', 3)]


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    cp = omd.copy()
    assert cp == omd
    assert cp is not omd
    cp.add('a', 4)
    assert omd.getlist('a') == [1, 3]


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd.items(multi=True) == [('a', None), ('b', None)]
    omd2 = OrderedMultiDict.fromkeys(['x'], default=0)
    assert omd2['x'] == 0


# ---------------------------------------------------------------------------
# OMD update / update_extend
# ---------------------------------------------------------------------------

def test_omd_update_self_noop():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.items(multi=True) == [('a', 1)]


def test_omd_update_with_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 10), ('c', 20)])
    omd.update(other)
    assert omd.items(multi=True) == [('a', 10), ('b', 3), ('c', 20)]


def test_omd_update_with_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update({'a': 9, 'c': 3})
    assert omd['a'] == 9
    assert omd.getlist('a') == [9]
    assert omd['c'] == 3


def test_omd_update_with_iterable():
    omd = OrderedMultiDict([('a', 1)])
    omd.update([('a', 2), ('a', 3), ('b', 4)])
    assert omd.getlist('a') == [2, 3]
    assert omd['b'] == 4


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


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 2, 1, 2]


def test_omd_update_extend_with_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 2), ('a', 3)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 3]


def test_omd_update_extend_with_dict():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


# ---------------------------------------------------------------------------
# OMD setitem / delitem / pop semantics
# ---------------------------------------------------------------------------

def test_omd_setitem_overwrites_all():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd['a'] = 5
    assert omd.getlist('a') == [5]
    assert omd.items(multi=True) == [('b', 2), ('a', 5)]


def test_omd_delitem():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]


def test_omd_delitem_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        del omd['nope']


def test_omd_pop():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.pop('a') == 3
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]


def test_omd_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('missing', 'default') == 'default'


def test_omd_pop_missing_raises():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.pop('missing')


def test_omd_popall():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.popall('a') == [1, 3]
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]


def test_omd_popall_default():
    omd = OrderedMultiDict()
    assert omd.popall('missing', []) == []


def test_omd_popall_missing_raises():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.popall('missing')


def test_omd_poplast_by_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_omd_poplast_no_arg():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
    assert omd.poplast() == 3
    assert 'c' not in omd
    assert list(omd) == ['a', 'b']


def test_omd_poplast_empty_raises():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='d') == 'd'


def test_omd_poplast_missing_key():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast('missing')
    assert omd.poplast('missing', default='x') == 'x'


# ---------------------------------------------------------------------------
# OMD iteration
# ---------------------------------------------------------------------------

def test_omd_iteritems():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]


def test_omd_iterkeys():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.iterkeys()) == ['a', 'b']


def test_omd_itervalues():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]
    assert list(omd.itervalues()) == [3, 2]


def test_omd_dunder_iter():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd) == ['a', 'b']


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_reversed_simple():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_keys_values_items_methods():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_omd_views():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert set(omd.viewkeys()) == {'a', 'b'}
    assert set(omd.viewvalues()) == {1, 2}
    assert set(omd.viewitems()) == {('a', 1), ('b', 2)}


# ---------------------------------------------------------------------------
# OMD todict / sorted / sortedvalues / inverted / counts
# ---------------------------------------------------------------------------

def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    assert omd.todict(multi=True) == {'a': [1, 3], 'b': [2]}


def test_omd_todict_multi_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    d = omd.todict(multi=True)
    d['a'].append(3)
    assert omd.getlist('a') == [1, 2]


def test_omd_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert isinstance(s, OrderedMultiDict)
    assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_omd_sorted_key():
    omd = OrderedMultiDict(zip('hello', 'world'))
    s = omd.sorted(key=lambda i: i[1])
    assert s.items(multi=True) == [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]


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


def test_omd_sortedvalues_reverse():
    omd = OrderedMultiDict([('a', 1), ('a', 3), ('a', 2)])
    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('a') == [3, 2, 1]


def test_omd_inverted():
    omd = OrderedMultiDict([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]
    assert omd.inverted().inverted() == omd


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts['a'] == 2
    assert counts['b'] == 1


def test_omd_repr():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"


# ---------------------------------------------------------------------------
# OMD equality
# ---------------------------------------------------------------------------

def test_omd_eq_self():
    omd = OrderedMultiDict([('a', 1)])
    assert omd == omd


def test_omd_eq_omd():
    a = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    b = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert a == b


def test_omd_ne_omd_different_order():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('b', 2), ('a', 1)])
    assert a != b


def test_omd_eq_dict_semantics():
    omd = OrderedMultiDict([('a', 1), ('a', 3), ('b', 2)])
    assert omd == {'a': 3, 'b': 2}


def test_omd_eq_dict_semantics_no():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert not (omd == {'a': 1})


def test_omd_eq_dict_missing_key():
    omd = OrderedMultiDict([('a', 1)])
    assert not (omd == {'a': 1, 'b': 2})


def test_omd_eq_non_dict():
    omd = OrderedMultiDict([('a', 1)])
    assert not (omd == [('a', 1)])
    assert omd != [('a', 1)]


def test_omd_eq_len_mismatch():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('a', 1)])
    assert a != b


def test_omd_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'b': 2}
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


# ---------------------------------------------------------------------------
# OMD pickle
# ---------------------------------------------------------------------------

def test_omd_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    restored = pickle.loads(pickle.dumps(omd))
    assert restored == omd
    assert restored.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_omd_setstate():
    omd = OrderedMultiDict()
    omd.__setstate__([('a', 1), ('a', 2)])
    assert omd.getlist('a') == [1, 2]


def test_omd_getstate():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.__getstate__() == [('a', 1), ('b', 2), ('a', 3)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def _make_fast(data):
    return FastIterOrderedMultiDict(data)


def test_fast_basic():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3)])
    assert omd['a'] == 3
    assert omd.getlist('a') == [1, 3]
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]


def test_fast_iterkeys():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.iterkeys()) == ['a', 'b']


def test_fast_reversed():
    omd = _make_fast([('a', 1), ('b', 2), ('c', 3)])
    assert list(reversed(omd)) == ['c', 'b', 'a']


def test_fast_reversed_duplicates():
    omd = _make_fast([('a', 1), ('a', 2), ('b', 3), ('a', 4)])
    rev = list(reversed(omd))
    assert rev == ['a', 'b']


def test_fast_popall():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3)])
    assert omd.popall('a') == [1, 3]
    assert 'a' not in omd
    assert list(omd.iteritems(multi=True)) == [('b', 2)]


def test_fast_poplast():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]


def test_fast_delitem():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3), ('c', 4)])
    del omd['a']
    assert 'a' not in omd
    assert list(omd.iterkeys(multi=True)) == ['b', 'c']


def test_fast_setitem_overwrite():
    omd = _make_fast([('a', 1), ('b', 2), ('a', 3)])
    omd['a'] = 9
    assert omd.getlist('a') == [9]
    assert omd.items() == [('a', 9), ('b', 2)]


def test_fast_clear_and_reuse():
    omd = _make_fast([('a', 1), ('b', 2)])
    omd.clear()
    assert list(omd.iteritems(multi=True)) == []
    omd.add('x', 1)
    omd.add('x', 2)
    assert list(omd.iteritems(multi=True)) == [('x', 1), ('x', 2)]


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_oto_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto['b'] == 2
    assert len(oto) == 2
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'


def test_oto_inv_overwrite():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2
    assert oto.inv[1] == 'c'


def test_oto_setitem():
    oto = OneToOne()
    oto['a'] = 1
    assert oto.inv[1] == 'a'
    oto['a'] = 2
    assert 1 not in oto.inv
    assert oto.inv[2] == 'a'
    oto['b'] = 2
    assert 'a' not in oto
    assert oto.inv[2] == 'b'


def test_oto_setitem_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = [1, 2]


def test_oto_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert len(oto) == 1


def test_oto_delitem_missing():
    oto = OneToOne({'a': 1})
    with pytest.raises(KeyError):
        del oto['nope']


def test_oto_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_oto_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    cp = oto.copy()
    assert cp == oto
    assert cp is not oto
    cp['c'] = 3
    assert 3 not in oto.inv


def test_oto_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert oto.pop('missing', 'dflt') == 'dflt'
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
    assert oto['a'] == 1


def test_oto_update():
    oto = OneToOne()
    oto.update({'a': 1, 'b': 2})
    assert oto.inv[1] == 'a'
    oto.update([('c', 3)])
    assert oto['c'] == 3
    oto.update(x=9)
    assert oto['x'] == 9


def test_oto_update_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto.update({'a': [1]})


def test_oto_dupe_values_collapse():
    # With default constructor, duplicate values collapse; only one key survives
    oto = OneToOne([('a', 1), ('b', 1)])
    assert len(oto) == 1
    assert len(oto.inv) == 1


def test_oto_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'


def test_oto_unique_dupe():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_oto_unique_dupe_across_inputs():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)


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
    m2m.add('b', 2)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([2])
    assert m2m.inv[2] == frozenset(['a', 'b'])
    assert m2m.inv[1] == frozenset(['a'])


def test_m2m_get():
    m2m = ManyToMany([('a', 1)])
    assert m2m.get('a') == frozenset([1])
    assert m2m.get('missing') == frozenset()
    assert m2m.get('missing', None) is None


def test_m2m_getitem_missing():
    m2m = ManyToMany()
    with pytest.raises(KeyError):
        m2m['missing']


def test_m2m_contains_iter_len():
    m2m = ManyToMany([('a', 1), ('b', 2)])
    assert 'a' in m2m
    assert 'z' not in m2m
    assert set(iter(m2m)) == {'a', 'b'}
    assert len(m2m) == 2


def test_m2m_setitem():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset([2, 3])
    assert m2m.inv[1] == frozenset()


def test_m2m_setitem_new_key():
    m2m = ManyToMany()
    m2m['a'] = [1, 2]
    assert m2m['a'] == frozenset([1, 2])


def test_m2m_delitem():
    m2m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
    del m2m['a']
    assert 'a' not in m2m
    assert m2m.inv[2] == frozenset(['b'])
    assert 1 not in m2m.inv


def test_m2m_remove():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m.remove('a', 1)
    assert m2m['a'] == frozenset([2])
    assert 1 not in m2m.inv
    m2m.remove('a', 2)
    assert 'a' not in m2m


def test_m2m_replace():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m.replace('a', 'b')
    assert 'a' not in m2m
    assert m2m['b'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['b'])


def test_m2m_replace_missing():
    m2m = ManyToMany([('a', 1)])
    m2m.replace('missing', 'b')
    assert 'b' not in m2m
    assert m2m['a'] == frozenset([1])


def test_m2m_update_iterable():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('b', 2)])
    assert m2m['a'] == frozenset([1])
    assert m2m['b'] == frozenset([2])


def test_m2m_update_dict():
    m2m = ManyToMany()
    m2m.update({'a': 1})
    assert m2m['a'] == frozenset([1])


def test_m2m_update_m2m():
    m2m = ManyToMany([('a', 1)])
    other = ManyToMany([('a', 2), ('b', 3)])
    m2m.update(other)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([3])
    assert m2m.inv[2] == frozenset(['a'])


def test_m2m_init_with_items():
    m2m = ManyToMany([('a', 1), ('b', 1)])
    assert m2m['a'] == frozenset([1])
    assert m2m['b'] == frozenset([1])
    assert m2m.inv[1] == frozenset(['a', 'b'])


def test_m2m_iteritems():
    m2m = ManyToMany([('a', 1)])
    assert list(m2m.iteritems()) == [('a', 1)]


def test_m2m_keys():
    m2m = ManyToMany([('a', 1), ('b', 2)])
    assert set(m2m.keys()) == {'a', 'b'}


def test_m2m_eq():
    a = ManyToMany([('a', 1)])
    b = ManyToMany([('a', 1)])
    assert a == b
    c = ManyToMany([('a', 2)])
    assert a != c
    assert a != [('a', 1)]


def test_m2m_repr():
    m2m = ManyToMany([('a', 1)])
    assert repr(m2m) == "ManyToMany([('a', 1)])"


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_identity():
    d = {'a': 1, 'b': 2}
    assert subdict(d) == {'a': 1, 'b': 2}
    assert subdict(d) is not d


def test_subdict_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_and_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_missing_keys_in_keep_ignored():
    d = {'a': 1}
    assert subdict(d, keep=['a', 'z']) == {'a': 1}


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert len(fd) == 2
    assert dict(fd) == {'a': 1, 'b': 2}


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert isinstance(h, int)
    # stable hash
    assert hash(fd) == h


def test_frozendict_as_set_and_dict_key():
    fd = FrozenDict({'a': 1})
    s = {fd}
    assert fd in s
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'


def test_frozendict_equal_and_hash():
    a = FrozenDict({'a': 1})
    b = FrozenDict({'a': 1})
    assert a == b
    assert hash(a) == hash(b)


def test_frozendict_unhashable_value_raises():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_frozenhasherror_is_typeerror():
    assert issubclass(FrozenHashError, TypeError)


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
    fd2 = fd.updated({'b': 2})
    assert isinstance(fd2, FrozenDict)
    assert fd2 == {'a': 1, 'b': 2}
    assert fd == {'a': 1}


def test_frozendict_updated_kwargs():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated(b=2)
    assert fd2 == {'a': 1, 'b': 2}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'])
    assert isinstance(fd, FrozenDict)
    assert fd == {'a': None, 'b': None}
    fd2 = FrozenDict.fromkeys(['x'], value=0)
    assert fd2 == {'x': 0}


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_copy_returns_self():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    restored = pickle.loads(pickle.dumps(fd))
    assert isinstance(restored, FrozenDict)
    assert restored == fd
