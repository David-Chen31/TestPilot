# -*- coding: utf-8 -*-
import copy as copy_module
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
# Aliases and basic construction
# ---------------------------------------------------------------------------

def test_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


def test_empty_construction():
    omd = OrderedMultiDict()
    assert len(omd) == 0
    assert list(omd) == []
    assert omd.items() == []
    assert omd.items(multi=True) == []


def test_construction_from_pairs():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items() == [('a', 3), ('b', 2)]


def test_construction_from_dict():
    omd = OrderedMultiDict({'a': 1, 'b': 2})
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_construction_from_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd2 = OrderedMultiDict(omd)
    assert omd2.items(multi=True) == [('a', 1), ('a', 2)]


def test_construction_kwargs():
    omd = OrderedMultiDict(a=1, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])


# ---------------------------------------------------------------------------
# get / getlist / getitem
# ---------------------------------------------------------------------------

def test_get_returns_most_recent():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]


def test_get_default():
    omd = OrderedMultiDict()
    assert omd.get('missing') is None
    assert omd.get('missing', 'x') == 'x'


def test_getitem_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd['missing']


def test_getlist_missing_default():
    omd = OrderedMultiDict()
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', 'dflt') == 'dflt'


def test_getlist_is_copy():
    omd = OrderedMultiDict([('a', 1)])
    lst = omd.getlist('a')
    lst.append(2)
    assert omd.getlist('a') == [1]


# ---------------------------------------------------------------------------
# add / addlist
# ---------------------------------------------------------------------------

def test_add_preserves_existing():
    omd = OrderedMultiDict([('a', 1)])
    omd.add('a', 2)
    assert omd.getlist('a') == [1, 2]


def test_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.items(multi=True) == [('a', -1), ('a', 0), ('a', 1), ('a', 2)]


def test_addlist_generator():
    omd = OrderedMultiDict()
    omd.addlist('a', (i for i in range(3)))
    assert omd.getlist('a') == [0, 1, 2]


def test_addlist_empty():
    omd = OrderedMultiDict()
    omd.addlist('a', [])
    assert 'a' not in omd
    assert list(omd) == []


# ---------------------------------------------------------------------------
# clear / copy / fromkeys
# ---------------------------------------------------------------------------

def test_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []


def test_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    c = omd.copy()
    assert c is not omd
    assert c.items(multi=True) == omd.items(multi=True)
    c.add('a', 4)
    assert omd.getlist('a') == [1, 3]


def test_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd['a'] is None
    assert omd['b'] is None


def test_fromkeys_default():
    omd = OrderedMultiDict.fromkeys(['a', 'b'], 0)
    assert omd['a'] == 0
    assert omd['b'] == 0


# ---------------------------------------------------------------------------
# update / update_extend
# ---------------------------------------------------------------------------

def test_update_overwrites():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    omd.update([('a', 10)])
    assert omd.getlist('a') == [10]
    assert omd['b'] == 3


def test_update_from_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 7), ('a', 8)])
    omd.update(other)
    assert omd.getlist('a') == [7, 8]
    assert omd['b'] == 3


def test_update_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.items(multi=True) == [('a', 1)]


def test_update_with_dict():
    omd = OrderedMultiDict()
    omd.update({'x': 1})
    assert omd['x'] == 1


def test_update_with_kwargs():
    omd = OrderedMultiDict()
    omd.update({'x': 1}, y=2)
    assert omd['x'] == 1
    assert omd['y'] == 2


def test_update_extend_keeps_existing():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('b', 3)])
    assert omd.getlist('a') == [1, 2]
    assert omd['b'] == 3


def test_update_extend_from_omd():
    omd = OrderedMultiDict([('a', 1)])
    other = OrderedMultiDict([('a', 2), ('a', 3)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 3]


def test_update_extend_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_update_extend_kwargs():
    omd = OrderedMultiDict()
    omd.update_extend([('a', 1)], b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


# ---------------------------------------------------------------------------
# setdefault
# ---------------------------------------------------------------------------

def test_setdefault_missing():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert 'a' in omd


def test_setdefault_existing():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.setdefault('a', 99) == 1


def test_setdefault_with_default():
    omd = OrderedMultiDict()
    assert omd.setdefault('a', 5) == 5


# ---------------------------------------------------------------------------
# __setitem__ / __delitem__
# ---------------------------------------------------------------------------

def test_setitem_overwrites_all():
    omd = OrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    omd['a'] = 3
    assert omd.getlist('a') == [3]


def test_delitem_removes_all():
    omd = OrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    del omd['a']
    assert 'a' not in omd
    assert list(omd) == []


def test_delitem_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        del omd['x']


# ---------------------------------------------------------------------------
# equality
# ---------------------------------------------------------------------------

def test_eq_self():
    omd = OrderedMultiDict([('a', 1)])
    assert omd == omd


def test_eq_same_multidict():
    a = OrderedMultiDict([('a', 1), ('a', 2)])
    b = OrderedMultiDict([('a', 1), ('a', 2)])
    assert a == b


def test_eq_diff_order_multidict():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('b', 2), ('a', 1)])
    assert a != b


def test_eq_with_dict():
    a = OrderedMultiDict([('a', 1), ('a', 2)])
    assert a == {'a': 2}


def test_eq_with_dict_mismatch():
    a = OrderedMultiDict([('a', 1)])
    assert a != {'a': 2}


def test_ne():
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 2)])
    assert a != b
    assert not (a != b)


def test_eq_non_mapping():
    a = OrderedMultiDict([('a', 1)])
    assert a != 5


def test_eq_missing_key_against_dict():
    a = OrderedMultiDict([('a', 1)])
    assert a != {'b': 1}


# ---------------------------------------------------------------------------
# pop / popall / poplast
# ---------------------------------------------------------------------------

def test_pop_returns_most_recent():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.pop('a') == 3
    assert 'a' not in omd
    assert omd['b'] == 2


def test_pop_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.pop('x')


def test_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('x', 'dflt') == 'dflt'


def test_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd


def test_popall_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.popall('x')


def test_popall_default():
    omd = OrderedMultiDict()
    assert omd.popall('x', 'dflt') == 'dflt'


def test_poplast_with_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]


def test_poplast_removes_key():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_poplast_no_key():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert omd.poplast() == 2
    assert omd.poplast() == 1
    with pytest.raises(KeyError):
        omd.poplast()


def test_poplast_empty_with_default():
    omd = OrderedMultiDict()
    assert omd.poplast(default='d') == 'd'


def test_poplast_missing_key():
    omd = OrderedMultiDict([('a', 1)])
    with pytest.raises(KeyError):
        omd.poplast('x')


def test_poplast_missing_key_default():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.poplast('x', 'dflt') == 'dflt'


# ---------------------------------------------------------------------------
# iteration
# ---------------------------------------------------------------------------

def test_iterkeys():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys()) == ['a', 'b']
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']


def test_itervalues():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.itervalues()) == [3, 2]
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]


def test_iteritems():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]


def test_iter():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd) == ['a', 'b']


def test_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_keys_values_items_lists():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


# ---------------------------------------------------------------------------
# todict
# ---------------------------------------------------------------------------

def test_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}


def test_todict_multi_is_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 3)])
    d = omd.todict(multi=True)
    assert d == {'a': [1, 3]}
    d['a'].append(5)
    assert omd.getlist('a') == [1, 3]


# ---------------------------------------------------------------------------
# sorted / sortedvalues / inverted / counts
# ---------------------------------------------------------------------------

def test_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_sorted_key():
    omd = OrderedMultiDict(zip('hello', 'world'))
    s = omd.sorted(key=lambda i: i[1])
    assert s.items(multi=True) == [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]


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


def test_sortedvalues_reverse():
    omd = OrderedMultiDict()
    omd.addlist('even', [4, 2])
    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('even') == [4, 2]


def test_inverted():
    omd = OMD([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]


def test_inverted_twice():
    omd = OMD([(0, 2), (1, 2)])
    assert omd.inverted().inverted().items(multi=True) == [(0, 2), (1, 2)]


def test_counts():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts['a'] == 2
    assert counts['b'] == 1


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2)])"


def test_repr_empty():
    assert repr(OrderedMultiDict()) == "OrderedMultiDict([])"


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def test_views():
    from collections.abc import KeysView, ValuesView, ItemsView
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert isinstance(omd.viewkeys(), KeysView)
    assert isinstance(omd.viewvalues(), ValuesView)
    assert isinstance(omd.viewitems(), ItemsView)
    assert set(omd.viewkeys()) == {'a', 'b'}


# ---------------------------------------------------------------------------
# pickling
# ---------------------------------------------------------------------------

def test_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    omd2 = pickle.loads(data)
    assert omd2.items(multi=True) == omd.items(multi=True)


def test_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= [('a', 2), ('b', 3)]
    assert omd.getlist('a') == [2]
    assert omd['b'] == 3


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

def test_fastiter_basic():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert list(omd.iterkeys()) == ['a', 'b']
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']


def test_fastiter_reversed():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_fastiter_remove():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.pop('a') == 3
    assert omd.items(multi=True) == [('b', 2)]


def test_fastiter_remove_all():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.popall('a') == [1, 3]
    assert omd.items(multi=True) == [('b', 2)]


def test_fastiter_poplast():
    omd = FastIterOrderedMultiDict([('a', 1), ('a', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1, 2]


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_onetoone_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2


def test_onetoone_overwrite_reverse():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_onetoone_setitem_removes_old_key_value():
    oto = OneToOne({'a': 1})
    oto['a'] = 2
    assert oto['a'] == 2
    assert 1 not in oto.inv
    assert oto.inv[2] == 'a'


def test_onetoone_setitem_value_reassigned():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['c'] = 1
    assert 'a' not in oto
    assert oto['c'] == 1
    assert oto.inv[1] == 'c'


def test_onetoone_delitem():
    oto = OneToOne({'a': 1})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_onetoone_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_copy():
    oto = OneToOne({'a': 1})
    c = oto.copy()
    assert c == oto
    assert c.inv == oto.inv


def test_onetoone_pop():
    oto = OneToOne({'a': 1})
    assert oto.pop('a') == 1
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_onetoone_pop_default():
    oto = OneToOne()
    assert oto.pop('a', 'dflt') == 'dflt'


def test_onetoone_pop_missing():
    oto = OneToOne()
    with pytest.raises(KeyError):
        oto.pop('a')


def test_onetoone_popitem():
    oto = OneToOne({'a': 1})
    k, v = oto.popitem()
    assert (k, v) == ('a', 1)
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_onetoone_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.setdefault('a', 2) == 1
    assert oto.inv[1] == 'a'


def test_onetoone_update():
    oto = OneToOne()
    oto.update({'a': 1})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'


def test_onetoone_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1)])
    assert oto['a'] == 1


def test_onetoone_update_kwargs():
    oto = OneToOne()
    oto.update({}, b=2)
    assert oto['b'] == 2


def test_onetoone_dupe_values_collapse():
    # without unique, duplicate values collapse (last wins)
    oto = OneToOne({'a': 1, 'b': 1})
    assert len(oto) == 1


def test_onetoone_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto['b'] == 2


def test_onetoone_unique_raises():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_onetoone_unique_across_inputs():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)


def test_onetoone_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = [1, 2]


def test_onetoone_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto).startswith("OneToOne(")


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_manytomany_basic():
    m2m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([2])
    assert m2m.inv[2] == frozenset(['a', 'b'])


def test_manytomany_get_default():
    m2m = ManyToMany()
    assert m2m.get('x') == frozenset()
    assert m2m.get('x', 'dflt') == 'dflt'


def test_manytomany_getitem_missing():
    m2m = ManyToMany()
    with pytest.raises(KeyError):
        m2m['x']


def test_manytomany_setitem():
    m2m = ManyToMany()
    m2m['a'] = [1, 2]
    assert m2m['a'] == frozenset([1, 2])
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset([2, 3])
    assert 1 not in m2m.inv


def test_manytomany_delitem():
    m2m = ManyToMany([('a', 1)])
    del m2m['a']
    assert 'a' not in m2m
    assert 1 not in m2m.inv


def test_manytomany_add_remove():
    m2m = ManyToMany()
    m2m.add('a', 1)
    assert m2m['a'] == frozenset([1])
    m2m.remove('a', 1)
    assert 'a' not in m2m
    assert 1 not in m2m.inv


def test_manytomany_update_pairs():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('b', 2)])
    assert m2m['a'] == frozenset([1])
    assert m2m['b'] == frozenset([2])


def test_manytomany_update_manytomany():
    a = ManyToMany([('x', 1)])
    b = ManyToMany([('x', 2), ('y', 3)])
    a.update(b)
    assert a['x'] == frozenset([1, 2])
    assert a['y'] == frozenset([3])
    assert a.inv[2] == frozenset(['x'])


def test_manytomany_update_dict():
    m2m = ManyToMany()
    m2m.update({'a': [1, 2]})
    assert m2m['a'] == frozenset([1, 2])


def test_manytomany_replace():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m.replace('a', 'b')
    assert 'a' not in m2m
    assert m2m['b'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['b'])


def test_manytomany_replace_missing():
    m2m = ManyToMany([('a', 1)])
    m2m.replace('missing', 'b')
    assert 'b' not in m2m


def test_manytomany_contains_iter_len():
    m2m = ManyToMany([('a', 1), ('b', 2)])
    assert 'a' in m2m
    assert 'z' not in m2m
    assert set(iter(m2m)) == {'a', 'b'}
    assert len(m2m) == 2


def test_manytomany_eq():
    a = ManyToMany([('a', 1)])
    b = ManyToMany([('a', 1)])
    assert a == b
    c = ManyToMany([('a', 2)])
    assert a != c


def test_manytomany_keys_iteritems():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    assert set(m2m.keys()) == {'a'}
    assert set(m2m.iteritems()) == {('a', 1), ('a', 2)}


def test_manytomany_repr():
    m2m = ManyToMany([('a', 1)])
    assert repr(m2m).startswith("ManyToMany(")


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
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_subdict_missing_keep_keys():
    assert subdict({'a': 1}, keep=['a', 'z']) == {'a': 1}


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1})
    assert fd['a'] == 1
    assert dict(fd) == {'a': 1}
    assert isinstance(fd, dict)


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert isinstance(h, int)
    assert hash(fd) == h


def test_frozendict_as_set_entry():
    fd = FrozenDict({'a': 1})
    s = {fd}
    assert fd in s


def test_frozendict_eq():
    assert FrozenDict({'a': 1}) == FrozenDict({'a': 1})
    assert FrozenDict({'a': 1}) == {'a': 1}


def test_frozendict_unhashable_value_raises():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)


def test_frozendict_frozen_hash_error_is_typeerror():
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


def test_frozendict_immutable_setdefault():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.setdefault('b', 2)


def test_frozendict_immutable_ior():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd |= {'b': 2}


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert fd2 == {'a': 1, 'b': 2}
    assert isinstance(fd2, FrozenDict)
    assert fd == {'a': 1}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 0)
    assert fd == {'a': 0, 'b': 0}
    assert isinstance(fd, FrozenDict)


def test_frozendict_copy_returns_self():
    fd = FrozenDict({'a': 1})
    assert copy_module.copy(fd) is fd


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1})
    fd2 = pickle.loads(pickle.dumps(fd))
    assert fd2 == fd
    assert isinstance(fd2, FrozenDict)
