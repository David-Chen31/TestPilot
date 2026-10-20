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
    FrozenHashError,
    FrozenDict,
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
    assert omd.keys() == []
    assert omd.values() == []
    assert omd.items(multi=True) == []
    assert omd.todict() == {}
    assert omd.todict(multi=True) == {}
    assert repr(omd) == 'OrderedMultiDict([])'


def test_omd_basic_setting_getting():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd['a'] == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.get('b') == 2
    assert omd.get('missing') is None
    assert omd.get('missing', 'x') == 'x'
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', ['d']) == ['d']


def test_omd_constructor():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]
    assert omd['a'] == 3

    omd2 = OrderedMultiDict(a=1, b=2)
    assert omd2['a'] == 1
    assert omd2['b'] == 2


def test_omd_constructor_too_many_args():
    with pytest.raises(TypeError) as exc_info:
        OrderedMultiDict([], [])
    msg = str(exc_info.value)
    assert msg == 'OrderedMultiDict expected at most 1 argument, got 2'


def test_omd_constructor_too_many_args_subclass():
    class MyOMD(OrderedMultiDict):
        pass
    with pytest.raises(TypeError) as exc_info:
        MyOMD([], [], [])
    assert str(exc_info.value) == 'MyOMD expected at most 1 argument, got 3'


def test_omd_setitem_removes_old_values():
    omd = OrderedMultiDict()
    omd.add('a', 1)
    omd.add('a', 2)
    omd['a'] = 3
    assert omd.getlist('a') == [3]


def test_omd_setitem_new_key():
    omd = OrderedMultiDict()
    omd['z'] = 99
    assert omd.getlist('z') == [99]
    assert omd.items(multi=True) == [('z', 99)]


def test_omd_add_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]
    # empty addlist is a no-op
    omd.addlist('b', [])
    assert 'b' not in omd
    # iterator support
    omd.addlist('c', iter([1, 2]))
    assert omd.getlist('c') == [1, 2]


def test_omd_add_addlist_values_preserved():
    # ensure the actual values are inserted, not None
    omd = OrderedMultiDict()
    omd.addlist('a', ['x', 'y', 'z'])
    assert omd.getlist('a') == ['x', 'y', 'z']
    assert omd.items(multi=True) == [('a', 'x'), ('a', 'y'), ('a', 'z')]


def test_omd_repr_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"


def test_omd_reversed():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_iterkeys_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iterkeys()) == ['a', 'b']
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']


def test_omd_iteritems_values():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.itervalues()) == [3, 2]
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.values() == [3, 2]


def test_omd_values_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.values(multi=False) == [3, 2]
    assert omd.values() == [3, 2]


def test_omd_pop():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.pop('a') == 3
    assert 'a' not in omd
    assert omd.pop('missing', 'default') == 'default'
    with pytest.raises(KeyError) as exc_info:
        omd.pop('missing')
    assert exc_info.value.args == ('missing',)


def test_omd_popall():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.popall('a') == [1, 3]
    assert 'a' not in omd
    assert omd.popall('missing', 'default') == 'default'
    with pytest.raises(KeyError) as exc_info:
        omd.popall('missing')
    assert exc_info.value.args == ('missing',)


def test_omd_popall_default_not_consuming_k():
    omd = OrderedMultiDict([('a', 1)])
    # default for an existing key returns the list; default unused
    assert omd.popall('a', 'd') == [1]
    assert 'a' not in omd


def test_omd_poplast():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd
    assert omd.poplast('missing', 'd') == 'd'
    with pytest.raises(KeyError) as exc_info:
        omd.poplast('missing')
    assert exc_info.value.args == ('missing',)

    # no-arg poplast returns most recently inserted key's last value
    omd2 = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd2.poplast() == 3
    assert omd2.poplast() == 2
    assert omd2.poplast() == 1
    assert len(omd2) == 0
    with pytest.raises(KeyError) as exc_info:
        omd2.poplast()
    msg = str(exc_info.value)
    assert "empty" in msg
    assert "OrderedMultiDict" in msg
    assert omd2.poplast(default='empty') == 'empty'


def test_omd_poplast_empty_message_for_subclass():
    class MyOMD(OrderedMultiDict):
        pass
    omd = MyOMD()
    with pytest.raises(KeyError) as exc_info:
        omd.poplast()
    msg = str(exc_info.value)
    assert "MyOMD" in msg
    assert "empty" in msg


def test_omd_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []


def test_omd_clear_docstring():
    assert OrderedMultiDict.clear.__doc__ == "Empty the dictionary."


def test_omd_copy_docstring():
    assert OrderedMultiDict.copy.__doc__ == "Return a shallow copy of the dictionary."


def test_omd_setdefault():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('a', 5) is None  # already present
    assert omd.setdefault('b', 5) == 5
    assert omd['b'] == 5


def test_omd_copy():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    c = omd.copy()
    assert isinstance(c, OrderedMultiDict)
    assert c == omd
    assert c.items(multi=True) == omd.items(multi=True)
    c.add('c', 4)
    assert 'c' not in omd


def test_omd_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'], 0)
    assert omd.items(multi=True) == [('a', 0), ('b', 0)]
    omd2 = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd2['a'] is None
    assert omd2['b'] is None


def test_omd_update_self_noop():
    omd = OrderedMultiDict([('a', 1)])
    omd.update(omd)
    assert omd.items(multi=True) == [('a', 1)]


def test_omd_update_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update({'a': 10, 'c': 3})
    assert omd.getlist('a') == [10]
    assert omd['b'] == 2
    assert omd['c'] == 3


def test_omd_update_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 10), ('a', 20), ('c', 30)])
    omd.update(other)
    assert omd.getlist('a') == [10, 20]
    assert omd.getlist('b') == [3]
    assert omd.getlist('c') == [30]


def test_omd_update_iterable():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update([('a', 10), ('a', 11), ('c', 3)])
    # once 'a' is deleted on first occurrence, subsequent items are just added
    assert omd.getlist('a') == [10, 11]
    assert omd.getlist('c') == [3]
    assert omd.getlist('b') == [2]


def test_omd_update_kwargs():
    omd = OrderedMultiDict([('a', 1)])
    omd.update({'b': 2}, c=3)
    assert omd['a'] == 1
    assert omd['b'] == 2
    assert omd['c'] == 3


def test_omd_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend({'a': 2, 'b': 3})
    assert omd.getlist('a') == [1, 2]
    assert omd.getlist('b') == [3]


def test_omd_update_extend_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    other = OrderedMultiDict([('a', 10), ('a', 20)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 10, 20]


def test_omd_update_extend_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_omd_update_extend_iterable():
    omd = OrderedMultiDict()
    omd.update_extend([('a', 1), ('a', 2)])
    assert omd.getlist('a') == [1, 2]


def test_omd_delete():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd.getlist('b') == [2]
    assert omd.items(multi=True) == [('b', 2)]


def test_omd_equality_omd():
    a = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    b = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    c = OrderedMultiDict([('a', 3), ('b', 2), ('a', 1)])
    assert a == b
    assert a != c
    assert a == a
    assert not (a != a)


def test_omd_equality_omd_trailing_self():
    # len differs -> False via length check
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 1), ('b', 2)])
    assert a != b
    assert b != a


def test_omd_equality_dict():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    assert a == {'a': 1, 'b': 2}
    assert a != {'a': 1, 'b': 2, 'c': 4}
    assert a != {'a': 1}
    assert a != [('a', 1), ('b', 2)]
    # non-sized object
    assert a != 1


def test_omd_equality_dict_differing_value():
    a = OrderedMultiDict([('a', 1)])
    assert not (a == {'a': 2})
    assert a != {'a': 2}


def test_omd_equality_dict_missing_key():
    a = OrderedMultiDict([('a', 1)])
    assert not (a == {'a': 1, 'b': 5})
    assert not (a == {'b': 1})


def test_omd_equality_multi_diff():
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 1), ('a', 2)])
    assert a != b


def test_omd_equality_omd_different_order():
    a = OrderedMultiDict([('a', 1), ('b', 2)])
    b = OrderedMultiDict([('b', 2), ('a', 1)])
    assert a != b


def test_omd_ior():
    a = OrderedMultiDict([('a', 1)])
    a |= {'a': 2, 'b': 3}
    assert a['a'] == 2
    assert a['b'] == 3


def test_omd_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    multi = omd.todict(multi=True)
    assert multi == {'a': [1, 3], 'b': [2]}
    # returned lists are copies
    multi['a'].append(99)
    assert omd.getlist('a') == [1, 3]


def test_omd_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert isinstance(s, OrderedMultiDict)
    assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_omd_sorted_key():
    omd = OrderedMultiDict([('b', 2), ('a', 1), ('c', 3)])
    s = omd.sorted(key=lambda i: i[0])
    assert s.items(multi=True) == [('a', 1), ('b', 2), ('c', 3)]


def test_omd_sorted_key_reverse():
    omd = OrderedMultiDict([('b', 2), ('a', 1), ('c', 3)])
    s = omd.sorted(key=lambda i: i[0], reverse=True)
    assert s.items(multi=True) == [('c', 3), ('b', 2), ('a', 1)]


def test_omd_sorted_multi_preserved():
    omd = OrderedMultiDict([('b', 2), ('a', 1), ('b', 3)])
    s = omd.sorted(key=lambda i: i[0])
    assert s.items(multi=True) == [('a', 1), ('b', 2), ('b', 3)]


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
    omd.addlist('a', [3, 1, 2])
    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('a') == [3, 2, 1]


def test_omd_sortedvalues_key():
    omd = OrderedMultiDict()
    omd.addlist('a', [1, 3, 2])
    somd = omd.sortedvalues(key=lambda v: -v)
    assert somd.getlist('a') == [3, 2, 1]


def test_omd_inverted():
    omd = OrderedMultiDict([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted() == omd


def test_omd_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert isinstance(counts, OrderedMultiDict)
    assert counts['a'] == 2
    assert counts['b'] == 1


def test_omd_views():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    vk = omd.viewkeys()
    vv = omd.viewvalues()
    vi = omd.viewitems()
    assert set(vk) == {'a', 'b'}
    assert set(vv) == {1, 2}
    assert set(vi) == {('a', 1), ('b', 2)}


def test_omd_view_docstrings():
    assert (OrderedMultiDict.viewkeys.__doc__
            == "OMD.viewkeys() -> a set-like object providing a view on OMD's keys")
    assert (OrderedMultiDict.viewvalues.__doc__
            == "OMD.viewvalues() -> an object providing a view on OMD's values")
    assert (OrderedMultiDict.viewitems.__doc__
            == "OMD.viewitems() -> a set-like object providing a view on OMD's items")


def test_omd_view_types():
    from collections.abc import KeysView, ValuesView, ItemsView
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert isinstance(omd.viewkeys(), KeysView)
    assert isinstance(omd.viewvalues(), ValuesView)
    assert isinstance(omd.viewitems(), ItemsView)


def test_omd_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('b', 2), ('a', 3)]
    new = OrderedMultiDict([('x', 9)])
    new.__setstate__(state)
    assert new.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_omd_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    data = pickle.dumps(omd)
    restored = pickle.loads(data)
    assert isinstance(restored, OrderedMultiDict)
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

def test_fomd_basics():
    omd = FastIterOrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd['a'] == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    # FastIter iterates via the skip-list (SNEXT) link, so the single-value
    # view yields the first-inserted item per key rather than the last.
    assert omd.items() == [('a', 1), ('b', 2)]


def test_fomd_remove():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3), ('c', 4)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.keys() == ['a', 'b', 'c']
    del omd['a']
    assert 'a' not in omd
    assert omd.keys() == ['b', 'c']


def test_fomd_popall():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.popall('a') == [1, 3]
    assert omd.keys(multi=True) == ['b']


def test_fomd_reversed():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3), ('c', 4)])
    rev = list(reversed(omd))
    # every unique key appears; order not strictly specified beyond set
    assert set(rev) == {'a', 'b', 'c'}


def test_fomd_reversed_simple():
    omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
    assert list(reversed(omd)) == ['c', 'b', 'a']


def test_fomd_clear_and_reuse():
    omd = FastIterOrderedMultiDict([('a', 1)])
    omd.clear()
    assert len(omd) == 0
    omd['x'] = 10
    assert omd['x'] == 10
    assert omd.keys() == ['x']


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

def test_oto_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2
    assert len(oto.inv) == 2


def test_oto_inv_overwrite():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_oto_setitem_overwrites_value():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['c'] = 1  # 1 already maps to 'a'
    assert 'a' not in oto
    assert oto['c'] == 1
    assert oto.inv[1] == 'c'


def test_oto_setitem_changes_key_value():
    oto = OneToOne({'a': 1})
    oto['a'] = 5
    assert oto['a'] == 5
    assert 1 not in oto.inv
    assert oto.inv[5] == 'a'


def test_oto_setitem_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = [1, 2]


def test_oto_delitem():
    oto = OneToOne({'a': 1})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_oto_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_oto_copy():
    oto = OneToOne({'a': 1})
    c = oto.copy()
    assert isinstance(c, OneToOne)
    assert c == oto
    assert c.inv == oto.inv
    c['b'] = 2
    assert 'b' not in oto


def test_oto_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 'a' not in oto
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
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert oto.setdefault('a', 99) == 1


def test_oto_update_dict():
    oto = OneToOne({'a': 1})
    oto.update({'b': 2, 'c': 3})
    assert oto['b'] == 2
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
    assert oto.inv[2] == 'b'


def test_oto_update_unhashable():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto.update({'a': [1, 2]})


def test_oto_duplicate_values():
    # non-unique values: last one wins (behavior of default constructor)
    oto = OneToOne({'a': 1, 'b': 1})
    assert len(oto) == 1
    assert oto.inv[1] in ('a', 'b')


def test_oto_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_oto_unique_duplicate():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_oto_unique_duplicate_message():
    with pytest.raises(ValueError) as exc_info:
        OneToOne.unique({'a': 1, 'b': 1})
    msg = str(exc_info.value)
    assert msg.startswith(
        'expected unique values, got multiple keys for the following values:'
    )
    assert "{1: ['a', 'b']}" in msg


def test_oto_unique_across_inputs():
    a_dict = {'a': 2}
    with pytest.raises(ValueError):
        OneToOne.unique(a_dict, b=2)


def test_oto_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto) == "OneToOne({'a': 1})"


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

def test_m2m_empty():
    m2m = ManyToMany()
    assert len(m2m) == 0
    assert list(m2m.iteritems()) == []
    assert m2m.get('missing') == frozenset()


def test_m2m_add_and_get():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.add('b', 2)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([2])
    assert m2m.inv[2] == frozenset(['a', 'b'])
    assert len(m2m) == 2
    assert 'a' in m2m


def test_m2m_get_default():
    m2m = ManyToMany()
    assert m2m.get('x') == frozenset()
    assert m2m.get('x', 'def') == 'def'


def test_m2m_setitem():
    m2m = ManyToMany()
    m2m['a'] = [1, 2, 3]
    assert m2m['a'] == frozenset([1, 2, 3])
    m2m['a'] = [2, 3]  # removes 1
    assert m2m['a'] == frozenset([2, 3])
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
    m2m.remove('a', 1)
    assert 'a' not in m2m
    assert 1 not in m2m.inv


def test_m2m_replace():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.replace('a', 'b')
    assert 'a' not in m2m
    assert m2m['b'] == frozenset([1, 2])
    assert m2m.inv[1] == frozenset(['b'])


def test_m2m_replace_missing():
    m2m = ManyToMany()
    m2m.replace('nope', 'x')  # no-op
    assert m2m == ManyToMany()


def test_m2m_update_iterable():
    m2m = ManyToMany()
    m2m.update([('a', 1), ('a', 2), ('b', 2)])
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([2])


def test_m2m_update_mapping():
    # Mapping input is consumed via add(k, iterable[k]) directly, so the
    # per-key value must itself be hashable (not a list of values).
    m2m = ManyToMany()
    with pytest.raises(TypeError):
        m2m.update({'a': [1, 2]})

    m2m2 = ManyToMany()
    m2m2.update({'a': 1, 'b': 2})
    assert m2m2['a'] == frozenset([1])
    assert m2m2['b'] == frozenset([2])


def test_m2m_update_other():
    m2m = ManyToMany()
    m2m.add('a', 1)
    other = ManyToMany()
    other.add('a', 2)
    other.add('b', 3)
    m2m.update(other)
    assert m2m['a'] == frozenset([1, 2])
    assert m2m['b'] == frozenset([3])
    assert m2m.inv[2] == frozenset(['a'])


def test_m2m_iter_and_eq():
    m2m = ManyToMany([('a', 1), ('b', 2)])
    assert set(m2m) == {'a', 'b'}
    assert set(m2m.keys()) == {'a', 'b'}
    assert set(m2m.iteritems()) == {('a', 1), ('b', 2)}
    other = ManyToMany([('a', 1), ('b', 2)])
    assert m2m == other
    assert m2m != ManyToMany([('a', 1)])


def test_m2m_repr():
    m2m = ManyToMany([('a', 1)])
    assert repr(m2m) == "ManyToMany([('a', 1)])"


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

def test_subdict_default():
    d = {'a': 1, 'b': 2}
    assert subdict(d) == {'a': 1, 'b': 2}


def test_subdict_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_keep_missing():
    d = {'a': 1}
    assert subdict(d, keep=['a', 'z']) == {'a': 1}


def test_subdict_both():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d, keep=['a', 'b', 'c'], drop=['b']) == {'a': 1, 'c': 3}


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert len(fd) == 2
    assert repr(fd) == "FrozenDict({'a': 1, 'b': 2})"


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    assert hash(fd) == hash(fd)
    s = {fd, FrozenDict({'a': 1})}
    assert len(s) == 1
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'


def test_frozendict_unhashable_content():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(TypeError):
        hash(fd)
    # repeated hash raises again
    with pytest.raises(TypeError):
        hash(fd)


def test_frozendict_frozen_hash_error_type():
    fd = FrozenDict({'a': [1, 2]})
    try:
        hash(fd)
    except FrozenHashError:
        pass
    else:
        pytest.fail("expected FrozenHashError")


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
    fd3 = fd.updated([('a', 10)])
    assert fd3['a'] == 10


def test_frozendict_updated_kwargs():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated(b=2)
    assert fd2 == {'a': 1, 'b': 2}


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 0)
    assert isinstance(fd, FrozenDict)
    assert fd == {'a': 0, 'b': 0}
    fd2 = FrozenDict.fromkeys(['a'])
    assert fd2['a'] is None


def test_frozendict_copy():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    restored = pickle.loads(pickle.dumps(fd))
    assert isinstance(restored, FrozenDict)
    assert restored == fd
