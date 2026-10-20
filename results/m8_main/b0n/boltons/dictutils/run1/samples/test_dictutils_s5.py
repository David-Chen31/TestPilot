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
# OrderedMultiDict
# ---------------------------------------------------------------------------

class TestOrderedMultiDictBasic:
    def test_aliases(self):
        assert OMD is OrderedMultiDict
        assert MultiDict is OrderedMultiDict

    def test_empty(self):
        omd = OrderedMultiDict()
        assert len(omd) == 0
        assert list(omd) == []
        assert omd.items() == []
        assert omd.keys() == []
        assert omd.values() == []

    def test_setitem_getitem(self):
        omd = OrderedMultiDict()
        omd['a'] = 1
        omd['b'] = 2
        assert omd['a'] == 1
        assert omd['b'] == 2
        assert len(omd) == 2
        assert list(omd) == ['a', 'b']

    def test_setitem_overwrite_replaces(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.getlist('a') == [1, 3]
        assert omd['a'] == 3
        omd['a'] = 99
        assert omd.getlist('a') == [99]
        assert omd['a'] == 99
        assert omd.todict(multi=True) == {'a': [99], 'b': [2]}

    def test_add_preserves(self):
        omd = OrderedMultiDict()
        omd['a'] = 1
        omd.add('a', 2)
        omd.add('a', 3)
        assert omd.getlist('a') == [1, 2, 3]
        assert omd['a'] == 3
        assert omd.get('a') == 3

    def test_addlist(self):
        omd = OrderedMultiDict([('a', -1)])
        omd.addlist('a', range(3))
        assert omd.getlist('a') == [-1, 0, 1, 2]

    def test_addlist_with_iterator(self):
        omd = OrderedMultiDict()
        omd.addlist('k', iter([1, 2, 3]))
        assert omd.getlist('k') == [1, 2, 3]

    def test_addlist_empty(self):
        omd = OrderedMultiDict()
        omd.addlist('k', [])
        assert 'k' not in omd
        assert omd.getlist('k') == []

    def test_get_default(self):
        omd = OrderedMultiDict()
        assert omd.get('x') is None
        assert omd.get('x', 'dflt') == 'dflt'
        omd['x'] = 1
        omd.add('x', 2)
        assert omd.get('x') == 2

    def test_getitem_missing(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd['nope']

    def test_getlist_default(self):
        omd = OrderedMultiDict()
        assert omd.getlist('x') == []
        assert omd.getlist('x', 'dflt') == 'dflt'
        omd.addlist('x', [1, 2])
        lst = omd.getlist('x')
        assert lst == [1, 2]
        # must be a copy
        lst.append(3)
        assert omd.getlist('x') == [1, 2]

    def test_clear(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2)])
        omd.clear()
        assert len(omd) == 0
        assert list(omd) == []
        omd.add('c', 3)
        assert list(omd) == ['c']
        assert omd['c'] == 3

    def test_setdefault(self):
        omd = OrderedMultiDict()
        assert omd.setdefault('a') is None
        assert omd['a'] is None
        assert omd.setdefault('a', 5) is None  # already present
        omd2 = OrderedMultiDict()
        assert omd2.setdefault('b', 7) == 7
        assert omd2['b'] == 7

    def test_copy(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        cp = omd.copy()
        assert isinstance(cp, OrderedMultiDict)
        assert cp == omd
        cp.add('a', 4)
        assert omd.getlist('a') == [1, 3]
        assert cp.getlist('a') == [1, 3, 4]

    def test_fromkeys(self):
        omd = OrderedMultiDict.fromkeys(['a', 'b'])
        assert omd.todict() == {'a': None, 'b': None}
        omd2 = OrderedMultiDict.fromkeys('xyz', 0)
        assert omd2.todict() == {'x': 0, 'y': 0, 'z': 0}


class TestOrderedMultiDictInit:
    def test_init_from_pairs(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        # update_extend: keeps all values
        assert omd.getlist('a') == [1, 3]
        assert omd.getlist('b') == [2]

    def test_init_kwargs(self):
        omd = OrderedMultiDict(a=1, b=2)
        assert omd.todict() == {'a': 1, 'b': 2}

    def test_init_too_many_args(self):
        with pytest.raises(TypeError):
            OrderedMultiDict([], [])

    def test_init_from_dict(self):
        omd = OrderedMultiDict({'a': 1, 'b': 2})
        assert omd.todict() == {'a': 1, 'b': 2}


class TestOrderedMultiDictUpdate:
    def test_update_from_dict(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        omd.update({'a': 10, 'b': 20})
        assert omd.getlist('a') == [10]
        assert omd.getlist('b') == [20]

    def test_update_from_list(self):
        omd = OrderedMultiDict()
        omd.update([('a', 1), ('b', 2), ('a', 3)])
        # update: for repeated keys within same input, first insert then add
        assert omd.getlist('a') == [1, 3]
        assert omd.getlist('b') == [2]

    def test_update_self_is_noop(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        omd.update(omd)
        assert omd.getlist('a') == [1, 2]

    def test_update_from_omd(self):
        a = OrderedMultiDict([('a', 1), ('a', 2)])
        b = OrderedMultiDict([('a', 5), ('b', 6)])
        a.update(b)
        assert a.getlist('a') == [5]
        assert a.getlist('b') == [6]

    def test_update_kwargs(self):
        omd = OrderedMultiDict()
        omd.update({'a': 1}, b=2)
        assert omd.todict() == {'a': 1, 'b': 2}

    def test_update_extend(self):
        omd = OrderedMultiDict([('a', 1)])
        omd.update_extend([('a', 2), ('b', 3)])
        assert omd.getlist('a') == [1, 2]
        assert omd.getlist('b') == [3]

    def test_update_extend_self(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2)])
        omd.update_extend(omd)
        # items() returns single per key
        assert omd.getlist('a') == [1, 1]
        assert omd.getlist('b') == [2, 2]

    def test_update_extend_from_omd(self):
        a = OrderedMultiDict([('a', 1), ('a', 2)])
        b = OrderedMultiDict()
        b.update_extend(a)
        assert b.getlist('a') == [1, 2]

    def test_ior(self):
        omd = OrderedMultiDict([('a', 1)])
        omd |= [('b', 2)]
        assert omd.todict() == {'a': 1, 'b': 2}


class TestOrderedMultiDictDelete:
    def test_delitem(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        del omd['a']
        assert 'a' not in omd
        assert omd.getlist('b') == [3]
        assert list(omd) == ['b']

    def test_delitem_missing(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            del omd['x']

    def test_pop(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        assert omd.pop('a') == 2
        assert 'a' not in omd

    def test_pop_default(self):
        omd = OrderedMultiDict()
        assert omd.pop('x', 'dflt') == 'dflt'

    def test_pop_missing(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.pop('x')

    def test_popall(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        assert omd.popall('a') == [1, 2]
        assert 'a' not in omd

    def test_popall_default(self):
        omd = OrderedMultiDict()
        assert omd.popall('x', []) == []
        with pytest.raises(KeyError):
            omd.popall('x')

    def test_poplast_by_key(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        assert omd.poplast('a') == 2
        assert omd.getlist('a') == [1]
        assert omd.poplast('a') == 1
        assert 'a' not in omd

    def test_poplast_no_key_gets_last(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
        assert omd.poplast() == 3
        assert list(omd) == ['a', 'b']
        assert omd.poplast() == 2
        assert omd.poplast() == 1
        assert list(omd) == []

    def test_poplast_empty(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.poplast()

    def test_poplast_empty_default(self):
        omd = OrderedMultiDict()
        assert omd.poplast(default='dflt') == 'dflt'

    def test_poplast_missing_key_default(self):
        omd = OrderedMultiDict()
        assert omd.poplast('x', 'dflt') == 'dflt'

    def test_poplast_missing_key_raises(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.poplast('x')


class TestOrderedMultiDictIteration:
    @pytest.fixture
    def omd(self):
        return OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])

    def test_iteritems_single(self, omd):
        assert list(omd.iteritems()) == [('a', 3), ('b', 2)]

    def test_iteritems_multi(self, omd):
        assert list(omd.iteritems(multi=True)) == [
            ('a', 1), ('b', 2), ('a', 3)]

    def test_iterkeys_single(self, omd):
        assert list(omd.iterkeys()) == ['a', 'b']

    def test_iterkeys_multi(self, omd):
        assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']

    def test_itervalues_single(self, omd):
        assert list(omd.itervalues()) == [3, 2]

    def test_itervalues_multi(self, omd):
        assert list(omd.itervalues(multi=True)) == [1, 2, 3]

    def test_keys_values_items(self, omd):
        assert omd.keys() == ['a', 'b']
        assert omd.keys(multi=True) == ['a', 'b', 'a']
        assert omd.values() == [3, 2]
        assert omd.values(multi=True) == [1, 2, 3]
        assert omd.items() == [('a', 3), ('b', 2)]
        assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]

    def test_iter(self, omd):
        assert list(iter(omd)) == ['a', 'b']

    def test_reversed(self, omd):
        assert list(reversed(omd)) == ['b', 'a']

    def test_reversed_simple(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
        assert list(reversed(omd)) == ['c', 'b', 'a']

    def test_reversed_duplicates(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3), ('b', 4)])
        # each key yielded once, at its last occurrence, in reverse order
        assert list(reversed(omd)) == ['b', 'a']

    def test_repr(self, omd):
        assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"

    def test_repr_empty(self):
        assert repr(OrderedMultiDict()) == "OrderedMultiDict([])"

    def test_views(self, omd):
        assert set(omd.viewkeys()) == {'a', 'b'}
        assert set(omd.viewvalues()) == {2, 3}
        assert set(omd.viewitems()) == {('a', 3), ('b', 2)}


class TestOrderedMultiDictEquality:
    def test_eq_self(self):
        omd = OrderedMultiDict([('a', 1)])
        assert omd == omd

    def test_eq_omd(self):
        a = OrderedMultiDict([('a', 1), ('b', 2)])
        b = OrderedMultiDict([('a', 1), ('b', 2)])
        assert a == b
        c = OrderedMultiDict([('b', 2), ('a', 1)])
        assert a != c

    def test_eq_omd_different_values(self):
        a = OrderedMultiDict([('a', 1)])
        b = OrderedMultiDict([('a', 2)])
        assert a != b

    def test_eq_dict(self):
        a = OrderedMultiDict([('a', 1), ('b', 2)])
        assert a == {'a': 1, 'b': 2}
        assert a != {'a': 1}
        assert a != {'a': 1, 'b': 2, 'c': 3}

    def test_eq_non_dict(self):
        a = OrderedMultiDict([('a', 1)])
        assert a != [('a', 1)]
        assert a != 5

    def test_ne(self):
        a = OrderedMultiDict([('a', 1)])
        b = OrderedMultiDict([('a', 2)])
        assert a != b
        assert not (a != a)


class TestOrderedMultiDictConverters:
    def test_todict(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.todict() == {'a': 3, 'b': 2}
        assert omd.todict(multi=True) == {'a': [1, 3], 'b': [2]}

    def test_todict_multi_copy(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        d = omd.todict(multi=True)
        d['a'].append(3)
        assert omd.getlist('a') == [1, 2]

    def test_sorted(self):
        omd = OrderedMultiDict(zip(range(3), range(3)))
        s = omd.sorted(reverse=True)
        assert isinstance(s, OrderedMultiDict)
        assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]

    def test_sorted_key(self):
        omd = OrderedMultiDict(zip('hello', 'world'))
        s = omd.sorted(key=lambda i: i[1])
        assert [v for _, v in s.items(multi=True)] == list('dllorw')

    def test_sortedvalues(self):
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

    def test_sortedvalues_reverse(self):
        omd = OrderedMultiDict([('a', 3), ('a', 1), ('a', 2)])
        somd = omd.sortedvalues(reverse=True)
        assert somd.getlist('a') == [3, 2, 1]

    def test_inverted(self):
        omd = OMD([(0, 2), (1, 2)])
        inv = omd.inverted()
        assert inv.getlist(2) == [0, 1]
        assert inv.inverted() == omd

    def test_inverted_roundtrip(self):
        omd = OMD([('a', 1), ('b', 2), ('a', 3)])
        assert omd.inverted().inverted() == omd

    def test_counts(self):
        omd = OMD([('a', 1), ('b', 2), ('a', 3)])
        c = omd.counts()
        assert c.todict() == {'a': 2, 'b': 1}


class TestOrderedMultiDictPickle:
    def test_pickle_roundtrip(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        data = pickle.dumps(omd)
        omd2 = pickle.loads(data)
        assert omd2 == omd
        assert omd2.items(multi=True) == omd.items(multi=True)

    def test_getstate_setstate(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        state = omd.__getstate__()
        assert state == [('a', 1), ('a', 2)]
        omd2 = OrderedMultiDict()
        omd2.__setstate__(state)
        assert omd2.items(multi=True) == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

class TestFastIterOrderedMultiDict:
    def test_basic(self):
        omd = FastIterOrderedMultiDict()
        omd.add('a', 1)
        omd.add('b', 2)
        omd.add('a', 3)
        assert omd.getlist('a') == [1, 3]
        assert omd['a'] == 3
        assert list(omd.iteritems(multi=True)) == [
            ('a', 1), ('b', 2), ('a', 3)]
        assert list(omd.iterkeys()) == ['a', 'b']

    def test_iter_multi(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(omd.iteritems(multi=True)) == [
            ('a', 1), ('b', 2), ('a', 3)]
        assert list(omd.iteritems()) == [('a', 1), ('b', 2)]
        assert list(omd.iterkeys()) == ['a', 'b']
        assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']

    def test_reversed(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(reversed(omd)) == ['b', 'a']

    def test_remove(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.poplast('a') == 3
        assert omd.getlist('a') == [1]
        assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2)]

    def test_remove_all(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        del omd['a']
        assert 'a' not in omd
        assert list(omd.iteritems(multi=True)) == [('b', 2)]

    def test_clear(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2)])
        omd.clear()
        assert list(omd.iteritems(multi=True)) == []
        omd.add('c', 3)
        assert list(omd.iteritems(multi=True)) == [('c', 3)]

    def test_is_omd_subclass(self):
        omd = FastIterOrderedMultiDict()
        assert isinstance(omd, OrderedMultiDict)


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

class TestOneToOne:
    def test_basic(self):
        oto = OneToOne({'a': 1, 'b': 2})
        assert oto['a'] == 1
        assert oto.inv[1] == 'a'
        assert len(oto) == 2
        assert len(oto.inv) == 2

    def test_overwrite_reverse(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto.inv[1] = 'c'
        assert oto.get('a') is None
        assert oto['c'] == 1
        assert len(oto) == 2
        assert oto.inv[1] == 'c'

    def test_overwrite_forward(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto['a'] = 3
        assert oto['a'] == 3
        assert 1 not in oto.inv
        assert oto.inv[3] == 'a'
        # previous owner of value 3 (none), fine

    def test_overwrite_value_removes_old(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto['b'] = 1
        assert oto['b'] == 1
        # 'a' used to map to 1, should be evicted
        assert 'a' not in oto
        assert oto.inv[1] == 'b'

    def test_setitem_unhashable(self):
        oto = OneToOne()
        with pytest.raises(TypeError):
            oto['a'] = []

    def test_delitem(self):
        oto = OneToOne({'a': 1})
        del oto['a']
        assert 'a' not in oto
        assert 1 not in oto.inv

    def test_clear(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto.clear()
        assert dict(oto) == {}
        assert dict(oto.inv) == {}

    def test_copy(self):
        oto = OneToOne({'a': 1})
        cp = oto.copy()
        assert isinstance(cp, OneToOne)
        assert cp == oto
        cp['b'] = 2
        assert 'b' not in oto

    def test_pop(self):
        oto = OneToOne({'a': 1, 'b': 2})
        assert oto.pop('a') == 1
        assert 'a' not in oto
        assert 1 not in oto.inv

    def test_pop_default(self):
        oto = OneToOne()
        assert oto.pop('x', 'dflt') == 'dflt'

    def test_pop_missing(self):
        oto = OneToOne()
        with pytest.raises(KeyError):
            oto.pop('x')

    def test_popitem(self):
        oto = OneToOne({'a': 1})
        k, v = oto.popitem()
        assert (k, v) == ('a', 1)
        assert 'a' not in oto
        assert 1 not in oto.inv

    def test_setdefault(self):
        oto = OneToOne()
        assert oto.setdefault('a', 1) == 1
        assert oto['a'] == 1
        assert oto.inv[1] == 'a'
        assert oto.setdefault('a', 5) == 1

    def test_update(self):
        oto = OneToOne()
        oto.update({'a': 1, 'b': 2})
        assert oto.todict() if hasattr(oto, 'todict') else dict(oto) == {
            'a': 1, 'b': 2}
        assert oto.inv[1] == 'a'

    def test_update_iterable(self):
        oto = OneToOne()
        oto.update([('a', 1), ('b', 2)])
        assert dict(oto) == {'a': 1, 'b': 2}

    def test_update_kwargs(self):
        oto = OneToOne()
        oto.update({'a': 1}, b=2)
        assert dict(oto) == {'a': 1, 'b': 2}
        assert oto.inv[2] == 'b'

    def test_update_unhashable(self):
        oto = OneToOne()
        with pytest.raises(TypeError):
            oto.update({'a': []})

    def test_repr(self):
        oto = OneToOne({'a': 1})
        r = repr(oto)
        assert r.startswith('OneToOne(') and r.endswith(')')

    def test_unique_ok(self):
        oto = OneToOne.unique({'a': 1, 'b': 2})
        assert dict(oto) == {'a': 1, 'b': 2}

    def test_unique_dupes(self):
        with pytest.raises(ValueError):
            OneToOne.unique({'a': 1, 'b': 1})

    def test_unique_dupes_across_inputs(self):
        a_dict = {'a': 2}
        with pytest.raises(ValueError):
            OneToOne.unique(a_dict, b=2)

    def test_default_dupe_handling(self):
        # On dupe values, non-unique constructor keeps the last one
        oto = OneToOne({'a': 1, 'b': 1})
        # lengths differ -> fallback path rebuilds from inv
        assert len(oto) == 1
        assert oto.inv[1] in ('a', 'b')


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

class TestManyToMany:
    def test_empty(self):
        m2m = ManyToMany()
        assert len(m2m) == 0
        assert m2m.get('x') == frozenset()

    def test_init_from_pairs(self):
        m2m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
        assert m2m['a'] == frozenset([1, 2])
        assert m2m['b'] == frozenset([2])
        assert m2m.inv[2] == frozenset(['a', 'b'])
        assert m2m.inv[1] == frozenset(['a'])

    def test_getitem_missing(self):
        m2m = ManyToMany()
        with pytest.raises(KeyError):
            m2m['nope']

    def test_get_default(self):
        m2m = ManyToMany()
        assert m2m.get('x') == frozenset()
        assert m2m.get('x', 'dflt') == 'dflt'

    def test_add_remove(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        assert m2m['a'] == frozenset([1, 2])
        assert m2m.inv[1] == frozenset(['a'])
        m2m.remove('a', 1)
        assert m2m['a'] == frozenset([2])
        assert 1 not in m2m.inv
        m2m.remove('a', 2)
        assert 'a' not in m2m

    def test_setitem(self):
        m2m = ManyToMany()
        m2m['a'] = [1, 2, 3]
        assert m2m['a'] == frozenset([1, 2, 3])
        m2m['a'] = [2, 4]
        assert m2m['a'] == frozenset([2, 4])
        assert 1 not in m2m.inv
        assert 3 not in m2m.inv
        assert m2m.inv[4] == frozenset(['a'])

    def test_delitem(self):
        m2m = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
        del m2m['a']
        assert 'a' not in m2m
        assert m2m.inv[1] == frozenset(['b'])
        assert 2 not in m2m.inv

    def test_update_from_iterable(self):
        m2m = ManyToMany()
        m2m.update([('a', 1), ('b', 2)])
        assert m2m['a'] == frozenset([1])
        assert m2m['b'] == frozenset([2])

    def test_update_from_mapping(self):
        m2m = ManyToMany()
        m2m.update({'a': 1, 'b': 2})
        assert m2m['a'] == frozenset([1])
        assert m2m['b'] == frozenset([2])

    def test_update_from_m2m(self):
        a = ManyToMany([('a', 1)])
        b = ManyToMany([('a', 2), ('b', 3)])
        a.update(b)
        assert a['a'] == frozenset([1, 2])
        assert a['b'] == frozenset([3])
        assert a.inv[2] == frozenset(['a'])

    def test_replace(self):
        m2m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
        m2m.replace('a', 'c')
        assert 'a' not in m2m
        assert m2m['c'] == frozenset([1, 2])
        assert m2m.inv[2] == frozenset(['c', 'b'])

    def test_replace_missing(self):
        m2m = ManyToMany([('a', 1)])
        m2m.replace('missing', 'x')
        assert 'x' not in m2m

    def test_iteritems_keys_iter_len_contains(self):
        m2m = ManyToMany([('a', 1), ('a', 2), ('b', 2)])
        items = set(m2m.iteritems())
        assert items == {('a', 1), ('a', 2), ('b', 2)}
        assert set(m2m.keys()) == {'a', 'b'}
        assert set(iter(m2m)) == {'a', 'b'}
        assert len(m2m) == 2
        assert 'a' in m2m
        assert 'c' not in m2m

    def test_eq(self):
        a = ManyToMany([('a', 1)])
        b = ManyToMany([('a', 1)])
        assert a == b
        c = ManyToMany([('a', 2)])
        assert a != c
        assert a != {'a': frozenset([1])}

    def test_repr(self):
        m2m = ManyToMany([('a', 1)])
        r = repr(m2m)
        assert r.startswith('ManyToMany(')


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

class TestSubdict:
    def test_no_args(self):
        d = {'a': 1, 'b': 2}
        assert subdict(d) == d
        assert subdict(d) is not d

    def test_drop(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, drop=['b', 'c']) == {'a': 1}

    def test_keep(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}

    def test_keep_and_drop(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, keep=['a', 'b', 'c'], drop=['b']) == {
            'a': 1, 'c': 3}

    def test_keep_missing(self):
        d = {'a': 1}
        assert subdict(d, keep=['a', 'z']) == {'a': 1}

    def test_preserves_type(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2)])
        result = subdict(omd, keep=['a'])
        assert isinstance(result, OrderedMultiDict)
        assert result.todict() == {'a': 1}


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

class TestFrozenDict:
    def test_basic(self):
        fd = FrozenDict({'a': 1, 'b': 2})
        assert fd['a'] == 1
        assert dict(fd) == {'a': 1, 'b': 2}
        assert len(fd) == 2

    def test_hashable(self):
        fd = FrozenDict({'a': 1})
        h = hash(fd)
        assert isinstance(h, int)
        # repeated hash uses cache
        assert hash(fd) == h

    def test_usable_as_key(self):
        fd = FrozenDict({'a': 1})
        d = {fd: 'value'}
        assert d[FrozenDict({'a': 1})] == 'value'

    def test_mutation_blocked(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd['b'] = 2
        with pytest.raises(TypeError):
            del fd['a']
        with pytest.raises(TypeError):
            fd.update({'b': 2})
        with pytest.raises(TypeError):
            fd.clear()
        with pytest.raises(TypeError):
            fd.pop('a')
        with pytest.raises(TypeError):
            fd.popitem()
        with pytest.raises(TypeError):
            fd.setdefault('b', 2)
        with pytest.raises(TypeError):
            fd |= {'b': 2}

    def test_updated(self):
        fd = FrozenDict({'a': 1})
        fd2 = fd.updated({'b': 2}, c=3)
        assert isinstance(fd2, FrozenDict)
        assert dict(fd2) == {'a': 1, 'b': 2, 'c': 3}
        assert dict(fd) == {'a': 1}

    def test_fromkeys(self):
        fd = FrozenDict.fromkeys(['a', 'b'])
        assert dict(fd) == {'a': None, 'b': None}
        fd2 = FrozenDict.fromkeys(['x'], 0)
        assert dict(fd2) == {'x': 0}

    def test_repr(self):
        fd = FrozenDict({'a': 1})
        assert repr(fd) == "FrozenDict({'a': 1})"

    def test_copy_is_self(self):
        fd = FrozenDict({'a': 1})
        assert copy.copy(fd) is fd

    def test_pickle(self):
        fd = FrozenDict({'a': 1, 'b': 2})
        fd2 = pickle.loads(pickle.dumps(fd))
        assert isinstance(fd2, FrozenDict)
        assert dict(fd2) == {'a': 1, 'b': 2}
        assert hash(fd2) == hash(fd)

    def test_unhashable_hash_error(self):
        fd = FrozenDict({'a': [1, 2]})
        with pytest.raises(FrozenHashError):
            hash(fd)
        # subsequent calls raise the same cached error
        with pytest.raises(FrozenHashError):
            hash(fd)

    def test_frozen_hash_error_is_type_error(self):
        assert issubclass(FrozenHashError, TypeError)
