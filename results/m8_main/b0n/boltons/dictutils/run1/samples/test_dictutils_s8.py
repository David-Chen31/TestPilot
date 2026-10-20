"""Unit tests for boltons.dictutils."""

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
# OrderedMultiDict - basics
# ---------------------------------------------------------------------------

class TestOrderedMultiDictBasic:
    def test_empty_construction(self):
        omd = OrderedMultiDict()
        assert len(omd) == 0
        assert list(omd) == []
        assert omd.items() == []
        assert omd.keys() == []
        assert omd.values() == []

    def test_aliases(self):
        assert OMD is OrderedMultiDict
        assert MultiDict is OrderedMultiDict

    def test_init_with_list_of_pairs(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
        assert omd.get('a') == 3
        assert omd.getlist('a') == [1, 3]
        assert omd['b'] == 2

    def test_init_with_dict(self):
        omd = OrderedMultiDict({'a': 1, 'b': 2})
        assert omd['a'] == 1
        assert omd['b'] == 2
        assert len(omd) == 2

    def test_init_kwargs(self):
        omd = OrderedMultiDict(a=1, b=2)
        assert omd['a'] == 1
        assert omd['b'] == 2

    def test_init_too_many_args_raises(self):
        with pytest.raises(TypeError):
            OrderedMultiDict([], [])

    def test_setitem_and_getitem(self):
        omd = OrderedMultiDict()
        omd['a'] = 1
        assert omd['a'] == 1
        assert len(omd) == 1
        # overwrite collapses
        omd['a'] = 2
        assert omd['a'] == 2
        assert omd.getlist('a') == [2]

    def test_add_preserves_values(self):
        omd = OrderedMultiDict()
        omd['a'] = 1
        omd.add('a', 3)
        assert omd.get('a') == 3
        assert omd.getlist('a') == [1, 3]
        assert omd['a'] == 3

    def test_addlist(self):
        omd = OrderedMultiDict([('a', -1)])
        omd.addlist('a', range(3))
        assert omd.getlist('a') == [-1, 0, 1, 2]
        assert omd.items(multi=True) == [('a', -1), ('a', 0), ('a', 1), ('a', 2)]

    def test_addlist_empty_noop(self):
        omd = OrderedMultiDict()
        omd.addlist('a', [])
        assert 'a' not in omd
        assert len(omd) == 0

    def test_addlist_with_iterator(self):
        omd = OrderedMultiDict()
        omd.addlist('a', iter([1, 2, 3]))
        assert omd.getlist('a') == [1, 2, 3]

    def test_get_default(self):
        omd = OrderedMultiDict()
        assert omd.get('missing') is None
        assert omd.get('missing', 'dflt') == 'dflt'
        omd['a'] = 1
        assert omd.get('a', 'x') == 1

    def test_getlist_default(self):
        omd = OrderedMultiDict()
        assert omd.getlist('missing') == []
        sentinel = object()
        assert omd.getlist('missing', sentinel) is sentinel
        omd.add('a', 1)
        got = omd.getlist('a')
        assert got == [1]
        got.append(5)
        # returned list is a copy
        assert omd.getlist('a') == [1]

    def test_setdefault_creates(self):
        omd = OrderedMultiDict()
        assert omd.setdefault('a') is None
        assert omd['a'] is None
        assert omd.setdefault('a', 5) is None
        omd2 = OrderedMultiDict()
        assert omd2.setdefault('b', 7) == 7
        assert omd2['b'] == 7

    def test_setdefault_existing_returns_value(self):
        omd = OrderedMultiDict()
        omd['a'] = 1
        assert omd.setdefault('a', 99) == 1
        assert omd.getlist('a') == [1]

    def test_contains(self):
        omd = OrderedMultiDict([('a', 1)])
        assert 'a' in omd
        assert 'b' not in omd

    def test_delitem(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        del omd['a']
        assert 'a' not in omd
        assert omd.getlist('b') == [3]
        with pytest.raises(KeyError):
            del omd['nope']

    def test_clear(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2)])
        omd.clear()
        assert len(omd) == 0
        assert omd.items() == []
        # can still be used
        omd['c'] = 3
        assert omd['c'] == 3

    def test_copy_shallow(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        cp = omd.copy()
        assert isinstance(cp, OrderedMultiDict)
        assert cp.items(multi=True) == omd.items(multi=True)
        cp.add('a', 4)
        assert omd.getlist('a') == [1, 3]
        assert cp.getlist('a') == [1, 3, 4]

    def test_fromkeys(self):
        omd = OrderedMultiDict.fromkeys(['a', 'b'])
        assert omd['a'] is None
        assert omd['b'] is None
        omd2 = OrderedMultiDict.fromkeys(['a', 'b'], 0)
        assert omd2['a'] == 0
        assert omd2.keys() == ['a', 'b']


# ---------------------------------------------------------------------------
# OrderedMultiDict - update / update_extend
# ---------------------------------------------------------------------------

class TestOrderedMultiDictUpdate:
    def test_update_with_plain_dict(self):
        omd = OrderedMultiDict([('a', 1)])
        omd.update({'a': 2, 'b': 3})
        assert omd['a'] == 2
        assert omd['b'] == 3

    def test_update_with_omd(self):
        src = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        dst = OrderedMultiDict([('a', 99)])
        dst.update(src)
        assert dst.getlist('a') == [1, 2]
        assert dst['b'] == 3

    def test_update_with_iterable(self):
        omd = OrderedMultiDict([('a', 1)])
        omd.update([('a', 2), ('b', 3), ('a', 4)])
        assert omd.getlist('a') == [2, 4]
        assert omd['b'] == 3

    def test_update_with_kwargs(self):
        omd = OrderedMultiDict()
        omd.update({'a': 1}, b=2)
        assert omd['a'] == 1
        assert omd['b'] == 2

    def test_update_with_self_is_noop(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        omd.update(omd)
        assert omd.getlist('a') == [1, 2]

    def test_ior(self):
        omd = OrderedMultiDict([('a', 1)])
        omd |= {'a': 2, 'b': 3}
        assert omd['a'] == 2
        assert omd['b'] == 3

    def test_update_extend(self):
        omd = OrderedMultiDict([('a', 1)])
        omd.update_extend([('a', 2), ('b', 3)])
        assert omd.getlist('a') == [1, 2]
        assert omd['b'] == 3

    def test_update_extend_with_omd(self):
        src = OrderedMultiDict([('a', 1), ('a', 2)])
        dst = OrderedMultiDict([('a', 3)])
        dst.update_extend(src)
        assert dst.getlist('a') == [3, 1, 2]

    def test_update_extend_with_dict_like(self):
        class DictLike:
            def keys(self):
                return ['a', 'b']

            def __getitem__(self, k):
                return {'a': 1, 'b': 2}[k]

        omd = OrderedMultiDict()
        omd.update_extend(DictLike())
        assert omd['a'] == 1
        assert omd['b'] == 2

    def test_update_extend_with_self(self):
        omd = OrderedMultiDict([('a', 1)])
        omd.update_extend(omd)
        assert omd.getlist('a') == [1, 1]


# ---------------------------------------------------------------------------
# OrderedMultiDict - pop variants
# ---------------------------------------------------------------------------

class TestOrderedMultiDictPop:
    def test_pop_returns_latest_and_removes_all(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.pop('a') == 3
        assert 'a' not in omd
        assert omd['b'] == 2

    def test_pop_missing_raises(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.pop('missing')

    def test_pop_missing_default(self):
        omd = OrderedMultiDict()
        assert omd.pop('missing', 'dflt') == 'dflt'

    def test_popall(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        assert omd.popall('a') == [1, 2]
        assert 'a' not in omd
        assert omd['b'] == 3

    def test_popall_missing_raises(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.popall('missing')

    def test_popall_missing_default(self):
        omd = OrderedMultiDict()
        assert omd.popall('missing', 'dflt') == 'dflt'

    def test_poplast_specific_key(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        assert omd.poplast('a') == 2
        assert omd.getlist('a') == [1]
        assert omd.poplast('a') == 1
        assert 'a' not in omd

    def test_poplast_missing_raises(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.poplast('missing')

    def test_poplast_missing_default(self):
        omd = OrderedMultiDict()
        assert omd.poplast('missing', 'dflt') == 'dflt'

    def test_poplast_no_key_returns_last(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('c', 3)])
        assert omd.poplast() == 3
        assert 'c' not in omd
        assert omd.poplast() == 2
        assert omd.poplast() == 1
        assert len(omd) == 0

    def test_poplast_empty_raises(self):
        omd = OrderedMultiDict()
        with pytest.raises(KeyError):
            omd.poplast()

    def test_poplast_empty_default(self):
        omd = OrderedMultiDict()
        assert omd.poplast(_default_marker()) == _default_marker() if False else True
        assert omd.poplast(default='dflt') == 'dflt'


def _default_marker():
    return object


# ---------------------------------------------------------------------------
# OrderedMultiDict - iteration / views
# ---------------------------------------------------------------------------

class TestOrderedMultiDictIteration:
    def setup_method(self):
        self.omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])

    def test_iterkeys_default(self):
        assert list(self.omd.iterkeys()) == ['a', 'b']

    def test_iterkeys_multi(self):
        assert list(self.omd.iterkeys(multi=True)) == ['a', 'b', 'a']

    def test_iteritems_default(self):
        assert list(self.omd.iteritems()) == [('a', 3), ('b', 2)]

    def test_iteritems_multi(self):
        assert list(self.omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]

    def test_itervalues_default(self):
        assert list(self.omd.itervalues()) == [3, 2]

    def test_itervalues_multi(self):
        assert list(self.omd.itervalues(multi=True)) == [1, 2, 3]

    def test_keys_values_items(self):
        assert self.omd.keys() == ['a', 'b']
        assert self.omd.keys(multi=True) == ['a', 'b', 'a']
        assert self.omd.values() == [3, 2]
        assert self.omd.values(multi=True) == [1, 2, 3]
        assert self.omd.items() == [('a', 3), ('b', 2)]
        assert self.omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]

    def test_dunder_iter(self):
        assert list(self.omd) == ['a', 'b']

    def test_reversed(self):
        assert list(reversed(self.omd)) == ['b', 'a']

    def test_reversed_multi(self):
        # reversed yields keys in reverse order according to last occurrence
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(reversed(omd)) == ['b', 'a']

    def test_viewkeys(self):
        vk = self.omd.viewkeys()
        assert 'a' in vk
        assert set(vk) == {'a', 'b'}

    def test_viewvalues(self):
        vv = self.omd.viewvalues()
        assert 3 in vv
        assert 2 in vv

    def test_viewitems(self):
        vi = self.omd.viewitems()
        assert ('a', 3) in vi
        assert ('b', 2) in vi


# ---------------------------------------------------------------------------
# OrderedMultiDict - todict / sorted / sortedvalues / inverted / counts
# ---------------------------------------------------------------------------

class TestOrderedMultiDictTransforms:
    def test_todict(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.todict() == {'a': 3, 'b': 2}
        multi = omd.todict(multi=True)
        assert multi == {'a': [1, 3], 'b': [2]}
        # lists are copies
        multi['a'].append(99)
        assert omd.getlist('a') == [1, 3]

    def test_sorted(self):
        omd = OrderedMultiDict(zip(range(3), range(3)))
        sorted_omd = omd.sorted(reverse=True)
        assert sorted_omd.items(multi=True) == [(2, 2), (1, 1), (0, 0)]
        assert omd.items(multi=True) == [(0, 0), (1, 1), (2, 2)]

    def test_sorted_with_key(self):
        omd = OrderedMultiDict(zip('hello', 'world'))
        result = omd.sorted(key=lambda i: i[1])
        assert result.items(multi=True) == [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]

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
        omd = OrderedMultiDict()
        omd.addlist('a', [3, 1, 2])
        somd = omd.sortedvalues(reverse=True)
        assert somd.getlist('a') == [3, 2, 1]

    def test_inverted(self):
        omd = OMD([(0, 2), (1, 2)])
        inv = omd.inverted()
        assert inv.getlist(2) == [0, 1]
        assert inv.inverted().items(multi=True) == [(0, 2), (1, 2)]

    def test_counts(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        counts = omd.counts()
        assert isinstance(counts, OrderedMultiDict)
        assert counts['a'] == 2
        assert counts['b'] == 1
        assert counts.keys() == ['a', 'b']

    def test_repr(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"

    def test_repr_empty(self):
        assert repr(OrderedMultiDict()) == "OrderedMultiDict([])"


# ---------------------------------------------------------------------------
# OrderedMultiDict - equality
# ---------------------------------------------------------------------------

class TestOrderedMultiDictEquality:
    def test_self_equal(self):
        omd = OrderedMultiDict([('a', 1)])
        assert omd == omd

    def test_equal_omd(self):
        a = OrderedMultiDict([('a', 1), ('b', 2)])
        b = OrderedMultiDict([('a', 1), ('b', 2)])
        assert a == b

    def test_not_equal_omd_order(self):
        a = OrderedMultiDict([('a', 1), ('b', 2)])
        b = OrderedMultiDict([('b', 2), ('a', 1)])
        # OMD equality considers multi-item order
        assert a != b

    def test_not_equal_multi(self):
        a = OrderedMultiDict([('a', 1)])
        b = OrderedMultiDict([('a', 1), ('a', 2)])
        assert a != b

    def test_equal_plain_dict(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2)])
        assert omd == {'a': 1, 'b': 2}

    def test_not_equal_plain_dict(self):
        omd = OrderedMultiDict([('a', 1)])
        assert omd != {'a': 2}
        assert omd != {'a': 1, 'b': 2}
        assert omd != {'b': 1}

    def test_not_equal_non_dict(self):
        omd = OrderedMultiDict([('a', 1)])
        assert omd != [('a', 1)]

    def test_ne(self):
        a = OrderedMultiDict([('a', 1)])
        b = OrderedMultiDict([('a', 2)])
        assert a != b
        assert not (a == b)


# ---------------------------------------------------------------------------
# OrderedMultiDict - pickle
# ---------------------------------------------------------------------------

class TestOrderedMultiDictPickle:
    def test_pickle_roundtrip_keeps_multi(self):
        omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        data = pickle.dumps(omd)
        restored = pickle.loads(data)
        assert restored.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]

    def test_getstate(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        assert omd.__getstate__() == [('a', 1), ('a', 2)]

    def test_setstate(self):
        omd = OrderedMultiDict()
        omd.__setstate__([('a', 1), ('a', 2)])
        assert omd.getlist('a') == [1, 2]

    def test_reduce(self):
        omd = OrderedMultiDict([('a', 1), ('a', 2)])
        cls, args, state = omd.__reduce__()
        assert cls is OrderedMultiDict
        assert args == ()
        assert state == [('a', 1), ('a', 2)]


# ---------------------------------------------------------------------------
# FastIterOrderedMultiDict
# ---------------------------------------------------------------------------

class TestFastIterOrderedMultiDict:
    def test_basic_construction(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert omd.getlist('a') == [1, 3]
        assert omd['b'] == 2

    def test_iteritems(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
        assert list(omd.iteritems()) == [('a', 3), ('b', 2)]

    def test_iterkeys(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
        assert list(omd.iterkeys()) == ['a', 'b']

    def test_reversed(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
        assert list(reversed(omd)) == ['b', 'a']

    def test_add_and_delete(self):
        omd = FastIterOrderedMultiDict()
        omd.add('a', 1)
        omd.add('b', 2)
        omd.add('a', 3)
        assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
        del omd['a']
        assert list(omd.iterkeys(multi=True)) == ['b']

    def test_poplast(self):
        omd = FastIterOrderedMultiDict([('a', 1), ('a', 2)])
        assert omd.poplast('a') == 2
        assert omd.getlist('a') == [1]

    def test_update(self):
        omd = FastIterOrderedMultiDict([('a', 1)])
        omd.update([('a', 2), ('b', 3)])
        assert omd.getlist('a') == [2]
        assert omd['b'] == 3


# ---------------------------------------------------------------------------
# OneToOne
# ---------------------------------------------------------------------------

class TestOneToOne:
    def test_basic(self):
        oto = OneToOne({'a': 1, 'b': 2})
        assert oto['a'] == 1
        assert oto.inv[1] == 'a'
        assert oto.inv[2] == 'b'
        assert len(oto) == 2

    def test_overwrite_via_inv(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto.inv[1] = 'c'
        assert oto.get('a') is None
        assert oto['c'] == 1
        assert oto.inv[1] == 'c'
        assert len(oto) == 2

    def test_setitem_overwrites_both_directions(self):
        oto = OneToOne({'a': 1})
        oto['a'] = 2
        assert oto['a'] == 2
        assert oto.inv[2] == 'a'
        assert 1 not in oto.inv

    def test_setitem_value_collision(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto['c'] = 1
        assert oto['c'] == 1
        # 'a' should be removed since value 1 now maps to 'c'
        assert 'a' not in oto
        assert oto.inv[1] == 'c'
        assert len(oto) == 2

    def test_unhashable_value_raises(self):
        oto = OneToOne()
        with pytest.raises(TypeError):
            oto['a'] = [1, 2]

    def test_delitem(self):
        oto = OneToOne({'a': 1})
        del oto['a']
        assert 'a' not in oto
        assert 1 not in oto.inv

    def test_clear(self):
        oto = OneToOne({'a': 1, 'b': 2})
        oto.clear()
        assert len(oto) == 0
        assert len(oto.inv) == 0

    def test_copy(self):
        oto = OneToOne({'a': 1})
        cp = oto.copy()
        assert isinstance(cp, OneToOne)
        assert cp == oto
        assert cp.inv[1] == 'a'
        cp['b'] = 2
        assert 'b' not in oto

    def test_pop(self):
        oto = OneToOne({'a': 1})
        assert oto.pop('a') == 1
        assert 'a' not in oto
        assert 1 not in oto.inv

    def test_pop_missing_raises(self):
        oto = OneToOne()
        with pytest.raises(KeyError):
            oto.pop('missing')

    def test_pop_missing_default(self):
        oto = OneToOne()
        assert oto.pop('missing', 'dflt') == 'dflt'

    def test_popitem(self):
        oto = OneToOne({'a': 1})
        k, v = oto.popitem()
        assert (k, v) == ('a', 1)
        assert len(oto) == 0
        assert len(oto.inv) == 0

    def test_setdefault(self):
        oto = OneToOne()
        assert oto.setdefault('a', 1) == 1
        assert oto['a'] == 1
        assert oto.inv[1] == 'a'
        assert oto.setdefault('a', 99) == 1

    def test_update_with_dict(self):
        oto = OneToOne()
        oto.update({'a': 1, 'b': 2})
        assert oto['a'] == 1
        assert oto.inv[2] == 'b'

    def test_update_with_iterable(self):
        oto = OneToOne()
        oto.update([('a', 1), ('b', 2)])
        assert oto['a'] == 1
        assert oto['b'] == 2

    def test_update_with_kwargs(self):
        oto = OneToOne()
        oto.update({'a': 1}, b=2)
        assert oto['a'] == 1
        assert oto['b'] == 2

    def test_unique_success(self):
        oto = OneToOne.unique({'a': 1, 'b': 2})
        assert oto['a'] == 1
        assert oto.inv[2] == 'b'

    def test_unique_duplicate_raises(self):
        with pytest.raises(ValueError):
            OneToOne.unique({'a': 1, 'b': 1})

    def test_unique_duplicate_across_inputs(self):
        a_dict = {'a': 2}
        with pytest.raises(ValueError):
            OneToOne.unique(a_dict, b=2)

    def test_non_unique_collapses(self):
        # non-strict constructor collapses duplicate values
        oto = OneToOne({'a': 1, 'b': 1})
        assert len(oto) == 1
        assert oto.inv[1] in ('a', 'b')

    def test_repr(self):
        oto = OneToOne({'a': 1})
        assert repr(oto) == "OneToOne({'a': 1})"


# ---------------------------------------------------------------------------
# ManyToMany
# ---------------------------------------------------------------------------

class TestManyToMany:
    def test_basic(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        m2m.add('b', 2)
        assert m2m['a'] == frozenset([1, 2])
        assert m2m['b'] == frozenset([2])
        assert m2m.inv[2] == frozenset(['a', 'b'])
        assert m2m.inv[1] == frozenset(['a'])

    def test_get_default(self):
        m2m = ManyToMany()
        assert m2m.get('missing') == frozenset()
        assert m2m.get('missing', 'dflt') == 'dflt'
        m2m.add('a', 1)
        assert m2m.get('a') == frozenset([1])

    def test_contains(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        assert 'a' in m2m
        assert 'b' not in m2m

    def test_len_and_iter(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('b', 2)
        assert len(m2m) == 2
        assert set(iter(m2m)) == {'a', 'b'}

    def test_setitem(self):
        m2m = ManyToMany()
        m2m['a'] = [1, 2]
        assert m2m['a'] == frozenset([1, 2])
        m2m['a'] = [2, 3]
        assert m2m['a'] == frozenset([2, 3])
        assert 1 not in m2m.inv

    def test_delitem(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        del m2m['a']
        assert 'a' not in m2m
        assert 1 not in m2m.inv
        assert 2 not in m2m.inv

    def test_update_with_iterable(self):
        m2m = ManyToMany()
        m2m.update([('a', 1), ('b', 2)])
        assert m2m['a'] == frozenset([1])
        assert m2m['b'] == frozenset([2])

    def test_update_with_m2m(self):
        src = ManyToMany([('a', 1), ('b', 2)])
        dst = ManyToMany([('a', 3)])
        dst.update(src)
        assert dst['a'] == frozenset([1, 3])
        assert dst['b'] == frozenset([2])

    def test_update_with_dict_like(self):
        class DictLike:
            def keys(self):
                return ['a', 'b']

            def __getitem__(self, k):
                return {'a': 1, 'b': 2}[k]

        m2m = ManyToMany()
        m2m.update(DictLike())
        assert m2m['a'] == frozenset([1])
        assert m2m['b'] == frozenset([2])

    def test_remove(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        m2m.remove('a', 1)
        assert m2m['a'] == frozenset([2])
        assert 1 not in m2m.inv

    def test_replace(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        m2m.replace('a', 'z')
        assert 'a' not in m2m
        assert m2m['z'] == frozenset([1, 2])
        assert m2m.inv[1] == frozenset(['z'])

    def test_replace_missing_key_noop(self):
        m2m = ManyToMany()
        m2m.replace('missing', 'z')
        assert 'z' not in m2m

    def test_iteritems(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('a', 2)
        items = list(m2m.iteritems())
        assert set(items) == {('a', 1), ('a', 2)}

    def test_keys(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        m2m.add('b', 2)
        assert set(m2m.keys()) == {'a', 'b'}

    def test_equality(self):
        a = ManyToMany([('a', 1)])
        b = ManyToMany([('a', 1)])
        c = ManyToMany([('a', 2)])
        assert a == b
        assert a != c
        assert a != {'a': {1}}

    def test_repr(self):
        m2m = ManyToMany()
        m2m.add('a', 1)
        assert repr(m2m) == "ManyToMany([('a', 1)])"

    def test_init_with_items(self):
        m2m = ManyToMany([('a', 1), ('b', 2)])
        assert m2m['a'] == frozenset([1])
        assert m2m['b'] == frozenset([2])


# ---------------------------------------------------------------------------
# subdict
# ---------------------------------------------------------------------------

class TestSubdict:
    def test_default_returns_copy(self):
        d = {'a': 1, 'b': 2}
        result = subdict(d)
        assert result == d
        assert result is not d

    def test_keep(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}

    def test_drop(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, drop=['b', 'c']) == {'a': 1}

    def test_keep_and_drop(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}

    def test_keep_missing_keys_ignored(self):
        d = {'a': 1}
        assert subdict(d, keep=['a', 'missing']) == {'a': 1}

    def test_preserves_type(self):
        d = {'a': 1}
        result = subdict(d)
        assert type(result) is dict


# ---------------------------------------------------------------------------
# FrozenDict
# ---------------------------------------------------------------------------

class TestFrozenDict:
    def test_basic(self):
        fd = FrozenDict({'a': 1, 'b': 2})
        assert fd['a'] == 1
        assert len(fd) == 2
        assert fd == {'a': 1, 'b': 2}

    def test_hashable(self):
        fd = FrozenDict({'a': 1})
        h = hash(fd)
        assert isinstance(h, int)
        # same content same hash
        assert hash(FrozenDict({'a': 1})) == h

    def test_usable_as_key(self):
        fd = FrozenDict({'a': 1})
        d = {fd: 'value'}
        assert d[fd] == 'value'
        s = {fd}
        assert fd in s

    def test_hash_cached(self):
        fd = FrozenDict({'a': 1})
        h1 = hash(fd)
        h2 = hash(fd)
        assert h1 == h2
        assert fd._hash == h1

    def test_unhashable_raises_frozen_hash_error(self):
        fd = FrozenDict({'a': [1, 2]})
        with pytest.raises(FrozenHashError):
            hash(fd)
        # FrozenHashError is a TypeError
        assert issubclass(FrozenHashError, TypeError)

    def test_unhashable_cached_raises(self):
        fd = FrozenDict({'a': [1, 2]})
        with pytest.raises(FrozenHashError):
            hash(fd)
        with pytest.raises(FrozenHashError):
            hash(fd)

    def test_immutable_setitem(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd['b'] = 2

    def test_immutable_delitem(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            del fd['a']

    def test_immutable_update(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd.update({'b': 2})

    def test_immutable_setdefault(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd.setdefault('b', 2)

    def test_immutable_pop(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd.pop('a')

    def test_immutable_popitem(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd.popitem()

    def test_immutable_clear(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd.clear()

    def test_immutable_ior(self):
        fd = FrozenDict({'a': 1})
        with pytest.raises(TypeError):
            fd |= {'b': 2}

    def test_updated_returns_new(self):
        fd = FrozenDict({'a': 1})
        fd2 = fd.updated({'b': 2})
        assert fd2 == {'a': 1, 'b': 2}
        assert isinstance(fd2, FrozenDict)
        assert fd == {'a': 1}

    def test_updated_with_iterable_and_kwargs(self):
        fd = FrozenDict({'a': 1})
        fd2 = fd.updated([('b', 2)], c=3)
        assert fd2 == {'a': 1, 'b': 2, 'c': 3}

    def test_fromkeys(self):
        fd = FrozenDict.fromkeys(['a', 'b'], 0)
        assert isinstance(fd, FrozenDict)
        assert fd == {'a': 0, 'b': 0}

    def test_repr(self):
        fd = FrozenDict({'a': 1})
        assert repr(fd) == "FrozenDict({'a': 1})"

    def test_copy_returns_self(self):
        fd = FrozenDict({'a': 1})
        assert copy.copy(fd) is fd

    def test_pickle_roundtrip(self):
        fd = FrozenDict({'a': 1, 'b': 2})
        data = pickle.dumps(fd)
        restored = pickle.loads(data)
        assert restored == fd
        assert isinstance(restored, FrozenDict)

    def test_reduce_ex(self):
        fd = FrozenDict({'a': 1})
        cls, args = fd.__reduce_ex__(2)
        assert cls is FrozenDict
        assert args == ({'a': 1},)
