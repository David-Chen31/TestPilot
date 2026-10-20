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


# ------------------------------------------------------------------ #
# OrderedMultiDict basics
# ------------------------------------------------------------------ #

def test_aliases():
    assert OMD is OrderedMultiDict
    assert MultiDict is OrderedMultiDict


def test_basic_add_get():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd['a'] == 3
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('b') == [2]
    assert omd.getlist('missing') == []


def test_get_default():
    omd = OrderedMultiDict()
    assert omd.get('missing') is None
    assert omd.get('missing', 42) == 42


def test_getlist_default():
    omd = OrderedMultiDict()
    assert omd.getlist('missing') == []
    assert omd.getlist('missing', 'DEF') == 'DEF'


def test_getlist_is_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    lst = omd.getlist('a')
    lst.append(3)
    assert omd.getlist('a') == [1, 2]


def test_init_too_many_args():
    with pytest.raises(TypeError):
        OrderedMultiDict([('a', 1)], [('b', 2)])


def test_init_kwargs():
    omd = OrderedMultiDict(a=1, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_reversed():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert list(reversed(omd)) == ['b', 'a']


def test_repr():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"


def test_iteration_default():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd) == ['a', 'b']
    assert list(omd.iteritems()) == [('a', 3), ('b', 2)]
    assert list(omd.iterkeys()) == ['a', 'b']
    assert list(omd.itervalues()) == [3, 2]


def test_iteration_multi():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(omd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(omd.itervalues(multi=True)) == [1, 2, 3]


def test_keys_values_items():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]


def test_todict():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.todict() == {'a': 3, 'b': 2}
    multi = omd.todict(multi=True)
    assert multi == {'a': [1, 3], 'b': [2]}
    multi['a'].append(99)
    assert omd.getlist('a') == [1, 3]


def test_addlist():
    omd = OrderedMultiDict([('a', -1)])
    omd.addlist('a', range(3))
    assert omd.getlist('a') == [-1, 0, 1, 2]


def test_addlist_generator():
    omd = OrderedMultiDict()
    omd.addlist('a', (i for i in range(3)))
    assert omd.getlist('a') == [0, 1, 2]


def test_addlist_empty():
    omd = OrderedMultiDict([('a', 1)])
    omd.addlist('a', [])
    assert omd.getlist('a') == [1]
    assert 'b' not in omd


def test_setdefault_missing():
    omd = OrderedMultiDict()
    assert omd.setdefault('a') is None
    assert omd['a'] is None
    assert omd.setdefault('b', 2) == 2
    assert omd['b'] == 2


def test_setdefault_present():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.setdefault('a', 99) == 1


def test_clear():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []
    omd.add('x', 1)
    assert omd['x'] == 1


def test_fromkeys():
    omd = OrderedMultiDict.fromkeys(['a', 'b'])
    assert omd['a'] is None
    assert omd['b'] is None
    omd2 = OrderedMultiDict.fromkeys(['a'], 5)
    assert omd2['a'] == 5


def test_copy():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    cp = omd.copy()
    assert cp is not omd
    assert cp.getlist('a') == [1, 2]
    cp.add('a', 3)
    assert omd.getlist('a') == [1, 2]


def test_update_overwrites():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    omd.update({'a': 10})
    assert omd.getlist('a') == [10]
    assert omd['b'] == 2


def test_update_with_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    other = OrderedMultiDict([('a', 9), ('c', 4)])
    omd.update(other)
    assert omd.getlist('a') == [9]
    assert omd.getlist('b') == [3]
    assert omd.getlist('c') == [4]


def test_update_with_iterable():
    # The implementation deletes an existing key only on the FIRST
    # occurrence in the iterable; subsequent duplicates are add()-ed,
    # so 'a' ends up with [2, 4].
    omd = OrderedMultiDict([('a', 1)])
    omd.update([('a', 2), ('b', 3), ('a', 4)])
    assert omd.getlist('a') == [2, 4]
    assert omd.getlist('b') == [3]


def test_update_self_noop():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd.update(omd)
    assert omd.getlist('a') == [1, 2]


def test_update_kwargs():
    omd = OrderedMultiDict()
    omd.update({'a': 1}, b=2)
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_update_extend():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend([('a', 2), ('b', 3)])
    assert omd.getlist('a') == [1, 2]
    assert omd.getlist('b') == [3]


def test_update_extend_with_omd():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    other = OrderedMultiDict([('a', 10), ('a', 20)])
    omd.update_extend(other)
    assert omd.getlist('a') == [1, 2, 10, 20]


def test_update_extend_with_mapping():
    omd = OrderedMultiDict()
    omd.update_extend({'a': 1, 'b': 2})
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_update_extend_self():
    omd = OrderedMultiDict([('a', 1)])
    omd.update_extend(omd)
    assert omd.getlist('a') == [1, 1]


def test_setitem_overwrites():
    omd = OrderedMultiDict([('a', 1), ('a', 2)])
    omd['a'] = 3
    assert omd.getlist('a') == [3]


def test_delitem():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    del omd['a']
    assert 'a' not in omd
    assert omd.getlist('a') == []
    assert omd['b'] == 3
    assert omd.items(multi=True) == [('b', 3)]


def test_pop():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.pop('a') == 2
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 3)]


def test_pop_default():
    omd = OrderedMultiDict()
    assert omd.pop('missing', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        omd.pop('missing')


def test_popall():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert omd.popall('a') == [1, 2]
    assert 'a' not in omd


def test_popall_default():
    omd = OrderedMultiDict()
    assert omd.popall('missing', []) == []
    with pytest.raises(KeyError):
        omd.popall('missing')


def test_poplast():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.items(multi=True) == [('a', 1), ('b', 2)]


def test_poplast_removes_key():
    omd = OrderedMultiDict([('a', 1)])
    assert omd.poplast('a') == 1
    assert 'a' not in omd


def test_poplast_no_arg():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert omd.poplast() == 2
    assert omd.items(multi=True) == [('a', 1)]


def test_poplast_empty():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='E') == 'E'


def test_poplast_missing():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.poplast('a')
    assert omd.poplast('a', 'D') == 'D'


def test_eq_omd():
    assert OrderedMultiDict([('a', 1)]) == OrderedMultiDict([('a', 1)])
    assert OrderedMultiDict([('a', 1)]) != OrderedMultiDict([('a', 2)])
    assert OrderedMultiDict([('a', 1)]) != OrderedMultiDict([('b', 1)])
    # order matters for multi
    assert OrderedMultiDict([('a', 1), ('b', 2)]) != \
        OrderedMultiDict([('b', 2), ('a', 1)])


def test_eq_self():
    omd = OrderedMultiDict([('a', 1)])
    assert (omd == omd) is True


def test_eq_plain_dict():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    assert omd == {'a': 1, 'b': 2}
    assert omd != {'a': 1}
    assert omd != {'a': 2, 'b': 2}


def test_eq_non_dict():
    omd = OrderedMultiDict([('a', 1)])
    assert (omd == 5) is False
    assert (omd != 5) is True


def test_eq_other_mapping_missing_key():
    # covers the KeyError branch at lines 369-375 in __eq__
    class FakeMapping:
        def keys(self):
            return ['a']

        def __getitem__(self, k):
            raise KeyError(k)

    omd = OrderedMultiDict([('a', 1)])
    assert (omd == FakeMapping()) is False


def test_eq_other_mapping_getitem_keyerror():
    # A mapping with matching length and matching keys() but whose
    # __getitem__ raises KeyError triggers the except KeyError branch.
    class FakeMapping:
        def __len__(self):
            return 1

        def keys(self):
            return ['a']

        def __getitem__(self, k):
            raise KeyError(k)

    omd = OrderedMultiDict([('a', 1)])
    assert (omd == FakeMapping()) is False


def test_eq_other_mapping_len_mismatch():
    # if len(other) != len(self), the early return False path
    omd = OrderedMultiDict([('a', 1)])
    assert (omd == {'a': 1, 'b': 2}) is False


def test_eq_no_keys_attr_non_omd():
    # other is not OMD and has no .keys() -> return False (line 377)
    omd = OrderedMultiDict([('a', 1)])
    class NoKeys:
        def __len__(self):
            return 1
    assert (omd == NoKeys()) is False


def test_eq_omd_leftover_items():
    # line 364-367: leftovers path. Two OMDs with equal lengths that
    # differ after zip_longest comparison can't easily happen since
    # lengths must be equal, but we can exercise the "next(...)" checks
    # by making iteritems produce nothing extra. Here we assert the
    # positive path via fresh generators.
    a = OrderedMultiDict([('a', 1)])
    b = OrderedMultiDict([('a', 1)])
    # self is not other (fresh objects), lengths equal
    assert (a == b) is True


def test_eq_omd_leftover_branch():
    # Exercise the leftover branch (lines 364-367) of __eq__.
    # The comparison loop compares zip_longest iterators but only the
    # underlying dict __len__ is checked up front. We craft an OMD
    # subclass whose iteritems produces an extra item not reflected in
    # __len__, so that after the zip_longest drains, one iterator has a
    # leftover item.
    class TrickyOMD(OrderedMultiDict):
        def iteritems(self, multi=False):
            if multi:
                yield ('a', 1)
                yield ('a', 1)  # extra that will be consumed by zip_longest
                return
            yield from super().iteritems(multi=multi)

    a = TrickyOMD([('a', 1)])
    b = OrderedMultiDict([('a', 1)])
    # Both are OMDs; len(a) == len(b) == 1. iteritems for a yields two
    # equal items (multi=True), and the zip_longest consumes both plus
    # a's fillers... the leftover check should catch the mismatch.
    # Actually zip_longest pairs a's two items with b's one item and a
    # (None, None) filler -- the second has different values. But make
    # the extra identical so the loop doesn't catch it, forcing the
    # leftover check.
    class TrickyOMD2(OrderedMultiDict):
        _extra_done = False
        def iteritems(self, multi=False):
            if multi and not self._extra_done:
                self._extra_done = True
                yield ('a', 1)
                yield ('a', 1)
                return
            yield from super().iteritems(multi=multi)

    a = TrickyOMD2([('a', 1)])
    b = OrderedMultiDict([('a', 1)])
    # Due to the extra item the leftover check should kick in.
    result = a == b
    # Confirm iteritems in multi mode really had the extra: we build the
    # same check manually to know expected result.
    # Since b has only one item, the leftover next() on a's iterator is
    # not _MISSING => returns False.
    assert result is False


def test_ior():
    omd = OrderedMultiDict([('a', 1)])
    omd |= {'b': 2}
    assert omd['a'] == 1
    assert omd['b'] == 2


def test_sorted():
    omd = OrderedMultiDict(zip(range(3), range(3)))
    s = omd.sorted(reverse=True)
    assert s.items(multi=True) == [(2, 2), (1, 1), (0, 0)]


def test_sorted_key():
    omd = OrderedMultiDict(zip('hello', 'world'))
    s = omd.sorted(key=lambda i: i[1])
    assert s.items(multi=True) == \
        [('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')]


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


def test_inverted():
    omd = OMD([(0, 2), (1, 2)])
    inv = omd.inverted()
    assert inv.getlist(2) == [0, 1]
    assert inv.inverted().items(multi=True) == [(0, 2), (1, 2)]


def test_counts():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    counts = omd.counts()
    assert counts['a'] == 2
    assert counts['b'] == 1


def test_views():
    # KeysView/ValuesView/ItemsView iterate over the dict's *iterkeys*,
    # *itervalues*, and *items* methods — which for OMD default to
    # multi=False (most-recent value per key). So the values view only
    # exposes {2, 3} and the items view only the latest value per key.
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert set(omd.viewkeys()) == {'a', 'b'}
    assert set(omd.viewvalues()) == {2, 3}
    assert set(omd.viewitems()) == {('a', 2), ('b', 3)}


def test_pickle_roundtrip():
    omd = OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    roundtrip = pickle.loads(pickle.dumps(omd))
    assert roundtrip.items(multi=True) == omd.items(multi=True)


def test_getstate_setstate():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    state = omd.__getstate__()
    assert state == [('a', 1), ('a', 2), ('b', 3)]
    new = OrderedMultiDict()
    new.__setstate__(state)
    assert new.items(multi=True) == [('a', 1), ('a', 2), ('b', 3)]


def test_getstate_empty():
    omd = OrderedMultiDict()
    assert omd.__getstate__() == []


# ------------------------------------------------------------------ #
# FastIterOrderedMultiDict
# ------------------------------------------------------------------ #

def test_fastiter_basic():
    fmd = FastIterOrderedMultiDict()
    fmd['a'] = 1
    fmd['b'] = 2
    fmd.add('a', 3)
    assert fmd['a'] == 3
    assert fmd['b'] == 2
    assert fmd.getlist('a') == [1, 3]


def test_fastiter_iteration():
    # With multi=False, FastIterOrderedMultiDict iterates the skip-list
    # link (SNEXT), which for duplicate keys only visits the *first*
    # occurrence of each key — unlike base OMD, it does NOT return the
    # most-recent value. So this yields ('a', 1) not ('a', 3).
    fmd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(fmd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(fmd.iteritems()) == [('a', 1), ('b', 2)]
    assert list(fmd.iterkeys(multi=True)) == ['a', 'b', 'a']
    assert list(fmd.iterkeys()) == ['a', 'b']


def test_fastiter_reversed():
    fmd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert list(reversed(fmd)) == ['b', 'a']


def test_fastiter_remove_all():
    fmd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    del fmd['a']
    assert 'a' not in fmd
    assert fmd.items(multi=True) == [('b', 2)]


def test_fastiter_poplast():
    fmd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])
    assert fmd.poplast('a') == 3
    assert fmd.getlist('a') == [1]


def test_fastiter_multi_deletions():
    # Exercise FastIter _remove / _remove_all with multiple duplicate keys
    fmd = FastIterOrderedMultiDict([('a', 1), ('b', 2), ('a', 3), ('c', 4), ('a', 5)])
    assert list(fmd.iterkeys(multi=True)) == ['a', 'b', 'a', 'c', 'a']
    assert fmd.poplast('a') == 5
    assert list(fmd.iterkeys(multi=True)) == ['a', 'b', 'a', 'c']
    del fmd['a']
    assert list(fmd.iterkeys(multi=True)) == ['b', 'c']
    # remaining keys still iterate correctly
    assert list(fmd.iteritems(multi=True)) == [('b', 2), ('c', 4)]
    assert list(reversed(fmd)) == ['c', 'b']


def test_fastiter_remove_middle_dupe():
    # Removing the last (most recent) item of a key that skips another
    # ensures the skip-list repair branch is hit.
    fmd = FastIterOrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert fmd.poplast('a') == 2
    assert list(fmd.iterkeys(multi=True)) == ['a', 'b']
    assert fmd.poplast('a') == 1
    assert list(fmd.iterkeys(multi=True)) == ['b']


def test_fastiter_reversed_dedups():
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('b', 2)
    fmd.add('a', 3)
    fmd.add('c', 4)
    assert list(reversed(fmd)) == ['c', 'b', 'a']


def test_fastiter_reversed_with_skipped():
    # build state where reverse iteration hits the "was skipped" branch
    fmd = FastIterOrderedMultiDict()
    fmd.add('x', 1)
    fmd.add('x', 2)
    fmd.add('y', 3)
    fmd.add('x', 4)
    rev = list(reversed(fmd))
    assert 'x' in rev
    assert 'y' in rev
    # x should appear exactly once in the reversed dedup output
    assert rev.count('x') == 1


def test_fastiter_remove_skipped_branch():
    # Exercise _remove branch where cell[PREV][SPREV][SNEXT] is cell.
    # Build a skip-list with duplicates such that removing the most
    # recent item routes through the `if cell[PREV][SPREV][SNEXT] is cell`
    # branch (line 712-713).
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('a', 2)
    # remove most-recent (2) for 'a'
    assert fmd.poplast('a') == 2
    assert fmd.getlist('a') == [1]
    assert list(fmd.iterkeys(multi=True)) == ['a']


def test_fastiter_remove_all_skipped_branch():
    # Exercise _remove_all branches on lines 723-726.
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('b', 2)
    fmd.add('a', 3)
    fmd.add('a', 4)
    del fmd['a']
    assert 'a' not in fmd
    assert list(fmd.iteritems(multi=True)) == [('b', 2)]


def test_fastiter_remove_all_full_branch():
    # Exercise _remove_all with the `if cell[PREV][SPREV][SNEXT] is cell`
    # branch (lines 723-724) and the `elif cell[SNEXT] is cell[NEXT]`
    # branch (lines 725-726) by constructing skip states with duplicates
    # at various positions.
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('a', 2)
    fmd.add('b', 3)
    fmd.add('a', 4)
    # first a skips, later a's chain differently
    del fmd['a']
    assert list(fmd.iteritems(multi=True)) == [('b', 3)]

    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('b', 2)
    fmd.add('b', 3)
    fmd.add('a', 4)
    del fmd['b']
    assert list(fmd.iteritems(multi=True)) == [('a', 1), ('a', 4)]


def test_fastiter_remove_alternate_branch():
    # Exercise the `elif cell[SNEXT] is cell[NEXT]` branch of _remove
    # (lines 714-715).
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('b', 2)
    fmd.add('a', 3)
    fmd.add('a', 4)
    assert fmd.poplast('a') == 4
    assert list(fmd.iteritems(multi=True)) == [('a', 1), ('b', 2), ('a', 3)]
    assert fmd.poplast('a') == 3
    assert list(fmd.iteritems(multi=True)) == [('a', 1), ('b', 2)]


def test_fastiter_reversed_break_at_root():
    # Exercise the reverse-iteration "skip to root" break at lines 753-754.
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('a', 2)
    fmd.add('b', 3)
    rev = list(reversed(fmd))
    # 'a' should appear once, 'b' once
    assert sorted(rev) == ['a', 'b']


def test_fastiter_reversed_hits_root_break():
    # Force the branch at line 753-754 where curr becomes root during
    # the SPREV walk. Removing a chain that leaves the skip list
    # pointing through root should exercise it.
    fmd = FastIterOrderedMultiDict()
    fmd.add('a', 1)
    fmd.add('a', 2)
    fmd.add('b', 3)
    fmd.add('c', 4)
    # poplast on 'a' removes the later 'a' but leaves the skip intact
    assert fmd.poplast('a') == 2
    rev = list(reversed(fmd))
    assert sorted(rev) == ['a', 'b', 'c']


# ------------------------------------------------------------------ #
# OneToOne
# ------------------------------------------------------------------ #

def test_onetoone_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'
    assert len(oto) == 2
    assert len(oto.inv) == 2


def test_onetoone_inverse_overwrite():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_onetoone_setitem_overwrite_both():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['a'] = 2
    assert oto['a'] == 2
    assert oto.inv[2] == 'a'
    assert 'b' not in oto


def test_onetoone_dupe_values_collapse():
    oto = OneToOne({'a': 1, 'b': 1})
    # values collide, so the later key wins in both directions
    assert len(oto) == 1
    assert len(oto.inv) == 1


def test_onetoone_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1


def test_onetoone_unique_dup_raises():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_onetoone_unique_across_inputs():
    a_dict = {'a': 2}
    with pytest.raises(ValueError):
        OneToOne.unique(a_dict, b=2)


def test_onetoone_unique_error_message():
    with pytest.raises(ValueError) as exc:
        OneToOne.unique({'a': 1, 'b': 1})
    msg = str(exc.value)
    assert 'expected unique values' in msg
    # the dupes mapping should map value -> list of keys
    assert "1: ['a', 'b']" in msg or "1: ['b', 'a']" in msg


def test_onetoone_delitem():
    oto = OneToOne({'a': 1, 'b': 2})
    del oto['a']
    assert 'a' not in oto
    assert 1 not in oto.inv


def test_onetoone_clear():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_onetoone_copy():
    oto = OneToOne({'a': 1, 'b': 2})
    cp = oto.copy()
    assert cp == {'a': 1, 'b': 2}
    assert cp.inv[1] == 'a'
    cp['c'] = 3
    assert 'c' not in oto


def test_onetoone_pop():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto.pop('a') == 1
    assert 'a' not in oto
    assert 1 not in oto.inv
    assert oto.pop('missing', 'D') == 'D'
    with pytest.raises(KeyError):
        oto.pop('missing')


def test_onetoone_popitem():
    oto = OneToOne({'a': 1, 'b': 2})
    k, v = oto.popitem()
    assert k in ('a', 'b')
    assert v not in oto.inv
    assert k not in oto


def test_onetoone_setdefault():
    oto = OneToOne()
    assert oto.setdefault('a', 1) == 1
    assert oto.inv[1] == 'a'
    assert oto.setdefault('a', 99) == 1


def test_onetoone_update_dict():
    oto = OneToOne({'a': 1})
    oto.update({'b': 2})
    assert oto['b'] == 2
    assert oto.inv[2] == 'b'


def test_onetoone_update_iterable():
    oto = OneToOne()
    oto.update([('a', 1), ('b', 2)])
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_onetoone_update_kwargs():
    oto = OneToOne()
    oto.update({'a': 1}, b=2)
    assert oto['b'] == 2
    assert oto.inv[1] == 'a'


def test_onetoone_hash_unhashable_value():
    oto = OneToOne()
    with pytest.raises(TypeError):
        oto['a'] = [1, 2, 3]


def test_onetoone_repr():
    oto = OneToOne({'a': 1})
    assert repr(oto).startswith('OneToOne(')


# ------------------------------------------------------------------ #
# ManyToMany
# ------------------------------------------------------------------ #

def test_manytomany_basic():
    mtm = ManyToMany()
    mtm.add('a', 1)
    mtm.add('a', 2)
    mtm.add('b', 1)
    assert mtm['a'] == frozenset({1, 2})
    assert mtm['b'] == frozenset({1})
    assert mtm.inv[1] == frozenset({'a', 'b'})
    assert len(mtm) == 2
    assert 'a' in mtm
    assert 'z' not in mtm


def test_manytomany_getitem_missing():
    mtm = ManyToMany()
    with pytest.raises(KeyError):
        mtm['missing']


def test_manytomany_get():
    mtm = ManyToMany()
    assert mtm.get('missing') == frozenset()
    assert mtm.get('missing', 'D') == 'D'
    mtm.add('a', 1)
    assert mtm.get('a') == frozenset({1})


def test_manytomany_init_iterable():
    mtm = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
    assert mtm['a'] == frozenset({1, 2})
    assert mtm.inv[1] == frozenset({'a', 'b'})


def test_manytomany_setitem():
    mtm = ManyToMany()
    mtm['a'] = {1, 2}
    assert mtm['a'] == frozenset({1, 2})
    mtm['a'] = {2, 3}
    assert mtm['a'] == frozenset({2, 3})
    assert 1 not in mtm.inv


def test_manytomany_delitem():
    mtm = ManyToMany([('a', 1), ('a', 2), ('b', 1)])
    del mtm['a']
    assert 'a' not in mtm
    assert 2 not in mtm.inv
    assert mtm.inv[1] == frozenset({'b'})


def test_manytomany_update():
    mtm = ManyToMany([('a', 1)])
    mtm.update([('a', 2), ('b', 1)])
    assert mtm['a'] == frozenset({1, 2})
    assert mtm['b'] == frozenset({1})


def test_manytomany_update_from_instance():
    a = ManyToMany([('a', 1)])
    b = ManyToMany([('b', 2), ('a', 3)])
    a.update(b)
    assert a['a'] == frozenset({1, 3})
    assert a['b'] == frozenset({2})
    assert a.inv[1] == frozenset({'a'})


def test_manytomany_update_mapping():
    # When given a mapping, ManyToMany.update adds each value via add(),
    # which requires the value to be hashable. A list value is unhashable.
    mtm = ManyToMany()
    with pytest.raises(TypeError):
        mtm.update({'a': [1, 2]})
    # Using a hashable value per key works:
    mtm.update({'b': 5})
    assert mtm['b'] == frozenset({5})


def test_manytomany_remove():
    mtm = ManyToMany([('a', 1), ('a', 2)])
    mtm.remove('a', 1)
    assert mtm['a'] == frozenset({2})
    mtm.remove('a', 2)
    assert 'a' not in mtm


def test_manytomany_replace():
    mtm = ManyToMany([('a', 1), ('a', 2)])
    mtm.replace('a', 'b')
    assert 'a' not in mtm
    assert mtm['b'] == frozenset({1, 2})
    assert mtm.inv[1] == frozenset({'b'})


def test_manytomany_replace_missing():
    mtm = ManyToMany()
    mtm.replace('missing', 'x')
    assert len(mtm) == 0


def test_manytomany_iteritems_keys_iter():
    mtm = ManyToMany([('a', 1)])
    items = list(mtm.iteritems())
    assert items == [('a', 1)]
    assert set(mtm.keys()) == {'a'}
    assert list(iter(mtm)) == ['a']


def test_manytomany_eq():
    a = ManyToMany([('a', 1)])
    b = ManyToMany([('a', 1)])
    assert a == b
    assert a != ManyToMany([('a', 2)])


def test_manytomany_eq_non_mtm():
    a = ManyToMany([('a', 1)])
    assert (a == {'a': 1}) is False


def test_manytomany_repr():
    mtm = ManyToMany([('a', 1)])
    assert repr(mtm).startswith('ManyToMany(')


def test_manytomany_remove_clears_inv_entry():
    # line 992: after removing a forward pair, if the inverse set becomes
    # empty, the inverse key is removed entirely.
    mtm = ManyToMany([('a', 1)])
    mtm.remove('a', 1)
    assert 'a' not in mtm
    assert 1 not in mtm.inv.data


def test_manytomany_update_instance_partial_overlap():
    # Exercise the branch where the key is already present in inv data
    # (line 966-970), so the sets are merged.
    a = ManyToMany([('a', 1), ('x', 5)])
    b = ManyToMany([('a', 2), ('x', 5)])
    a.update(b)
    # 'a' merged forward
    assert a['a'] == frozenset({1, 2})
    # inv entries for existing shared value 5 and new value 2 both merge
    assert a.inv[5] == frozenset({'x'})
    assert a.inv[2] == frozenset({'a'})


def test_manytomany_remove_keeps_other_inv_entry():
    # touch the non-clearing path of remove: inv.data[val] still nonempty.
    mtm = ManyToMany([('a', 1), ('b', 1)])
    mtm.remove('a', 1)
    assert mtm.inv[1] == frozenset({'b'})
    mtm.remove('b', 1)
    assert 1 not in mtm.inv.data


def test_manytomany_setitem_removes_old_remove_path():
    # When changing 'a' to a smaller set, the removed values trigger
    # remove(); if other keys still point at that value, the inv entry
    # stays. This exercises the non-empty branch of remove() (line 992).
    mtm = ManyToMany([('a', 1), ('b', 1), ('a', 2)])
    mtm['a'] = {2}
    assert mtm.inv[1] == frozenset({'b'})
    assert mtm.inv[2] == frozenset({'a'})


# ------------------------------------------------------------------ #
# subdict
# ------------------------------------------------------------------ #

def test_subdict_identity():
    assert subdict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}


def test_subdict_keep():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']) == {'a': 1, 'c': 3}


def test_subdict_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c']) == {'a': 1}


def test_subdict_keep_nonexistent():
    assert subdict({'a': 1}, keep=['a', 'z']) == {'a': 1}


def test_subdict_type_preserved_omd():
    omd = OrderedMultiDict([('a', 1), ('b', 2)])
    result = subdict(omd, keep=['a'])
    assert isinstance(result, OrderedMultiDict)
    assert result['a'] == 1


def test_subdict_keep_and_drop():
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'b'], drop=['b']) == {'a': 1}


# ------------------------------------------------------------------ #
# FrozenDict
# ------------------------------------------------------------------ #

def test_frozendict_basic():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert fd['a'] == 1
    assert dict(fd) == {'a': 1, 'b': 2}


def test_frozendict_hashable():
    fd = FrozenDict({'a': 1})
    h = hash(fd)
    assert isinstance(h, int)
    d = {fd: 'value'}
    assert d[FrozenDict({'a': 1})] == 'value'


def test_frozendict_hash_cached():
    fd = FrozenDict({'a': 1})
    assert hash(fd) == hash(fd)


def test_frozendict_unhashable_content():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)
    assert issubclass(FrozenHashError, TypeError)


def test_frozendict_unhashable_repeated():
    # second hash() must re-raise the cached FrozenHashError
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)
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


def test_frozendict_immutable_clear():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.clear()


def test_frozendict_immutable_pop():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.pop('a')


def test_frozendict_immutable_setdefault():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.setdefault('b', 2)


def test_frozendict_immutable_popitem():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd.popitem()


def test_frozendict_immutable_error_message():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError) as exc:
        fd['b'] = 2
    assert 'FrozenDict object is immutable' in str(exc.value)


def test_frozendict_ior():
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        fd |= {'b': 2}


def test_frozendict_updated():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated({'b': 2})
    assert fd2 == {'a': 1, 'b': 2}
    assert isinstance(fd2, FrozenDict)
    assert dict(fd) == {'a': 1}


def test_frozendict_updated_kwargs():
    fd = FrozenDict({'a': 1})
    fd2 = fd.updated(b=2)
    assert fd2 == {'a': 1, 'b': 2}
    assert isinstance(fd2, FrozenDict)


def test_frozendict_fromkeys():
    fd = FrozenDict.fromkeys(['a', 'b'], 0)
    assert fd == {'a': 0, 'b': 0}
    assert isinstance(fd, FrozenDict)


def test_frozendict_fromkeys_default_none():
    fd = FrozenDict.fromkeys(['a'])
    assert fd['a'] is None
    assert isinstance(fd, FrozenDict)


def test_frozendict_copy_returns_self():
    fd = FrozenDict({'a': 1})
    assert copy.copy(fd) is fd


def test_frozendict_repr():
    fd = FrozenDict({'a': 1})
    assert repr(fd) == "FrozenDict({'a': 1})"


def test_frozendict_pickle():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert pickle.loads(pickle.dumps(fd)) == fd
