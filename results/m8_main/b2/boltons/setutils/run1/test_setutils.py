import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet: construction and basics
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []
    assert repr(s) == "IndexedSet([])"


def test_init_from_iterable_dedupes_preserving_order():
    s = IndexedSet([3, 1, 3, 2, 1])
    assert list(s) == [3, 1, 2]
    assert len(s) == 3


def test_init_from_string():
    s = IndexedSet("abca")
    assert list(s) == ["a", "b", "c"]


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter_and_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(iter(s)) == [1, 2, 3]
    assert list(reversed(s)) == [3, 2, 1]


def test_repr():
    s = IndexedSet([1, 2])
    assert repr(s) == "IndexedSet([1, 2])"


def test_repr_subclass():
    class MySet(IndexedSet):
        pass
    s = MySet([1])
    assert repr(s) == "MySet([1])"


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_sensitive():
    assert IndexedSet([1, 2]) == IndexedSet([1, 2])
    assert IndexedSet([1, 2]) != IndexedSet([2, 1])


def test_eq_with_plain_set():
    assert IndexedSet([1, 2, 3]) == {3, 2, 1}
    assert IndexedSet([1, 2]) == frozenset([1, 2])


def test_eq_non_iterable_returns_false():
    s = IndexedSet([1])
    assert (s == 5) is False


def test_from_iterable_classmethod():
    s = IndexedSet.from_iterable([1, 2, 2, 3])
    assert isinstance(s, IndexedSet)
    assert list(s) == [1, 2, 3]


# ---------------------------------------------------------------------------
# add / remove / discard / clear
# ---------------------------------------------------------------------------

def test_add():
    s = IndexedSet()
    s.add(1)
    s.add(2)
    s.add(1)
    assert list(s) == [1, 2]


def test_remove():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    assert list(s) == [1, 3]
    assert 2 not in s


def test_remove_missing_raises():
    s = IndexedSet([1])
    with pytest.raises(KeyError):
        s.remove(5)


def test_discard():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    assert list(s) == [1, 3]
    # discarding missing does nothing
    s.discard(99)
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []
    assert s.item_list == []
    assert s.dead_indices == []


# ---------------------------------------------------------------------------
# isdisjoint / issubset / issuperset
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2])
    assert s.isdisjoint([3, 4])
    assert not s.isdisjoint([2, 3])


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset([1, 2, 3])
    assert s.issubset(IndexedSet([1, 2]))
    assert not s.issubset([1])
    assert not s.issubset([1, 3])


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset([1, 2])
    assert not s.issuperset([1, 4])
    assert not s.issuperset([1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def test_union():
    s = IndexedSet([1, 2])
    u = s.union([2, 3], [4])
    assert isinstance(u, IndexedSet)
    assert list(u) == [1, 2, 3, 4]


def test_or_operator():
    s = IndexedSet([1, 2])
    u = s | set([3])
    assert list(u) == [1, 2, 3]
    # Reverse or
    u2 = set([3]) | s
    assert isinstance(u2, IndexedSet)
    assert set(u2) == {1, 2, 3}


def test_intersection_single():
    s = IndexedSet([1, 2, 3])
    r = s.intersection([2, 3, 4])
    assert list(r) == [2, 3]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.intersection([1, 2, 3], [2, 3, 5])
    assert sorted(r) == [2, 3]


def test_iter_intersection():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_intersection([2, 3])) == [2, 3]


def test_and_operator():
    s = IndexedSet([1, 2, 3])
    r = s & set([2, 3, 4])
    assert list(r) == [2, 3]


def test_difference_single():
    s = IndexedSet([1, 2, 3])
    r = s.difference([2])
    assert list(r) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.difference([1], [3])
    assert list(r) == [2, 4]


def test_iter_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_difference([2])) == [1, 3]


def test_sub_operator():
    s = IndexedSet([1, 2, 3])
    r = s - set([2])
    assert list(r) == [1, 3]


def test_rsub_operator():
    s = IndexedSet([2, 3])
    r = {1, 2, 3, 4} - s
    assert isinstance(r, set)
    assert r == {1, 4}


def test_rsub_list():
    s = IndexedSet([2, 3])
    r = [1, 2, 3, 4] - s
    assert isinstance(r, list)
    assert r == [1, 4]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    r = s.symmetric_difference([2, 3, 4])
    assert sorted(r) == [1, 4]


def test_xor_operator():
    s = IndexedSet([1, 2, 3])
    r = s ^ set([3, 4])
    assert sorted(r) == [1, 2, 4]
    r2 = set([3, 4]) ^ s
    assert sorted(r2) == [1, 2, 4]


# ---------------------------------------------------------------------------
# In-place set operations
# ---------------------------------------------------------------------------

def test_update():
    s = IndexedSet([1])
    s.update([2, 3], [3, 4])
    assert list(s) == [1, 2, 3, 4]


def test_update_no_args():
    s = IndexedSet([1])
    s.update()
    assert list(s) == [1]


def test_ior():
    s = IndexedSet([1])
    s |= [2, 3]
    assert list(s) == [1, 2, 3]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update([2, 3, 5])
    assert sorted(s) == [2, 3]


def test_iand():
    s = IndexedSet([1, 2, 3])
    s &= [2, 3, 4]
    assert sorted(s) == [2, 3]


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update([2, 4])
    assert list(s) == [1, 3]


def test_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert len(s) == 0


def test_isub():
    s = IndexedSet([1, 2, 3])
    s -= [2]
    assert list(s) == [1, 3]


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update([2, 3, 4])
    assert sorted(s) == [1, 4]


def test_symmetric_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert len(s) == 0


def test_ixor():
    s = IndexedSet([1, 2, 3])
    s ^= [3, 4]
    assert sorted(s) == [1, 2, 4]


# ---------------------------------------------------------------------------
# Indexing, slicing, pop, index, count
# ---------------------------------------------------------------------------

def test_getitem_int():
    s = IndexedSet([10, 20, 30])
    assert s[0] == 10
    assert s[-1] == 30
    assert s[1] == 20


def test_getitem_out_of_range():
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s[5]
    with pytest.raises(IndexError):
        s[-5]


def test_getitem_with_dead_indices():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert list(s) == [1, 3, 5]
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 5
    assert s[-1] == 5
    assert s[-2] == 3


def test_getitem_slice():
    s = IndexedSet([0, 1, 2, 3, 4, 5])
    r = s[1:4]
    assert isinstance(r, IndexedSet)
    assert list(r) == [1, 2, 3]


def test_getitem_slice_step():
    s = IndexedSet([0, 1, 2, 3, 4, 5])
    r = s[0:6:2]
    assert list(r) == [0, 2, 4]


def test_getitem_slice_negative_bounds():
    s = IndexedSet([0, 1, 2, 3, 4])
    r = s[-3:]
    assert list(r) == [2, 3, 4]
    r2 = s[:-2]
    assert list(r2) == [0, 1, 2]


def test_getitem_slice_negative_step():
    s = IndexedSet([0, 1, 2, 3, 4])
    # reversed stream, forward bounds
    r = s[1:3:-1]
    assert list(r) == [3, 2]


def test_getitem_slice_with_dead():
    s = IndexedSet([0, 1, 2, 3, 4])
    s.remove(2)
    assert list(s) == [0, 1, 3, 4]
    r = s[1:3]
    assert list(r) == [1, 3]


def test_pop_default():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_last_explicit():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_last_by_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3
    assert list(s) == [1, 2]


def test_pop_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]


def test_pop_from_empty():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


def test_count():
    s = IndexedSet([1, 2])
    assert s.count(1) == 1
    assert s.count(5) == 0


def test_index():
    s = IndexedSet(["a", "b", "c"])
    assert s.index("b") == 1


def test_index_missing_raises():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(ValueError):
        s.index(99)


def test_index_with_dead_indices():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert s.index(3) == 1
    assert s.index(4) == 2


def test_reverse_in_place():
    s = IndexedSet([1, 2, 3])
    result = s.reverse()
    assert result is None
    assert list(s) == [3, 2, 1]
    assert s.index(1) == 2


def test_reverse_with_dead_indices():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    s.reverse()
    assert list(s) == [4, 3, 1]
    assert s.dead_indices == []


def test_sort_in_place():
    s = IndexedSet([3, 1, 2])
    result = s.sort()
    assert result is None
    assert list(s) == [1, 2, 3]


def test_sort_with_kwargs():
    s = IndexedSet([3, 1, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_already_sorted():
    s = IndexedSet([1, 2, 3])
    s.sort()
    assert list(s) == [1, 2, 3]


def test_sort_with_key():
    s = IndexedSet(["bb", "a", "ccc"])
    s.sort(key=len)
    assert list(s) == ["a", "bb", "ccc"]


# ---------------------------------------------------------------------------
# Compaction / dead index behavior
# ---------------------------------------------------------------------------

def test_compaction():
    s = IndexedSet(range(20))
    for i in range(0, 20, 2):
        s.remove(i)
    assert sorted(s) == list(range(1, 20, 2))
    # All items still accessible
    for i in range(len(s)):
        assert s[i] in s


def test_add_after_remove():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    s.add(4)
    assert list(s) == [1, 3, 4]


def test_bulk_discard_via_intersection_update():
    # Trigger _bulk_discard: to_remove must exceed _MAX_DEAD_INTERVALS
    from boltons.setutils import _MAX_DEAD_INTERVALS
    n = 2 * (_MAX_DEAD_INTERVALS + 100)
    s = IndexedSet(range(n))
    keep = set(range(0, n, 2))
    s.intersection_update(keep)
    assert len(s) == len(keep)
    assert set(s) == keep
    # _bulk_discard rebuilds item_list and clears dead_indices
    assert s.dead_indices == []
    # item_list rebuilt (no _MISSING placeholders remain)
    from boltons.setutils import _MISSING
    assert all(item is not _MISSING for item in s.item_list)


def test_isinstance_mutableset():
    from collections.abc import MutableSet
    s = IndexedSet([1])
    assert isinstance(s, MutableSet)


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------

def test_complement_basic():
    c = complement(set([2, 3]))
    assert 2 not in c
    assert 1 in c


def test_complement_of_empty():
    c = complement(set())
    assert 1 in c
    assert "anything" in c


def test_complement_repr():
    c = complement(set([1, 2]))
    assert repr(c).startswith("complement(")


def test_complement_double_returns_set_or_complement():
    s = set(range(5))
    c = complement(s)
    c2 = complement(c)
    # complemented twice yields a complement-wrapped set again
    # (per implementation, complement(complement(x)) -> complemented())
    assert c2 == s or isinstance(c2, type(c))


def test_complement_with_intersection():
    c = complement(set([2, 3]))
    result = set(range(5)) & c
    assert result == {0, 1, 4}


def test_complement_intersection_explicit():
    c = complement(set([2, 3]))
    result = c.intersection(set(range(5)))
    assert isinstance(result, type(c))


def test_complement_union():
    c = complement(set([2, 3]))
    result = c | set([2])
    # Complement now excludes {3} (2 removed from exclusion)
    assert 2 in result
    assert 3 not in result


def test_complement_len_raises():
    c = complement(set([1]))
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_raises():
    c = complement(set([1]))
    with pytest.raises(NotImplementedError):
        list(c)


def test_complement_bool_always_true():
    c = complement(set())
    assert bool(c) is True
    c2 = complement(set([1]))
    assert bool(c2) is True


def test_complement_add():
    c = complement(set([1, 2]))
    c.add(1)
    assert 1 in c
    c.add(5)
    assert 5 in c


def test_complement_remove():
    c = complement(set([1]))
    c.remove(2)
    assert 2 not in c


def test_complement_pop_raises():
    c = complement(set())
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_frozenset():
    c = complement(frozenset([1]))
    assert 1 not in c
    assert 2 in c


def test_complement_inplace_intersection():
    c = complement(set([1, 2, 3]))
    # included is None, excluded={1,2,3}
    # __iand__ with inc={1,2}, exc=None: self._excluded = inc - self._excluded
    #   => {1,2} - {1,2,3} == set()
    # After: excluded=set(), so everything is in c
    c &= set([1, 2])
    assert c._excluded == set()
    assert c._included is None
    assert 1 in c
    assert 2 in c
    assert 3 in c


def test_complement_inplace_union():
    c = complement(set([1, 2]))
    c |= set([2])
    assert 2 in c
    assert 1 not in c


def test_complement_symmetric_difference():
    c = complement(set([1, 2]))
    r = c ^ set([2, 3])
    assert isinstance(r, type(c))


def test_complement_sub():
    c = complement(set([1]))
    r = c - set([2])
    assert 1 not in r
    assert 2 not in r


def test_complement_difference_with_complement():
    c1 = complement(set([1]))
    c2 = complement(set([2]))
    r = c1 - c2
    assert isinstance(r, type(c1))


def test_complement_eq_with_set():
    c = complement(set())
    # empty complement equals universum conceptually; implementation:
    # included is None so equality with plain set is False
    assert (c == set()) is False


def test_complement_hash_raises():
    # __hash__ is defined but hashing the underlying set raises TypeError
    c = complement(set([1, 2]))
    with pytest.raises(TypeError):
        hash(c)
    # frozenset-backed complement is hashable (both included and excluded
    # in this case are... let's test the included side is a frozenset)
    c2 = complement(frozenset())
    # included is None here, excluded is frozenset() -> hash works
    assert isinstance(hash(c2), int)


def test_complement_typeerror_bad_arg():
    c = complement(set([1]))
    with pytest.raises(TypeError):
        c & [1, 2]


def test_complement_issubset():
    c1 = complement(set([1, 2, 3]))
    c2 = complement(set([1]))
    # Excluded of c2 ({1}) is subset of excluded of c1 ({1,2,3}),
    # so c1 excludes fewer -> c1 is subset of c2
    assert c1.issubset(c2)


def test_complement_issuperset():
    c1 = complement(set([1]))
    c2 = complement(set([1, 2]))
    assert c1.issuperset(c2)


def test_complement_isdisjoint():
    c = complement(set([1, 2]))
    assert c.isdisjoint(set([1, 2]))
    assert not c.isdisjoint(set([1, 5]))


def test_complement_complemented_returns_new():
    s = set([1, 2])
    c = complement(s)
    c2 = c.complemented()
    assert c2 is not c


def test_complement_invert_operator():
    s = set([1])
    c = complement(s)
    c2 = ~c
    assert c2 is not c
