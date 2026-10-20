import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet basics
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []
    assert repr(s) == "IndexedSet([])"


def test_init_dedup_order():
    s = IndexedSet([3, 1, 2, 1, 3, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_init_from_string():
    s = IndexedSet("freecreditreport.com")
    assert list(s) == list("frecditpo.m")


def test_repr():
    s = IndexedSet([1, 2, 3])
    assert repr(s) == "IndexedSet([1, 2, 3])"


def test_from_iterable():
    s = IndexedSet.from_iterable(range(3))
    assert isinstance(s, IndexedSet)
    assert list(s) == [0, 1, 2]


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter_and_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(s) == [1, 2, 3]
    assert list(reversed(s)) == [3, 2, 1]


# ---------------------------------------------------------------------------
# Indexing / slicing
# ---------------------------------------------------------------------------

def test_getitem_int():
    s = IndexedSet([10, 20, 30])
    assert s[0] == 10
    assert s[1] == 20
    assert s[-1] == 30
    assert s[-3] == 10


def test_getitem_out_of_range():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(IndexError):
        s[3]
    with pytest.raises(IndexError):
        s[-4]


def test_getitem_slice():
    s = IndexedSet([0, 1, 2, 3, 4])
    assert list(s[1:3]) == [1, 2]
    assert list(s[:2]) == [0, 1]
    assert list(s[3:]) == [3, 4]
    assert list(s[::2]) == [0, 2, 4]
    assert isinstance(s[1:3], IndexedSet)


def test_getitem_slice_negative():
    s = IndexedSet([0, 1, 2, 3, 4])
    assert list(s[-2:]) == [3, 4]
    assert list(s[:-2]) == [0, 1, 2]


def test_getitem_negative_step():
    s = IndexedSet([0, 1, 2, 3, 4])
    assert list(s[::-1]) == [4, 3, 2, 1, 0]


def test_iter_slice_negative_start_stop():
    s = IndexedSet([0, 1, 2, 3, 4])
    assert list(s.iter_slice(-2, None)) == [3, 4]


def test_getitem_after_dead_indices():
    s = IndexedSet(range(6))
    s.remove(1)
    s.remove(3)
    assert list(s) == [0, 2, 4, 5]
    assert s[0] == 0
    assert s[1] == 2
    assert s[2] == 4
    assert s[3] == 5


# ---------------------------------------------------------------------------
# add / remove / discard / clear
# ---------------------------------------------------------------------------

def test_add():
    s = IndexedSet()
    s.add(1)
    s.add(2)
    s.add(1)
    assert list(s) == [1, 2]
    assert len(s) == 2


def test_remove():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    assert list(s) == [1, 3]
    assert 2 not in s


def test_remove_missing():
    s = IndexedSet([1, 2])
    with pytest.raises(KeyError):
        s.remove(5)


def test_discard():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    s.discard(99)
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []
    assert s.item_list == []


def test_remove_all_triggers_empty():
    s = IndexedSet([1, 2, 3])
    s.remove(1)
    s.remove(2)
    s.remove(3)
    assert len(s) == 0
    assert list(s) == []


# ---------------------------------------------------------------------------
# set operations (non-mutating)
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint([4, 5]) is True
    assert s.isdisjoint([3, 4]) is False


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset([1, 2, 3]) is True
    assert s.issubset([1]) is False


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset([1, 2]) is True
    assert s.issuperset([1, 5]) is False


def test_union():
    s = IndexedSet([1, 2])
    r = s.union([2, 3], [4])
    assert isinstance(r, IndexedSet)
    assert list(r) == [1, 2, 3, 4]


def test_union_operator():
    s = IndexedSet([1, 2])
    r = s | [2, 3]
    assert list(r) == [1, 2, 3]


def test_intersection():
    s = IndexedSet([1, 2, 3, 4])
    r = s.intersection([2, 3, 5])
    assert list(r) == [2, 3]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.intersection([2, 3, 4], [3, 4, 5])
    assert list(r) == [3, 4]


def test_intersection_operator():
    s = IndexedSet([1, 2, 3])
    r = s & [2, 3, 4]
    assert list(r) == [2, 3]


def test_difference():
    s = IndexedSet([1, 2, 3, 4])
    r = s.difference([2, 4])
    assert list(r) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.difference([2], [4])
    assert list(r) == [1, 3]


def test_difference_operator():
    s = IndexedSet([1, 2, 3])
    r = s - [2]
    assert list(r) == [1, 3]


def test_rsub():
    s = IndexedSet([1, 2])
    r = [1, 2, 3] - s
    assert r == [3]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    r = s.symmetric_difference([2, 3, 4])
    assert set(r) == {1, 4}


def test_symmetric_difference_operator():
    s = IndexedSet([1, 2, 3])
    r = s ^ [2, 3, 4]
    assert set(r) == {1, 4}


def test_iter_intersection():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.iter_intersection([2, 3, 4], [3, 4])) == [3, 4]


def test_iter_difference():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.iter_difference([2], [4])) == [1, 3]


# ---------------------------------------------------------------------------
# in-place set operations
# ---------------------------------------------------------------------------

def test_update():
    s = IndexedSet([1, 2])
    s.update([2, 3], [4])
    assert list(s) == [1, 2, 3, 4]


def test_update_no_args():
    s = IndexedSet([1, 2])
    s.update()
    assert list(s) == [1, 2]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update([2, 3, 4, 5])
    assert list(s) == [2, 3, 4]


def test_intersection_update_operator():
    s = IndexedSet([1, 2, 3, 4])
    s &= [2, 3]
    assert list(s) == [2, 3]


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update([2, 4])
    assert list(s) == [1, 3]


def test_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert len(s) == 0


def test_difference_update_operator():
    s = IndexedSet([1, 2, 3])
    s -= [2]
    assert list(s) == [1, 3]


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update([3, 4])
    assert set(s) == {1, 2, 4}


def test_symmetric_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert len(s) == 0


def test_symmetric_difference_update_operator():
    s = IndexedSet([1, 2, 3])
    s ^= [3, 4]
    assert set(s) == {1, 2, 4}


def test_ior():
    s = IndexedSet([1])
    s |= [2, 3]
    assert list(s) == [1, 2, 3]


def test_bulk_discard_paths():
    # exceed _MAX_DEAD_INTERVALS to trigger _bulk_discard
    s = IndexedSet(range(2000))
    s.intersection_update(range(0, 2000, 2))
    assert list(s) == list(range(0, 2000, 2))


def test_bulk_difference_update():
    s = IndexedSet(range(2000))
    s.difference_update(range(0, 2000, 2))
    assert list(s) == list(range(1, 2000, 2))


# ---------------------------------------------------------------------------
# list-like ops
# ---------------------------------------------------------------------------

def test_pop_default():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(0) == 1
    assert list(s) == [2, 3]


def test_pop_negative():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_last_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3
    assert list(s) == [1, 2]


def test_pop_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]


def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_reverse():
    s = IndexedSet([1, 2, 3])
    s.reverse()
    assert list(s) == [3, 2, 1]


def test_reverse_after_removal():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    s.reverse()
    assert list(s) == [4, 3, 1]


def test_sort():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]


def test_sort_reverse_keyword():
    s = IndexedSet([3, 1, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_index():
    s = IndexedSet([1, 2, 3])
    assert s.index(1) == 0
    assert s.index(3) == 2


def test_index_missing():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(ValueError):
        s.index(99)


def test_index_after_removal():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert s.index(3) == 1
    assert s.index(4) == 2


# ---------------------------------------------------------------------------
# equality
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_sensitive():
    a = IndexedSet([1, 2, 3])
    b = IndexedSet([1, 2, 3])
    c = IndexedSet([3, 2, 1])
    assert a == b
    assert a != c


def test_eq_set():
    a = IndexedSet([1, 2, 3])
    assert a == {1, 2, 3}
    assert a == {3, 2, 1}


def test_eq_non_iterable():
    a = IndexedSet([1, 2])
    assert (a == 5) is False


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement_repr():
    c = complement(set([1, 2]))
    assert repr(c) == "complement({1, 2})"


def test_complement_contains():
    c = complement(set([1, 2]))
    assert 1 not in c
    assert 3 in c


def test_complement_double():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_empty_is_universal():
    c = complement(set())
    assert 1 in c
    assert "anything" in c


def test_complement_intersection_with_set():
    c = complement(set([2, 3]))
    result = set(range(5)) & c
    assert result == {0, 1, 4}


def test_complement_len_raises():
    c = complement(set([1, 2]))
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_raises():
    c = complement(set([1, 2]))
    with pytest.raises(NotImplementedError):
        list(c)


def test_complement_of_included_len():
    c = complement(set([1, 2])).complemented()
    assert len(c) == 2
    assert sorted(c) == [1, 2]


def test_complement_add_remove():
    c = complement(set([1, 2]))
    c.add(1)
    assert 1 in c
    c.remove(3)
    assert 3 not in c


def test_complement_pop_raises():
    c = complement(set([1, 2]))
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_bool():
    assert bool(complement(set())) is True
    assert bool(complement(set([1]))) is True
    included = complement(set([1])).complemented()
    assert bool(included) is True


def test_complement_invert():
    c = complement(set([1, 2]))
    inv = ~c
    assert sorted(inv) == [1, 2]


def test_complement_complement_inplace():
    c = complement(set([1, 2]))
    c.complement()
    assert sorted(c) == [1, 2]


def test_complement_frozenset():
    c = complement(frozenset([1, 2]))
    assert 1 not in c
    assert repr(c) == "complement(frozenset({1, 2}))"


def test_complement_union():
    c = complement(set([1, 2]))
    r = c | set([3])
    assert 1 not in r
    assert 2 not in r
    assert 3 not in r
    assert 4 in r


def test_complement_difference():
    c = complement(set([1, 2]))
    r = c - set([3])
    assert 1 not in r
    assert 2 not in r
    assert 3 not in r


def test_complement_typeerror():
    c = complement(set([1, 2]))
    with pytest.raises(TypeError):
        c.union([1, 2, 3])


def test_complement_issubset():
    c = complement(set([1, 2]))
    assert c.issubset(complement(set([1]))) is True


def test_complement_issuperset():
    c = complement(set([1, 2]))
    assert c.issuperset(complement(set([1, 2, 3]))) is True


def test_complement_isdisjoint():
    c = complement(set([1, 2]))
    assert c.isdisjoint(set([1, 2])) is True
    assert c.isdisjoint(set([1, 3])) is False


def test_complement_symmetric_difference():
    c = complement(set([1, 2]))
    r = c ^ set([2])
    assert 1 not in r
    assert 2 in r


def test_complement_update_inplace():
    c = complement(set([1, 2, 3]))
    c.update(set([1]))
    assert 1 not in c
    assert 2 not in c
    assert 3 not in c


def test_complement_discard_inplace():
    c = complement(set([1, 2, 3]))
    c.discard(set([1]))
    assert 1 in c
    assert 2 not in c


def test_complement_eq():
    a = complement(set([1, 2]))
    b = complement(set([1, 2]))
    assert a == b
    assert (a == set([1, 2])) is False


def test_complement_hash():
    a = complement(set([1, 2]))
    b = complement(set([1, 2]))
    assert hash(a) == hash(b)


def test_complement_included_excluded_validation():
    with pytest.raises((ValueError, AssertionError)):
        from boltons.setutils import _ComplementSet
        _ComplementSet(included=set([1]), excluded=set([2]))
    with pytest.raises((ValueError, AssertionError)):
        from boltons.setutils import _ComplementSet
        _ComplementSet()


def test_complement_rsub():
    c = complement(set([1, 2]))
    r = set([1, 3]) - c
    assert sorted(r) == [1]


def test_complement_rsub_complement():
    c = complement(set([1, 2]))
    other = complement(set([1, 3]))
    r = other - c
    assert sorted(r) == [3]
