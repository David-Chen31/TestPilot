import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet: construction, basic container protocol
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []


def test_init_with_iterable_dedups_preserving_order():
    s = IndexedSet([3, 1, 2, 3, 1, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_init_with_empty_iterable():
    s = IndexedSet([])
    assert len(s) == 0


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 2 in s
    assert 5 not in s


def test_iter_and_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(s) == [1, 2, 3]
    assert list(reversed(s)) == [3, 2, 1]


def test_repr():
    s = IndexedSet([1, 2])
    assert repr(s) == "IndexedSet([1, 2])"


def test_from_iterable_classmethod():
    s = IndexedSet.from_iterable([1, 1, 2])
    assert isinstance(s, IndexedSet)
    assert list(s) == [1, 2]


# ---------------------------------------------------------------------------
# IndexedSet: add / remove / discard / clear / pop
# ---------------------------------------------------------------------------

def test_add_and_duplicate():
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


def test_remove_missing_raises_keyerror():
    s = IndexedSet([1])
    with pytest.raises(KeyError):
        s.remove(99)


def test_discard_present_and_missing():
    s = IndexedSet([1, 2])
    s.discard(1)
    assert list(s) == [2]
    s.discard(99)  # no error
    assert list(s) == [2]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []


def test_pop_default_last():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_explicit_last_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3
    assert list(s) == [1, 2]


def test_pop_minus_one():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_first():
    s = IndexedSet([1, 2, 3])
    assert s.pop(0) == 1
    assert list(s) == [2, 3]


def test_pop_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]
    assert len(s) == 3


def test_pop_empty_raises():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


# ---------------------------------------------------------------------------
# IndexedSet: indexing and slicing
# ---------------------------------------------------------------------------

def test_getitem_positive_and_negative():
    s = IndexedSet([10, 20, 30])
    assert s[0] == 10
    assert s[1] == 20
    assert s[-1] == 30
    assert s[-3] == 10


def test_getitem_out_of_range():
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s[5]
    with pytest.raises(IndexError):
        s[-5]


def test_getitem_after_removal():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 4
    assert s[-1] == 5


def test_slice_basic():
    s = IndexedSet([1, 2, 3, 4, 5])
    result = s[1:3]
    assert isinstance(result, IndexedSet)
    assert list(result) == [2, 3]


def test_slice_negative_bounds():
    s = IndexedSet([1, 2, 3, 4, 5])
    assert list(s[-2:]) == [4, 5]
    assert list(s[:-2]) == [1, 2, 3]


def test_slice_step():
    s = IndexedSet([1, 2, 3, 4, 5])
    assert list(s[0:5:2]) == [1, 3, 5]


def test_slice_after_removal():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    assert list(s[0:2]) == [1, 3]
    assert list(s[:]) == [1, 3, 4, 5]


def test_slice_reverse():
    s = IndexedSet([1, 2, 3, 4, 5])
    # negative step slices reversed stream with forward bounds
    assert list(s[::-1]) == [5, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# IndexedSet: index / count / sort / reverse
# ---------------------------------------------------------------------------

def test_index():
    s = IndexedSet([10, 20, 30])
    assert s.index(10) == 0
    assert s.index(20) == 1
    assert s.index(30) == 2


def test_index_after_removal():
    s = IndexedSet([10, 20, 30, 40])
    s.remove(20)
    assert s.index(10) == 0
    assert s.index(30) == 1
    assert s.index(40) == 2


def test_index_missing_raises_valueerror():
    s = IndexedSet([1, 2])
    with pytest.raises(ValueError):
        s.index(99)


def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_reverse():
    s = IndexedSet([1, 2, 3])
    s.reverse()
    assert list(s) == [3, 2, 1]
    assert s.index(3) == 0


def test_sort():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]
    assert s.index(1) == 0


def test_sort_with_key():
    s = IndexedSet([1, 3, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_after_removal():
    s = IndexedSet([5, 3, 1, 4])
    s.remove(3)
    s.sort()
    assert list(s) == [1, 4, 5]


# ---------------------------------------------------------------------------
# IndexedSet: set relations
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint([4, 5]) is True
    assert s.isdisjoint([3, 4]) is False


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset({1, 2, 3}) is True
    assert s.issubset({1}) is False


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset({1, 2}) is True
    assert s.issuperset({1, 8}) is False


# ---------------------------------------------------------------------------
# IndexedSet: set operations producing new sets
# ---------------------------------------------------------------------------

def test_union():
    s = IndexedSet([1, 2])
    r = s.union([2, 3], [4])
    assert isinstance(r, IndexedSet)
    assert list(r) == [1, 2, 3, 4]


def test_intersection_single():
    s = IndexedSet([1, 2, 3])
    r = s.intersection({2, 3, 4})
    assert list(r) == [2, 3]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.intersection({1, 2, 3}, {2, 3, 4})
    assert list(r) == [2, 3]


def test_difference_single():
    s = IndexedSet([1, 2, 3])
    r = s.difference({2})
    assert list(r) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.difference({1}, {4})
    assert list(r) == [2, 3]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    r = s.symmetric_difference({2, 3, 4})
    assert set(r) == {1, 4}


def test_iter_intersection():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_intersection({1, 2})) == [1, 2]


def test_iter_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_difference({2})) == [1, 3]


def test_operators():
    s = IndexedSet([1, 2, 3])
    assert set(s | {4}) == {1, 2, 3, 4}
    assert set(s & {2, 3, 4}) == {2, 3}
    assert set(s - {2}) == {1, 3}
    assert set(s ^ {3, 4}) == {1, 2, 4}


def test_rsub():
    s = IndexedSet([1, 2])
    # set - IndexedSet uses set.__sub__; IndexedSet has no __rsub__
    # IndexedSet.__rsub__ builds type(other)(vals)
    result = s.__rsub__({1, 2, 3})
    assert result == {3}


# ---------------------------------------------------------------------------
# IndexedSet: in-place operations
# ---------------------------------------------------------------------------

def test_update_single():
    s = IndexedSet([1])
    s.update([2, 3])
    assert list(s) == [1, 2, 3]


def test_update_multiple():
    s = IndexedSet([1])
    s.update([2], [3, 4])
    assert list(s) == [1, 2, 3, 4]


def test_update_no_args():
    s = IndexedSet([1])
    s.update()
    assert list(s) == [1]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update([2, 3, 4, 5])
    assert list(s) == [2, 3, 4]


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update([2, 4])
    assert list(s) == [1, 3]


def test_difference_update_self_clears():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert len(s) == 0


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update([2, 3, 4])
    assert set(s) == {1, 4}


def test_symmetric_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert len(s) == 0


def test_inplace_operators():
    s = IndexedSet([1, 2, 3])
    s |= {4}
    assert set(s) == {1, 2, 3, 4}
    s &= {3, 4}
    assert set(s) == {3, 4}
    s -= {4}
    assert set(s) == {3}
    s ^= {3, 5}
    assert set(s) == {5}


# ---------------------------------------------------------------------------
# IndexedSet: equality
# ---------------------------------------------------------------------------

def test_eq_indexedset():
    assert IndexedSet([1, 2]) == IndexedSet([1, 2])
    # ordering matters for IndexedSet vs IndexedSet
    assert not (IndexedSet([1, 2]) == IndexedSet([2, 1]))


def test_eq_regular_set():
    assert IndexedSet([1, 2]) == {2, 1}
    assert not (IndexedSet([1, 2]) == {1, 3})


def test_eq_incomparable():
    assert (IndexedSet([1]) == 5) is False


# ---------------------------------------------------------------------------
# IndexedSet: compaction / culling internals indirectly
# ---------------------------------------------------------------------------

def test_many_removals_keep_consistency():
    s = IndexedSet(range(20))
    for i in range(0, 20, 2):
        s.remove(i)
    assert list(s) == list(range(1, 20, 2))
    assert len(s) == 10
    assert s[0] == 1
    assert s[-1] == 19
    assert s.index(19) == 9


def test_bulk_removal_path():
    # more than _MAX_DEAD_INTERVALS removals exercise _bulk_discard
    n = 500
    s = IndexedSet(range(n))
    s.intersection_update(range(0, n, 2))
    assert list(s) == list(range(0, n, 2))
    assert len(s) == n // 2


def test_readd_after_removal():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    s.add(2)
    assert list(s) == [1, 3, 2]
    assert s.index(2) == 2


def test_compaction_after_many_removals():
    s = IndexedSet(range(100))
    for i in range(90):
        s.discard(i)
    assert list(s) == list(range(90, 100))
    assert len(s) == 10


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement_contains():
    c = complement({2, 3})
    assert 2 not in c
    assert 5 in c


def test_complement_repr():
    c = complement({2, 3})
    assert repr(c) == "complement({2, 3})"


def test_complement_intersection_with_set():
    c = complement({2, 3})
    result = set(range(5)) & c
    assert set(result) == {0, 1, 4}


def test_complement_double_complement():
    s = set(range(5))
    c = complement(s)
    c2 = complement(c)
    assert c2 == s


def test_complement_of_frozenset():
    c = complement(frozenset({1}))
    assert 1 not in c
    assert 2 in c


def test_complement_non_set_iterable():
    c = complement([1, 2])
    assert 1 not in c
    assert 3 in c


def test_complement_len_undefined():
    c = complement({1})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_undefined():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_pop():
    c = complement({1, 2, 3})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_pop_included():
    s = set(range(5))
    c = complement(s)
    c.complement()  # now has included
    assert c.pop() in range(5)


def test_complement_complement_method_inplace():
    c = complement({1, 2})
    c.complement()
    assert 1 in c
    assert 2 in c
    assert 5 not in c


def test_complement_invert():
    c = complement({1})
    inv = ~c
    assert 1 in inv
    assert 2 not in inv


def test_complement_add_remove():
    c = complement({1, 2})
    c.add(1)  # removing from excluded means 1 now allowed
    assert 1 in c
    c.remove(3)  # adds to excluded
    assert 3 not in c


def test_complement_bool():
    c = complement(set())
    assert bool(c) is True


def test_complement_bool_included_empty():
    c = complement(frozenset())
    c.complement()  # becomes empty included set
    assert bool(c) is False


def test_complement_union():
    c = complement({2})
    result = c.union({2, 3})
    # complement({2}) union {2,3}: excluded = {2} - {2,3} = {}
    # So result is complement(set()) => everything is in it
    assert 2 in result
    assert 3 in result


def test_complement_intersection_method():
    c = complement({2})
    result = c.intersection({1, 2, 3})
    assert 1 in result
    assert 2 not in result
    assert 3 in result


def test_complement_difference():
    c = complement({2})
    result = c.difference({1})
    # complement({2}) - {1}: excluded = {2} | {1} = {1,2}
    assert 1 not in result
    assert 2 not in result
    assert 3 in result


def test_complement_symmetric_difference():
    c = complement({2})
    result = c.symmetric_difference({2, 3})
    # (- , +) branch: excluded = {2} - {2,3} = {}
    # So result is complement(set())
    assert 2 in result
    assert 3 in result


def test_complement_isdisjoint():
    c = complement({1, 2})
    assert c.isdisjoint({1, 2}) is True
    assert c.isdisjoint({1, 3}) is False


def test_complement_issubset_superset():
    c1 = complement({1, 2})
    c2 = complement({1})
    # c1 (excludes {1,2}) is subset of c2 (excludes {1})
    assert c1.issubset(c2) is True
    assert c2.issuperset(c1) is True


def test_complement_typeerror_on_bad_arg():
    c = complement({1})
    with pytest.raises(TypeError):
        c.union([2, 3])
    with pytest.raises(TypeError):
        c.intersection([2, 3])
    with pytest.raises(TypeError):
        c.difference([2, 3])


def test_complement_hash():
    # complement of a frozenset hashes fine
    fs = frozenset({1})
    c = complement(fs)
    assert hash(c) == hash(None) ^ hash(fs)
    # complement of a plain set raises because set is unhashable
    c2 = complement({1})
    with pytest.raises(TypeError):
        hash(c2)


def test_complement_eq():
    assert complement({1}) == complement({1})
    assert not (complement({1}) == complement({2}))


def test_complement_update():
    c = complement({1, 2, 3})
    c.update([1])
    # update with a non-set iterable intersects excluded with it
    # excluded = {1,2,3} & {1} = {1}
    assert 2 in c
    assert 1 not in c


def test_complement_update_with_set():
    c = complement({1, 2, 3})
    c.update({1})
    # excluded &= included => {1,2,3} & {1} = {1}
    assert 2 in c
    assert 1 not in c


def test_complement_discard():
    c = complement({1, 2})
    c.discard({2})
    # discard with a non-ComplementSet iterable: excluded.update(inc)
    # excluded = {1,2} | {2} = {1,2}
    assert 2 not in c
    assert 1 not in c


def test_complement_inplace_operators():
    c = complement({1, 2, 3})
    c &= {1, 2}
    # self._included is None, exc is None: self._excluded = inc - self._excluded
    # = {1,2} - {1,2,3} = {}
    assert 1 in c
    assert 2 in c
    assert 3 in c


def test_complement_two_complement_ops():
    c1 = complement({1, 2})
    c2 = complement({2, 3})
    inter = c1 & c2
    # both excluded: excluded = {1,2} | {2,3} = {1,2,3}
    assert 1 not in inter
    assert 2 not in inter
    assert 3 not in inter
    assert 4 in inter


def test_complement_two_union():
    c1 = complement({1, 2})
    c2 = complement({2, 3})
    union = c1 | c2
    # both excluded: excluded = {1,2} & {2,3} = {2}
    assert 1 in union
    assert 2 not in union
    assert 3 in union
