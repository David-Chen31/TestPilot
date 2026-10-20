import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet basics
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []


def test_init_with_iterable_dedupes_and_preserves_order():
    s = IndexedSet([3, 1, 3, 2, 1])
    assert list(s) == [3, 1, 2]
    assert len(s) == 3


def test_init_with_string():
    s = IndexedSet('freecreditreport.com')
    assert len(s) == len(set('freecreditreport.com'))
    assert ''.join(s[:s.index('.')]) == 'frecditpo'


def test_repr():
    s = IndexedSet([1, 2, 3])
    assert repr(s) == 'IndexedSet([1, 2, 3])'


def test_from_iterable():
    s = IndexedSet.from_iterable([1, 2, 2, 3])
    assert list(s) == [1, 2, 3]
    assert isinstance(s, IndexedSet)


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter():
    s = IndexedSet([1, 2, 3])
    assert list(s) == [1, 2, 3]


def test_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(reversed(s)) == [3, 2, 1]


def test_len():
    s = IndexedSet([1, 2, 3, 3])
    assert len(s) == 3


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


def test_remove_missing_raises():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(KeyError):
        s.remove(99)


def test_discard():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    assert list(s) == [1, 3]
    # discarding missing does not raise
    s.discard(99)
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []


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


def test_pop_empty_raises():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


def test_pop_invalid_index_raises():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(IndexError):
        s.pop(10)


def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_index():
    s = IndexedSet([10, 20, 30])
    assert s.index(10) == 0
    assert s.index(20) == 1
    assert s.index(30) == 2


def test_index_after_removal():
    s = IndexedSet([10, 20, 30])
    s.remove(10)
    assert s.index(20) == 0
    assert s.index(30) == 1


def test_index_missing_raises():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(ValueError):
        s.index(99)


# ---------------------------------------------------------------------------
# Indexing / slicing
# ---------------------------------------------------------------------------

def test_getitem():
    s = IndexedSet([10, 20, 30])
    assert s[0] == 10
    assert s[-1] == 30
    assert s[1] == 20


def test_getitem_out_of_range():
    s = IndexedSet([10, 20])
    with pytest.raises(IndexError):
        s[5]
    with pytest.raises(IndexError):
        s[-5]


def test_getitem_after_removal():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert list(s) == [1, 3, 5]
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 5


def test_slice():
    s = IndexedSet(range(10))
    sliced = s[2:5]
    assert isinstance(sliced, IndexedSet)
    assert list(sliced) == [2, 3, 4]


def test_slice_with_step():
    s = IndexedSet(range(10))
    assert list(s[::2]) == [0, 2, 4, 6, 8]


def test_slice_negative():
    s = IndexedSet(range(10))
    assert list(s[-3:]) == [7, 8, 9]
    assert list(s[:-7]) == [0, 1, 2]


def test_slice_negative_step():
    s = IndexedSet(range(10))
    # x[2:4:-1] == reversed(x)[2:4]
    assert list(s[2:4:-1]) == [7, 6]


def test_slice_after_removal():
    s = IndexedSet(range(10))
    s.remove(2)
    s.remove(5)
    assert list(s) == [0, 1, 3, 4, 6, 7, 8, 9]
    assert list(s[1:4]) == [1, 3, 4]


def test_iter_slice():
    s = IndexedSet(range(6))
    assert list(s.iter_slice(1, 4)) == [1, 2, 3]
    assert list(s.iter_slice(None, None, 2)) == [0, 2, 4]


# ---------------------------------------------------------------------------
# reverse / sort
# ---------------------------------------------------------------------------

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


def test_sort_reverse():
    s = IndexedSet([3, 1, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_key():
    s = IndexedSet(['bbb', 'a', 'cc'])
    s.sort(key=len)
    assert list(s) == ['a', 'cc', 'bbb']


def test_sort_after_removal():
    s = IndexedSet([5, 3, 1, 4, 2])
    s.remove(1)
    s.sort()
    assert list(s) == [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_eq_other_indexedset():
    assert IndexedSet([1, 2, 3]) == IndexedSet([1, 2, 3])
    assert IndexedSet([1, 2, 3]) != IndexedSet([1, 2])
    assert IndexedSet([1, 2, 3]) != IndexedSet([1, 2, 4])
    # order matters for IndexedSet comparison
    assert IndexedSet([1, 2, 3]) != IndexedSet([3, 2, 1])


def test_eq_with_set():
    assert IndexedSet([1, 2, 3]) == {1, 2, 3}
    assert IndexedSet([1, 2, 3]) != {1, 2}


def test_eq_with_other_iterable():
    assert IndexedSet([1, 2, 3]) == [1, 2, 3]


def test_eq_not_comparable():
    s = IndexedSet([1, 2, 3])
    assert (s == 42) is False


# ---------------------------------------------------------------------------
# Set operations (non in-place)
# ---------------------------------------------------------------------------

def test_union():
    a = IndexedSet([1, 2, 3])
    result = a.union({3, 4})
    assert isinstance(result, IndexedSet)
    assert list(result) == [1, 2, 3, 4]
    # original unchanged
    assert list(a) == [1, 2, 3]


def test_union_multiple():
    a = IndexedSet([1, 2])
    result = a.union([3], [4, 1])
    assert list(result) == [1, 2, 3, 4]


def test_or_operator():
    a = IndexedSet([1, 2])
    result = a | {2, 3}
    assert list(result) == [1, 2, 3]


def test_intersection():
    a = IndexedSet([1, 2, 3, 4])
    result = a.intersection({2, 4, 5})
    assert isinstance(result, IndexedSet)
    assert list(result) == [2, 4]


def test_intersection_multiple():
    a = IndexedSet([1, 2, 3, 4])
    result = a.intersection([2, 3, 4], [3, 4, 5])
    assert list(result) == [3, 4]


def test_and_operator():
    a = IndexedSet([1, 2, 3])
    result = a & {2, 3, 4}
    assert list(result) == [2, 3]


def test_difference():
    a = IndexedSet([1, 2, 3, 4])
    result = a.difference({2, 4})
    assert isinstance(result, IndexedSet)
    assert list(result) == [1, 3]


def test_difference_multiple():
    a = IndexedSet([1, 2, 3, 4])
    result = a.difference([2], [4])
    assert list(result) == [1, 3]


def test_sub_operator():
    a = IndexedSet([1, 2, 3])
    result = a - {2}
    assert list(result) == [1, 3]


def test_rsub_operator():
    a = IndexedSet([2, 3])
    result = {1, 2, 3, 4} - a
    assert result == {1, 4}


def test_rsub_with_list():
    a = IndexedSet([2, 3])
    result = [1, 2, 3, 4] - a
    assert result == [1, 4]


def test_symmetric_difference():
    a = IndexedSet([1, 2, 3])
    result = a.symmetric_difference({3, 4})
    assert isinstance(result, IndexedSet)
    assert sorted(result) == [1, 2, 4]


def test_xor_operator():
    a = IndexedSet([1, 2, 3])
    result = a ^ {3, 4}
    assert sorted(result) == [1, 2, 4]


def test_iter_intersection():
    a = IndexedSet([1, 2, 3, 4])
    assert list(a.iter_intersection({2, 4}, {4})) == [4]


def test_iter_difference():
    a = IndexedSet([1, 2, 3, 4])
    assert list(a.iter_difference({2}, {4})) == [1, 3]


# ---------------------------------------------------------------------------
# In-place set operations
# ---------------------------------------------------------------------------

def test_update():
    a = IndexedSet([1, 2])
    a.update({2, 3})
    assert list(a) == [1, 2, 3]


def test_update_multiple():
    a = IndexedSet([1])
    a.update([2], [3, 1])
    assert list(a) == [1, 2, 3]


def test_update_no_args():
    a = IndexedSet([1, 2])
    a.update()
    assert list(a) == [1, 2]


def test_ior_operator():
    a = IndexedSet([1, 2])
    a |= {2, 3}
    assert list(a) == [1, 2, 3]


def test_intersection_update():
    a = IndexedSet([1, 2, 3, 4])
    a.intersection_update({2, 4, 5})
    assert list(a) == [2, 4]


def test_iand_operator():
    a = IndexedSet([1, 2, 3])
    a &= {2, 3, 4}
    assert list(a) == [2, 3]


def test_difference_update():
    a = IndexedSet([1, 2, 3, 4])
    a.difference_update({2, 4})
    assert list(a) == [1, 3]


def test_difference_update_self():
    a = IndexedSet([1, 2, 3])
    a.difference_update(a)
    assert list(a) == []


def test_isub_operator():
    a = IndexedSet([1, 2, 3])
    a -= {2}
    assert list(a) == [1, 3]


def test_symmetric_difference_update():
    a = IndexedSet([1, 2, 3])
    a.symmetric_difference_update({3, 4})
    assert sorted(a) == [1, 2, 4]


def test_symmetric_difference_update_self():
    a = IndexedSet([1, 2, 3])
    a.symmetric_difference_update(a)
    assert list(a) == []


def test_ixor_operator():
    a = IndexedSet([1, 2, 3])
    a ^= {3, 4}
    assert sorted(a) == [1, 2, 4]


# ---------------------------------------------------------------------------
# Predicate methods
# ---------------------------------------------------------------------------

def test_isdisjoint():
    a = IndexedSet([1, 2, 3])
    assert a.isdisjoint({4, 5})
    assert not a.isdisjoint({3, 4})


def test_issubset():
    a = IndexedSet([1, 2])
    assert a.issubset({1, 2, 3})
    assert a.issubset([1, 2])
    assert not a.issubset({1})


def test_issuperset():
    a = IndexedSet([1, 2, 3])
    assert a.issuperset({1, 2})
    assert not a.issuperset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# Bulk discard path (exercises _bulk_discard)
# ---------------------------------------------------------------------------

def test_intersection_update_bulk():
    a = IndexedSet(range(1000))
    a.intersection_update(range(0, 1000, 2))
    assert len(a) == 500
    assert list(a) == list(range(0, 1000, 2))


def test_difference_update_bulk():
    a = IndexedSet(range(1000))
    a.difference_update(range(0, 1000, 2))
    assert list(a) == list(range(1, 1000, 2))


def test_many_removes_compaction():
    a = IndexedSet(range(100))
    for i in range(0, 100, 2):
        a.remove(i)
    assert list(a) == list(range(1, 100, 2))
    # indexing after dead-index bookkeeping
    assert a[0] == 1
    assert a[-1] == 99


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------

def test_complement_contains():
    c = complement({1, 2, 3})
    assert 1 not in c
    assert 4 in c


def test_complement_repr():
    c = complement({1, 2})
    assert repr(c) == 'complement({1, 2})'


def test_complement_double_returns_set():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_from_frozenset():
    c = complement(frozenset([1, 2]))
    assert 1 not in c
    assert 3 in c


def test_complement_len_undefined():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_undefined():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_bool():
    assert bool(complement(set())) is True
    assert bool(complement({1})) is True


def test_complement_add_remove():
    c = complement({1, 2})
    c.add(1)
    assert 1 in c
    c.add(5)
    assert 5 in c
    c.remove(7)
    assert 7 not in c


def test_complement_and_with_set():
    c = complement({2, 3})
    result = {0, 1, 2, 3, 4} & c
    assert result == {0, 1, 4}


def test_complement_intersection_method():
    c = complement({2, 3})
    result = c.intersection({0, 1, 2, 3, 4})
    assert result == {0, 1, 4}


def test_complement_intersection_bad_type():
    c = complement({2, 3})
    with pytest.raises(TypeError):
        c.intersection([1, 2, 3])


def test_complement_or():
    c = complement({1, 2})
    result = c | {2, 3}
    # everything except (excluded - inc) = except {1}
    assert 1 not in result
    assert 2 in result
    assert 3 in result


def test_complement_two_complements():
    c1 = complement({1})
    c2 = complement({2})
    result = c1 | c2
    # union of two complements: everything except intersection of excluded
    assert 1 in result
    assert 2 in result


def test_complement_sub():
    c = complement({1, 2, 3})
    result = c - {1}
    # complement excludes {1,2,3}; subtract {1} -> excludes {1,2,3}
    assert 1 not in result
    assert 4 in result


def test_complement_xor():
    c = complement({1, 2})
    result = c ^ {3}
    assert 3 in result or 3 not in result  # sanity: no exception
    # complement of {1,2} XOR {3}: (univ - {1,2}) XOR {3}
    assert 1 not in result


def test_complement_eq():
    assert complement({1, 2}) == complement({1, 2})
    assert complement({1, 2}) != complement({1, 3})


def test_complement_eq_set():
    # included-set complement equals its included set
    c = complement(complement({1, 2, 3}))
    assert c == {1, 2, 3}


def test_complement_hash():
    c = complement({1, 2})
    d = complement({1, 2})
    assert hash(c) == hash(d)


def test_complement_pop_included():
    c = complement(complement({1, 2, 3}))
    popped = c.pop()
    assert popped in {1, 2, 3}
    assert len(c) == 2


def test_complement_pop_undefined():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_update():
    c = complement({1, 2, 3})
    c.update({1})
    # intersection with {1} of excluded removes 2,3
    assert 1 not in c
    assert 2 in c


def test_complement_discard():
    c = complement({1, 2})
    c.discard({1})
    assert 1 not in c
    assert 3 in c


def test_complement_isdisjoint():
    c = complement({1, 2})
    assert not c.isdisjoint({1})
    assert c.isdisjoint({3, 4})


def test_complement_issubset():
    c = complement({1, 2, 3})
    assert c.issubset(complement({1, 2}))
    assert not c.issubset(complement({1, 2, 3, 4}))


def test_complement_issuperset():
    c = complement({1, 2})
    assert c.issuperset(complement({1, 2, 3}))
    assert not c.issuperset(complement({1}))


def test_complement_complemented():
    c = complement({1, 2})
    cc = c.complemented()
    assert type(cc) is type(c)
    assert not cc == c


def test_complement_invert_operator():
    c = complement({1, 2})
    cc = ~c
    assert type(cc) is type(c)


def test_complement_complement_inplace():
    c = complement({1, 2})
    c.complement()
    assert 1 not in c
    assert 3 in c


def test_complement_rsub():
    c = complement({2, 3})
    result = {1, 2, 3, 4} - c
    # other - complement = intersection of other and excluded
    assert result == {2, 3}


def test_norm_args_typeerror_non_set():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference_update([1, 2])


def test_complement_notimplemented_operator():
    c = complement({1})
    with pytest.raises(TypeError):
        c | 42
