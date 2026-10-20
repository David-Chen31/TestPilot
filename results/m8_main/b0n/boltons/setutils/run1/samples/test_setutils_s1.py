import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet basics
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []


def test_init_with_duplicates_and_order():
    s = IndexedSet([3, 1, 3, 2, 1, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_init_from_string():
    s = IndexedSet('freecreditreport.com')
    assert ''.join(s) == 'frecditpo.m'


def test_repr():
    s = IndexedSet([1, 2, 3])
    assert repr(s) == 'IndexedSet([1, 2, 3])'


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter_and_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(s) == [1, 2, 3]
    assert list(reversed(s)) == [3, 2, 1]


def test_len():
    s = IndexedSet([1, 2, 3])
    assert len(s) == 3
    s.add(4)
    assert len(s) == 4
    s.remove(4)
    assert len(s) == 3


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_matters():
    a = IndexedSet([1, 2, 3])
    b = IndexedSet([1, 2, 3])
    c = IndexedSet([3, 2, 1])
    assert a == b
    assert not (a == c)


def test_eq_with_set():
    a = IndexedSet([1, 2, 3])
    assert a == {1, 2, 3}
    assert a == {3, 2, 1}
    assert not (a == {1, 2})


def test_eq_with_unhashable_returns_false():
    a = IndexedSet([1, 2, 3])
    assert not (a == [1, 2, 3])


# ---------------------------------------------------------------------------
# from_iterable / add / remove / discard / clear
# ---------------------------------------------------------------------------

def test_from_iterable():
    s = IndexedSet.from_iterable(iter([1, 2, 2, 3]))
    assert list(s) == [1, 2, 3]


def test_add_new_and_existing():
    s = IndexedSet([1, 2])
    s.add(3)
    s.add(1)
    assert list(s) == [1, 2, 3]


def test_remove_missing_raises():
    s = IndexedSet([1, 2])
    with pytest.raises(KeyError):
        s.remove(5)


def test_remove_present():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    assert list(s) == [1, 3]
    assert 2 not in s


def test_discard_missing_no_error():
    s = IndexedSet([1, 2])
    s.discard(5)
    assert list(s) == [1, 2]


def test_discard_present():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []


# ---------------------------------------------------------------------------
# isdisjoint / issubset / issuperset
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint([4, 5])
    assert not s.isdisjoint([3, 4])


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset({1, 2, 3})
    assert s.issubset([1, 2])
    assert not s.issubset({1})
    assert not s.issubset([1, 5])


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset({1, 2})
    assert s.issuperset([1, 2, 3])
    assert not s.issuperset({1, 5})
    assert not s.issuperset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# union / intersection / difference / symmetric_difference
# ---------------------------------------------------------------------------

def test_union():
    s = IndexedSet([1, 2])
    u = s.union([2, 3], [4])
    assert isinstance(u, IndexedSet)
    assert list(u) == [1, 2, 3, 4]


def test_union_operator():
    s = IndexedSet([1, 2])
    u = s | {2, 3}
    assert list(u) == [1, 2, 3]
    # reversed
    u2 = {2, 3} | s
    assert set(u2) == {1, 2, 3}


def test_intersection_single():
    s = IndexedSet([1, 2, 3])
    r = s.intersection({2, 3, 4})
    assert list(r) == [2, 3]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    r = s.intersection({1, 2, 3}, {2, 3, 4})
    assert list(r) == [2, 3]


def test_intersection_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s & {2, 3, 4}) == [2, 3]
    assert set({2, 3, 4} & s) == {2, 3}


def test_iter_intersection():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.iter_intersection({2, 3, 4}, {3, 4})) == [3, 4]


def test_difference_single():
    s = IndexedSet([1, 2, 3])
    assert list(s.difference({2})) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.difference({1}, {3, 4})) == [2]


def test_difference_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s - {2}) == [1, 3]


def test_iter_difference():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.iter_difference({2}, {4})) == [1, 3]


def test_rsub():
    s = IndexedSet([1, 2, 3])
    r = {1, 2, 3, 4} - s
    assert isinstance(r, set)
    assert r == {4}


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    r = s.symmetric_difference({2, 3, 4})
    assert set(r) == {1, 4}


def test_symmetric_difference_operator():
    s = IndexedSet([1, 2, 3])
    assert set(s ^ {2, 3, 4}) == {1, 4}


# ---------------------------------------------------------------------------
# In-place operations
# ---------------------------------------------------------------------------

def test_update_no_args():
    s = IndexedSet([1])
    s.update()
    assert list(s) == [1]


def test_update_single_and_multiple():
    s = IndexedSet([1])
    s.update([2, 3])
    assert list(s) == [1, 2, 3]
    s.update([3, 4], [5])
    assert list(s) == [1, 2, 3, 4, 5]


def test_ior():
    s = IndexedSet([1, 2])
    orig = s
    s |= {3}
    assert s is orig
    assert list(s) == [1, 2, 3]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update({2, 3, 4}, {3, 4})
    assert list(s) == [3, 4]


def test_intersection_update_iand():
    s = IndexedSet([1, 2, 3])
    orig = s
    s &= {2, 3}
    assert s is orig
    assert list(s) == [2, 3]


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update({2}, {4})
    assert list(s) == [1, 3]


def test_difference_update_with_self_clears():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert len(s) == 0


def test_isub():
    s = IndexedSet([1, 2, 3])
    orig = s
    s -= {2}
    assert s is orig
    assert list(s) == [1, 3]


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update({2, 3, 4})
    assert set(s) == {1, 4}


def test_symmetric_difference_update_with_self():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert len(s) == 0


def test_ixor():
    s = IndexedSet([1, 2, 3])
    orig = s
    s ^= {2, 3, 4}
    assert s is orig
    assert set(s) == {1, 4}


# ---------------------------------------------------------------------------
# Indexing / slicing
# ---------------------------------------------------------------------------

def test_getitem_positive():
    s = IndexedSet([1, 2, 3, 4])
    assert s[0] == 1
    assert s[3] == 4


def test_getitem_negative():
    s = IndexedSet([1, 2, 3, 4])
    assert s[-1] == 4
    assert s[-4] == 1


def test_getitem_out_of_range():
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s[5]
    with pytest.raises(IndexError):
        s[-5]


def test_getitem_after_removal():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 4
    assert s[-1] == 4


def test_getitem_slice():
    s = IndexedSet([1, 2, 3, 4, 5])
    r = s[1:4]
    assert isinstance(r, IndexedSet)
    assert list(r) == [2, 3, 4]


def test_getitem_slice_step():
    s = IndexedSet([1, 2, 3, 4, 5])
    assert list(s[::2]) == [1, 3, 5]


def test_getitem_slice_negative_step():
    s = IndexedSet([1, 2, 3, 4, 5])
    assert list(s[::-1]) == [5, 4, 3, 2, 1]


def test_getitem_slice_negative_indices():
    s = IndexedSet([1, 2, 3, 4, 5])
    assert list(s[-2:]) == [4, 5]
    assert list(s[:-2]) == [1, 2, 3]


def test_getitem_slice_after_removal():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert list(s) == [1, 3, 5]
    assert list(s[1:]) == [3, 5]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

def test_pop_default():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_minus_one():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(0) == 1
    assert list(s) == [2, 3]


def test_pop_index_after_remove():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(1)
    assert s.pop(0) == 2
    assert list(s) == [3, 4]


def test_pop_empty_raises():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


# ---------------------------------------------------------------------------
# count / index / reverse / sort
# ---------------------------------------------------------------------------

def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(5) == 0


def test_index():
    s = IndexedSet([10, 20, 30])
    assert s.index(10) == 0
    assert s.index(30) == 2


def test_index_missing_raises():
    s = IndexedSet([1, 2])
    with pytest.raises(ValueError):
        s.index(5)


def test_index_after_removal():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert s.index(1) == 0
    assert s.index(3) == 1
    assert s.index(4) == 2


def test_reverse():
    s = IndexedSet([1, 2, 3])
    s.reverse()
    assert list(s) == [3, 2, 1]
    assert s.index(1) == 2


def test_reverse_after_removal():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    s.reverse()
    assert list(s) == [4, 3, 1]


def test_sort():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]


def test_sort_with_key():
    s = IndexedSet(['ccc', 'a', 'bb'])
    s.sort(key=len)
    assert list(s) == ['a', 'bb', 'ccc']


def test_sort_reverse():
    s = IndexedSet([1, 2, 3])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_after_removal():
    s = IndexedSet([3, 1, 2, 4])
    s.remove(1)
    s.sort()
    assert list(s) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Compaction / dead index behavior
# ---------------------------------------------------------------------------

def test_many_removals_trigger_compaction():
    s = IndexedSet(range(100))
    for i in range(50):
        s.remove(i)
    # Some compaction has happened
    assert s._compactions > 0
    assert list(s) == list(range(50, 100))
    assert len(s) == 50


def test_remove_from_end():
    s = IndexedSet([1, 2, 3])
    s.remove(3)
    assert list(s) == [1, 2]
    assert len(s) == 2


def test_large_bulk_intersection_update():
    s = IndexedSet(range(500))
    s.intersection_update(range(400, 600))
    assert list(s) == list(range(400, 500))


def test_large_bulk_difference_update():
    s = IndexedSet(range(500))
    s.difference_update(range(400))
    assert list(s) == list(range(400, 500))


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------

def test_complement_returns_complementset():
    c = complement({1, 2})
    assert repr(c) == 'complement({1, 2})'


def test_complement_contains():
    c = complement({1, 2})
    assert 3 in c
    assert 1 not in c


def test_complement_of_frozenset():
    c = complement(frozenset([1, 2]))
    assert 1 not in c
    assert 3 in c


def test_complement_double_negation_is_set_equality():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_of_complement_returns_set_like():
    c = complement({1, 2})
    inner = complement(c)
    # The inturned out complement should equal {1, 2}
    assert inner == {1, 2}


def test_complement_intersection_with_set():
    assert list(set(range(5)) & complement(set([2, 3]))) == [0, 1, 4]


def test_complement_intersection_method():
    c = complement({2, 3})
    r = c.intersection({0, 1, 2, 3, 4})
    assert set(r) == {0, 1, 4}


def test_complement_intersection_typeerror():
    c = complement({2})
    with pytest.raises(TypeError):
        c.intersection([1, 2])


def test_complement_union_with_set():
    c = complement({1, 2})
    r = c | {1, 2, 3}
    # complement({1,2}) union {1,2,3} = everything except 2... wait
    # complement({1,2}) contains everything except 1,2.
    # union with {1,2,3} -> everything (universal-ish), so exclusion set is empty
    assert 0 in r
    assert 3 in r


def test_complement_union_method():
    c = complement({1})
    r = c.union({1})
    assert 1 in r


def test_complement_union_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.union([1, 2])


def test_complement_add_remove():
    c = complement({1, 2})
    c.add(1)  # remove from excluded
    assert 1 in c
    c.remove(2)  # add to excluded
    assert 2 not in c


def test_complement_pop_excluded_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_len_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_bool_true():
    c = complement({1, 2})
    assert bool(c) is True


def test_complement_invert():
    c = complement({1, 2})
    inv = ~c
    assert isinstance(inv, type(c))
    assert inv == {1, 2}


def test_complement_hash():
    c = complement({1, 2})
    assert isinstance(hash(c), int)


def test_complement_eq_with_set():
    c = complement({1, 2})
    assert not (c == {1, 2})
    assert c == c


def test_complement_subtraction():
    c = complement({1, 2})
    r = c - {3}
    # complement({1,2}) - {3} excludes 1,2,3 -> complement({1,2,3})
    assert 1 not in r
    assert 3 not in r
    assert 0 in r


def test_complement_rsub():
    c = complement({1, 2})
    r = {1, 2, 3, 4} - c
    # {1,2,3,4} - complement({1,2}) = {1,2,3,4} intersect {1,2} = {1,2}
    assert set(r) == {1, 2}


def test_complement_update_and_discard():
    c = complement({1, 2, 3})
    c.update({1})
    assert 1 in c
    c.discard({2})
    assert 2 not in c


def test_complement_update_with_complement():
    a = complement({1, 2})
    b = complement({2, 3})
    a.update(b)
    # Both exclude 2, so 'a' should exclude at least 2
    assert 2 not in a


def test_complement_le_ge():
    # complement({1,2,3}) <= complement({1,2}) since excluded of other subset of excluded of self? 
    a = complement({1, 2, 3})
    b = complement({1, 2})
    assert not (a <= b)
    assert a >= b


def test_complement_symmetric_difference():
    a = complement({1, 2})
    r = a ^ complement({2, 3})
    # (-,-) case: symmetric differential of excluded -> included
    assert 1 not in r
    assert 3 not in r


def test_complement_isdisjoint():
    a = complement({1, 2})
    assert not a.isdisjoint({5})  # 5 is in a
    assert a.isdisjoint({1, 2})


def test_complement_issubset_issuperset():
    a = complement({1, 2})
    b = complement({1, 2, 3})
    # b excludes more -> b has fewer elements -> b subset of a
    assert b.issubset(a)
    assert a.issuperset(b)


def test_complement_isub():
    a = complement({1, 2})
    a -= {3}
    assert 3 not in a


def test_complement_iand():
    a = complement({1, 2})
    a &= {1, 2, 3}
    # complement({1,2}) & {1,2,3} = {3} -> complement includes {3}
    assert 3 in a
    assert 1 not in a


def test_complement_ior():
    a = complement({1, 2})
    a |= {1}
    assert 1 in a


def test_complement_ixor():
    a = complement({1, 2})
    a ^= complement({2, 3})
    assert 1 not in a


def test_complement_init_both_none_raises():
    from boltons.setutils import _ComplementSet
    with pytest.raises(AssertionError):
        _ComplementSet()


def test_complement_init_both_given_raises():
    from boltons.setutils import _ComplementSet
    with pytest.raises(ValueError):
        _ComplementSet(included={1}, excluded={2})


def test_complement_invert_frozenset_stays():
    c = complement(frozenset([1, 2]))
    inv = ~c
    # Inverting a frozenset complement yields the frozenset itself
    assert inv == frozenset([1, 2])


def test_complement_of_complement_double():
    s = frozenset([1, 2])
    c1 = complement(s)
    c2 = complement(c1)
    assert c2 == s


def test_complement_difference_method_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.difference([1, 2])


def test_complement_symmetric_difference_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference([1, 2])


def test_complement_issubset_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.issubset([1, 2])


def test_complement_issuperset_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.issuperset([1, 2])


def test_complement_symmetric_difference_update_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference_update([1, 2])


def test_complement_isdisjoint_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.isdisjoint([1, 2])
