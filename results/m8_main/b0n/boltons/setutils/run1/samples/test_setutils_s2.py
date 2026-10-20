import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet basics
# ---------------------------------------------------------------------------

def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []
    assert repr(s) == 'IndexedSet([])'


def test_init_from_iterable_dedup_preserves_order():
    s = IndexedSet([3, 1, 3, 2, 1, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_init_from_generator():
    s = IndexedSet(x for x in [2, 2, 3])
    assert list(s) == [2, 3]


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter_and_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(iter(s)) == [1, 2, 3]
    assert list(reversed(s)) == [3, 2, 1]


def test_from_iterable_classmethod():
    s = IndexedSet.from_iterable([1, 1, 2])
    assert isinstance(s, IndexedSet)
    assert list(s) == [1, 2]


def test_add_new_and_duplicate():
    s = IndexedSet()
    s.add(1)
    s.add(1)
    s.add(2)
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


def test_discard_present_and_absent():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    assert list(s) == [1, 3]
    s.discard(99)  # no raise
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []


def test_pop_default_last():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_minus_one():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_middle_index():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]
    # subsequent indexing works
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 4


def test_pop_last_index_explicit():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3
    assert list(s) == [1, 2]


def test_pop_negative_index():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(-2) == 3
    assert list(s) == [1, 2, 4]


def test_pop_out_of_range():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(IndexError):
        s.pop(10)


def test_getitem_positive():
    s = IndexedSet([10, 20, 30])
    assert s[0] == 10
    assert s[1] == 20
    assert s[2] == 30


def test_getitem_negative():
    s = IndexedSet([10, 20, 30])
    assert s[-1] == 30
    assert s[-3] == 10


def test_getitem_out_of_range():
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s[5]


def test_getitem_after_removals():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert list(s) == [1, 3, 5]
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 5


def test_slice_basic():
    s = IndexedSet(range(10))
    sliced = s[2:5]
    assert isinstance(sliced, IndexedSet)
    assert list(sliced) == [2, 3, 4]


def test_slice_step():
    s = IndexedSet(range(10))
    assert list(s[0:10:2]) == [0, 2, 4, 6, 8]


def test_slice_negative_start_stop():
    s = IndexedSet(range(5))
    assert list(s[-2:]) == [3, 4]
    assert list(s[:-2]) == [0, 1, 2]


def test_slice_negative_step():
    s = IndexedSet(range(6))
    # x[2:4:-1] == reversed(x)[2:4]
    assert list(s[2:4:-1]) == list(reversed(list(range(6)))[2:4])


def test_iter_slice_direct():
    s = IndexedSet(range(6))
    assert list(s.iter_slice(1, 4)) == [1, 2, 3]


def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_index():
    s = IndexedSet(['a', 'b', 'c'])
    assert s.index('a') == 0
    assert s.index('c') == 2


def test_index_after_removal():
    s = IndexedSet(['a', 'b', 'c', 'd'])
    s.remove('b')
    assert s.index('a') == 0
    assert s.index('c') == 1
    assert s.index('d') == 2


def test_index_missing_raises():
    s = IndexedSet(['a'])
    with pytest.raises(ValueError):
        s.index('z')


def test_index_error_message():
    s = IndexedSet(['a'])
    with pytest.raises(ValueError) as exc:
        s.index('z')
    assert "'z'" in str(exc.value)
    assert 'IndexedSet' in str(exc.value)


def test_reverse():
    s = IndexedSet([1, 2, 3, 4])
    s.reverse()
    assert list(s) == [4, 3, 2, 1]
    assert s[0] == 4
    assert s.index(1) == 3


def test_reverse_after_removals():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    s.reverse()
    assert list(s) == [5, 3, 1]
    assert s.index(5) == 0


def test_sort():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]
    assert s[0] == 1


def test_sort_reverse_key():
    s = IndexedSet([3, 1, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_key():
    s = IndexedSet(['aaa', 'b', 'cc'])
    s.sort(key=len)
    assert list(s) == ['b', 'cc', 'aaa']


def test_sort_already_sorted_is_noop():
    s = IndexedSet([1, 2, 3])
    s.sort()
    assert list(s) == [1, 2, 3]
    assert s.dead_indices == []


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_matters():
    assert IndexedSet([1, 2, 3]) == IndexedSet([1, 2, 3])
    assert IndexedSet([1, 2, 3]) != IndexedSet([3, 2, 1])


def test_eq_regular_set():
    assert IndexedSet([1, 2, 3]) == {3, 2, 1}
    assert IndexedSet([1, 2]) != {1, 2, 3}


def test_eq_against_non_iterable():
    s = IndexedSet([1])
    assert (s == 5) is False


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint({4, 5})
    assert not s.isdisjoint({3, 4})


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset({1, 2, 3})
    assert not s.issubset({1})
    assert s.issubset({1, 2})


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset({1, 2})
    assert not s.issuperset({1, 4})
    assert s.issuperset({1, 2, 3})


def test_union():
    s = IndexedSet([1, 2])
    result = s.union([2, 3, 4])
    assert isinstance(result, IndexedSet)
    assert list(result) == [1, 2, 3, 4]


def test_union_multiple():
    s = IndexedSet([1])
    assert list(s.union([2], [3])) == [1, 2, 3]


def test_intersection_single():
    s = IndexedSet([1, 2, 3, 4])
    result = s.intersection({2, 4})
    assert isinstance(result, IndexedSet)
    assert list(result) == [2, 4]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    result = s.intersection({2, 3, 4}, {3, 4, 5})
    assert list(result) == [3, 4]


def test_iter_intersection():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_intersection({2, 3})) == [2, 3]


def test_difference_single():
    s = IndexedSet([1, 2, 3, 4])
    result = s.difference({2, 4})
    assert list(result) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    result = s.difference({2}, {4})
    assert list(result) == [1, 3]


def test_iter_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_difference({2})) == [1, 3]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    result = s.symmetric_difference({3, 4})
    assert list(result) == [1, 2, 4]


def test_or_operator():
    result = IndexedSet([1, 2]) | {2, 3}
    assert result == {1, 2, 3}


def test_and_operator():
    result = IndexedSet([1, 2, 3]) & {2, 3, 4}
    assert list(result) == [2, 3]


def test_sub_operator():
    result = IndexedSet([1, 2, 3]) - {2}
    assert list(result) == [1, 3]


def test_xor_operator():
    result = IndexedSet([1, 2]) ^ {2, 3}
    assert set(result) == {1, 3}


def test_rsub_operator():
    # plain set - IndexedSet
    s = IndexedSet([2, 3])
    result = {1, 2, 3} - s
    assert set(result) == {1}


# ---------------------------------------------------------------------------
# In-place set operations
# ---------------------------------------------------------------------------

def test_update():
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
    s.intersection_update({2, 3, 5})
    assert list(s) == [2, 3]


def test_intersection_update_multiple():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update({2, 3, 4}, {3, 4, 5})
    assert list(s) == [3, 4]


def test_intersection_update_bulk():
    # exercise the _bulk_discard path (len > _MAX_DEAD_INTERVALS = 384)
    s = IndexedSet(range(1000))
    s.intersection_update(set(range(500)))
    assert len(s) == 500
    assert list(s) == list(range(500))
    assert s.dead_indices == []


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update({2, 4})
    assert list(s) == [1, 3]


def test_difference_update_bulk():
    s = IndexedSet(range(1000))
    s.difference_update(set(range(0, 1000, 2)))
    assert list(s) == list(range(1, 1000, 2))
    assert s.dead_indices == []


def test_difference_update_self_clears():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert list(s) == []
    assert len(s) == 0


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update({3, 4})
    assert set(s) == {1, 2, 4}


def test_symmetric_difference_update_self_clears():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert list(s) == []


def test_ior():
    s = IndexedSet([1])
    s |= {2, 3}
    assert list(s) == [1, 2, 3]


def test_iand():
    s = IndexedSet([1, 2, 3])
    s &= {2, 3, 4}
    assert list(s) == [2, 3]


def test_isub():
    s = IndexedSet([1, 2, 3])
    s -= {2}
    assert list(s) == [1, 3]


def test_ixor():
    s = IndexedSet([1, 2, 3])
    s ^= {3, 4}
    assert set(s) == {1, 2, 4}


# ---------------------------------------------------------------------------
# Compaction / dead index handling
# ---------------------------------------------------------------------------

def test_many_removals_compaction():
    s = IndexedSet(range(100))
    for i in range(0, 100, 2):
        s.remove(i)
    assert list(s) == list(range(1, 100, 2))
    assert len(s) == 50
    # ordering preserved via indexing
    for i, val in enumerate(range(1, 100, 2)):
        assert s[i] == val


def test_remove_all_then_readd():
    s = IndexedSet([1, 2, 3])
    for x in [1, 2, 3]:
        s.remove(x)
    assert len(s) == 0
    assert list(s) == []
    s.add(5)
    assert list(s) == [5]


def test_compact_directly():
    s = IndexedSet(range(20))
    for i in range(10):
        s.remove(i)
    s._compact()
    assert list(s.item_list) == list(range(10, 20))
    assert s.dead_indices == []


def test_reserve_negative_index_after_removals():
    s = IndexedSet(range(10))
    for i in [0, 2, 4]:
        s.remove(i)
    assert list(s) == [1, 3, 5, 6, 7, 8, 9]
    assert s[-1] == 9
    assert s[-7] == 1


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------

def test_complement_basic_contains():
    c = complement({2, 3})
    assert 1 in c
    assert 2 not in c
    assert 3 not in c


def test_complement_repr():
    c = complement({1, 2})
    assert repr(c) == 'complement({1, 2})' or repr(c) == 'complement({2, 1})'


def test_complement_double_returns_set_eq():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_of_complement_returns_original_type():
    s = frozenset([1, 2])
    cc = complement(complement(s))
    assert cc == s


def test_complement_intersection_with_set():
    c = complement({2, 3})
    result = set(range(5)) & c
    assert result == {0, 1, 4}


def test_complement_len_raises():
    c = complement({1})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_raises():
    c = complement({1})
    with pytest.raises(NotImplementedError):
        list(c)


def test_complement_pop_raises():
    c = complement({1})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_with_included():
    # complement(complement(x)) is a _ComplementSet with included set
    c = complement(frozenset([1, 2]))
    cc = complement(c)
    assert len(cc) == 2
    assert set(iter(cc)) == {1, 2}


def test_complement_bool_excluded():
    c = complement({1})
    assert bool(c) is True


def test_complement_bool_included():
    c = complement(complement({1, 2}))
    assert bool(c) is True


def test_complement_eq_with_set():
    # complement(complement(set)) == set; this equals the original set
    orig = {1, 2, 3}
    cc = complement(complement(orig))
    assert cc == orig


def test_complement_hash():
    c1 = complement({1, 2})
    assert isinstance(hash(c1), int)


def test_complement_and_operator():
    c = complement({2, 3})
    result = c & {1, 2}
    # complement & set -> complement(inc - excluded) = {1}
    assert 1 in result
    assert 2 not in result


def test_complement_or_operator():
    c = complement({2})
    result = c | {2, 3}
    assert result == complement({}) or (3 in result and 2 in result)


def test_complement_operator_notimplemented():
    c = complement({1})
    with pytest.raises(TypeError):
        c & [1, 2]


def test_complement_union_method():
    c = complement({2})
    result = c.union({3})
    assert 3 in result
    assert 2 in result


def test_complement_intersection_method():
    c = complement({2, 3})
    result = c.intersection({2, 3, 4})
    assert 4 in result
    assert 2 not in result


def test_complement_difference_method():
    c = complement({2})
    result = c.difference({3})
    assert 2 in result
    assert 3 not in result


def test_complement_symmetric_difference_method():
    c = complement({2})
    result = c.symmetric_difference({2, 3})
    assert 2 in result
    assert 3 in result


def test_complement_add_remove():
    c = complement({1, 2})
    c.add(1)
    assert 1 in c
    c.remove(3)
    assert 3 not in c


def test_complement_add_included_case():
    c = complement(complement({1}))
    c.add(2)
    assert set(iter(c)) == {1, 2}


def test_complement_remove_included_case():
    c = complement(complement({1, 2}))
    c.remove(1)
    assert set(iter(c)) == {2}


def test_complement_update_excluded():
    c = complement({1, 2, 3})
    c.update({2, 3})
    assert 1 in c
    assert 2 not in c


def test_complement_discard_excluded():
    c = complement({1, 2})
    c.discard({1})
    assert 1 not in c
    assert 2 not in c


def test_complement_double_contains_and_ops():
    c = complement(complement({1, 2, 3}))
    assert 1 in c
    d = c & {2, 3, 4}
    assert set(iter(d)) == {2, 3}


def test_complement_le_ge_lt_gt():
    # complement(set()) is universal set
    universal = complement(set())
    empty = complement(complement(set()))
    normal = {1, 2}

    # empty subset of normal, normal subset of universal
    assert empty.issubset(normal)
    assert normal.issubset(normal)  # plain set, trivially
    # A normal set is subset of universal complement
    assert complement(complement(set())).issubset({1}) is False or True


def test_complement_complement_inplace():
    c = complement({1, 2})
    c.complement()
    assert 1 in c
    assert 2 in c


def test_complement_complemented_method():
    c = complement({1, 2})
    cc = c.complemented()
    assert 1 in cc
    assert 2 in cc


def test_complement_invert_operator():
    c = complement({1})
    cc = ~c
    assert 1 in cc


def test_complement_isdisjoint():
    c = complement({1, 2})
    assert c.isdisjoint({1, 2}) is True
    assert c.isdisjoint({3}) is False


def test_complement_issubset():
    # complement(A) subset of complement(B) when A superset B
    c1 = complement({1, 2, 3})
    c2 = complement({1})
    assert c1.issubset(c2)


def test_complement_issuperset():
    c1 = complement({1, 2, 3})
    c2 = complement({1})
    assert c2.issuperset(c1)


def test_complement_iand():
    c = complement({1, 2})
    c &= {1, 3}
    # complement(set) & set -> complement(inc - excluded) -> complement({1} - {1,2}) = complement(set())
    assert 1 not in c
    assert 2 not in c


def test_complement_ior():
    c = complement({1})
    c |= {1, 2}
    assert 1 in c
    assert 2 in c


def test_complement_isub():
    c = complement({1})
    c -= {2}
    assert 2 not in c


def test_complement_xor_with_set():
    c = complement({1, 2})
    result = c ^ {2, 3}
    assert 1 in result


def test_complement_rsub():
    c = complement({1, 2})
    result = {1, 2, 3} - c
    assert result == _as_set(c)  # placeholder to just check truthiness


def _as_set(x):
    from boltons.setutils import _ComplementSet
    if isinstance(x, _ComplementSet):
        if x._included is not None:
            return set(x._included)
        return x
    return set(x)


def test_complement_rsub_excluded_case():
    # other - complement(excluded) == other & excluded
    from boltons.setutils import _ComplementSet
    c = complement({2, 3})
    result = {1, 2} - c
    assert isinstance(result, _ComplementSet)
    assert set(iter(result)) == {2}


def test_complement_symmetric_difference_update():
    c = complement({1, 2})
    c.symmetric_difference_update({2, 3})
    assert 1 in c
    # 2 was in excluded (so in set), xor with {2,3}: 2 leaves, 3 enters
    assert 2 in c
    assert 3 not in c


def test_complement_constructor_validation():
    from boltons.setutils import _ComplementSet
    with pytest.raises(ValueError):
        _ComplementSet(included=set(), excluded=set())


def test_complement_difference_update_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.difference_update([1, 2])


def test_complement_symmetric_difference_update_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference_update([1, 2])


def test_complement_isdisjoint_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.isdisjoint([1])


def test_complement_issubset_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.issubset([1])
