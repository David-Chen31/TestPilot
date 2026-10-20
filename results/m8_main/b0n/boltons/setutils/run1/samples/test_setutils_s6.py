import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet: construction, iter, repr, len, contains
# ---------------------------------------------------------------------------

def test_construction_default_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []


def test_construction_with_duplicates_preserves_first_order():
    s = IndexedSet([3, 1, 3, 2, 1, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_from_iterable_classmethod():
    s = IndexedSet.from_iterable(iter('aabbc'))
    assert list(s) == ['a', 'b', 'c']


def test_repr():
    s = IndexedSet([1, 2, 3])
    assert repr(s) == 'IndexedSet([1, 2, 3])'


def test_repr_empty():
    assert repr(IndexedSet()) == 'IndexedSet([])'


def test_contains():
    s = IndexedSet([1, 2, 3])
    assert 1 in s
    assert 4 not in s


def test_iter_skips_missing_after_remove():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert list(s) == [1, 3, 4]


def test_reversed():
    s = IndexedSet([1, 2, 3])
    assert list(reversed(s)) == [3, 2, 1]


def test_reversed_after_remove():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(3)
    assert list(reversed(s)) == [4, 2, 1]


# ---------------------------------------------------------------------------
# add / remove / discard / clear
# ---------------------------------------------------------------------------

def test_add_new_and_existing():
    s = IndexedSet()
    s.add(1)
    s.add(2)
    s.add(1)
    assert list(s) == [1, 2]
    assert len(s) == 2


def test_remove_present():
    s = IndexedSet([1, 2, 3])
    s.remove(2)
    assert list(s) == [1, 3]
    assert 2 not in s


def test_remove_missing_raises_keyerror():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(KeyError):
        s.remove(99)


def test_discard_missing_does_not_raise():
    s = IndexedSet([1, 2, 3])
    s.discard(99)
    assert list(s) == [1, 2, 3]


def test_discard_present():
    s = IndexedSet([1, 2, 3])
    s.discard(2)
    assert list(s) == [1, 3]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []
    assert s.item_list == []
    assert s.dead_indices == []


# ---------------------------------------------------------------------------
# indexing / slicing / getitem
# ---------------------------------------------------------------------------

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
    s = IndexedSet([10, 20, 30])
    with pytest.raises(IndexError):
        s[3]
    with pytest.raises(IndexError):
        s[-4]


def test_getitem_after_removal():
    s = IndexedSet([0, 1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert s[0] == 0
    assert s[1] == 1
    assert s[2] == 3
    assert s[3] == 5


def test_getitem_slice():
    s = IndexedSet(range(6))
    sub = s[1:4]
    assert isinstance(sub, IndexedSet)
    assert list(sub) == [1, 2, 3]


def test_getitem_slice_with_step():
    s = IndexedSet(range(6))
    sub = s[0:6:2]
    assert list(sub) == [0, 2, 4]


def test_getitem_slice_negative_step():
    s = IndexedSet(range(5))
    sub = s[::-1]
    assert list(sub) == [4, 3, 2, 1, 0]


def test_getitem_slice_negative_bounds():
    s = IndexedSet(range(6))
    sub = s[-4:-1]
    assert list(sub) == [2, 3, 4]


def test_slice_after_removals():
    s = IndexedSet(range(6))
    s.remove(1)
    s.remove(3)
    assert list(s[1:3]) == [2, 4]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

def test_pop_default_last():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_minus_one():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_explicit_last_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3


def test_pop_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]


def test_pop_index_zero():
    s = IndexedSet([1, 2, 3])
    assert s.pop(0) == 1
    assert list(s) == [2, 3]


def test_pop_negative_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-2) == 2
    assert list(s) == [1, 3]


def test_pop_out_of_range():
    s = IndexedSet([1, 2, 3])
    with pytest.raises(IndexError):
        s.pop(10)


def test_pop_empty():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


# ---------------------------------------------------------------------------
# count / index
# ---------------------------------------------------------------------------

def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_index():
    s = IndexedSet([10, 20, 30])
    assert s.index(10) == 0
    assert s.index(30) == 2


def test_index_missing():
    s = IndexedSet([10, 20])
    with pytest.raises(ValueError):
        s.index(99)


def test_index_after_removals():
    s = IndexedSet([0, 1, 2, 3, 4])
    s.remove(1)
    s.remove(3)
    assert s.index(2) == 1
    assert s.index(4) == 2


# ---------------------------------------------------------------------------
# reverse / sort
# ---------------------------------------------------------------------------

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
    s = IndexedSet(['aaa', 'a', 'aa'])
    s.sort(key=len, reverse=True)
    assert list(s) == ['aaa', 'aa', 'a']


def test_sort_after_removal():
    s = IndexedSet([5, 3, 4, 1, 2])
    s.remove(3)
    s.sort()
    assert list(s) == [1, 2, 4, 5]


def test_sort_already_sorted_noop():
    s = IndexedSet([1, 2, 3])
    s.sort()
    assert list(s) == [1, 2, 3]


# ---------------------------------------------------------------------------
# equality and set comparisons
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_sensitive():
    a = IndexedSet([1, 2, 3])
    b = IndexedSet([1, 2, 3])
    c = IndexedSet([3, 2, 1])
    assert a == b
    assert not (a == c)


def test_eq_with_regular_set():
    a = IndexedSet([1, 2, 3])
    assert a == {1, 2, 3}
    assert not (a == {1, 2})


def test_eq_with_non_set_returns_false():
    a = IndexedSet([1, 2])
    assert not (a == 5)


def test_eq_with_unhashable_raises_typeerror_returns_false():
    a = IndexedSet([1, 2])
    assert not (a == [[1], [2]])


# ---------------------------------------------------------------------------
# isdisjoint / issubset / issuperset
# ---------------------------------------------------------------------------

def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint([4, 5])
    assert not s.isdisjoint([3, 4])


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset([1, 2, 3])
    assert s.issubset({1, 2})
    assert not s.issubset([1])


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset([1, 2])
    assert s.issuperset({1, 2, 3})
    assert not s.issuperset([1, 2, 3, 4])


# ---------------------------------------------------------------------------
# set operations (union, intersection, difference, symmetric_difference)
# ---------------------------------------------------------------------------

def test_union_returns_new_indexedset():
    s = IndexedSet([1, 2])
    u = s.union([2, 3], [4])
    assert isinstance(u, IndexedSet)
    assert list(u) == [1, 2, 3, 4]
    assert list(s) == [1, 2]


def test_union_operator_multiple_via_operators():
    s = IndexedSet([1, 2])
    assert list(s | {3}) == [1, 2, 3]


def test_ror():
    s = IndexedSet([2, 3])
    result = {1} | s
    assert list(result) == [1, 2, 3]


def test_intersection():
    s = IndexedSet([1, 2, 3, 4])
    i = s.intersection([2, 4, 5])
    assert isinstance(i, IndexedSet)
    assert list(i) == [2, 4]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    i = s.intersection([2, 3, 4], [3, 4, 5])
    assert list(i) == [3, 4]


def test_intersection_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s & {2, 3, 9}) == [2, 3]


def test_rand():
    s = IndexedSet([2, 3])
    assert list({1, 2, 3} & s) == [2, 3]


def test_iter_intersection():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_intersection([2, 3], [3, 4])) == [3]


def test_difference():
    s = IndexedSet([1, 2, 3, 4])
    d = s.difference([2, 4])
    assert isinstance(d, IndexedSet)
    assert list(d) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    d = s.difference([2], [4])
    assert list(d) == [1, 3]


def test_difference_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s - {2}) == [1, 3]


def test_iter_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_difference([2])) == [1, 3]


def test_rsub_with_set():
    s = IndexedSet([2, 3])
    result = {1, 2, 3, 4} - s
    assert result == {1, 4}


def test_rsub_with_list():
    s = IndexedSet([2, 3])
    result = [1, 2, 3, 4] - s
    assert result == [1, 4]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    sd = s.symmetric_difference([2, 3, 4])
    assert isinstance(sd, IndexedSet)
    assert sorted(sd) == [1, 4]


def test_symmetric_difference_operator():
    s = IndexedSet([1, 2, 3])
    assert sorted(s ^ {2, 3, 4}) == [1, 4]


def test_rxor():
    s = IndexedSet([1, 2, 3])
    assert sorted({2, 3, 4} ^ s) == [1, 4]


# ---------------------------------------------------------------------------
# in-place operations
# ---------------------------------------------------------------------------

def test_update():
    s = IndexedSet([1])
    result = s.update([2, 3])
    assert result is None
    assert list(s) == [1, 2, 3]


def test_update_no_args():
    s = IndexedSet([1])
    s.update()
    assert list(s) == [1]


def test_update_multiple_args():
    s = IndexedSet([1])
    s.update([2], [3, 4])
    assert list(s) == [1, 2, 3, 4]


def test_ior():
    s = IndexedSet([1])
    s |= {2, 3}
    assert list(s) == [1, 2, 3]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update([2, 3, 4], [3, 4, 5])
    assert list(s) == [3, 4]


def test_iand():
    s = IndexedSet([1, 2, 3])
    s &= {2, 3, 4}
    assert list(s) == [2, 3]


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update([2, 4])
    assert list(s) == [1, 3]


def test_difference_update_with_self_clears():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert list(s) == []


def test_isub():
    s = IndexedSet([1, 2, 3])
    s -= {2}
    assert list(s) == [1, 3]


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update([2, 3, 4])
    assert sorted(s) == [1, 4]


def test_symmetric_difference_update_self_clears():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert list(s) == []


def test_ixor():
    s = IndexedSet([1, 2, 3])
    s ^= {2, 3, 4}
    assert sorted(s) == [1, 4]


# ---------------------------------------------------------------------------
# large operation triggering bulk discard path
# ---------------------------------------------------------------------------

def test_intersection_update_large_triggers_bulk():
    big = IndexedSet(range(1000))
    big.intersection_update(range(500))
    assert len(big) == 500
    assert list(big) == list(range(500))


def test_difference_update_large_triggers_bulk():
    big = IndexedSet(range(1000))
    big.difference_update(range(500))
    assert list(big) == list(range(500, 1000))


# ---------------------------------------------------------------------------
# many removals triggering compaction / culling
# ---------------------------------------------------------------------------

def test_many_removals_keep_order():
    s = IndexedSet(range(20))
    for i in range(0, 20, 2):
        s.remove(i)
    assert list(s) == list(range(1, 20, 2))
    assert s._compactions >= 1


def test_remove_all_by_one():
    s = IndexedSet(range(10))
    for i in range(10):
        s.remove(i)
    assert list(s) == []
    assert len(s) == 0


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement_set_contains():
    c = complement(set([1, 2]))
    assert 3 in c
    assert 1 not in c
    assert 2 not in c


def test_complement_frozenset():
    c = complement(frozenset([1, 2]))
    assert 3 in c
    assert 1 not in c


def test_complement_repr():
    c = complement(set([1, 2]))
    assert repr(c) == 'complement({1, 2})'


def test_complement_double_returns_set():
    s = {1, 2, 3}
    result = complement(complement(s))
    assert result == s
    assert type(result) is set


def test_complement_of_complement_object_returns_complemented():
    c = complement(set([1]))
    double = complement(c)
    assert double == {1}


def test_complement_intersection_doc_example():
    result = list(set(range(5)) & complement(set([2, 3])))
    assert sorted(result) == [0, 1, 4]


def test_complement_add():
    c = complement(set([1, 2]))
    c.add(1)
    assert 1 in c
    c.add(5)
    assert 5 in c


def test_complement_remove():
    c = complement(set([1, 2]))
    c.remove(1)
    assert 1 not in c
    c.remove(7)
    assert 7 not in c


def test_complement_pop_undefined():
    c = complement(set([1]))
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_len_undefined():
    c = complement(set([1]))
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_undefined():
    c = complement(set([1]))
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_bool():
    assert complement(set([1])) is not None
    assert bool(complement(set())) is True
    regular = complement(set())
    double = complement(regular)
    assert bool(double) is False


def test_complement_included_set_bool_and_len():
    c = complement(set([1, 2]))
    inner = complement(c)  # now included={1,2}
    assert len(inner) == 2
    assert sorted(inner) == [1, 2]
    assert bool(inner) is True


def test_complement_and():
    a = complement(set([1, 2]))
    b = complement(set([2, 3]))
    result = a & b
    assert isinstance(result, type(a))
    assert 1 in result
    assert 2 not in result
    assert 3 in result


def test_complement_or():
    a = complement(set([1, 2]))
    b = complement(set([2, 3]))
    result = a | b
    assert 1 not in result
    assert 2 not in result
    assert 3 not in result
    assert 4 in result


def test_complement_xor():
    c = complement(set([1, 2]))
    result = c ^ {2}
    assert 1 not in result
    assert 3 in result


def test_complement_sub():
    c = complement(set([1, 2]))
    result = c - {3}
    assert 1 not in result
    assert 3 not in result
    assert 4 in result


def test_complement_rsub():
    c = complement(set([1, 2]))
    result = {2, 3} - c
    assert result == {2}


def test_complement_iand():
    c = complement(set([1, 2]))
    c &= {3, 4}
    assert 1 not in c
    assert 3 in c


def test_complement_ior():
    c = complement(set([1, 2]))
    c |= {2, 3}
    assert 1 in c
    assert 2 not in c


def test_complement_isub():
    c = complement(set([1, 2]))
    c -= {3}
    assert 1 not in c
    assert 3 not in c


def test_complement_ixor():
    c = complement(set([1, 2]))
    c ^= {2}
    assert 1 in c
    assert 2 not in c


def test_complement_update():
    c = complement(set([1, 2, 3]))
    c.update([1])
    assert 1 not in c
    assert 2 in c


def test_complement_discard():
    c = complement(set([1, 2]))
    c.discard([1])
    assert 1 in c


def test_complement_isdisjoint():
    c = complement(set([1, 2]))
    assert c.isdisjoint({1, 2})
    assert not c.isdisjoint({3})


def test_complement_issubset():
    c = complement(set([1, 2]))
    assert c.issubset(complement(set([1])))
    assert not c.issubset(complement(set([1, 2, 3])))


def test_complement_issuperset():
    c = complement(set([1, 2]))
    assert c.issuperset(complement(set([1, 2, 3])))
    assert not c.issuperset(complement(set([1])))


def test_complement_comparison_operators():
    a = complement(set([1, 2]))
    b = complement(set([1, 2, 3]))
    assert a < b is True or a < b is False  # exercises __lt__
    assert (a <= a) is True
    assert (a >= a) is True
    assert (a > a) is False


def test_complement_le_gt():
    a = complement(set([1, 2]))
    b = complement(set([1, 2, 3]))
    assert a <= b
    assert not (a > b)


def test_complement_eq_with_set():
    c = complement(set([1]))
    assert not (c == {2})
    assert (complement(complement(set([1, 2]))) == {1, 2})


def test_complement_eq_with_complement():
    a = complement(set([1]))
    b = complement(set([1]))
    assert a == b
    assert not (a == complement(set([2])))


def test_complement_hash():
    c = complement(set([1]))
    assert hash(c) == (hash(set([1])) ^ hash(None))


def test_complement_complement_method():
    c = complement(set([1]))
    c.complement()
    assert 1 in c
    assert 2 not in c


def test_complement_complemented_method():
    c = complement(set([1]))
    result = c.complemented()
    assert 1 in result


def test_complement_invert():
    c = complement(set([1]))
    result = ~c
    assert 1 in result


def test_complement_intersection_method():
    c = complement(set([1, 2]))
    result = c.intersection({2, 3})
    assert isinstance(result, type(c))


def test_complement_union_method():
    c = complement(set([1, 2]))
    result = c.union({2, 3})
    assert isinstance(result, type(c))


def test_complement_difference_method():
    c = complement(set([1, 2]))
    result = c.difference({3})
    assert isinstance(result, type(c))


def test_complement_symmetric_difference_method():
    c = complement(set([1, 2]))
    result = c.symmetric_difference({2})
    assert isinstance(result, type(c))


def test_complement_invalid_intersection_raises_typeerror():
    c = complement(set([1]))
    with pytest.raises(TypeError):
        c.intersection([1, 2])


def test_complement_invalid_update_with_complement():
    c = complement(set([1, 2]))
    c.update(complement(set([1])))
    assert 1 in c


def test_complement_operators_with_plain_set_both_sides():
    c = complement(set([1]))
    assert 2 in (c | {2})


def test_complement_symmetric_difference_update():
    c = complement(set([1, 2]))
    c.symmetric_difference_update({2, 3})
    assert 1 in c
    assert 2 not in c


def test_complement_difference_update():
    c = complement(set([1, 2]))
    c.difference_update({3})
    assert 1 not in c
    assert 3 not in c


def test_complement_slot_included_and_excluded():
    c = complement(set([1]))
    d = complement(frozenset([1]))
    assert c == d


def test_complement_complemented_frozenset():
    f = frozenset([1, 2])
    c = complement(f)
    result = complement(c)
    assert result == f
