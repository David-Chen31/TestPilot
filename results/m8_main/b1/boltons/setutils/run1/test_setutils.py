import pytest
from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# Construction / basic protocol
# ---------------------------------------------------------------------------
def test_init_empty():
    s = IndexedSet()
    assert len(s) == 0
    assert list(s) == []
    assert s.item_list == []
    assert s.item_index_map == {}


def test_init_from_iterable_dedupes_preserving_order():
    s = IndexedSet([3, 1, 3, 2, 1, 4])
    assert list(s) == [3, 1, 2, 4]
    assert len(s) == 4


def test_init_from_string():
    s = IndexedSet("freecreditreport.com")
    assert list(s) == list(dict.fromkeys("freecreditreport.com"))


def test_init_falsy_other():
    assert list(IndexedSet([])) == []
    assert list(IndexedSet(())) == []


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


def test_eq_indexedset():
    assert IndexedSet([1, 2, 3]) == IndexedSet([1, 2, 3])
    assert IndexedSet([1, 2, 3]) != IndexedSet([1, 3, 2])


def test_eq_with_set():
    assert IndexedSet([1, 2, 3]) == {3, 2, 1}
    assert IndexedSet([1, 2]) == frozenset([1, 2])
    assert IndexedSet([1, 2]) != {1, 2, 3}


def test_eq_typeerror_returns_false():
    assert IndexedSet([1, 2]) != 5


def test_from_iterable():
    s = IndexedSet.from_iterable([1, 2, 2, 3])
    assert isinstance(s, IndexedSet)
    assert list(s) == [1, 2, 3]


# ---------------------------------------------------------------------------
# add / remove / discard / clear / pop
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


def test_remove_missing_raises():
    s = IndexedSet([1])
    with pytest.raises(KeyError):
        s.remove(5)


def test_discard_missing_no_raise():
    s = IndexedSet([1, 2])
    s.discard(99)
    assert list(s) == [1, 2]
    s.discard(1)
    assert list(s) == [2]


def test_clear():
    s = IndexedSet([1, 2, 3])
    s.clear()
    assert len(s) == 0
    assert list(s) == []
    assert s.item_list == []
    assert s.dead_indices == []


def test_pop_default_last():
    s = IndexedSet([1, 2, 3])
    assert s.pop() == 3
    assert list(s) == [1, 2]


def test_pop_minus_one():
    s = IndexedSet([1, 2, 3])
    assert s.pop(-1) == 3
    assert list(s) == [1, 2]


def test_pop_last_index():
    s = IndexedSet([1, 2, 3])
    assert s.pop(2) == 3


def test_pop_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(1) == 2
    assert list(s) == [1, 3, 4]
    assert len(s) == 3


def test_pop_negative_middle():
    s = IndexedSet([1, 2, 3, 4])
    assert s.pop(-2) == 3
    assert list(s) == [1, 2, 4]


def test_pop_empty_raises():
    s = IndexedSet()
    with pytest.raises(IndexError):
        s.pop()


def test_pop_out_of_range():
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s.pop(5)


# ---------------------------------------------------------------------------
# indexing / slicing
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
    s = IndexedSet([1, 2])
    with pytest.raises(IndexError):
        s[5]
    with pytest.raises(IndexError):
        s[-5]


def test_getitem_after_removals():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(2)
    s.remove(4)
    assert list(s) == [1, 3, 5]
    assert s[0] == 1
    assert s[1] == 3
    assert s[2] == 5
    assert s[-1] == 5


def test_getitem_slice_returns_indexedset():
    s = IndexedSet([0, 1, 2, 3, 4])
    r = s[1:4]
    assert isinstance(r, IndexedSet)
    assert list(r) == [1, 2, 3]


def test_getitem_slice_step():
    s = IndexedSet(range(6))
    assert list(s[0:6:2]) == [0, 2, 4]


def test_getitem_slice_negative_step():
    s = IndexedSet([0, 1, 2, 3, 4])
    assert list(s[2:4:-1]) == [2, 1]


def test_getitem_negative_slice_bounds():
    s = IndexedSet(range(6))
    assert list(s[-2:]) == [4, 5]
    assert list(s[:-2]) == [0, 1, 2, 3]


def test_index():
    s = IndexedSet([10, 20, 30])
    assert s.index(10) == 0
    assert s.index(30) == 2


def test_index_missing_raises():
    s = IndexedSet([1, 2])
    with pytest.raises(ValueError):
        s.index(99)


def test_index_after_removals():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert s.index(3) == 1
    assert s.index(4) == 2


# ---------------------------------------------------------------------------
# count / reverse / sort
# ---------------------------------------------------------------------------
def test_count():
    s = IndexedSet([1, 2, 3])
    assert s.count(1) == 1
    assert s.count(99) == 0


def test_reverse():
    s = IndexedSet([1, 2, 3])
    s.reverse()
    assert list(s) == [3, 2, 1]
    assert s[0] == 3
    assert s.dead_indices == []


def test_reverse_with_dead_indices():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    s.reverse()
    assert list(s) == [4, 3, 1]
    assert s.dead_indices == []
    assert s[0] == 4


def test_sort():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]


def test_sort_reverse_kwarg():
    s = IndexedSet([3, 1, 2])
    s.sort(reverse=True)
    assert list(s) == [3, 2, 1]


def test_sort_key():
    s = IndexedSet([3, 1, 2])
    s.sort(key=lambda x: -x)
    assert list(s) == [3, 2, 1]


def test_sort_already_sorted_no_change():
    s = IndexedSet([1, 2, 3])
    before = list(s.item_list)
    s.sort()
    assert s.item_list == before


# ---------------------------------------------------------------------------
# subset / superset / disjoint
# ---------------------------------------------------------------------------
def test_isdisjoint():
    s = IndexedSet([1, 2, 3])
    assert s.isdisjoint([4, 5])
    assert not s.isdisjoint([3, 4])


def test_issubset():
    s = IndexedSet([1, 2])
    assert s.issubset({1, 2, 3})
    assert not s.issubset({1})
    assert not s.issubset({0, 1})


def test_issuperset():
    s = IndexedSet([1, 2, 3])
    assert s.issuperset({1, 2})
    assert not s.issuperset({1, 2, 3, 4})
    assert not s.issuperset({0})


# ---------------------------------------------------------------------------
# set operations
# ---------------------------------------------------------------------------
def test_union():
    s = IndexedSet([1, 2])
    r = s.union([3, 4], {2, 5})
    assert isinstance(r, IndexedSet)
    assert list(r) == [1, 2, 3, 4, 5]


def test_or_operator():
    s = IndexedSet([1, 2])
    r = s | {3}
    assert list(r) == [1, 2, 3]


def test_intersection_single():
    s = IndexedSet([1, 2, 3])
    assert list(s.intersection({2, 3, 4})) == [2, 3]


def test_intersection_multiple():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.intersection({2, 3, 4}, {3, 4, 5})) == [3, 4]


def test_and_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s & {2, 3}) == [2, 3]


def test_iter_intersection():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_intersection({2, 3}, {2})) == [2]


def test_difference_single():
    s = IndexedSet([1, 2, 3])
    assert list(s.difference({2})) == [1, 3]


def test_difference_multiple():
    s = IndexedSet([1, 2, 3, 4])
    assert list(s.difference({1}, {4})) == [2, 3]


def test_sub_operator():
    s = IndexedSet([1, 2, 3])
    assert list(s - {2}) == [1, 3]


def test_iter_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.iter_difference({1}, {3})) == [2]


def test_symmetric_difference():
    s = IndexedSet([1, 2, 3])
    assert list(s.symmetric_difference({2, 3, 4})) == [1, 4]


def test_xor_operator():
    s = IndexedSet([1, 2])
    assert list(s ^ {2, 3}) == [1, 3]


def test_rsub():
    s = IndexedSet([2, 3])
    r = [1, 2, 3, 4] - s
    assert list(r) == [1, 4]


# ---------------------------------------------------------------------------
# in-place set operations
# ---------------------------------------------------------------------------
def test_update():
    s = IndexedSet([1])
    s.update([2, 3], {1, 4})
    assert list(s) == [1, 2, 3, 4]


def test_update_no_args():
    s = IndexedSet([1])
    s.update()
    assert list(s) == [1]


def test_intersection_update():
    s = IndexedSet([1, 2, 3, 4])
    s.intersection_update({2, 3, 4}, {3, 4, 5})
    assert list(s) == [2, 3, 4]


def test_intersection_update_large():
    s = IndexedSet(range(400))
    s.intersection_update(set(range(0, 400, 2)))
    assert list(s) == list(range(0, 400, 2))


def test_difference_update():
    s = IndexedSet([1, 2, 3, 4])
    s.difference_update({2}, {4})
    assert list(s) == [1, 2, 3, 4]


def test_difference_update_self_clears():
    s = IndexedSet([1, 2, 3])
    s.difference_update(s)
    assert list(s) == []


def test_difference_update_large():
    s = IndexedSet(range(400))
    s.difference_update(set(range(0, 400, 2)))
    assert list(s) == list(range(1, 400, 2))


def test_symmetric_difference_update():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update({2, 3, 4})
    assert list(s) == [1, 4]


def test_symmetric_difference_update_self():
    s = IndexedSet([1, 2, 3])
    s.symmetric_difference_update(s)
    assert list(s) == []


def test_ior():
    s = IndexedSet([1, 2])
    orig = s
    s |= {3}
    assert s is orig
    assert list(s) == [1, 2, 3]


def test_iand():
    s = IndexedSet([1, 2, 3])
    s &= {2, 3}
    assert list(s) == [2, 3]


def test_isub():
    s = IndexedSet([1, 2, 3])
    s -= {2}
    assert list(s) == [1, 3]


def test_ixor():
    s = IndexedSet([1, 2, 3])
    s ^= {2, 4}
    assert list(s) == [1, 3, 4]


def test_ior_returns_self():
    s = IndexedSet([1])
    assert s.__ior__({2}) is s


def test_iand_returns_self():
    s = IndexedSet([1])
    assert s.__iand__({1}) is s


def test_isub_returns_self():
    s = IndexedSet([1])
    assert s.__isub__({1}) is s


def test_ixor_returns_self():
    s = IndexedSet([1])
    assert s.__ixor__({1}) is s


# ---------------------------------------------------------------------------
# compaction / dead-index culling
# ---------------------------------------------------------------------------
def test_many_removals_compact():
    s = IndexedSet(range(2000))
    for i in range(0, 2000, 2):
        s.remove(i)
    assert list(s) == list(range(1, 2000, 2))
    assert len(s) == 1000


def test_remove_right_hand_side():
    s = IndexedSet([1, 2, 3, 4, 5])
    s.remove(5)
    assert list(s) == [1, 2, 3, 4]
    assert s.item_list[-1] == 4


def test_compact_increments_counter():
    s = IndexedSet(range(100))
    s._compactions = 0
    for i in range(50):
        s.remove(i)
    assert s._compactions >= 1


def test_cull_empty_after_all_removed():
    s = IndexedSet([1])
    s.remove(1)
    assert len(s) == 0
    assert s.item_list == []
    assert s.dead_indices == []


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------
def test_complement_basic():
    c = complement({2, 3})
    assert 1 in c
    assert 2 not in c


def test_complement_of_empty_universal():
    c = complement(set())
    assert 1 in c
    assert 0 in c


def test_complement_double_returns_set():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_repr():
    c = complement({1, 2})
    assert repr(c) == "complement({1, 2})"


def test_complement_intersection_with_set():
    c = complement({2, 3})
    assert set(range(5)) & c == {0, 1, 4}


def test_complement_len_included():
    c = complement(complement({1, 2, 3}))
    assert len(c) == 3
    assert set(c) == {1, 2, 3}


def test_complement_len_excluded_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_excluded_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_bool():
    assert bool(complement(set())) is True
    assert bool(complement({1})) is True
    assert bool(complement(complement(set()))) is False


def test_complement_add_remove():
    c = complement({2, 3})
    c.add(2)
    assert 2 in c
    c.remove(3)
    assert 3 not in c


def test_complement_add_included():
    c = complement(complement({1}))
    c.add(2)
    assert 2 in c
    assert set(c) == {1, 2}


def test_complement_remove_included():
    c = complement(complement({1, 2}))
    c.remove(1)
    assert 1 not in c
    assert set(c) == {2}


def test_complement_pop_included():
    c = complement(complement({1}))
    assert c.pop() == 1
    with pytest.raises(KeyError):
        c.pop()


def test_complement_pop_excluded():
    c = complement({1})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_union():
    c = complement({1, 2})
    r = c | {1}
    assert 1 in r
    assert 2 not in r


def test_complement_intersection_ops():
    c = complement({1, 2})
    r = c & {1, 3}
    assert set(r) == {3}


def test_complement_difference():
    c = complement({1, 2})
    r = c - {3}
    assert 1 not in r
    assert 2 not in r
    assert 3 not in r
    assert 4 in r


def test_complement_symmetric_difference():
    c = complement({1, 2})
    r = c ^ {2}
    assert 1 not in r
    assert 2 in r
    assert 3 in r


def test_complement_complemented_method():
    c = complement({1, 2})
    inv = c.complemented()
    assert inv == complement(complement({1, 2}))


def test_complement_invert_operator():
    c = complement({1, 2})
    inv = ~c
    assert 1 in inv
    assert 2 in inv
    assert 3 not in inv


def test_complement_complement_inplace():
    c = complement({1, 2})
    c.complement()
    assert 1 in c
    assert 3 not in c


def test_complement_isdisjoint():
    c = complement({1, 2})
    assert c.isdisjoint({1, 2})
    assert not c.isdisjoint({3})


def test_complement_issubset():
    c = complement({1, 2})
    assert c.issubset(complement({1}))
    assert not c.issubset(complement({1, 2, 3}))


def test_complement_issuperset():
    c = complement({1, 2})
    assert c.issuperset(complement({1, 2, 3}))
    assert not c.issuperset(complement({1}))


def test_complement_le_ge():
    a = complement({1, 2})
    b = complement({1})
    assert a <= b
    assert b >= a
    assert a < b
    assert b > a


def test_complement_eq_with_set():
    c = complement(complement({1, 2, 3}))
    assert c == {1, 2, 3}
    assert c == frozenset({1, 2, 3})


def test_complement_eq_complement():
    assert complement({1}) == complement({1})
    assert complement({1}) != complement({2})


def test_complement_hash():
    c = complement(frozenset({1}))
    assert isinstance(hash(c), int)


def test_complement_typeerror_on_bad_operand():
    c = complement({1})
    with pytest.raises(TypeError):
        c.intersection([1, 2])


def test_complement_union_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.union([1, 2])


def test_complement_symmetric_difference_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference([1, 2])


def test_complement_difference_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.difference([1, 2])


def test_complement_update_inplace():
    c = complement({1, 2, 3})
    c |= {1}
    assert 1 in c
    assert 2 not in c


def test_complement_update_with_complement():
    c = complement({1, 2, 3})
    c.update(complement({1, 2}))
    assert 3 not in c


def test_complement_update_plain_iterable():
    c = complement({1, 2, 3})
    c.update([1, 2])
    assert 1 not in c
    assert 2 not in c
    assert 3 in c


def test_complement_discard():
    # discard() on an excluded-complement takes the `exc is None` branch
    # and performs `self._excluded.update(inc)`, adding the discarded
    # elements to the exclusion set (items REMAIN excluded).
    c = complement({1, 2})
    c.discard({1})
    assert 1 not in c
    assert 2 not in c


def test_complement_discard_iterable():
    c = complement({1, 2, 3})
    c.discard([1, 2])
    assert 1 not in c
    assert 2 not in c
    assert 3 not in c


def test_complement_update_returns():
    c = complement({1})
    c2 = complement(complement({1}))
    c2.update({1, 2})
    assert set(c2) == {1, 2}


def test_complement_update_method_on_included_by_complement():
    # self is an included-complement (_included={1,2}, _excluded=None).
    # Updated by an excluded-complement (_included=None, _excluded={1}).
    # In update(): inc is None (other._included), exc={1}. Since
    # self._included is not None and inc is None, it runs
    #   self._included &= exc        -> {1,2} &= {1}  -> {1}
    #   self._included, self._excluded = None, self._excluded  -> None, None
    # No error is raised. Verify the resulting internal state.
    c = complement(complement({1, 2}))
    c.update(complement({1}))
    assert c._included is None
    assert c._excluded is None


def test_complement_update_method_excluded_by_complement():
    c = complement({1, 2, 3})
    c.update(complement({1, 2}))
    assert 3 not in c


def test_complement_symmetric_difference_update_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.symmetric_difference_update([1, 2])


def test_complement_isdisjoint_typeerror():
    c = complement({1})
    with pytest.raises(TypeError):
        c.isdisjoint([1, 2])
