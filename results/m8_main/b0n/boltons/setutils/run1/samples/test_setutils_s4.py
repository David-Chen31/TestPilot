import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet basics
# ---------------------------------------------------------------------------

def test_init_empty():
    x = IndexedSet()
    assert len(x) == 0
    assert list(x) == []


def test_init_with_iterable_dedupes_preserving_order():
    x = IndexedSet([3, 1, 3, 2, 1, 4])
    assert list(x) == [3, 1, 2, 4]
    assert len(x) == 4


def test_init_with_falsey_other():
    assert list(IndexedSet([])) == []
    assert list(IndexedSet("")) == []


def test_repr():
    x = IndexedSet([1, 2, 3])
    assert repr(x) == "IndexedSet([1, 2, 3])"
    assert repr(IndexedSet()) == "IndexedSet([])"


def test_from_iterable():
    x = IndexedSet.from_iterable([1, 2, 2, 3])
    assert isinstance(x, IndexedSet)
    assert list(x) == [1, 2, 3]


def test_contains():
    x = IndexedSet([1, 2, 3])
    assert 1 in x
    assert 4 not in x


def test_iter():
    x = IndexedSet([5, 4, 3])
    assert list(x) == [5, 4, 3]


def test_reversed():
    x = IndexedSet([5, 4, 3])
    assert list(reversed(x)) == [3, 4, 5]


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_eq_indexedset_order_matters():
    a = IndexedSet([1, 2, 3])
    b = IndexedSet([1, 2, 3])
    c = IndexedSet([3, 2, 1])
    assert a == b
    assert not (a == c)


def test_eq_with_set_ignores_order():
    a = IndexedSet([1, 2, 3])
    assert a == {1, 2, 3}
    assert a == frozenset({1, 2, 3})
    assert not (a == {1, 2})


def test_eq_with_non_iterable():
    a = IndexedSet([1, 2, 3])
    assert (a == 5) is False


def test_eq_with_unhashable_iterable():
    a = IndexedSet([1, 2, 3])
    # list of lists -> set() raises TypeError -> False
    assert (a == [[1], [2]]) is False


# ---------------------------------------------------------------------------
# add / remove / discard / clear / pop
# ---------------------------------------------------------------------------

def test_add():
    x = IndexedSet()
    x.add(1)
    x.add(2)
    x.add(1)
    assert list(x) == [1, 2]
    assert len(x) == 2


def test_remove():
    x = IndexedSet([1, 2, 3])
    x.remove(2)
    assert list(x) == [1, 3]
    assert 2 not in x
    assert len(x) == 2


def test_remove_missing_raises():
    x = IndexedSet([1, 2, 3])
    with pytest.raises(KeyError):
        x.remove(99)


def test_discard_existing_and_missing():
    x = IndexedSet([1, 2, 3])
    x.discard(2)
    assert list(x) == [1, 3]
    x.discard(99)  # no error
    assert list(x) == [1, 3]


def test_clear():
    x = IndexedSet([1, 2, 3])
    x.clear()
    assert len(x) == 0
    assert list(x) == []
    assert x.item_list == []
    assert x.item_index_map == {}
    assert x.dead_indices == []


def test_pop_default_removes_last():
    x = IndexedSet([1, 2, 3])
    assert x.pop() == 3
    assert list(x) == [1, 2]


def test_pop_last_with_index():
    x = IndexedSet([1, 2, 3])
    assert x.pop(-1) == 3
    assert list(x) == [1, 2]


def test_pop_middle():
    x = IndexedSet([1, 2, 3, 4])
    assert x.pop(1) == 2
    assert list(x) == [1, 3, 4]
    assert x.index(3) == 1


def test_pop_index_out_of_range():
    x = IndexedSet([1, 2, 3])
    with pytest.raises(IndexError):
        x.pop(10)


# ---------------------------------------------------------------------------
# Indexing / slicing
# ---------------------------------------------------------------------------

def test_getitem_positive():
    x = IndexedSet([10, 20, 30])
    assert x[0] == 10
    assert x[1] == 20
    assert x[2] == 30


def test_getitem_negative():
    x = IndexedSet([10, 20, 30])
    assert x[-1] == 30
    assert x[-3] == 10


def test_getitem_out_of_range():
    x = IndexedSet([10, 20, 30])
    with pytest.raises(IndexError):
        x[3]
    with pytest.raises(IndexError):
        x[-4]


def test_getitem_after_removal():
    x = IndexedSet(range(10))
    x.remove(3)
    x.remove(7)
    assert x[3] == 4
    assert x[6] == 8
    assert list(x) == [0, 1, 2, 4, 5, 6, 8, 9]


def test_slicing():
    x = IndexedSet(range(6))
    res = x[1:4]
    assert isinstance(res, IndexedSet)
    assert list(res) == [1, 2, 3]


def test_slicing_with_step():
    x = IndexedSet(range(6))
    assert list(x[::2]) == [0, 2, 4]
    assert list(x[1::2]) == [1, 3, 5]


def test_slicing_negative():
    x = IndexedSet(range(6))
    assert list(x[-2:]) == [4, 5]
    assert list(x[:-2]) == [0, 1, 2, 3]


def test_slicing_negative_step():
    x = IndexedSet(range(6))
    # x[2:4:-1] == reversed(x)[2:4] per docs
    assert list(x[2:4:-1]) == [3, 2]


def test_slicing_after_removal():
    x = IndexedSet(range(10))
    x.remove(2)
    x.remove(5)
    assert list(x[2:6]) == [3, 4, 6, 7]


# ---------------------------------------------------------------------------
# count / index
# ---------------------------------------------------------------------------

def test_count():
    x = IndexedSet([1, 2, 3])
    assert x.count(1) == 1
    assert x.count(99) == 0


def test_index():
    x = IndexedSet([10, 20, 30])
    assert x.index(20) == 1


def test_index_after_removal():
    x = IndexedSet(range(10))
    x.remove(2)
    x.remove(3)
    assert x.index(4) == 2
    assert x.index(5) == 3


def test_index_missing_raises_value_error():
    x = IndexedSet([1, 2, 3])
    with pytest.raises(ValueError):
        x.index(99)


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def test_isdisjoint():
    x = IndexedSet([1, 2, 3])
    assert x.isdisjoint([4, 5]) is True
    assert x.isdisjoint([3, 4]) is False


def test_issubset():
    x = IndexedSet([1, 2])
    assert x.issubset({1, 2, 3}) is True
    assert x.issubset({1}) is False


def test_issuperset():
    x = IndexedSet([1, 2, 3])
    assert x.issuperset({1, 2}) is True
    assert x.issuperset({1, 5}) is False


def test_union():
    x = IndexedSet([1, 2])
    res = x.union({2, 3}, [3, 4])
    assert isinstance(res, IndexedSet)
    assert list(res) == [1, 2, 3, 4]


def test_or_operator():
    x = IndexedSet([1, 2])
    res = x | {3}
    assert list(res) == [1, 2, 3]


def test_intersection():
    x = IndexedSet([1, 2, 3, 4])
    res = x.intersection({2, 4, 5})
    assert list(res) == [2, 4]


def test_intersection_multiple():
    x = IndexedSet([1, 2, 3, 4])
    res = x.intersection({2, 3, 4}, {3, 4, 5})
    assert list(res) == [3, 4]


def test_and_operator():
    x = IndexedSet([1, 2, 3])
    res = x & {2, 3, 4}
    assert list(res) == [2, 3]


def test_difference():
    x = IndexedSet([1, 2, 3, 4])
    res = x.difference({2, 4})
    assert list(res) == [1, 3]


def test_difference_multiple():
    x = IndexedSet([1, 2, 3, 4])
    res = x.difference({1}, {4})
    assert list(res) == [2, 3]


def test_sub_operator():
    x = IndexedSet([1, 2, 3])
    res = x - {2}
    assert list(res) == [1, 3]


def test_rsub_operator():
    x = IndexedSet([2, 3])
    res = {1, 2, 3, 4} - x
    assert res == {1, 4}


def test_symmetric_difference():
    x = IndexedSet([1, 2, 3])
    res = x.symmetric_difference({2, 3, 4})
    assert list(res) == [1, 4]


def test_xor_operator():
    x = IndexedSet([1, 2, 3])
    res = x ^ {2, 3, 4}
    assert set(res) == {1, 4}


def test_iter_intersection():
    x = IndexedSet([1, 2, 3, 4])
    assert list(x.iter_intersection({2, 4})) == [2, 4]


def test_iter_difference():
    x = IndexedSet([1, 2, 3, 4])
    assert list(x.iter_difference({2, 4})) == [1, 3]


# ---------------------------------------------------------------------------
# In-place set operations
# ---------------------------------------------------------------------------

def test_update():
    x = IndexedSet([1])
    x.update([2, 3], [3, 4])
    assert list(x) == [1, 2, 3, 4]


def test_update_no_args():
    x = IndexedSet([1])
    x.update()
    assert list(x) == [1]


def test_ior():
    x = IndexedSet([1])
    x |= {2, 3}
    assert list(x) == [1, 2, 3]


def test_intersection_update():
    x = IndexedSet([1, 2, 3, 4])
    x.intersection_update({2, 4, 5})
    assert list(x) == [2, 4]


def test_iand():
    x = IndexedSet([1, 2, 3, 4])
    x &= {2, 4}
    assert list(x) == [2, 4]


def test_difference_update():
    x = IndexedSet([1, 2, 3, 4])
    x.difference_update({2, 4})
    assert list(x) == [1, 3]


def test_difference_update_self():
    x = IndexedSet([1, 2, 3])
    x.difference_update(x)
    assert len(x) == 0
    assert list(x) == []


def test_isub():
    x = IndexedSet([1, 2, 3])
    x -= {2}
    assert list(x) == [1, 3]


def test_symmetric_difference_update():
    x = IndexedSet([1, 2, 3])
    x.symmetric_difference_update({2, 3, 4})
    assert list(x) == [1, 4]


def test_symmetric_difference_update_self():
    x = IndexedSet([1, 2, 3])
    x.symmetric_difference_update(x)
    # after clear, other is iterated; here it's the same object
    # clear empties, then iteration over cleared set yields nothing
    assert list(x) == []


def test_ixor():
    x = IndexedSet([1, 2, 3])
    x ^= {2, 3, 4}
    assert list(x) == [1, 4]


# ---------------------------------------------------------------------------
# reverse / sort
# ---------------------------------------------------------------------------

def test_reverse():
    x = IndexedSet([1, 2, 3])
    x.reverse()
    assert list(x) == [3, 2, 1]
    assert x[0] == 3


def test_sort():
    x = IndexedSet([3, 1, 2])
    x.sort()
    assert list(x) == [1, 2, 3]
    assert x[0] == 1


def test_sort_with_key():
    x = IndexedSet(["aaa", "b", "cc"])
    x.sort(key=len)
    assert list(x) == ["b", "cc", "aaa"]


def test_sort_after_removal():
    x = IndexedSet([5, 3, 1, 4, 2])
    x.remove(3)
    x.remove(4)
    x.sort()
    assert list(x) == [1, 2, 5]


def test_reverse_after_removal():
    x = IndexedSet([1, 2, 3, 4])
    x.remove(2)
    x.reverse()
    assert list(x) == [4, 3, 1]


# ---------------------------------------------------------------------------
# Internal compaction behaviors (via public API)
# ---------------------------------------------------------------------------

def test_many_removals_keep_consistency():
    x = IndexedSet(range(1000))
    for i in range(0, 1000, 2):
        x.remove(i)
    expected = list(range(1, 1000, 2))
    assert list(x) == expected
    assert len(x) == len(expected)
    for i, v in enumerate(expected):
        assert x[i] == v


def test_bulk_discard_via_intersection_update():
    x = IndexedSet(range(1000))
    x.intersection_update(range(0, 1000, 2))
    assert list(x) == list(range(0, 1000, 2))


def test_bulk_discard_via_difference_update():
    x = IndexedSet(range(1000))
    x.difference_update(range(0, 1000, 2))
    assert list(x) == list(range(1, 1000, 2))


def test_dead_index_after_middle_removals():
    x = IndexedSet(range(5))
    x.remove(1)
    x.remove(3)
    assert list(x) == [0, 2, 4]
    assert x[0] == 0
    assert x[1] == 2
    assert x[2] == 4
    assert x.index(2) == 1
    assert x.index(4) == 2


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

def test_complement_repr():
    c = complement({1, 2})
    assert repr(c) == "complement({1, 2})"


def test_complement_contains():
    c = complement({1, 2})
    assert 3 in c
    assert 1 not in c


def test_complement_double_negation():
    s = {1, 2, 3}
    assert complement(complement(s)) == s


def test_complement_double_via_invert():
    s = {1, 2, 3}
    c = complement(s)
    assert (~c) == s


def test_complement_intersection():
    s = set(range(5))
    c = complement({2, 3})
    assert list(s & c) == [0, 1, 4]


def test_complement_union():
    a = complement({1})
    b = complement({2})
    res = a | b
    # complement({1}) | complement({2}) => complement({1} & {2}) = complement(set())
    assert res == set()
    assert 100 in res


def test_complement_add_remove():
    c = complement({1, 2})
    c.add(1)
    assert 1 in c
    c.remove(1)
    assert 1 not in c


def test_complement_isdisjoint():
    c = complement({1, 2})
    assert c.isdisjoint({3, 4}) is True
    assert c.isdisjoint({1, 3}) is False


def test_complement_issubset():
    c1 = complement({1, 2, 3})
    c2 = complement({1, 2})
    assert c1.issubset(c2) is True


def test_complement_issuperset():
    c1 = complement({1, 2})
    c2 = complement({1, 2, 3})
    assert c1.issuperset(c2) is True


def test_complement_difference():
    a = complement({1, 2, 3})
    b = complement({2, 3, 4})
    res = a - b
    # {1,2,3} excluded a; {2,3,4} excluded b; a - b = excluded elements not in b
    assert res == {4}


def test_complement_symmetric_difference():
    a = complement({1, 2})
    b = complement({2, 3})
    res = a ^ b
    assert res == {1, 3}


def test_complement_eq_with_set():
    c = complement({1, 2})
    assert not (c == {3, 4})


def test_complement_len_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        len(c)


def test_complement_iter_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        iter(c)


def test_complement_pop_raises():
    c = complement({1, 2})
    with pytest.raises(NotImplementedError):
        c.pop()


def test_complement_type_error_non_set():
    c = complement({1, 2})
    with pytest.raises(TypeError):
        c | [1, 2]


def test_complement_frozenset():
    c = complement(frozenset({1, 2}))
    assert 1 not in c
    assert 3 in c


def test_complement_bool():
    c = complement({1, 2})
    assert bool(c) is True


def test_complement_complemented_from_positive():
    # Build a positive complement explicitly via complemented()
    c = complement({1, 2})
    inner = c.complemented()  # not a _ComplementSet now, wait—returns ComplementSet
    # complement of complement({1,2}) should be _ComplementSet(included={1,2})
    assert inner == {1, 2}
    assert 1 in inner
    assert 3 not in inner


def test_complement_hash():
    c1 = complement({1, 2})
    c2 = complement({1, 2})
    assert hash(c1) == hash(c2)


def test_complement_in_place_ops():
    c = complement({1, 2})
    c |= {3}
    # complement({1,2}) | {3} → complement({1,2} - {3}) = complement({1,2})
    assert 1 not in c
    assert 2 not in c
    assert 3 in c


def test_complement_symmetric_difference_update():
    c = complement({1, 2})
    c.symmetric_difference_update({1, 3})
    # - + : excluded |= inc → excluded = {1,2,3}
    assert 1 not in c
    assert 2 not in c
    assert 3 not in c
    assert 4 in c


def test_complement_update_non_set():
    c = complement({1, 2})
    c.update([1])
    # excluded &= {1} → {1}
    assert 1 not in c
    assert 2 in c
