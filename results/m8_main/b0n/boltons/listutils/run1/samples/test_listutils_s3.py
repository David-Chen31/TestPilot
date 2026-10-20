import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_empty_construction():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]


def test_construction_from_iterable():
    bl = BarrelList([1, 2, 3])
    assert len(bl) == 3
    assert list(bl) == [1, 2, 3]


def test_construction_none_iterable():
    bl = BarrelList(None)
    assert len(bl) == 0


def test_blist_alias():
    assert BList is BarrelList


def test_from_iterable_classmethod():
    bl = BarrelList.from_iterable(iter(range(5)))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Append / insert / extend
# ---------------------------------------------------------------------------

def test_append():
    bl = BarrelList()
    bl.append(1)
    bl.append(2)
    assert list(bl) == [1, 2]
    assert len(bl) == 2


def test_extend():
    bl = BarrelList([1])
    bl.extend([2, 3, 4])
    assert list(bl) == [1, 2, 3, 4]


def test_insert_basic():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 99)
    assert list(bl) == [1, 99, 2, 3]


def test_insert_at_beginning():
    bl = BarrelList([1, 2, 3])
    bl.insert(0, 0)
    assert list(bl) == [0, 1, 2, 3]


def test_insert_into_empty():
    bl = BarrelList()
    bl.insert(0, 5)
    assert list(bl) == [5]


def test_insert_clamp_append():
    bl = BarrelList([1, 2, 3])
    bl.insert(100, 4)
    assert list(bl) == [1, 2, 3, 4]


def test_insert_clamp_prepend():
    bl = BarrelList([1, 2, 3])
    bl.insert(-100, 0)
    assert list(bl) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_getitem_positive():
    bl = BarrelList([10, 20, 30])
    assert bl[0] == 10
    assert bl[1] == 20
    assert bl[2] == 30


def test_getitem_negative():
    bl = BarrelList([10, 20, 30])
    assert bl[-1] == 30
    assert bl[-3] == 10


def test_getitem_out_of_range_positive():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]


def test_getitem_out_of_range_negative():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[-4]


def test_getitem_empty():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl[0]


def test_getitem_slice():
    bl = BarrelList([0, 1, 2, 3, 4])
    result = bl[1:4]
    assert isinstance(result, BarrelList)
    assert list(result) == [1, 2, 3]


def test_getitem_slice_with_step():
    bl = BarrelList([0, 1, 2, 3, 4, 5])
    result = bl[::2]
    assert isinstance(result, BarrelList)
    assert list(result) == [0, 2, 4]


def test_getitem_slice_negative_step():
    bl = BarrelList([0, 1, 2, 3, 4])
    result = bl[::-1]
    assert list(result) == [4, 3, 2, 1, 0]


# ---------------------------------------------------------------------------
# Setitem
# ---------------------------------------------------------------------------

def test_setitem_index():
    bl = BarrelList([1, 2, 3])
    bl[1] = 99
    assert list(bl) == [1, 99, 3]


def test_setitem_negative_index():
    bl = BarrelList([1, 2, 3])
    bl[-1] = 99
    assert list(bl) == [1, 2, 99]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[10] = 5


def test_setitem_slice():
    bl = BarrelList([1, 2, 3, 4])
    bl[1:3] = [20, 30]
    assert list(bl) == [1, 20, 30, 4]


# ---------------------------------------------------------------------------
# Delitem
# ---------------------------------------------------------------------------

def test_delitem_index():
    bl = BarrelList([1, 2, 3])
    del bl[1]
    assert list(bl) == [1, 3]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        del bl[10]


def test_delitem_slice():
    bl = BarrelList([0, 1, 2, 3, 4])
    del bl[1:3]
    assert list(bl) == [0, 3, 4]


def test_delitem_slice_open():
    bl = BarrelList([0, 1, 2, 3, 4])
    del bl[2:]
    assert list(bl) == [0, 1]


# ---------------------------------------------------------------------------
# Pop
# ---------------------------------------------------------------------------

def test_pop_no_args():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_last_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-1) == 3
    assert list(bl) == [1, 2]


def test_pop_specific_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(1) == 2
    assert list(bl) == [1, 3]


def test_pop_negative_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-2) == 2
    assert list(bl) == [1, 3]


def test_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(10)


def test_pop_empty():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


# ---------------------------------------------------------------------------
# Iteration / reversed / contains / len
# ---------------------------------------------------------------------------

def test_iter():
    bl = BarrelList([1, 2, 3])
    assert list(iter(bl)) == [1, 2, 3]


def test_reversed():
    bl = BarrelList([1, 2, 3])
    assert list(reversed(bl)) == [3, 2, 1]


def test_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 99 not in bl


def test_len():
    bl = BarrelList(range(10))
    assert len(bl) == 10


# ---------------------------------------------------------------------------
# Sorting / reversing
# ---------------------------------------------------------------------------

def test_sort_single_list():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_reverse():
    bl = BarrelList([1, 2, 3])
    bl.reverse()
    assert list(bl) == [3, 2, 1]


# ---------------------------------------------------------------------------
# Count / index
# ---------------------------------------------------------------------------

def test_count():
    bl = BarrelList([1, 1, 2, 1])
    assert bl.count(1) == 3
    assert bl.count(2) == 1
    assert bl.count(99) == 0


def test_index_found():
    bl = BarrelList([5, 6, 7, 6])
    assert bl.index(6) == 1
    assert bl.index(7) == 2


def test_index_not_found():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(ValueError):
        bl.index(99)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == 'BarrelList([1, 2, 3])'


# ---------------------------------------------------------------------------
# Larger-scale balancing test
# ---------------------------------------------------------------------------

def test_large_extend_and_access():
    n = 10000
    bl = BarrelList(range(n))
    assert len(bl) == n
    assert list(bl) == list(range(n))
    assert len(bl.lists) > 1
    # Spot-check random-ish (deterministic) accesses
    for idx in (0, 1, 50, 500, 5000, n - 1):
        assert bl[idx] == idx


def test_large_insert_balance():
    bl = BarrelList()
    # Insertions using insert() trigger balancing
    for i in range(5000):
        bl.insert(0, i)
    assert len(bl) == 5000
    # Inserting 0-front means reversed
    assert bl[0] == 4999
    assert bl[-1] == 0


def test_large_pop():
    n = 5000
    bl = BarrelList(range(n))
    # Popping from the end repeatedly
    for expected in range(n - 1, -1, -1):
        assert bl.pop() == expected
    assert len(bl) == 0


def test_large_sort():
    n = 3000
    data = [(i * 37) % n for i in range(n)]
    bl = BarrelList(data)
    assert len(bl.lists) > 1
    bl.sort()
    assert list(bl) == sorted(data)


def test_large_reverse():
    n = 3000
    bl = BarrelList(range(n))
    assert len(bl.lists) > 1
    bl.reverse()
    assert list(bl) == list(range(n - 1, -1, -1))


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splay_shift_same_index():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splay_shift_default_dest():
    sl = SplayList([1, 2, 3])
    sl.shift(2)
    assert list(sl) == [3, 1, 2]


def test_splay_shift_to_index():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(1, 3)
    assert list(sl) == [1, 3, 4, 2]


def test_splay_swap():
    sl = SplayList([1, 2, 3])
    sl.swap(0, 2)
    assert list(sl) == [3, 2, 1]


def test_splay_swap_same():
    sl = SplayList([1, 2, 3])
    sl.swap(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splay_is_list_subclass():
    sl = SplayList([1, 2])
    assert isinstance(sl, list)
    assert sl == [1, 2]
