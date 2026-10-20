import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------------------------------------------------------------------------
# BarrelList basics
# ---------------------------------------------------------------------------

def test_blist_is_barrellist_alias():
    assert BList is BarrelList


def test_empty_init():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]


def test_init_with_iterable():
    bl = BarrelList([1, 2, 3])
    assert len(bl) == 3
    assert list(bl) == [1, 2, 3]


def test_init_with_empty_iterable():
    bl = BarrelList([])
    assert len(bl) == 0


def test_init_with_falsy_iterable_not_extended():
    # An empty generator is falsy? No -- but empty list is falsy.
    bl = BarrelList([])
    assert bl.lists == [[]]


def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == "BarrelList([1, 2, 3])"


def test_append_and_getitem():
    bl = BarrelList()
    bl.append(1)
    bl.append(2)
    assert bl[0] == 1
    assert bl[1] == 2
    assert bl[-1] == 2
    assert bl[-2] == 1


def test_getitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]
    with pytest.raises(IndexError):
        bl[-4]


def test_setitem():
    bl = BarrelList([1, 2, 3])
    bl[1] = 20
    assert list(bl) == [1, 20, 3]
    bl[-1] = 30
    assert list(bl) == [1, 20, 30]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[5] = 1


def test_delitem():
    bl = BarrelList([1, 2, 3])
    del bl[1]
    assert list(bl) == [1, 3]
    del bl[0]
    assert list(bl) == [3]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        del bl[10]


def test_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 5 not in bl


def test_len():
    bl = BarrelList(range(10))
    assert len(bl) == 10


def test_iter():
    bl = BarrelList(range(5))
    assert list(bl) == [0, 1, 2, 3, 4]


def test_reversed():
    bl = BarrelList(range(5))
    assert list(reversed(bl)) == [4, 3, 2, 1, 0]


def test_equality_with_list():
    bl = BarrelList([1, 2, 3])
    assert bl == [1, 2, 3]


def test_count():
    bl = BarrelList([1, 2, 2, 3, 2])
    assert bl.count(2) == 3
    assert bl.count(99) == 0


def test_index():
    bl = BarrelList([1, 2, 3, 2])
    assert bl.index(2) == 1
    assert bl.index(3) == 2
    with pytest.raises(ValueError):
        bl.index(99)


def test_insert():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 10)
    assert list(bl) == [1, 10, 2, 3]


def test_insert_clamp_high():
    bl = BarrelList([1, 2, 3])
    bl.insert(100, 99)
    assert list(bl) == [1, 2, 3, 99]


def test_insert_clamp_low():
    bl = BarrelList([1, 2, 3])
    bl.insert(-100, 99)
    assert list(bl) == [99, 1, 2, 3]


def test_insert_negative():
    bl = BarrelList([1, 2, 3])
    bl.insert(-1, 99)
    assert list(bl) == [1, 2, 99, 3]


def test_extend():
    bl = BarrelList([1])
    bl.extend([2, 3])
    assert list(bl) == [1, 2, 3]


def test_pop_default():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_no_arg_single_underlying_list():
    bl = BarrelList([1])
    assert bl.pop() == 1
    assert list(bl) == []


def test_pop_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(0) == 1
    assert list(bl) == [2, 3]


def test_pop_negative_one_explicit():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-1) == 3
    assert list(bl) == [1, 2]


def test_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(10)


def test_pop_empty_raises():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


def test_from_iterable():
    bl = BarrelList.from_iterable(iter([1, 2, 3]))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def test_getitem_slice_returns_barrellist():
    bl = BarrelList(range(10))
    sl = bl[2:5]
    assert isinstance(sl, BarrelList)
    assert list(sl) == [2, 3, 4]


def test_getitem_slice_with_step():
    bl = BarrelList(range(10))
    assert list(bl[::2]) == [0, 2, 4, 6, 8]


def test_getitem_reverse_slice():
    bl = BarrelList(range(10))
    assert list(bl[::-1]) == list(range(9, -1, -1))


def test_getitem_reverse_slice_partial():
    bl = BarrelList(range(10))
    assert list(bl[8:2:-2]) == [8, 6, 4]


def test_delitem_slice():
    bl = BarrelList(range(10))
    del bl[2:5]
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_delitem_slice_full():
    bl = BarrelList(range(5))
    del bl[:]
    assert list(bl) == []


def test_delitem_slice_with_step():
    bl = BarrelList(range(10))
    del bl[::2]
    assert list(bl) == [1, 3, 5, 7, 9]


def test_setitem_slice():
    bl = BarrelList([1, 2, 3])
    bl[1:2] = [20, 21]
    assert list(bl) == [1, 20, 21, 3]


def test_setitem_slice_single_list():
    bl = BarrelList([1, 2, 3])
    bl[0:1] = [10]
    assert list(bl) == [10, 2, 3]


# ---------------------------------------------------------------------------
# iter_slice / del_slice
# ---------------------------------------------------------------------------

def test_iter_slice():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 5)) == [2, 3, 4]


def test_iter_slice_step():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(0, 10, 2)) == [0, 2, 4, 6, 8]


def test_iter_slice_negative_step():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(None, None, -1)) == list(range(9, -1, -1))


def test_del_slice():
    bl = BarrelList(range(10))
    bl.del_slice(2, 5)
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_del_slice_none_start():
    bl = BarrelList(range(10))
    bl.del_slice(None, 3)
    assert list(bl) == [3, 4, 5, 6, 7, 8, 9]


def test_del_slice_none_stop():
    bl = BarrelList(range(10))
    bl.del_slice(7, None)
    assert list(bl) == [0, 1, 2, 3, 4, 5, 6]


def test_del_slice_with_step():
    bl = BarrelList(range(10))
    bl.del_slice(0, 10, 2)
    assert list(bl) == [1, 3, 5, 7, 9]


# ---------------------------------------------------------------------------
# Sorting and reversing
# ---------------------------------------------------------------------------

def test_sort():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_reverse():
    bl = BarrelList([1, 2, 3])
    bl.reverse()
    assert list(bl) == [3, 2, 1]


# ---------------------------------------------------------------------------
# Balancing / multiple sublists
# ---------------------------------------------------------------------------

def test_large_list_multiple_sublists():
    bl = BarrelList(range(100000))
    assert len(bl) == 100000
    assert len(bl.lists) > 1
    assert bl[50000] == 50000
    assert bl[99999] == 99999


def test_large_list_pop_middle():
    bl = BarrelList(range(100000))
    assert bl.pop(50000) == 50000
    assert len(bl) == 99999
    assert bl[50000] == 50001


def test_large_list_contains():
    bl = BarrelList(range(10000))
    assert 5000 in bl
    assert 20000 not in bl


def test_large_list_index():
    bl = BarrelList(range(10000))
    assert bl.index(5000) == 5000


def test_large_list_slice():
    bl = BarrelList(range(100000))
    sl = bl[10:20]
    assert list(sl) == list(range(10, 20))


def test_large_list_del_slice():
    bl = BarrelList(range(100000))
    del bl[10:20]
    assert len(bl) == 99990
    assert bl[10] == 20


def test_large_list_sort():
    data = list(range(1000, 0, -1))
    bl = BarrelList(data)
    bl.sort()
    assert list(bl) == sorted(data)


def test_large_list_reverse():
    bl = BarrelList(range(1000))
    bl.reverse()
    assert list(bl) == list(range(999, -1, -1))


def test_large_list_setitem():
    bl = BarrelList(range(100000))
    bl[50000] = -1
    assert bl[50000] == -1


def test_large_list_delitem():
    bl = BarrelList(range(100000))
    del bl[50000]
    assert len(bl) == 99999
    assert bl[50000] == 50001


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splaylist_shift():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(3)
    assert list(sl) == [4, 1, 2, 3]


def test_splaylist_shift_to_index():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(1, 3)
    assert list(sl) == [1, 3, 4, 2]


def test_splaylist_shift_same_index_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3])
    sl.swap(0, 2)
    assert list(sl) == [3, 2, 1]


def test_splaylist_is_list_subclass():
    sl = SplayList([1, 2, 3])
    assert isinstance(sl, list)
    assert sl == [1, 2, 3]


def test_splaylist_shift_negative():
    sl = SplayList([1, 2, 3])
    sl.shift(-1)
    assert list(sl) == [3, 1, 2]
