import pytest

from boltons.listutils import BList, BarrelList, SplayList


# ---------------------------------------------------------------------------
# Basic BarrelList / BList API
# ---------------------------------------------------------------------------

def test_blist_alias():
    assert BList is BarrelList


def test_empty_init():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]


def test_init_with_iterable():
    bl = BarrelList([1, 2, 3])
    assert list(bl) == [1, 2, 3]
    assert len(bl) == 3


def test_init_with_empty_iterable():
    bl = BarrelList([])
    assert list(bl) == []
    assert bl.lists == [[]]


def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == "BarrelList([1, 2, 3])"


def test_len_and_iter():
    bl = BarrelList(range(10))
    assert len(bl) == 10
    assert list(bl) == list(range(10))


def test_reversed():
    bl = BarrelList(range(10))
    assert list(reversed(bl)) == list(reversed(range(10)))


def test_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 99 not in bl


def test_count():
    bl = BarrelList([1, 1, 2, 1, 3])
    assert bl.count(1) == 3
    assert bl.count(2) == 1
    assert bl.count(99) == 0


def test_index():
    bl = BarrelList([10, 20, 30, 20])
    assert bl.index(20) == 1
    assert bl.index(30) == 2
    with pytest.raises(ValueError):
        bl.index(99)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_getitem():
    bl = BarrelList([10, 20, 30])
    assert bl[0] == 10
    assert bl[1] == 20
    assert bl[2] == 30
    assert bl[-1] == 30
    assert bl[-2] == 20


def test_getitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]
    with pytest.raises(IndexError):
        bl[-4]


def test_getitem_slice():
    bl = BarrelList(range(10))
    result = bl[2:5]
    assert isinstance(result, BarrelList)
    assert list(result) == [2, 3, 4]


def test_getitem_slice_step():
    bl = BarrelList(range(10))
    result = bl[0:10:2]
    assert list(result) == [0, 2, 4, 6, 8]


def test_getitem_negative_slice():
    bl = BarrelList(range(10))
    assert list(bl[-3:]) == [7, 8, 9]


def test_setitem():
    bl = BarrelList([1, 2, 3])
    bl[1] = 20
    assert list(bl) == [1, 20, 3]
    bl[-1] = 30
    assert list(bl) == [1, 20, 30]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3] = 0


def test_setitem_slice_single_list():
    bl = BarrelList([1, 2, 3, 4])
    bl[1:3] = [20, 30]
    assert list(bl) == [1, 20, 30, 4]


def test_setitem_slice_multi_list():
    bl = BarrelList(range(3000))
    # force multiple lists
    assert len(bl.lists) > 1
    bl[0:3] = [100, 200, 300]
    assert bl[0:3] == BarrelList([100, 200, 300])


def test_delitem():
    bl = BarrelList([1, 2, 3])
    del bl[1]
    assert list(bl) == [1, 3]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        del bl[5]


def test_delitem_slice():
    bl = BarrelList(range(10))
    del bl[2:5]
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_delitem_slice_step():
    bl = BarrelList(range(10))
    del bl[0:10:2]
    assert list(bl) == [1, 3, 5, 7, 9]


# ---------------------------------------------------------------------------
# Insert / append / extend / pop
# ---------------------------------------------------------------------------

def test_insert():
    bl = BarrelList([1, 3])
    bl.insert(1, 2)
    assert list(bl) == [1, 2, 3]


def test_insert_at_front():
    bl = BarrelList([2, 3])
    bl.insert(0, 1)
    assert list(bl) == [1, 2, 3]


def test_insert_at_end():
    bl = BarrelList([1, 2])
    bl.insert(99, 3)
    assert list(bl) == [1, 2, 3]


def test_insert_negative_huge():
    bl = BarrelList([1, 2])
    bl.insert(-99, 0)
    assert list(bl) == [0, 1, 2]


def test_append():
    bl = BarrelList([1])
    bl.append(2)
    bl.append(3)
    assert list(bl) == [1, 2, 3]


def test_extend():
    bl = BarrelList([1])
    bl.extend([2, 3])
    assert list(bl) == [1, 2, 3]


def test_pop_no_args():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(0) == 1
    assert list(bl) == [2, 3]


def test_pop_negative():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-1) == 3
    assert list(bl) == [1, 2]


def test_pop_single_list_no_args():
    bl = BarrelList([1, 2])
    assert len(bl.lists) == 1
    assert bl.pop() == 2
    assert list(bl) == [1]


def test_pop_empty():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


def test_pop_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl.pop(5)


# ---------------------------------------------------------------------------
# Big list / balancing behavior
# ---------------------------------------------------------------------------

def test_large_growth_creates_multiple_lists():
    bl = BarrelList(range(100000))
    assert len(bl) == 100000
    assert list(bl) == list(range(100000))
    assert len(bl.lists) > 1


def test_large_pop_middle():
    bl = BarrelList(range(100000))
    assert bl.pop(50000) == 50000
    assert len(bl) == 99999


def test_slice_across_borders():
    bl = BarrelList(range(100000))
    assert len(bl.lists) > 1
    slice_idx = bl.lists[0][-1]
    result = bl[slice_idx:slice_idx + 2]
    assert isinstance(result, BarrelList)
    assert list(result) == [slice_idx, slice_idx + 1]


# ---------------------------------------------------------------------------
# sort / reverse
# ---------------------------------------------------------------------------

def test_sort_single_list():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_sort_multi_list():
    bl = BarrelList(range(3000, 0, -1))
    assert len(bl.lists) > 1
    bl.sort()
    assert list(bl) == list(range(1, 3001))


def test_reverse():
    bl = BarrelList([1, 2, 3])
    bl.reverse()
    assert list(bl) == [3, 2, 1]


def test_reverse_multi_list():
    bl = BarrelList(range(3000))
    assert len(bl.lists) > 1
    bl.reverse()
    assert list(bl) == list(range(2999, -1, -1))


# ---------------------------------------------------------------------------
# iter_slice / del_slice
# ---------------------------------------------------------------------------

def test_iter_slice_forward():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 6)) == [2, 3, 4, 5]


def test_iter_slice_step():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(0, 10, 2)) == [0, 2, 4, 6, 8]


def test_iter_slice_reverse():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(8, 2, -1)) == [8, 7, 6, 5, 4, 3]


def test_del_slice_simple():
    bl = BarrelList(range(10))
    bl.del_slice(2, 5)
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_del_slice_with_step():
    bl = BarrelList(range(10))
    bl.del_slice(0, 10, 2)
    assert list(bl) == [1, 3, 5, 7, 9]


def test_del_slice_start_none():
    bl = BarrelList(range(10))
    bl.del_slice(None, 3)
    assert list(bl) == [3, 4, 5, 6, 7, 8, 9]


def test_del_slice_stop_none():
    bl = BarrelList(range(10))
    bl.del_slice(7, None)
    assert list(bl) == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# from_iterable
# ---------------------------------------------------------------------------

def test_from_iterable():
    result = BarrelList.from_iterable(range(5))
    assert isinstance(result, BarrelList)
    assert list(result) == [0, 1, 2, 3, 4]


def test_from_iterable_subclass():
    class MyList(BarrelList):
        pass

    result = MyList.from_iterable([1, 2, 3])
    assert isinstance(result, MyList)
    assert list(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splaylist_is_list():
    sl = SplayList([1, 2, 3])
    assert isinstance(sl, list)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift_to_front():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert list(sl) == [3, 1, 2, 4]


def test_splaylist_shift_same_index():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift_to_dest():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(0, 2)
    assert list(sl) == [2, 3, 1, 4]


def test_splaylist_shift_negative():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(-1)
    assert list(sl) == [4, 1, 2, 3]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 3)
    assert list(sl) == [4, 2, 3, 1]


def test_splaylist_swap_same():
    sl = SplayList([1, 2, 3])
    sl.swap(1, 1)
    assert list(sl) == [1, 2, 3]
