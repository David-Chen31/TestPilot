import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------------------------------------------------------------------------
# Construction and basic list protocol
# ---------------------------------------------------------------------------

def test_blist_alias():
    assert BList is BarrelList


def test_empty_construction():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]
    assert isinstance(bl, list)


def test_construction_none():
    bl = BarrelList(None)
    assert len(bl) == 0
    assert list(bl) == []


def test_construction_iterable():
    bl = BarrelList([1, 2, 3])
    assert len(bl) == 3
    assert list(bl) == [1, 2, 3]


def test_construction_from_generator():
    bl = BarrelList(x for x in range(5))
    assert list(bl) == [0, 1, 2, 3, 4]


def test_from_iterable_classmethod():
    bl = BarrelList.from_iterable(range(4))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [0, 1, 2, 3]


def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == "BarrelList([1, 2, 3])"


def test_repr_empty():
    bl = BarrelList()
    assert repr(bl) == "BarrelList([])"


def test_iter_and_reversed():
    bl = BarrelList(range(10))
    assert list(iter(bl)) == list(range(10))
    assert list(reversed(bl)) == list(range(9, -1, -1))


def test_len_and_contains():
    bl = BarrelList([1, 2, 3])
    assert len(bl) == 3
    assert 2 in bl
    assert 5 not in bl


# ---------------------------------------------------------------------------
# Indexing and slicing
# ---------------------------------------------------------------------------

def test_getitem_simple():
    bl = BarrelList([10, 20, 30])
    assert bl[0] == 10
    assert bl[1] == 20
    assert bl[2] == 30
    assert bl[-1] == 30
    assert bl[-3] == 10


def test_getitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]
    with pytest.raises(IndexError):
        bl[-4]


def test_getitem_slice():
    bl = BarrelList(range(10))
    sliced = bl[2:5]
    assert isinstance(sliced, BarrelList)
    assert list(sliced) == [2, 3, 4]


def test_getitem_slice_with_step():
    bl = BarrelList(range(10))
    assert list(bl[0:10:2]) == [0, 2, 4, 6, 8]


def test_getitem_slice_negative_step():
    bl = BarrelList(range(10))
    assert list(bl[::-1]) == list(range(9, -1, -1))


def test_getitem_slice_across_borders():
    # Force multiple sublists
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    sliced = bl[100:200]
    assert list(sliced) == list(range(100, 200))


def test_getslice():
    bl = BarrelList(range(5))
    assert list(bl.__getslice__(1, 3)) == [1, 2]


def test_setitem_simple():
    bl = BarrelList([1, 2, 3])
    bl[1] = 99
    assert list(bl) == [1, 99, 3]


def test_setitem_negative_index():
    bl = BarrelList([1, 2, 3])
    bl[-1] = 42
    assert list(bl) == [1, 2, 42]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3] = 10
    with pytest.raises(IndexError):
        bl[-4] = 10


def test_setitem_slice_single_list():
    bl = BarrelList([1, 2, 3, 4])
    bl[1:3] = [10, 20]
    assert list(bl) == [1, 10, 20, 4]


def test_setitem_slice_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    bl[0:2] = [100, 200, 300]
    assert bl[0] == 100
    assert bl[1] == 200
    assert bl[2] == 300
    assert len(bl) == 5001


def test_setslice():
    bl = BarrelList([1, 2, 3, 4])
    bl.__setslice__(1, 3, [10, 20])
    assert list(bl) == [1, 10, 20, 4]


def test_delitem_by_index():
    bl = BarrelList([1, 2, 3])
    del bl[1]
    assert list(bl) == [1, 3]
    del bl[-1]
    assert list(bl) == [1]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        del bl[5]


def test_delitem_slice():
    bl = BarrelList(range(10))
    del bl[2:5]
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_delitem_slice_with_step():
    bl = BarrelList(range(10))
    del bl[0:10:2]
    assert list(bl) == [1, 3, 5, 7, 9]


def test_delitem_open_ended_slice():
    bl = BarrelList(range(5))
    del bl[2:]
    assert list(bl) == [0, 1]


# ---------------------------------------------------------------------------
# insert / append / extend / pop
# ---------------------------------------------------------------------------

def test_insert_basic():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 10)
    assert list(bl) == [1, 10, 2, 3]


def test_insert_clamped_append():
    bl = BarrelList([1, 2, 3])
    bl.insert(100, 10)
    assert list(bl) == [1, 2, 3, 10]


def test_insert_clamped_prepend():
    bl = BarrelList([1, 2, 3])
    bl.insert(-100, 10)
    assert list(bl) == [10, 1, 2, 3]


def test_insert_negative():
    bl = BarrelList([1, 2, 3])
    bl.insert(-1, 10)
    assert list(bl) == [1, 2, 10, 3]


def test_insert_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    bl.insert(2500, -1)
    assert bl[2500] == -1
    assert len(bl) == 5001


def test_append_and_extend():
    bl = BarrelList()
    bl.append(1)
    bl.append(2)
    bl.extend([3, 4])
    assert list(bl) == [1, 2, 3, 4]


def test_pop_default():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_empty_raises():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


def test_pop_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(0) == 1
    assert list(bl) == [2, 3]


def test_pop_negative_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-2) == 2
    assert list(bl) == [1, 3]


def test_pop_explicit_neg_one():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-1) == 3
    assert list(bl) == [1, 2]


def test_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(10)


def test_pop_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    assert bl.pop(2500) == 2500
    assert len(bl) == 4999
    assert bl[2500] == 2501


# ---------------------------------------------------------------------------
# iter_slice / del_slice
# ---------------------------------------------------------------------------

def test_iter_slice_forward():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 5)) == [2, 3, 4]


def test_iter_slice_with_step():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(0, 10, 2)) == [0, 2, 4, 6, 8]


def test_iter_slice_reverse():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(0, 10, -1)) == list(range(9, -1, -1))


def test_del_slice_basic():
    bl = BarrelList(range(10))
    bl.del_slice(2, 5)
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_del_slice_with_step():
    bl = BarrelList(range(10))
    bl.del_slice(0, 10, 2)
    assert list(bl) == [1, 3, 5, 7, 9]


def test_del_slice_start_none():
    bl = BarrelList(range(5))
    bl.del_slice(None, 2)
    assert list(bl) == [2, 3, 4]


def test_del_slice_stop_none():
    bl = BarrelList(range(5))
    bl.del_slice(2, None)
    assert list(bl) == [0, 1]


def test_delslice_alias():
    bl = BarrelList(range(5))
    bl.__delslice__(1, 3)
    assert list(bl) == [0, 3, 4]


def test_del_slice_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    bl.del_slice(100, 4000)
    assert len(bl) == 1100
    assert bl[0] == 0
    assert bl[99] == 99
    assert bl[100] == 4000


# ---------------------------------------------------------------------------
# sort / reverse / count / index
# ---------------------------------------------------------------------------

def test_sort_single_list():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_sort_multi_list():
    bl = BarrelList(range(5000, 0, -1))
    assert len(bl.lists) > 1
    bl.sort()
    assert list(bl) == list(range(1, 5001))


def test_reverse():
    bl = BarrelList([1, 2, 3, 4])
    bl.reverse()
    assert list(bl) == [4, 3, 2, 1]


def test_reverse_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    bl.reverse()
    assert list(bl) == list(range(4999, -1, -1))


def test_count():
    bl = BarrelList([1, 2, 2, 3, 2])
    assert bl.count(2) == 3
    assert bl.count(5) == 0


def test_index():
    bl = BarrelList([1, 2, 3, 2])
    assert bl.index(2) == 1
    assert bl.index(3) == 2


def test_index_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    assert bl.index(2500) == 2500


def test_index_missing():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(ValueError):
        bl.index(5)


# ---------------------------------------------------------------------------
# _translate_index and _balance_list (internal, but exercised indirectly)
# ---------------------------------------------------------------------------

def test_translate_index():
    bl = BarrelList([1, 2, 3])
    assert bl._translate_index(0) == (0, 0)
    assert bl._translate_index(-1) == (0, 2)
    assert bl._translate_index(5) == (None, None)


def test_balance_list_single():
    bl = BarrelList([1, 2, 3])
    assert bl._balance_list(0) is False


def test_balance_list_negative_index():
    bl = BarrelList(iter(range(5000)))
    assert len(bl.lists) > 1
    # balancing last list with negative index
    res = bl._balance_list(-1)
    assert res is False


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splaylist_is_list():
    sl = SplayList([1, 2, 3])
    assert isinstance(sl, list)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert list(sl) == [3, 1, 2, 4]


def test_splaylist_shift_to_dest():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(1, 3)
    assert list(sl) == [1, 3, 4, 2]


def test_splaylist_shift_same_index_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift_negative():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(-1)
    assert list(sl) == [4, 1, 2, 3]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 2)
    assert list(sl) == [3, 2, 1, 4]


def test_splaylist_swap_negative():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(-1, 0)
    assert list(sl) == [4, 2, 3, 1]
