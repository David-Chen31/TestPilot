import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------------------------------------------------------------------------
# BarrelList / BList basics
# ---------------------------------------------------------------------------

def test_blist_is_barrellist_alias():
    assert BList is BarrelList


def test_init_empty():
    bl = BarrelList()
    assert isinstance(bl, list)
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]


def test_init_none():
    bl = BarrelList(None)
    assert len(bl) == 0
    assert bl.lists == [[]]


def test_init_with_iterable():
    bl = BarrelList([1, 2, 3])
    assert list(bl) == [1, 2, 3]
    assert len(bl) == 3


def test_init_with_empty_iterable():
    bl = BarrelList([])
    assert len(bl) == 0
    assert bl.lists == [[]]


def test_from_iterable():
    bl = BarrelList.from_iterable(iter(range(5)))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [0, 1, 2, 3, 4]


def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == 'BarrelList([1, 2, 3])'


def test_repr_subclass_name():
    class MyBL(BarrelList):
        pass

    b = MyBL([1])
    assert repr(b).startswith('MyBL(')


# ---------------------------------------------------------------------------
# append / extend / insert
# ---------------------------------------------------------------------------

def test_append_extend():
    bl = BarrelList([1])
    bl.append(2)
    bl.extend([3, 4])
    assert list(bl) == [1, 2, 3, 4]


def test_insert_into_single_list():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 99)
    assert list(bl) == [1, 99, 2, 3]


def test_insert_clamp_append():
    bl = BarrelList([1, 2, 3])
    # force multiple lists
    bl.lists = [[1, 2], [3]]
    bl.insert(100, 9)
    assert list(bl) == [1, 2, 3, 9]


def test_insert_clamp_prepend():
    bl = BarrelList([1, 2, 3])
    bl.lists = [[1, 2], [3]]
    bl.insert(-100, 9)
    assert list(bl) == [9, 1, 2, 3]


def test_insert_multi_list_middle():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3], [4, 5]]
    bl.insert(3, 99)
    assert list(bl) == [0, 1, 2, 99, 3, 4, 5]


def test_insert_balances():
    bl = BarrelList()
    bl._size_factor = 4
    # make current list large enough to trigger balancing
    for i in range(20):
        bl.insert(0, i)
    assert len(bl) == 20
    assert list(bl) == list(reversed(range(20)))


# ---------------------------------------------------------------------------
# __getitem__ / indexing
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


def test_getitem_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl[5]
    with pytest.raises(IndexError):
        bl[-5]


def test_getitem_multi_list():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3], [4, 5]]
    assert bl[2] == 2
    assert bl[4] == 4
    assert bl[-1] == 5


def test_getitem_slice():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3], [4, 5]]
    result = bl[1:4]
    assert isinstance(result, BarrelList)
    assert list(result) == [1, 2, 3]


def test_getitem_slice_step():
    bl = BarrelList(list(range(10)))
    result = bl[::2]
    assert list(result) == [0, 2, 4, 6, 8]


def test_getitem_slice_negative_step():
    bl = BarrelList(list(range(5)))
    result = bl[::-1]
    assert list(result) == [4, 3, 2, 1, 0]


def test_getslice_dunder():
    bl = BarrelList([1, 2, 3, 4])
    result = bl.__getslice__(1, 3)
    assert list(result) == [2, 3]


# ---------------------------------------------------------------------------
# __setitem__
# ---------------------------------------------------------------------------

def test_setitem_single():
    bl = BarrelList([1, 2, 3])
    bl[1] = 99
    assert list(bl) == [1, 99, 3]


def test_setitem_negative():
    bl = BarrelList([1, 2, 3])
    bl[-1] = 99
    assert list(bl) == [1, 2, 99]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl[5] = 0


def test_setitem_slice_single_list():
    bl = BarrelList([1, 2, 3, 4])
    bl[1:3] = [9, 9]
    assert list(bl) == [1, 9, 9, 4]


def test_setitem_slice_multi_list():
    bl = BarrelList()
    bl.lists = [[1, 2], [3, 4]]
    bl[1:3] = [9]
    assert list(bl) == [1, 9, 4]


def test_setslice_dunder():
    bl = BarrelList([1, 2, 3, 4])
    bl.__setslice__(1, 3, [9, 9, 9])
    assert list(bl) == [1, 9, 9, 9, 4]


# ---------------------------------------------------------------------------
# __delitem__ / del_slice
# ---------------------------------------------------------------------------

def test_delitem_single():
    bl = BarrelList([1, 2, 3])
    del bl[1]
    assert list(bl) == [1, 3]


def test_delitem_negative():
    bl = BarrelList([1, 2, 3])
    del bl[-1]
    assert list(bl) == [1, 2]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        del bl[5]


def test_delitem_slice():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3], [4, 5]]
    del bl[1:4]
    assert list(bl) == [0, 4, 5]


def test_del_slice_across():
    bl = BarrelList(list(range(10)))
    bl.del_slice(2, 7)
    assert list(bl) == [0, 1, 7, 8, 9]


def test_del_slice_with_step():
    bl = BarrelList(list(range(10)))
    bl.del_slice(1, 8, 2)
    assert list(bl) == [0, 2, 4, 6, 8, 9]


def test_del_slice_none_start():
    bl = BarrelList(list(range(5)))
    bl.del_slice(None, 2)
    assert list(bl) == [2, 3, 4]


def test_del_slice_none_stop():
    bl = BarrelList(list(range(5)))
    bl.del_slice(2, None)
    assert list(bl) == [0, 1]


def test_del_slice_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl.del_slice(5, 10)


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

def test_pop_no_args():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_empty_no_args():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


def test_pop_explicit_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(0) == 1
    assert list(bl) == [2, 3]


def test_pop_negative_index():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3]]
    assert bl.pop(-1) == 3
    assert list(bl) == [0, 1, 2]


def test_pop_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl.pop(10)


def test_pop_cleans_empty_tail():
    bl = BarrelList()
    bl.lists = [[1], []]
    assert bl.pop() == 1
    assert bl.lists == [[1]]


def test_pop_explicit_removes_empty_sublist():
    bl = BarrelList()
    bl.lists = [[0, 1], [2]]
    assert bl.pop(2) == 2
    assert bl.lists == [[0, 1]]


# ---------------------------------------------------------------------------
# iteration / reversed / len / contains
# ---------------------------------------------------------------------------

def test_iter():
    bl = BarrelList()
    bl.lists = [[0, 1], [2], [3, 4]]
    assert list(iter(bl)) == [0, 1, 2, 3, 4]


def test_reversed():
    bl = BarrelList()
    bl.lists = [[0, 1], [2], [3, 4]]
    assert list(reversed(bl)) == [4, 3, 2, 1, 0]


def test_len():
    bl = BarrelList()
    bl.lists = [[0, 1], [2], [3, 4, 5]]
    assert len(bl) == 6


def test_contains():
    bl = BarrelList()
    bl.lists = [[0, 1], [2], [3, 4]]
    assert 2 in bl
    assert 4 in bl
    assert 9 not in bl


# ---------------------------------------------------------------------------
# iter_slice
# ---------------------------------------------------------------------------

def test_iter_slice_positive():
    bl = BarrelList(list(range(5)))
    assert list(bl.iter_slice(1, 4)) == [1, 2, 3]


def test_iter_slice_step():
    bl = BarrelList(list(range(10)))
    assert list(bl.iter_slice(0, 10, 2)) == [0, 2, 4, 6, 8]


def test_iter_slice_negative_step():
    bl = BarrelList(list(range(5)))
    assert list(bl.iter_slice(4, 0, -1)) == [4, 3, 2, 1]


# ---------------------------------------------------------------------------
# sort / reverse / count / index
# ---------------------------------------------------------------------------

def test_sort_single_list():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_sort_multi_list():
    bl = BarrelList()
    bl.lists = [[3], [1, 4], [2]]
    bl.sort()
    assert list(bl) == [1, 2, 3, 4]


def test_reverse():
    bl = BarrelList()
    bl.lists = [[1, 2], [3], [4, 5]]
    bl.reverse()
    assert list(bl) == [5, 4, 3, 2, 1]


def test_count():
    bl = BarrelList()
    bl.lists = [[1, 2, 1], [1], [3]]
    assert bl.count(1) == 3
    assert bl.count(3) == 1
    assert bl.count(9) == 0


def test_index_found():
    bl = BarrelList()
    bl.lists = [[0, 1], [2], [3, 4]]
    assert bl.index(2) == 2
    assert bl.index(4) == 4


def test_index_not_found():
    bl = BarrelList()
    bl.lists = [[0, 1], [2]]
    with pytest.raises(ValueError):
        bl.index(99)


# ---------------------------------------------------------------------------
# _translate_index behavior
# ---------------------------------------------------------------------------

def test_translate_index_negative_out_of_range():
    bl = BarrelList([1, 2, 3])
    list_idx, rel_idx = bl._translate_index(-100)
    assert list_idx is None
    assert rel_idx is None


def test_translate_index_good():
    bl = BarrelList()
    bl.lists = [[0, 1], [2, 3]]
    assert bl._translate_index(0) == (0, 0)
    assert bl._translate_index(2) == (1, 0)
    assert bl._translate_index(-1) == (1, 1)


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splaylist_shift():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert list(sl) == [3, 1, 2, 4]


def test_splaylist_shift_same_index_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift_to_dest():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(3, 1)
    assert list(sl) == [1, 4, 2, 3]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 3)
    assert list(sl) == [4, 2, 3, 1]


def test_splaylist_swap_same_indices():
    sl = SplayList([1, 2, 3])
    sl.swap(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_is_list_subclass():
    sl = SplayList([1, 2, 3])
    assert isinstance(sl, list)
    assert sl[0] == 1


# ---------------------------------------------------------------------------
# Integration: many ops keep consistency
# ---------------------------------------------------------------------------

def test_large_insert_pop_consistency():
    bl = BarrelList()
    for i in range(100):
        bl.append(i)
    assert len(bl) == 100
    assert bl[50] == 50
    assert bl.pop(50) == 50
    assert len(bl) == 99
    assert list(bl)[:3] == [0, 1, 2]


def test_slice_returns_independent_copy():
    bl = BarrelList([1, 2, 3, 4, 5])
    s = bl[1:3]
    s.append(99)
    assert list(bl) == [1, 2, 3, 4, 5]
