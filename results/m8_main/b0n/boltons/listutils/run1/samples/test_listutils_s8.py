import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def test_blist_alias():
    assert BList is BarrelList


def test_empty_init():
    bl = BarrelList()
    assert bl.lists == [[]]
    assert len(bl) == 0
    assert list(bl) == []


def test_init_none():
    bl = BarrelList(None)
    assert bl.lists == [[]]
    assert len(bl) == 0


def test_init_iterable():
    bl = BarrelList(range(5))
    assert list(bl) == [0, 1, 2, 3, 4]
    assert len(bl) == 5


def test_from_iterable():
    bl = BarrelList.from_iterable(iter([1, 2, 3]))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Iteration / len / contains / repr
# ---------------------------------------------------------------------------

def test_iter_and_reversed():
    bl = BarrelList([1, 2, 3, 4, 5])
    assert list(iter(bl)) == [1, 2, 3, 4, 5]
    assert list(reversed(bl)) == [5, 4, 3, 2, 1]


def test_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 10 not in bl
    assert "x" not in bl


def test_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == "BarrelList([1, 2, 3])"
    assert repr(BarrelList()) == "BarrelList([])"


def test_repr_subclass():
    class MyBL(BarrelList):
        pass
    assert repr(MyBL([1])) == "MyBL([1])"


# ---------------------------------------------------------------------------
# getitem
# ---------------------------------------------------------------------------

def test_getitem_basic():
    bl = BarrelList([10, 20, 30, 40])
    assert bl[0] == 10
    assert bl[3] == 40
    assert bl[-1] == 40
    assert bl[-4] == 10


def test_getitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]
    with pytest.raises(IndexError):
        bl[-4]


def test_getitem_slice():
    bl = BarrelList([1, 2, 3, 4, 5, 6])
    res = bl[1:4]
    assert isinstance(res, BarrelList)
    assert list(res) == [2, 3, 4]
    assert list(bl[::2]) == [1, 3, 5]
    assert list(bl[::-1]) == [6, 5, 4, 3, 2, 1]


def test_getitem_slice_negative_step():
    bl = BarrelList(range(10))
    assert list(bl[8:2:-2]) == [8, 6, 4]


# ---------------------------------------------------------------------------
# insert / append / extend
# ---------------------------------------------------------------------------

def test_append_and_extend():
    bl = BarrelList()
    bl.append(1)
    bl.append(2)
    bl.extend([3, 4])
    assert list(bl) == [1, 2, 3, 4]


def test_insert_basic():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 99)
    assert list(bl) == [1, 99, 2, 3]
    bl.insert(-1, 88)
    assert list(bl) == [1, 99, 2, 88, 3]


def test_insert_clamps():
    bl = BarrelList([1, 2, 3])
    bl.insert(100, "end")
    assert list(bl) == [1, 2, 3, "end"]
    bl.insert(-100, "start")
    assert list(bl) == ["start", 1, 2, 3, "end"]


def test_insert_triggers_balance():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

def test_pop_empty():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()


def test_pop_no_args():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_index():
    bl = BarrelList([1, 2, 3, 4])
    assert bl.pop(1) == 2
    assert list(bl) == [1, 3, 4]
    assert bl.pop(-1) == 4
    assert list(bl) == [1, 3]


def test_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(5)


def test_pop_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    val = bl.pop(0)
    assert val == 0
    assert bl[0] == 1
    last = bl.pop()
    assert last == 4999
    assert bl[-1] == 4998


# ---------------------------------------------------------------------------
# delitem / del_slice
# ---------------------------------------------------------------------------

def test_delitem_index():
    bl = BarrelList([1, 2, 3, 4])
    del bl[1]
    assert list(bl) == [1, 3, 4]
    del bl[-1]
    assert list(bl) == [1, 3]


def test_delitem_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        del bl[5]


def test_del_slice_basic():
    bl = BarrelList([1, 2, 3, 4, 5])
    del bl[1:3]
    assert list(bl) == [1, 4, 5]


def test_del_slice_to_end():
    bl = BarrelList([1, 2, 3, 4, 5])
    del bl[2:]
    assert list(bl) == [1, 2]


def test_del_slice_from_start():
    bl = BarrelList([1, 2, 3, 4, 5])
    del bl[:2]
    assert list(bl) == [3, 4, 5]


def test_del_slice_with_step():
    bl = BarrelList(list(range(10)))
    del bl[0:10:2]
    assert list(bl) == [1, 3, 5, 7, 9]


def test_del_slice_empty_result():
    bl = BarrelList([1, 2, 3])
    del bl[:]
    assert list(bl) == []
    assert len(bl) == 0


# ---------------------------------------------------------------------------
# setitem
# ---------------------------------------------------------------------------

def test_setitem_index():
    bl = BarrelList([1, 2, 3])
    bl[1] = 20
    assert list(bl) == [1, 20, 3]
    bl[-1] = 30
    assert list(bl) == [1, 20, 30]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2])
    with pytest.raises(IndexError):
        bl[5] = 99


def test_setitem_slice():
    bl = BarrelList([1, 2, 3, 4, 5])
    bl[1:3] = [20, 30, 40]
    assert list(bl) == [1, 20, 30, 40, 4, 5]


def test_setitem_slice_multi_list():
    bl = BarrelList(range(5000))
    assert len(bl.lists) > 1
    bl[0:2] = [-1, -2]
    assert bl[0] == -1
    assert bl[1] == -2
    assert list(bl)[2:] == list(range(2, 5000))


def test_setslice_legacy():
    bl = BarrelList([1, 2, 3])
    bl.__setslice__(1, 2, [9, 9])
    assert list(bl) == [1, 9, 9, 3]


def test_getslice_legacy():
    bl = BarrelList([1, 2, 3, 4])
    res = bl.__getslice__(1, 3)
    assert isinstance(res, BarrelList)
    assert list(res) == [2, 3]


# ---------------------------------------------------------------------------
# sort / reverse / count / index
# ---------------------------------------------------------------------------

def test_sort_single_list():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_sort_multi_list():
    bl = BarrelList([5, 3, 1, 4, 2] * 1000)
    assert len(bl.lists) > 1
    bl.sort()
    assert list(bl) == sorted([5, 3, 1, 4, 2] * 1000)


def test_reverse():
    bl = BarrelList([1, 2, 3, 4])
    bl.reverse()
    assert list(bl) == [4, 3, 2, 1]


def test_reverse_multi_list():
    data = list(range(5000))
    bl = BarrelList(data)
    assert len(bl.lists) > 1
    bl.reverse()
    assert list(bl) == list(reversed(data))


def test_count():
    bl = BarrelList([1, 2, 2, 3, 2])
    assert bl.count(2) == 3
    assert bl.count(99) == 0


def test_index_found():
    bl = BarrelList([5, 6, 7, 8])
    assert bl.index(6) == 1
    assert bl.index(8) == 3


def test_index_not_found():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(ValueError):
        bl.index(99)


def test_index_multi_list():
    data = list(range(5000))
    bl = BarrelList(data)
    assert len(bl.lists) > 1
    assert bl.index(4999) == 4999


# ---------------------------------------------------------------------------
# iter_slice
# ---------------------------------------------------------------------------

def test_iter_slice_basic():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 5)) == [2, 3, 4]
    assert list(bl.iter_slice(0, None, 2)) == [0, 2, 4, 6, 8]


def test_iter_slice_negative_step():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(8, 2, -1)) == [8, 7, 6, 5, 4, 3]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_translate_index():
    bl = BarrelList([1, 2, 3])
    assert bl._translate_index(0) == (0, 0)
    assert bl._translate_index(-1) == (0, 2)
    assert bl._translate_index(5) == (None, None)


def test_cur_size_limit_positive():
    bl = BarrelList()
    assert bl._cur_size_limit > 0


# ---------------------------------------------------------------------------
# SplayList
# ---------------------------------------------------------------------------

def test_splaylist_shift_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_shift():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert list(sl) == [3, 1, 2, 4]
    sl.shift(3, 1)
    assert list(sl) == [3, 4, 1, 2]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 3)
    assert list(sl) == [4, 2, 3, 1]
    sl.swap(1, 2)
    assert list(sl) == [4, 3, 2, 1]
