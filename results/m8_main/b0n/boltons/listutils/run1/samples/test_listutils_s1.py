import pytest

from boltons.listutils import BarrelList, BList, SplayList


# ---------- BarrelList basics ----------

def test_barrel_list_empty():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]


def test_barrel_list_from_iterable():
    bl = BarrelList(range(5))
    assert len(bl) == 5
    assert list(bl) == [0, 1, 2, 3, 4]


def test_barrel_list_none_iterable():
    bl = BarrelList(None)
    assert len(bl) == 0
    assert list(bl) == []


def test_blist_alias():
    assert BList is BarrelList
    bl = BList([1, 2, 3])
    assert isinstance(bl, BarrelList)
    assert list(bl) == [1, 2, 3]


def test_barrel_list_repr():
    bl = BarrelList([1, 2, 3])
    assert repr(bl) == "BarrelList([1, 2, 3])"


def test_barrel_list_from_iterable_classmethod():
    bl = BarrelList.from_iterable(iter([1, 2, 3]))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [1, 2, 3]


# ---------- getitem ----------

def test_barrel_list_getitem():
    bl = BarrelList(range(10))
    assert bl[0] == 0
    assert bl[5] == 5
    assert bl[-1] == 9
    assert bl[-10] == 0


def test_barrel_list_getitem_out_of_range():
    bl = BarrelList(range(5))
    with pytest.raises(IndexError):
        bl[5]
    with pytest.raises(IndexError):
        bl[-6]


def test_barrel_list_getitem_slice():
    bl = BarrelList(range(10))
    sliced = bl[2:5]
    assert isinstance(sliced, BarrelList)
    assert list(sliced) == [2, 3, 4]


def test_barrel_list_getitem_slice_step():
    bl = BarrelList(range(10))
    assert list(bl[::2]) == [0, 2, 4, 6, 8]
    assert list(bl[::-1]) == list(range(9, -1, -1))
    assert list(bl[8:2:-2]) == [8, 6, 4]


def test_barrel_list_getitem_negative_slice():
    bl = BarrelList(range(10))
    assert list(bl[-3:]) == [7, 8, 9]
    assert list(bl[:-7]) == [0, 1, 2]


# ---------- setitem ----------

def test_barrel_list_setitem_index():
    bl = BarrelList(range(10))
    bl[0] = 100
    bl[-1] = 200
    assert list(bl) == [100, 1, 2, 3, 4, 5, 6, 7, 8, 200]


def test_barrel_list_setitem_out_of_range():
    bl = BarrelList(range(5))
    with pytest.raises(IndexError):
        bl[10] = 1


def test_barrel_list_setitem_slice():
    bl = BarrelList(range(10))
    bl[2:5] = [20, 30]
    assert list(bl) == [0, 1, 20, 30, 5, 6, 7, 8, 9]


# ---------- delitem ----------

def test_barrel_list_delitem():
    bl = BarrelList(range(5))
    del bl[0]
    assert list(bl) == [1, 2, 3, 4]
    del bl[-1]
    assert list(bl) == [1, 2, 3]


def test_barrel_list_delitem_out_of_range():
    bl = BarrelList(range(5))
    with pytest.raises(IndexError):
        del bl[10]


def test_barrel_list_delitem_slice():
    bl = BarrelList(range(10))
    del bl[2:5]
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_barrel_list_delitem_slice_step():
    bl = BarrelList(range(10))
    del bl[::2]
    assert list(bl) == [1, 3, 5, 7, 9]


# ---------- append / extend / insert ----------

def test_barrel_list_append_extend():
    bl = BarrelList()
    bl.append(1)
    assert list(bl) == [1]
    bl.extend([2, 3, 4])
    assert list(bl) == [1, 2, 3, 4]


def test_barrel_list_insert_beginning():
    bl = BarrelList([1, 2, 3])
    bl.insert(0, 0)
    assert list(bl) == [0, 1, 2, 3]


def test_barrel_list_insert_middle():
    bl = BarrelList([1, 2, 3])
    bl.insert(2, 99)
    assert list(bl) == [1, 2, 99, 3]


def test_barrel_list_insert_negative():
    bl = BarrelList([1, 2, 3])
    bl.insert(-1, 99)
    assert list(bl) == [1, 2, 99, 3]


def test_barrel_list_insert_clamp_append():
    bl = BarrelList([1, 2, 3])
    bl.insert(100, 4)
    assert list(bl) == [1, 2, 3, 4]


def test_barrel_list_insert_clamp_prepend():
    bl = BarrelList([1, 2, 3])
    bl.insert(-100, 0)
    assert list(bl) == [0, 1, 2, 3]


# ---------- pop ----------

def test_barrel_list_pop():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]
    assert bl.pop(0) == 1
    assert list(bl) == [2]


def test_barrel_list_pop_negative():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-2) == 2
    assert list(bl) == [1, 3]


def test_barrel_list_pop_empty_raises():
    bl = BarrelList()
    with pytest.raises(IndexError):
        bl.pop()
    with pytest.raises(IndexError):
        bl.pop(0)


def test_barrel_list_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(10)


# ---------- iter / reversed / len / contains ----------

def test_barrel_list_iter():
    bl = BarrelList(range(5))
    assert list(iter(bl)) == [0, 1, 2, 3, 4]


def test_barrel_list_reversed():
    bl = BarrelList(range(5))
    assert list(reversed(bl)) == [4, 3, 2, 1, 0]


def test_barrel_list_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 99 not in bl


def test_barrel_list_len():
    bl = BarrelList(range(100))
    assert len(bl) == 100


# ---------- multi-list behaviors (trigger balancing) ----------

def test_barrel_list_balancing_grows():
    bl = BarrelList(range(10000))
    # For a large list, balancing happens so multiple sublists exist.
    assert len(bl.lists) > 1
    assert list(bl) == list(range(10000))
    assert len(bl) == 10000


def test_barrel_list_multi_sublist_getitem():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    for idx in (0, 1, 100, 5000, 9999):
        assert bl[idx] == idx
    for idx in range(10000):
        assert bl[idx] == idx


def test_barrel_list_multi_sublist_insert_pop():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    bl.insert(5000, -1)
    assert bl[5000] == -1
    assert len(bl) == 10001
    assert bl.pop(5000) == -1
    assert len(bl) == 10000
    assert list(bl) == list(range(10000))


def test_barrel_list_multi_sublist_delete():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    del bl[5000]
    assert len(bl) == 9999
    assert bl[4999] == 4999
    assert bl[5000] == 5001


def test_barrel_list_multi_sublist_setitem():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    bl[5000] = -5
    assert bl[5000] == -5


def test_barrel_list_slicing_across_borders():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    first_len = len(bl.lists[0])
    sliced = bl[first_len - 1:first_len + 1]
    assert list(sliced) == [first_len - 1, first_len]


# ---------- iter_slice / del_slice ----------

def test_barrel_list_iter_slice():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 5)) == [2, 3, 4]
    assert list(bl.iter_slice(0, 10, 2)) == [0, 2, 4, 6, 8]


def test_barrel_list_iter_slice_negative_step():
    bl = BarrelList(range(10))
    result = list(bl.iter_slice(8, 2, -2))
    assert result == [8, 6, 4]


def test_barrel_list_del_slice():
    bl = BarrelList(range(10))
    bl.del_slice(2, 5)
    assert list(bl) == [0, 1, 5, 6, 7, 8, 9]


def test_barrel_list_del_slice_with_step():
    bl = BarrelList(range(10))
    bl.del_slice(0, 10, 2)
    assert list(bl) == [1, 3, 5, 7, 9]


# ---------- sort / reverse / count / index ----------

def test_barrel_list_sort():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_barrel_list_sort_large():
    data = list(range(10000, 0, -1))
    bl = BarrelList(data)
    bl.sort()
    assert list(bl) == list(range(1, 10001))


def test_barrel_list_reverse():
    bl = BarrelList([1, 2, 3])
    bl.reverse()
    assert list(bl) == [3, 2, 1]


def test_barrel_list_reverse_large():
    bl = BarrelList(range(10000))
    bl.reverse()
    assert list(bl) == list(range(9999, -1, -1))


def test_barrel_list_count():
    bl = BarrelList([1, 2, 1, 3, 1])
    assert bl.count(1) == 3
    assert bl.count(99) == 0


def test_barrel_list_index():
    bl = BarrelList([1, 2, 3])
    assert bl.index(2) == 1


def test_barrel_list_index_not_found():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(ValueError):
        bl.index(99)


def test_barrel_list_index_large():
    bl = BarrelList(range(10000))
    assert len(bl.lists) > 1
    assert bl.index(9000) == 9000


# ---------- SplayList ----------

def test_splay_list_basic():
    sl = SplayList([1, 2, 3])
    assert isinstance(sl, list)
    assert sl == [1, 2, 3]


def test_splay_list_shift_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert sl == [1, 2, 3]


def test_splay_list_shift_to_front():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert sl == [3, 1, 2, 4]


def test_splay_list_shift_to_dest():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(0, 2)
    assert sl == [2, 3, 1, 4]


def test_splay_list_shift_negative():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(-1, 0)
    assert sl == [4, 1, 2, 3]


def test_splay_list_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 3)
    assert sl == [4, 2, 3, 1]


def test_splay_list_swap_same():
    sl = SplayList([1, 2, 3])
    sl.swap(1, 1)
    assert sl == [1, 2, 3]
