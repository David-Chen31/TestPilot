import pytest

from boltons.listutils import BarrelList, BList, SplayList


# -------------------------
# Basic BarrelList behavior
# -------------------------

def test_blist_alias():
    assert BList is BarrelList


def test_empty_barrel_list():
    bl = BarrelList()
    assert len(bl) == 0
    assert list(bl) == []
    assert bl.lists == [[]]
    assert repr(bl) == "BarrelList([])"


def test_init_with_iterable():
    bl = BarrelList([1, 2, 3])
    assert list(bl) == [1, 2, 3]
    assert len(bl) == 3


def test_init_with_empty_iterable():
    bl = BarrelList([])
    assert len(bl) == 0
    assert list(bl) == []


def test_init_with_generator():
    bl = BarrelList(x for x in range(5))
    assert list(bl) == [0, 1, 2, 3, 4]


def test_repr_roundtrip():
    bl = BarrelList([1, 2, 3])
    r = repr(bl)
    assert r == "BarrelList([1, 2, 3])"


def test_from_iterable():
    bl = BarrelList.from_iterable(range(4))
    assert isinstance(bl, BarrelList)
    assert list(bl) == [0, 1, 2, 3]


# -------------------------
# Indexing and getting
# -------------------------

def test_getitem_positive():
    bl = BarrelList(range(10))
    assert bl[0] == 0
    assert bl[5] == 5
    assert bl[9] == 9


def test_getitem_negative():
    bl = BarrelList(range(10))
    assert bl[-1] == 9
    assert bl[-10] == 0


def test_getitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[3]
    with pytest.raises(IndexError):
        bl[-4]


def test_getitem_slice_returns_barrel_list():
    bl = BarrelList(range(10))
    sliced = bl[2:5]
    assert isinstance(sliced, BarrelList)
    assert list(sliced) == [2, 3, 4]


def test_getitem_slice_with_step():
    bl = BarrelList(range(10))
    assert list(bl[::2]) == [0, 2, 4, 6, 8]
    assert list(bl[1::3]) == [1, 4, 7]


def test_getitem_negative_step():
    bl = BarrelList(range(5))
    assert list(bl[::-1]) == [4, 3, 2, 1, 0]


# -------------------------
# Insert/append/extend
# -------------------------

def test_insert_at_end():
    bl = BarrelList([1, 2, 3])
    bl.insert(10, 4)
    assert list(bl) == [1, 2, 3, 4]


def test_insert_at_front():
    bl = BarrelList([1, 2, 3])
    bl.insert(-10, 0)
    assert list(bl) == [0, 1, 2, 3]


def test_insert_middle():
    bl = BarrelList([1, 2, 3])
    bl.insert(1, 99)
    assert list(bl) == [1, 99, 2, 3]


def test_append_and_extend():
    bl = BarrelList([1])
    bl.append(2)
    bl.extend([3, 4])
    assert list(bl) == [1, 2, 3, 4]


# -------------------------
# Setitem and delitem
# -------------------------

def test_setitem_index():
    bl = BarrelList([1, 2, 3])
    bl[1] = 20
    assert list(bl) == [1, 20, 3]


def test_setitem_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl[5] = 100


def test_setitem_slice():
    bl = BarrelList([1, 2, 3, 4])
    bl[1:3] = [20, 30]
    assert list(bl) == [1, 20, 30, 4]


def test_delitem_index():
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


def test_delitem_slice_with_step():
    bl = BarrelList(range(10))
    del bl[::2]
    assert list(bl) == [1, 3, 5, 7, 9]


def test_del_slice_negative_step():
    bl = BarrelList(range(10))
    bl.del_slice(9, 0, -3)
    result = list(bl)
    # values at indices 9,6,3 were removed
    assert 9 not in result
    assert 6 not in result
    assert 3 not in result


# -------------------------
# Pop
# -------------------------

def test_pop_no_arg():
    bl = BarrelList([1, 2, 3])
    assert bl.pop() == 3
    assert list(bl) == [1, 2]


def test_pop_with_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(0) == 1
    assert list(bl) == [2, 3]


def test_pop_negative_index():
    bl = BarrelList([1, 2, 3])
    assert bl.pop(-1) == 3
    assert list(bl) == [1, 2]


def test_pop_out_of_range():
    bl = BarrelList([1, 2, 3])
    with pytest.raises(IndexError):
        bl.pop(10)


# -------------------------
# Iteration, contains, count, index
# -------------------------

def test_iter_and_reversed():
    bl = BarrelList([1, 2, 3])
    assert list(iter(bl)) == [1, 2, 3]
    assert list(reversed(bl)) == [3, 2, 1]


def test_contains():
    bl = BarrelList([1, 2, 3])
    assert 2 in bl
    assert 5 not in bl


def test_count():
    bl = BarrelList([1, 1, 2, 1])
    assert bl.count(1) == 3
    assert bl.count(2) == 1
    assert bl.count(9) == 0


def test_index():
    bl = BarrelList([10, 20, 30])
    assert bl.index(20) == 1
    assert bl.index(10) == 0


def test_index_not_found():
    bl = BarrelList([10, 20, 30])
    with pytest.raises(ValueError):
        bl.index(99)


# -------------------------
# Sort / reverse
# -------------------------

def test_sort():
    bl = BarrelList([3, 1, 2])
    bl.sort()
    assert list(bl) == [1, 2, 3]


def test_sort_large_forces_balance():
    bl = BarrelList(range(5000, 0, -1))
    bl.sort()
    result = list(bl)
    assert result == sorted(range(1, 5001))


def test_reverse():
    bl = BarrelList([1, 2, 3])
    bl.reverse()
    assert list(bl) == [3, 2, 1]


# -------------------------
# Multiple sublists / balancing
# -------------------------

def test_extend_large_balances():
    bl = BarrelList()
    bl.extend(range(5000))
    # After balancing there should be multiple sublists
    assert len(bl.lists) > 1
    assert list(bl) == list(range(5000))


def test_insert_large_balances_and_ordering():
    bl = BarrelList(range(3000))
    for i in range(100):
        bl.insert(1500, i)
    assert len(bl) == 3100
    # first 1500 are 0..1499
    assert list(bl[:1500]) == list(range(1500))
    # last part is 1500..2999
    assert list(bl[-1500:]) == list(range(1500, 3000))


def test_iter_slice_positive():
    bl = BarrelList(range(10))
    assert list(bl.iter_slice(2, 6)) == [2, 3, 4, 5]


def test_iter_slice_negative_step():
    bl = BarrelList(range(5))
    assert list(bl.iter_slice(None, None, -1)) == [4, 3, 2, 1, 0]


# -------------------------
# SplayList
# -------------------------

def test_splaylist_shift():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(2)
    assert list(sl) == [3, 1, 2, 4]


def test_splaylist_shift_to_dest():
    sl = SplayList([1, 2, 3, 4])
    sl.shift(3, 1)
    assert list(sl) == [1, 4, 2, 3]


def test_splaylist_shift_same_index_is_noop():
    sl = SplayList([1, 2, 3])
    sl.shift(1, 1)
    assert list(sl) == [1, 2, 3]


def test_splaylist_swap():
    sl = SplayList([1, 2, 3, 4])
    sl.swap(0, 3)
    assert list(sl) == [4, 2, 3, 1]


def test_splaylist_is_list():
    sl = SplayList()
    sl.append(1)
    sl.append(2)
    assert list(sl) == [1, 2]
    assert sl[0] == 1
