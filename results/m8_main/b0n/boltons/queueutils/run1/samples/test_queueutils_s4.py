import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


# -------------------------
# Module-level aliases / API
# -------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


# -------------------------
# BasePriorityQueue basics
# -------------------------

def test_default_priority_key():
    key = BasePriorityQueue._default_priority_key
    assert key(None) == -0.0
    assert key(0) == -0.0
    assert key(5) == -5.0
    assert key(-3) == 3.0
    assert key('2') == -2.0


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_empty_queue(cls):
    pq = cls()
    assert len(pq) == 0


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_add_and_len(cls):
    pq = cls()
    pq.add('a')
    pq.add('b', 5)
    pq.add('c', -2)
    assert len(pq) == 3


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_pop_order_higher_priority_first(cls):
    pq = cls()
    pq.add('low', 0)
    pq.add('high', 2)
    pq.add('med1', 1)
    pq.add('med2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high'
    assert pq.peek() == 'med1'
    assert len(pq) == 3
    assert pq.pop() == 'med1'
    assert pq.pop() == 'med2'
    assert pq.pop() == 'low'
    assert len(pq) == 0


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add('first')
    pq.add('second')
    # Same priority, insertion order preserved via counter
    assert pq.pop() == 'first'
    assert pq.pop() == 'second'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_negative_priority_lower_than_zero(cls):
    pq = cls()
    pq.add('zero', 0)
    pq.add('neg', -1)
    pq.add('pos', 1)
    assert pq.pop() == 'pos'
    assert pq.pop() == 'zero'
    assert pq.pop() == 'neg'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add('a', 1)
    assert pq.peek() == 'a'
    assert pq.peek() == 'a'
    assert len(pq) == 1


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_peek_empty_raises_index_error(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_peek_empty_returns_default(cls):
    pq = cls()
    assert pq.peek(default='sentinel') == 'sentinel'
    assert pq.peek(None) is None


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_pop_empty_raises_index_error(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_pop_empty_returns_default(cls):
    pq = cls()
    assert pq.pop(default='sentinel') == 'sentinel'
    assert pq.pop(None) is None


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.add('c', 3)
    pq.remove('b')
    assert len(pq) == 2
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_remove_absent_raises_key_error(cls):
    pq = cls()
    pq.add('a', 1)
    with pytest.raises(KeyError):
        pq.remove('missing')


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_remove_top_then_pop(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('b')
    assert pq.pop() == 'a'
    assert len(pq) == 0


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_remove_all_then_peek_empty(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    pq.remove('b')
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.peek()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_add_existing_task_changes_priority(cls):
    pq = cls()
    pq.add('a', 0)
    pq.add('b', 5)
    pq.add('c', 1)
    # Bump 'a' to highest priority
    pq.add('a', 10)
    assert len(pq) == 3
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_add_existing_task_lowers_priority(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.add('a', 0)
    assert len(pq) == 2
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_key_custom(cls):
    # priority_key returning -p means higher p = higher effective priority
    pq = cls(priority_key=lambda p: -float(p or 0))
    pq.add('x', 1)
    pq.add('y', 3)
    pq.add('z', 2)
    assert pq.pop() == 'y'
    assert pq.pop() == 'z'
    assert pq.pop() == 'x'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_key_inverted(cls):
    # Inverted priority: smaller numeric values win
    pq = cls(priority_key=lambda p: p if p is not None else 0)
    pq.add('a', 3)
    pq.add('b', 1)
    pq.add('c', 2)
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_unexpected_kwargs_raise_type_error(cls):
    with pytest.raises(TypeError):
        cls(bogus_keyword=123)


def test_base_priority_queue_directly():
    # Base is not truly abstract; instantiate and exercise it.
    pq = BasePriorityQueue()
    assert len(pq) == 0
    pq.add('a', 1)
    assert len(pq) == 1


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_backend_is_initialized_empty(cls):
    pq = cls()
    assert list(pq._pq) == []
    assert pq._entry_map == {}


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_entry_map_tracks_entries(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert set(pq._entry_map.keys()) == {'a', 'b'}
    pq.pop()
    # 'b' should have been popped (priority 2)
    assert set(pq._entry_map.keys()) == {'a'}


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_counter_increments(cls):
    pq = cls()
    pq.add('a')
    pq.add('b')
    pq.add('c')
    counts = sorted(entry[1] for entry in pq._entry_map.values())
    assert counts == [0, 1, 2]


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_pop_with_multiple_equal_priorities_fifo(cls):
    pq = cls()
    for i in range(5):
        pq.add(i, 1)
    results = [pq.pop() for _ in range(5)]
    assert results == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_mixed_operations(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 3)
    pq.add('c', 2)
    assert pq.peek() == 'b'
    pq.remove('b')
    assert pq.peek() == 'c'
    pq.add('d', 5)
    assert pq.pop() == 'd'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'
    assert len(pq) == 0


def test_heap_priority_queue_uses_heap():
    pq = HeapPriorityQueue()
    assert isinstance(pq._pq, list)
    pq.add('a', 1)
    pq.add('b', 2)
    # heap root is the highest-priority entry
    assert pq._pq[0][2] == 'b'


def test_sorted_priority_queue_is_sorted():
    pq = SortedPriorityQueue()
    pq.add('a', 3)
    pq.add('b', 1)
    pq.add('c', 2)
    # entries stored in ascending order of negated priority -> b, c, a
    tasks = [entry[2] for entry in pq._pq]
    assert tasks == ['b', 'c', 'a']


# -------------------------
# _cull behavior
# -------------------------

@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_cull_removes_marked_entries(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 5)
    pq.remove('b')
    # entry marked removed but still in backend
    assert len(pq._pq) == 2
    pq._cull()
    assert len(pq._pq) == 1
    assert pq._pq[0][2] == 'a'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_cull_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq._cull()


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_cull_empty_no_raise(cls):
    pq = cls()
    # Should not raise
    pq._cull(raise_exc=False)


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_cull_all_removed_no_raise(cls):
    pq = cls()
    pq.add('a', 1)
    pq.remove('a')
    pq._cull(raise_exc=False)
    assert len(pq._pq) == 0
