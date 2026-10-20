import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------------------------------------------------------------------
# PriorityQueue alias / module-level sanity
# ---------------------------------------------------------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue, BasePriorityQueue])
def test_constructor_empty(cls):
    pq = cls()
    assert len(pq) == 0
    assert pq._entry_map == {}
    assert list(pq._pq) == []


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_constructor_unexpected_kwargs(cls):
    with pytest.raises(TypeError) as excinfo:
        cls(bogus=1)
    assert "unexpected keyword arguments" in str(excinfo.value)


def test_constructor_custom_priority_key():
    # Using a priority_key that treats lower numbers as higher priority
    pq = SortedPriorityQueue(priority_key=lambda p: float(p or 0))
    pq.add('a', 3)
    pq.add('b', 1)
    pq.add('c', 2)
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


# ---------------------------------------------------------------------------
# Default priority_key behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_default_priority_key_negation(cls):
    # higher numeric priority -> popped first
    pq = cls()
    pq.add('low', 0)
    pq.add('high', 5)
    pq.add('mid', 2)
    assert pq.pop() == 'high'
    assert pq.pop() == 'mid'
    assert pq.pop() == 'low'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_none_priority_equivalent_to_zero(cls):
    pq = cls()
    pq.add('none', None)
    pq.add('zero', 0)
    # both get priority 0; FIFO via counter
    assert pq.pop() == 'none'
    assert pq.pop() == 'zero'


# ---------------------------------------------------------------------------
# add / len
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_add_and_len(cls):
    pq = cls()
    assert len(pq) == 0
    pq.add('a')
    assert len(pq) == 1
    pq.add('b')
    pq.add('c')
    assert len(pq) == 3
    assert set(pq._entry_map.keys()) == {'a', 'b', 'c'}


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_add_existing_task_updates_priority(cls):
    pq = cls()
    pq.add('task', 1)
    pq.add('other', 2)
    assert len(pq) == 2
    # Re-add with higher priority
    pq.add('task', 10)
    assert len(pq) == 2
    assert pq.pop() == 'task'
    assert pq.pop() == 'other'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_add_existing_task_lower_priority(cls):
    pq = cls()
    pq.add('task', 10)
    pq.add('other', 5)
    pq.add('task', 1)
    assert len(pq) == 2
    assert pq.pop() == 'other'
    assert pq.pop() == 'task'


def test_fifo_tiebreak_by_counter():
    pq = SortedPriorityQueue()
    pq.add('first', 1)
    pq.add('second', 1)
    pq.add('third', 1)
    assert pq.pop() == 'first'
    assert pq.pop() == 'second'
    assert pq.pop() == 'third'


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.peek() == 'b'
    assert pq.peek() == 'b'
    assert len(pq) == 2


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.peek()
    assert 'peek on empty queue' in str(excinfo.value)


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_empty_default(cls):
    pq = cls()
    assert pq.peek(default='fallback') == 'fallback'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_nonempty_ignores_default(cls):
    pq = cls()
    pq.add('x', 1)
    assert pq.peek(default='fallback') == 'x'


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_order_and_len(cls):
    pq = cls()
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.peek() == 'medium priority task 1'
    assert len(pq) == 3
    assert pq.pop() == 'medium priority task 1'
    assert pq.pop() == 'medium priority task 2'
    assert pq.pop() == 'low priority task'
    assert len(pq) == 0


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.pop()
    assert 'pop on empty queue' in str(excinfo.value)


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_empty_default(cls):
    pq = cls()
    assert pq.pop(default=None) is None
    assert pq.pop(default='x') == 'x'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_removes_from_entry_map(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.pop() == 'b'
    assert 'b' not in pq._entry_map
    assert 'a' in pq._entry_map
    assert len(pq) == 1


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_present(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.add('c', 3)
    pq.remove('a')
    assert len(pq) == 2
    assert 'a' not in pq._entry_map
    assert pq.pop() == 'c'
    assert pq.pop() == 'b'
    assert len(pq) == 0


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_absent_raises(cls):
    pq = cls()
    pq.add('a', 1)
    with pytest.raises(KeyError):
        pq.remove('missing')


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_then_peek_skips_removed(cls):
    pq = cls()
    pq.add('a', 10)
    pq.add('b', 5)
    pq.remove('a')
    # 'a' was highest; peek should cull it and return 'b'
    assert pq.peek() == 'b'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_all_then_pop_raises(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    pq.remove('b')
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_all_then_peek_empty_default(cls):
    pq = cls()
    pq.add('a', 1)
    pq.remove('a')
    assert pq.peek(default='empty') == 'empty'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_re_add_after_remove(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    pq.add('a', 10)
    assert len(pq) == 2
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


# ---------------------------------------------------------------------------
# _cull behavior directly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_cull_removes_tombstones(cls):
    pq = cls()
    pq.add('a', 10)
    pq.add('b', 5)
    pq.remove('a')
    pq._cull()
    assert len(pq._pq) == 1
    assert pq._pq[0][-1] == 'b'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_cull_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq._cull()


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_cull_empty_no_raise(cls):
    pq = cls()
    pq._cull(raise_exc=False)


# ---------------------------------------------------------------------------
# BasePriorityQueue as abstract backend (no push/pop implementations)
# ---------------------------------------------------------------------------

def test_base_priority_queue_len_and_add():
    # Base class doesn't implement push/pop but add still records state
    pq = BasePriorityQueue()
    pq.add('a', 1)
    assert len(pq) == 1
    assert 'a' in pq._entry_map
    # Abstract _push_entry is a no-op, so backend stays empty
    assert len(pq._pq) == 0


def test_base_priority_queue_peek_empty_raises():
    pq = BasePriorityQueue()
    with pytest.raises(IndexError):
        pq.peek()


# ---------------------------------------------------------------------------
# BasePriorityQueue abstract entries return None
# ---------------------------------------------------------------------------

def test_base_push_entry_returns_none():
    backend = []
    entry = [-1.0, 0, 'x']
    result = BasePriorityQueue._push_entry(backend, entry)
    assert result is None
    # Abstract implementation is a no-op: backend is unmodified.
    assert backend == []


def test_base_pop_entry_returns_none():
    backend = [(-1.0, 0, 'x')]
    result = BasePriorityQueue._pop_entry(backend)
    assert result is None
    # Abstract implementation does not consume the backend entry.
    assert backend == [(-1.0, 0, 'x')]


# ---------------------------------------------------------------------------
# HeapPriorityQueue / SortedPriorityQueue backend entry shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_entry_shape(cls):
    pq = cls()
    pq.add('task', 3)
    assert len(pq._pq) == 1
    entry = pq._pq[0]
    assert entry[0] == -3.0  # negated float priority
    assert entry[-1] == 'task'


def test_sorted_backend_is_sorted_by_priority():
    pq = SortedPriorityQueue()
    pq.add('a', 1)
    pq.add('b', 3)
    pq.add('c', 2)
    entries = list(pq._pq)
    assert [e[0] for e in entries] == [-3.0, -2.0, -1.0]


# ---------------------------------------------------------------------------
# Staticmethod overrides operate on given backend
# ---------------------------------------------------------------------------

def test_heap_push_pop_entry():
    backend = []
    entry = [-1.0, 0, 'x']
    HeapPriorityQueue._push_entry(backend, entry)
    assert backend == [entry]
    assert HeapPriorityQueue._pop_entry(backend) == entry
    assert backend == []


def test_sorted_push_pop_entry():
    backend = []
    SortedPriorityQueue._push_entry(backend, [-2.0, 0, 'y'])
    SortedPriorityQueue._push_entry(backend, [-1.0, 1, 'x'])
    assert len(backend) == 2
    assert SortedPriorityQueue._pop_entry(backend)[-1] == 'y'
