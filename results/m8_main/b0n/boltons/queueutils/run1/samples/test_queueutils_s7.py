import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------- Basic API across both concrete queues ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_basic_add_pop_ordering(cls):
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


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add('a')
    pq.add('b')
    pq.add('c', 1)
    assert pq.pop() == 'c'
    # remaining equal-priority items come out in insertion order
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_none_priority_treated_as_zero(cls):
    pq = cls()
    pq.add('none', None)
    pq.add('zero', 0)
    pq.add('one', 1)
    assert pq.pop() == 'one'
    assert pq.pop() == 'none'
    assert pq.pop() == 'zero'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_len_and_contains_via_len(cls):
    pq = cls()
    assert len(pq) == 0
    pq.add('x')
    assert len(pq) == 1
    pq.add('y')
    assert len(pq) == 2


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add('a', 1)
    assert pq.peek() == 'a'
    assert pq.peek() == 'a'
    assert len(pq) == 1


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_empty_raises_indexerror(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_empty_with_default(cls):
    pq = cls()
    sentinel = object()
    assert pq.peek(default=sentinel) is sentinel
    assert pq.peek('fallback') == 'fallback'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_empty_raises_indexerror(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_empty_with_default(cls):
    pq = cls()
    sentinel = object()
    assert pq.pop(default=sentinel) is sentinel
    assert pq.pop('fallback') == 'fallback'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    assert len(pq) == 1
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_absent_raises_keyerror(cls):
    pq = cls()
    with pytest.raises(KeyError):
        pq.remove('nope')


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_top_then_peek_culls(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    # removed entry should be culled from the backend
    assert pq.peek() == 'b'
    assert pq.pop() == 'b'
    assert len(pq) == 0


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_remove_all_then_peek_raises(cls):
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


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_add_existing_updates_priority(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.peek() == 'b'
    # re-add 'a' with higher priority
    pq.add('a', 5)
    assert len(pq) == 2
    assert pq.peek() == 'a'
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_add_existing_lower_priority(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    assert pq.peek() == 'a'
    pq.add('a', -10)
    assert pq.peek() == 'b'
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


# ---------------- Custom priority_key ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_custom_priority_key(cls):
    # lower number = higher priority
    pq = cls(priority_key=lambda p: p)
    pq.add('low', 10)
    pq.add('high', 1)
    pq.add('mid', 5)
    assert pq.pop() == 'high'
    assert pq.pop() == 'mid'
    assert pq.pop() == 'low'


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_custom_priority_key_none(cls):
    pq = cls(priority_key=lambda p: p if p is not None else 100)
    pq.add('none', None)
    pq.add('zero', 0)
    assert pq.pop() == 'zero'
    assert pq.pop() == 'none'


# ---------------- Constructor errors ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_unexpected_kwarg_raises_typeerror(cls):
    with pytest.raises(TypeError):
        cls(bogus_kwarg=1)


# ---------------- Ordering with many equal priorities ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_equal_priorities_fifo(cls):
    pq = cls()
    tasks = ['t%d' % i for i in range(20)]
    for t in tasks:
        pq.add(t, 1)
    assert [pq.pop() for _ in tasks] == tasks


# ---------------- Mixed priorities ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_mixed_priorities_sorted_desc(cls):
    pq = cls()
    data = [('a', 3), ('b', 1), ('c', 2), ('d', 5), ('e', 4)]
    for task, pr in data:
        pq.add(task, pr)
    result = [pq.pop() for _ in data]
    assert result == ['d', 'e', 'a', 'c', 'b']


# ---------------- BasePriorityQueue abstract behavior ----------------

def test_base_priority_queue_backend_is_list():
    pq = BasePriorityQueue()
    assert pq._pq == []
    assert pq._entry_map == {}
    assert len(pq) == 0


def test_base_priority_queue_unsupported_push():
    pq = BasePriorityQueue()
    # Base doesn't implement push, so entry is not stored in backend
    pq.add('x', 1)
    assert len(pq) == 1
    # backend remains empty because _push_entry is a no-op
    assert pq._pq == []


def test_base_priority_queue_peek_unsupported():
    pq = BasePriorityQueue()
    pq.add('x', 1)
    # backend empty so peek raises
    with pytest.raises(IndexError):
        pq.peek()


def test_base_priority_queue_pop_unsupported():
    pq = BasePriorityQueue()
    pq.add('x', 1)
    with pytest.raises(IndexError):
        pq.pop()


def test_base_priority_queue_custom_kwarg_priority_key():
    calls = []

    def pk(p):
        calls.append(p)
        return p

    pq = BasePriorityQueue(priority_key=pk)
    pq.add('x', 7)
    assert calls == [7]


def test_base_priority_queue_remove_and_len():
    pq = BasePriorityQueue()
    pq.add('x', 1)
    pq.add('y', 2)
    assert len(pq) == 2
    pq.remove('x')
    assert len(pq) == 1
    with pytest.raises(KeyError):
        pq.remove('x')


# ---------------- PriorityQueue alias ----------------

def test_priority_queue_alias_is_sorted():
    assert PriorityQueue is SortedPriorityQueue


# ---------------- Subclass/backend type checks ----------------

def test_heap_backend_is_list():
    pq = HeapPriorityQueue()
    assert isinstance(pq._pq, list)


def test_sorted_uses_sorted_insertion():
    pq = SortedPriorityQueue()
    pq.add('a', 1)
    pq.add('b', 3)
    pq.add('c', 2)
    # internal backend should be sorted ascending by (priority, count, task)
    # (priority negated), so highest priority first
    assert [entry[2] for entry in pq._pq] == ['b', 'c', 'a']


# ---------------- repr / misc ----------------

@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_pop_returns_task_object(cls):
    pq = cls()
    obj = object()
    pq.add(obj, 1)
    assert pq.pop() is obj


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_returns_task_object(cls):
    pq = cls()
    obj = object()
    pq.add(obj, 1)
    assert pq.peek() is obj


@pytest.mark.parametrize("cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_peek_skips_removed_entries(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 4)
    pq.add('c', 3)
    pq.remove('a')
    pq.remove('b')
    assert pq.peek() == 'c'
    assert len(pq) == 1
