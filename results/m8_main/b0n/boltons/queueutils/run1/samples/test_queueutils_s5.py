import itertools

import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


ALL_PQ_CLASSES = [HeapPriorityQueue, SortedPriorityQueue, BasePriorityQueue]


@pytest.fixture(params=ALL_PQ_CLASSES)
def pq_cls(request):
    return request.param


def _make_subclass():
    class ConcretePQ(BasePriorityQueue):
        _backend_type = list

        @staticmethod
        def _push_entry(backend, entry):
            backend.append(entry)

        @staticmethod
        def _pop_entry(backend):
            return backend.pop(0)

    return ConcretePQ


# --------------------------- module-level ---------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


def test_exports():
    import boltons.queueutils as qu
    for name in ('PriorityQueue', 'BasePriorityQueue',
                 'HeapPriorityQueue', 'SortedPriorityQueue'):
        assert name in qu.__all__


# --------------------------- construction ---------------------------

def test_basic_construction(pq_cls):
    if pq_cls is BasePriorityQueue:
        pq = pq_cls()
        assert len(pq) == 0
        assert pq._counter is not None
        return
    pq = pq_cls()
    assert isinstance(pq._entry_map, dict)
    assert len(pq._entry_map) == 0
    assert list(itertools.islice(pq._counter, 0)) == []


def test_unexpected_kwargs_raise():
    with pytest.raises(TypeError):
        HeapPriorityQueue(unexpected=1)


def test_priority_key_constructor_kwarg():
    pq = HeapPriorityQueue(priority_key=lambda p: p)
    pq.add('a', 1)
    pq.add('b', 5)
    # priority_key returns p directly, larger number = higher priority? 
    # With default, priorities are negated. Here not negated, so heappop
    # returns smallest first -> 'a' (priority 1).
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


# --------------------------- add / pop ---------------------------

def test_add_single(pq_cls):
    pq = pq_cls if pq_cls is not BasePriorityQueue else _make_subclass()
    pq = pq()
    pq.add('task')
    assert len(pq) == 1
    assert 'task' in pq._entry_map
    assert pq.pop() == 'task'
    assert len(pq) == 0


def test_default_priority_zero(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a')
    pq.add('b')
    # Both priority 0; FIFO by counter
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


def test_higher_priority_first():
    for cls in (HeapPriorityQueue, SortedPriorityQueue):
        pq = cls()
        pq.add('low', 0)
        pq.add('high', 10)
        pq.add('medium', 5)
        assert pq.pop() == 'high'
        assert pq.pop() == 'medium'
        assert pq.pop() == 'low'


def test_negative_priority_lower():
    pq = HeapPriorityQueue()
    pq.add('a', -5)
    pq.add('b', -1)
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


def test_equal_priority_fifo():
    for cls in (HeapPriorityQueue, SortedPriorityQueue):
        pq = cls()
        pq.add('a', 1)
        pq.add('b', 1)
        pq.add('c', 1)
        assert pq.pop() == 'a'
        assert pq.pop() == 'b'
        assert pq.pop() == 'c'


def test_len_multiple():
    pq = HeapPriorityQueue()
    assert len(pq) == 0
    pq.add('a')
    pq.add('b')
    pq.add('c')
    assert len(pq) == 3
    pq.pop()
    assert len(pq) == 2
    pq.pop()
    pq.pop()
    assert len(pq) == 0


def test_len_counts_tasks_not_backend():
    pq = HeapPriorityQueue()
    pq.add('a')
    pq.add('b')
    pq.remove('a')
    # backend still has removed entry until culled
    assert len(pq) == 1


# --------------------------- peek ---------------------------

def test_peek_returns_top(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.peek() == 'b'
    # peek doesn't remove
    assert len(pq) == 2
    assert pq.peek() == 'b'


def test_peek_does_not_consume(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('x', 1)
    for _ in range(3):
        assert pq.peek() == 'x'
    assert pq.pop() == 'x'


def test_peek_empty_raises(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    with pytest.raises(IndexError):
        pq.peek()


def test_peek_empty_default(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    sentinel = object()
    assert pq.peek(sentinel) is sentinel


# --------------------------- pop edge cases ---------------------------

def test_pop_empty_raises(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    with pytest.raises(IndexError):
        pq.pop()


def test_pop_empty_default(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    sentinel = object()
    assert pq.pop(sentinel) is sentinel


def test_pop_all(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    for i in range(5):
        pq.add(i, i)
    results = [pq.pop() for _ in range(5)]
    assert results == [4, 3, 2, 1, 0]
    sentinel = object()
    assert pq.pop(sentinel) is sentinel


# --------------------------- remove ---------------------------

def test_remove_raises_on_missing(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    with pytest.raises(KeyError):
        pq.remove('nope')


def test_remove_existing(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('b')
    assert len(pq) == 1
    assert pq.pop() == 'a'


def test_remove_then_pop_culls(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    # 'a' is top but marked removed; cull should drop it
    assert pq.pop() == 'b'
    assert len(pq) == 0


def test_peek_culls_removed(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    assert pq.peek() == 'b'


def test_remove_all_then_pop_default(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    pq.remove('b')
    sentinel = object()
    assert pq.pop(sentinel) is sentinel


# --------------------------- add same task updates priority ---------------------------

def test_add_existing_updates_priority(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.peek() == 'b'
    pq.add('a', 10)  # re-add with higher priority
    assert len(pq) == 2
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


def test_add_existing_lowers_priority(pq_cls):
    if pq_cls is BasePriorityQueue:
        pytest.skip('abstract backend')
    pq = pq_cls()
    pq.add('a', 10)
    pq.add('b', 5)
    assert pq.peek() == 'a'
    pq.add('a', 1)
    assert len(pq) == 2
    assert pq.pop() == 'b'


# --------------------------- default priority key ---------------------------

def test_default_priority_key_none():
    key = BasePriorityQueue._default_priority_key
    assert key(None) == 0.0
    assert key(0) == 0.0
    assert key(5) == -5.0
    assert key(-3) == 3.0


def test_default_priority_key_string_number():
    key = BasePriorityQueue._default_priority_key
    assert key('5') == -5.0


# --------------------------- equality of tasks / unhashable behavior ---------------------------

def test_unhashable_task_raises():
    pq = HeapPriorityQueue()
    with pytest.raises(TypeError):
        pq.add([1, 2, 3])


# --------------------------- backend type ---------------------------

def test_heap_backend_type():
    pq = HeapPriorityQueue()
    assert isinstance(pq._pq, list)


def test_sorted_backend_type():
    pq = SortedPriorityQueue()
    from boltons.listutils import BList
    assert isinstance(pq._pq, BList)


def test_base_backend_type():
    pq = BasePriorityQueue()
    assert isinstance(pq._pq, list)


# --------------------------- integrated scenario ---------------------------

def test_docstring_scenario():
    pq = PriorityQueue()
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.peek() == 'medium priority task 1'
    assert len(pq) == 3


def test_many_operations():
    for cls in (HeapPriorityQueue, SortedPriorityQueue):
        pq = cls()
        for i in range(20):
            pq.add(i, i % 4)
        results = [pq.pop() for _ in range(20)]
        # tasks with priority 3 first, then 2, 1, 0
        priorities = [i % 4 for i in results]
        assert priorities == sorted(priorities, reverse=True)
        # within equal priority, insertion order (counter) preserved
        for p in (3, 2, 1, 0):
            grp = [t for t in results if t % 4 == p]
            assert grp == sorted(grp)


# --------------------------- abstract base hooks ---------------------------

def test_base_push_pop_hooks_are_noops():
    backend = []
    BasePriorityQueue._push_entry(backend, ['entry'])
    assert backend == []
    assert BasePriorityQueue._pop_entry(backend) is None


def test_base_add_remove_with_custom_backend():
    Concrete = _make_subclass()
    pq = Concrete()
    pq.add('a', 1)
    pq.add('b', 2)
    assert len(pq) == 2
    assert pq._pq  # backend has entries
    # entries are [priority, count, task]; push/append order preserved
    assert pq._pq[0][2] == 'a'
    assert pq._pq[1][2] == 'b'


def test_counter_increments():
    pq = HeapPriorityQueue()
    pq.add('a')
    pq.add('b')
    counters = [entry[1] for entry in pq._entry_map.values()]
    assert counters == [0, 1]


# --------------------------- remove uses sentinel ---------------------------

def test_remove_marks_entry():
    from boltons.queueutils import _REMOVED
    pq = HeapPriorityQueue()
    pq.add('a')
    entry = pq._entry_map['a']
    pq.remove('a')
    assert entry[-1] is _REMOVED
    assert 'a' not in pq._entry_map
