import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------------------------------------------------------------------
# Test double used to exercise the abstract base class directly
# ---------------------------------------------------------------------------

class _ListBackedPriorityQueue(BasePriorityQueue):
    """A minimal concrete subclass using an unsorted list and linear scans.

    This lets us exercise BasePriorityQueue's own logic independently of the
    heapq/bisect subclasses.
    """

    _backend_type = list

    @staticmethod
    def _push_entry(backend, entry):
        backend.append(entry)

    @staticmethod
    def _pop_entry(backend):
        # Find the minimum entry, remove and return it.
        best_idx = 0
        best = backend[0]
        for idx in range(1, len(backend)):
            if backend[idx] < best:
                best = backend[idx]
                best_idx = idx
        return backend.pop(best_idx)


ALL_CONCRETE_CLASSES = [
    HeapPriorityQueue,
    SortedPriorityQueue,
    _ListBackedPriorityQueue,
]

# Classes whose backends maintain their head at index 0 (heap and sorted).
HEAD_ORDERED_CLASSES = [
    HeapPriorityQueue,
    SortedPriorityQueue,
]


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


def test_subclass_relationships():
    assert issubclass(HeapPriorityQueue, BasePriorityQueue)
    assert issubclass(SortedPriorityQueue, BasePriorityQueue)


# ---------------------------------------------------------------------------
# Basic behavior across all concrete implementations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_empty_queue_len(cls):
    pq = cls()
    assert len(pq) == 0


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_add_and_len(cls):
    pq = cls()
    pq.add('a', 1)
    assert len(pq) == 1
    pq.add('b', 2)
    assert len(pq) == 2


# The full peek-then-pop ordering sequence only holds for backends that keep
# the minimum entry at _pq[0]. The generic list-backed double does not, so its
# peek returns the first-added surviving entry rather than the highest-priority.
@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_ordering_higher_priority_first(cls):
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


@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_ordering_pop_only(cls):
    pq = cls()
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.pop() == 'medium priority task 1'
    assert pq.pop() == 'medium priority task 2'
    assert pq.pop() == 'low priority task'
    assert len(pq) == 0


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add('a')
    pq.add('b', -1)
    # default flavor: larger priority = higher priority, default 0
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_priority_accepts_none_as_zero(cls):
    pq = cls()
    pq.add('a', None)
    pq.add('b', 1)
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_fifo_among_equal_priorities(cls):
    pq = cls()
    pq.add('first', 1)
    pq.add('second', 1)
    pq.add('third', 1)
    assert pq.pop() == 'first'
    assert pq.pop() == 'second'
    assert pq.pop() == 'third'


# Peek only returns the head of the backend for head-ordered backends.
@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add('only', 5)
    assert pq.peek() == 'only'
    assert pq.peek() == 'only'
    assert len(pq) == 1
    assert pq.pop() == 'only'


# For all backends, adding an existing task replaces its entry (the old one
# is tombstoned). Because the generic list backend does not cull non-head
# tombstones, only pop-ordering is meaningful for it. Restrict peek/length
# interactions to head-ordered backends and test pop ordering everywhere.
@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_add_existing_task_updates_priority(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.add('a', 5)  # re-add with higher priority
    assert len(pq) == 2
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_add_existing_task_replaces_entry(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.add('a', 5)
    # The re-added 'a' should come out first.
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'
    assert len(pq) == 0


# ---------------------------------------------------------------------------
# Empty-queue behavior: peek/pop with and without default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_peek_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_pop_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_peek_empty_with_default(cls):
    pq = cls()
    assert pq.peek(default='fallback') == 'fallback'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_pop_empty_with_default(cls):
    pq = cls()
    assert pq.pop(default='fallback') == 'fallback'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_empty_default_message(cls):
    pq = cls()
    with pytest.raises(IndexError, match='peek on empty queue'):
        pq.peek()
    with pytest.raises(IndexError, match='pop on empty queue'):
        pq.pop()


# ---------------------------------------------------------------------------
# Remove behavior
# ---------------------------------------------------------------------------

# Only head-ordered backends reliably cull a removed entry that has become
# the head; the generic list backend leaves tombstones in place until they
# happen to reach index 0. Restrict these removal semantics accordingly.

@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_remove_decrements_len(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    assert len(pq) == 1
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_remove_absent_raises_keyerror(cls):
    pq = cls()
    pq.add('a', 1)
    with pytest.raises(KeyError):
        pq.remove('missing')


@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_remove_then_peek_skips_removed(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 1)
    pq.remove('a')
    assert pq.peek() == 'b'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_remove_all_then_empty(cls):
    pq = cls()
    pq.add('a', 2)
    pq.add('b', 1)
    pq.remove('a')
    pq.remove('b')
    assert len(pq) == 0


@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_remove_all_then_pop_raises(cls):
    pq = cls()
    pq.add('a', 2)
    pq.add('b', 1)
    pq.remove('a')
    pq.remove('b')
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_remove_absent_does_not_change_len(cls):
    pq = cls()
    pq.add('a', 1)
    with pytest.raises(KeyError):
        pq.remove('b')
    assert len(pq) == 1


# ---------------------------------------------------------------------------
# priority_key constructor argument
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_custom_priority_key(cls):
    # Invert the default ordering: lower number = higher priority.
    pq = cls(priority_key=lambda p: float(p or 0))
    pq.add('low', 5)
    pq.add('high', 1)
    assert pq.pop() == 'high'
    assert pq.pop() == 'low'


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_custom_priority_key_gets_raw_value(cls):
    seen = []

    def key(p):
        seen.append(p)
        return -float(p or 0)

    pq = cls(priority_key=key)
    pq.add('a', 3)
    assert seen == [3]


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_unexpected_kwarg_raises_typeerror(cls):
    with pytest.raises(TypeError):
        cls(bogus_argument=123)


# ---------------------------------------------------------------------------
# Various task types (hashable objects)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_tuple_tasks(cls):
    pq = cls()
    pq.add(('x', 1), 1)
    pq.add(('y', 2), 2)
    assert pq.pop() == ('y', 2)


@pytest.mark.parametrize("cls", ALL_CONCRETE_CLASSES)
def test_int_and_string_mixed_by_priority(cls):
    pq = cls()
    pq.add(100, 1)
    pq.add('hello', 2)
    pq.add(3.5, 0)
    assert pq.pop() == 'hello'
    assert pq.pop() == 100
    assert pq.pop() == 3.5


# ---------------------------------------------------------------------------
# Backend-specific behavior sanity checks
# ---------------------------------------------------------------------------

def test_heap_backend_is_heap_ordered():
    pq = HeapPriorityQueue()
    for task, prio in [('a', 1), ('b', 3), ('c', 2), ('d', 5)]:
        pq.add(task, prio)
    assert pq.pop() == 'd'
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


def test_sorted_backend_maintains_order():
    pq = SortedPriorityQueue()
    for task, prio in [('a', 1), ('b', 3), ('c', 2), ('d', 5)]:
        pq.add(task, prio)
    assert pq._pq[0][2] == 'd'
    assert pq.pop() == 'd'
    assert pq._pq[0][2] == 'b'
    assert pq.pop() == 'b'
    assert pq._pq[0][2] == 'c'
    assert pq.pop() == 'c'
    assert pq._pq[0][2] == 'a'
    assert pq.pop() == 'a'


# ---------------------------------------------------------------------------
# Base class internals / abstract defaults
# ---------------------------------------------------------------------------

def test_base_default_backend_type_is_list():
    assert BasePriorityQueue._backend_type is list


def test_base_push_pop_entry_are_noops():
    backend = []
    entry = [1, 2, 'task']
    assert BasePriorityQueue._push_entry(backend, entry) is None
    assert backend == []
    assert BasePriorityQueue._pop_entry(backend) is None


def test_base_default_priority_key_negates():
    key = BasePriorityQueue._default_priority_key
    assert key(5) == -5.0
    assert key(None) == -0.0
    assert key(0) == -0.0


def test_concrete_queue_backend_types():
    assert isinstance(HeapPriorityQueue()._pq, list)
    assert HeapPriorityQueue()._pq == []
    sorted_pq = SortedPriorityQueue()
    assert isinstance(sorted_pq._pq, list)
    assert len(sorted_pq._pq) == 0


# ---------------------------------------------------------------------------
# Repeated remove/add cycling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", HEAD_ORDERED_CLASSES)
def test_remove_readd_cycle(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    pq.add('a', 3)
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'
    assert len(pq) == 0
