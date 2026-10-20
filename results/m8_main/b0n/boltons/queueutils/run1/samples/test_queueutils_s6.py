import itertools

import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BACKEND_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


@pytest.fixture(params=BACKEND_CLASSES, ids=["heap", "sorted"])
def pq_class(request):
    return request.param


@pytest.fixture
def pq(pq_class):
    return pq_class()


# ---------------------------------------------------------------------------
# Module-level exports and aliases
# ---------------------------------------------------------------------------

def test_priority_queue_alias_is_sorted():
    assert PriorityQueue is SortedPriorityQueue


def test_both_backends_are_base_subclasses():
    assert issubclass(HeapPriorityQueue, BasePriorityQueue)
    assert issubclass(SortedPriorityQueue, BasePriorityQueue)


# ---------------------------------------------------------------------------
# Basic add / pop / peek / len
# ---------------------------------------------------------------------------

def test_empty_queue_len(pq):
    assert len(pq) == 0


def test_add_increases_len(pq):
    pq.add('task')
    assert len(pq) == 1
    pq.add('task2')
    assert len(pq) == 2


def test_basic_priority_ordering(pq):
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)

    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.peek() == 'medium priority task 1'
    assert len(pq) == 3


def test_pop_order_multiple(pq):
    pq.add('a', 1)
    pq.add('b', 3)
    pq.add('c', 2)
    assert [pq.pop() for _ in range(3)] == ['b', 'c', 'a']


def test_default_priority_is_zero(pq):
    pq.add('t')
    assert pq.peek() == 't'


def test_peek_does_not_remove(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    assert pq.peek() == 'b'
    assert pq.peek() == 'b'
    assert len(pq) == 2
    assert pq.pop() == 'b'
    assert pq.peek() == 'a'


def test_higher_priority_popped_first(pq):
    pq.add('one', 1)
    pq.add('two', 10)
    pq.add('three', 5)
    assert pq.pop() == 'two'
    assert pq.pop() == 'three'
    assert pq.pop() == 'one'


def test_len_decreases_on_pop(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    pq.pop()
    assert len(pq) == 1
    pq.pop()
    assert len(pq) == 0


# ---------------------------------------------------------------------------
# Priority as None / falsy / string-coercible
# ---------------------------------------------------------------------------

def test_priority_none_defaults_to_zero(pq):
    pq.add('x', None)
    assert len(pq) == 1
    assert pq.pop() == 'x'


def test_priority_string_coercible(pq):
    pq.add('low', '1')
    pq.add('high', '5')
    assert pq.pop() == 'high'
    assert pq.pop() == 'low'


# ---------------------------------------------------------------------------
# add with existing task changes priority
# ---------------------------------------------------------------------------

def test_add_existing_task_updates_priority(pq):
    pq.add('task', 1)
    pq.add('other', 2)
    assert pq.peek() == 'other'

    pq.add('task', 10)
    assert len(pq) == 2
    assert pq.peek() == 'task'
    assert pq.pop() == 'task'
    assert pq.pop() == 'other'


def test_add_existing_task_does_not_duplicate(pq):
    for _ in range(5):
        pq.add('same', 1)
    assert len(pq) == 1
    assert pq.pop() == 'same'
    assert len(pq) == 0


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_existing(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    assert len(pq) == 1
    assert pq.peek() == 'b'


def test_remove_then_pop(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    pq.add('c', 3)

    pq.remove('c')  # the top
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'
    assert len(pq) == 0


def test_remove_all_then_pop_empty(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    pq.remove('b')
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.pop()


def test_remove_missing_raises_keyerror(pq):
    pq.add('a', 1)
    with pytest.raises(KeyError):
        pq.remove('missing')


def test_remove_missing_on_empty_raises_keyerror(pq):
    with pytest.raises(KeyError):
        pq.remove('missing')


def test_remove_already_removed_raises_keyerror(pq):
    pq.add('a', 1)
    pq.remove('a')
    with pytest.raises(KeyError):
        pq.remove('a')


# ---------------------------------------------------------------------------
# Empty queue behavior: pop / peek
# ---------------------------------------------------------------------------

def test_pop_empty_raises_indexerror(pq):
    with pytest.raises(IndexError):
        pq.pop()


def test_peek_empty_raises_indexerror(pq):
    with pytest.raises(IndexError):
        pq.peek()


def test_pop_empty_returns_default(pq):
    assert pq.pop(default='empty') == 'empty'


def test_peek_empty_returns_default(pq):
    assert pq.peek(default=None) is None
    assert pq.peek(default='empty') == 'empty'


def test_pop_default_after_emptying(pq):
    pq.add('a', 1)
    assert pq.pop() == 'a'
    assert pq.pop(default='done') == 'done'


# ---------------------------------------------------------------------------
# Culling of removed entries from the backend
# ---------------------------------------------------------------------------

def test_culling_via_peek(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('b')
    # peek should cull the removed 'b' entry from the backend
    assert pq.peek() == 'a'
    # after culling the removed entry, the backend should hold only 'a'
    assert len(pq._pq) == 1


def test_cull_raise_when_all_removed(pq):
    pq.add('a', 1)
    pq.remove('a')
    # backend still holds the removed entry; _cull should raise
    with pytest.raises(IndexError):
        pq._cull()
    # culling with raise_exc=False should not raise and should empty backend
    pq.add('b', 1)
    pq.remove('b')
    pq._cull(raise_exc=False)
    assert len(pq._pq) == 0


# ---------------------------------------------------------------------------
# Custom priority_key
# ---------------------------------------------------------------------------

def test_custom_priority_key():
    # lowest value = highest priority, i.e. do not negate
    def priority_key(p):
        return float(p or 0)

    pq = SortedPriorityQueue(priority_key=priority_key)
    pq.add('a', 5)
    pq.add('b', 1)
    pq.add('c', 3)
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


def test_custom_priority_key_heap():
    def priority_key(p):
        return float(p or 0)

    pq = HeapPriorityQueue(priority_key=priority_key)
    pq.add('a', 5)
    pq.add('b', 1)
    pq.add('c', 3)
    assert pq.pop() == 'b'
    assert pq.pop() == 'c'
    assert pq.pop() == 'a'


def test_unexpected_kwarg_raises_typeerror():
    with pytest.raises(TypeError):
        SortedPriorityQueue(bogus_kwarg=1)


def test_unexpected_kwarg_raises_typeerror_heap():
    with pytest.raises(TypeError):
        HeapPriorityQueue(bogus_kwarg=1)


# ---------------------------------------------------------------------------
# Internal state / tie-breaking
# ---------------------------------------------------------------------------

def test_counter_increments(pq):
    pq.add('a', 1)
    pq.add('b', 1)
    assert pq._counter is not None
    # Ensure counter has advanced from initial state by popping entries
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


def test_entry_map_tracks_tasks(pq):
    pq.add('a', 1)
    pq.add('b', 2)
    assert set(pq._entry_map.keys()) == {'a', 'b'}
    pq.remove('a')
    assert set(pq._entry_map.keys()) == {'b'}
    pq.pop()
    assert pq._entry_map == {}


def test_fifo_tiebreak_ordering(pq):
    pq.add('first', 1)
    pq.add('second', 1)
    pq.add('third', 1)
    assert pq.pop() == 'first'
    assert pq.pop() == 'second'
    assert pq.pop() == 'third'


# ---------------------------------------------------------------------------
# Mixed operations / larger scenario
# ---------------------------------------------------------------------------

def test_mixed_operations(pq):
    tasks = ['t%d' % i for i in range(10)]
    for i, t in enumerate(tasks):
        pq.add(t, i)
    assert len(pq) == 10

    # pop highest (t9)
    assert pq.pop() == 't9'

    # change priorities: lift t0 high
    pq.add('t0', 100)
    assert pq.pop() == 't0'

    # remove one and confirm others remain
    pq.remove('t5')
    popped = [pq.pop() for _ in range(7)]
    assert 't5' not in popped
    assert popped == ['t8', 't7', 't6', 't4', 't3', 't2', 't1']
    assert len(pq) == 0


def test_add_none_priority_then_explicit(pq):
    pq.add('a')
    pq.add('a', 5)
    pq.add('b', 1)
    assert len(pq) == 2
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


# ---------------------------------------------------------------------------
# Base class abstract-ish backends
# ---------------------------------------------------------------------------

def test_base_priority_queue_abstract_push_pop_do_nothing():
    q = BasePriorityQueue()
    # abstract _push_entry / _pop_entry are no-ops
    q._push_entry(q._pq, [1, 0, 'x'])
    assert q._pop_entry(q._pq) is None


def test_backends_constant_types():
    assert SortedPriorityQueue._backend_type is not None
    assert HeapPriorityQueue._backend_type is list


# ---------------------------------------------------------------------------
# Default priority key behavior
# ---------------------------------------------------------------------------

def test_default_priority_key_negates():
    key = BasePriorityQueue._default_priority_key
    assert key(5) == -5.0
    assert key(None) == 0.0
    assert key(0) == 0.0
    assert key(0.5) == -0.5
