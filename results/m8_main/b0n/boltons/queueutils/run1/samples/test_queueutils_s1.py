import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------------------------------------------------------------------
# A concrete subclass exercising the abstract-ish BasePriorityQueue hooks.
# ---------------------------------------------------------------------------
class ListPriorityQueue(BasePriorityQueue):
    """Simple backend: list, push appends, pop pops from front."""

    @staticmethod
    def _push_entry(backend, entry):
        backend.append(entry)

    @staticmethod
    def _pop_entry(backend):
        return backend.pop(0)


# ---------------------------------------------------------------------------
# Basic behavior on every concrete queue class
# ---------------------------------------------------------------------------
QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue, ListPriorityQueue]


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_empty_queue_len(cls):
    pq = cls()
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_add_and_len(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    assert len(pq) == 2


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_order_by_priority(cls):
    pq = cls()
    pq.add('low', 0)
    pq.add('high', 2)
    pq.add('medium1', 1)
    pq.add('medium2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high'
    assert pq.peek() == 'medium1'
    assert len(pq) == 3
    assert pq.pop() == 'medium1'
    assert pq.pop() == 'medium2'
    assert pq.pop() == 'low'
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_priority_default_is_zero(cls):
    pq = cls()
    pq.add('a')
    pq.add('b', 0)
    pq.add('c', -1)
    # a and b have equal priority (0); c has lower priority.
    assert len(pq) == 3
    first = pq.pop()
    second = pq.pop()
    assert {first, second} == {'a', 'b'}
    assert pq.pop() == 'c'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_none_priority_treated_as_zero(cls):
    pq = cls()
    pq.add('a', None)
    pq.add('b', 1)
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_empty_raises_index_error(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_empty_with_default(cls):
    pq = cls()
    assert pq.pop('default') == 'default'
    assert pq.pop(None) is None


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_empty_raises_index_error(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_empty_with_default(cls):
    pq = cls()
    assert pq.peek('default') == 'default'
    assert pq.peek(None) is None


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add('a', 1)
    assert pq.peek() == 'a'
    assert pq.peek() == 'a'
    assert len(pq) == 1


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    pq.remove('a')
    assert len(pq) == 1
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_missing_raises_key_error(cls):
    pq = cls()
    with pytest.raises(KeyError):
        pq.remove('nope')


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_then_peek_skips_removed(cls):
    pq = cls()
    pq.add('a', 2)
    pq.add('b', 1)
    pq.remove('a')
    assert pq.peek() == 'b'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_all_then_pop_raises(cls):
    pq = cls()
    pq.add('a', 2)
    pq.add('b', 1)
    pq.remove('a')
    pq.remove('b')
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_re_add_updates_priority(cls):
    pq = cls()
    pq.add('a', 1)
    pq.add('b', 2)
    # Re-adding 'a' with higher priority moves it to the front.
    pq.add('a', 5)
    assert len(pq) == 2
    assert pq.pop() == 'a'
    assert pq.pop() == 'b'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_re_add_lower_priority(cls):
    pq = cls()
    pq.add('a', 5)
    pq.add('b', 2)
    pq.add('a', 1)
    assert len(pq) == 2
    assert pq.pop() == 'b'
    assert pq.pop() == 'a'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_unhashable_task_raises():
    pq = HeapPriorityQueue()
    with pytest.raises(TypeError):
        pq.add(['unhashable'], 1)


# ---------------------------------------------------------------------------
# Custom priority_key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_custom_priority_key(cls):
    # Without negation, larger number = lower priority.
    pq = cls(priority_key=lambda p: float(p or 0))
    pq.add('low', 5)
    pq.add('high', 1)
    assert pq.pop() == 'high'
    assert pq.pop() == 'low'


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_custom_priority_key_called(cls):
    calls = []

    def key(p):
        calls.append(p)
        return -float(p or 0)

    pq = cls(priority_key=key)
    pq.add('a', 3)
    assert calls == [3]


def test_unexpected_kwargs_raises_type_error():
    with pytest.raises(TypeError):
        HeapPriorityQueue(bogus_kwarg=1)


def test_default_priority_key_value():
    key = BasePriorityQueue._default_priority_key
    assert key(5) == -5.0
    assert key(None) == 0.0
    assert key(0) == 0.0
    assert key(-3) == 3.0


# ---------------------------------------------------------------------------
# Base class semantics
# ---------------------------------------------------------------------------
def test_base_backend_type_is_list():
    pq = BasePriorityQueue()
    assert isinstance(pq._pq, list)


def test_base_push_pop_are_noop():
    pq = BasePriorityQueue()
    # Adding works (backend list, push noop), but pop will fail to find entry.
    pq.add('a', 1)
    assert len(pq) == 1
    # _pop_entry is a noop returning None; unpacking from it fails.
    with pytest.raises(TypeError):
        pq.pop()


def test_base_constructor_creates_entry_map_and_counter():
    pq = BasePriorityQueue()
    assert pq._entry_map == {}
    assert next(pq._counter) == 0


# ---------------------------------------------------------------------------
# Backend-specific behavior
# ---------------------------------------------------------------------------
def test_heap_backend_is_list():
    pq = HeapPriorityQueue()
    assert isinstance(pq._pq, list)


def test_sorted_backend_type():
    from boltons.queueutils import BList
    pq = SortedPriorityQueue()
    assert isinstance(pq._pq, BList)


def test_module_alias():
    assert PriorityQueue is SortedPriorityQueue


def test_heap_pop_entry():
    backend = []
    HeapPriorityQueue._push_entry(backend, [1, 0, 'a'])
    HeapPriorityQueue._push_entry(backend, [2, 1, 'b'])
    entry = HeapPriorityQueue._pop_entry(backend)
    assert entry == [2, 1, 'b']


def test_sorted_pop_entry():
    backend = []
    SortedPriorityQueue._push_entry(backend, [2, 1, 'b'])
    SortedPriorityQueue._push_entry(backend, [1, 0, 'a'])
    entry = SortedPriorityQueue._pop_entry(backend)
    assert entry == [1, 0, 'a']


def test_counter_increments():
    pq = HeapPriorityQueue()
    pq.add('a', 1)
    pq.add('b', 1)
    assert next(pq._counter) == 2


def test_equal_priority_fifo_order():
    pq = HeapPriorityQueue()
    pq.add('first', 1)
    pq.add('second', 1)
    pq.add('third', 1)
    assert pq.pop() == 'first'
    assert pq.pop() == 'second'
    assert pq.pop() == 'third'
