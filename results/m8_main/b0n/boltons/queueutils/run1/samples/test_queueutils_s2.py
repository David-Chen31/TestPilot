import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


class CustomQueue(BasePriorityQueue):
    """A minimal concrete subclass exercising the base hooks."""

    _backend_type = list

    @staticmethod
    def _push_entry(backend, entry):
        backend.append(entry)
        backend.sort(key=lambda e: (e[0], e[1]))

    @staticmethod
    def _pop_entry(backend):
        return backend.pop(0)


# ---------------------------------------------------------------------------
# Aliasing / module-level facts
# ---------------------------------------------------------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


def test_all_subclasses_of_base():
    assert issubclass(HeapPriorityQueue, BasePriorityQueue)
    assert issubclass(SortedPriorityQueue, BasePriorityQueue)


# ---------------------------------------------------------------------------
# Basic behavior across both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_empty_queue_len(QueueCls):
    q = QueueCls()
    assert len(q) == 0


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_add_and_len(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('b', 2)
    q.add('c', 3)
    assert len(q) == 3


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_pop_order_higher_priority_first(QueueCls):
    q = QueueCls()
    q.add('low', 0)
    q.add('high', 2)
    q.add('medium', 1)
    assert q.pop() == 'high'
    assert q.pop() == 'medium'
    assert q.pop() == 'low'
    assert len(q) == 0


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_peek_does_not_remove(QueueCls):
    q = QueueCls()
    q.add('low', 0)
    q.add('high', 2)
    assert q.peek() == 'high'
    assert q.peek() == 'high'
    assert len(q) == 2
    assert q.pop() == 'high'
    assert q.peek() == 'low'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_fifo_stability_equal_priority(QueueCls):
    q = QueueCls()
    q.add('first', 1)
    q.add('second', 1)
    q.add('third', 1)
    assert q.pop() == 'first'
    assert q.pop() == 'second'
    assert q.pop() == 'third'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_default_priority_is_zero(QueueCls):
    q = QueueCls()
    q.add('zero_default')
    q.add('negative', -1)
    q.add('positive', 1)
    assert q.pop() == 'positive'
    assert q.pop() == 'zero_default'
    assert q.pop() == 'negative'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_none_priority_treated_as_zero(QueueCls):
    q = QueueCls()
    q.add('a', None)
    q.add('b', 1)
    assert q.pop() == 'b'
    assert q.pop() == 'a'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_readd_changes_priority(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('b', 5)
    assert q.peek() == 'b'
    # raise a's priority above b
    q.add('a', 10)
    assert len(q) == 2
    assert q.pop() == 'a'
    assert q.pop() == 'b'
    assert len(q) == 0


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_remove(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('b', 2)
    q.add('c', 3)
    q.remove('b')
    assert len(q) == 2
    assert q.pop() == 'c'
    assert q.pop() == 'a'
    assert len(q) == 0


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_remove_top(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('b', 2)
    q.remove('b')
    assert q.peek() == 'a'
    assert q.pop() == 'a'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_remove_missing_raises(QueueCls):
    q = QueueCls()
    with pytest.raises(KeyError):
        q.remove('missing')
    q.add('a', 1)
    with pytest.raises(KeyError):
        q.remove('missing')


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_pop_empty_raises_indexerror(QueueCls):
    q = QueueCls()
    with pytest.raises(IndexError):
        q.pop()


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_peek_empty_raises_indexerror(QueueCls):
    q = QueueCls()
    with pytest.raises(IndexError):
        q.peek()


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_pop_empty_default(QueueCls):
    q = QueueCls()
    sentinel = object()
    assert q.pop(sentinel) is sentinel
    assert q.pop('default-value') == 'default-value'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_peek_empty_default(QueueCls):
    q = QueueCls()
    sentinel = object()
    assert q.peek(sentinel) is sentinel
    assert q.peek('default-value') == 'default-value'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_pop_default_not_used_when_nonempty(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    assert q.pop('unused-default') == 'a'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_cull_after_removing_all(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('b', 2)
    q.remove('a')
    q.remove('b')
    assert len(q) == 0
    with pytest.raises(IndexError):
        q.peek()
    assert q.peek('empty') == 'empty'
    with pytest.raises(IndexError):
        q.pop()
    assert q.pop('empty') == 'empty'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_iterator_of_pops(QueueCls):
    q = QueueCls()
    for i in range(5):
        q.add(i, i)
    results = [q.pop() for _ in range(5)]
    assert results == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_priority_key_customization(QueueCls):
    # reverse: lower numbers = higher priority
    q = QueueCls(priority_key=lambda p: float(p if p is not None else 0))
    q.add('a', 3)
    q.add('b', 1)
    q.add('c', 2)
    assert q.pop() == 'b'
    assert q.pop() == 'c'
    assert q.pop() == 'a'


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_unexpected_kwargs_raises(QueueCls):
    with pytest.raises(TypeError):
        QueueCls(bogus=True)


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_priority_key_receives_none(QueueCls):
    captured = []

    def key(p):
        captured.append(p)
        return -(p or 0)

    q = QueueCls(priority_key=key)
    q.add('a', None)
    assert captured == [None]


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_duplicate_add_does_not_grow_len(QueueCls):
    q = QueueCls()
    q.add('a', 1)
    q.add('a', 2)
    q.add('a', 3)
    assert len(q) == 1
    assert q.pop() == 'a'
    assert len(q) == 0


@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_float_and_int_priorities(QueueCls):
    q = QueueCls()
    q.add('one', 1)
    q.add('half', 0.5)
    q.add('two', 2)
    assert q.pop() == 'two'
    assert q.pop() == 'one'
    assert q.pop() == 'half'


# ---------------------------------------------------------------------------
# Base class behavior
# ---------------------------------------------------------------------------

def test_base_class_abstract_push_pop_are_noops():
    q = BasePriorityQueue()
    # abstract methods do nothing by default
    q.add('a', 1)
    assert len(q) == 1
    # _push_entry noop means backend remains empty
    assert q._pq == []
    with pytest.raises(IndexError):
        q.pop()


def test_base_class_default_backend_type():
    q = BasePriorityQueue()
    assert isinstance(q._pq, list)
    assert q._entry_map == {}
    assert len(q) == 0


def test_custom_subclass_works():
    q = CustomQueue()
    q.add('a', 1)
    q.add('b', 3)
    q.add('c', 2)
    assert q.pop() == 'b'
    assert q.pop() == 'c'
    assert q.pop() == 'a'


def test_entry_map_contains_entries():
    q = HeapPriorityQueue()
    q.add('a', 1)
    assert 'a' in q._entry_map
    q.pop()
    assert 'a' not in q._entry_map


def test_remove_marks_entry_removed():
    q = HeapPriorityQueue()
    q.add('a', 1)
    entry = q._entry_map['a']
    q.remove('a')
    # entry last element replaced with sentinel
    assert entry[-1] is not 'a' or entry[-1] != 'a'


# ---------------------------------------------------------------------------
# Docstring example
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("QueueCls", QUEUE_CLASSES)
def test_docstring_example(QueueCls):
    pq = QueueCls()
    pq.add('low priority task', 0)
    pq.add('high priority task', 2)
    pq.add('medium priority task 1', 1)
    pq.add('medium priority task 2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high priority task'
    assert pq.peek() == 'medium priority task 1'
    assert len(pq) == 3
