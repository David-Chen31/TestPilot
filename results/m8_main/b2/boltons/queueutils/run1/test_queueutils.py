import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


# ---------------------------------------------------------------------------
# Basic construction / attributes
# ---------------------------------------------------------------------------

def test_priority_queue_alias():
    assert PriorityQueue is SortedPriorityQueue


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_empty_queue_initial_state(cls):
    pq = cls()
    assert len(pq) == 0
    assert pq._pq == pq._backend_type()
    assert pq._entry_map == {}


def test_unexpected_kwargs_raises():
    with pytest.raises(TypeError):
        BasePriorityQueue(foo=1)


def test_unexpected_kwargs_message():
    # The kind/type of the exception matters, as does the mention of
    # "unexpected keyword arguments".
    with pytest.raises(TypeError) as excinfo:
        BasePriorityQueue(foo=1)
    assert "unexpected keyword arguments" in str(excinfo.value)


def test_unexpected_kwargs_exact_message():
    # Pin down the exact prefix of the error message.
    with pytest.raises(TypeError) as excinfo:
        BasePriorityQueue(foo=1)
    assert str(excinfo.value).startswith("unexpected keyword arguments: ")


def test_unexpected_kwargs_multiple_message():
    with pytest.raises(TypeError) as excinfo:
        BasePriorityQueue(foo=1, bar=2)
    msg = str(excinfo.value)
    assert "unexpected keyword arguments" in msg
    # 'foo' and 'bar' should be referenced in the message via %r of keys
    assert "foo" in msg
    assert "bar" in msg


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_default_priority_key(cls):
    pq = cls()
    # default key negates priority; None -> 0
    assert pq._get_priority(None) == 0.0
    assert pq._get_priority(5) == -5.0
    assert pq._get_priority(-3) == 3.0


def test_custom_priority_key():
    # The priority_key result is used directly as the entry's priority,
    # with lower values being popped first. With an identity key, the
    # lowest priority number is therefore popped first.
    pq = HeapPriorityQueue(priority_key=lambda p: p)
    pq.add("a", 1)
    pq.add("b", 5)
    pq.add("c", 3)
    assert pq.pop() == "a"
    assert pq.pop() == "c"
    assert pq.pop() == "b"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_custom_priority_key_negating(cls):
    # Using an explicit negating key reproduces "higher number first".
    pq = cls(priority_key=lambda p: -p)
    pq.add("a", 1)
    pq.add("b", 5)
    pq.add("c", 3)
    assert pq.pop() == "b"
    assert pq.pop() == "c"
    assert pq.pop() == "a"


# ---------------------------------------------------------------------------
# add / pop / peek ordering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_basic_ordering(cls):
    pq = cls()
    pq.add("low priority task", 0)
    pq.add("high priority task", 2)
    pq.add("medium priority task 1", 1)
    pq.add("medium priority task 2", 1)
    assert len(pq) == 4
    assert pq.pop() == "high priority task"
    assert pq.peek() == "medium priority task 1"
    assert len(pq) == 3
    assert pq.pop() == "medium priority task 1"
    assert pq.pop() == "medium priority task 2"
    assert pq.pop() == "low priority task"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add("a")
    pq.add("b")
    # equal priorities, FIFO via counter
    assert pq.pop() == "a"
    assert pq.pop() == "b"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_fifo_for_equal_priorities(cls):
    pq = cls()
    for i in range(5):
        pq.add(i, 1)
    assert [pq.pop() for _ in range(5)] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_negative_and_zero_priority(cls):
    pq = cls()
    pq.add("zero", 0)
    pq.add("neg", -1)
    pq.add("pos", 1)
    assert pq.pop() == "pos"
    assert pq.pop() == "zero"
    assert pq.pop() == "neg"


def test_non_numeric_priority_raises():
    pq = HeapPriorityQueue()
    with pytest.raises(ValueError):
        pq.add("bad", "not a number")


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add("x", 1)
    assert pq.peek() == "x"
    assert pq.peek() == "x"
    assert len(pq) == 1
    assert pq.pop() == "x"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_empty_message(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.peek()
    assert str(excinfo.value) == "peek on empty queue"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_peek_empty_with_default(cls):
    pq = cls()
    sentinel = object()
    assert pq.peek(default=sentinel) is sentinel
    assert pq.peek(default=None) is None


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_empty_message(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.pop()
    assert str(excinfo.value) == "pop on empty queue"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_pop_empty_with_default(cls):
    pq = cls()
    sentinel = object()
    assert pq.pop(default=sentinel) is sentinel
    assert pq.pop(default=None) is None


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_existing(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.remove("a")
    assert len(pq) == 1
    assert pq.pop() == "b"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_absent_raises_keyerror(cls):
    pq = cls()
    pq.add("a", 1)
    with pytest.raises(KeyError):
        pq.remove("missing")


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_then_peek_culls(cls):
    pq = cls()
    pq.add("a", 5)
    pq.add("b", 1)
    pq.remove("a")
    # a was top; after culling should be b
    assert pq.peek() == "b"
    assert pq.pop() == "b"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_remove_all_then_pop_empty(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.remove("a")
    pq.remove("b")
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.pop()


# ---------------------------------------------------------------------------
# Re-adding an existing task updates priority
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_readd_updates_priority(cls):
    pq = cls()
    pq.add("a", 0)
    pq.add("b", 5)
    pq.add("c", 10)
    # Raise a's priority to the top
    pq.add("a", 100)
    assert len(pq) == 3
    assert pq.pop() == "a"
    assert pq.pop() == "c"
    assert pq.pop() == "b"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_readd_same_priority_no_duplicate(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("a", 1)
    assert len(pq) == 1
    assert pq.pop() == "a"
    with pytest.raises(IndexError):
        pq.pop()


# ---------------------------------------------------------------------------
# BasePriorityQueue abstract behavior
# ---------------------------------------------------------------------------

def test_base_push_pop_entry_are_noops():
    backend = []
    entry = [1, 0, "task"]
    BasePriorityQueue._push_entry(backend, entry)
    assert backend == []
    assert BasePriorityQueue._pop_entry(backend) is None


def test_base_cull_raise_exc_false_on_empty():
    pq = BasePriorityQueue()
    # _cull with raise_exc=False should not raise
    pq._cull(raise_exc=False)


def test_base_cull_raises_on_empty():
    pq = BasePriorityQueue()
    with pytest.raises(IndexError):
        pq._cull()


def test_base_cull_empty_message():
    pq = BasePriorityQueue()
    with pytest.raises(IndexError) as excinfo:
        pq._cull()
    assert str(excinfo.value) == "empty priority queue"


@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_dunder_len_reflects_entry_map(cls):
    pq = cls()
    assert len(pq) == 0
    pq.add("a")
    assert len(pq) == 1
    pq.add("b")
    assert len(pq) == 2
    pq.remove("a")
    assert len(pq) == 1
    pq.pop()
    assert len(pq) == 0


# ---------------------------------------------------------------------------
# Unhashable tasks raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES)
def test_unhashable_task_raises(cls):
    pq = cls()
    with pytest.raises(TypeError):
        pq.add([1, 2, 3])


# ---------------------------------------------------------------------------
# Docstring assertions (to catch docstring mutations)
# ---------------------------------------------------------------------------

def test_cull_docstring_exact():
    assert BasePriorityQueue._cull.__doc__ == (
        "Remove entries marked as removed by previous :meth:`remove` calls."
    )


def test_len_docstring_exact():
    assert BasePriorityQueue.__len__.__doc__ == (
        "Return the number of tasks in the queue."
    )
