import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]
QUEUE_IDS = ["heap", "sorted"]


# ---------------------------------------------------------------------------
# Basic construction / module-level behavior
# ---------------------------------------------------------------------------

def test_priority_queue_alias_is_sorted():
    assert PriorityQueue is SortedPriorityQueue


def test_unexpected_kwargs_raise_type_error():
    with pytest.raises(TypeError):
        HeapPriorityQueue(foo=1)

    with pytest.raises(TypeError):
        SortedPriorityQueue(bar=2)


def test_default_backend_types():
    assert HeapPriorityQueue()._pq.__class__ is list
    # SortedPriorityQueue uses BList (or list fallback)
    hq = HeapPriorityQueue()
    sq = SortedPriorityQueue()
    assert isinstance(hq._pq, list)
    assert hasattr(sq._pq, "pop") and hasattr(sq._pq, "insert")


# ---------------------------------------------------------------------------
# add / len / peek / pop behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_add_len_peek_pop_basic(cls):
    pq = cls()
    assert len(pq) == 0

    pq.add("low priority task", 0)
    pq.add("high priority task", 2)
    pq.add("medium priority task 1", 1)
    pq.add("medium priority task 2", 1)

    assert len(pq) == 4
    assert pq.peek() == "high priority task"
    # peek does not remove
    assert len(pq) == 4

    assert pq.pop() == "high priority task"
    assert len(pq) == 3
    assert pq.peek() == "medium priority task 1"

    # FIFO among equal priorities
    assert pq.pop() == "medium priority task 1"
    assert pq.pop() == "medium priority task 2"
    assert pq.pop() == "low priority task"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add("a")
    pq.add("b", None)
    pq.add("c", 0)
    # all equal priority -> insertion order
    assert pq.pop() == "a"
    assert pq.pop() == "b"
    assert pq.pop() == "c"


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_negative_priority_lower_than_zero(cls):
    pq = cls()
    pq.add("zero", 0)
    pq.add("negative", -5)
    pq.add("positive", 3)
    assert pq.pop() == "positive"
    assert pq.pop() == "zero"
    assert pq.pop() == "negative"


# ---------------------------------------------------------------------------
# peek/pop on empty queue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_peek_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.peek()
    assert "peek on empty queue" in str(excinfo.value)


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_pop_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError) as excinfo:
        pq.pop()
    assert "pop on empty queue" in str(excinfo.value)


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_peek_empty_returns_default(cls):
    pq = cls()
    assert pq.peek("default") == "default"
    assert pq.peek(None) is None


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_pop_empty_returns_default(cls):
    pq = cls()
    assert pq.pop("default") == "default"
    assert pq.pop(None) is None


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_peek_default_when_tasks_removed(cls):
    pq = cls()
    pq.add("a", 1)
    pq.remove("a")
    # all entries culled
    assert pq.peek("empty") == "empty"
    assert pq.pop("empty") == "empty"


# ---------------------------------------------------------------------------
# remove behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_remove_missing_raises_keyerror(cls):
    pq = cls()
    with pytest.raises(KeyError):
        pq.remove("nonexistent")


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_remove_reduces_length_and_skips_task(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.add("c", 3)
    assert len(pq) == 3

    pq.remove("c")  # highest priority
    assert len(pq) == 2
    assert pq.pop() == "b"
    assert pq.pop() == "a"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_remove_then_peek_culls(cls):
    pq = cls()
    pq.add("a", 1)
    pq.remove("a")
    assert len(pq) == 0
    with pytest.raises(IndexError):
        pq.peek()


# ---------------------------------------------------------------------------
# re-adding an existing task changes priority
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_add_existing_task_updates_priority(cls):
    pq = cls()
    pq.add("task", 1)
    pq.add("other", 5)
    pq.add("task", 10)  # re-add with higher priority
    assert len(pq) == 2
    assert pq.pop() == "task"
    assert pq.pop() == "other"


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_add_existing_task_keeps_single_entry(cls):
    pq = cls()
    pq.add("task", 1)
    pq.add("task", 2)
    pq.add("task", 3)
    assert len(pq) == 1
    assert pq.pop() == "task"
    assert len(pq) == 0


# ---------------------------------------------------------------------------
# custom priority_key
# ---------------------------------------------------------------------------

def test_custom_priority_key_inverted():
    # higher priority number should now be LOWER priority
    pq = HeapPriorityQueue(priority_key=lambda p: p or 0)
    pq.add("a", 1)
    pq.add("b", 2)
    pq.add("c", 3)
    assert pq.pop() == "a"
    assert pq.pop() == "b"
    assert pq.pop() == "c"


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_custom_priority_key_used(cls):
    calls = []

    def key(p):
        calls.append(p)
        return -(p or 0)

    pq = cls(priority_key=key)
    pq.add("a", 5)
    assert calls == [5]
    pq.add("b", 3)
    assert calls == [5, 3]
    assert pq.pop() == "a"


@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_custom_priority_key_with_none(cls):
    pq = cls(priority_key=lambda p: float("-inf") if p is None else p)
    pq.add("a", None)
    pq.add("b", 1)
    assert pq.pop() == "a"


# ---------------------------------------------------------------------------
# Ordering stability among equal priorities (insertion order)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_equal_priorities_fifo(cls):
    pq = cls()
    for i in range(10):
        pq.add(i, 0)
    results = [pq.pop() for _ in range(10)]
    assert results == list(range(10))


# ---------------------------------------------------------------------------
# Direct static methods
# ---------------------------------------------------------------------------

def test_heap_push_pop_entry():
    backend = []
    HeapPriorityQueue._push_entry(backend, [1, 0, "a"])
    HeapPriorityQueue._push_entry(backend, [2, 1, "b"])
    HeapPriorityQueue._push_entry(backend, [0, 2, "c"])
    # heappop returns smallest entry
    assert HeapPriorityQueue._pop_entry(backend) == [0, 2, "c"]
    assert HeapPriorityQueue._pop_entry(backend) == [1, 0, "a"]
    assert HeapPriorityQueue._pop_entry(backend) == [2, 1, "b"]


def test_sorted_push_pop_entry():
    backend = []
    SortedPriorityQueue._push_entry(backend, [1, 0, "a"])
    SortedPriorityQueue._push_entry(backend, [2, 1, "b"])
    SortedPriorityQueue._push_entry(backend, [0, 2, "c"])
    assert SortedPriorityQueue._pop_entry(backend) == [0, 2, "c"]
    assert SortedPriorityQueue._pop_entry(backend) == [1, 0, "a"]
    assert SortedPriorityQueue._pop_entry(backend) == [2, 1, "b"]


# ---------------------------------------------------------------------------
# Base class abstract stubs
# ---------------------------------------------------------------------------

def test_base_priority_queue_stubs():
    # _push_entry and _pop_entry are abstract no-ops returning None
    assert BasePriorityQueue._push_entry([], [1, 0, "t"]) is None
    assert BasePriorityQueue._pop_entry([]) is None


def test_base_priority_queue_backend_type_is_list():
    assert BasePriorityQueue._backend_type is list


def test_base_priority_queue_default_priority_key():
    key = BasePriorityQueue._default_priority_key
    assert key(5) == -5.0
    assert key(None) == 0.0
    assert key(0) == 0.0


def test_base_priority_queue_entry_map_and_counter():
    pq = BasePriorityQueue()
    assert pq._entry_map == {}
    # add uses stubs so entry is pushed but no real backend push
    pq.add("x", 3)
    assert "x" in pq._entry_map
    entry = pq._entry_map["x"]
    assert entry[0] == -3.0
    assert entry[2] == "x"
    assert len(pq) == 1


# ---------------------------------------------------------------------------
# Mixed sequence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", QUEUE_CLASSES, ids=QUEUE_IDS)
def test_mixed_operations(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.add("c", 3)
    pq.remove("b")
    pq.add("d", 0)
    pq.add("a", 5)  # move a up
    assert len(pq) == 3
    assert pq.pop() == "a"
    assert pq.pop() == "c"
    assert pq.pop() == "d"
    assert len(pq) == 0
