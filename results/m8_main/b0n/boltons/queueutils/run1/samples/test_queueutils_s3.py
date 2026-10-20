import pytest

from boltons.queueutils import (
    BasePriorityQueue,
    HeapPriorityQueue,
    SortedPriorityQueue,
    PriorityQueue,
)


# ---------------------------------------------------------------------------
# Basic API / shared behavior across both concrete implementations
# ---------------------------------------------------------------------------

CONCRETE = [HeapPriorityQueue, SortedPriorityQueue]


@pytest.mark.parametrize("cls", CONCRETE)
def test_empty_queue_len_and_pop_default(cls):
    pq = cls()
    assert len(pq) == 0
    assert pq.pop("empty") == "empty"
    assert pq.peek("empty") == "empty"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_pop_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()


@pytest.mark.parametrize("cls", CONCRETE)
def test_peek_empty_raises(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.peek()


@pytest.mark.parametrize("cls", CONCRETE)
def test_add_pop_order(cls):
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


@pytest.mark.parametrize("cls", CONCRETE)
def test_default_priority_is_zero(cls):
    pq = cls()
    pq.add("a")
    pq.add("b", 0)
    pq.add("c", None)
    # all same priority, FIFO ordering by insertion (counter tiebreak)
    assert pq.pop() == "a"
    assert pq.pop() == "b"
    assert pq.pop() == "c"


@pytest.mark.parametrize("cls", CONCRETE)
def test_higher_priority_pops_first(cls):
    pq = cls()
    for i in range(5):
        pq.add(i, priority=i)
    assert [pq.pop() for _ in range(5)] == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("cls", CONCRETE)
def test_negative_priorities(cls):
    pq = cls()
    pq.add("zero", 0)
    pq.add("neg", -5)
    pq.add("pos", 3)
    assert pq.pop() == "pos"
    assert pq.pop() == "zero"
    assert pq.pop() == "neg"


@pytest.mark.parametrize("cls", CONCRETE)
def test_peek_does_not_remove(cls):
    pq = cls()
    pq.add("x", 1)
    assert pq.peek() == "x"
    assert pq.peek() == "x"
    assert len(pq) == 1
    assert pq.pop() == "x"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_peek_default_none(cls):
    pq = cls()
    assert pq.peek(None) is None
    assert pq.pop(None) is None


@pytest.mark.parametrize("cls", CONCRETE)
def test_readding_task_updates_priority(cls):
    pq = cls()
    pq.add("t", 1)
    pq.add("other", 2)
    assert len(pq) == 2
    # Re-adding same task bumps its priority, length stays 1 for it
    pq.add("t", 5)
    assert len(pq) == 2
    assert pq.pop() == "t"
    assert pq.pop() == "other"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_readd_same_task_same_priority_no_duplicate(cls):
    pq = cls()
    pq.add("t", 1)
    pq.add("t", 1)
    assert len(pq) == 1
    assert pq.pop() == "t"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_remove_task(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.add("c", 3)
    pq.remove("b")
    assert len(pq) == 2
    assert pq.pop() == "c"
    assert pq.pop() == "a"
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_remove_absent_raises_keyerror(cls):
    pq = cls()
    with pytest.raises(KeyError):
        pq.remove("missing")

    pq.add("a", 1)
    with pytest.raises(KeyError):
        pq.remove("missing")


@pytest.mark.parametrize("cls", CONCRETE)
def test_remove_then_len(cls):
    pq = cls()
    pq.add("a", 1)
    pq.remove("a")
    assert len(pq) == 0
    # Pop skips removed entries and raises on empty
    with pytest.raises(IndexError):
        pq.pop()
    assert pq.pop("d") == "d"


@pytest.mark.parametrize("cls", CONCRETE)
def test_remove_all_then_peek_default(cls):
    pq = cls()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.remove("a")
    pq.remove("b")
    assert pq.peek("none") == "none"


@pytest.mark.parametrize("cls", CONCRETE)
def test_interleaved_remove_and_pop(cls):
    pq = cls()
    for i in range(10):
        pq.add(i, i)
    pq.remove(9)
    pq.remove(5)
    result = [pq.pop() for _ in range(8)]
    assert result == [8, 7, 6, 4, 3, 2, 1, 0]


@pytest.mark.parametrize("cls", CONCRETE)
def test_custom_priority_key(cls):
    # priority_key transforms priority into effective priority.
    # Using identity on priority where negative means lower first.
    pq = cls(priority_key=lambda p: p)
    pq.add("a", 3)
    pq.add("b", 1)
    pq.add("c", 2)
    assert pq.pop() == "b"
    assert pq.pop() == "c"
    assert pq.pop() == "a"


@pytest.mark.parametrize("cls", CONCRETE)
def test_custom_priority_key_none_handling(cls):
    # custom key receives raw priority (may be None) and must handle it
    pq = cls(priority_key=lambda p: p if p is not None else 0)
    pq.add("a")
    pq.add("b", 1)
    assert pq.pop() == "b"
    assert pq.pop() == "a"


@pytest.mark.parametrize("cls", CONCRETE)
def test_unexpected_kwargs_raises(cls):
    with pytest.raises(TypeError):
        cls(bogus_kwarg=1)


@pytest.mark.parametrize("cls", CONCRETE)
def test_len_reflects_entry_map(cls):
    pq = cls()
    assert len(pq) == 0
    pq.add("a", 1)
    assert len(pq) == 1
    pq.add("b", 2)
    assert len(pq) == 2
    pq.add("a", 3)
    assert len(pq) == 2
    pq.remove("b")
    assert len(pq) == 1
    pq.pop()
    assert len(pq) == 0


@pytest.mark.parametrize("cls", CONCRETE)
def test_unhashable_task_raises(cls):
    pq = cls()
    with pytest.raises(TypeError):
        pq.add(["unhashable"], 1)


@pytest.mark.parametrize("cls", CONCRETE)
def test_tasks_with_dicts_as_values(cls):
    # test that arbitrary hashable objects work as tasks
    pq = cls()
    t1 = frozenset([1])
    t2 = frozenset([2])
    pq.add(t1, 1)
    pq.add(t2, 2)
    assert pq.pop() == t2
    assert pq.pop() == t1


@pytest.mark.parametrize("cls", CONCRETE)
def test_priority_string_coerced_to_float(cls):
    # default priority key does float(p or 0); strings convertible to
    # numbers should work
    pq = cls()
    pq.add("a", "1")
    pq.add("b", "2")
    assert pq.pop() == "b"
    assert pq.pop() == "a"


@pytest.mark.parametrize("cls", CONCRETE)
def test_priority_zero_string_falsy(cls):
    # "" is falsy -> treated as 0
    pq = cls()
    pq.add("a", "")
    pq.add("b", 1)
    assert pq.pop() == "b"
    assert pq.pop() == "a"


@pytest.mark.parametrize("cls", CONCRETE)
def test_private_backend_attributes_exist(cls):
    pq = cls()
    assert hasattr(pq, "_pq")
    assert hasattr(pq, "_entry_map")
    assert isinstance(pq._entry_map, dict)
    assert len(pq._pq) == 0


# ---------------------------------------------------------------------------
# HeapPriorityQueue specifics
# ---------------------------------------------------------------------------

def test_heap_is_subclass_of_base():
    assert issubclass(HeapPriorityQueue, BasePriorityQueue)


def test_heap_backend_is_list():
    pq = HeapPriorityQueue()
    assert isinstance(pq._pq, list)


def test_heap_push_pop_entries_static():
    backend = []
    HeapPriorityQueue._push_entry(backend, [1, 0, "x"])
    HeapPriorityQueue._push_entry(backend, [0, 1, "y"])
    assert HeapPriorityQueue._pop_entry(backend) == [0, 1, "y"]
    assert HeapPriorityQueue._pop_entry(backend) == [1, 0, "x"]


# ---------------------------------------------------------------------------
# SortedPriorityQueue specifics
# ---------------------------------------------------------------------------

def test_sorted_is_subclass_of_base():
    assert issubclass(SortedPriorityQueue, BasePriorityQueue)


def test_sorted_push_pop_entries_static():
    backend = []
    SortedPriorityQueue._push_entry(backend, [1, 0, "x"])
    SortedPriorityQueue._push_entry(backend, [0, 1, "y"])
    assert SortedPriorityQueue._pop_entry(backend) == [0, 1, "y"]
    assert SortedPriorityQueue._pop_entry(backend) == [1, 0, "x"]


def test_priorityqueue_alias():
    assert PriorityQueue is SortedPriorityQueue


# ---------------------------------------------------------------------------
# BasePriorityQueue base behavior
# ---------------------------------------------------------------------------

def test_base_default_priority_key():
    assert BasePriorityQueue._default_priority_key(5) == -5.0
    assert BasePriorityQueue._default_priority_key(None) == 0.0
    assert BasePriorityQueue._default_priority_key(0) == 0.0


def test_base_backend_type_is_list():
    assert BasePriorityQueue._backend_type is list


def test_base_abstract_push_pop_do_nothing():
    # _push_entry/_pop_entry are abstract no-ops returning None
    backend = []
    assert BasePriorityQueue._push_entry(backend, [1, 0, "x"]) is None
    assert backend == []
    assert BasePriorityQueue._pop_entry(backend) is None


def test_base_len_without_operations():
    pq = BasePriorityQueue()
    assert len(pq) == 0


def test_base_custom_priority_key_stored():
    key = lambda p: p
    pq = BasePriorityQueue(priority_key=key)
    assert pq._get_priority is key


def test_base_remove_marks_removed():
    pq = BasePriorityQueue()
    pq.add("t", 1)
    assert len(pq) == 1
    pq.remove("t")
    assert len(pq) == 0
    # task marked removed in the underlying entry
    entries = list(pq._pq)
    assert len(entries) == 1
    assert entries[0][2] is not "t"


def test_base_add_returns_none():
    pq = BasePriorityQueue()
    assert pq.add("t", 1) is None


# ---------------------------------------------------------------------------
# Doctest-like scenario from module docstring
# ---------------------------------------------------------------------------

def test_module_docstring_scenario():
    pq = PriorityQueue()
    pq.add("low priority task", 0)
    pq.add("high priority task", 2)
    pq.add("medium priority task 1", 1)
    pq.add("medium priority task 2", 1)
    assert len(pq) == 4
    assert pq.pop() == "high priority task"
    assert pq.peek() == "medium priority task 1"
    assert len(pq) == 3
