import operator

import pytest

from toolz.itertoolz import (
    remove,
    accumulate,
    groupby,
    merge_sorted,
    interleave,
    unique,
    isiterable,
    isdistinct,
    take,
    drop,
    take_nth,
    first,
    second,
    nth,
    last,
    get,
    concat,
    concatv,
    mapcat,
    cons,
    interpose,
    frequencies,
    reduceby,
    iterate,
    sliding_window,
    partition,
    partition_all,
    count,
    pluck,
    join,
    tail,
    diff,
    topk,
    peek,
    peekn,
    random_sample,
    rest,
)


def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [])) == []


def test_accumulate():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]
    assert list(accumulate(operator.add, [])) == []


def test_groupby():
    names = ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"]
    assert groupby(len, names) == {
        3: ["Bob", "Dan"],
        5: ["Alice", "Edith", "Frank"],
        7: ["Charlie"],
    }

    assert groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }

    people = [
        {"name": "Alice", "gender": "F"},
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]
    assert groupby("gender", people) == {
        "F": [people[0]],
        "M": [people[1], people[2]],
    }

    pairs = [("a", 1), ("b", 2), ("a", 3)]
    assert groupby(0, pairs) == {
        "a": [("a", 1), ("a", 3)],
        "b": [("b", 2)],
    }

    assert groupby(len, []) == {}

    with pytest.raises(KeyError):
        groupby("missing", [{"a": 1}])


def test_merge_sorted():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]
    assert list(interleave([])) == []
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")
    assert list(unique([])) == []

    with pytest.raises(TypeError):
        list(unique([[1], [2]]))


def test_isiterable_and_isdistinct():
    assert isiterable([1, 2, 3]) is True
    assert isiterable("abc") is True
    assert isiterable(iter([1])) is True
    assert isiterable(5) is False

    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False

    with pytest.raises(TypeError):
        isdistinct([[1], [2]])


def test_take_drop_take_nth_tail_rest():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(10, [1, 2])) == [1, 2]
    assert list(take(0, [1, 2])) == []

    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []

    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]

    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)

    assert list(rest([1, 2, 3])) == [2, 3]

    with pytest.raises(ValueError):
        list(take_nth(0, [1, 2, 3]))


def test_first_second_nth_last():
    assert first("ABC") == "A"
    assert second("ABC") == "B"
    assert nth(1, "ABC") == "B"
    assert nth(2, iter([10, 20, 30, 40])) == 30
    assert nth(-1, [10, 20, 30]) == 30
    assert last("ABC") == "C"
    assert last(iter([1, 2, 3])) == 3

    with pytest.raises(StopIteration):
        first([])

    with pytest.raises(StopIteration):
        second([1])

    with pytest.raises(IndexError):
        nth(5, [1, 2])

    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))

    with pytest.raises(IndexError):
        last([])


def test_get():
    phonebook = {
        "Alice": "555-1234",
        "Bob": "555-5678",
        "Charlie": "555-9999",
    }

    assert get(1, "ABC") == "B"
    assert get([1, 2], "ABC") == ("B", "C")
    assert get([1], "ABC") == ("B",)
    assert get([], "ABC") == ()

    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)

    assert get("Dennis", phonebook, "unknown") == "unknown"
    assert get([0, 5], [10, 20], "missing") == (10, "missing")
    assert get("a", [1, 2], "fallback") == "fallback"
    assert get(1, 5, "fallback") == "fallback"

    with pytest.raises(KeyError):
        get("Dennis", phonebook)

    with pytest.raises(IndexError):
        get(5, [1, 2])

    with pytest.raises(TypeError):
        get(1, 5)


def test_concat_concatv_mapcat_cons_interpose():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]

    assert list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    ) == ["A", "B", "C", "D", "E"]

    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(0, iter([1, 2]))) == [0, 1, 2]

    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


def test_frequencies_reduceby():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }
    assert frequencies([]) == {}

    assert reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5]) == {
        False: 9,
        True: 6,
    }
    assert reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5]) == {
        False: 15,
        True: 8,
    }
    assert reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5], 0) == {
        False: 9,
        True: 6,
    }

    def set_add(s, i):
        s.add(i)
        return s

    assert reduceby(
        lambda x: x % 2 == 0,
        set_add,
        [1, 2, 3, 4, 1, 2, 3],
        set,
    ) == {False: {1, 3}, True: {2, 4}}

    projects = [
        {"name": "build roads", "state": "CA", "cost": 1000000},
        {"name": "fight crime", "state": "IL", "cost": 100000},
        {"name": "help farmers", "state": "IL", "cost": 2000000},
        {"name": "help farmers", "state": "CA", "cost": 200000},
    ]
    assert reduceby(
        "state", lambda acc, x: acc + x["cost"], projects, 0
    ) == {"CA": 1200000, "IL": 2100000}

    assert reduceby(lambda x: x, operator.add, [], 0) == {}


def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]

    powers_of_two = iterate(lambda x: x * 2, 1)
    assert [next(powers_of_two) for _ in range(4)] == [1, 2, 4, 8]


def test_sliding_window_and_partition():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
    ]
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]
    assert list(sliding_window(5, [1, 2, 3])) == []
    assert list(sliding_window(0, [1, 2, 3])) == []

    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]
    assert list(partition(3, [1, 2, 3, 4, 5], pad=0)) == [
        (1, 2, 3),
        (4, 5, 0),
    ]
    assert list(partition(2, [])) == []
    assert list(partition(0, [1, 2])) == []


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]

    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]

    assert list(partition_all(2, [])) == []
    assert list(partition_all(0, [1, 2])) == []


def test_count_and_pluck():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0

    data = [
        {"id": 1, "name": "Cheese"},
        {"id": 2, "name": "Pies"},
    ]

    assert list(pluck("name", data)) == ["Cheese", "Pies"]
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]
    assert list(pluck("missing", data, default="unknown")) == [
        "unknown",
        "unknown",
    ]
    assert list(pluck([0, 2], [[1, 2], [3, 4, 5]], default=None)) == [
        (1, None),
        (3, 5),
    ]

    with pytest.raises(KeyError):
        list(pluck("missing", data))


def test_join():
    left = [("a", 1), ("b", 2), ("a", 3), ("d", 4)]
    right = [("a", 10), ("c", 30), ("b", 20)]

    inner = [
        (("a", 1), ("a", 10)),
        (("a", 3), ("a", 10)),
        (("b", 2), ("b", 20)),
    ]

    assert list(join(0, left, 0, right)) == inner
    assert list(
        join(lambda x: x[0], left, lambda x: x[0], right)
    ) == inner

    assert list(join(0, left, 0, right, left_default="L")) == [
        (("a", 1), ("a", 10)),
        (("a", 3), ("a", 10)),
        ("L", ("c", 30)),
        (("b", 2), ("b", 20)),
    ]

    assert list(join(0, left, 0, right, right_default="R")) == [
        (("a", 1), ("a", 10)),
        (("a", 3), ("a", 10)),
        (("b", 2), ("b", 20)),
        (("d", 4), "R"),
    ]

    assert list(
        join(0, left, 0, right, left_default="L", right_default="R")
    ) == [
        (("a", 1), ("a", 10)),
        (("a", 3), ("a", 10)),
        ("L", ("c", 30)),
        (("b", 2), ("b", 20)),
        (("d", 4), "R"),
    ]

    assert list(
        join(
            lambda x: x,
            [1, 2, 3],
            lambda x: x,
            [2, 3, 4],
            left_default=None,
            right_default=None,
        )
    ) == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]
    assert list(
        diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)
    ) == [("bananas", "Oranges")]
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]
    assert list(diff([1, 2], [1, 2])) == []

    with pytest.raises(TypeError):
        list(diff())

    with pytest.raises(TypeError):
        list(diff([1]))


def test_topk_peek_peekn():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie",
        "Alice",
    )
    assert topk(2, [{"x": 1}, {"x": 3}, {"x": 2}], key="x") == (
        {"x": 3},
        {"x": 2},
    )
    assert topk(0, [1, 2, 3]) == ()
    assert topk(5, [1, 3, 2]) == (3, 2, 1)

    item, seq = peek([0, 1, 2])
    assert item == 0
    assert list(seq) == [0, 1, 2]

    with pytest.raises(StopIteration):
        peek([])

    first_two, seq = peekn(2, [0, 1, 2, 3])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3]

    no_items, seq = peekn(0, [0, 1])
    assert no_items == ()
    assert list(seq) == [0, 1]

    first_two, seq = peekn(5, [0, 1])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1]

    no_items, seq = peekn(2, [])
    assert no_items == ()
    assert list(seq) == []


def test_random_sample():
    class StubRandom:
        def __init__(self, values):
            self._values = iter(values)

        def random(self):
            return next(self._values)

    assert list(
        random_sample(
            0.5,
            [10, 20, 30, 40],
            random_state=StubRandom([0.0, 0.5, 0.499, 0.9]),
        )
    ) == [10, 30]

    assert list(
        random_sample(0.0, [1, 2], random_state=StubRandom([0.0, 0.5]))
    ) == []
    assert list(
        random_sample(1.0, [1, 2], random_state=StubRandom([0.0, 0.5]))
    ) == [1, 2]

    assert list(random_sample(0.1, range(100), random_state=2016)) == [
        7,
        9,
        19,
        25,
        30,
        32,
        34,
        48,
        59,
        60,
        81,
        98,
    ]
