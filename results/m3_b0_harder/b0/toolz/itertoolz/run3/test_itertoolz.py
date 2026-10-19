import itertools
from operator import add, mul

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
)


def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []
    assert list(remove(lambda x: False, range(3))) == [0, 1, 2]


def test_accumulate():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4])) == [1, 2, 6, 24]
    assert list(accumulate(add, [])) == []
    assert list(accumulate(add, [], 1)) == [1]
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(lambda a, b: a - b, [10, 2, 3])) == [10, 8, 5]


def test_groupby():
    assert groupby(lambda x: x % 2, [1, 2, 3, 4, 5, 6]) == {
        0: [2, 4, 6],
        1: [1, 3, 5],
    }
    assert groupby(lambda x: x, []) == {}

    data = [
        {"name": "Alice", "gender": "F"},
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]
    assert groupby("gender", data) == {
        "F": [data[0]],
        "M": [data[1], data[2]],
    }

    assert groupby(0, [(1, "a"), (2, "b"), (1, "c")]) == {
        1: [(1, "a"), (1, "c")],
        2: [(2, "b")],
    }


def test_merge_sorted():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([1, 2])) == [1, 2]
    assert list(merge_sorted([1, 3], [2, 4], [5, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([], [1, 2], [3])) == [1, 2, 3]
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"

    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]
    assert list(merge_sorted([], [1, 2], key=lambda x: x)) == [1, 2]


def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"
    assert list(interleave([])) == []
    assert list(interleave([[1, 2], [3]])) == [1, 3, 2]
    assert list(interleave([[1], [], [2]])) == [1, 2]
    assert list(take(6, interleave([itertools.count(), itertools.count(100)]))) == [
        0,
        100,
        1,
        101,
        2,
        102,
    ]


def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == (
        "cat",
        "mouse",
    )
    assert list(unique([])) == []
    assert list(unique([[1], [2], [1]], key=tuple)) == [[1], [2]]


def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable("abc") is True
    assert isiterable(iter([1])) is True
    assert isiterable(5) is False
    assert isiterable(None) is False


def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True
    assert isdistinct([]) is True
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_take_drop_take_nth():
    assert list(take(2, [10, 20, 30])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(5, [1, 2])) == [1, 2]

    assert list(drop(2, [10, 20, 30, 40])) == [30, 40]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []

    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]

    with pytest.raises(ValueError):
        take_nth(0, [1, 2])


def test_first_second_nth_last():
    assert first("ABC") == "A"
    assert first(iter([10, 20])) == 10
    with pytest.raises(StopIteration):
        first([])

    assert second("ABC") == "B"
    assert second(iter([10, 20, 30])) == 20
    with pytest.raises(StopIteration):
        second([1])

    assert nth(1, "ABC") == "B"
    assert nth(-1, [1, 2, 3]) == 3
    assert nth(2, iter([1, 2, 3, 4])) == 3
    with pytest.raises(IndexError):
        nth(10, [1, 2])
    with pytest.raises(StopIteration):
        nth(10, iter([1, 2]))
    with pytest.raises(ValueError):
        nth(-1, iter([1, 2]))

    assert last("ABC") == "C"
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3
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
    assert get("Dennis", phonebook, None) is None
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)
    assert get([0, 1, 2], "ab", default="x") == ("a", "b", "x")
    assert get(1, "abc", "z") == "b"
    assert get(5, "abc", "z") == "z"
    assert get(None, [1, 2], "d") == "d"

    with pytest.raises(IndexError):
        get(5, "abc")
    with pytest.raises(KeyError):
        get("Dennis", phonebook)
    with pytest.raises(TypeError):
        get(None, [1, 2])


def test_concat_concatv_mapcat_cons_interpose():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]
    assert list(concat([])) == []

    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]
    assert list(concatv()) == []

    assert list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c"]])
    ) == ["A", "B", "C"]

    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(take(3, cons(0, itertools.count(1)))) == [0, 1, 2]

    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


def test_frequencies():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }
    assert frequencies([]) == {}
    assert type(frequencies([])) is dict
    assert frequencies(iter("abca")) == {"a": 2, "b": 1, "c": 1}


def test_reduceby():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]

    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}
    assert reduceby(iseven, add, [], 0) == {}

    projects = [
        {"name": "build roads", "state": "CA", "cost": 1000000},
        {"name": "fight crime", "state": "IL", "cost": 100000},
        {"name": "help farmers", "state": "IL", "cost": 2000000},
        {"name": "help farmers", "state": "CA", "cost": 200000},
    ]
    assert reduceby(
        "state", lambda acc, x: acc + x["cost"], projects, 0
    ) == {"CA": 1200000, "IL": 2100000}

    def set_add(s, i):
        s.add(i)
        return s

    assert reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set) == {
        True: {2, 4},
        False: {1, 3},
    }

    assert reduceby(lambda x: x % 2, add, [1, 3, 5]) == {1: 9}


def test_iterate():
    assert list(take(5, iterate(lambda x: x + 1, 0))) == [0, 1, 2, 3, 4]
    assert list(take(4, iterate(lambda x: x * 2, 1))) == [1, 2, 4, 8]


def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
    ]
    assert list(sliding_window(2, [1])) == []
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]
    assert list(sliding_window(2, [])) == []
    assert list(sliding_window(0, [1, 2])) == []


def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]
    assert list(partition(3, [1, 2], pad=0)) == [(1, 2, 0)]
    assert list(partition(0, [1, 2])) == []


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == [
        (1, 2, 3),
        (4, 5, 6),
        (7,),
    ]
    assert list(partition_all(1, [1, 2, 3])) == [(1,), (2,), (3,)]
    assert list(partition_all(2, [])) == []
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == [
        (1, 2, 3),
        (4, 5),
    ]

    class BadLen(list):
        def __len__(self):
            return 4

    with pytest.raises(LookupError, match="invalid length"):
        list(partition_all(2, BadLen([1, 2, 3])))


def test_count():
    assert count([1, 2, 3]) == 3
    assert count((1,)) == 1
    assert count(iter([1, 2, 3])) == 3
    assert count(x for x in range(5)) == 5
    assert count([]) == 0


def test_pluck():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]

    assert list(pluck("name", data)) == ["Cheese", "Pies"]
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]
    assert list(pluck([], data)) == [(), ()]

    assert list(pluck("name", data, default="missing")) == ["Cheese", "Pies"]
    assert list(pluck("x", data, default="missing")) == ["missing", "missing"]
    assert list(
        pluck(
            ["name", "x"],
            [{"name": "A"}, {"name": "B", "x": 2}],
            default=None,
        )
    ) == [("A", None), ("B", 2)]

    with pytest.raises(KeyError):
        list(pluck("x", [{}]))
    with pytest.raises(IndexError):
        list(pluck(2, [[1]]))


def test_join():
    identity = lambda x: x

    assert list(join(identity, [1, 2, 3], identity, [2, 3, 4])) == [
        (2, 2),
        (3, 3),
    ]
    assert list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], left_default=None)
    ) == [(2, 2), (3, 3), (None, 4)]
    assert list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], right_default=None)
    ) == [(2, 2), (3, 3), (1, None)]
    assert list(
        join(
            identity,
            [1, 2, 3],
            identity,
            [2, 3, 4],
            left_default=None,
            right_default=None,
        )
    ) == [(2, 2), (3, 3), (None, 4), (1, None)]

    assert list(
        join(0, [(1, "a"), (2, "b")], 0, [(2, "c"), (1, "d")])
    ) == [((2, "b"), (2, "c")), ((1, "a"), (1, "d"))]

    left = [("a", 1), ("a", 2), ("b", 3)]
    right = [("a", "x"), ("c", "y")]
    assert list(join(0, left, 0, right)) == [
        (("a", 1), ("a", "x")),
        (("a", 2), ("a", "x")),
    ]

    assert list(join(identity, [], identity, [1, 2], right_default=None)) == []
    assert list(join(identity, [], identity, [1], left_default="L")) == [
        ("L", 1)
    ]


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (1, 2, 3)) == (2, 3)
    assert tail(2, "abcd") == "cd"
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)
    assert tail(5, [1, 2]) == [1, 2]


def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]
    assert list(
        diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)
    ) == [("bananas", "Oranges")]
    assert list(diff([1, 2, 3], [1, 2, 4], [1, 2, 3])) == [(3, 4, 3)]
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]

    with pytest.raises(TypeError, match="Too few"):
        list(diff((1, 2, 3)))
    with pytest.raises(TypeError, match="Too few"):
        list(diff())


def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie",
        "Alice",
    )
    assert topk(
        2,
        [{"v": 1}, {"v": 3}, {"v": 2}],
        key="v",
    ) == ({"v": 3}, {"v": 2})
    assert topk(0, [1, 2, 3]) == ()
    assert topk(5, [1, 2]) == (2, 1)
    assert topk(2, []) == ()


def test_peek():
    item, restored = peek([0, 1, 2])
    assert item == 0
    assert list(restored) == [0, 1, 2]

    iterator = iter([5, 6])
    item, restored = peek(iterator)
    assert item == 5
    assert list(restored) == [5, 6]
    assert list(iterator) == []

    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    peeked, restored = peekn(2, [0, 1, 2, 3, 4])
    assert peeked == (0, 1)
    assert list(restored) == [0, 1, 2, 3, 4]

    peeked, restored = peekn(2, [])
    assert peeked == ()
    assert list(restored) == []

    peeked, restored = peekn(0, iter([1, 2]))
    assert peeked == ()
    assert list(restored) == [1, 2]

    peeked, restored = peekn(5, [1, 2])
    assert peeked == (1, 2)
    assert list(restored) == [1, 2]


def test_random_sample():
    class FixedRandom:
        def __init__(self, values):
            self._values = iter(values)

        def random(self):
            return next(self._values)

    assert list(
        random_sample(
            0.5,
            [1, 2, 3, 4],
            random_state=FixedRandom([0.1, 0.6, 0.5, 0.9]),
        )
    ) == [1]

    assert list(
        random_sample(
            0.0,
            [1, 2, 3],
            random_state=FixedRandom([0.0, 0.0, 0.0]),
        )
    ) == []

    assert list(
        random_sample(
            1.0,
            [1, 2, 3],
            random_state=FixedRandom([0.0, 0.999, 0.5]),
        )
    ) == [1, 2, 3]

    # Integer seed branch; with probability 0, no random values can pass.
    assert list(random_sample(0.0, [1, 2, 3], random_state=2016)) == []

    # random_sample is lazy and consumes random values as items are requested.
    sample = random_sample(
        0.5,
        [10, 20, 30],
        random_state=FixedRandom([0.9, 0.1, 0.0]),
    )
    assert next(sample) == 20
    assert next(sample) == 30
    with pytest.raises(StopIteration):
        next(sample)
