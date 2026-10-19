"""Unit tests for toolz.itertoolz."""
import itertools
import operator

import pytest

from toolz.itertoolz import (
    accumulate,
    concat,
    concatv,
    cons,
    count,
    diff,
    drop,
    first,
    frequencies,
    get,
    groupby,
    interleave,
    interpose,
    isdistinct,
    isiterable,
    iterate,
    join,
    last,
    mapcat,
    merge_sorted,
    nth,
    partition,
    partition_all,
    peek,
    peekn,
    pluck,
    random_sample,
    reduceby,
    remove,
    rest,
    second,
    sliding_window,
    tail,
    take,
    take_nth,
    topk,
    unique,
)


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_single():
    assert list(accumulate(operator.add, [7])) == [7]


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert result == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_string_key():
    data = [
        {"g": "F", "n": "Alice"},
        {"g": "M", "n": "Bob"},
        {"g": "F", "n": "Eve"},
    ]
    assert groupby("g", data) == {
        "F": [data[0], data[2]],
        "M": [data[1]],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_list_key():
    data = [(1, "a"), (1, "b"), (2, "c")]
    # a single-element list is wrapped into a 1-tuple
    assert groupby([0], data) == {
        (1,): [(1, "a"), (1, "b")],
        (2,): [(2, "c")],
    }


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    # A single sequence is returned unchanged (as an iterator)
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_three():
    assert list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9])) == list(range(1, 10))


def test_merge_sorted_generators():
    assert list(merge_sorted(iter([1, 3]), iter([2, 4]))) == [1, 2, 3, 4]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_three_with_key():
    result = list(merge_sorted([3, 6, 9], [2, 5, 8], [1, 4, 7], key=lambda x: x))
    assert result == list(range(1, 10))


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_single_sequence():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_empty():
    assert list(interleave([])) == []


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_unique_generator():
    assert tuple(unique(iter([1, 2, 1, 3, 2]))) == (1, 2, 3)


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable / isdistinct
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [([1, 2], True), ("abc", True), ((1,), True), (5, False), (None, False)],
)
def test_isiterable(value, expected):
    assert isiterable(value) is expected


def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_string():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_tuple():
    assert isdistinct((1, 2, 3)) is True
    assert isdistinct((1, 2, 1)) is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth / rest
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40])) == [10, 20]
    assert list(take(10, [1, 2])) == [1, 2]
    assert list(take(0, [1, 2])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40])) == [30, 40]
    assert list(drop(10, [1, 2])) == []


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first("ABC") == "A"
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, "ABC") == "B"
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30
    assert nth(-1, (10, 20, 30)) == 30
    assert nth(2, iter([10, 20, 30])) == 30


def test_last():
    assert last("ABC") == "C"
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_generator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_len():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]
    assert tail(10, iter([1, 2, 3])) == (1, 2, 3)


def test_tail_dict():
    assert tail(2, {"a": 1, "b": 2, "c": 3}) == ("b", "c")


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_index():
    assert get(1, "ABC") == "B"
    assert get(1, [10, 20, 30]) == 20
    assert get(-1, [10, 20, 30]) == 30
    assert get("a", {"a": 1}) == 1


def test_get_multiple():
    assert get([1, 2], "ABC") == ("B", "C")
    assert get([0, 2], [10, 20, 30]) == (10, 30)


def test_get_single_element_list():
    assert get([1], "ABC") == ("B",)


def test_get_empty_list():
    assert get([], "ABC") == ()
    assert get([], "ABC", "X") == ()


def test_get_default():
    assert get(5, "ABC", "X") == "X"
    assert get("missing", {"a": 1}, "X") == "X"
    assert get([1, 5], "ABC", "X") == ("B", "X")
    assert get(["Alice", "Dennis"], {"Alice": "555"}, None) == ("555", None)


def test_get_missing_no_default():
    with pytest.raises(IndexError):
        get(5, "ABC")
    with pytest.raises(KeyError):
        get("missing", {"a": 1})
    with pytest.raises(KeyError):
        get(["a", "b"], {"a": 1})


def test_get_typeerror_default():
    assert get(5, 5, "X") == "X"


def test_get_typeerror_no_default():
    with pytest.raises(TypeError):
        get(5, 5)


def test_get_list_of_indices_on_non_indexable():
    # ``_get`` only swallows KeyError/IndexError, not TypeError, so the
    # TypeError from indexing an int propagates.
    with pytest.raises(TypeError):
        get([0], 5, "X")


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator_of_sequences():
    gen = (x for x in [[1, 2], [3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]
    assert list(concatv()) == []


def test_mapcat():
    assert list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    ) == ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


# ---------------------------------------------------------------------------
# frequencies / reduceby
# ---------------------------------------------------------------------------
def test_frequencies():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }
    assert frequencies([]) == {}


def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_with_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5], 0)
    assert result == {False: 9, True: 6}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_string_key():
    data = [
        {"s": "CA", "c": 10},
        {"s": "IL", "c": 5},
        {"s": "CA", "c": 3},
    ]
    result = reduceby("s", lambda acc, x: acc + x["c"], data, 0)
    assert result == {"CA": 13, "IL": 5}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate / sliding_window
# ---------------------------------------------------------------------------
def test_iterate():
    assert list(itertools.islice(iterate(lambda x: x * 2, 1), 5)) == [1, 2, 4, 8, 16]


def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
    ]


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


def test_sliding_window_generator():
    assert list(sliding_window(2, iter([1, 2, 3]))) == [(1, 2), (2, 3)]


def test_sliding_window_too_large():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == [(1, 2, 3), (4, 0, 0)]


def test_partition_larger_than_seq():
    assert list(partition(5, [1, 2, 3])) == []
    assert list(partition(5, [1, 2, 3], pad=None)) == [(1, 2, 3, None, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_generator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [(1, 2), (3, 4), (5,)]


def test_partition_all_partial_generator():
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]


def test_partition_all_short():
    assert list(partition_all(3, [1, 2])) == [(1, 2)]
    assert list(partition_all(3, iter([1, 2]))) == [(1, 2)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count():
    assert count([1, 2, 3]) == 3
    assert count((1, 2)) == 2
    assert count("abc") == 3
    assert count([]) == 0


def test_count_lazy():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(x for x in range(5)) == 5


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------
def test_pluck_scalar():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{"a": 1}, {"b": 2}]
    assert list(pluck("a", data, default=None)) == [1, None]
    assert list(pluck(["a", "b"], data, default=None)) == [(1, None), (None, 2)]


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    left = [("a", 1), ("b", 2)]
    right = [("a", "x"), ("c", "y")]
    result = list(join(first, left, first, right))
    assert result == [(("a", 1), ("a", "x"))]


def test_join_inner_duplicate_keys():
    left = [("a", 1), ("a", 2)]
    right = [("a", "x")]
    assert list(join(first, left, first, right)) == [
        (("a", 1), ("a", "x")),
        (("a", 2), ("a", "x")),
    ]


def test_join_right_outer():
    left = [("a", 1), ("b", 2)]
    right = [("a", "x"), ("c", "y")]
    result = list(join(first, left, first, right, left_default=None))
    assert result == [
        (("a", 1), ("a", "x")),
        (None, ("c", "y")),
    ]


def test_join_left_outer():
    left = [("a", 1), ("b", 2)]
    right = [("a", "x"), ("c", "y")]
    result = list(join(first, left, first, right, right_default=None))
    assert result == [
        (("a", 1), ("a", "x")),
        (("b", 2), None),
    ]


def test_join_full_outer():
    left = [("a", 1), ("b", 2)]
    right = [("a", "x"), ("c", "y")]
    result = list(
        join(first, left, first, right, left_default=None, right_default=None)
    )
    assert result == [
        (("a", 1), ("a", "x")),
        (None, ("c", "y")),
        (("b", 2), None),
    ]


def test_join_non_callable_keys():
    left = [("a", 1), ("b", 2)]
    right = [("a", "x"), ("c", "y")]
    result = list(join(0, left, 0, right))
    assert result == [(("a", 1), ("a", "x"))]


def test_join_identity_example():
    identity = lambda x: x
    result = list(
        join(
            identity,
            [1, 2, 3],
            identity,
            [2, 3, 4],
            left_default=None,
            right_default=None,
        )
    )
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]


def test_diff_with_key():
    assert list(
        diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)
    ) == [("bananas", "Oranges")]


def test_diff_three_sequences():
    assert list(diff([1, 2, 3], [1, 5, 3], [1, 7, 3])) == [(2, 5, 7)]


def test_diff_single_list_of_sequences():
    assert list(diff([[1, 2], [3, 4], [5, 6]])) == [(1, 3, 5), (2, 4, 6)]


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_too_few_sequences():
    with pytest.raises(TypeError):
        list(diff([[1, 2]]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie",
        "Alice",
    )


def test_topk_with_non_callable_key():
    data = [{"x": 3}, {"x": 1}, {"x": 2}]
    assert topk(2, data, key="x") == ({"x": 3}, {"x": 2})


def test_topk_k_larger_than_sequence():
    assert topk(10, [3, 1, 2]) == (3, 2, 1)


def test_topk_empty():
    assert topk(2, []) == ()


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    item, seq = peek([0, 1, 2, 3])
    assert item == 0
    assert list(seq) == [0, 1, 2, 3]


def test_peek_generator():
    item, seq = peek(iter([10, 20, 30]))
    assert item == 10
    assert list(seq) == [10, 20, 30]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    items, seq = peekn(2, [0, 1, 2, 3])
    assert items == (0, 1)
    assert list(seq) == [0, 1, 2, 3]


def test_peekn_larger_than_sequence():
    items, seq = peekn(5, [0, 1, 2])
    assert items == (0, 1, 2)
    assert list(seq) == [0, 1, 2]


def test_peekn_zero():
    items, seq = peekn(0, [1, 2])
    assert items == ()
    assert list(seq) == [1, 2]


def test_peekn_empty():
    items, seq = peekn(3, [])
    assert items == ()
    assert list(seq) == []


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_prob_zero():
    assert list(random_sample(0, range(100), random_state=42)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1, range(10), random_state=42)) == list(range(10))


def test_random_sample_deterministic_seed():
    a = list(random_sample(0.5, range(20), random_state=42))
    b = list(random_sample(0.5, range(20), random_state=42))
    assert a == b


def test_random_sample_custom_state():
    class FixedRandom:
        def __init__(self, values):
            self._it = iter(values)

        def random(self):
            return next(self._it)

    r = FixedRandom([0.1, 0.9, 0.3, 0.7])
    assert list(random_sample(0.5, [1, 2, 3, 4], random_state=r)) == [1, 3]
