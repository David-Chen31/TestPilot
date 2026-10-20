import itertools
import operator
import random
from functools import partial
from collections import deque

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
    tail,
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
    getter,
    join,
    diff,
    topk,
    peek,
    peekn,
    random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_lazy():
    it = remove(lambda x: x > 1, [1, 2, 3])
    assert next(it) == 1
    with pytest.raises(StopIteration):
        next(it)


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_no_initial():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_generator():
    result = accumulate(operator.add, [1, 2, 3])
    assert iter(result) is result


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    res = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert res == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_indexing():
    data = [
        {"name": "Alice", "gender": "F"},
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]
    res = groupby("gender", data)
    assert res == {
        "F": [{"name": "Alice", "gender": "F"}],
        "M": [
            {"name": "Bob", "gender": "M"},
            {"name": "Charlie", "gender": "M"},
        ],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_returns_dict():
    res = groupby(lambda x: x, [1, 1, 2])
    assert isinstance(res, dict)
    assert res == {1: [1, 1], 2: [2]}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    res = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert res == [2, 1, 3, 3]


def test_merge_sorted_key_three():
    res = list(merge_sorted([2, 3], [1, 3], [0, 4], key=lambda x: x // 3))
    assert res == [2, 1, 0, 3, 3, 4]


def test_merge_sorted_one_empty_two():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_with_key_one_empty():
    assert list(merge_sorted([], [1, 2], key=lambda x: x)) == [1, 2]


def test_merge_sorted_equal_elements():
    assert list(merge_sorted([1, 2, 2], [2, 3])) == [1, 2, 2, 2, 3]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_with_empty_sex():
    assert list(interleave([[], [1, 2], [3]])) == [1, 3, 2]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)


def test_unique_dupes():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable_list():
    assert isiterable([1, 2, 3]) is True


def test_isiterable_string():
    assert isiterable("abc") is True


def test_isiterable_int():
    assert isiterable(5) is False


def test_isiterable_generator():
    assert isiterable(iter([])) is True


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_on_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ---------------------------------------------------------------------------
# take / drop / tail / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_len():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_lazy():
    it = take(2, iter([1, 2, 3]))
    assert list(it) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_step_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_len():
    assert tail(10, [1, 2]) == [1, 2]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first("ABC") == "A"


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"


def test_nth_list():
    assert nth(1, "ABC") == "B"
    assert nth(2, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(1, iter("ABC")) == "B"


def test_nth_negative_list():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last("ABC") == "C"
    assert last([1, 2, 3, 4]) == 4


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_single():
    assert get(1, "ABC") == "B"


def test_get_multiple():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_dict():
    phonebook = {
        "Alice": "555-1234",
        "Bob": "555-5678",
        "Charlie": "555-9999",
    }
    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_multiple_with_default():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get("Dennis", {"Alice": 1})


def test_get_missing_index_default():
    assert get(10, [1, 2], None) is None


def test_get_empty_list():
    assert get([], "ABC") == ()


def test_get_single_list_ind():
    assert get([1], "ABC") == ("B",)


def test_get_type_error_with_default():
    # get on a non-indexable with default returns default
    assert get("x", 5, default="d") == "d"


def test_get_type_error_no_default_raises():
    with pytest.raises(TypeError):
        get("x", 5)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_of_seqs():
    assert list(concat(iter([iter([1]), iter([2])]))) == [1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_mapcat():
    result = list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    )
    assert result == ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_on_generator():
    assert list(cons(0, iter([1, 2]))) == [0, 1, 2]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------
def test_frequencies():
    res = frequencies(["cat", "cat", "ox", "pig", "pig", "cat"])
    assert res == {"cat": 3, "ox": 1, "pig": 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


def test_frequencies_returns_plain_dict():
    res = frequencies([1, 1, 2])
    assert type(res) is dict


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.mul, [1, 2, 3, 4, 5]) == {False: 15, True: 8}


def test_reduceby_with_init_constant():
    projects = [
        {"name": "build roads", "state": "CA", "cost": 1000000},
        {"name": "fight crime", "state": "IL", "cost": 100000},
        {"name": "help farmers", "state": "IL", "cost": 2000000},
        {"name": "help farmers", "state": "CA", "cost": 200000},
    ]
    res = reduceby("state", lambda acc, x: acc + x["cost"], projects, 0)
    assert res == {"CA": 1200000, "IL": 2100000}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    res = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert res == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


def test_reduceby_indexing_key():
    data = [{"k": "a", "v": 1}, {"k": "a", "v": 2}, {"k": "b", "v": 3}]
    res = reduceby("k", lambda acc, x: acc + x["v"], data, 0)
    assert res == {"a": 3, "b": 3}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


def test_iterate_take():
    assert list(take(5, iterate(lambda x: x + 1, 10))) == [10, 11, 12, 13, 14]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    result = list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    assert result == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------
def test_partition_exact():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]


def test_partition_n_larger_than_seq_no_pad():
    assert list(partition(5, [1, 2])) == []


def test_partition_n_larger_than_seq_with_pad():
    assert list(partition(5, [1, 2], pad=0)) == [(1, 2, 0, 0, 0)]


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------
def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter(range(7)))) == [
        (0, 1, 2),
        (3, 4, 5),
        (6,),
    ]


def test_partition_all_len_sequence():
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_n_one():
    assert list(partition_all(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_generator():
    assert count(x for x in range(5)) == 5


def test_count_empty():
    assert count([]) == 0


def test_count_string():
    assert count("abc") == 3


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{"id": 1}, {"name": "x"}]
    assert list(pluck("name", data, default="missing")) == ["missing", "x"]


def test_pluck_with_list_and_default():
    data = [[1, 2], [3]]
    res = list(pluck([0, 1], data, default=None))
    assert res == [(1, 2), (3, None)]


def test_getter_single():
    g = getter(1)
    assert g("ABC") == "B"


def test_getter_single_list():
    g = getter([1])
    assert g("ABC") == ("B",)


def test_getter_multiple():
    g = getter([0, 2])
    assert g("ABC") == ("A", "C")


def test_getter_empty():
    g = getter([])
    assert g("ABC") == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    friends = [
        ("Alice", "Edith"),
        ("Alice", "Zhao"),
        ("Edith", "Alice"),
        ("Zhao", "Alice"),
        ("Zhao", "Edith"),
    ]
    cities = [
        ("Alice", "NYC"),
        ("Alice", "Chicago"),
        ("Dan", "Sydney"),
        ("Edith", "Paris"),
        ("Edith", "Berlin"),
        ("Zhao", "Shanghai"),
    ]
    result = list(join(second, friends, first, cities))
    result_set = set(result)
    expected = {
        (("Alice", "Edith"), ("Edith", "Paris")),
        (("Alice", "Edith"), ("Edith", "Berlin")),
        (("Alice", "Zhao"), ("Zhao", "Shanghai")),
        (("Edith", "Alice"), ("Alice", "NYC")),
        (("Edith", "Alice"), ("Alice", "Chicago")),
        (("Zhao", "Alice"), ("Alice", "NYC")),
        (("Zhao", "Alice"), ("Alice", "Chicago")),
        (("Zhao", "Edith"), ("Edith", "Paris")),
        (("Zhao", "Edith"), ("Edith", "Berlin")),
    }
    assert result_set == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4],
             left_default=None, right_default=None)
    )
    assert set(result) == {(2, 2), (None, 4), (1, None), (3, 3)}


def test_join_right_default():
    identity = lambda x: x
    result = set(join(identity, [1, 2], identity, [2, 3]))
    assert result == {(2, 2)}


def test_join_left_default():
    identity = lambda x: x
    result = set(
        join(identity, [1, 2], identity, [2, 3], left_default=None)
    )
    assert result == {(2, 2), (None, 3)}


def test_join_left_only_default():
    identity = lambda x: x
    result = set(
        join(identity, [1, 2], identity, [2, 3], right_default=None)
    )
    assert result == {(2, 2), (1, None)}


def test_join_index_keys():
    friends = [("Alice", "Edith"), ("Alice", "Zhao")]
    cities = [("Alice", "NYC"), ("Zhao", "Shanghai")]
    result = set(join(1, friends, 0, cities))
    assert result == {
        (("Alice", "Edith"), ("Alice", "NYC")),
        (("Alice", "Zhao"), ("Zhao", "Shanghai")),
    }


def test_join_empty_left():
    result = list(join(first, [], first, [("a", 1)]))
    assert result == []


def test_join_empty_right():
    result = list(join(first, [("a", 1)], first, []))
    assert result == []


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    res = list(diff([1, 2, 3], [1, 2, 10, 100], default=None))
    assert res == [(3, 10), (None, 100)]


def test_diff_key():
    res = list(diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower))
    assert res == [("bananas", "Oranges")]


def test_diff_single_list_of_seqs():
    res = list(diff([[1, 2, 3], [1, 2, 10]]))
    assert res == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    res = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
    assert res == [(3, 3, 4)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie",
        "Alice",
    )


def test_topk_key_index():
    data = [{"v": 1}, {"v": 100}, {"v": 10}]
    assert topk(2, data, key="v") == ({"v": 100}, {"v": 10})


def test_topk_k_larger_than_seq():
    assert topk(10, [3, 1, 2]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_elem, rest = peek(seq)
    assert first_elem == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_on_iterator():
    seq = iter([10, 20, 30])
    elem, rest = peek(seq)
    assert elem == 10
    assert list(rest) == [10, 20, 30]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_empty():
    first_elems, rest = peekn(2, [])
    assert first_elems == ()
    assert list(rest) == []


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
class _AlwaysRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_random_sample_prob_zero():
    data = list(range(100))
    res = list(random_sample(0.0, data, random_state=42))
    assert res == []


def test_random_sample_prob_one():
    data = list(range(10))
    res = list(random_sample(1.0, data, random_state=42))
    assert res == data


def test_random_sample_random_object():
    data = list(range(10))
    res = list(random_sample(0.5, data, random_state=_AlwaysRandom(0.3)))
    assert res == data


def test_random_sample_random_object_zero():
    data = list(range(10))
    res = list(random_sample(0.5, data, random_state=_AlwaysRandom(0.9)))
    assert res == []


def test_random_sample_deterministic_with_seed():
    data = list(range(100))
    res1 = list(random_sample(0.1, data, random_state=2016))
    res2 = list(random_sample(0.1, data, random_state=2016))
    assert res1 == res2
    assert res1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_lazy():
    data = range(1000000)
    it = random_sample(0.0, data, random_state=0)
    assert list(it) == []
