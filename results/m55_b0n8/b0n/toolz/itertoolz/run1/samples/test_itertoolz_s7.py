import itertools
import operator
from functools import reduce

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


# remove ----------------------------------------------------------------------

def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_all():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_none():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# accumulate ------------------------------------------------------------------

def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_initial_empty():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


# groupby ---------------------------------------------------------------------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_key_not_callable():
    data = [
        {"name": "Alice", "gender": "F"},
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]
    result = groupby("gender", data)
    assert result["F"] == [{"name": "Alice", "gender": "F"}]
    assert result["M"] == [
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]


def test_groupby_empty():
    assert groupby(len, []) == {}


# merge_sorted ----------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_single_seq():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_seq():
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


def test_merge_sorted_key_empty():
    assert list(merge_sorted([1, 2], [], key=lambda x: x)) == [1, 2]


# interleave ------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty():
    assert list(interleave([])) == []


# unique ----------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_unique_empty():
    assert tuple(unique([])) == ()


# isiterable ------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable("abc") is True
    assert isiterable(5) is False


# isdistinct ------------------------------------------------------------------

def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_string():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# take / drop / tail / take_nth ----------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterable():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_longer_than_seq():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]
    assert tail(10, iter([1, 2, 3])) == (1, 2, 3)


# first / second / nth / last -------------------------------------------------

def test_first():
    assert first("ABC") == "A"


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"


def test_second_empty():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_list():
    assert nth(1, "ABC") == "B"
    assert nth(0, [10, 20, 30]) == 10


def test_nth_iterator():
    assert nth(1, iter("ABC")) == "B"


def test_nth_out_of_range_list():
    with pytest.raises(IndexError):
        nth(10, [1, 2, 3])


def test_last():
    assert last("ABC") == "C"


def test_last_list():
    assert last([1, 2, 3]) == 3


# get -------------------------------------------------------------------------

def test_get_single():
    assert get(1, "ABC") == "B"


def test_get_list_of_indices():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_dict():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678", "Charlie": "555-9999"}
    assert get("Alice", phonebook) == "555-1234"


def test_get_dict_list():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678"}
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_default():
    phonebook = {"Alice": "555-1234"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)


def test_get_default_single_missing():
    assert get("x", {}, "default") == "default"


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get("x", {})


def test_get_empty_list_indices():
    assert get([], "ABC") == ()


def test_get_single_list_no_default():
    assert get([1], "ABC") == ("B",)


# concat / concatv / mapcat ---------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]]))
    assert result == ["A", "B", "C", "D", "E"]


# cons / interpose ------------------------------------------------------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# frequencies -----------------------------------------------------------------

def test_frequencies():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }


def test_frequencies_empty():
    assert frequencies([]) == {}


# reduceby --------------------------------------------------------------------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}


def test_reduceby_no_init_mul():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_constant():
    projects = [
        {"name": "build roads", "state": "CA", "cost": 1000000},
        {"name": "fight crime", "state": "IL", "cost": 100000},
        {"name": "help farmers", "state": "IL", "cost": 2000000},
        {"name": "help farmers", "state": "CA", "cost": 200000},
    ]
    result = reduceby("state", lambda acc, x: acc + x["cost"], projects, 0)
    assert result == {"CA": 1200000, "IL": 2100000}


def test_reduceby_init_callable():

    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# iterate ---------------------------------------------------------------------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    gen = iterate(lambda x: x * 2, 1)
    assert [next(gen) for _ in range(4)] == [1, 2, 4, 8]


# sliding_window --------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_shorter_than_n():
    assert list(sliding_window(3, [1, 2])) == []


# partition / partition_all ---------------------------------------------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]


# count -----------------------------------------------------------------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# pluck / getter --------------------------------------------------------------

def test_pluck_dict():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_multiple_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{"name": "Cheese"}, {"name": "Pies"}]
    result = list(pluck("id", data, default=None))
    assert result == [None, None]


def test_pluck_default_list():
    data = [[1, 2], [3]]
    result = list(pluck([0, 1], data, default=None))
    assert result == [(1, 2), (3, None)]


def test_getter_single():
    g = getter(1)
    assert g("ABC") == "B"


def test_getter_list():
    g = getter([0, 2])
    assert g("ABC") == ("A", "C")


def test_getter_list_one():
    g = getter([1])
    assert g("ABC") == ("B",)


def test_getter_empty_list():
    g = getter([])
    assert g("ABC") == ()


# join ------------------------------------------------------------------------

def test_join_inner():
    friends = [
        ("Alice", "Edith"),
        ("Alice", "Zhao"),
        ("Edith", "Alice"),
    ]
    cities = [
        ("Alice", "NYC"),
        ("Edith", "Paris"),
    ]
    result = list(join(second, friends, first, cities))
    assert result == [
        (("Edith", "Alice"), ("Alice", "NYC")),
        (("Alice", "Edith"), ("Edith", "Paris")),
        (("Edith", "Alice"), ("Edith", "Paris")),
    ]


def test_join_right_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], left_default=None)
    )
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_full_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4],
             left_default=None, right_default=None)
    )
    assert sorted(result, key=lambda t: (t[0] is None, str(t[0]), t[1])) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)],
        key=lambda t: (t[0] is None, str(t[0]), t[1]),
    )


def test_join_left_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], right_default=None)
    )
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_index_keys():
    friends = [("Alice", "Edith"), ("Alice", "Zhao")]
    cities = [("Alice", "NYC"), ("Edith", "Paris")]
    result = list(join(1, friends, 0, cities))
    assert result == [
        (("Alice", "Edith"), ("Alice", "NYC")),
        (("Alice", "Zhao"), ("Alice", "NYC")),
        (("Alice", "Edith"), ("Edith", "Paris")),
    ]


# diff ------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)) == [
        ("bananas", "Oranges")
    ]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    assert list(diff([1, 2], [1, 3], [1, 4])) == [(2, 3, 4)]


# topk ------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == ("Charlie", "Alice")


def test_topk_key_not_callable():
    data = [{"v": 1}, {"v": 10}, {"v": 5}]
    assert topk(2, data, key="v") == ({"v": 10}, {"v": 5})


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# peek / peekn ----------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest_seq = peek(seq)
    assert item == 0
    assert list(rest_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    seq = iter([0, 1, 2])
    item, rest_seq = peek(seq)
    assert item == 0
    assert list(rest_seq) == [0, 1, 2]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest_seq = peekn(10, [0, 1])
    assert peeked == (0, 1)
    assert list(rest_seq) == [0, 1]


# random_sample ---------------------------------------------------------------

def test_random_sample_prob_one():
    seq = list(range(100))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(100))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_deterministic():
    seq = list(range(100))
    first = list(random_sample(0.1, seq, random_state=2016))
    second = list(random_sample(0.1, seq, random_state=2016))
    assert first == second
    assert first == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    from random import Random

    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
