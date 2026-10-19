import itertools
import operator
import random

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
def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_returns_iterator():
    it = remove(lambda x: x > 1, [1, 2, 3])
    assert not isinstance(it, list)
    assert iter(it) is it


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    result = groupby(len, ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"])
    assert result == {3: ["Bob", "Dan"], 5: ["Alice", "Edith", "Frank"],
                      7: ["Charlie"]}


def test_groupby_iseven():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }


def test_groupby_non_callable():
    data = [{"name": "Alice", "gender": "F"},
            {"name": "Bob", "gender": "M"},
            {"name": "Charlie", "gender": "M"}]
    result = groupby("gender", data)
    assert result == {
        "F": [{"name": "Alice", "gender": "F"}],
        "M": [{"name": "Bob", "gender": "M"},
              {"name": "Charlie", "gender": "M"}],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_unbalanced():
    assert list(merge_sorted([1], [2, 3, 4, 5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([1, 2, 3, 4, 5], [])) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty():
    assert list(interleave([])) == []


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == (
        "cat", "mouse")


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [[1, 2, 3], "abc", (1,), {1: 2}, iter([])])
def test_isiterable_true(value):
    assert isiterable(value) is True


@pytest.mark.parametrize("value", [5, None, 3.14])
def test_isiterable_false(value):
    assert isiterable(value) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct("World") is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, "abcde") == "de"


def test_tail_iterator():
    result = tail(2, iter([10, 20, 30, 40, 50]))
    assert result == (40, 50)


def test_tail_more_than_length():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first("ABC") == "A"
    assert first([10, 20]) == 10


def test_second():
    assert second("ABC") == "B"


def test_nth_sequence():
    assert nth(1, "ABC") == "B"
    assert nth(2, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_last():
    assert last("ABC") == "C"
    assert last([1, 2, 3]) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_single_index():
    assert get(1, "ABC") == "B"


def test_get_list_index():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_single_item_list():
    assert get([1], "ABC") == ("B",)


def test_get_empty_list():
    assert get([], "ABC") == ()


def test_get_dict():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678"}
    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_default():
    phonebook = {"Alice": "555-1234"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)
    assert get("Dennis", phonebook, "unknown") == "unknown"


def test_get_missing_raises():
    phonebook = {"Alice": "555-1234"}
    with pytest.raises(KeyError):
        get("Dennis", phonebook)
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_type_error_default():
    assert get(1.5, {"a": 1}, "default") == "default"


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([iter([1]), iter([2, 3])]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ["A", "B", "C", "D", "E"]


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------------------------------------------------------------------------
# interpose
# ---------------------------------------------------------------------------
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
    result = frequencies(["cat", "cat", "ox", "pig", "pig", "cat"])
    assert result == {"cat": 3, "ox": 1, "pig": 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    projects = [{"name": "build roads", "state": "CA", "cost": 1000000},
                {"name": "fight crime", "state": "IL", "cost": 100000},
                {"name": "help farmers", "state": "IL", "cost": 2000000},
                {"name": "help farmers", "state": "CA", "cost": 200000}]
    result = reduceby("state", lambda acc, x: acc + x["cost"], projects, 0)
    assert result == {"CA": 1200000, "IL": 2100000}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(5)] == [1, 2, 4, 8, 16]


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


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_with_len():
    # list has __len__, exercises fast path
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_without_len():
    # generator has no __len__, exercises binary search path
    result = list(partition_all(3, iter([1, 2, 3, 4])))
    assert result == [(1, 2, 3), (4,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{"id": 1}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data, "unknown")) == ["unknown", "Pies"]


def test_pluck_default_list():
    data = [[1], [2, 3]]
    result = list(pluck([0, 1], data, "missing"))
    assert result == [(1, "missing"), (2, 3)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------
def test_getter_single():
    get = getter(1)
    assert get("ABC") == "B"


def test_getter_list():
    get = getter([0, 2])
    assert get("ABC") == ("A", "C")


def test_getter_single_item_list():
    get = getter([1])
    assert get("ABC") == ("B",)


def test_getter_empty_list():
    get = getter([])
    assert get("ABC") == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right))
    assert result == [(2, 2), (3, 3)]


def test_join_right_outer():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_outer():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_full_outer():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_index_keys():
    friends = [("Alice", "Edith"),
               ("Alice", "Zhao"),
               ("Edith", "Alice"),
               ("Zhao", "Alice"),
               ("Zhao", "Edith")]
    cities = [("Alice", "NYC"),
              ("Alice", "Chicago"),
              ("Dan", "Sydney"),
              ("Edith", "Paris"),
              ("Edith", "Berlin"),
              ("Zhao", "Shanghai")]
    result = join(1, friends, 0, cities)
    result_set = set(result)
    assert result_set == {
        (("Alice", "Edith"), ("Edith", "Paris")),
        (("Alice", "Edith"), ("Edith", "Berlin")),
        (("Edith", "Alice"), ("Alice", "NYC")),
        (("Edith", "Alice"), ("Alice", "Chicago")),
        (("Zhao", "Alice"), ("Alice", "NYC")),
        (("Zhao", "Alice"), ("Alice", "Chicago")),
        (("Alice", "Zhao"), ("Zhao", "Shanghai")),
        (("Zhao", "Alice"), ("Alice", "NYC")),
        (("Zhao", "Edith"), ("Edith", "Paris")),
        (("Zhao", "Edith"), ("Edith", "Berlin")),
    } - {(("Zhao", "Alice"), ("Alice", "NYC")), (("Zhao", "Alice"), ("Alice", "Chicago"))} | {
        (("Zhao", "Alice"), ("Alice", "NYC")),
        (("Zhao", "Alice"), ("Alice", "Chicago")),
    }


def test_join_no_matches():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [3, 4]))
    assert result == []


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(["apples", "bananas"], ["Apples", "Oranges"],
                       key=str.lower))
    assert result == [("bananas", "Oranges")]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_three_seqs():
    assert list(diff([1, 1, 1], [1, 1, 2], [1, 1, 3])) == [(1, 2, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie", "Alice")


def test_topk_more_than_length():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


def test_topk_non_callable_key():
    data = [{"x": 1}, {"x": 5}, {"x": 3}]
    assert topk(2, data, key="x") == ({"x": 5}, {"x": 3})


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([10, 20, 30]))
    assert item == 10
    assert list(rest) == [10, 20, 30]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------------------------------------------------------------------------
# peekn
# ---------------------------------------------------------------------------
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


def test_peekn_zero():
    peeked, rest = peekn(0, [1, 2, 3])
    assert peeked == ()
    assert list(rest) == [1, 2, 3]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_seed_deterministic():
    seq = list(range(100))
    first = list(random_sample(0.1, seq, random_state=2016))
    second = list(random_sample(0.1, seq, random_state=2016))
    assert first == second
    assert first == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, list(range(100)), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, list(range(10)), random_state=1)) == list(
        range(10))
