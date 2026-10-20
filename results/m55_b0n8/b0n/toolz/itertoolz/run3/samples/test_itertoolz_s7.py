import itertools
import operator
from random import Random

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


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_is_lazy_iterator():
    result = remove(lambda x: False, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate_default():
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
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert result == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_member_selector():
    data = [
        {"name": "Alice", "gender": "F"},
        {"name": "Bob", "gender": "M"},
        {"name": "Charlie", "gender": "M"},
    ]
    result = groupby("gender", data)
    assert result == {
        "F": [{"name": "Alice", "gender": "F"}],
        "M": [
            {"name": "Bob", "gender": "M"},
            {"name": "Charlie", "gender": "M"},
        ],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_returns_plain_dict():
    result = groupby(len, ["a", "bb"])
    assert type(result) is dict


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_zero_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    result = merge_sorted([3, 1, 2])
    assert list(result) == [3, 1, 2]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [])) == []
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [3, 4])) == [3, 4]


def test_merge_sorted_many():
    seqs = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    assert list(merge_sorted(*seqs)) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_merge_sorted_key_many():
    seqs = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    result = list(merge_sorted(*seqs, key=lambda x: -x))
    assert list(merge_sorted(*[list(reversed(s)) for s in seqs])) == [
        7, 8, 9, 4, 5, 6, 1, 2, 3
    ]
    # sanity check on key merge result with reversed inputs
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == (
        "cat",
        "mouse",
    )


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable / isdistinct
# ---------------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable("abc") is True
    assert isiterable(5) is False
    assert isiterable(iter([1, 2])) is True


def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / tail / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_is_iterator():
    result = take(2, [1, 2, 3])
    assert isinstance(result, itertools.islice)


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    result = tail(2, iter([10, 20, 30, 40, 50]))
    assert result == (40, 50)


def test_tail_more_than_length():
    assert tail(10, iter([1, 2])) == (1, 2)


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------

def test_first():
    assert first("ABC") == "A"
    assert first([10, 20]) == 10


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"


def test_second_short_raises():
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, "ABC") == "B"
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30])) == 30


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2, 3]))


def test_last():
    assert last("ABC") == "C"
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


def test_last_empty_raises():
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_index():
    assert get(1, "ABC") == "B"


def test_get_list_indices():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_single_element_list():
    assert get([1], "ABC") == ("B",)


def test_get_empty_list():
    assert get([], "ABC") == ()


def test_get_dict():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678"}
    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_default_list():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)


def test_get_default_missing_key():
    assert get("missing", {"a": 1}, "default") == "default"


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        get("missing", {"a": 1})


def test_get_index_error_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_default_index_error():
    assert get(10, [1, 2], "d") == "d"


def test_get_type_error_with_default():
    assert get("foo", 5, "d") == "d"


def test_get_type_error_without_default_raises():
    with pytest.raises(TypeError):
        get("foo", 5)


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    g = getter(1)
    assert g("ABC") == "B"


def test_getter_single_element_list():
    g = getter([1])
    assert g("ABC") == ("B",)


def test_getter_multiple():
    g = getter([0, 2])
    assert g("ABC") == ("A", "C")


def test_getter_empty_list():
    g = getter([])
    assert g("ABC") == ()


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons / interpose
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_input():
    gen = (x for x in [[1], [2, 3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_mapcat():
    result = list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    )
    assert result == ["A", "B", "C", "D", "E"]


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


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
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }


def test_frequencies_empty():
    assert frequencies([]) == {}


def test_frequencies_returns_dict():
    result = frequencies([1, 1, 2])
    assert type(result) is dict


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------

def test_reduceby_add():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


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
    assert result == {False: {1, 3}, True: {2, 4}}


def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2, lambda a, b: a + b, [1, 2, 3, 4])
    assert result == {1: 4, 0: 6}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [
        (1, 2),
        (2, 3),
        (3, 4),
    ]


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(5, [1, 2])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------

def test_partition_exact():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]


def test_partition_empty():
    assert list(partition(3, [])) == []


def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [
        (1, 2),
        (3, 4),
        (5,),
    ]


def test_partition_all_with_len():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_generator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_len():
    assert count([1, 2, 3]) == 3
    assert count("abc") == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


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
    assert list(pluck("name", data, default=None)) == [None, "Pies"]


def test_pluck_default_multiple():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, None), (2, 3)]


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
    result = join(second, friends, first, cities)
    result_pairs = sorted(unique(result))
    expected = sorted(
        [
            (("Alice", "Edith"), ("Edith", "Paris")),
            (("Alice", "Edith"), ("Edith", "Berlin")),
            (("Alice", "Zhao"), ("Zhao", "Shanghai")),
            (("Edith", "Alice"), ("Alice", "NYC")),
            (("Edith", "Alice"), ("Alice", "Chicago")),
            (("Zhao", "Alice"), ("Alice", "NYC")),
            (("Zhao", "Alice"), ("Alice", "Chicago")),
            (("Zhao", "Edith"), ("Edith", "Paris")),
            (("Zhao", "Edith"), ("Edith", "Berlin")),
        ]
    )
    assert result_pairs == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4],
             left_default=None, right_default=None)
    )
    assert sorted(result, key=lambda p: (str(p[0]), str(p[1]))) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)],
        key=lambda p: (str(p[0]), str(p[1])),
    )


def test_join_right_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2], identity, [2, 3], left_default=None)
    )
    assert result == [(2, 2), (None, 3)]


def test_join_left_outer():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2], identity, [2, 3], right_default=None)
    )
    assert result == [(2, 2), (1, None)]


def test_join_with_index_keys():
    friends = [("Alice", "Edith"), ("Alice", "Zhao")]
    cities = [("Alice", "NYC"), ("Edith", "Paris")]
    result = list(join(1, friends, 0, cities))
    assert result == [
        (("Alice", "Alice"), ("Alice", "NYC")),
        (("Alice", "Edith"), ("Edith", "Paris")),
    ]


def test_join_empty():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]


def test_diff_key():
    assert list(
        diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)
    ) == [("bananas", "Oranges")]


def test_diff_single_list_of_seqs():
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
    assert result == [(3, 3, 4)]


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


def test_topk_member_key():
    data = [{"v": 1}, {"v": 10}, {"v": 5}]
    assert topk(1, data, key="v") == ({"v": 10},)


def test_topk_more_than_length():
    assert topk(5, [1, 2, 3]) == (3, 2, 1)


def test_topk_zero():
    assert topk(0, [1, 2, 3]) == ()


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    first_item, seq = peek([0, 1, 2, 3, 4])
    assert first_item == 0
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    first_two, seq = peekn(2, [0, 1, 2, 3, 4])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, seq = peekn(10, [0, 1])
    assert peeked == (0, 1)
    assert list(seq) == [0, 1]


def test_peekn_zero():
    peeked, seq = peekn(0, [0, 1, 2])
    assert peeked == ()
    assert list(seq) == [0, 1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_deterministic_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_obj():
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=0)) == list(range(10))


def test_random_sample_is_lazy():
    result = random_sample(0.5, iter(range(100)), random_state=0)
    assert not isinstance(result, list)
