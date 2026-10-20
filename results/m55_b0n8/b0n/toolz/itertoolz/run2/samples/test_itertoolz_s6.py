import itertools
import operator
import collections

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


# ------------------------- remove ---------------------------------------
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_all():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_none():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_lazy():
    result = remove(lambda x: x < 2, iter([1, 2, 3]))
    assert not isinstance(result, list)
    assert list(result) == [2, 3]


# ------------------------- accumulate -----------------------------------
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


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


def test_accumulate_initial_none():
    # initial=None should be treated as a value, not default
    assert list(accumulate(operator.add, [1, 2], None)) == [None]


def test_accumulate_lazy():
    result = accumulate(operator.add, [1, 2, 3])
    assert not isinstance(result, list)
    assert next(result) == 1
    assert next(result) == 3


# ------------------------- groupby --------------------------------------
def test_groupby_callable_key():
    result = groupby(len, ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"])
    assert result == {3: ["Bob", "Dan"], 5: ["Alice", "Edith", "Frank"], 7: ["Charlie"]}


def test_groupby_iseven():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_member_key():
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


def test_groupby_preserves_order():
    result = groupby(lambda x: x % 2, [3, 1, 2, 4, 5])
    assert result == {1: [3, 1, 5], 0: [2, 4]}


# ------------------------- merge_sorted --------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_empty_seq():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_many():
    a = [1, 4, 7, 10]
    b = [2, 5, 8, 11]
    c = [3, 6, 9, 12]
    assert list(merge_sorted(a, b, c)) == list(sorted(a + b + c))


def test_merge_sorted_key_with_generator():
    gen = (x for x in [1, 5, 9])
    assert list(merge_sorted([2, 6], gen, key=lambda x: x)) == [1, 2, 5, 6, 9]


# ------------------------- interleave ----------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_with_empty():
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]


# ------------------------- unique --------------------------------------
def test_unique_tuple():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)


def test_unique_get_rid_of_dupes():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_lazy():
    res = unique(iter([1, 1, 2, 3, 3]))
    assert next(res) == 1
    assert next(res) == 2
    assert next(res) == 3


# ------------------------- isiterable ----------------------------------
def test_isiterable_list():
    assert isiterable([1, 2, 3]) is True


def test_isiterable_str():
    assert isiterable("abc") is True


def test_isiterable_int():
    assert isiterable(5) is False


def test_isiterable_generator():
    assert isiterable((x for x in range(3))) is True


# ------------------------- isdistinct ----------------------------------
def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ------------------------- take / drop / tail --------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_take_more_than_len():
    assert list(take(10, [1, 2, 3])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2, 3])) == []


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_len():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


def test_tail_n_zero():
    assert tail(0, [1, 2, 3]) == []


# ------------------------- take_nth ------------------------------------
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_1():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ------------------------- first/second/nth/last -----------------------
def test_first():
    assert first("ABC") == "A"


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"


def test_second_too_short():
    with pytest.raises(StopIteration):
        second("A")


def test_nth_sequence():
    assert nth(1, "ABC") == "B"


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30])) == 30


def test_nth_out_of_bounds():
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2, 3]))


def test_nth_list():
    assert nth(0, [1, 2, 3]) == 1


def test_last():
    assert last("ABC") == "C"


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3


def test_last_empty():
    with pytest.raises(IndexError):
        last([])


# ------------------------- get -----------------------------------------
def test_get_int_string():
    assert get(1, "ABC") == "B"


def test_get_list_of_inds():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_dict():
    phonebook = {
        "Alice": "555-1234",
        "Bob": "555-5678",
        "Charlie": "555-9999",
    }
    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_default():
    phonebook = {"Alice": "555-1234"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)


def test_get_default_single():
    assert get("missing", {"a": 1}, "default") == "default"


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get("missing", {"a": 1})


def test_get_index_error_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_empty_list_no_default():
    assert get([], "ABC") == ()


def test_get_single_list_no_default():
    assert get([1], "ABC") == ("B",)


def test_get_list_index_of_int():
    # get with list index on a non-subscriptable-by-list object falls back
    with pytest.raises(TypeError):
        get([0, 1], 5)


def test_get_list_type_error_default():
    assert get([0, 1], 5, "x") == ("x", "x")


# ------------------------- concat / concatv / mapcat --------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat((x for x in [[1], [2]]))) == [1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]]))
    assert result == ["A", "B", "C", "D", "E"]


# ------------------------- cons / interpose ----------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ------------------------- frequencies ---------------------------------
def test_frequencies():
    result = frequencies(["cat", "cat", "ox", "pig", "pig", "cat"])
    assert result == {"cat": 3, "ox": 1, "pig": 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ------------------------- reduceby ------------------------------------
def test_reduceby_no_init():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, data, 0)
    assert result == {False: 9, True: 6}


def test_reduceby_with_callable_init():
    iseven = lambda x: x % 2 == 0

    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_member_key():
    projects = [
        {"name": "build roads", "state": "CA", "cost": 1000000},
        {"name": "fight crime", "state": "IL", "cost": 100000},
        {"name": "help farmers", "state": "IL", "cost": 2000000},
        {"name": "help farmers", "state": "CA", "cost": 200000},
    ]
    result = reduceby(
        "state", lambda acc, x: acc + x["cost"], projects, 0
    )
    assert result == {"CA": 1200000, "IL": 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ------------------------- iterate -------------------------------------
def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ------------------------- sliding_window ------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ------------------------- partition -----------------------------------
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


# ------------------------- partition_all -------------------------------
def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(3, [])) == []


def test_partition_all_generator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [
        (1, 2),
        (3, 4),
        (5,),
    ]


# ------------------------- count --------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(x for x in range(5)) == 5


def test_count_empty():
    assert count([]) == 0


def test_count_string():
    assert count("abc") == 3


# ------------------------- pluck --------------------------------------
def test_pluck_single():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_index_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{"a": 1}, {"b": 2}]
    assert list(pluck("a", data, default=0)) == [1, 0]


def test_pluck_multiple_with_default():
    data = [{"a": 1}, {"a": 2}]
    assert list(pluck(["a", "b"], data, default=None)) == [
        (1, None),
        (2, None),
    ]


# ------------------------- getter ------------------------------------
def test_getter_single():
    g = getter(1)
    assert g("ABC") == "B"


def test_getter_list_single():
    g = getter([1])
    assert g("ABC") == ("B",)


def test_getter_list_multi():
    g = getter([0, 1])
    assert g("ABC") == ("A", "B")


def test_getter_list_empty():
    g = getter([])
    assert g("ABC") == ()


# ------------------------- join ---------------------------------------
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
    result = set(join(second, friends, first, cities))
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
    assert result == expected


def test_join_outer_full():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4],
             left_default=None, right_default=None)
    )
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right_only():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], left_default=None)
    )
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_only():
    identity = lambda x: x
    result = list(
        join(identity, [1, 2, 3], identity, [2, 3, 4], right_default=None)
    )
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_member_keys():
    friends = [("Alice", "Edith"), ("Zhao", "Alice")]
    cities = [("Alice", "NYC"), ("Edith", "Paris")]
    result = list(join(1, friends, 0, cities))
    assert result == [
        (("Alice", "Edith"), ("Edith", "Paris")),
        (("Zhao", "Alice"), ("Alice", "NYC")),
    ]


# ------------------------- diff --------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]


def test_diff_key():
    assert list(diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)) == [
        ("bananas", "Oranges")
    ]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_multiple():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3])) == [(2, 2, 5)]


# ------------------------- topk --------------------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == ("Charlie", "Alice")


def test_topk_member_key():
    data = [{"v": 1}, {"v": 5}, {"v": 3}]
    assert topk(2, data, key="v") == ({"v": 5}, {"v": 3})


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ------------------------- peek / peekn ------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter(range(5))
    first_item, rest = peek(it)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_less_than_n():
    seq = [0, 1]
    peeked, rest = peekn(5, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1]


# ------------------------- random_sample -----------------------------
def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    seq = list(range(10))
    result = list(random_sample(1.0, seq, random_state=0))
    assert result == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    result = list(random_sample(0.0, seq, random_state=0))
    assert result == []


def test_random_sample_random_object():
    from random import Random

    randobj = Random(2016)
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
