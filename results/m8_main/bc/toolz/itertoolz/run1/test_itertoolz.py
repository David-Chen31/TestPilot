import itertools
import operator
from collections import OrderedDict

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


# ---------------------------------------------------------------- remove
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------------------------------------------------------------- accumulate
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


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


def test_accumulate_is_lazy():
    def gen():
        yield 1
        yield 2
        yield 3

    it = accumulate(operator.add, gen())
    assert next(it) == 1
    assert next(it) == 3
    assert next(it) == 6


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    assert groupby(len, ["Alice", "Bob", "Charlie", "Dan"]) == {
        5: ["Alice"],
        3: ["Bob", "Dan"],
        7: ["Charlie"],
    }


def test_groupby_iseven():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_non_callable_key():
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


def test_groupby_returns_plain_dict():
    assert type(groupby(len, ["a", "bb"])) is dict


# ---------------------------------------------------------------- merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert "".join(merge_sorted("abc", "abc", "abc")) == "aaabbbccc"


def test_merge_sorted_no_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_sequence_inside():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_three():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3, 4, 5], [10])) == [1, 2, 3, 4, 5, 10]


def test_merge_sorted_four_sequences():
    # Exercises recursive _merge_sorted_binary path where len(L1) > 1 (line 141)
    assert list(merge_sorted([1, 5], [2, 6], [3, 7], [4, 8])) == list(range(1, 9))


def test_merge_sorted_five_sequences():
    assert list(
        merge_sorted([1, 6], [2, 7], [3, 8], [4, 9], [5, 10])
    ) == list(range(1, 11))


def test_merge_sorted_seq2_entirely_less_than_val1():
    # Exercises the inner-loop-exhaustion branch (line 164 break) followed by
    # the trailing `yield val1` / `yield from seq1` (lines 171-172).
    assert list(merge_sorted([10, 20], [1, 2, 3])) == [1, 2, 3, 10, 20]


def test_merge_sorted_key_four_sequences():
    assert list(
        merge_sorted([1, 5], [2, 6], [3, 7], [4, 8], key=lambda x: x)
    ) == list(range(1, 9))


def test_merge_sorted_key_empty_subseqs():
    # Exercises the StopIteration path of _merge_sorted_binary_key
    # (lines 190-192) by having a recursively-generated seq2 that yields nothing.
    assert list(merge_sorted([1], [], [], key=lambda x: x)) == [1]


def test_merge_sorted_key_seq2_entirely_less_than_val1():
    # Exercises line 202 (yield val2 in inner loop), line 207 break,
    # and trailing lines 214-215.
    assert list(merge_sorted([10, 20], [1, 2, 3], key=lambda x: x)) == [
        1,
        2,
        3,
        10,
        20,
    ]


def test_merge_sorted_key_inner_break_then_continue():
    # Exercises line 203 else -> yield val1, break, then outer loop continues.
    assert list(merge_sorted([2, 4], [1, 3], key=lambda x: x)) == [1, 2, 3, 4]


def test_merge_sorted_key_three_sequences_recursion():
    # Three seqs: L2 has 2 elements -> line 186 recursion.
    assert list(merge_sorted([1, 4], [2, 5], [3, 6], key=lambda x: x)) == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


# ---------------------------------------------------------------- interleave
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert "".join(interleave(("ABC", "XY"))) == "AXBYC"


def test_interleave_empty_outer():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_empty_inner():
    assert list(interleave([[], [1, 2]])) == [1, 2]


def test_interleave_lazy():
    it = interleave([itertools.count(), [10, 20]])
    assert next(it) == 0
    assert next(it) == 10


# ---------------------------------------------------------------- unique
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_unique_empty():
    assert list(unique([])) == []


def test_unique_lazy():
    def gen():
        yield 1
        yield 1
        yield 2

    assert list(unique(gen())) == [1, 2]


# ---------------------------------------------------------------- isiterable
def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable("abc") is True
    assert isiterable(iter([])) is True


def test_isiterable_false():
    assert isiterable(5) is False
    assert isiterable(None) is False


# ---------------------------------------------------------------- isdistinct
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


# ---------------------------------------------------------------- take / drop / tail
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_len():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2])) == []


def test_take_lazy():
    assert list(take(3, itertools.count())) == [0, 1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2])) == []


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_string():
    assert tail(2, "abcde") == "de"


# ---------------------------------------------------------------- take_nth
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_infinite():
    assert list(take(3, take_nth(2, itertools.count()))) == [0, 2, 4]


# ---------------------------------------------------------------- first/second/nth/last
def test_first():
    assert first("ABC") == "A"
    assert first([1, 2, 3]) == 1


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second("ABC") == "B"


def test_second_too_short():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, "ABC") == "B"
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


def test_last():
    assert last("ABC") == "C"
    assert last([1, 2, 3]) == 3


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------- get
def test_get_sequence():
    assert get(1, "ABC") == "B"


def test_get_dict():
    phonebook = {"Alice": "555-1234", "Bob": "555-5678", "Charlie": "555-9999"}
    assert get("Alice", phonebook) == "555-1234"
    assert get(["Alice", "Bob"], phonebook) == ("555-1234", "555-5678")


def test_get_list_indices():
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_list_indices_single():
    assert get([1], "ABC") == ("B",)


def test_get_list_indices_empty():
    assert get([], "ABC") == ()


def test_get_default():
    phonebook = {"Alice": "555-1234"}
    assert get(["Alice", "Dennis"], phonebook, None) == ("555-1234", None)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get("Dennis", {"Alice": "555-1234"})
    with pytest.raises(IndexError):
        get(5, "ABC")


def test_get_missing_with_default():
    assert get("Dennis", {}, "x") == "x"
    assert get(10, [1, 2], "y") == "y"


def test_get_non_indexable_with_default():
    # Trigger the TypeError fallback with a non-list index and a default.
    assert get("a", 5, "default") == "default"


def test_get_non_indexable_no_default_reraises():
    # TypeError with a non-list index and no default -> re-raise (line 463).
    with pytest.raises(TypeError):
        get("a", 5)


def test_get_single_element_list_no_default():
    assert get([0], "ABC") == ("A",)


# ---------------------------------------------------------------- concat/concatv
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_of_iterables():
    assert list(concat([range(3) for _ in range(2)])) == [0, 1, 2, 0, 1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]


def test_concatv_empty():
    assert list(concatv()) == []


# ---------------------------------------------------------------- mapcat
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]]))
    assert result == ["A", "B", "C", "D", "E"]


def test_mapcat_empty():
    assert list(mapcat(lambda s: s, [])) == []


# ---------------------------------------------------------------- cons
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_lazy():
    assert list(take(3, cons(1, itertools.count(10)))) == [1, 10, 11]


# ---------------------------------------------------------------- interpose
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, "a", 2, "a", 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {
        "cat": 3,
        "ox": 1,
        "pig": 2,
    }


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_add():
    assert reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5]) == {
        False: 9,
        True: 6,
    }


def test_reduceby_mul():
    assert reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5]) == {
        False: 15,
        True: 8,
    }


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

    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_size_1():
    assert list(sliding_window(1, [1, 2])) == [(1,), (2,)]


def test_sliding_window_too_big():
    assert list(sliding_window(5, [1, 2])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ---------------------------------------------------------------- partition
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------- partition_all
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator():
    def gen():
        yield from [1, 2, 3, 4, 5]
        yield 6
        yield 7

    assert list(partition_all(3, gen())) == [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_sized_with_remainder():
    # Uses the __len__ fast path in partition_all.
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_remainder_is_no_pad_sentinel():
    # Remainder only via generator, forcing binary-search branch.
    def gen():
        for i in range(7):
            yield i

    assert list(partition_all(4, gen())) == [(0, 1, 2, 3), (4, 5, 6)]


def test_partition_all_len_raises_typeerror_binary_search():
    # A sized object whose __len__ raises TypeError forces the implementation
    # through the binary-search path. The equality check `prev[-1] is no_pad`
    # is True here, then `len(seq)` raises TypeError, which is caught and the
    # binary search computes the first index of no_pad. Result is (1, 2, 3)
    # for partition size 4 over the 3-element sequence [1, 2, 3].
    class BrokenLen:
        def __iter__(self):
            return iter([1, 2, 3])

        def __len__(self):
            raise TypeError("no len")

    assert list(partition_all(2, BrokenLen())) == [(1, 2), (3,)]


# ---------------------------------------------------------------- count
def test_count_sized():
    assert count([1, 2, 3]) == 3
    assert count("abcd") == 4
    assert count({}) == 0


def test_count_lazy():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(x for x in range(5)) == 5


def test_count_infinite_take():
    assert count(take(4, itertools.count())) == 4


# ---------------------------------------------------------------- pluck
def test_pluck_single():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default_single():
    data = [{"id": 1}, {"other": 2}]
    assert list(pluck("id", data, default=None)) == [1, None]


def test_pluck_default_multiple():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, 2), (3, None)]


# ---------------------------------------------------------------- getter
def test_getter_callable():
    assert getter(1)("ABC") == "B"


def test_getter_list_single():
    assert getter([1])("ABC") == ("B",)


def test_getter_list_multiple():
    assert getter([0, 2])("ABC") == ("A", "C")


def test_getter_list_empty():
    assert getter([])("ABC") == ()


# ---------------------------------------------------------------- join
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
    result = sorted(unique(join(second, friends, first, cities)))
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
    assert result == expected


def test_join_inner_index():
    friends = [("Alice", "Edith"), ("Bob", "Alice")]
    cities = [("Alice", "NYC"), ("Bob", "LA")]
    result = list(join(1, friends, 0, cities))
    assert set(result) == {((("Bob", "Alice")), ("Alice", "NYC"))}


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}


def test_join_right_join():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [2, 3],
                       left_default=None))
    assert set(result) == {(2, 2), (None, 3)}


def test_join_left_join():
    # right_default set, left_default unset -> lines 904-911 branch.
    result = list(join(lambda x: x, [1, 2], lambda x: x, [2, 3],
                       right_default=None))
    assert set(result) == {(2, 2), (1, None)}


def test_join_left_join_all_left_unmatched():
    # right_default set, no matches: all left items emitted with right_default.
    result = list(join(lambda x: x, [1, 2], lambda x: x, [5, 6],
                       right_default=None))
    assert list(result) == [(1, None), (2, None)]


def test_join_empty_left():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []


def test_join_empty_right():
    assert list(join(lambda x: x, [1, 2], lambda x: x, [])) == []


def test_join_empty_left_full_outer():
    result = list(join(lambda x: x, [], lambda x: x, [1, 2],
                       left_default=None, right_default=None))
    assert list(result) == [(None, 1), (None, 2)]


def test_join_right_empty_full_outer():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [],
                       left_default=None, right_default=None))
    assert set(result) == {(1, None), (2, None)}


# ---------------------------------------------------------------- diff
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]


def test_diff_key():
    result = list(diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower))
    assert result == [("bananas", "Oranges")]


def test_diff_multiple_seqs():
    assert list(diff([1, 2], [1, 3], [1, 4])) == [(2, 3, 4)]


def test_diff_single_list_argument():
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_single_empty_list_argument():
    # One list argument that unwraps to a single (empty) sequence -> N < 2 raise.
    with pytest.raises(TypeError) as excinfo:
        list(diff([]))
    assert str(excinfo.value) == "Too few sequences given (min 2 required)"


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ---------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == (
        "Charlie",
        "Alice",
    )


def test_topk_non_callable_key():
    data = [{"v": 1}, {"v": 5}, {"v": 3}]
    assert topk(1, data, key="v") == ({"v": 5},)


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


def test_topk_finite_prefix():
    assert topk(3, list(range(5))) == (4, 3, 2)


# ---------------------------------------------------------------- peek
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    first_item, rest = peek(it)
    assert first_item == 10
    assert list(rest) == [10, 20, 30]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------------------------------------------------------------- peekn
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(5, [0, 1])
    assert peeked == (0, 1)
    assert list(rest) == [0, 1]


def test_peekn_zero():
    peeked, rest = peekn(0, [1, 2, 3])
    assert peeked == ()
    assert list(rest) == [1, 2, 3]


# ---------------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98,
    ]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, list(range(100)), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, list(range(10)), random_state=1)) == list(range(10))


def test_random_sample_custom_state():
    import random as _random

    rnd = _random.Random(2016)
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=rnd)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98,
    ]


def test_random_sample_no_state_lazy():
    it = random_sample(0.5, range(20), random_state=0)
    result = list(it)
    assert all(0 <= x < 20 for x in result)
