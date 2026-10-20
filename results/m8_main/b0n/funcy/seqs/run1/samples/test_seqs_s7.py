import random
import pytest

from funcy.seqs import (
    repeatedly, iterate, take, drop, first, second, nth, last, rest, butlast,
    shuffled, ilen, lmap, lfilter, map, filter, lremove, remove, lkeep, keep,
    without, lwithout, lconcat, lcat, flatten, lflatten, lmapcat, mapcat,
    interleave, interpose, takewhile, dropwhile, ldistinct, distinct, split,
    lsplit, split_at, lsplit_at, split_by, lsplit_by, group_by, group_by_keys,
    group_values, count_by, count_reps, partition, lpartition, chunks, lchunks,
    partition_by, lpartition_by, with_prev, with_next, pairwise, reductions,
    lreductions, sums, lsums,
)


# --------------------------------------------------------------------------
# repeatedly / iterate
# --------------------------------------------------------------------------
class TestRepeatedly:
    def test_infinite(self):
        counter = iter(range(1000))
        gen = repeatedly(lambda: next(counter))
        assert take(5, gen) == [0, 1, 2, 3, 4]

    def test_limited(self):
        calls = []

        def f():
            calls.append(1)
            return len(calls)

        result = list(repeatedly(f, 4))
        assert result == [1, 2, 3, 4]

    def test_limited_zero(self):
        assert list(repeatedly(lambda: 1, 0)) == []


class TestIterate:
    def test_iterate(self):
        assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]

    def test_iterate_numeric(self):
        assert take(4, iterate(lambda x: x + 1, 0)) == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# take / drop / first / second / nth / last / rest / butlast
# --------------------------------------------------------------------------
class TestTake:
    def test_short(self):
        assert take(2, [1, 2, 3, 4]) == [1, 2]

    def test_more_than_length(self):
        assert take(10, [1, 2, 3]) == [1, 2, 3]

    def test_infinite(self):
        assert take(3, iterate(lambda x: x + 1, 0)) == [0, 1, 2]

    def test_zero(self):
        assert take(0, [1, 2, 3]) == []

    def test_iterator(self):
        assert take(2, iter([5, 6, 7])) == [5, 6]


class TestDrop:
    def test_drop(self):
        assert list(drop(2, [1, 2, 3, 4])) == [3, 4]

    def test_drop_all(self):
        assert list(drop(10, [1, 2, 3])) == []

    def test_drop_zero(self):
        assert list(drop(0, [1, 2, 3])) == [1, 2, 3]

    def test_drop_on_iterator(self):
        assert list(drop(1, iter([1, 2, 3]))) == [2, 3]


class TestFirst:
    def test_first(self):
        assert first([1, 2, 3]) == 1

    def test_empty(self):
        assert first([]) is None

    def test_iterator(self):
        assert first(iter([9, 8])) == 9

    def test_empty_iterator(self):
        assert first(iter([])) is None


class TestSecond:
    def test_second(self):
        assert second([1, 2, 3]) == 2

    def test_too_short(self):
        assert second([1]) is None

    def test_empty(self):
        assert second([]) is None


class TestNth:
    def test_nth_list(self):
        assert nth(2, [10, 20, 30, 40]) == 30

    def test_nth_out_of_range(self):
        assert nth(10, [1, 2]) is None

    def test_nth_iterator(self):
        assert nth(1, iter([5, 6, 7])) == 6

    def test_nth_iterator_out_of_range(self):
        assert nth(5, iter([1, 2])) is None

    def test_nth_zero(self):
        assert nth(0, [7, 8, 9]) == 7

    def test_nth_negative(self):
        assert nth(-1, [1, 2, 3]) == 3


class TestLast:
    def test_last(self):
        assert last([1, 2, 3]) == 3

    def test_empty(self):
        assert last([]) is None

    def test_iterator(self):
        assert last(iter([1, 2, 3])) == 3

    def test_empty_iterator(self):
        assert last(iter([])) is None

    def test_string(self):
        assert last("abc") == "c"


class TestRest:
    def test_rest(self):
        assert list(rest([1, 2, 3])) == [2, 3]

    def test_rest_empty(self):
        assert list(rest([])) == []

    def test_rest_single(self):
        assert list(rest([1])) == []


class TestButlast:
    def test_butlast(self):
        assert list(butlast([1, 2, 3, 4])) == [1, 2, 3]

    def test_empty(self):
        assert list(butlast([])) == []

    def test_single(self):
        assert list(butlast([1])) == []

    def test_two(self):
        assert list(butlast([1, 2])) == [1]


# --------------------------------------------------------------------------
# shuffled
# --------------------------------------------------------------------------
class TestShuffled:
    def test_same_elements(self):
        items = list(range(20))
        result = shuffled(items)
        assert sorted(result) == items
        assert result is not items

    def test_does_not_mutate_input(self):
        items = [1, 2, 3, 4, 5]
        original = list(items)
        random.seed(0)
        shuffled(items)
        assert items == original

    def test_empty(self):
        assert shuffled([]) == []


# --------------------------------------------------------------------------
# ilen
# --------------------------------------------------------------------------
class TestIlen:
    def test_list(self):
        assert ilen([1, 2, 3]) == 3

    def test_empty(self):
        assert ilen([]) == 0

    def test_iterator(self):
        assert ilen(iter([1, 2, 3, 4])) == 4

    def test_range(self):
        assert ilen(range(100)) == 100


# --------------------------------------------------------------------------
# lmap / lfilter / map / filter with funcmaker support
# --------------------------------------------------------------------------
class TestLmap:
    def test_callable(self):
        assert lmap(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

    def test_none(self):
        assert lmap(None, [1, 2, 3]) == [1, 2, 3]

    def test_itemgetter_int(self):
        assert lmap(1, [(1, 2), (3, 4)]) == [2, 4]

    def test_itemgetter_slice(self):
        assert lmap(slice(2), ["abcd", "efgh"]) == ["ab", "ef"]

    def test_dict(self):
        d = {"a": 1, "b": 2}
        assert lmap(d, ["a", "b"]) == [1, 2]

    def test_multiple_seqs(self):
        assert lmap(lambda a, b: a + b, [1, 2], [3, 4]) == [4, 6]


class TestLfilter:
    def test_callable(self):
        assert lfilter(lambda x: x > 2, [1, 2, 3, 4]) == [3, 4]

    def test_none(self):
        assert lfilter(None, [0, 1, 2, 0, 3]) == [1, 2, 3]

    def test_int(self):
        assert lfilter(0, [(1, "a"), (0, "b")]) == [(1, "a")]

    def test_set(self):
        assert lfilter({1, 2}, [1, 2, 3, 4]) == [1, 2]


class TestMap:
    def test_callable(self):
        assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]

    def test_none(self):
        assert list(map(None, [1, 2, 3])) == [1, 2, 3]

    def test_int(self):
        assert list(map(0, [(1, 2), (3, 4)])) == [1, 3]

    def test_multiple_seqs(self):
        assert list(map(lambda a, b: a * b, [1, 2], [3, 4])) == [3, 8]


class TestFilter:
    def test_callable(self):
        assert list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [2, 4]

    def test_none(self):
        assert list(filter(None, [0, 1, "", "a"])) == [1, "a"]

    def test_string_regex(self):
        result = list(filter(r"^a", ["apple", "banana", "avocado"]))
        assert result == ["apple", "avocado"]


# --------------------------------------------------------------------------
# lremove / remove
# --------------------------------------------------------------------------
class TestLremove:
    def test_remove(self):
        assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]

    def test_none(self):
        assert lremove(None, [0, 1, 2]) == [0]


class TestRemove:
    def test_remove(self):
        assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]

    def test_none(self):
        assert list(remove(None, [0, 1, 2])) == [0]


# --------------------------------------------------------------------------
# lkeep / keep
# --------------------------------------------------------------------------
class TestLkeep:
    def test_one_arg(self):
        assert lkeep([0, 1, 2, None, 3, ""]) == [1, 2, 3]

    def test_two_arg(self):
        assert lkeep(lambda x: x * 2, [0, 1, 2, 3]) == [2, 4, 6]

    def test_one_arg_iterator(self):
        assert lkeep(iter([0, 5, 0, 7])) == [5, 7]


class TestKeep:
    def test_one_arg(self):
        assert list(keep([0, 1, 2, None, 3])) == [1, 2, 3]

    def test_two_arg(self):
        assert list(keep(lambda x: x if x > 1 else None, [1, 2, 3])) == [2, 3]

    def test_none_mapping(self):
        assert list(keep(None, [0, 1, 2])) == [1, 2]


# --------------------------------------------------------------------------
# without / lwithout
# --------------------------------------------------------------------------
class TestWithout:
    def test_without(self):
        assert list(without([1, 2, 3, 4, 5], 2, 4)) == [1, 3, 5]

    def test_without_none(self):
        assert list(without([1, 2, 3])) == [1, 2, 3]

    def test_without_all(self):
        assert list(without([1, 2], 1, 2)) == []


class TestLwithout:
    def test_without(self):
        assert lwithout([1, 2, 3, 4], 2, 3) == [1, 4]

    def test_without_no_items(self):
        assert lwithout([1, 2]) == [1, 2]


# --------------------------------------------------------------------------
# lconcat / lcat / flatten / lflatten / mapcat / lmapcat
# --------------------------------------------------------------------------
class TestLconcat:
    def test_lconcat(self):
        assert lconcat([1, 2], [3], [4, 5]) == [1, 2, 3, 4, 5]

    def test_lconcat_empty(self):
        assert lconcat() == []


class TestLcat:
    def test_lcat(self):
        assert lcat([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_lcat_empty(self):
        assert lcat([]) == []

    def test_lcat_strings(self):
        assert lcat(["ab", "cd"]) == ["a", "b", "c", "d"]


class TestFlatten:
    def test_flatten_nested(self):
        assert list(flatten([1, [2, [3, [4]]]])) == [1, 2, 3, 4]

    def test_flatten_flat(self):
        assert list(flatten([1, 2, 3])) == [1, 2, 3]

    def test_flatten_skips_strings(self):
        assert list(flatten(["ab", ["cd"]])) == ["ab", "cd"]

    def test_custom_follow(self):
        def follow(x):
            return isinstance(x, tuple)

        assert list(flatten([1, (2, (3,)), [4]], follow)) == [1, 2, 3, [4]]

    def test_flatten_infinite_guard(self):
        # self-referential check is just for practical sequences
        result = flatten([[], [1], [2, [3]]])
        assert list(result) == [1, 2, 3]


class TestLflatten:
    def test_lflatten(self):
        assert lflatten([1, [2, [3]]]) == [1, 2, 3]

    def test_lflatten_flat(self):
        assert lflatten([1, 2, 3]) == [1, 2, 3]


class TestLmapcat:
    def test_lmapcat(self):
        assert lmapcat(lambda x: [x, x], [1, 2]) == [1, 1, 2, 2]

    def test_lmapcat_multiple(self):
        assert lmapcat(lambda a, b: [a + b], [1, 2], [3, 4]) == [4, 6]


class TestMapcat:
    def test_mapcat(self):
        assert list(mapcat(lambda x: [x, -x], [1, 2])) == [1, -1, 2, -2]

    def test_mapcat_strings(self):
        assert list(mapcat(lambda x: x, ["ab", "cd"])) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------
# interleave / interpose
# --------------------------------------------------------------------------
class TestInterleave:
    def test_interleave(self):
        assert list(interleave([1, 2], [3, 4])) == [1, 3, 2, 4]

    def test_interleave_uneven(self):
        assert list(interleave([1, 2, 3], [4])) == [1, 4]

    def test_interleave_empty(self):
        assert list(interleave()) == []


class TestInterpose:
    def test_interpose(self):
        assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]

    def test_interpose_single(self):
        assert list(interpose(0, [1])) == [1]

    def test_interpose_empty(self):
        assert list(interpose(0, [])) == []


# --------------------------------------------------------------------------
# takewhile / dropwhile
# --------------------------------------------------------------------------
class TestTakewhile:
    def test_pred(self):
        assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]

    def test_one_arg(self):
        assert list(takewhile([1, 2, 0, 3])) == [1, 2]

    def test_pred_none(self):
        assert list(takewhile(None, [1, 2, 0, 3])) == [1, 2]


class TestDropwhile:
    def test_pred(self):
        assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]

    def test_one_arg(self):
        assert list(dropwhile([0, 0, 1, 0, 2])) == [1, 0, 2]

    def test_pred_none(self):
        assert list(dropwhile(None, [0, 0, 1, 0, 2])) == [1, 0, 2]


# --------------------------------------------------------------------------
# ldistinct / distinct
# --------------------------------------------------------------------------
class TestLdistinct:
    def test_ldistinct(self):
        assert ldistinct([1, 2, 1, 3, 2, 4]) == [1, 2, 3, 4]

    def test_ldistinct_key(self):
        assert ldistinct([1, 2, 3, 4], key=lambda x: x % 2) == [1, 2]


class TestDistinct:
    def test_distinct(self):
        assert list(distinct([1, 1, 2, 3, 3, 1])) == [1, 2, 3]

    def test_distinct_key(self):
        result = list(distinct(["a", "A", "b", "B"], key=str.lower))
        assert result == ["a", "b"]

    def test_distinct_empty(self):
        assert list(distinct([])) == []

    def test_distinct_key_int(self):
        assert list(distinct([(1, 2), (1, 3), (2, 4)], key=0)) == [(1, 2), (2, 4)]


# --------------------------------------------------------------------------
# split / lsplit
# --------------------------------------------------------------------------
class TestSplit:
    def test_split_lazy(self):
        yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
        assert list(yes) == [2, 4]
        assert list(no) == [1, 3, 5]

    def test_split_interleaved(self):
        yes, no = split(lambda x: x > 2, [1, 4, 2, 5, 3])
        # Consume alternating
        assert next(yes) == 4
        assert next(no) == 1
        assert next(yes) == 5
        assert next(no) == 2
        assert next(no) == 3
        with pytest.raises(StopIteration):
            next(yes)

    def test_split_pred_maker(self):
        yes, no = split(1, [(1, "a"), (0, "b"), (2, "c")])
        assert list(no) == [(0, "b")]
        assert list(yes) == [(1, "a"), (2, "c")]


class TestLsplit:
    def test_lsplit(self):
        yes, no = lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4])
        assert (yes, no) == ([2, 4], [1, 3])

    def test_lsplit_pred_maker(self):
        yes, no = lsplit(None, [0, 1, 2, 3])
        assert yes == [1, 2, 3]
        assert no == [0]


# --------------------------------------------------------------------------
# split_at / lsplit_at
# --------------------------------------------------------------------------
class TestSplitAt:
    def test_split_at(self):
        a, b = split_at(2, [1, 2, 3, 4])
        assert list(a) == [1, 2]
        assert list(b) == [3, 4]

    def test_split_at_beyond(self):
        a, b = split_at(10, [1, 2])
        assert list(a) == [1, 2]
        assert list(b) == []

    def test_split_at_zero(self):
        a, b = split_at(0, [1, 2])
        assert list(a) == []
        assert list(b) == [1, 2]


class TestLsplitAt:
    def test_lsplit_at(self):
        a, b = lsplit_at(2, [1, 2, 3, 4])
        assert a == [1, 2]
        assert b == [3, 4]

    def test_lsplit_at_beyond(self):
        a, b = lsplit_at(5, [1, 2])
        assert a == [1, 2]
        assert b == []


# --------------------------------------------------------------------------
# split_by / lsplit_by
# --------------------------------------------------------------------------
class TestSplitBy:
    def test_split_by(self):
        a, b = split_by(lambda x: x < 3, [1, 2, 3, 4, 1])
        assert list(a) == [1, 2]
        assert list(b) == [3, 4, 1]

    def test_split_by_all(self):
        a, b = split_by(lambda x: x < 3, [1, 2])
        assert list(a) == [1, 2]
        assert list(b) == []

    def test_split_by_none(self):
        a, b = split_by(lambda x: x < 3, [5, 1, 2])
        assert list(a) == []
        assert list(b) == [5, 1, 2]


class TestLsplitBy:
    def test_lsplit_by(self):
        a, b = lsplit_by(lambda x: x < 3, [1, 2, 3, 4])
        assert a == [1, 2]
        assert b == [3, 4]

    def test_lsplit_by_maker(self):
        a, b = lsplit_by(None, [1, 2, 0, 3])
        assert a == [1, 2]
        assert b == [0, 3]


# --------------------------------------------------------------------------
# group_by / group_by_keys / group_values / count_by / count_reps
# --------------------------------------------------------------------------
class TestGroupBy:
    def test_group_by(self):
        result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
        assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}

    def test_group_by_maker(self):
        result = group_by(0, [(1, "a"), (2, "b"), (1, "c")])
        assert dict(result) == {1: [(1, "a"), (1, "c")], 2: [(2, "b")]}

    def test_group_by_empty(self):
        result = group_by(lambda x: x, [])
        assert dict(result) == {}
        assert result == {}

    def test_group_by_missing_key_raises(self):
        result = group_by(lambda x: x, [1, 2])
        with pytest.raises(KeyError):
            result[99]


class TestGroupByKeys:
    def test_group_by_keys(self):
        items = [1, 2, 3, 4, 5, 6]
        result = group_by_keys(lambda x: [x % 2, x % 3], items)
        assert dict(result) == {
            1: [1, 3, 5],
            0: [2, 4, 6],
            2: [2, 5],
        }

    def test_group_by_keys_maker(self):
        result = group_by_keys(0, [((1, 2), "a"), ((3, 4), "b")])
        # key int -> itemgetter(0) -> get_keys(item) should be iterable
        assert dict(result) == {}


class TestGroupValues:
    def test_group_values(self):
        result = group_values([("a", 1), ("b", 2), ("a", 3)])
        assert dict(result) == {"a": [1, 3], "b": [2]}

    def test_group_values_empty(self):
        result = group_values([])
        assert dict(result) == {}


class TestCountBy:
    def test_count_by(self):
        result = count_by(lambda x: x % 2, [1, 2, 3, 4, 5])
        assert dict(result) == {1: 3, 0: 2}

    def test_count_by_maker(self):
        result = count_by(0, [(1, "a"), (2, "b"), (1, "c")])
        assert dict(result) == {1: 2, 2: 1}


class TestCountReps:
    def test_count_reps(self):
        result = count_reps([1, 2, 1, 3, 1, 2])
        assert dict(result) == {1: 3, 2: 2, 3: 1}

    def test_count_reps_empty(self):
        result = count_reps([])
        assert dict(result) == {}


# --------------------------------------------------------------------------
# partition / lpartition / chunks / lchunks
# --------------------------------------------------------------------------
class TestPartition:
    def test_partition(self):
        assert lpartition(2, [1, 2, 3, 4, 5]) == [(1, 2), (3, 4)]

    def test_partition_step(self):
        assert lpartition(2, 3, [1, 2, 3, 4, 5, 6, 7, 8]) == [(1, 2), (4, 5), (7, 8)]

    def test_partition_step_greater(self):
        assert lpartition(2, 4, list(range(10))) == [(0, 1), (4, 5), (8, 9)]

    def test_partition_step_equal(self):
        assert lpartition(3, list(range(7))) == [(0, 1, 2), (3, 4, 5)]

    def test_partition_step_smaller(self):
        assert lpartition(3, 1, list(range(5))) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    def test_partition_step_smaller_seq(self):
        # Sequence implementation with step < n
        assert lpartition(3, 1, [0, 1, 2, 3, 4]) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    def test_partition_iterator(self):
        assert lpartition(2, iter([1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]

    def test_partition_iterator_step(self):
        assert lpartition(2, 1, iter([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_partition_iterator_step_greater(self):
        assert lpartition(2, 3, iter([1, 2, 3, 4, 5, 6])) == [(1, 2), (4, 5)]

    def test_partition_zero_n(self):
        with pytest.raises(ValueError):
            lpartition(0, [1, 2, 3])

    def test_partition_zero_step(self):
        with pytest.raises(ValueError):
            lpartition(2, 0, [1, 2, 3])

    def test_partition_lazy(self):
        it = partition(2, [1, 2, 3, 4])
        assert next(iter(it)) == (1, 2)

    def test_partition_empty(self):
        assert lpartition(2, []) == []

    def test_partition_exact_fit(self):
        assert lpartition(2, [1, 2, 3, 4]) == [(1, 2), (3, 4)]


class TestChunks:
    def test_chunks(self):
        assert lchunks(2, [1, 2, 3, 4, 5]) == [(1, 2), (3, 4), (5,)]

    def test_chunks_step(self):
        assert lchunks(2, 3, list(range(8))) == [(0, 1), (3, 4), (6, 7)]

    def test_chunks_step_greater(self):
        assert lchunks(2, 4, list(range(10))) == [(0, 1), (4, 5), (8, 9)]

    def test_chunks_iterator(self):
        assert lchunks(2, iter([1, 2, 3])) == [(1, 2), (3,)]

    def test_chunks_iterator_step_smaller(self):
        assert lchunks(3, 1, iter(range(5))) == [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4), (4,)]

    def test_chunks_iterator_step_greater(self):
        assert lchunks(2, 3, iter(range(7))) == [(0, 1), (3, 4), (6,)]

    def test_chunks_iterator_step_equal(self):
        assert lchunks(2, iter(range(5))) == [(0, 1), (2, 3), (4,)]

    def test_chunks_zero(self):
        with pytest.raises(ValueError):
            lchunks(0, [1, 2])

    def test_chunks_empty(self):
        assert lchunks(2, []) == []


# --------------------------------------------------------------------------
# partition_by / lpartition_by
# --------------------------------------------------------------------------
class TestPartitionBy:
    def test_partition_by(self):
        result = lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5])
        assert result == [[1, 3], [2, 4], [5]]

    def test_partition_by_maker(self):
        result = lpartition_by(0, [(1, "a"), (1, "b"), (2, "c")])
        assert result == [[(1, "a"), (1, "b")], [(2, "c")]]

    def test_partition_by_empty(self):
        assert lpartition_by(lambda x: x, []) == []

    def test_partition_by_lazy(self):
        it = partition_by(lambda x: x, [1, 1, 2, 2, 3])
        first_group = next(iter(it))
        assert list(first_group) == [1, 1]


class TestLpartitionBy:
    def test_lpartition_by(self):
        assert lpartition_by(lambda x: x, [1, 1, 2, 2, 1]) == [[1, 1], [2, 2], [1]]


# --------------------------------------------------------------------------
# with_prev / with_next / pairwise
# --------------------------------------------------------------------------
class TestWithPrev:
    def test_with_prev(self):
        assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]

    def test_with_prev_fill(self):
        assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]

    def test_with_prev_empty(self):
        assert list(with_prev([])) == []


class TestWithNext:
    def test_with_next(self):
        assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]

    def test_with_next_fill(self):
        assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]

    def test_with_next_empty(self):
        assert list(with_next([])) == []

    def test_with_next_single(self):
        assert list(with_next([1])) == [(1, None)]


class TestPairwise:
    def test_pairwise(self):
        assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_pairwise_one(self):
        assert list(pairwise([1])) == []

    def test_pairwise_empty(self):
        assert list(pairwise([])) == []


# --------------------------------------------------------------------------
# reductions / lreductions / sums / lsums
# --------------------------------------------------------------------------
class TestReductions:
    def test_reductions_add(self):
        assert list(reductions(lambda a, b: a + b, [1, 2, 3])) == [1, 3, 6]

    def test_reductions_with_acc(self):
        assert list(reductions(lambda a, b: a + b, [1, 2, 3], 10)) == [11, 13, 16]

    def test_reductions_with_acc_zero(self):
        # acc=0 should still be used (not EMPTY)
        assert list(reductions(lambda a, b: a + b, [1, 2], 0)) == [1, 3]

    def test_reductions_empty(self):
        assert list(reductions(lambda a, b: a + b, [])) == []

    def test_reductions_mult(self):
        assert list(reductions(lambda a, b: a * b, [1, 2, 3, 4])) == [1, 2, 6, 24]


class TestLreductions:
    def test_lreductions(self):
        assert lreductions(lambda a, b: a + b, [1, 2, 3]) == [1, 3, 6]

    def test_lreductions_acc(self):
        assert lreductions(lambda a, b: a + b, [1, 2], 5) == [6, 8]


class TestSums:
    def test_sums(self):
        assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]

    def test_sums_acc(self):
        assert list(sums([1, 2, 3], 10)) == [11, 13, 16]

    def test_sums_empty(self):
        assert list(sums([])) == []


class TestLsums:
    def test_lsums(self):
        assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]

    def test_lsums_acc(self):
        assert lsums([1, 2, 3], 10) == [11, 13, 16]
