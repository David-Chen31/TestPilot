import pytest
import operator
import sys

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
from funcy.primitives import EMPTY


# ---------------------------------------------------------------- repeatedly
def test_repeatedly_infinite():
    gen = repeatedly(lambda: 7)
    assert take(3, gen) == [7, 7, 7]


def test_repeatedly_finite():
    assert take(10, repeatedly(lambda: 1, 3)) == [1, 1, 1]


def test_repeatedly_zero():
    assert list(repeatedly(lambda: 1, 0)) == []


def test_repeatedly_side_effects():
    calls = []

    def f():
        calls.append(1)
        return len(calls)

    assert take(3, repeatedly(f)) == [1, 2, 3]
    assert len(calls) == 3


# ---------------------------------------------------------------- iterate
def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


def test_iterate_infinite_no_consume():
    it = iterate(lambda x: x + 1, 0)
    assert next(it) == 0
    assert next(it) == 1


# ---------------------------------------------------------------- take/drop
def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []


def test_take_iter():
    assert take(3, iter([1, 2, 3, 4])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---------------------------------------------------------------- first etc
def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([5])) == 5
    assert first(iter([])) is None


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None


def test_nth():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20]) is None
    # iterator path
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(5, iter([10, 20])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None
    assert last((1, 2)) == 2


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []
    assert list(butlast(iter([1, 2, 3]))) == [1, 2]


# ---------------------------------------------------------------- shuffled
def test_shuffled_preserves_multiset():
    src = [1, 2, 3, 4, 5]
    result = shuffled(src)
    assert isinstance(result, list)
    assert sorted(result) == src
    # original list unchanged
    assert src == [1, 2, 3, 4, 5]


def test_shuffled_from_iterator():
    result = shuffled(iter([1, 2, 3]))
    assert sorted(result) == [1, 2, 3]


# ---------------------------------------------------------------- ilen
def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3, 4])) == 4
    assert ilen(range(10)) == 10


# ---------------------------------------------------------------- lmap/map
def test_lmap_basic():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_lmap_multiple_seqs():
    assert lmap(operator.add, [1, 2], [10, 20]) == [11, 22]


def test_lmap_with_int():
    assert lmap(0, [[1, 2], [3, 4]]) == [1, 3]


def test_lmap_with_slice():
    assert lmap(slice(0, 2), [[1, 2, 3], [4, 5, 6]]) == [[1, 2], [4, 5]]


def test_lmap_with_dict():
    assert lmap({'a': 1}, ['a', 'b']) == [1, 1]  # KeyError? Actually __getitem__


def test_lmap_with_set():
    assert lmap({1, 2}, [1, 2, 3]) == [True, True, False]


def test_map_returns_iterator():
    result = map(lambda x: x + 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [2, 3, 4]


# ---------------------------------------------------------------- filter
def test_lfilter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]


def test_lfilter_with_none():
    assert lfilter(None, [0, 1, 2, 3]) == [1, 2, 3]


def test_lfilter_with_regex():
    assert lfilter(r'^a', ['apple', 'banana', 'avocado']) == ['apple', 'avocado']


def test_filter_returns_iterator():
    result = filter(lambda x: x > 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [2, 3]


# ---------------------------------------------------------------- remove
def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]


def test_remove_returns_iterator():
    result = remove(lambda x: x > 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [1]


# ---------------------------------------------------------------- keep
def test_lkeep_one_arg():
    assert lkeep([0, 1, 2, 0, 3]) == [1, 2, 3]


def test_lkeep_two_args():
    assert lkeep(lambda x: x * 2 if x % 2 else 0, [1, 2, 3]) == [2, 6]


def test_keep_one_arg_returns_iterator():
    result = keep([0, 1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [1, 2, 3]


def test_keep_two_args():
    result = keep(lambda x: x if x % 2 else 0, [1, 2, 3])
    assert list(result) == [1, 3]


# ---------------------------------------------------------------- without
def test_without():
    assert list(without([1, 2, 3, 4], 2, 4)) == [1, 3]


def test_without_iterator():
    assert list(without(iter([1, 2, 3]), 1)) == [2, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 4], 2, 4) == [1, 3]
    assert lwithout([1, 2], 5) == [1, 2]


# ---------------------------------------------------------------- concat
def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]


def test_lconcat_empty():
    assert lconcat() == []


def test_lcat():
    assert lcat([[1, 2], [3], []]) == [1, 2, 3]


def test_lcat_empty():
    assert lcat([]) == []


# ---------------------------------------------------------------- flatten
def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


def test_flatten_strings_are_not_unpacked():
    # strings are not sequence containers per is_seqcont
    assert list(flatten(['ab', 'cd'])) == ['ab', 'cd']


def test_flatten_custom_follow():
    assert list(flatten([1, 2, 3], follow=lambda x: isinstance(x, list))) == [1, 2, 3]


def test_lflatten():
    assert lflatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_lflatten_custom_follow():
    assert lflatten([[1, 2], [3]], follow=lambda x: isinstance(x, list)) == [1, 2, 3]


# ---------------------------------------------------------------- mapcat
def test_lmapcat():
    assert lmapcat(lambda x: [x, x], [1, 2]) == [1, 1, 2, 2]


def test_mapcat():
    result = mapcat(lambda x: [x, x], [1, 2, 3])
    assert list(result) == [1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------- interleave
def test_interleave():
    assert list(interleave([1, 2], [3, 4])) == [1, 3, 2, 4]


def test_interleave_different_lengths():
    assert list(interleave([1, 2, 3], [4])) == [1, 4]


def test_interleave_empty():
    assert list(interleave()) == []


# ---------------------------------------------------------------- interpose
def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]


def test_interpose_empty():
    assert list(interpose(0, [])) == []


def test_interpose_single():
    assert list(interpose(0, [1])) == [1]


# ---------------------------------------------------------------- takewhile
def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]


def test_takewhile_one_arg_truthy():
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]


def test_takewhile_empty():
    assert list(takewhile(lambda x: True, [])) == []


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]


def test_dropwhile_one_arg_truthy():
    assert list(dropwhile([1, 2, 0, 3])) == [0, 3]


# ---------------------------------------------------------------- distinct
def test_distinct():
    assert list(distinct([1, 1, 2, 2, 3])) == [1, 2, 3]


def test_distinct_key():
    assert list(distinct(['aa', 'ab', 'bc'], key=lambda x: x[0])) == ['aa', 'bc']


def test_distinct_key_int():
    assert list(distinct([(1, 2), (1, 3), (2, 4)], key=0)) == [(1, 2), (2, 4)]


def test_ldistinct():
    assert ldistinct([1, 2, 1, 3]) == [1, 2, 3]


def test_ldistinct_key():
    assert ldistinct(['aa', 'ab', 'bc'], key=lambda x: x[0]) == ['aa', 'bc']


# ---------------------------------------------------------------- split
def test_split():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert list(yes) == [2, 4]
    assert list(no) == [1, 3]


def test_split_interleaved_consumption():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert next(yes) == 2
    assert next(no) == 1
    assert next(yes) == 4
    assert next(no) == 3
    assert next(yes) == 6
    assert next(no) == 5


def test_lsplit():
    yes, no = lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert yes == [2, 4]
    assert no == [1, 3]


# ---------------------------------------------------------------- split_at
def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_split_at_beyond_length():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_split_at_zero():
    a, b = split_at(0, [1, 2])
    assert list(a) == []
    assert list(b) == [1, 2]


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])


# ---------------------------------------------------------------- split_by
def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1]


def test_split_by_all_pass():
    a, b = split_by(lambda x: True, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 1]) == ([1, 2], [3, 1])


# ---------------------------------------------------------------- group_by
def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_int_key():
    result = group_by(0, [(1, 2), (1, 3), (2, 4)])
    assert dict(result) == {1: [(1, 2), (1, 3)], 2: [(2, 4)]}


def test_group_by_keys():
    result = group_by_keys(lambda x: [x % 2, x % 3], [1, 2, 3])
    assert dict(result) == {1: [1, 3], 0: [3], 2: [2]}


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}


def test_count_by():
    result = count_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: 3, 0: 2}


def test_count_by_int():
    result = count_by(0, [(1, 2), (1, 3), (2, 4)])
    assert dict(result) == {1: 2, 2: 1}


def test_count_reps():
    result = count_reps([1, 1, 2, 3, 3, 3])
    assert dict(result) == {1: 2, 2: 1, 3: 3}


# ---------------------------------------------------------------- partition
def test_partition_seq():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_partition_seq_with_step():
    assert lpartition(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5]]


def test_partition_iter():
    assert lpartition(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]


def test_partition_iter_step_gt_n():
    assert lpartition(2, 4, iter([1, 2, 3, 4, 5, 6, 7, 8, 9])) == [[1, 2], [5, 6], [9, 10][:0]]


def test_partition_iter_step_eq_n():
    assert lpartition(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]


def test_partition_iter_step_lt_n():
    assert lpartition(3, 1, iter([1, 2, 3, 4, 5])) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_partition_zero_n_error():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2]))


def test_partition_zero_step_error():
    with pytest.raises(ValueError):
        list(partition(2, 0, [1, 2]))


def test_partition_is_lazy():
    result = partition(2, [1, 2, 3, 4])
    assert not isinstance(result, list)
    assert list(result) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------- chunks
def test_chunks_seq():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]


def test_chunks_seq_with_step():
    assert lchunks(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5], [7]]


def test_chunks_iter():
    assert lchunks(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]


def test_chunks_iter_step_gt_n():
    assert lchunks(2, 4, iter([1, 2, 3, 4, 5])) == [[1, 2], [5]]


def test_chunks_iter_step_lt_n():
    assert lchunks(3, 1, iter([1, 2, 3, 4, 5])) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5], [5]]


def test_chunks_zero_error():
    with pytest.raises(ValueError):
        list(chunks(0, [1, 2]))


def test_chunks_empty():
    assert lchunks(2, []) == []


def test_chunks_is_lazy():
    result = chunks(2, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [[1, 2], [3]]


# ---------------------------------------------------------------- partition_by
def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x % 2, [1, 3, 2, 4, 5])]
    assert result == [[1, 3], [2, 4], [5]]


def test_partition_by_int():
    result = lpartition_by(0, [(1, 'a'), (1, 'b'), (2, 'c')])
    assert result == [[(1, 'a'), (1, 'b')], [(2, 'c')]]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5]) == [[1, 3], [2, 4], [5]]


# ---------------------------------------------------------------- with_prev / next
def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2], fill=0)) == [(1, 0), (2, 1)]


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2], fill=0)) == [(1, 2), (2, 0)]


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_short():
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------------------------------------------------------------- reductions
def test_reductions_no_acc():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_other_func():
    assert list(reductions(operator.mul, [2, 3, 4])) == [2, 6, 24]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]


def test_sums():
    assert list(sums([1, 2, 3])) == [1, 3, 6]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]


def test_lsums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
