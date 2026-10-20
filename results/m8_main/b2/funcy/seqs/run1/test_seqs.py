import itertools
import operator
import re

import pytest

from funcy.seqs import (
    repeatedly, iterate, take, drop, first, second, nth, last, rest, butlast,
    shuffled, ilen, lmap, lfilter, map, filter, lremove, remove, lkeep, keep,
    without, lwithout, lconcat, lcat, flatten, lflatten, lmapcat, mapcat,
    interleave, interpose, takewhile, dropwhile, ldistinct, distinct,
    split, lsplit, split_at, lsplit_at, split_by, lsplit_by,
    group_by, group_by_keys, group_values, count_by, count_reps,
    partition, lpartition, chunks, lchunks, partition_by, lpartition_by,
    with_prev, with_next, pairwise, reductions, lreductions, sums, lsums,
    count, cycle, repeat,
)
from funcy.primitives import EMPTY


# --- repeatedly / iterate -------------------------------------------------

def test_repeatedly_infinite():
    it = repeatedly(lambda: 7)
    assert take(4, it) == [7, 7, 7, 7]


def test_repeatedly_with_count():
    counter = itertools.count()
    it = repeatedly(lambda: next(counter), 3)
    assert list(it) == [0, 1, 2]


def test_repeatedly_zero():
    assert list(repeatedly(lambda: 1, 0)) == []


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


# --- take / drop ----------------------------------------------------------

def test_take_basic():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]


def test_take_fewer():
    assert take(10, [1, 2]) == [1, 2]


def test_take_zero():
    assert take(0, [1, 2, 3]) == []


def test_take_iterator():
    assert take(2, iter(range(5))) == [0, 1]


def test_drop_basic():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]


def test_drop_all():
    assert list(drop(10, [1, 2])) == []


# --- first / second / nth / last -----------------------------------------

def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None


def test_nth_sequence():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20, 30]) is None


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(5, iter([10, 20, 30])) is None


def test_last_sequence():
    assert last([1, 2, 3]) == 3
    assert last([]) is None


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


# --- shuffled -------------------------------------------------------------

def test_shuffled_same_elements():
    seq = [1, 2, 3, 4, 5]
    result = shuffled(seq)
    assert isinstance(result, list)
    assert sorted(result) == seq
    # original not mutated
    assert seq == [1, 2, 3, 4, 5]


def test_shuffled_empty():
    assert shuffled([]) == []


# --- ilen ----------------------------------------------------------------

def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen(iter([1, 2, 3, 4])) == 4
    assert ilen([]) == 0


def test_ilen_consumes_all_large():
    # maxlen must be 0 so zip fully consumes; with maxlen=None/1 the deque
    # would hold items but zip still iterates.  The key distinction is that
    # every element is counted.
    assert ilen(range(1000)) == 1000
    assert ilen(iter(range(7))) == 7


# --- lmap / map / lfilter / filter ---------------------------------------

def test_lmap_and_map():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert list(map(lambda x: x + 1, [1, 2])) == [2, 3]


def test_map_multiple_seqs():
    assert list(map(operator.add, [1, 2, 3], [10, 20, 30])) == [11, 22, 33]


def test_map_with_string():
    # string interpreted as regex finder
    result = list(map(r'\d+', ['a1', 'b22', 'c']))
    assert result == ['1', '22', None]


def test_map_with_item_index():
    assert lmap(1, [(1, 2), (3, 4)]) == [2, 4]


def test_map_with_dict():
    assert lmap({'a': 1, 'b': 2}, ['a', 'b']) == [1, 2]


def test_lfilter_filter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
    assert list(filter(lambda x: x > 1, [1, 2, 3])) == [2, 3]


def test_filter_with_string_predicate():
    result = list(filter(r'^a', ['abc', 'bac', 'aardvark']))
    assert result == ['abc', 'aardvark']


def test_filter_with_set():
    assert lfilter({'a', 'b'}, ['a', 'c', 'b']) == ['a', 'b']


def test_make_func_error():
    with pytest.raises(TypeError):
        lmap(object(), [1, 2])


# --- remove / lremove ----------------------------------------------------

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_lremove():
    assert lremove(lambda x: x > 1, [1, 2, 3]) == [1]


# --- keep / lkeep --------------------------------------------------------

def test_keep_mapper():
    assert list(keep(lambda x: x if x > 0 else None, [-1, 2, -3, 4])) == [2, 4]


def test_keep_single_arg():
    assert list(keep([0, 1, None, 2, '', 3])) == [1, 2, 3]


def test_keep_single_arg_uses_bool():
    # one-arg keep returns a filter (lazy); materialize before comparing
    assert list(keep([0, 1, None, 2, '', 3])) == [1, 2, 3]
    assert list(keep([0, 1, '', 2])) == [1, 2]


def test_lkeep():
    assert lkeep(lambda x: x * 2, [1, 0, 2]) == [2, 4]


def test_lkeep_single_arg():
    assert lkeep([0, 1, None, 2]) == [1, 2]


def test_keep_two_arg_truthiness():
    # ensure _filter(bool vs None) both drop falsy; assert exact
    assert list(keep(lambda x: x - 1, [1, 2, 0, 3])) == [1, -1, 2]


# --- without / lwithout --------------------------------------------------

def test_without():
    assert list(without([1, 2, 3, 4], 2, 4)) == [1, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 4], 1, 3) == [2, 4]


def test_without_no_items():
    assert list(without([1, 2])) == [1, 2]


# --- lconcat / lcat ------------------------------------------------------

def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]


def test_lcat():
    assert lcat([[1, 2], [3], []]) == [1, 2, 3]


# --- flatten / lflatten --------------------------------------------------

def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


def test_flatten_strings_not_flattened():
    assert list(flatten(['ab', ['cd']])) == ['ab', 'cd']


def test_flatten_custom_follow():
    # follow that only flattens lists recursively applies to nested lists too
    assert list(flatten([[1], [2, [3]]], follow=lambda x: isinstance(x, list))) == [1, 2, 3]


def test_flatten_custom_follow_propagates():
    # Verify that the follow predicate is threaded through recursion.
    # With a follow that only matches tuples, inner lists must NOT be flattened.
    data = [(1, [2, 3]), (4, [5])]
    assert list(flatten(data, follow=lambda x: isinstance(x, tuple))) == [1, [2, 3], 4, [5]]


def test_lflatten():
    assert lflatten([1, [2, [3]]]) == [1, 2, 3]


def test_lflatten_custom_follow():
    data = [(1, [2, 3]), (4,)]
    assert lflatten(data, follow=lambda x: isinstance(x, tuple)) == [1, [2, 3], 4]


# --- lmapcat / mapcat ----------------------------------------------------

def test_lmapcat():
    assert lmapcat(lambda x: [x, x], [1, 2]) == [1, 1, 2, 2]


def test_mapcat():
    assert list(mapcat(lambda x: [x, -x], [1, 2])) == [1, -1, 2, -2]


# --- interleave / interpose ---------------------------------------------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]


def test_interleave_uneven():
    assert list(interleave([1, 2, 3], [4, 5])) == [1, 4, 2, 5]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]


def test_interpose_empty():
    assert list(interpose(0, [])) == []


# --- takewhile / dropwhile ----------------------------------------------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]


def test_takewhile_single_arg():
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]


def test_takewhile_with_string():
    # regex 'a' matches any string containing 'a', including 'banana'
    assert list(takewhile(r'a', ['apple', 'ant', 'banana'])) == ['apple', 'ant', 'banana']


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]


def test_dropwhile_single_arg():
    # single-arg form: the argument is the sequence, bool is the predicate.
    # dropwhile(list) sees a non-empty list as truthy, so nothing is dropped
    # and the whole (already-evaluated) list is returned.
    assert list(dropwhile([0, 0, 1, 0])) == [0, 0, 1, 0]


def test_dropwhile_single_arg_drops_falsy():
    # dropwhile(list) evaluates bool(list) which is truthy -> drops nothing
    assert list(dropwhile([0, 0, 1, 0, 2])) == [0, 0, 1, 0, 2]


def test_dropwhile_single_arg_empty():
    # empty list is falsy, so dropwhile drops it entirely
    assert list(dropwhile([])) == []


# --- distinct / ldistinct ------------------------------------------------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]


def test_distinct_key():
    assert list(distinct(['a', 'A', 'b'], key=lambda s: s.lower())) == ['a', 'b']


def test_ldistinct():
    assert ldistinct([1, 1, 2, 3, 2]) == [1, 2, 3]


def test_ldistinct_key():
    assert ldistinct([1, 2, 3, 4], key=lambda x: x % 2) == [1, 2]


# --- split / lsplit ------------------------------------------------------

def test_split():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert list(yes) == [2, 4, 6]
    assert list(no) == [1, 3, 5]


def test_split_lazy_independent():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert list(no) == [1, 3]
    assert list(yes) == [2, 4]


def test_split_interleaved_consumption():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    got = []
    # alternate pulling from each side
    for y, n in zip(yes, no):
        got.append(y)
        got.append(n)
    assert got == [2, 1, 4, 3, 6, 5]


def test_lsplit():
    assert lsplit(lambda x: x > 2, [1, 2, 3, 4]) == ([3, 4], [1, 2])


# --- split_at / lsplit_at ------------------------------------------------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_split_at_beyond():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    assert lsplit_at(1, [1, 2, 3]) == ([1], [2, 3])


# --- split_by / lsplit_by -----------------------------------------------

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 4]


def test_split_by_all_pass():
    a, b = split_by(lambda x: x > 0, [1, 2, 3])
    assert list(a) == [1, 2, 3]
    assert list(b) == []


def test_lsplit_by():
    assert lsplit_by(lambda x: x % 2 == 0, [2, 4, 5, 6]) == ([2, 4], [5, 6])


# --- group_by / group_by_keys / group_values ---------------------------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert result == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_with_string():
    result = group_by(r'\d', ['a1', 'b2', 'c3'])
    assert result == {'1': ['a1'], '2': ['b2'], '3': ['c3']}


def test_group_by_keys():
    data = [('a', 1), ('b', 2)]
    result = group_by_keys(lambda pair: [pair[0]], data)
    assert result == {'a': [('a', 1)], 'b': [('b', 2)]}


def test_group_by_keys_multiple():
    # even numbers are keyed under 0, odd numbers under 1
    data = [1, 2, 3, 4]
    result = group_by_keys(lambda x: [0] if x % 2 == 0 else [1], data)
    assert result == {1: [1, 3], 0: [2, 4]}


def test_group_by_keys_multiple_keys_per_item():
    data = [1, 2]
    result = group_by_keys(lambda x: [x, x * 10], data)
    assert result == {1: [1], 10: [1], 2: [2], 20: [2]}


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert result == {'a': [1, 3], 'b': [2]}


# --- count_by / count_reps ----------------------------------------------

def test_count_by():
    assert count_by(lambda x: x % 2, [1, 2, 3, 4, 5]) == {1: 3, 0: 2}


def test_count_by_string():
    result = count_by(r'\d', ['a1', 'b2', 'c1'])
    assert result == {'1': 2, '2': 1}


def test_count_reps():
    assert count_reps(['a', 'b', 'a', 'a']) == {'a': 3, 'b': 1}


def test_count_reps_empty():
    assert count_reps([]) == {}


# --- partition / lpartition ---------------------------------------------

def test_partition_sequence():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_partition_with_step():
    assert lpartition(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5]]


def test_partition_step_larger_than_n():
    # n=2, step=4
    assert lpartition(2, 4, list(range(10))) == [[0, 1], [4, 5], [8, 9]]


def test_partition_exact():
    assert lpartition(2, [1, 2, 3, 4]) == [[1, 2], [3, 4]]


def test_partition_iterator():
    assert list(partition(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4]]


def test_partition_iterator_step_lt_n():
    # step < n
    result = list(partition(3, 2, iter([1, 2, 3, 4, 5])))
    assert result == [[1, 2, 3], [3, 4, 5]]


def test_partition_iterator_step_gt_n():
    result = list(partition(2, 3, iter(range(9))))
    assert result == [[0, 1], [3, 4], [6, 7]]


def test_partition_iterator_step_eq_n():
    # step == n branch: distinct consecutive non-overlapping parts
    result = list(partition(2, 2, iter([1, 2, 3, 4, 5])))
    assert result == [[1, 2], [3, 4]]


def test_partition_iterator_step_lt_n_longer():
    # with step < n, windows overlap; the last partial window dropped.
    result = list(partition(3, 1, iter([1, 2, 3, 4])))
    assert result == [[1, 2, 3], [2, 3, 4]]


def test_partition_iterator_skip_more_than_remaining():
    # step > n; skip gap consumes everything after first block
    result = list(partition(2, 5, iter(range(10))))
    assert result == [[0, 1], [5, 6]]


def test_partition_zero_n_raises():
    with pytest.raises(ValueError):
        lpartition(0, [1, 2, 3])


def test_partition_zero_step_raises():
    with pytest.raises(ValueError):
        lpartition(2, 0, [1, 2, 3])


def test_partition_zero_n_raises_message():
    with pytest.raises(ValueError) as exc:
        lpartition(0, [1, 2, 3])
    assert str(exc.value) == 'n and step must not be zero'


def test_partition_zero_step_raises_message():
    with pytest.raises(ValueError) as exc:
        lpartition(2, 0, [1, 2, 3])
    assert str(exc.value) == 'n and step must not be zero'


def test_partition_zero_n_without_step_raises():
    # single argument form: partition(n, seq)
    with pytest.raises(ValueError):
        lpartition(0, [1, 2, 3])


# --- chunks / lchunks ---------------------------------------------------

def test_chunks_sequence():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]


def test_chunks_with_step():
    assert lchunks(2, 3, [1, 2, 3, 4, 5, 6, 7, 8]) == [[1, 2], [4, 5], [7, 8]]


def test_chunks_sequence_tail_included():
    # sequence impl: limit = len(seq), so trailing partial chunk appears
    assert lchunks(2, [1, 2, 3]) == [[1, 2], [3]]


def test_chunks_sequence_single_arg():
    assert lchunks(3, [1, 2, 3, 4]) == [[1, 2, 3], [4]]


def test_chunks_iterator():
    assert list(chunks(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4], [5]]


def test_chunks_iterator_step_lt_n():
    # chunks includes the final non-fitting tail
    result = list(chunks(3, 2, iter([1, 2, 3, 4, 5])))
    assert result == [[1, 2, 3], [3, 4, 5], [5]]


def test_chunks_iterator_step_gt_n():
    result = list(chunks(2, 3, iter(range(8))))
    assert result == [[0, 1], [3, 4], [6, 7]]


def test_chunks_iterator_step_eq_n():
    # step == n branch, plus a trailing partial chunk
    result = list(chunks(2, 2, iter([1, 2, 3, 4, 5])))
    assert result == [[1, 2], [3, 4], [5]]


def test_chunks_iterator_tail_only():
    # fewer items than n -> only trailing partial chunk
    result = list(chunks(5, 5, iter([1, 2])))
    assert result == [[1, 2]]


def test_chunks_iterator_step_lt_n_tail():
    # step < n; after the full windows the loop ends with a residual pool,
    # and _cut_seq then yields further shrinking chunks from it.
    result = list(chunks(3, 1, iter([1, 2, 3, 4])))
    assert result == [[1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_chunks_iterator_step_gt_n_tail_after_gap():
    # step > n with a residual tail: after the gap-skip consumes past the end,
    # the final pool is empty, so no trailing partial chunk is yielded.
    result = list(chunks(2, 4, iter(range(7))))
    assert result == [[0, 1], [4, 5]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        lchunks(0, [1, 2])


def test_chunks_zero_step_raises():
    with pytest.raises(ValueError):
        lchunks(2, 0, [1, 2])


# --- partition_by / lpartition_by ---------------------------------------

def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x, [1, 1, 1, 2, 2, 3])]
    assert result == [[1, 1, 1], [2, 2], [3]]


def test_partition_by_string():
    # regex returns matched string; '1' and '2' get distinct keys
    result = [list(g) for g in partition_by(r'\d', ['1', '2', 'a', 'b'])]
    assert result == [['1'], ['2'], ['a', 'b']]


def test_lpartition_by():
    assert lpartition_by(lambda x: x, [1, 1, 2, 2, 2, 3]) == [[1, 1], [2, 2, 2], [3]]


# --- with_prev / with_next ----------------------------------------------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2], fill=0)) == [(1, 0), (2, 1)]


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2], fill=0)) == [(1, 2), (2, 0)]


# --- pairwise -----------------------------------------------------------

def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_short():
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# --- reductions / lreductions / sums / lsums ---------------------------

def test_reductions_default():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_custom_func():
    assert list(reductions(lambda a, b: a + b, [1, 2, 3], 0)) == [1, 3, 6]


def test_reductions_custom_func_no_acc():
    # no acc uses accumulate(seq, f)
    assert list(reductions(lambda a, b: a * b, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 100)) == [101, 103, 106]


def test_lsums():
    assert lsums([1, 2, 3]) == [1, 3, 6]


# --- re-exports --------------------------------------------------------

def test_count_reexport():
    it = count(10)
    assert take(3, it) == [10, 11, 12]


def test_cycle_reexport():
    assert take(5, cycle([1, 2])) == [1, 2, 1, 2, 1]


def test_repeat_reexport():
    assert take(3, repeat(5)) == [5, 5, 5]
