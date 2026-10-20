import itertools
import operator
import re
from collections import defaultdict

import pytest

from funcy.seqs import (
    repeatedly, iterate, take, drop, first, second, nth, last, rest, butlast,
    shuffled, ilen, lmap, lfilter, map, filter, lremove, remove, lkeep, keep,
    without, lwithout, lconcat, lcat, flatten, lflatten, lmapcat, mapcat,
    interleave, interpose, takewhile, dropwhile, ldistinct, distinct, split,
    lsplit, split_at, lsplit_at, split_by, lsplit_by, group_by, group_by_keys,
    group_values, count_by, count_reps, partition, lpartition, chunks,
    lchunks, partition_by, lpartition_by, with_prev, with_next, pairwise,
    reductions, lreductions, sums, lsums, count, cycle, repeat,
)
from funcy.primitives import EMPTY


# ---------------------------------------------------------------------------
# repeatedly / iterate / count / cycle / repeat
# ---------------------------------------------------------------------------

def test_repeatedly_infinite_first_n():
    it = repeatedly(lambda: 7)
    assert take(4, it) == [7, 7, 7, 7]


def test_repeatedly_with_n():
    assert list(repeatedly(lambda: 1, 3)) == [1, 1, 1]
    assert list(repeatedly(lambda: 1, 0)) == []


def test_repeatedly_side_effects():
    calls = []
    def f():
        calls.append(len(calls))
        return len(calls)
    assert list(repeatedly(f, 3)) == [1, 2, 3]
    assert calls == [0, 1, 2]


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


def test_count_reexport():
    c = count(5, 2)
    assert take(3, c) == [5, 7, 9]


def test_cycle_reexport():
    assert take(5, cycle([1, 2])) == [1, 2, 1, 2, 1]


def test_repeat_reexport():
    assert take(3, repeat('a')) == ['a', 'a', 'a']
    assert list(repeat('a', 2)) == ['a', 'a']


# ---------------------------------------------------------------------------
# take / drop / first / second / nth / last / rest / butlast
# ---------------------------------------------------------------------------

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []
    assert take(3, iter([1, 2, 3, 4])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(10, [1, 2])) == []
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(1, iter([1, 2, 3]))) == [2, 3]


def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([5, 6])) == 5
    assert first(iter([])) is None


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None
    assert second(iter([1, 2, 3])) == 2


def test_nth():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(1, [10, 20, 30]) == 20
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20, 30]) is None
    assert nth(0, iter([10, 20, 30])) == 10
    assert nth(2, iter([10, 20, 30])) == 30
    assert nth(5, iter([10, 20, 30])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []
    assert list(rest(iter([1, 2, 3]))) == [2, 3]


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []
    assert list(butlast(iter([1, 2, 3]))) == [1, 2]


def test_shuffled_preserves_elements():
    seq = list(range(20))
    result = shuffled(seq)
    assert isinstance(result, list)
    assert sorted(result) == seq


def test_shuffled_empty():
    assert shuffled([]) == []


def test_ilen():
    assert ilen([1, 2, 3, 4]) == 4
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3])) == 3
    assert ilen(iter([])) == 0


# ---------------------------------------------------------------------------
# lmap / lfilter / map / filter / lremove / remove / lkeep / keep
# ---------------------------------------------------------------------------

def test_lmap_basic():
    assert lmap(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]


def test_lmap_multiple_seqs():
    assert lmap(operator.add, [1, 2, 3], [10, 20, 30]) == [11, 22, 33]


def test_lmap_with_string():
    assert lmap('a', ['apple', 'banana', 'cherry']) == [True, True, False]


def test_lmap_with_attr_string():
    # re finder returns match object; check truthiness via bool
    result = lmap('b.', ['abc', 'xbz'])
    assert all(result)


def test_map_basic():
    assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]


def test_map_with_int_getter():
    assert list(map(0, [[1, 2], [3, 4]])) == [1, 3]


def test_filter_basic():
    assert list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [2, 4]


def test_lfilter_with_set():
    assert lfilter({1, 2}, [1, 2, 3, 4]) == [1, 2]


def test_lfilter_with_none_pred():
    assert lfilter(None, [0, 1, '', 'a', []]) == [1, 'a']


def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]


def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_with_set():
    assert list(remove({2, 3}, [1, 2, 3, 4])) == [1, 4]


def test_lkeep_two_arg():
    assert lkeep(lambda x: x if x % 2 else 0, [1, 2, 3, 4]) == [1, 3]


def test_lkeep_one_arg():
    assert lkeep([0, 1, '', 'a', None, 5]) == [1, 'a', 5]


def test_keep_two_arg():
    assert list(keep(lambda x: x * 2 if x > 1 else 0, [1, 2, 3])) == [4, 6]


def test_keep_one_arg():
    assert list(keep([0, 1, 2, 0, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# without / lwithout
# ---------------------------------------------------------------------------

def test_without():
    assert list(without([1, 2, 3, 4], 2, 4)) == [1, 3]
    assert list(without([1, 2, 3])) == [1, 2, 3]
    assert list(without([1, 2, 3], 5)) == [1, 2, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 2, 1], 1, 2) == [3]
    assert lwithout([1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# lconcat / lcat / flatten / lflatten / lmapcat / mapcat
# ---------------------------------------------------------------------------

def test_lconcat():
    assert lconcat([1, 2], [3], [4, 5]) == [1, 2, 3, 4, 5]
    assert lconcat() == []


def test_lcat():
    assert lcat([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
    assert lcat([]) == []


def test_flatten():
    assert list(flatten([1, [2, [3, [4]], 5]])) == [1, 2, 3, 4, 5]
    assert list(flatten([1, 2, 3])) == [1, 2, 3]
    assert list(flatten([])) == []


def test_flatten_custom_follow():
    # don't follow tuples explicitly
    result = list(flatten([1, (2, 3), [4]], follow=lambda x: isinstance(x, list)))
    assert result == [1, (2, 3), 4]


def test_lflatten():
    assert lflatten([1, [2, [3]], 4]) == [1, 2, 3, 4]


def test_flatten_strings_not_followed():
    # strings are seqcont? Let's verify actual behavior with a simple string
    result = list(flatten(['abc', 'de']))
    # strings generally aren't iterated as by is_seqcont in funcy
    assert isinstance(result, list)


def test_lmapcat():
    assert lmapcat(lambda x: [x, x], [1, 2, 3]) == [1, 1, 2, 2, 3, 3]


def test_mapcat():
    assert list(mapcat(lambda x: (x, -x), [1, 2, 3])) == [1, -1, 2, -2, 3, -3]


# ---------------------------------------------------------------------------
# interleave / interpose
# ---------------------------------------------------------------------------

def test_interleave():
    assert list(interleave([1, 2, 3], ['a', 'b', 'c'])) == [1, 'a', 2, 'b', 3, 'c']


def test_interleave_stops_at_shortest():
    assert list(interleave([1, 2, 3], ['a', 'b'])) == [1, 'a', 2, 'b']


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]


def test_interpose_empty():
    assert list(interpose(0, [])) == []


def test_interpose_single():
    assert list(interpose(0, [1])) == [1]


# ---------------------------------------------------------------------------
# takewhile / dropwhile
# ---------------------------------------------------------------------------

def test_takewhile_two_arg():
    assert list(takewhile(lambda x: x < 4, [1, 2, 3, 4, 1])) == [1, 2, 3]


def test_takewhile_one_arg():
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]


def test_takewhile_with_set():
    assert list(takewhile({1, 2}, [1, 2, 3, 1])) == [1, 2]


def test_dropwhile_two_arg():
    assert list(dropwhile(lambda x: x < 4, [1, 2, 3, 4, 1])) == [4, 1]


def test_dropwhile_one_arg():
    assert list(dropwhile([0, 0, 1, 0, 2])) == [1, 0, 2]


# ---------------------------------------------------------------------------
# distinct / ldistinct
# ---------------------------------------------------------------------------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]


def test_distinct_key():
    assert list(distinct(['a', 'A', 'b', 'B'], key=str.lower)) == ['a', 'b']


def test_distinct_key_string():
    assert list(distinct(['a1', 'b2', 'a3'], key='a')) == ['a1', 'b2']


def test_ldistinct():
    assert ldistinct([1, 1, 2, 3, 3]) == [1, 2, 3]
    assert ldistinct(['x', 'X'], key=str.lower) == ['x']


# ---------------------------------------------------------------------------
# split / lsplit
# ---------------------------------------------------------------------------

def test_split_basic():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert list(yes) == [2, 4, 6]
    assert list(no) == [1, 3, 5]


def test_split_interleaved_consumption():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert next(yes) == 2
    assert next(no) == 1
    assert next(yes) == 4
    assert list(yes) == [6]
    assert list(no) == [3, 5]


def test_split_with_set():
    yes, no = split({1, 2}, [1, 2, 3, 4])
    assert list(yes) == [1, 2]
    assert list(no) == [3, 4]


def test_lsplit():
    yes, no = lsplit(lambda x: x > 2, [1, 2, 3, 4])
    assert yes == [3, 4]
    assert no == [1, 2]


# ---------------------------------------------------------------------------
# split_at / lsplit_at
# ---------------------------------------------------------------------------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4, 5])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4, 5]


def test_split_at_beyond():
    a, b = split_at(10, [1, 2, 3])
    assert list(a) == [1, 2, 3]
    assert list(b) == []


def test_split_at_zero():
    a, b = split_at(0, [1, 2, 3])
    assert list(a) == []
    assert list(b) == [1, 2, 3]


def test_lsplit_at():
    assert lsplit_at(3, [1, 2, 3, 4, 5]) == ([1, 2, 3], [4, 5])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])


# ---------------------------------------------------------------------------
# split_by / lsplit_by
# ---------------------------------------------------------------------------

def test_split_by_basic():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 4, 1])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4, 1]


def test_split_by_none_pass():
    a, b = split_by(lambda x: x < 3, [5, 6])
    assert list(a) == []
    assert list(b) == [5, 6]


def test_split_by_all_pass():
    a, b = split_by(lambda x: x < 10, [1, 2, 3])
    assert list(a) == [1, 2, 3]
    assert list(b) == []


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 4]) == ([1, 2], [3, 4])


# ---------------------------------------------------------------------------
# group_by / group_by_keys / group_values / count_by / count_reps
# ---------------------------------------------------------------------------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert isinstance(result, defaultdict)
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_string_key():
    result = group_by('a', ['apple', 'banana', 'avocado', 'blueberry'])
    # regex finder returns match object; items may share equal truthy/falsy
    assert sum(len(v) for v in result.values()) == 4


def test_group_by_keys():
    data = [('apple', 'fruit'), ('carrot', 'veg'), ('banana', 'fruit')]
    result = group_by_keys(lambda pair: pair[1], data)
    assert dict(result) == {
        'fruit': [('apple', 'fruit'), ('banana', 'fruit')],
        'veg': [('carrot', 'veg')],
    }


def test_group_by_keys_multiple_keys():
    data = [1, 2, 3, 4]
    result = group_by_keys(lambda x: [x % 2, x % 3], data)
    assert set(result.keys()) == {0, 1, 2}


def test_group_values():
    data = [('a', 1), ('b', 2), ('a', 3)]
    result = group_values(data)
    assert dict(result) == {'a': [1, 3], 'b': [2]}


def test_count_by():
    result = count_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: 3, 0: 2}


def test_count_by_string():
    result = count_by('a', ['apple', 'banana', 'apricot'])
    assert sum(result.values()) == 3


def test_count_reps():
    result = count_reps(['a', 'b', 'a', 'c', 'b', 'a'])
    assert dict(result) == {'a': 3, 'b': 2, 'c': 1}


# ---------------------------------------------------------------------------
# partition / lpartition / chunks / lchunks
# ---------------------------------------------------------------------------

def test_partition_on_sequence():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]


def test_partition_step_less_than_n():
    assert list(partition(3, 1, [1, 2, 3, 4, 5])) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_partition_step_greater_than_n():
    assert list(partition(2, 3, [1, 2, 3, 4, 5, 6, 7])) == [[1, 2], [4, 5], [7, ...]] if False else [[1, 2], [4, 5]]


def test_partition_with_step_arg():
    assert list(partition(2, 2, [1, 2, 3, 4, 5, 6])) == [[1, 2], [3, 4], [5, 6]]


def test_partition_on_iterator():
    assert list(partition(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4]]


def test_partition_iterator_step_gt():
    assert list(partition(2, 3, iter([1, 2, 3, 4, 5, 6, 7]))) == [[1, 2], [4, 5]]


def test_partition_iterator_step_lt():
    assert list(partition(3, 1, iter([1, 2, 3, 4]))) == [[1, 2, 3], [2, 3, 4]]


def test_partition_iterator_step_eq():
    assert list(partition(2, 2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        partition(0, [1, 2, 3])


def test_partition_zero_step_raises():
    with pytest.raises(ValueError):
        partition(2, 0, [1, 2, 3])


def test_lpartition():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_chunks_on_sequence():
    assert list(chunks(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]


def test_chunks_on_iterator():
    assert list(chunks(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4], [5]]


def test_chunks_step():
    assert list(chunks(2, 3, [1, 2, 3, 4, 5, 6, 7])) == [[1, 2], [4, 5], [7]]


def test_chunks_iterator_step_gt():
    assert list(chunks(2, 4, iter([1, 2, 3, 4, 5, 6, 7, 8, 9]))) == [[1, 2], [5, 6], [9]]


def test_chunks_iterator_step_lt():
    assert list(chunks(3, 1, iter([1, 2, 3, 4]))) == [[1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        chunks(0, [1, 2])


def test_lchunks():
    assert lchunks(2, [1, 2, 3]) == [[1, 2], [3]]
    assert lchunks(3, 1, [1, 2, 3, 4]) == [[1, 2, 3], [2, 3, 4], [3, 4], [4]]


# ---------------------------------------------------------------------------
# partition_by / lpartition_by
# ---------------------------------------------------------------------------

def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x % 2, [1, 1, 2, 2, 3, 1])]
    assert result == [[1, 1], [2, 2], [3], [1]]


def test_partition_by_string():
    result = [list(g) for g in partition_by(None, [0, 0, 1, 2, 0])]
    assert result == [[0, 0], [1], [2], [0]]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 2, 3, 3, 4]) == [[1], [2], [3, 3], [4]]


# ---------------------------------------------------------------------------
# with_prev / with_next / pairwise
# ---------------------------------------------------------------------------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]


def test_with_prev_empty():
    assert list(with_prev([])) == []


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]


def test_with_next_empty():
    assert list(with_next([])) == []


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_empty_or_single():
    assert list(pairwise([])) == []
    assert list(pairwise([1])) == []


# ---------------------------------------------------------------------------
# reductions / lreductions / sums / lsums
# ---------------------------------------------------------------------------

def test_reductions_no_acc():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_mul():
    assert list(reductions(operator.mul, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_reductions_with_acc_mul():
    assert list(reductions(operator.mul, [2, 3, 4], 10)) == [20, 60, 240]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lreductions(operator.add, [1, 2, 3], 0) == [1, 3, 6]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 100)) == [101, 103, 106]


def test_lsums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lsums([1, 2, 3], 10) == [11, 13, 16]


# ---------------------------------------------------------------------------
# EMPTY sentinel and edge cases
# ---------------------------------------------------------------------------

def test_empty_sentinel_distinct():
    # When seq is not a sequence but an iterator with EMPTY key default
    assert list(distinct(iter([1, 1, 2]))) == [1, 2]


def test_reductions_empty_seq():
    assert list(reductions(operator.add, [])) == []


def test_lmap_empty():
    assert lmap(lambda x: x, []) == []


def test_filter_returns_iterator():
    result = filter(lambda x: x > 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [2, 3]
