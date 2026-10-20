import itertools
import operator

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
from funcy.primitives import EMPTY


# ---------- repeatedly / iterate ----------

def test_repeatedly_infinite():
    it = repeatedly(lambda: 7)
    assert take(5, it) == [7, 7, 7, 7, 7]


def test_repeatedly_with_n():
    assert list(repeatedly(lambda: 1, 3)) == [1, 1, 1]
    assert list(repeatedly(lambda: 1, 0)) == []


def test_repeatedly_side_effects():
    calls = []
    list(repeatedly(lambda: calls.append(1), 4))
    assert len(calls) == 4


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


def test_iterate_string():
    assert take(3, iterate(lambda s: s + '!', 'a')) == ['a', 'a!', 'a!!']


# ---------- take / drop / first / second / nth / last ----------

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2, 3]) == []


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(10, [1, 2])) == []


def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([5, 6])) == 5


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None


def test_nth():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20, 30]) is None
    # generator path
    assert nth(1, iter([1, 2, 3])) == 2
    assert nth(5, iter([1, 2, 3])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None
    assert last('abc') == 'c'


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


def test_shuffled():
    data = [1, 2, 3, 4, 5]
    result = shuffled(data)
    assert isinstance(result, list)
    assert sorted(result) == data
    # original untouched
    assert data == [1, 2, 3, 4, 5]


def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter(range(10))) == 10


# ---------- map / filter ----------

def test_lmap():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert lmap(lambda a, b: a + b, [1, 2], [10, 20]) == [11, 22]


def test_lmap_derived():
    assert lmap(0, [(1, 2), (3, 4)]) == [1, 3]
    assert lmap('a', [{'a': 1}, {'a': 2}]) == [1, 2]


def test_map_returns_iterator():
    result = map(lambda x: x + 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [2, 3, 4]


def test_lfilter():
    assert lfilter(lambda x: x % 2 == 0, range(6)) == [0, 2, 4]


def test_lfilter_derived():
    assert lfilter(1, [(0, 'a'), (1, 'b'), (2, 'c')]) == [(1, 'b')]


def test_filter_returns_iterator():
    result = filter(lambda x: x > 1, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [2, 3]


def test_lremove():
    assert lremove(lambda x: x % 2 == 0, range(6)) == [1, 3, 5]


def test_remove():
    result = remove(lambda x: x < 3, [1, 2, 3, 4])
    assert not isinstance(result, list)
    assert list(result) == [3, 4]


def test_keep():
    assert list(keep(lambda x: x if x > 2 else None, [1, 2, 3, 4])) == [3, 4]
    # one-argument version
    assert list(keep([0, 1, 2, 0, 3])) == [1, 2, 3]


def test_lkeep():
    assert lkeep(lambda x: x if x % 2 else 0, [1, 2, 3, 4]) == [1, 3]
    assert lkeep([0, None, 5, '', 'x']) == [5, 'x']


# ---------- without ----------

def test_without():
    result = without([1, 2, 3, 4, 5], 1, 3)
    assert list(result) == [2, 4, 5]


def test_lwithout():
    assert lwithout([1, 2, 3, 4, 5], 1, 3) == [2, 4, 5]
    assert lwithout([1, 2], 3) == [1, 2]


# ---------- concat / cat / flatten / mapcat ----------

def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]


def test_lcat():
    assert lcat([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]


def test_flatten():
    assert lflatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_flatten_custom_follow():
    def follow(x):
        return isinstance(x, list)
    assert list(flatten([[1, 2], (3, 4)], follow=follow)) == [1, 2, (3, 4)]


def test_lflatten():
    assert lflatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]


def test_mapcat():
    result = mapcat(lambda x: [x, x], [1, 2])
    assert not isinstance(result, list)
    assert list(result) == [1, 1, 2, 2]


def test_lmapcat():
    assert lmapcat(lambda x: [x, x * 10], [1, 2]) == [1, 10, 2, 20]


# ---------- interleave / interpose ----------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]
    # shorter sequence terminates
    assert list(interleave([1, 2, 3], [4])) == [1, 4]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose('-', ['a'])) == ['a']
    assert list(interpose('-', [])) == []


# ---------- takewhile / dropwhile ----------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]


def test_takewhile_one_arg():
    assert list(takewhile([1, 2, 0, 4])) == [1, 2]


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]


def test_dropwhile_one_arg():
    assert list(dropwhile([0, 0, 1, 0])) == [1, 0]


# ---------- distinct ----------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]


def test_distinct_with_key():
    data = [{'a': 1}, {'a': 1, 'b': 2}, {'a': 3}]
    result = list(distinct(data, key='a'))
    assert result == [{'a': 1}, {'a': 3}]


def test_ldistinct():
    assert ldistinct([1, 1, 2, 2, 3]) == [1, 2, 3]
    assert ldistinct([1, 2, 3], key=lambda x: x % 2) == [1, 2]


# ---------- split / lsplit ----------

def test_split():
    yes, no = split(lambda x: x % 2 == 0, range(6))
    assert list(yes) == [0, 2, 4]
    assert list(no) == [1, 3, 5]


def test_split_interleaved():
    yes, no = split(lambda x: x < 3, [1, 4, 2, 5, 3, 6])
    # consume alternately
    assert next(yes) == 1
    assert next(no) == 4
    assert next(yes) == 2
    assert list(yes) == [3]
    assert list(no) == [5, 6]


def test_lsplit():
    yes, no = lsplit(lambda x: x % 2 == 0, range(6))
    assert yes == [0, 2, 4]
    assert no == [1, 3, 5]


# ---------- split_at ----------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4, 5])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4, 5]


def test_split_at_beyond():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    a, b = lsplit_at(2, [1, 2, 3, 4])
    assert a == [1, 2]
    assert b == [3, 4]


# ---------- split_by ----------

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 2])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 2]


def test_lsplit_by():
    a, b = lsplit_by(lambda x: x < 3, [1, 2, 3, 1])
    assert a == [1, 2]
    assert b == [3, 1]


def test_lsplit_by_empty():
    a, b = lsplit_by(lambda x: x < 3, [])
    assert a == []
    assert b == []


# ---------- grouping ----------

def test_group_by():
    result = group_by(lambda x: x % 2, range(5))
    assert result[0] == [0, 2, 4]
    assert result[1] == [1, 3]


def test_group_by_derived():
    result = group_by('a', [{'a': 1}, {'a': 2}, {'a': 1}])
    assert result[1] == [{'a': 1}, {'a': 1}]
    assert result[2] == [{'a': 2}]


def test_group_by_keys():
    data = [{'keys': ['a', 'b']}, {'keys': ['a']}]
    result = group_by_keys('keys', data)
    assert result['a'] == data
    assert result['b'] == [data[0]]


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert result['a'] == [1, 3]
    assert result['b'] == [2]


def test_count_by():
    result = count_by(lambda x: x % 2, range(5))
    assert result[0] == 3
    assert result[1] == 2


def test_count_by_derived():
    result = count_by('a', [{'a': 1}, {'a': 2}, {'a': 1}])
    assert result[1] == 2
    assert result[2] == 1


def test_count_reps():
    result = count_reps('ababc')
    assert result['a'] == 2
    assert result['b'] == 2
    assert result['c'] == 1


# ---------- partition / chunks ----------

def test_partition_sequence():
    assert lpartition(2, range(6)) == [[0, 1], [2, 3], [4, 5]]
    assert lpartition(2, range(5)) == [[0, 1], [2, 3]]


def test_partition_with_step():
    assert lpartition(2, 3, range(7)) == [[0, 1], [3, 4], [6]]


def test_partition_iterable():
    assert lpartition(2, iter(range(5))) == [[0, 1], [2, 3]]


def test_partition_iterable_step_eq_n():
    assert lpartition(2, iter(range(5))) == [[0, 1], [2, 3]]


def test_partition_iterable_step_lt_n():
    assert lpartition(3, 2, iter(range(7))) == [[0, 1, 2], [2, 3, 4], [4, 5, 6]]


def test_partition_iterable_step_gt_n():
    assert lpartition(2, 4, iter(range(9))) == [[0, 1], [4, 5], [8]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        partition(0, [1, 2, 3])
    with pytest.raises(ValueError):
        partition(2, 0, [1, 2, 3])


def test_partition_returns_iterator():
    result = partition(2, [1, 2, 3, 4])
    assert not isinstance(result, list)
    assert list(result) == [[1, 2], [3, 4]]


def test_chunks_sequence():
    assert lchunks(3, range(7)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunks_with_step():
    assert lchunks(2, 3, range(7)) == [[0, 1], [3, 4], [6]]


def test_chunks_iterable():
    assert lchunks(3, iter(range(7))) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunks_iterable_step_lt_n():
    assert lchunks(3, 2, iter(range(5))) == [[0, 1, 2], [2, 3, 4], [4]]


def test_chunks_iterable_step_gt_n():
    assert lchunks(2, 4, iter(range(7))) == [[0, 1], [4, 5], [6]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        chunks(0, [1, 2])
    with pytest.raises(ValueError):
        chunks(2, 0, [1, 2])


def test_chunks_returns_iterator():
    result = chunks(2, [1, 2, 3])
    assert not isinstance(result, list)


# ---------- partition_by ----------

def test_partition_by():
    data = [1, 1, 2, 2, 2, 3, 1, 1]
    parts = [list(p) for p in partition_by(lambda x: x, data)]
    assert parts == [[1, 1], [2, 2, 2], [3], [1, 1]]


def test_lpartition_by():
    data = [1, 1, 2, 2]
    assert lpartition_by(lambda x: x, data) == [[1, 1], [2, 2]]


def test_lpartition_by_derived():
    data = [{'k': 1}, {'k': 1}, {'k': 2}]
    assert lpartition_by('k', data) == [[{'k': 1}, {'k': 1}], [{'k': 2}]]


# ---------- with_prev / with_next / pairwise ----------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]
    assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------- reductions / sums ----------

def test_reductions():
    assert lreductions(operator.add, [1, 2, 3, 4]) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert lreductions(operator.add, [1, 2, 3], 10) == [11, 13, 16]


def test_reductions_mul():
    assert lreductions(operator.mul, [1, 2, 3, 4]) == [1, 2, 6, 24]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]


def test_sums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]


def test_sums_with_acc():
    assert lsums([1, 2, 3], 10) == [11, 13, 16]


def test_sums_empty():
    assert lsums([]) == []


def test_lsums():
    assert lsums([5, 5, 5]) == [5, 10, 15]


def test_reductions_returns_iterator():
    result = reductions(operator.add, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == [1, 3, 6]


# ---------- tuple-like sequences and strings ----------

def test_partition_string():
    assert lpartition(2, 'abcd') == ['ab', 'cd']


def test_chunks_tuple():
    assert lchunks(2, (1, 2, 3)) == [(1, 2), (3,)]
