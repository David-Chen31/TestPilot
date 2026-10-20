import itertools
import operator
import re
from collections import defaultdict

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


# ---------- generators / iterators ----------

def test_repeatedly_infinite():
    it = repeatedly(lambda: 7)
    assert take(5, it) == [7, 7, 7, 7, 7]


def test_repeatedly_with_n():
    assert repeatedly(lambda: 'x', 3) == ['x', 'x', 'x']
    assert repeatedly(lambda: 'x', 0) == []


def test_repeatedly_side_effects():
    counter = {'n': 0}

    def inc():
        counter['n'] += 1
        return counter['n']

    assert take(3, repeatedly(inc)) == [1, 2, 3]


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]
    assert take(3, iterate(lambda x: x + 1, 0)) == [0, 1, 2]


def test_count_cycle_repeat_reexports():
    assert take(5, count(10)) == [10, 11, 12, 13, 14]
    assert take(5, cycle([1, 2])) == [1, 2, 1, 2, 1]
    assert take(3, repeat('a')) == ['a', 'a', 'a']
    assert take(3, repeat('a', 10)) == ['a', 'a', 'a']


# ---------- take / drop etc ----------

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []
    assert take(2, iter([1, 2, 3])) == [1, 2]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(10, [1, 2])) == []
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(1, iter([1, 2, 3]))) == [2, 3]


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
    # iterator path
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(5, iter([10, 20, 30])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    # iterator path
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3, 4])) == [1, 2, 3]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


def test_shuffled_preserves_items():
    src = list(range(20))
    out = shuffled(src)
    assert sorted(out) == src
    assert out is not src


def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter(range(10))) == 10
    assert ilen(x for x in 'abcd') == 4


# ---------- map / filter family ----------

def test_lmap_and_map():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]
    # multi-sequence
    assert lmap(operator.add, [1, 2], [10, 20]) == [11, 22]


def test_lmap_with_key():
    # int / slice / str derived mapper
    assert lmap(0, [(1, 2), (3, 4)]) == [1, 3]
    assert lmap('a', [{'a': 1}, {'a': 2}]) == [1, 2]


def test_lfilter_and_filter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
    assert list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [2, 4]


def test_filter_with_pred():
    assert lfilter(0, [(0, 'x'), (1, 'y')]) == [(0, 'x')]
    assert lfilter({'a', 'b'}, ['a', 'c', 'b']) == ['a', 'b']


def test_lremove_and_remove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_keep():
    assert list(keep(lambda x: x if x > 0 else None, [-1, 1, 2, -2])) == [1, 2]
    # one-arg version filters truthy
    assert list(keep([0, 1, '', 'a', None, 2])) == [1, 'a', 2]
    assert lkeep(lambda x: x if x > 0 else None, [-1, 1, 2]) == [1, 2]
    assert lkeep([0, 1, '', 'a']) == [1, 'a']


def test_without():
    assert list(without([1, 2, 3, 4, 5], 2, 4)) == [1, 3, 5]
    assert lwithout([1, 2, 3, 4, 5], 2, 4) == [1, 3, 5]
    assert list(without([1, 2, 3])) == [1, 2, 3]


# ---------- concat / flatten ----------

def test_lconcat_concat():
    assert lconcat([1, 2], [3], [4, 5]) == [1, 2, 3, 4, 5]
    assert list(lconcat()) == []


def test_lcat_cat():
    assert lcat([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
    assert list(lcat([])) == []


def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]
    assert list(flatten([1, (2, 3), [4, [5]]])) == [1, 2, 3, 4, 5]
    assert list(flatten([])) == []


def test_flatten_custom_follow():
    # don't descend into strings
    nested = ['ab', 'cd']
    assert list(flatten(nested)) == ['ab', 'cd']
    # custom follow: always unpack lists
    assert list(flatten([[1, [2]], [3]], follow=lambda x: isinstance(x, list))) == [1, [2], 3]


def test_lflatten():
    assert lflatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_lmapcat_mapcat():
    assert lmapcat(lambda x: [x, x * 2], [1, 2, 3]) == [1, 2, 2, 4, 3, 6]
    assert list(mapcat(lambda x: [x, x], [1, 2])) == [1, 1, 2, 2]


# ---------- interleave / interpose ----------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]
    # Stops at shortest
    assert list(interleave([1, 2], [3, 4], [5])) == [1, 3, 5]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [1])) == [1]
    assert list(interpose(0, [])) == []


# ---------- takewhile / dropwhile ----------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 4, 1])) == [1, 2]
    # one-arg version checks truthiness
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]
    assert list(takewhile([])) == []


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 4, 1])) == [3, 4, 1]
    # one-arg version checks truthiness
    assert list(dropwhile([1, 2, 0, 3])) == [0, 3]
    assert list(dropwhile([1, 2])) == []


# ---------- distinct ----------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]
    assert ldistinct([1, 2, 1, 3, 2, 4]) == [1, 2, 3, 4]


def test_distinct_key():
    assert list(distinct(['A', 'a', 'b', 'B'], key=str.lower)) == ['A', 'b']
    assert ldistinct(['A', 'a', 'b', 'B'], str.lower) == ['A', 'b']


# ---------- split family ----------

def test_split():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert list(yes) == [2, 4, 6]
    assert list(no) == [1, 3, 5]


def test_split_alternating():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    # Interleaved consumption
    assert next(yes) == 2
    assert next(no) == 1
    assert next(yes) == 4
    assert next(no) == 3


def test_lsplit():
    assert lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4]) == ([2, 4], [1, 3])


def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4, 5])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4, 5]


def test_split_at_empty_and_large():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4, 5]) == ([1, 2], [3, 4, 5])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])


def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 4]


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 1, 4]) == ([1, 2], [3, 1, 4])
    assert lsplit_by(lambda x: x < 3, [5, 6]) == ([], [5, 6])


# ---------- grouping ----------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert isinstance(result, defaultdict)
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_key():
    result = group_by(lambda x: x['k'], [{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}, {'k': 'a', 'v': 3}])
    assert dict(result) == {'a': [{'k': 'a', 'v': 1}, {'k': 'a', 'v': 3}],
                            'b': [{'k': 'b', 'v': 2}]}


def test_group_by_keys():
    items = [(1, [10, 20]), (2, [20, 30])]
    result = group_by_keys(lambda x: x[1], items)
    assert dict(result) == {10: [(1, [10, 20])],
                            20: [(1, [10, 20]), (2, [20, 30])],
                            30: [(2, [20, 30])]}


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}


def test_count_by():
    result = count_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: 3, 0: 2}


def test_count_reps():
    result = count_reps('aabbbc')
    assert dict(result) == {'a': 2, 'b': 3, 'c': 1}


# ---------- partition / chunks ----------

def test_partition_seq():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]
    assert lpartition(3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2, 3], [4, 5, 6]]


def test_partition_with_step():
    assert list(partition(2, 3, range(10))) == [[0, 1], [3, 4], [6, 7]]
    assert lpartition(3, 2, range(10)) == [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]]


def test_partition_iter():
    assert list(partition(2, iter(range(5)))) == [[0, 1], [2, 3]]
    assert list(partition(2, 1, iter(range(5)))) == [[0, 1], [1, 2], [2, 3], [3, 4]]


def test_partition_step_greater_than_n_iter():
    assert list(partition(2, 4, iter(range(10)))) == [[0, 1], [4, 5], [8, 9]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        partition(0, [1, 2, 3])
    with pytest.raises(ValueError):
        partition(2, 0, [1, 2, 3])


def test_chunks():
    assert list(chunks(3, [1, 2, 3, 4, 5])) == [[1, 2, 3], [4, 5]]
    assert lchunks(2, [1, 2, 3]) == [[1, 2], [3]]
    assert list(chunks(2, iter([1, 2, 3]))) == [[1, 2], [3]]


def test_chunks_with_step():
    assert list(chunks(3, 2, range(6))) == [[0, 1, 2], [2, 3, 4], [4, 5]]
    assert lchunks(2, 3, range(7)) == [[0, 1], [3, 4], [6]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        chunks(0, [1, 2, 3])


def test_partition_by():
    assert [list(g) for g in partition_by(lambda x: x % 2, [1, 3, 2, 4, 5, 7])] == [
        [1, 3], [2, 4], [5, 7]
    ]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5, 7]) == [
        [1, 3], [2, 4], [5, 7]
    ]
    assert lpartition_by(lambda x: x, []) == []


# ---------- with_prev / with_next / pairwise ----------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]
    assert list(with_prev([])) == []


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]
    assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]
    assert list(with_next([])) == []


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------- reductions / sums ----------

def test_reductions():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(reductions(operator.mul, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]
    assert list(reductions(operator.mul, [2, 3], 4)) == [8, 24]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lreductions(operator.add, [1, 2, 3], 10) == [11, 13, 16]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]
    assert list(sums([])) == []


def test_lsums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lsums([1, 2, 3], 10) == [11, 13, 16]
    assert lsums([]) == []
