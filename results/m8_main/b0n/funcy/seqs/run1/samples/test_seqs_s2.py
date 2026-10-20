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


# ---------- repeatedly / iterate ----------

def test_repeatedly_infinite():
    gen = repeatedly(lambda: 7)
    assert take(3, gen) == [7, 7, 7]


def test_repeatedly_infinite_with_argument():
    # first argument is a function f, but here we pass a callable taking no args
    counter = itertools.count()
    gen = repeatedly(lambda: next(counter))
    assert take(4, gen) == [0, 1, 2, 3]


def test_repeatedly_with_n():
    assert list(repeatedly(lambda: 'x', 3)) == ['x', 'x', 'x']
    assert list(repeatedly(lambda: 'x', 0)) == []


def test_repeatedly_takes_none_as_finite():
    # n=None -> repeat(None, None) -> repeat with None count -> infinite
    # Actually repeat(None, None) raises TypeError, so check behavior
    with pytest.raises(TypeError):
        list(take(3, repeatedly(lambda: 1, None)))


def test_iterate():
    gen = iterate(lambda x: x * 2, 1)
    assert take(5, gen) == [1, 2, 4, 8, 16]


def test_iterate_with_zero():
    gen = iterate(lambda x: x + 1, 0)
    assert take(3, gen) == [0, 1, 2]


# ---------- take / drop ----------

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(0, [1, 2]) == []
    assert take(10, [1, 2]) == [1, 2]
    assert take(3, iter([1, 2, 3, 4])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []
    assert list(drop(1, iter([1, 2, 3]))) == [2, 3]


# ---------- first / second / nth / last ----------

def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([5, 6])) == 5
    assert first(x for x in []) is None


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None
    assert second(iter([4, 5, 6])) == 5


def test_nth():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20]) is None
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(5, iter([10])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(x for x in []) is None


# ---------- rest / butlast ----------

def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []
    assert list(butlast(iter([1, 2, 3, 4]))) == [1, 2, 3]


# ---------- shuffled ----------

def test_shuffled():
    result = shuffled([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert len(result) == 5


def test_shuffled_empty():
    assert shuffled([]) == []


# ---------- ilen ----------

def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3, 4])) == 4
    assert ilen(x for x in range(5)) == 5


# ---------- lmap / map ----------

def test_lmap():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    # string as a mapper -> re_finder
    assert lmap('a', ['abc', 'bcd', 'xyz']) == ['a', None, None]


def test_lmap_multiple_seqs():
    assert lmap(lambda a, b: a + b, [1, 2], [10, 20]) == [11, 22]


def test_lmap_int_mapper():
    assert lmap(0, [(1, 2), (3, 4)]) == [1, 3]


def test_map():
    assert list(map(lambda x: x + 1, [1, 2, 3])) == [2, 3, 4]
    is_iter = map(lambda x: x, [1, 2])
    assert not isinstance(is_iter, list)
    assert list(is_iter) == [1, 2]


# ---------- lfilter / filter ----------

def test_lfilter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
    # int as predicate -> itemgetter -> truthy filter
    assert lfilter(1, [(1, 0), (0, 1), (1, 1)]) == [(1, 0), (1, 1)]


def test_filter():
    assert list(filter(lambda x: x > 2, [1, 2, 3, 4])) == [3, 4]
    res = filter(lambda x: x, [0, 1, 2])
    assert not isinstance(res, list)
    assert list(res) == [1, 2]


def test_filter_string_pred():
    result = list(filter('x', ['xyz', 'abc', 'xx']))
    assert result == ['xyz', 'xx']


# ---------- lremove / remove ----------

def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]


def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_with_int_pred():
    assert list(remove(0, [(1, 2), (0, 3), (4, 5)])) == [(1, 2), (4, 5)]


# ---------- lkeep / keep ----------

def test_lkeep_two_args():
    assert lkeep(lambda x: x if x > 2 else None, [1, 2, 3, 4]) == [3, 4]


def test_lkeep_one_arg():
    assert lkeep([0, 1, 2, 3, None, '', 'x']) == [1, 2, 3, 'x']


def test_keep_two_args():
    assert list(keep(lambda x: x if x > 1 else 0, [1, 2, 3])) == [2, 3]


def test_keep_one_arg():
    result = keep([0, 1, 2, None, 'a'])
    assert not isinstance(result, list)
    assert list(result) == [1, 2, 'a']


# ---------- without / lwithout ----------

def test_without():
    assert list(without([1, 2, 3, 2, 1], 1, 3)) == [2, 2]
    assert list(without([1, 2, 3])) == [1, 2, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 2, 1], 1, 3) == [2, 2]
    assert lwithout(['a', 'b', 'c'], 'b') == ['a', 'c']


# ---------- lconcat / lcat ----------

def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]
    assert lconcat() == []


def test_lcat():
    assert lcat([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
    assert lcat([]) == []


# ---------- flatten / lflatten ----------

def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


def test_flatten_with_custom_follow():
    # follow returns False for strings, so strings not unpacked
    result = list(flatten(['ab', ['cd']], follow=lambda x: isinstance(x, (list, tuple))))
    assert result == ['ab', 'cd']


def test_lflatten():
    assert lflatten([1, [2, [3]]]) == [1, 2, 3]
    assert lflatten([]) == []


# ---------- lmapcat / mapcat ----------

def test_lmapcat():
    assert lmapcat(lambda x: [x, x * 2], [1, 2, 3]) == [1, 2, 2, 4, 3, 6]


def test_mapcat():
    result = mapcat(lambda x: [x, -x], [1, 2])
    assert not isinstance(result, list)
    assert list(result) == [1, -1, 2, -2]


# ---------- interleave / interpose ----------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]


def test_interleave_uneven():
    assert list(interleave([1, 2, 3], [4, 5])) == [1, 4, 2, 5]


def test_interleave_single():
    assert list(interleave([1, 2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [1])) == [1]
    assert list(interpose(0, [])) == []


# ---------- takewhile / dropwhile ----------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]


def test_takewhile_one_arg():
    assert list(takewhile([1, 1, 0, 1, 1])) == [1, 1]


def test_takewhile_with_int_pred():
    assert list(takewhile(0, [(1, 2), (3, 4), (0, 5), (6, 7)])) == [(1, 2), (3, 4)]


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]


def test_dropwhile_one_arg():
    assert list(dropwhile([0, 0, 1, 0])) == [1, 0]


# ---------- ldistinct / distinct ----------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2])) == [1, 2, 3]
    assert list(distinct([])) == []


def test_distinct_with_key():
    result = list(distinct(['a', 'A', 'b', 'B'], key=str.lower))
    assert result == ['a', 'b']


def test_ldistinct():
    assert ldistinct([1, 2, 1, 3]) == [1, 2, 3]
    assert ldistinct(['a', 'A', 'b'], key=str.lower) == ['a', 'b']


# ---------- split / lsplit ----------

def test_split():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert list(yes) == [2, 4, 6]
    assert list(no) == [1, 3, 5]


def test_split_interleaved_access():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    # consume alternately
    assert next(no) == 1
    assert next(yes) == 2
    assert next(no) == 3
    assert next(yes) == 4
    with pytest.raises(StopIteration):
        next(yes)


def test_lsplit():
    yes, no = lsplit(lambda x: x > 2, [1, 2, 3, 4, 1])
    assert yes == [3, 4]
    assert no == [1, 2, 1]


# ---------- split_at / lsplit_at ----------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4, 5])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4, 5]


def test_split_at_with_iterator():
    a, b = split_at(2, iter([1, 2, 3, 4]))
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])
    assert lsplit_at(5, [1, 2]) == ([1, 2], [])


# ---------- split_by / lsplit_by ----------

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 2])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 2]


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 2]) == ([1, 2], [3, 2])
    assert lsplit_by(lambda x: x > 0, [-1, -2, 1]) == ([], [-1, -2, 1])


# ---------- group_by / group_by_keys / group_values ----------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_returns_defaultdict():
    result = group_by(len, ['a', 'bb', 'cc', 'ddd'])
    assert result['missing'] == []
    assert dict(result) == {1: ['a'], 2: ['bb', 'cc'], 3: ['ddd']}


def test_group_by_keys():
    data = [('a', 1), ('b', 2)]
    result = group_by_keys(lambda x: x[0], data)
    assert dict(result) == {'a': [('a', 1)], 'b': [('b', 2)]}


def test_group_by_keys_multiple():
    data = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    result = group_by_keys(lambda x: x, data)
    assert dict(result) == {
        'a': [('a', 'b'), ('c', 'a')],
        'b': [('a', 'b'), ('b', 'c')],
        'c': [('b', 'c'), ('c', 'a')],
    }


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}


# ---------- count_by / count_reps ----------

def test_count_by():
    result = count_by(len, ['a', 'bb', 'cc', 'ddd'])
    assert dict(result) == {1: 1, 2: 2, 3: 1}


def test_count_reps():
    result = count_reps('abacb')
    assert dict(result) == {'a': 2, 'b': 2, 'c': 1}


# ---------- partition / chunks (and error handling) ----------

def test_partition_sequence():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_partition_with_step_sequence():
    assert lpartition(2, 3, list(range(7))) == [[0, 1], [3, 4]]


def test_partition_iterator():
    assert lpartition(2, iter(range(5))) == [[0, 1], [2, 3]]


def test_partition_iterator_with_step():
    assert lpartition(2, 3, iter(range(7))) == [[0, 1], [3, 4]]


def test_partition_step_less_than_n():
    # overlapping partitions
    assert lpartition(3, 1, [1, 2, 3, 4]) == [[1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_partition_step_less_than_n_iterator():
    assert lpartition(3, 1, iter([1, 2, 3, 4])) == [[1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_partition_invalid_zero_n():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2, 3]))


def test_partition_invalid_zero_step():
    with pytest.raises(ValueError):
        list(partition(2, 0, [1, 2, 3]))


def test_chunks_sequence():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]


def test_chunks_with_step():
    assert lchunks(2, 3, list(range(7))) == [[0, 1], [3, 4], [6]]


def test_chunks_iterator():
    assert lchunks(2, iter(range(5))) == [[0, 1], [2, 3], [4]]


def test_chunks_iterator_with_step():
    assert lchunks(2, 3, iter(range(7))) == [[0, 1], [3, 4], [6]]


def test_chunks_short_seq():
    assert lchunks(3, [1, 2]) == [[1, 2]]


# ---------- partition_by / lpartition_by ----------

def test_partition_by():
    result = partition_by(lambda x: x < 3, [1, 2, 3, 4, 1, 2])
    chunks_ = [list(c) for c in result]
    assert chunks_ == [[1, 2], [3, 4], [1, 2]]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5]) == [[1, 3], [2, 4], [5]]


# ---------- with_prev / with_next / pairwise ----------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]


def test_with_prev_empty():
    assert list(with_prev([])) == []


def test_with_next_empty():
    assert list(with_next([])) == []


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------- reductions / lreductions / sums / lsums ----------

def test_reductions():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(lambda a, b: a + b, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_mul():
    assert list(reductions(operator.mul, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]
    assert lreductions(operator.add, [1, 2, 3], 5) == [6, 8, 11]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]


def test_lsums():
    assert lsums([1, 2, 3]) == [1, 3, 6]
    assert lsums([1, 2, 3], 100) == [101, 103, 106]
