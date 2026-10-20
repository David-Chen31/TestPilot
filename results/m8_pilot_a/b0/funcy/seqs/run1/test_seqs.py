import pytest

from funcy.seqs import (
    repeatedly, iterate, take, drop, first, second, nth, last, rest, butlast,
    shuffled, ilen, lmap, lfilter, map, filter, lremove, remove, lkeep, keep,
    without, lwithout, lconcat, lcat, flatten, lflatten, lmapcat, mapcat,
    interleave, interpose, takewhile, dropwhile, ldistinct, distinct, split,
    lsplit, split_at, lsplit_at, split_by, lsplit_by, group_by, group_by_keys,
    group_values, count_by, count_reps, partition, lpartition, chunks, lchunks,
    partition_by, lpartition_by, with_prev, with_next, pairwise, reductions,
    lreductions, sums, lsums, lzip
)
from funcy.primitives import EMPTY


# ---------- generators / infinite ----------

def test_repeatedly_infinite():
    it = repeatedly(lambda: 7)
    assert take(3, it) == [7, 7, 7]

def test_repeatedly_n():
    calls = []
    def f():
        calls.append(1)
        return len(calls)
    assert list(repeatedly(f, 3)) == [1, 2, 3]
    assert list(repeatedly(f, 0)) == []

def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


# ---------- accessors ----------

def test_take():
    assert take(3, [1, 2, 3, 4]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []

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
    assert nth(0, [10, 20]) == 10
    assert nth(1, (10, 20)) == 20
    assert nth(5, [1, 2]) is None
    # iterator path (TypeError)
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(5, iter([1, 2])) is None

def test_last():
    assert last([1, 2, 3]) == 3
    assert last((1, 2, 3)) == 3
    assert last([]) is None
    # iterator path
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None

def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []

def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []
    assert list(butlast(iter([1, 2, 3]))) == [1, 2]

def test_shuffled():
    # no randomness dependence on order, just content
    result = shuffled([1, 2, 3, 4])
    assert sorted(result) == [1, 2, 3, 4]
    assert shuffled([]) == []

def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3, 4, 5])) == 5


# ---------- map/filter family ----------

def test_lmap_basic():
    assert lmap(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert lmap(lambda a, b: a + b, [1, 2], [10, 20]) == [11, 22]

def test_lmap_derivers():
    assert lmap('name', [{'name': 'a'}, {'name': 'b'}]) == ['a', 'b']
    assert lmap(0, [(1, 2), (3, 4)]) == [1, 3]
    assert lmap(slice(0, 2), [[1, 2, 3], [4, 5, 6]]) == [[1, 2], [4, 5]]

def test_map_basic():
    assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]
    assert list(map('x', [{'x': 1}, {'x': 2}])) == [1, 2]

def test_lfilter_basic():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]

def test_lfilter_derivers():
    assert lfilter(None, [0, 1, 2, '', 'a']) == [1, 2, 'a']
    assert lfilter('a', ['abc', 'bcd', 'cde']) == ['abc']
    assert lfilter(1, [(1, 2), (3, 4)]) == [(3, 4)]

def test_filter_basic():
    assert list(filter(lambda x: x > 1, [1, 2, 3])) == [2, 3]
    assert list(filter(None, [0, 1, '', 'x'])) == [1, 'x']

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]

def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]
    assert lremove(None, [0, 1, None, 2]) == [0, None]

def test_keep():
    assert list(keep(lambda x: x * 2, [1, 0, 2, 0])) == [2, 4]
    assert list(keep([0, 1, None, 2, '', 3])) == [1, 2, 3]

def test_lkeep():
    assert lkeep(lambda x: x * 2, [1, 0, 2, 0]) == [2, 4]
    assert lkeep([0, 1, None, 2, '', 3]) == [1, 2, 3]

def test_without():
    assert list(without([1, 2, 3, 4, 5], 2, 4)) == [1, 3, 5]
    assert list(without([], 1)) == []

def test_lwithout():
    assert lwithout([1, 2, 3, 4, 5], 2, 4) == [1, 3, 5]


# ---------- concat / flatten ----------

def test_lconcat():
    assert lconcat([1, 2], [3, 4], (5,)) == [1, 2, 3, 4, 5]
    assert lconcat() == []

def test_lcat():
    assert lcat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert lcat([]) == []

def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]
    assert list(flatten([])) == []
    # custom follow
    assert list(flatten([1, [2, 3]], follow=lambda x: False)) == [1, [2, 3]]

def test_lflatten():
    assert lflatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert lflatten([1, [2, 3]], follow=lambda x: False) == [1, [2, 3]]

def test_mapcat():
    assert list(mapcat(lambda x: [x, x], [1, 2, 3])) == [1, 1, 2, 2, 3, 3]
    assert list(mapcat(lambda a, b: [a, b], [1, 2], [3, 4])) == [1, 3, 2, 4]

def test_lmapcat():
    assert lmapcat(lambda x: [x, x], [1, 2, 3]) == [1, 1, 2, 2, 3, 3]

def test_interleave():
    assert list(interleave([1, 2], [3, 4])) == [1, 3, 2, 4]
    assert list(interleave([1, 2, 3], [4, 5])) == [1, 4, 2, 5]
    assert list(interleave()) == []

def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [])) == []
    assert list(interpose(0, [1])) == [1]


# ---------- takewhile / dropwhile ----------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]
    # one arg version - truthy
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]

def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]
    assert list(dropwhile([1, 2, 0, 3])) == [0, 3]


# ---------- distinct ----------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2])) == [1, 2, 3]
    assert list(distinct([])) == []

def test_distinct_key():
    assert list(distinct([1, 2, 3, 4], key=lambda x: x % 2)) == [1, 2]
    assert list(distinct(['a', 'A', 'b'], key=str.lower)) == ['a', 'b']

def test_ldistinct():
    assert ldistinct([1, 2, 1, 3, 2]) == [1, 2, 3]
    assert ldistinct([1, 2, 3, 4], key=lambda x: x % 2) == [1, 2]


# ---------- split ----------

def test_split_lazy():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
    assert list(yes) == [2, 4]
    assert list(no) == [1, 3, 5]

def test_lsplit():
    assert lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]) == ([2, 4], [1, 3, 5])
    assert lsplit(None, [0, 1, 2]) == ([1, 2], [0])

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]
    a, b = split_at(0, [1, 2])
    assert list(a) == []
    assert list(b) == [1, 2]

def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert lsplit_at(10, [1, 2]) == ([1, 2], [])

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 4]

def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 1, 4]) == ([1, 2], [3, 1, 4])


# ---------- grouping / counting ----------

def test_group_by():
    assert dict(group_by(lambda x: x % 2, [1, 2, 3, 4, 5])) == {1: [1, 3, 5], 0: [2, 4]}

def test_group_by_deriver():
    data = [{'k': 1, 'v': 'a'}, {'k': 2, 'v': 'b'}, {'k': 1, 'v': 'c'}]
    assert dict(group_by('k', data)) == {
        1: [{'k': 1, 'v': 'a'}, {'k': 1, 'v': 'c'}],
        2: [{'k': 2, 'v': 'b'}],
    }

def test_group_by_keys():
    data = [(1, ['a', 'b']), (2, ['b'])]
    result = dict(group_by_keys(1, data))
    assert result == {'a': [(1, ['a', 'b'])], 'b': [(1, ['a', 'b']), (2, ['b'])]}

def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}

def test_count_by():
    assert dict(count_by(lambda x: x % 2, [1, 2, 3, 4, 5])) == {1: 3, 0: 2}

def test_count_reps():
    assert dict(count_reps(['a', 'b', 'a', 'a'])) == {'a': 3, 'b': 1}
    assert dict(count_reps([])) == {}


# ---------- partition / chunks ----------

def test_partition_seq():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]
    assert lpartition(3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2, 3], [4, 5, 6]]

def test_partition_step():
    assert lpartition(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5]]

def test_partition_iter():
    assert lpartition(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]

def test_partition_iter_step():
    assert lpartition(2, 3, iter([1, 2, 3, 4, 5, 6, 7])) == [[1, 2], [4, 5]]
    assert lpartition(2, 1, iter([1, 2, 3, 4])) == [[1, 2], [2, 3], [3, 4]]

def test_chunks_seq():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]

def test_chunks_iter():
    assert lchunks(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]
    assert lchunks(3, iter([1, 2])) == [[1, 2]]

def test_chunks_step():
    assert lchunks(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5], [7]]

def test_partition_zero_raises():
    with pytest.raises(ValueError):
        partition(0, [1, 2, 3])

def test_partition_zero_step_raises():
    with pytest.raises(ValueError):
        partition(2, 0, [1, 2, 3])


def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x // 3, [1, 2, 4, 5, 7])]
    assert result == [[1, 2], [4, 5], [7]]

def test_lpartition_by():
    assert lpartition_by(lambda x: x // 3, [1, 2, 4, 5, 7]) == [[1, 2], [4, 5], [7]]


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


# ---------- lzip ----------

def test_lzip():
    assert lzip([1, 2], [3, 4]) == [(1, 3), (2, 4)]
    assert lzip([1, 2, 3], [4]) == [(1, 4)]

def test_lzip_strict():
    assert lzip([1, 2], [3, 4], strict=True) == [(1, 3), (2, 4)]
    with pytest.raises(ValueError):
        lzip([1, 2, 3], [4, 5], strict=True)
    with pytest.raises(ValueError):
        lzip([1, 2], [3, 4, 5], strict=True)

def test_lzip_strict_non_sized():
    # Generator / iterator path
    assert lzip(iter([1, 2]), iter([3, 4]), strict=True) == [(1, 3), (2, 4)]
    with pytest.raises(ValueError):
        lzip(iter([1, 2, 3]), iter([4, 5]), strict=True)
    with pytest.raises(ValueError):
        lzip(iter([1, 2]), iter([3, 4, 5]), strict=True)


# ---------- reductions / sums ----------

def test_reductions():
    assert list(reductions(lambda a, b: a + b, [1, 2, 3, 4])) == [1, 3, 6, 10]
    # with acc
    assert list(reductions(lambda a, b: a + b, [1, 2, 3], 0)) == [1, 3, 6]
    assert list(reductions(lambda a, b: a + b, [1, 2, 3], 10)) == [11, 13, 16]

def test_reductions_empty():
    assert list(reductions(lambda a, b: a + b, [])) == []
    assert list(reductions(lambda a, b: a + b, [], 5)) == []

def test_lreductions():
    assert lreductions(lambda a, b: a + b, [1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lreductions(lambda a, b: a - b, [1, 2, 3, 4], 10) == [9, 7, 4, 0]

def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]
    assert list(sums([])) == []

def test_lsums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lsums([1, 2, 3], 10) == [11, 13, 16]
