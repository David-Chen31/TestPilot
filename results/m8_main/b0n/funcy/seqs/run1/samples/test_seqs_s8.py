import pytest
from collections import defaultdict

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
    it = repeatedly(lambda: 5)
    assert take(3, it) == [5, 5, 5]


def test_repeatedly_count():
    it = repeatedly(lambda: 5, 3)
    assert list(it) == [5, 5, 5]


def test_repeatedly_zero():
    assert list(repeatedly(lambda: 1, 0)) == []


def test_iterate():
    it = iterate(lambda x: x * 2, 1)
    assert take(5, it) == [1, 2, 4, 8, 16]


# ---------- take / drop ----------

def test_take():
    assert take(3, [1, 2, 3, 4]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []


def test_take_iterator():
    assert take(2, iter([1, 2, 3])) == [1, 2]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(10, [1, 2])) == []
    assert list(drop(0, [1, 2])) == [1, 2]


def test_drop_iterator():
    assert list(drop(1, iter([1, 2, 3]))) == [2, 3]


# ---------- first / second / nth / last ----------

def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([7, 8])) == 7


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
    assert nth(5, iter([10, 20])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None


# ---------- rest / butlast ----------

def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3])) == [1, 2]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


def test_butlast_iterator():
    assert list(butlast(iter([1, 2, 3, 4]))) == [1, 2, 3]


# ---------- shuffled ----------

def test_shuffled_same_items():
    result = shuffled([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert isinstance(result, list)


def test_shuffled_empty():
    assert shuffled([]) == []


# ---------- ilen ----------

def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3, 4])) == 4


# ---------- map / filter / lmap / lfilter ----------

def test_map():
    assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]


def test_map_multiple():
    assert list(map(lambda a, b: a + b, [1, 2], [3, 4])) == [4, 6]


def test_map_string_key():
    data = [{'a': 1}, {'a': 2}]
    assert list(map('a', data)) == [1, 2]


def test_lmap():
    assert lmap(lambda x: x + 1, [1, 2]) == [2, 3]


def test_lmap_string():
    data = [{'a': 1}, {'a': 2}]
    assert lmap('a', data) == [1, 2]


def test_filter():
    assert list(filter(lambda x: x > 1, [1, 2, 3])) == [2, 3]


def test_lfilter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]


def test_filter_none_predicates_truthy():
    assert list(filter(None, [0, 1, 2, 0, 3])) == [1, 2, 3]


# ---------- remove / lremove ----------

def test_remove():
    assert list(remove(lambda x: x > 1, [1, 2, 3])) == [1]


def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]


def test_lremove_by_pred_int():
    # make_pred(int) -> itemgetter(int)
    data = [(0, 1), (1, 0), (2, 3)]
    assert lremove(1, data) == [(0, 1), (2, 3)]


# ---------- keep / lkeep ----------

def test_keep_with_seq():
    assert list(keep(lambda x: x if x > 1 else None, [1, 2, 3])) == [2, 3]


def test_keep_one_arg():
    assert list(keep([0, 1, None, 2, '', 3])) == [1, 2, 3]


def test_lkeep_with_seq():
    assert lkeep(lambda x: x if x % 2 else 0, [1, 2, 3]) == [1, 3]


def test_lkeep_one_arg():
    assert lkeep([0, 1, None, 2]) == [1, 2]


# ---------- without / lwithout ----------

def test_without():
    assert list(without([1, 2, 3, 4], 2, 4)) == [1, 3]


def test_without_multiple():
    assert list(without([1, 2, 3], 5)) == [1, 2, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 4], 1, 3) == [2, 4]


# ---------- concat / lconcat / cat / lcat ----------

def test_lconcat():
    assert lconcat([1, 2], [3, 4]) == [1, 2, 3, 4]


def test_lconcat_empty():
    assert lconcat() == []


def test_lcat():
    assert lcat([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_lcat_empty():
    assert lcat([]) == []


# ---------- flatten / lflatten ----------

def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


def test_flatten_strings_not_followed():
    # strings are not seqcont
    assert list(flatten(['ab', ['c']])) == ['ab', 'c']


def test_flatten_custom_follow():
    assert list(flatten([1, 2, 3], follow=lambda x: x == 2)) == [1, 2, 3]


def test_lflatten():
    assert lflatten([1, [2, [3]]]) == [1, 2, 3]


def test_flatten_empty():
    assert list(flatten([])) == []


# ---------- mapcat / lmapcat ----------

def test_mapcat():
    assert list(mapcat(lambda x: [x, x], [1, 2])) == [1, 1, 2, 2]


def test_lmapcat():
    assert lmapcat(lambda x: range(x), [2, 3]) == [0, 1, 0, 1, 2]


# ---------- interleave / interpose ----------

def test_interleave():
    assert list(interleave([1, 2], [3, 4])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert list(interleave([1, 2, 3], [4])) == [1, 4]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]


def test_interpose_empty():
    assert list(interpose(0, [])) == []


# ---------- takewhile / dropwhile ----------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]


def test_takewhile_one_arg():
    assert list(takewhile([1, 2, 0, 4])) == [1, 2]


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]


def test_dropwhile_one_arg():
    assert list(dropwhile([1, 2, 0, 4])) == [0, 4]


def test_takewhile_pred_int():
    data = [(1, 'a'), (1, 'b'), (0, 'c')]
    assert list(takewhile(0, data)) == [(1, 'a'), (1, 'b')]


# ---------- distinct / ldistinct ----------

def test_distinct():
    assert list(distinct([1, 1, 2, 3, 2, 1])) == [1, 2, 3]


def test_distinct_key():
    items = [{'a': 1}, {'a': 1}, {'a': 2}]
    assert list(distinct(items, key='a')) == [{'a': 1}, {'a': 2}]


def test_ldistinct():
    assert ldistinct([1, 2, 2, 3, 3]) == [1, 2, 3]


def test_ldistinct_key():
    items = [{'a': 1}, {'a': 1}, {'a': 2}]
    assert ldistinct(items, key='a') == [{'a': 1}, {'a': 2}]


# ---------- split / lsplit ----------

def test_split():
    yes, no = split(lambda x: x > 2, [1, 2, 3, 4])
    assert list(yes) == [3, 4]
    assert list(no) == [1, 2]


def test_split_interleaved():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert list(no) == [1, 3]
    assert list(yes) == [2, 4]


def test_lsplit():
    yes, no = lsplit(lambda x: x > 2, [1, 2, 3, 4])
    assert yes == [3, 4]
    assert no == [1, 2]


def test_lsplit_pred_int():
    yes, no = lsplit(1, [(0, 'a'), (1, 'b'), (1, 'c')])
    assert yes == [(1, 'b'), (1, 'c')]
    assert no == [(0, 'a')]


# ---------- split_at / lsplit_at ----------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_split_at_beyond():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3]) == ([1, 2], [3])


def test_lsplit_at_zero():
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])


# ---------- split_by / lsplit_by ----------

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1]


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 4]) == ([1, 2], [3, 4])


def test_lsplit_by_all():
    assert lsplit_by(lambda x: x < 10, [1, 2]) == ([1, 2], [])


# ---------- group_by / group_by_keys / group_values ----------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_string_key():
    data = [{'a': 1}, {'a': 2}, {'a': 1}]
    result = group_by('a', data)
    assert len(result[1]) == 2
    assert len(result[2]) == 1


def test_group_by_is_defaultdict():
    result = group_by(lambda x: x, [1])
    assert isinstance(result, defaultdict)


def test_group_by_keys():
    data = [('a', 'b'), ('b', 'c')]
    result = group_by_keys(lambda p: p, data)
    assert dict(result) == {'a': [('a', 'b')], 'b': [('a', 'b'), ('b', 'c')], 'c': [('b', 'c')]}


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}


# ---------- count_by / count_reps ----------

def test_count_by():
    result = count_by(lambda x: x % 2, [1, 2, 3, 4])
    assert dict(result) == {1: 2, 0: 2}


def test_count_by_string():
    result = count_by('a', [{'a': 1}, {'a': 1}, {'a': 2}])
    assert dict(result) == {1: 2, 2: 1}


def test_count_reps():
    result = count_reps([1, 1, 2, 3, 3, 3])
    assert dict(result) == {1: 2, 2: 1, 3: 3}


# ---------- partition / lpartition ----------

def test_partition_seq():
    assert lpartition(2, [1, 2, 3, 4, 5, 6]) == [[1, 2], [3, 4], [5, 6]]


def test_partition_seq_drops_tail():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_partition_step():
    assert lpartition(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5]]


def test_partition_iter_step_equal():
    assert lpartition(3, iter([1, 2, 3, 4, 5, 6])) == [[1, 2, 3], [4, 5, 6]]


def test_partition_iter_step_less():
    result = lpartition(3, 1, iter(range(6)))
    assert result == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_partition_iter_step_greater():
    result = lpartition(2, 4, iter(range(8)))
    assert result == [[0, 1], [4, 5]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2, 3]))


def test_partition_step_zero_raises():
    with pytest.raises(ValueError):
        list(partition(2, 0, [1, 2, 3]))


# ---------- chunks / lchunks ----------

def test_chunks_seq():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]


def test_chunks_exact():
    assert lchunks(2, [1, 2, 3, 4]) == [[1, 2], [3, 4]]


def test_chunks_step():
    assert lchunks(2, 3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2], [4, 5], [7]]


def test_chunks_iter():
    assert lchunks(2, iter([1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]


def test_chunks_iter_step_less():
    assert lchunks(3, 1, iter(range(5))) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        list(chunks(0, [1, 2]))


# ---------- partition_by / lpartition_by ----------

def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x % 2, [1, 3, 2, 4, 5])]
    assert result == [[1, 3], [2, 4], [5]]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5]) == [[1, 3], [2, 4], [5]]


def test_lpartition_by_empty():
    assert lpartition_by(lambda x: x, []) == []


# ---------- with_prev / with_next / pairwise ----------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2], fill=0)) == [(1, 2), (2, 0)]


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_short():
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------- reductions / lreductions / sums / lsums ----------

def test_reductions():
    assert list(reductions(lambda a, b: a + b, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(lambda a, b: a + b, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_add_uses_accumulate():
    # f is operator.add -> uses accumulate
    import operator
    assert list(reductions(operator.add, [1, 2, 3])) == [1, 3, 6]


def test_lreductions():
    assert lreductions(lambda a, b: a * b, [1, 2, 3, 4]) == [1, 2, 6, 24]


def test_lreductions_with_acc():
    assert lreductions(lambda a, b: a * b, [2, 3], 2) == [4, 12]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]


def test_lsums():
    assert lsums([1, 2, 3]) == [1, 3, 6]


def test_lsums_with_acc():
    assert lsums([1, 2, 3], 100) == [101, 103, 106]


def test_reductions_empty():
    assert list(reductions(lambda a, b: a + b, [])) == []
