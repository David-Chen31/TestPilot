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


# ---------------------------------- basics ----------------------------------

def test_repeatedly_infinite():
    g = repeatedly(lambda: 7)
    assert take(3, g) == [7, 7, 7]


def test_repeatedly_finite():
    assert repeatedly(lambda: 1, 3) == [1, 1, 1] or list(repeatedly(lambda: 1, 3)) == [1, 1, 1]
    assert take(5, repeatedly(lambda: 1, 3)) == [1, 1, 1]


def test_repeatedly_side_effects():
    counter = iter(range(100))
    assert take(3, repeatedly(lambda: next(counter))) == [0, 1, 2]


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]


def test_iterate_lazy():
    it = iterate(lambda x: x + 1, 0)
    assert next(it) == 0
    assert next(it) == 1


def test_take():
    assert take(3, [1, 2, 3, 4]) == [1, 2, 3]
    assert take(0, [1, 2]) == []
    assert take(10, [1, 2]) == [1, 2]
    assert take(2, iter([1, 2, 3])) == [1, 2]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []
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
    assert nth(5, [10, 20]) is None
    # iterators path (TypeError branch)
    assert nth(1, iter([1, 2, 3])) == 2
    assert nth(5, iter([1, 2])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
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


def test_shuffled_preserves_items():
    src = [1, 2, 3, 4, 5]
    out = shuffled(src)
    assert sorted(out) == src
    assert out is not src
    assert shuffled([]) == []


def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter([1, 2, 3, 4])) == 4
    assert ilen("abcd") == 4


# ------------------------------- map / filter -------------------------------

def test_lmap_and_map():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert list(map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]
    # multiple seqs
    assert lmap(lambda a, b: a + b, [1, 2], [3, 4]) == [4, 6]


def test_lmap_string_mapper():
    # string maps to regex finder
    import re
    items = ['123', 'abc', '456']
    result = lmap(r'\d+', items)
    assert result == ['123', '', '456']


def test_lmap_int_itemgetter():
    data = [(1, 2), (3, 4)]
    assert lmap(0, data) == [1, 3]
    assert lmap(1, data) == [2, 4]


def test_lfilter_and_filter():
    assert lfilter(lambda x: x > 1, [1, 2, 3]) == [2, 3]
    assert list(filter(lambda x: x > 1, [1, 2, 3])) == [2, 3]


def test_filter_pred_variants():
    data = ['a', 'bb', 'ccc']
    assert lfilter(2, data) == ['ccc']  # itemgetter index 2 works only on strings
    d = {'x': 1, 'y': 2}
    assert lfilter(d, [('x', 1), ('z', 9)]) == [('x', 1)]
    s = {1, 2}
    assert lfilter(s, [1, 3]) == [1]


def test_remove_and_lremove():
    assert list(remove(lambda x: x > 1, [1, 2, 3])) == [1]
    assert lremove(lambda x: x > 1, [1, 2, 3]) == [1]


def test_keep():
    assert list(keep(lambda x: x if x > 1 else None, [1, 2, 3])) == [2, 3]
    assert lkeep(lambda x: x if x > 1 else None, [1, 2, 3]) == [2, 3]


def test_keep_one_arg():
    # one-arg version just filters truthy values
    assert list(keep([0, 1, 2, 0, 3])) == [1, 2, 3]
    assert lkeep([None, 0, 'a', '', 'b']) == ['a', 'b']


def test_without():
    assert list(without([1, 2, 3, 4], 2, 4)) == [1, 3]
    assert list(without([1, 2, 3])) == [1, 2, 3]
    assert lwithout([1, 2, 1, 3], 1) == [2, 3]


# ------------------------------ concat / flatten ----------------------------

def test_lconcat():
    assert lconcat([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert lconcat() == []
    assert lconcat([1], [], [2]) == [1, 2]


def test_lcat():
    assert lcat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert lcat([]) == []


def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]
    assert list(flatten([])) == []
    assert list(flatten([1, 2])) == [1, 2]


def test_flatten_custom_follow():
    # treat only lists as containers
    result = list(flatten([1, (2, 3), [4]], follow=lambda x: isinstance(x, list)))
    assert result == [1, (2, 3), 4]


def test_lflatten():
    assert lflatten([1, [2, [3]]]) == [1, 2, 3]


def test_mapcat_lmapcat():
    assert list(mapcat(lambda x: [x, x], [1, 2])) == [1, 1, 2, 2]
    assert lmapcat(lambda x: [x, x], [1, 2]) == [1, 1, 2, 2]


# -------------------------- interleave / interpose --------------------------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]
    # shortest wins
    assert list(interleave([1, 2], [3, 4, 5])) == [1, 3, 2, 4]
    assert list(interleave([])) == []


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [1])) == [1]
    assert list(interpose(0, [])) == []


# --------------------------- takewhile / dropwhile --------------------------

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 1])) == [1, 2]
    assert list(takewhile(lambda x: x < 3, [5, 1])) == []


def test_takewhile_one_arg():
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]
    assert list(takewhile([0, 1])) == []


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1])) == [3, 1]
    assert list(dropwhile(lambda x: x < 3, [5, 6])) == [5, 6]


def test_dropwhile_one_arg():
    assert list(dropwhile([0, 0, 1, 2])) == [1, 2]
    assert list(dropwhile([1, 2])) == [1, 2]


# --------------------------------- distinct ---------------------------------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2])) == [1, 2, 3]
    assert list(distinct([])) == []


def test_ldistinct():
    assert ldistinct([1, 2, 1, 3]) == [1, 2, 3]


def test_distinct_with_key():
    data = ['a', 'A', 'b', 'B', 'a']
    assert list(distinct(data, key=str.lower)) == ['a', 'b']


def test_ldistinct_with_key():
    data = [(1, 'a'), (1, 'b'), (2, 'c')]
    assert ldistinct(data, key=0) == [(1, 'a'), (2, 'c')]


# ----------------------------------- split ----------------------------------

def test_split():
    yes, no = split(lambda x: x % 2 == 0, range(10))
    assert list(yes) == [0, 2, 4, 6, 8]
    assert list(no) == [1, 3, 5, 7, 9]


def test_split_interleaved_consumption():
    yes, no = split(lambda x: x % 2 == 0, [0, 1, 2, 3, 4])
    assert next(yes) == 0
    assert next(no) == 1
    assert list(yes) == [2, 4]
    assert list(no) == [3]


def test_lsplit():
    yes, no = lsplit(lambda x: x > 0, [-1, 0, 1, 2])
    assert yes == [1, 2]
    assert no == [-1, 0]


def test_split_at():
    a, b = split_at(3, [1, 2, 3, 4, 5])
    assert list(a) == [1, 2, 3]
    assert list(b) == [4, 5]


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])
    assert lsplit_at(10, [1, 2]) == ([1, 2], [])


def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1, 2])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1, 2]


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 1]) == ([1, 2], [3, 1])
    assert lsplit_by(lambda x: x < 3, [5, 6]) == ([], [5, 6])


# --------------------------------- grouping ---------------------------------

def test_group_by():
    result = group_by(lambda x: x % 2, range(6))
    assert dict(result) == {0: [0, 2, 4], 1: [1, 3, 5]}


def test_group_by_keys():
    data = [{'tags': ['a', 'b']}, {'tags': ['b', 'c']}]
    result = group_by_keys(lambda d: d['tags'], data)
    assert dict(result) == {
        'a': [{'tags': ['a', 'b']}],
        'b': [{'tags': ['a', 'b']}, {'tags': ['b', 'c']}],
        'c': [{'tags': ['b', 'c']}],
    }


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert dict(result) == {'a': [1, 3], 'b': [2]}


def test_count_by():
    result = count_by(lambda x: x % 2, range(5))
    assert dict(result) == {0: 3, 1: 2}


def test_count_reps():
    result = count_reps([1, 1, 2, 3, 3, 3])
    assert dict(result) == {1: 2, 2: 1, 3: 3}


# -------------------------------- partition ---------------------------------

def test_partition():
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]
    assert lpartition(2, [1, 2, 3, 4]) == [[1, 2], [3, 4]]


def test_partition_with_step():
    assert lpartition(2, 1, [1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4]]


def test_partition_n_gt_seq():
    assert lpartition(5, [1, 2, 3]) == []


def test_partition_iterator():
    assert list(partition(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4]]


def test_partition_iterator_step_lt_n():
    assert lpartition(3, 1, iter([1, 2, 3, 4, 5])) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_partition_iterator_step_gt_n():
    assert lpartition(2, 3, iter(range(10))) == [[0, 1], [3, 4], [6, 7]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2, 3]))
    with pytest.raises(ValueError):
        list(partition(2, 0, [1, 2, 3]))


def test_chunks():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]
    assert lchunks(3, [1, 2]) == [[1, 2]]


def test_chunks_with_step():
    assert lchunks(2, 1, [1, 2, 3]) == [[1, 2], [2, 3], [3]]


def test_chunks_iterator():
    assert list(chunks(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4], [5]]


def test_chunks_iterator_step_variants():
    assert lchunks(2, 1, iter(range(5))) == [[0, 1], [1, 2], [2, 3], [3, 4], [4]]
    assert lchunks(2, 3, iter(range(7))) == [[0, 1], [3, 4], [6]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        list(chunks(0, [1, 2]))


# ------------------------------- partition_by -------------------------------

def test_partition_by():
    result = list(partition_by(lambda x: x % 2, [1, 1, 2, 2, 2, 3, 1]))
    assert [list(g) for g in result] == [[1, 1], [2, 2, 2], [3], [1]]


def test_lpartition_by():
    result = lpartition_by(lambda x: x % 2, [1, 1, 2, 2, 3])
    assert result == [[1, 1], [2, 2], [3]]


# ------------------------- with_prev / with_next / pairwise -----------------

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


# -------------------------------- reductions --------------------------------

def test_reductions():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_reductions_other_func():
    assert list(reductions(operator.mul, [2, 3, 4])) == [2, 6, 24]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]
    assert lsums([1, 2, 3]) == [1, 3, 6]


def test_sums_with_acc():
    assert list(sums([1, 2, 3], 100)) == [101, 103, 106]


def test_sums_empty():
    assert list(sums([])) == []
    assert lsums([]) == []
