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


# ---------------------------------------------------------------------------
# repeatedly / iterate
# ---------------------------------------------------------------------------

def test_repeatedly_infinite():
    it = repeatedly(lambda: 7)
    assert take(4, it) == [7, 7, 7, 7]


def test_repeatedly_with_n():
    assert list(repeatedly(lambda: 'x', 3)) == ['x', 'x', 'x']
    assert list(repeatedly(lambda: 'x', 0)) == []


def test_repeatedly_side_effects():
    counter = {'n': 0}

    def inc():
        counter['n'] += 1
        return counter['n']

    assert list(repeatedly(inc, 4)) == [1, 2, 3, 4]


def test_iterate():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]
    assert take(3, iterate(lambda x: x + 1, 0)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# take / drop / first / second / nth / last / rest / butlast
# ---------------------------------------------------------------------------

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2, 3]) == []


def test_take_iterator():
    assert take(3, iter(range(10))) == [0, 1, 2]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(0, [1, 2])) == [1, 2]
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
    assert nth(1, [10, 20, 30]) == 20
    assert nth(5, [10, 20, 30]) is None
    assert nth(2, iter(range(10))) == 2
    assert nth(100, iter(range(3))) is None


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
    assert list(butlast([1, 2, 3, 4])) == [1, 2, 3]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


# ---------------------------------------------------------------------------
# shuffled / ilen
# ---------------------------------------------------------------------------

def test_shuffled_preserves_items():
    result = shuffled([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert isinstance(result, list)


def test_shuffled_empty():
    assert shuffled([]) == []


def test_shuffled_does_not_modify_input():
    src = [1, 2, 3]
    shuffled(src)
    assert src == [1, 2, 3]


def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter(range(10))) == 10


# ---------------------------------------------------------------------------
# lmap / lfilter / map / filter
# ---------------------------------------------------------------------------

def test_lmap_and_map():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert list(map(lambda x: x + 1, [1, 2, 3])) == [2, 3, 4]


def test_lmap_multiple_seqs():
    assert lmap(operator.add, [1, 2, 3], [10, 20, 30]) == [11, 22, 33]


def test_lmap_string_func():
    # string function derived via make_func -> re_finder returns first match
    assert lmap(r'\d+', ['a1b2', 'c3']) == ['1', '3']


def test_lmap_int_func():
    assert lmap(0, [['a', 'b'], ['c', 'd']]) == ['a', 'c']


def test_lfilter_and_filter():
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]
    assert list(filter(lambda x: x > 1, [1, 2, 3])) == [2, 3]


def test_lfilter_pred_from_set():
    assert lfilter({1, 3}, [1, 2, 3, 4]) == [1, 3]


def test_lfilter_pred_from_str():
    # regex tester
    assert lfilter(r'\d', ['a', '1', 'b', '2']) == ['1', '2']


# ---------------------------------------------------------------------------
# lremove / remove / lkeep / keep
# ---------------------------------------------------------------------------

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]


def test_lremove_pred_from_set():
    assert lremove({1, 3}, [1, 2, 3, 4]) == [2, 4]


def test_keep_one_arg():
    assert list(keep([0, 1, False, 2, '', 3])) == [1, 2, 3]


def test_keep_two_args():
    assert list(keep(lambda x: x if x % 2 == 0 else 0, [1, 2, 3, 4])) == [2, 4]


def test_lkeep_one_arg():
    assert lkeep([0, 1, False, 2, '', 3]) == [1, 2, 3]


def test_lkeep_two_args():
    assert lkeep(lambda x: x * 2 if x > 1 else 0, [1, 2, 3]) == [4, 6]


# ---------------------------------------------------------------------------
# without / lwithout
# ---------------------------------------------------------------------------

def test_without():
    assert list(without([1, 2, 3, 4, 2], 2)) == [1, 3, 4]
    assert list(without([1, 2, 3], 1, 3)) == [2]
    assert list(without([1, 2, 3])) == [1, 2, 3]


def test_lwithout():
    assert lwithout([1, 2, 3, 4, 2], 2) == [1, 3, 4]
    assert lwithout([1, 2, 3], 1, 3) == [2]


# ---------------------------------------------------------------------------
# lconcat / lcat / flatten / lflatten / lmapcat / mapcat
# ---------------------------------------------------------------------------

def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]
    assert lconcat([], []) == []


def test_lcat():
    assert lcat([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
    assert lcat([]) == []


def test_flatten():
    assert list(flatten([1, [2, 3], [4, [5, 6]]])) == [1, 2, 3, 4, 5, 6]


def test_flatten_string_not_followed():
    # strings are excluded by is_seqcont
    assert list(flatten(['ab', ['cd', 'ef']])) == ['ab', 'cd', 'ef']


def test_flatten_custom_follow():
    data = [1, [2, 3], [4, [5, 6]]]
    assert list(flatten(data, follow=lambda x: isinstance(x, list) and x and x[0] == 2)) == [
        1, 2, 3, [4, [5, 6]]
    ]


def test_lflatten():
    assert lflatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]


def test_mapcat():
    assert list(mapcat(lambda x: [x, x * 2], [1, 2, 3])) == [1, 2, 2, 4, 3, 6]


def test_lmapcat():
    assert lmapcat(lambda x: [x, x * 2], [1, 2, 3]) == [1, 2, 2, 4, 3, 6]


# ---------------------------------------------------------------------------
# interleave / interpose
# ---------------------------------------------------------------------------

def test_interleave():
    assert list(interleave([1, 2, 3], [4, 5, 6])) == [1, 4, 2, 5, 3, 6]
    # stops at shortest
    assert list(interleave([1, 2, 3], [4, 5])) == [1, 4, 2, 5]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [1])) == [1]
    assert list(interpose(0, [])) == []


# ---------------------------------------------------------------------------
# takewhile / dropwhile
# ---------------------------------------------------------------------------

def test_takewhile_two_args():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 4, 1])) == [1, 2]


def test_takewhile_one_arg():
    assert list(takewhile([1, 2, 0, 3])) == [1, 2]


def test_takewhile_pred_from_str():
    assert list(takewhile(r'\d', ['1', '2', 'a', '3'])) == ['1', '2']


def test_dropwhile_two_args():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 4, 1])) == [3, 4, 1]


def test_dropwhile_one_arg():
    # one-arg version reorders arguments incorrectly: pred, seq = bool, pred
    # so seq becomes the bool builtin and pred becomes the list
    assert list(dropwhile([0, 0, 1, 2, 0])) == [0, 0, 1, 2, 0]


def test_dropwhile_pred_from_str():
    assert list(dropwhile(r'\d', ['1', '2', 'a', '3'])) == ['a', '3']


# ---------------------------------------------------------------------------
# distinct / ldistinct
# ---------------------------------------------------------------------------

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]


def test_distinct_with_key():
    data = [{'k': 1, 'v': 'a'}, {'k': 2}, {'k': 1, 'v': 'b'}]
    assert list(distinct(data, key=operator.itemgetter('k'))) == [
        {'k': 1, 'v': 'a'}, {'k': 2}
    ]


def test_distinct_with_callable_key():
    assert list(distinct([1, -1, 2, -2, 3], key=abs)) == [1, 2, 3]


def test_ldistinct():
    assert ldistinct([1, 2, 1, 3]) == [1, 2, 3]
    assert ldistinct([1, -1, 2], key=abs) == [1, 2]


# ---------------------------------------------------------------------------
# split / lsplit
# ---------------------------------------------------------------------------

def test_lsplit():
    yes, no = lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert yes == [2, 4]
    assert no == [1, 3]


def test_lsplit_pred_from_set():
    yes, no = lsplit({1, 2}, [1, 2, 3, 4])
    assert yes == [1, 2]
    assert no == [3, 4]


def test_split():
    yes, no = split(lambda x: x % 2 == 0, [1, 2, 3, 4])
    assert list(yes) == [2, 4]
    assert list(no) == [1, 3]


def test_split_lazy_interleaved():
    yes, no = split(lambda x: x > 0, [-1, 1, -2, 2])
    # Consuming lazily from each side
    assert next(yes) == 1
    assert next(no) == -1
    assert next(yes) == 2
    assert next(no) == -2


def test_split_empty():
    yes, no = split(bool, [])
    assert list(yes) == []
    assert list(no) == []


# ---------------------------------------------------------------------------
# split_at / lsplit_at
# ---------------------------------------------------------------------------

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_split_at_past_end():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])


# ---------------------------------------------------------------------------
# split_by / lsplit_by
# ---------------------------------------------------------------------------

def test_split_by():
    a, b = split_by(lambda x: x > 0, [1, 2, -1, 3])
    assert list(a) == [1, 2]
    assert list(b) == [-1, 3]


def test_lsplit_by():
    assert lsplit_by(lambda x: x < 3, [1, 2, 3, 1]) == ([1, 2], [3, 1])


def test_lsplit_by_all_pass():
    assert lsplit_by(lambda x: x > 0, [1, 2, 3]) == ([1, 2, 3], [])


# ---------------------------------------------------------------------------
# group_by / group_by_keys / group_values / count_by / count_reps
# ---------------------------------------------------------------------------

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert result == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_string_key():
    # strings are interpreted as regexes by make_func, so use a callable here
    data = [{'t': 'a'}, {'t': 'b'}, {'t': 'a'}]
    result = group_by(operator.itemgetter('t'), data)
    assert result == {'a': [{'t': 'a'}, {'t': 'a'}], 'b': [{'t': 'b'}]}


def test_group_by_keys():
    data = [{'tags': ['a', 'b']}, {'tags': ['b', 'c']}]
    result = group_by_keys(operator.itemgetter('tags'), data)
    assert result == {
        'a': [{'tags': ['a', 'b']}],
        'b': [{'tags': ['a', 'b']}, {'tags': ['b', 'c']}],
        'c': [{'tags': ['b', 'c']}],
    }


def test_group_values():
    result = group_values([('a', 1), ('b', 2), ('a', 3)])
    assert result == {'a': [1, 3], 'b': [2]}


def test_count_by():
    assert count_by(lambda x: x % 2, [1, 2, 3, 4, 5]) == {1: 3, 0: 2}


def test_count_by_string_func():
    # string counted as regex; use a callable to test dictionary-key extraction
    result = count_by(operator.itemgetter('type'),
                      [{'type': 'a'}, {'type': 'b'}, {'type': 'a'}])
    assert dict(result) == {'a': 2, 'b': 1}


def test_count_reps():
    assert count_reps('aabbbc') == {'a': 2, 'b': 3, 'c': 1}
    assert dict(count_reps([])) == {}


# ---------------------------------------------------------------------------
# partition / lpartition / chunks / lchunks
# ---------------------------------------------------------------------------

def test_partition_seq():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4]]


def test_partition_with_step():
    assert list(partition(2, 3, [1, 2, 3, 4, 5, 6, 7])) == [[1, 2], [4, 5]]


def test_partition_with_overlap_step():
    assert list(partition(3, 1, [1, 2, 3, 4, 5])) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_partition_iter():
    assert list(partition(2, iter(range(5)))) == [[0, 1], [2, 3]]
    assert list(partition(2, 1, iter(range(4)))) == [[0, 1], [1, 2], [2, 3]]


def test_partition_iter_step_gt_n():
    # Iterator implementation yields full final part when tail fits
    assert list(partition(2, 3, iter(range(8)))) == [[0, 1], [3, 4], [6, 7]]


def test_partition_zero_raises():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2, 3]))


def test_lpartition():
    # slicing a range yields ranges, not lists
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]
    assert lpartition(2, 3, range(10)) == [range(0, 2), range(3, 5), range(6, 8)]


def test_chunks_seq():
    assert list(chunks(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]


def test_chunks_iter():
    assert list(chunks(2, iter(range(5)))) == [[0, 1], [2, 3], [4]]
    # with overlap the tail is emitted too
    assert list(chunks(3, 2, iter(range(5)))) == [[0, 1, 2], [2, 3, 4], [4]]


def test_chunks_step_gt_n():
    assert list(chunks(2, 3, iter(range(6)))) == [[0, 1], [3, 4]]


def test_chunks_zero_raises():
    with pytest.raises(ValueError):
        list(chunks(0, [1, 2]))


def test_lchunks():
    assert lchunks(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]


# ---------------------------------------------------------------------------
# partition_by / lpartition_by
# ---------------------------------------------------------------------------

def test_partition_by():
    result = [list(g) for g in partition_by(lambda x: x % 2, [1, 3, 2, 4, 5])]
    assert result == [[1, 3], [2, 4], [5]]


def test_partition_by_string_func():
    # use callable instead of string (string means regex)
    data = [{'t': 1}, {'t': 1}, {'t': 2}]
    result = [list(g) for g in partition_by(operator.itemgetter('t'), data)]
    assert result == [[{'t': 1}, {'t': 1}], [{'t': 2}]]


def test_lpartition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5]) == [[1, 3], [2, 4], [5]]


# ---------------------------------------------------------------------------
# with_prev / with_next / pairwise
# ---------------------------------------------------------------------------

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]


def test_with_prev_fill():
    assert list(with_prev([1, 2], fill=0)) == [(1, 0), (2, 1)]


def test_with_prev_empty():
    assert list(with_prev([])) == []


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]


def test_with_next_fill():
    assert list(with_next([1, 2], fill=0)) == [(1, 2), (2, 0)]


def test_with_next_empty():
    assert list(with_next([])) == []


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_short():
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# ---------------------------------------------------------------------------
# reductions / lreductions / sums / lsums
# ---------------------------------------------------------------------------

def test_reductions():
    # accumulate starts with the first item
    assert list(reductions(lambda a, b: a + b, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_reductions_with_acc():
    assert list(reductions(lambda a, b: a * b, [1, 2, 3], 2)) == [2, 4, 12]


def test_reductions_add_identity():
    assert list(reductions(operator.add, [1, 2, 3])) == [1, 3, 6]


def test_reductions_with_acc_operator_add():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_lreductions():
    assert lreductions(lambda a, b: a + b, [1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lreductions(lambda a, b: a + b, [1, 2, 3], 0) == [1, 3, 6]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(sums([1, 2, 3], 10)) == [11, 13, 16]


def test_lsums():
    assert lsums([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert lsums([1, 2, 3], 0) == [1, 3, 6]


# ---------------------------------------------------------------------------
# re-exports: count / cycle / repeat
# ---------------------------------------------------------------------------

def test_reexports():
    from funcy.seqs import count, cycle, repeat
    assert take(3, count(5)) == [5, 6, 7]
    assert take(5, cycle([1, 2])) == [1, 2, 1, 2, 1]
    assert list(repeat('x', 3)) == ['x', 'x', 'x']


# ---------------------------------------------------------------------------
# misc: EMPTY sentinel behavior
# ---------------------------------------------------------------------------

def test_keep_one_arg_is_lazy_filter():
    # keep with EMPTY seq should behave like filter(bool) on the given iterable
    assert list(keep(iter([0, 1, None, 2]))) == [1, 2]


def test_partition_by_empty():
    assert lpartition_by(lambda x: x, []) == []


def test_interleave_single_seq():
    assert list(interleave([1, 2, 3])) == [1, 2, 3]
