import itertools
import operator
import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join,
    diff, topk, peek, peekn, random_sample, no_pad,
)
from toolz.utils import no_default


# ---------- remove ----------
def test_remove():
    def iseven(x):
        return x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: False, [1, 2])) == [1, 2]
    assert list(remove(lambda x: True, [1, 2])) == []


# ---------- accumulate ----------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_lazy_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6


# ---------- groupby ----------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_member_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [data[0]], 'M': [data[1], data[2]]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------- merge_sorted ----------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_zero_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_arg():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_many():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


# ---------- interleave ----------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


# ---------- unique ----------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------- isiterable ----------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


# ---------- isdistinct ----------
def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------- take ----------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


# ---------- tail ----------
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    result = tail(2, iter([1, 2, 3, 4]))
    assert result == (3, 4)


def test_tail_more_than_length():
    assert tail(10, [1, 2]) == [1, 2]


# ---------- drop ----------
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---------- take_nth ----------
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# ---------- first / second / nth / last ----------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30])) == 30


def test_nth_negative_on_sequence():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------- get ----------
def test_get_single():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555-1234'}) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get(['Alice', 'Bob'],
               {'Alice': '555-1234', 'Bob': '555-5678'}) == ('555-1234', '555-5678')


def test_get_list_with_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_single_default():
    assert get(10, 'ABC', None) is None
    assert get('missing', {'a': 1}, 'default') == 'default'


def test_get_missing_no_default_raises():
    with pytest.raises(IndexError):
        get(10, 'ABC')
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_non_indexable_with_default():
    assert get(0, 5, 'd') == 'd'


# ---------- concat / concatv ----------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_accepts_generator():
    assert list(concat(iter([[], [1], [2, 3]]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ---------- mapcat ----------
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------- cons ----------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------- interpose ----------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------- frequencies ----------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------- reduceby ----------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}


def test_reduceby_constant_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5], 0) == {False: 9, True: 6}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_member_key():
    projects = [{'name': 'a', 'state': 'CA', 'cost': 100},
                {'name': 'b', 'state': 'IL', 'cost': 200},
                {'name': 'c', 'state': 'IL', 'cost': 300}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 100, 'IL': 500}


# ---------- iterate ----------
def test_iterate():
    inc = lambda x: x + 1
    it = iterate(inc, 0)
    assert [next(it) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    double = lambda x: x * 2
    it = iterate(double, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_few():
    assert list(sliding_window(3, [1, 2])) == []


# ---------- partition ----------
def test_partition_exact():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_partial():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------- partition_all ----------
def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_partial():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_iterator_no_len():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_single_element_partial():
    assert list(partition_all(3, iter([1]))) == [(1,)]


# ---------- count ----------
def test_count_len():
    assert count([1, 2, 3]) == 3
    assert count('abc') == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------- pluck ----------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=0)) == [1, 0]


def test_pluck_multiple_default():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, None), (2, 3)]


# ---------- getter ----------
def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list():
    g = getter([0, 1])
    assert g([10, 20]) == (10, 20)


def test_getter_single_element_list():
    g = getter([1])
    assert g([10, 20]) == (20,)


def test_getter_empty_list():
    g = getter([])
    assert g([10, 20]) == ()


# ---------- join ----------
def test_join_inner():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10), ('c', 30)]
    result = list(join(first, left, first, right))
    assert result == [(('a', 1), ('a', 10))]


def test_join_empty_left():
    assert list(join(first, [], first, [('a', 1)])) == []


def test_join_right_join():
    left = [('a', 1)]
    right = [('a', 10), ('c', 30)]
    result = list(join(first, left, first, right, left_default=None))
    assert result == [(('a', 1), ('a', 10)), (None, ('c', 30))]


def test_join_left_join():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10)]
    result = list(join(first, left, first, right, right_default=None))
    assert result == [(('a', 1), ('a', 10)), (('b', 2), None)]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=str) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)], key=str)


def test_join_non_callable_keys():
    left = [('a', 1)]
    right = [('a', 10)]
    assert list(join(0, left, 0, right)) == [(('a', 1), ('a', 10))]


# ---------- diff ----------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


# ---------- topk ----------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


# ---------- peek ----------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, new_seq = peek(seq)
    assert item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, new_seq = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(new_seq) == [1, 2, 3]


# ---------- peekn ----------
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    items, new_seq = peekn(2, seq)
    assert items == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    items, new_seq = peekn(10, [1, 2])
    assert items == (1, 2)
    assert list(new_seq) == [1, 2]


# ---------- random_sample ----------
def test_random_sample_deterministic():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_random_object():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


# ---------- no_pad sentinel ----------
def test_no_pad_sentinel():
    assert no_pad == '__no__pad__'
