import pytest
import random
from operator import add, mul
from itertools import count as icount

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join,
    diff, topk, peek, peekn, random_sample,
)


# ---------- remove ----------
def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


# ---------- accumulate ----------
def test_accumulate_basic():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


# ---------- groupby ----------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7], True: [2, 4, 6, 8]
    }


def test_groupby_key_string():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}],
    }


def test_groupby_returns_real_lists():
    result = groupby(lambda x: x % 2, [1, 3, 2, 4])
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# ---------- merge_sorted ----------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_empty_seq():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


# ---------- interleave ----------
def test_interleave_basic():
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
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


# ---------- isiterable ----------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


# ---------- isdistinct ----------
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------- take / drop / tail / take_nth ----------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


# ---------- first / second / nth / last ----------
def test_first():
    assert first('ABC') == 'A'
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(1, [10, 20, 30]) == 20
    assert nth(0, (1, 2, 3)) == 1


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------- get ----------
def test_get_single():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555-1234'}) == '555-1234'


def test_get_multiple():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_single_index_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_missing_no_default():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})


def test_get_missing_index():
    with pytest.raises(IndexError):
        get(10, 'ABC')


def test_get_no_index_support_with_default():
    assert get(0, 5, default='x') == 'x'


def test_get_no_index_support_no_default():
    with pytest.raises(TypeError):
        get(0, 5)


# ---------- concat / concatv ----------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


# ---------- mapcat ----------
def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == \
        ['A', 'B', 'C', 'D', 'E']


# ---------- cons ----------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


# ---------- interpose ----------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


# ---------- frequencies ----------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}


# ---------- reduceby ----------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data, 0) == {False: 9, True: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_key_string():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


# ---------- iterate ----------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


# ---------- sliding_window ----------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == \
        [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---------- partition ----------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


# ---------- partition_all ----------
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


# ---------- count ----------
def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3])) == 3
    assert count([]) == 0


# ---------- pluck ----------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default='unknown')) == \
        ['unknown', 'Pies']


# ---------- getter ----------
def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------- join ----------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris')),
                      (('Alice', 'Edith'), ('Alice', 'NYC'))]


def test_join_left_default():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_right_default():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert (None, 4) in result
    assert (1, None) in result
    assert (2, 2) in result
    assert (3, 3) in result


# ---------- diff ----------
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_multiple_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------- topk ----------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


# ---------- peek ----------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, new_seq = peek(seq)
    assert item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------- peekn ----------
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, new_seq = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    peeked, new_seq = peekn(10, [0, 1])
    assert peeked == (0, 1)
    assert list(new_seq) == [0, 1]


# ---------- random_sample ----------
def test_random_sample_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2
    assert r1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_with_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(5), random_state=0)) == [0, 1, 2, 3, 4]
