import itertools
import operator
from collections import deque

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join,
    diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------- remove
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]
    assert list(remove(lambda x: x < 0, [])) == []


# ---------------------------------------------------------------- accumulate
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty():
    assert list(accumulate(operator.add, [])) == []


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_noncallable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [data[0]], 'M': [data[1], data[2]]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------- merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_edge():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_many_sequences():
    result = list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9], [10]))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_merge_sorted_key_many():
    result = list(merge_sorted([3, 1], [2, 4], [5], key=lambda x: -x))
    # Each input not sorted but key behaviour only depends on comparison
    assert result == sorted([3, 1, 2, 4, 5], key=lambda x: -x)


# ---------------------------------------------------------------- interleave
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_different_lengths():
    assert list(interleave([[1, 2, 3], [4]])) == [1, 4, 2, 3]


# ---------------------------------------------------------------- unique
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')
    assert tuple(unique([])) == ()
    assert tuple(unique([1, 1, 1])) == (1,)


# ---------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


# ---------------------------------------------------------------- isdistinct
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True
    assert isdistinct([]) is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------- take
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]
    assert list(take(2, iter([1, 2, 3]))) == [1, 2]


# ---------------------------------------------------------------- tail
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (10, 20, 30)) == (20, 30)
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)
    assert tail(10, iter([1, 2])) == (1, 2)


# ---------------------------------------------------------------- drop
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---------------------------------------------------------------- take_nth
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]
    assert list(take_nth(3, [1, 2])) == [1]


# ---------------------------------------------------------------- first, second, last
def test_first():
    assert first('ABC') == 'A'
    assert first([42]) == 42
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second([1])
    with pytest.raises(StopIteration):
        second([])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------- nth
def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3
    assert nth(1, iter([1, 2, 3])) == 2
    with pytest.raises(IndexError):
        nth(5, [1, 2])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


# ---------------------------------------------------------------- get
def test_get_simple():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_list_index():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([1], 'ABC') == ('B',)
    assert get([], 'ABC') == ()


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'default') == 'default'
    assert get([0, 5], [1, 2, 3], None) == (1, None)


def test_get_errors():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(5, [1, 2, 3])


def test_get_type_error():
    with pytest.raises(TypeError):
        get([0, 1], 5)
    assert get([0, 1], 5, 'default') == 'default'


# ---------------------------------------------------------------- concat / concatv
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat(iter([iter([1, 2]), iter([3])]))) == [1, 2, 3]
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ---------------------------------------------------------------- mapcat
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------- cons
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]
    assert list(cons(1, iter([2, 3]))) == [1, 2, 3]


# ---------------------------------------------------------------- interpose
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_basic():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_noncallable_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(operator.itemgetter(0), operator.add, []) == {}


# ---------------------------------------------------------------- iterate
def test_iterate():
    inc = lambda x: x + 1
    counter = iterate(inc, 0)
    assert [next(counter) for _ in range(5)] == [0, 1, 2, 3, 4]
    doubled = list(take(5, iterate(lambda x: x * 2, 1)))
    assert doubled == [1, 2, 4, 8, 16]


# ---------------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]
    assert list(sliding_window(2, [1])) == []
    assert list(sliding_window(2, [])) == []


# ---------------------------------------------------------------- partition
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------- partition_all
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(2, [])) == []
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


# ---------------------------------------------------------------- count
def test_count():
    assert count([1, 2, 3]) == 3
    assert count([]) == 0
    assert count(iter([1, 2, 3, 4])) == 4
    assert count('abc') == 3


# ---------------------------------------------------------------- pluck
def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {}]
    assert list(pluck('id', data, default=None)) == [1, None]
    assert list(pluck(['id', 'name'], data, default='x')) == [(1, 'x'), ('x', 'x')]


# ---------------------------------------------------------------- getter
def test_getter():
    assert getter(0)([1, 2]) == 1
    assert getter([0, 1])([1, 2, 3]) == (1, 2)
    assert getter([0])([1, 2]) == (1,)
    assert getter([])([1, 2]) == ()


# ---------------------------------------------------------------- join
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    assert sorted(result) == sorted([
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
    ])


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda t: (str(t[0]), str(t[1]))) == [
        (1, None), (2, 2), (3, 3), (None, 4)]


def test_join_right():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [1, 3],
                       left_default=None))
    assert sorted(result, key=str) == [(1, 1), (None, 3)]


def test_join_left():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [1, 3],
                       right_default=None))
    assert sorted(result, key=str) == [(1, 1), (2, None)]


def test_join_noncallable_keys():
    result = list(join(0, [(1, 'a'), (2, 'b')], 0, [(2, 'c')]))
    assert result == [((2, 'b'), (2, 'c'))]


# ---------------------------------------------------------------- diff
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_single_list_argument():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_multiple():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


# ---------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_noncallable_key():
    data = [{'x': 1}, {'x': 5}, {'x': 3}]
    assert topk(2, data, key='x') == ({'x': 5}, {'x': 3})


def test_topk_empty():
    assert topk(2, []) == ()


# ---------------------------------------------------------------- peek
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    item, it2 = peek(it)
    assert item == 10
    assert list(it2) == [10, 20, 30]


# ---------------------------------------------------------------- peekn
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq2 = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, seq = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(seq) == [1, 2]


# ---------------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=0)) == list(range(10))


def test_random_sample_custom_random_state():
    class FakeRandom:
        def __init__(self):
            self.vals = iter([0.0, 0.5, 0.9, 0.1])
        def random(self):
            return next(self.vals)
    result = list(random_sample(0.5, [1, 2, 3, 4], random_state=FakeRandom()))
    assert result == [1, 4]
