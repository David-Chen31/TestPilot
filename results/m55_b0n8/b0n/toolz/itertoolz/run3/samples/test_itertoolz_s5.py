import itertools
import operator
import random
from collections import OrderedDict

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join,
    diff, topk, peek, peekn, random_sample,
)


# ---------------------------------------------------------------- remove

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]
    assert list(remove(lambda x: x < 0, [])) == []


# ------------------------------------------------------------- accumulate

def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(accumulate(operator.mul, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


# ---------------------------------------------------------------- groupby

def test_groupby_callable():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan'])
    assert result == {5: ['Alice'], 3: ['Bob', 'Dan'], 7: ['Charlie']}


def test_groupby_iseven():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert result == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ----------------------------------------------------------- merge_sorted

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_single_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_with_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


# ------------------------------------------------------------- interleave

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_with_empty_seq():
    assert list(interleave([[1, 2, 3], []])) == [1, 2, 3]


# ----------------------------------------------------------------- unique

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == (
        'cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ------------------------------------------------------------- isiterable

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ------------------------------------------------------------- isdistinct

def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_string():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ------------------------------------------------------------------- take

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


# ------------------------------------------------------------------- tail

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(0, [1, 2, 3]) == []


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


# ------------------------------------------------------------------- drop

def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(10, [1, 2])) == []


# --------------------------------------------------------------- take_nth

def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# ------------------------------------------------------- first/second/nth

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1
    assert first(iter([5, 6])) == 5


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([10, 20, 30]) == 20


def test_second_too_short():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_iterator_too_short():
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# -------------------------------------------------------------------- get

def test_get_single():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_list_with_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_single_default():
    assert get(10, [1, 2, 3], default='missing') == 'missing'
    assert get('unknown', {'a': 1}, default=0) == 0


def test_get_missing_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])
    with pytest.raises(KeyError):
        get('unknown', {'a': 1})


def test_get_empty_list_index():
    assert get([], 'ABC') == ()


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_typeerror_with_default():
    assert get(1.5, [1, 2, 3], default='x') == 'x'


# ----------------------------------------------------------------- concat

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]


# ---------------------------------------------------------------- concatv

def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


# ----------------------------------------------------------------- mapcat

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ------------------------------------------------------------------- cons

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# -------------------------------------------------------------- interpose

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ------------------------------------------------------------ frequencies

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# --------------------------------------------------------------- reduceby

def test_reduceby_basic():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    result = reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_with_init_constant():
    data = [1, 2, 3, 4, 5]
    result = reduceby(lambda x: x % 2 == 0, operator.add, data, 0)
    assert result == {False: 9, True: 6}


def test_reduceby_with_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_with_non_callable_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


# ---------------------------------------------------------------- iterate

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


# --------------------------------------------------------- sliding_window

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# -------------------------------------------------------------- partition

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == (
        [(1, 2), (3, 4), (5, None)])


# ---------------------------------------------------------- partition_all

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_incomplete():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == (
        [(1, 2), (3, 4), (5,)])


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_with_len():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == (
        [(1, 2), (3, 4), (5,)])


# ------------------------------------------------------------------ count

def test_count_sequence():
    assert count([1, 2, 3]) == 3
    assert count('abc') == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(iter([])) == 0


# ------------------------------------------------------------------ pluck

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    seqs = [{'a': 1}, {'b': 2}, {'a': 3}]
    assert list(pluck('a', seqs, default=None)) == [1, None, 3]


def test_pluck_multiple_with_default():
    seqs = [{'a': 1, 'b': 2}, {'a': 3}]
    assert list(pluck(['a', 'b'], seqs, default=None)) == (
        [(1, 2), (3, None)])


# ----------------------------------------------------------------- getter

def test_getter_single():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_list_single():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_list_empty():
    g = getter([])
    assert g([10, 20, 30]) == ()


# ------------------------------------------------------------------- join

def test_join_inner():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right))
    assert sorted(result) == [(2, 2), (3, 3)]


def test_join_right():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None))
    assert sorted(result, key=str) == sorted(
        [(2, 2), (3, 3), (None, 4)], key=str)


def test_join_left():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       right_default=None))
    assert sorted(result, key=str) == sorted(
        [(2, 2), (3, 3), (1, None)], key=str)


def test_join_full():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None, right_default=None))
    assert sorted(result, key=str) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)], key=str)


def test_join_with_index_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(1, friends, 0, cities))
    expected = [(('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
                (('Alice', 'Edith'), ('Edith', 'Paris')),
                (('Alice', 'Edith'), ('Edith', 'Berlin')),
                (('Edith', 'Alice'), ('Alice', 'NYC')),
                (('Edith', 'Alice'), ('Alice', 'Chicago')),
                (('Zhao', 'Alice'), ('Alice', 'NYC')),
                (('Zhao', 'Alice'), ('Alice', 'Chicago')),
                (('Zhao', 'Edith'), ('Edith', 'Paris')),
                (('Zhao', 'Edith'), ('Edith', 'Berlin'))]
    assert sorted(result) == sorted(expected)


# ------------------------------------------------------------------- diff

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == (
        [(3, 10), (None, 100)])


def test_diff_with_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 5, 3], [1, 2, 3])) == [(2, 5, 2)]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


# ------------------------------------------------------------------- topk

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


# ------------------------------------------------------------------ peek

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest_seq = peek(seq)
    assert item == 0
    assert list(rest_seq) == [0, 1, 2, 3, 4]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ----------------------------------------------------------------- peekn

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest_seq = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, rest_seq = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(rest_seq) == [1, 2, 3]


# ---------------------------------------------------------- random_sample

def test_random_sample_all():
    seq = list(range(20))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_none():
    seq = list(range(20))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b


def test_random_sample_random_object():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=random.Random(2016)))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
