import itertools
import operator
import random

import pytest

from toolz.itertoolz import (
    remove,
    accumulate,
    groupby,
    merge_sorted,
    interleave,
    unique,
    isiterable,
    isdistinct,
    take,
    tail,
    drop,
    take_nth,
    first,
    second,
    nth,
    last,
    get,
    concat,
    concatv,
    mapcat,
    cons,
    interpose,
    frequencies,
    reduceby,
    iterate,
    sliding_window,
    partition,
    partition_all,
    count,
    pluck,
    getter,
    join,
    diff,
    topk,
    peek,
    peekn,
    random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: True, [])) == []
    assert list(remove(lambda x: False, [1, 2])) == [1, 2]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {5: ['Alice', 'Edith', 'Frank'],
                      3: ['Bob', 'Dan'],
                      7: ['Charlie']}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert result['M'] == [{'name': 'Bob', 'gender': 'M'},
                           {'name': 'Charlie', 'gender': 'M'}]


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_or_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 100], [3, 4, 5])) == [1, 2, 3, 4, 5, 100]
    assert list(merge_sorted([1], [2, 3, 4, 5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(None) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct('Hello') is False
    assert isdistinct('World') is True
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / tail / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (10, 20, 30, 40, 50)) == (40, 50)
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)
    assert tail(10, [1, 2]) == [1, 2]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5])) == [1, 4]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first_second():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second_too_short():
    with pytest.raises(StopIteration):
        second(['a'])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3
    assert nth(1, iter([1, 2, 3])) == 2


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_basic():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'unknown') == 'unknown'


def test_get_single_list_index():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()
    assert get([], 'ABC', None) == ()


def test_get_list_with_default():
    assert get([1, 5], 'ABC', 'X') == ('B', 'X')


def test_get_missing_raises():
    with pytest.raises(IndexError):
        get(5, 'ABC')
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_type_error_raises():
    with pytest.raises(TypeError):
        get(object(), [1, 2, 3])


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_basic():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init_value():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_with_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'state': 'CA', 'cost': 1000000},
                {'state': 'IL', 'cost': 100000},
                {'state': 'IL', 'cost': 2000000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1000000, 'IL': 2100000}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]

    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]


def test_sliding_window_short_seq():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]
    assert list(partition(2, [1, 2, 3], pad=0)) == [(1, 2), (3, 0)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_list_path():
    # Uses __len__ fast-path
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_iterator_path():
    # Uses binary search path for iterators (no __len__)
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2, 3), (4, 5)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3])) == 3
    assert count([]) == 0
    assert count('abc') == 3


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------
def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2}]
    assert list(pluck('name', data, 'N/A')) == ['N/A', 'N/A']
    assert list(pluck(['id', 'name'], data, 'N/A')) == \
        [(1, 'N/A'), (2, 'N/A')]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------
def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list_single():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([10, 20]) == (10, 20)


def test_getter_list_empty():
    g = getter([])
    assert g([10, 20]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    pairs = set((a, d) for ((a, b), (c, d)) in result)
    expected = {
        ('Alice', 'Berlin'), ('Alice', 'Paris'), ('Alice', 'Shanghai'),
        ('Edith', 'Chicago'), ('Edith', 'NYC'),
        ('Zhao', 'Chicago'), ('Zhao', 'NYC'),
        ('Zhao', 'Berlin'), ('Zhao', 'Paris'),
    }
    assert pairs == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}
    assert result[-1] == (1, None)


def test_join_right_only():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4)}


def test_join_left_only():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert set(result) == {(2, 2), (3, 3), (1, None)}


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert set(result) == {(('Alice', 'Edith'), ('Edith', 'Paris')),
                          (('Edith', 'Alice'), ('Alice', 'NYC'))}


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_with_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_multiple():
    assert list(diff([1, 2], [1, 3], [1, 4])) == [(2, 3, 4)]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10], [1, 2, 3]])) == [(3, 10, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_returns_tuple():
    result = topk(1, [1, 2, 3])
    assert isinstance(result, tuple)
    assert result == (3,)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, seq2 = peek(seq)
    assert first_item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, seq2 = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    peeked, seq2 = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(seq2) == [1, 2, 3]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_deterministic():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert result1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=0)) == list(range(10))
