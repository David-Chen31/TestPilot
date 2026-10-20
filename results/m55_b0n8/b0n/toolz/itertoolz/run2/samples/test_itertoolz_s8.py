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


# ---------------------------------------------------------------- remove
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------------------------------------------------------------- accumulate
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_initial_empty():
    assert list(accumulate(operator.add, [], 1)) == [1]


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    assert groupby(lambda x: x % 2 == 0, [1, 2, 3, 4]) == {
        False: [1, 3],
        True: [2, 4],
    }


def test_groupby_key_member():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
    ]
    assert groupby('gender', data) == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'}],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------- merge_sorted
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_many():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3], [4])) == [1, 2, 3, 4]
    assert list(merge_sorted([4], [1, 2, 3])) == [1, 2, 3, 4]
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


# ---------------------------------------------------------------- interleave
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_outer():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], [1, 2]])) == [1, 2]


# ---------------------------------------------------------------- unique
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------------------------------------------------------- isdistinct
def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct('World') is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct('Hello') is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------- take / drop / take_nth
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------- tail
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_length():
    assert tail(10, [1, 2]) == [1, 2]


# ---------------------------------------------------------------- first / second / nth / last
def test_first():
    assert first('ABC') == 'A'
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3
    assert nth(0, iter([7, 8])) == 7
    with pytest.raises(IndexError):
        nth(5, [1, 2])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------- get
def test_get_single():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555'}) == '555'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get(['Alice', 'Bob'], {'Alice': 1, 'Bob': 2}) == (1, 2)


def test_get_list_default():
    assert get(['Alice', 'Dennis'], {'Alice': 1}, None) == (1, None)


def test_get_single_default():
    assert get('missing', {'a': 1}, None) is None
    assert get(5, [1, 2], 'x') == 'x'


def test_get_single_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})
    with pytest.raises(IndexError):
        get(5, [1, 2])


def test_get_empty_list():
    assert get([], 'ABC') == ()
    assert get([], 'ABC', None) == ()


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


# ---------------------------------------------------------------- concat / concatv / mapcat / cons / interpose
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[], [1], [2]]))) == [1, 2]


def test_concatv():
    assert list(concatv([], ['a'], ['b', 'c'])) == ['a', 'b', 'c']


def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [['a', 'b'], ['c', 'd', 'e']])) == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose('a', [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose('a', [])) == []
    assert list(interpose('a', [1])) == [1]


# ---------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2
    }
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_no_init():
    assert reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5]) == {
        False: 9, True: 6
    }


def test_reduceby_mul():
    assert reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5]) == {
        False: 15, True: 8
    }


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_constant_init():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


# ---------------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------- partition / partition_all
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)
    ]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(3, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter(range(7)))) == [(0, 1, 2), (3, 4, 5), (6,)]


# ---------------------------------------------------------------- count
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------------------------------------------------------------- pluck / getter
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2}]
    assert list(pluck('name', data, default=None)) == [None, None]


def test_pluck_list_default():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=0)) == [(1, 0), (2, 3)]


def test_getter_single():
    g = getter(0)
    assert g([7, 8]) == 7


def test_getter_list_multi():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_list_single():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_list_empty():
    g = getter([])
    assert g([1, 2]) == ()


# ---------------------------------------------------------------- join
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    expected = {('Alice', 'Berlin'), ('Alice', 'Paris'), ('Alice', 'Shanghai'),
                ('Edith', 'Chicago'), ('Edith', 'NYC'),
                ('Zhao', 'Chicago'), ('Zhao', 'NYC'),
                ('Zhao', 'Berlin'), ('Zhao', 'Paris')}
    assert set(result) == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_key_getter():
    friends = [('Alice', 'Edith')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = join(1, friends, 0, cities)
    assert set(result) == {('Alice', 'NYC'), ('Edith', 'Paris')}


def test_join_empty():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []


# ---------------------------------------------------------------- diff
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)
    ]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice'
    )


def test_topk_key_getter():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


# ---------------------------------------------------------------- peek / peekn
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, restored = peek(seq)
    assert item == 0
    assert list(restored) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    seq = iter([5, 6, 7])
    item, restored = peek(seq)
    assert item == 5
    assert list(restored) == [5, 6, 7]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, restored = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(restored) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    first_three, restored = peekn(3, [1])
    assert first_three == (1,)
    assert list(restored) == [1]


# ---------------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98
    ]


def test_random_sample_randomobj():
    seq = list(range(100))
    randobj = random.Random(2016)
    assert list(random_sample(0.1, seq, random_state=randobj)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98
    ]


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_empty():
    assert list(random_sample(0.5, [], random_state=0)) == []
