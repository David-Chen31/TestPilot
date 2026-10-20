import operator
import random
from collections import deque
from itertools import count as count_up

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
def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []
    # lazy: works on infinite sequences
    assert list(take(3, remove(iseven, count_up(1)))) == [1, 3, 5]


# ---------------------------------------------------------------- accumulate
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]
    assert list(accumulate(operator.add, [1, 2], 0)) == [0, 1, 3]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_lazy():
    # should not consume the full sequence
    gen = (x for x in count_up())
    result = accumulate(operator.add, gen)
    assert take(3, result) == [0, 1, 3]


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }


def test_groupby_string_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'},
        ],
    }


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# ---------------------------------------------------------------- merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_or_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_many():
    assert list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9])) == list(range(1, 10))


def test_merge_sorted_key_many():
    seqs = [[3, 1], [2, 4], [6, 5]]
    # all values within each seq share the same key so ordering by key stable
    result = list(merge_sorted(*seqs, key=lambda x: x))
    assert result == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------- interleave
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[]])) == []


def test_interleave_infinite():
    result = interleave([count_up(), count_up()])
    assert take(6, result) == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------- unique
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_lazy():
    result = unique(count_up())
    assert take(3, result) == [0, 1, 2]


# ---------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------------------------------------------------------- isdistinct
def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False
    assert isdistinct(iter([])) is True


# ---------------------------------------------------------------- take
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]
    assert list(take(3, count_up())) == [0, 1, 2]


# ---------------------------------------------------------------- tail
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_len():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


def test_tail_empty():
    assert list(tail(3, [])) == []


# ---------------------------------------------------------------- drop
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


# ---------------------------------------------------------------- take_nth
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ---------------------------------------------------------------- first/second/nth/last
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second(iter([1, 2, 3])) == 2
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(10, [1, 2, 3])
    with pytest.raises(StopIteration):
        nth(10, iter([1, 2, 3]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------- get
def test_get_index():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get(1, 'ABC') == 'B'
    assert get('Alice', phonebook) == '555-1234'


def test_get_multiple():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'unknown') == 'unknown'
    assert get(5, [1, 2, 3], 'x') == 'x'


def test_get_single_element_list():
    assert get(['x'], {'x': 1}) == (1,)
    assert get([], {'x': 1}) == ()


def test_get_missing_no_default():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})
    with pytest.raises(IndexError):
        get(5, [1, 2])


def test_get_type_error_default():
    # non-list index whose subscript raises TypeError, with default
    assert get(object(), [1, 2, 3], 'default') == 'default'


# ---------------------------------------------------------------- concat / concatv / mapcat
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    assert list(concat([x] for x in ([1], [2, 3]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(
        mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    )
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------- cons / interpose
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]
    assert take(3, cons(1, count_up())) == [1, 0, 1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3,
        'ox': 1,
        'pig': 2,
    }
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_value():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby(
        'state', lambda acc, x: acc + x['cost'], projects, 0
    )
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]

    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]


def test_sliding_window_too_small():
    assert list(sliding_window(2, [1])) == []
    assert list(sliding_window(2, [])) == []


# ---------------------------------------------------------------- partition
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2),
        (3, 4),
        (5, None),
    ]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------- partition_all
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator():
    gen = (x for x in range(5))
    assert list(partition_all(2, gen)) == [(0, 1), (2, 3), (4,)]


def test_partition_all_with_len():
    class HasLen:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __iter__(self):
            return iter(self.items)

    assert list(partition_all(3, HasLen([1, 2, 3, 4, 5]))) == [
        (1, 2, 3),
        (4, 5),
    ]


# ---------------------------------------------------------------- count
def test_count_with_len():
    assert count([1, 2, 3]) == 3
    assert count("abc") == 3
    assert count({1: 2}) == 1


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(iter([])) == 0


# ---------------------------------------------------------------- pluck
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck('id', data)) == [1, 2]


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, None)) == [None, 'Pies']
    assert list(pluck([0, 1], [[1], [4, 5]], None)) == [(1, None), (4, 5)]


# ---------------------------------------------------------------- getter
def test_getter():
    assert getter(0)([10, 20]) == 10
    assert getter('a')({'a': 1}) == 1
    assert getter([0, 1])([10, 20, 30]) == (10, 20)
    assert getter([0])([10, 20, 30]) == (10,)
    assert getter([])([10, 20]) == ()


# ---------------------------------------------------------------- join
def test_join_inner():
    friends = [
        ('Alice', 'Edith'),
        ('Alice', 'Zhao'),
        ('Edith', 'Alice'),
        ('Zhao', 'Alice'),
        ('Zhao', 'Edith'),
    ]
    cities = [
        ('Alice', 'NYC'),
        ('Alice', 'Chicago'),
        ('Dan', 'Sydney'),
        ('Edith', 'Paris'),
        ('Edith', 'Berlin'),
        ('Zhao', 'Shanghai'),
    ]
    result = join(second, friends, first, cities)
    assert sorted(result) == sorted(
        [
            (('Alice', 'Edith'), ('Edith', 'Paris')),
            (('Alice', 'Edith'), ('Edith', 'Berlin')),
            (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
            (('Edith', 'Alice'), ('Alice', 'NYC')),
            (('Edith', 'Alice'), ('Alice', 'Chicago')),
            (('Zhao', 'Alice'), ('Alice', 'NYC')),
            (('Zhao', 'Alice'), ('Alice', 'Chicago')),
            (('Zhao', 'Edith'), ('Edith', 'Paris')),
            (('Zhao', 'Edith'), ('Edith', 'Berlin')),
        ]
    )


def test_join_full_outer():
    identity = lambda x: x
    result = list(
        join(
            identity,
            [1, 2, 3],
            identity,
            [2, 3, 4],
            left_default=None,
            right_default=None,
        )
    )
    assert sorted(result, key=lambda x: (x[0] is None, x[0] or 0, x[1] or 0)) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)],
        key=lambda x: (x[0] is None, x[0] or 0, x[1] or 0),
    )


def test_join_right():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4], left_default=None))
    assert sorted(result) == [(2, 2), (3, 3), (None, 4)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4], right_default=None))
    assert sorted(result) == [(1, None), (2, 2), (3, 3)]


def test_join_index_keys():
    left = [('a', 1), ('b', 2)]
    right = [(1, 'x'), (2, 'y'), (3, 'z')]
    result = list(join(1, left, 0, right))
    assert result == [(('a', 1), (1, 'x')), (('b', 2), (2, 'y'))]


# ---------------------------------------------------------------- diff
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10),
        (None, 100),
    ]


def test_diff_key():
    assert list(
        diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)
    ) == [('bananas', 'Oranges')]


def test_diff_single_list_argument():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3])) == [(2, 2, 5)]


# ---------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie',
        'Alice',
    )


def test_topk_index_key():
    data = [{'x': 1}, {'x': 5}, {'x': 3}]
    assert topk(2, data, key='x') == ({'x': 5}, {'x': 3})


def test_topk_fewer_than_k():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------------------------------------------------------------- peek / peekn
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    first_item, rest = peek(iter([1, 2, 3]))
    assert first_item == 1
    assert list(rest) == [1, 2, 3]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# ---------------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98,
    ]
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98,
    ]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    assert list(random_sample(0.1, seq, random_state=randobj)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98,
    ]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]
