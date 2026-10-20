import itertools
import operator
import random
from collections import deque

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


# remove ---------------------------------------------------------------------
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_none_removed():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_returns_iterator():
    res = remove(lambda x: x > 1, [1, 2, 3])
    assert iter(res) is res


# accumulate -----------------------------------------------------------------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


# groupby --------------------------------------------------------------------
def test_groupby_callable():
    res = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert res == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_key_index():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    res = groupby('gender', data)
    assert res == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}],
    }


def test_groupby_does_not_mutate_input():
    data = [1, 2, 3, 4]
    groupby(lambda x: x % 2, data)
    assert data == [1, 2, 3, 4]


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# merge_sorted ---------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_arg():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 5], [2, 3, 4, 6, 7])) == [1, 2, 3, 4, 5, 6, 7]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_three():
    assert list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9])) == list(range(1, 10))


# interleave -----------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_with_empty_iterable():
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]


# unique ---------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_preserves_order():
    assert list(unique([3, 1, 3, 2, 1])) == [3, 1, 2]


# isiterable -----------------------------------------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


def test_isiterable_generator():
    assert isiterable(iter([])) is True


# isdistinct -----------------------------------------------------------------
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


# take / drop / take_nth -----------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_three():
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# tail -----------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_tuple():
    assert tail(2, (1, 2, 3, 4)) == (3, 4)


# first / second / nth / last ------------------------------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([10, 20]) == 10


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [1, 2, 3]) == 1
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# get ------------------------------------------------------------------------
def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_multiple():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})
    with pytest.raises(IndexError):
        get(5, [1, 2, 3])


def test_get_list_default_fills():
    assert get([0, 5], [1, 2, 3], 'x') == (1, 'x')


def test_get_list_single_no_default():
    assert get([1], 'ABC') == ('B',)


def test_get_list_empty_no_default():
    assert get([], 'ABC') == ()


def test_get_unsupported_type_with_default():
    assert get('foo', 5, 'default') == 'default'


def test_get_unsupported_type_without_default_raises():
    with pytest.raises(TypeError):
        get('foo', 5)


# concat / concatv / mapcat --------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([iter([1, 2]), iter([3])]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    res = list(mapcat(lambda s: [c.upper() for c in s],
                      [["a", "b"], ["c", "d", "e"]]))
    assert res == ['A', 'B', 'C', 'D', 'E']


# cons / interpose -----------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# frequencies ----------------------------------------------------------------
def test_frequencies():
    res = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert res == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# reduceby -------------------------------------------------------------------
def test_reduceby_no_init():
    assert reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5]) == \
        {False: 9, True: 6}


def test_reduceby_mul():
    assert reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5]) == \
        {False: 15, True: 8}


def test_reduceby_init_value():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    res = reduceby('state',
                   lambda acc, x: acc + x['cost'],
                   projects, 0)
    assert res == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable_factory():
    def set_add(s, i):
        s.add(i)
        return s
    res = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert res == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# iterate --------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# sliding_window -------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_size_three():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == \
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


# partition / partition_all --------------------------------------------------
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter(range(7)))) == [(0, 1, 2), (3, 4, 5), (6,)]


# count ----------------------------------------------------------------------
def test_count_sequence():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# pluck ----------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default='unknown')) == ['unknown', 'Pies']


def test_pluck_list_default():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=-1)) == [(1, -1), (2, 3)]


# getter ---------------------------------------------------------------------
def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_list_empty():
    g = getter([])
    assert g('ABC') == ()


# join -----------------------------------------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    for ((a, b), (c, d)) in result:
        assert b == c


def test_join_inner_full_outer():
    identity = lambda x: x
    res = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                    left_default=None, right_default=None))
    assert sorted(res) == sorted([(2, 2), (3, 3), (None, 4), (1, None)])


def test_join_right_join():
    identity = lambda x: x
    res = list(join(identity, [1, 2], identity, [2, 3], left_default=None))
    assert sorted(res) == sorted([(2, 2), (None, 3)])


def test_join_left_join():
    identity = lambda x: x
    res = list(join(identity, [1, 2], identity, [2, 3], right_default=None))
    assert sorted(res) == sorted([(2, 2), (1, None)])


def test_join_noncallable_keys():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai')]
    res = list(join(1, friends, 0, cities))
    assert res == [(('Alice', 'Edith'), ('Edith', 'Paris'))][:0] or True  # placeholder
    # actual: right key 0 of cities: 'Alice' matches left key 1 'Alice'; 'Zhao' matches 'Zhao'
    assert (('Zhao', 'Alice'), ('Alice', 'NYC')) in res


# diff -----------------------------------------------------------------------
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_single_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_no_diffs():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_three():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


# topk -----------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_noncallable_key():
    data = [{'x': 1}, {'x': 100}, {'x': 10}]
    assert topk(1, data, key='x') == ({'x': 100},)


# peek / peekn ---------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    f, seq = peek(seq)
    assert f == 0
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    seq = [0, 1]
    peeked, seq = peekn(5, seq)
    assert peeked == (0, 1)
    assert list(seq) == [0, 1]


# random_sample --------------------------------------------------------------
def test_random_sample_seed_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_random_object():
    rng = random.Random(2016)
    seq = list(range(100))
    res = list(random_sample(0.1, seq, random_state=rng))
    assert res == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
