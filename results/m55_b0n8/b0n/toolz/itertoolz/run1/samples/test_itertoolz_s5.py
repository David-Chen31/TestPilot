import itertools
import operator
from collections import deque
from functools import reduce

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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_is_lazy_filterfalse():
    result = remove(lambda x: x > 0, [1, -2, 3])
    assert list(result) == [-2]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

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


def test_accumulate_single_element():
    assert list(accumulate(operator.add, [5])) == [5]


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_index():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert len(result['M']) == 2


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_preserves_order():
    assert groupby(lambda x: 0, ['a', 'b', 'c']) == {0: ['a', 'b', 'c']}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3, 4], [5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([5], [1, 2, 3, 4])) == [1, 2, 3, 4, 5]


def test_merge_sorted_many_seqs():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], [1, 2]])) == [1, 2]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------

def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take, drop, take_nth, first, second, nth, last, tail
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_tuple():
    assert tail(2, (10, 20, 30)) == (20, 30)


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40])) == (30, 40)


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_first():
    assert first('ABC') == 'A'


def test_second():
    assert second('ABC') == 'B'


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(-1, [10, 20]) == 20


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_multiple():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_single_list_missing():
    assert get('Dennis', {'Alice': 1}, 'x') == 'x'


def test_get_list_default():
    assert get(['a', 'b'], {'a': 1}, None) == (1, None)


def test_get_empty_list():
    assert get([], [1, 2, 3]) == ()


def test_get_single_element_list():
    assert get([0], [1, 2, 3]) == (1,)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_index_error_default():
    assert get(10, [1, 2], None) is None


def test_get_type_error_default():
    assert get('a', 5, None) is None


# ---------------------------------------------------------------------------
# concat, concatv, mapcat, cons, interpose
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2]]))) == [1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    iseven = lambda x: x % 2 == 0
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state',
                      lambda acc, x: acc + x['cost'],
                      projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert list(itertools.islice(counter, 4)) == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert list(itertools.islice(powers, 4)) == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# partition, partition_all
# ---------------------------------------------------------------------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter(range(5)))) == [(0, 1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_sequence():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------

def test_pluck_single_key():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=None)) == [1, None]


def test_pluck_multiple_default():
    data = [{'a': 1}, {'a': 2, 'b': 3}]
    assert list(pluck(['a', 'b'], data, default=0)) == [(1, 0), (2, 3)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    assert getter(1)([10, 20, 30]) == 20


def test_getter_list():
    assert getter([0, 2])([10, 20, 30]) == (10, 30)


def test_getter_single_list():
    assert getter([1])([10, 20, 30]) == (20,)


def test_getter_empty_list():
    assert getter([])([10, 20, 30]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    friends = [('Alice', 'Edith'),
               ('Alice', 'Zhao'),
               ('Edith', 'Alice'),
               ('Zhao', 'Alice'),
               ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'),
              ('Alice', 'Chicago'),
              ('Dan', 'Sydney'),
              ('Edith', 'Paris'),
              ('Edith', 'Berlin'),
              ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    pairs = sorted(set((a, d) for ((a, b), (c, d)) in result))
    assert pairs == [('Alice', 'Berlin'),
                     ('Alice', 'Paris'),
                     ('Alice', 'Shanghai'),
                     ('Edith', 'Chicago'),
                     ('Edith', 'NYC'),
                     ('Zhao', 'Berlin'),
                     ('Zhao', 'Chicago'),
                     ('Zhao', 'Paris'),
                     ('Zhao', 'NYC')]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert (2, 2) in result
    assert (3, 3) in result
    assert (None, 4) in result
    assert (1, None) in result


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2],
                       identity, [2, 3],
                       left_default=None))
    assert (2, 2) in result
    assert (None, 3) in result
    assert len(result) == 2


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2],
                       identity, [2, 3],
                       right_default=None))
    assert (2, 2) in result
    assert (1, None) in result
    assert len(result) == 2


def test_join_index_keys():
    friends = [('Alice', 'Edith')]
    cities = [('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_single_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 5, 3], [1, 2, 3])) == [(2, 5, 2)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_getter_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


# ---------------------------------------------------------------------------
# peek, peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    first_item, rest = peek(iter([5, 6, 7]))
    assert first_item == 5
    assert list(rest) == [5, 6, 7]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_seed_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2
    assert r1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=1)) == []


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=1)) == seq


def test_random_sample_random_object():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
