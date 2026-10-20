import itertools
import math
import operator
import random
from collections import OrderedDict

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

def test_remove_filters_matching():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_returns_iterator():
    result = remove(lambda x: x > 0, [1, 2, 3])
    assert not isinstance(result, list)
    assert list(result) == []


def test_remove_empty():
    assert list(remove(bool, [])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_returns_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


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


def test_groupby_preserves_order():
    result = groupby(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert result == {1: [1, 3, 5], 0: [2, 4]}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single_arg():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_many():
    result = list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9]))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_with_empty_element():
    assert list(interleave([[1, 2], []])) == [1, 2]


def test_interleave_infinite_inner():
    gen = interleave([itertools.count(), [10, 20, 30]])
    assert [next(gen) for _ in range(6)] == [0, 10, 1, 20, 2, 30]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_unhashable_items_raises():
    with pytest.raises(TypeError):
        list(unique([[1], [2]]))


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1, 2])) is True
    assert isiterable(None) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------

def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True
    assert isdistinct(iter([])) is True


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_available():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_available():
    assert list(drop(10, [1, 2])) == []


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_second_too_short():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_out_of_range_iterator():
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


def test_nth_out_of_range_sequence():
    with pytest.raises(IndexError):
        nth(5, [1, 2])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


def test_last_empty():
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_single_index():
    assert get(1, 'ABC') == 'B'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_dict_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'missing') == 'missing'


def test_get_single_list_index():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_non_indexable_default():
    assert get(0, 42, 'default') == 'default'


def test_get_non_indexable_no_default():
    with pytest.raises(TypeError):
        get(0, 42)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_of_seqs():
    gen = (x for x in [[1], [2, 3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_with_infinite():
    assert list(take(3, cons(0, itertools.count(1)))) == [0, 1, 2]


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
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    result = reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_init_value():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(operator.neg, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_means():
    mean = lambda seq: float(sum(seq)) / len(seq)
    result = list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    assert result == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------

def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2, 3), (4, 5)]


def test_partition_all_single_element():
    assert list(partition_all(3, [1])) == [(1,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_with_len():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------

def test_pluck_single_key():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=None)) == [1, None]


def test_pluck_multiple_with_default():
    data = [[1, 2], [3]]
    result = list(pluck([0, 1], data, default=None))
    assert result == [(1, 2), (3, None)]


def test_pluck_missing_no_default():
    data = [{'a': 1}, {'b': 2}]
    with pytest.raises(KeyError):
        list(pluck('a', data))


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    left = [('a', 1), ('a', 2), ('b', 3)]
    right = [('a', 10), ('c', 30)]
    result = list(join(first, left, first, right))
    assert sorted(result) == [(('a', 1), ('a', 10)), (('a', 2), ('a', 10))]


def test_join_right():
    left = [('a', 1)]
    right = [('a', 10), ('c', 30)]
    result = list(join(first, left, first, right, left_default=None))
    assert result == [(('a', 1), ('a', 10)), (None, ('c', 30))]


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=repr) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)], key=repr)


def test_join_left():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10)]
    result = list(join(first, left, first, right, right_default=None))
    assert sorted(result) == [(('a', 1), ('a', 10)), (('b', 2), None)]


def test_join_docstring_example():
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
    result = set(join(second, friends, first, cities))
    expected = {
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
    }
    assert result == expected


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = set(join(1, friends, 0, cities))
    assert result == {
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
    }


def test_join_empty_left():
    result = list(join(first, [], first, [('a', 1)]))
    assert result == []


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_with_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 10]]))
    assert result == [(3, 10)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_key_non_callable():
    data = [{'v': 3}, {'v': 1}, {'v': 5}, {'v': 2}]
    assert topk(2, data, key='v') == ({'v': 5}, {'v': 3})


def test_topk_more_than_available():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, new_seq = peek(seq)
    assert first_item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    item, new_it = peek(it)
    assert item == 10
    assert list(new_it) == [10, 20, 30]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, new_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    first_items, new_it = peekn(10, [1, 2])
    assert first_items == (1, 2)
    assert list(new_it) == [1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_deterministic():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result2 == result


def test_random_sample_with_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, list(range(10)), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, list(range(10)), random_state=0)) == \
        list(range(10))


def test_random_sample_empty():
    assert list(random_sample(0.5, [], random_state=0)) == []
