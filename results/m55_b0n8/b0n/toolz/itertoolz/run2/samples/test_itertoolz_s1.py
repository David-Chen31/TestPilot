import itertools
import operator
from collections import OrderedDict

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join, diff,
    topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# remove / accumulate
# ---------------------------------------------------------------------------

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_returns_generator():
    result = accumulate(operator.add, [1, 2, 3])
    assert itertools.islice(result, 2) is not None
    assert list(result) == [1, 3, 6]


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable():
    assert groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']) == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_iseven():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


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


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_and_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted(iter([3, 2, 1]))) == [3, 2, 1]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [])) == []


def test_merge_sorted_key_empty():
    assert list(merge_sorted([], [1, 2], key=lambda x: x)) == [1, 2]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[]])) == []
    assert list(interleave([[], []])) == []


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_unhashable_key():
    data = [[1, 2], [1, 2], [3]]
    result = list(unique(data, key=tuple))
    assert result == [[1, 2], [3]]


# ---------------------------------------------------------------------------
# isiterable / isdistinct
# ---------------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / tail / drop / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (1, 2, 3, 4)) == (3, 4)


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)


def test_tail_more_than_length():
    assert tail(10, (1, 2)) == (1, 2)


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second('A')


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [1, 2, 3]) == 1
    assert nth(1, (10, 20, 30)) == 20


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'
    assert nth(2, iter([10, 20, 30])) == 30


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(10, [1, 2, 3])
    with pytest.raises(StopIteration):
        nth(10, iter([1, 2, 3]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3, 4]) == 4


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_basic():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default_scalar():
    assert get(10, 'ABC', None) is None
    assert get('missing', {'a': 1}, 'default') == 'default'


def test_get_missing_raises():
    with pytest.raises(IndexError):
        get(10, 'ABC')
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons / interpose
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []
    assert list(concat(iter([iter([1, 2]), iter([3])]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies / reduceby / iterate
# ---------------------------------------------------------------------------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_constant():
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

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]

    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    result = list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    assert result == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(2, [1])) == []


def test_sliding_window_exact():
    assert list(sliding_window(2, [1, 2])) == [(1, 2)]


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------

def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


def test_partition_pad_custom():
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == [(1, 2, 3), (4, 0, 0)]


def test_partition_empty():
    assert list(partition(2, [])) == []


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_single():
    assert list(partition_all(3, [1])) == [(1,)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count('abc') == 3
    assert count([]) == 0


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------

def test_pluck_dict():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_default_list():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, None), (2, 3)]


def test_getter_scalar():
    g = getter('name')
    assert g({'name': 'Alice'}) == 'Alice'


def test_getter_single_list():
    g = getter(['name'])
    assert g({'name': 'Alice'}) == ('Alice',)


def test_getter_multi_list():
    g = getter(['a', 'b'])
    assert g({'a': 1, 'b': 2}) == (1, 2)


def test_getter_empty_list():
    g = getter([])
    assert g({'a': 1}) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    assert ('Alice', 'Edith', 'Paris') not in result
    # Alice -> Edith; Edith->Paris because (Alice, Edith) and ('Alice','NYC')
    assert ('Alice', 'Edith') not in result
    assert set(result) == {('Alice', 'NYC'), ('Edith', 'Paris')}


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], left_default=None))
    assert set(result) == {(2, 2), (None, 3)}


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], right_default=None))
    assert set(result) == {(2, 2), (1, None)}


def test_join_key_indices():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC')]
    result = list(join(0, friends, 0, cities))
    assert set(result) == {('Alice', 'Edith'), ('Alice', 'NYC')}


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 5, 3], [1, 2, 3]))
    assert result == [(2, 5, 2)]


def test_diff_single_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 10]]))
    assert result == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key_callable():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_key_index():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_more_than_length():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_val, rest = peek(seq)
    assert first_val == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    first_val, rest = peek(iter([10, 20, 30]))
    assert first_val == 10
    assert list(rest) == [10, 20, 30]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    first_vals, rest = peekn(10, [0, 1])
    assert first_vals == (0, 1)
    assert list(rest) == [0, 1]


def test_peekn_zero():
    first_vals, rest = peekn(0, [0, 1, 2])
    assert first_vals == ()
    assert list(rest) == [0, 1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_deterministic():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert result1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]


def test_random_sample_random_obj():
    import random as random_mod
    randobj = random_mod.Random(2016)
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
