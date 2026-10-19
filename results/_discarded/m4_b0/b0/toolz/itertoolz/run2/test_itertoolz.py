import itertools
import operator
import random
from functools import partial

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------- remove ---
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: x > 0, [-1, -2])) == [-1, -2]
    assert list(remove(lambda x: True, [])) == []


# ------------------------------------------------------------ accumulate ---
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_initial_zero():
    # 0 is a legitimate initial, not no_default
    assert list(accumulate(operator.add, [1, 2], 0)) == [0, 1, 3]


# --------------------------------------------------------------- groupby ---
def test_groupby_callable():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {3: ['Bob', 'Dan'],
                      5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_non_callable():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [data[0]], 'M': [data[1], data[2]]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------- merge_sorted ---
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_arg():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_empty_iterables():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [])) == []


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_many():
    seqs = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
    assert list(merge_sorted(*seqs)) == list(range(12))


# ------------------------------------------------------------ interleave ---
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_outer():
    assert list(interleave([])) == []


def test_interleave_with_empty_inner():
    assert list(interleave([[1], [], [2, 3]])) == [1, 2, 3]


# ---------------------------------------------------------------- unique ---
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert list(unique([])) == []


# ------------------------------------------------------------ isiterable ---
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True
    assert isiterable(None) is False


# ------------------------------------------------------------ isdistinct ---
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True
    assert isdistinct([]) is True


# ------------------------------------------------------------- take/drop ---
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ------------------------------------------------------------------ tail ---
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (10, 20, 30, 40, 50)) == (40, 50)
    assert tail(2, "abcde") == "de"


def test_tail_iterator():
    result = tail(2, iter([1, 2, 3, 4, 5]))
    assert result == (4, 5)


def test_tail_n_greater_than_len():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


# ------------------------------------------------------------- take_nth ---
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6])) == [1, 4]


# --------------------------------------------------- first/second/nth/last ---
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([1, 2, 3])) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ------------------------------------------------------------------- get ---
def test_get_single():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([0], 'ABC') == ('A',)
    assert get([], 'ABC') == ()


def test_get_list_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default():
    assert get('missing', {'a': 1}, default=42) == 42
    assert get(100, [1, 2], default=None) is None


def test_get_type_error_default():
    assert get('bad', [1, 2], default='x') == 'x'


def test_get_type_error_no_default_raises():
    with pytest.raises(TypeError):
        get('bad', [1, 2])


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        get('missing', {})


def test_get_missing_index_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2])


# --------------------------------------------------- concat/concatv/mapcat ---
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    gen = (x for x in [[1, 2], [3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------- cons/interpose ---
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ----------------------------------------------------------- frequencies ---
def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# -------------------------------------------------------------- reduceby ---
def test_reduceby_no_init():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_constant_init():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'state': 'CA', 'cost': 1000000},
                {'state': 'IL', 'cost': 100000}]
    result = reduceby('state',
                      lambda acc, x: acc + x['cost'],
                      projects, 0)
    assert result == {'CA': 1000000, 'IL': 100000}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# -------------------------------------------------------------- iterate ---
def test_iterate():
    inc = lambda x: x + 1
    counter = iterate(inc, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert list(itertools.islice(powers, 5)) == [1, 2, 4, 8, 16]


# --------------------------------------------------------- sliding_window ---
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == \
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == \
        [1.5, 2.5, 3.5]


# ------------------------------------------------------------- partition ---
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_uneven_drop():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_uneven_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_pad_zero():
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == [(1, 2, 3), (4, 0, 0)]


# --------------------------------------------------------- partition_all ---
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_uneven():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


def test_partition_all_empty():
    assert list(partition_all(3, [])) == []


def test_partition_all_n_three():
    assert list(partition_all(3, range(7))) == [(0, 1, 2), (3, 4, 5), (6,)]


# ----------------------------------------------------------------- count ---
def test_count_len():
    assert count([1, 2, 3]) == 3
    assert count("abc") == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count(iter([])) == 0


# ----------------------------------------------------------------- pluck ---
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_default():
    data = [[1], [1, 2]]
    assert list(pluck([0, 1], data, default=0)) == [(1, 0), (1, 2)]


# ---------------------------------------------------------------- getter ---
def test_getter_single():
    g = getter(0)
    assert g([7, 8]) == 7


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g([1, 2, 3]) == (1, 3)


def test_getter_list_single():
    g = getter([1])
    assert g([1, 2, 3]) == (2,)


def test_getter_empty():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ------------------------------------------------------------------ join ---
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
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


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 3],
                       left_default=None))
    assert result == [(1, 1), (None, 3)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 3],
                       right_default=None))
    assert result == [(1, 1), (2, None)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = set(join(1, friends, 0, cities))
    assert result == {((('Alice', 'Edith')), ('Edith', 'Paris')),
                      ((('Zhao', 'Alice')), ('Alice', 'NYC'))}


# ------------------------------------------------------------------ diff ---
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 9, 3])) == [(2, 2, 9)]


# ------------------------------------------------------------------ topk ---
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_empty():
    assert topk(2, []) == ()


# ------------------------------------------------------------------ peek ---
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ----------------------------------------------------------------- peekn ---
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    items, seq2 = peekn(2, seq)
    assert items == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_fewer_than_n():
    seq = [0, 1]
    items, seq2 = peekn(5, seq)
    assert items == (0, 1)
    assert list(seq2) == [0, 1]


# --------------------------------------------------------- random_sample ---
def test_random_sample_all():
    assert list(random_sample(1.0, [1, 2, 3, 4])) == [1, 2, 3, 4]


def test_random_sample_none():
    assert list(random_sample(0.0, [1, 2, 3, 4])) == []


def test_random_sample_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_obj():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_deterministic():
    seq = list(range(50))
    a = list(random_sample(0.3, seq, random_state=42))
    b = list(random_sample(0.3, seq, random_state=42))
    assert a == b
