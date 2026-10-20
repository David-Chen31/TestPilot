import itertools
import operator
import collections
import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join, diff,
    topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ----------------------------- remove -----------------------------
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_all():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_none():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


# ----------------------------- accumulate -----------------------------
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


# ----------------------------- groupby -----------------------------
def test_groupby_callable():
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    result = groupby(lambda x: x % 2 == 0, data)
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_non_callable():
    data = [{'gender': 'F', 'name': 'Alice'},
            {'gender': 'M', 'name': 'Bob'},
            {'gender': 'M', 'name': 'Charlie'}]
    result = groupby('gender', data)
    assert result['F'] == [{'gender': 'F', 'name': 'Alice'}]
    assert len(result['M']) == 2
    assert list(result.keys()) == ['F', 'M'] or set(result.keys()) == {'F', 'M'}


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# ----------------------------- merge_sorted -----------------------------
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1], [2, 3, 4])) == [1, 2, 3, 4]
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


def test_merge_sorted_many():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


# ----------------------------- interleave -----------------------------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ----------------------------- unique -----------------------------
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ----------------------------- isiterable -----------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ----------------------------- isdistinct -----------------------------
def test_isdistinct_list():
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


# ----------------------------- take / tail / drop / take_nth -----------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (10, 20, 30)) == (20, 30)
    assert tail(2, iter([10, 20, 30])) == (20, 30)


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# ----------------------------- first / second / nth / last -----------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([1, 2, 3])) == 2


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ----------------------------- get -----------------------------
def test_get_basic():
    assert get(1, 'ABC') == 'B'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([1], 'ABC') == ('B',)
    assert get([], 'ABC') == ()


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('missing', phonebook, 'default') == 'default'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_list_missing_raises():
    with pytest.raises(KeyError):
        get(['missing'], {'a': 1})


def test_get_type_error_default():
    assert get(object(), [1, 2, 3], 'default') == 'default'


def test_get_type_error_raises():
    with pytest.raises(TypeError):
        get(object(), [1, 2, 3])


# ----------------------------- concat / concatv -----------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    gen = (x for x in [[1], [2], [3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ----------------------------- mapcat -----------------------------
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ----------------------------- cons -----------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_cons_infinite():
    result = list(take(4, cons(0, itertools.count(1))))
    assert result == [0, 1, 2, 3]


# ----------------------------- interpose -----------------------------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ----------------------------- frequencies -----------------------------
def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ----------------------------- reduceby -----------------------------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_constant_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'name': 'a', 'state': 'CA', 'cost': 100},
                {'name': 'b', 'state': 'IL', 'cost': 200},
                {'name': 'c', 'state': 'IL', 'cost': 300}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 100, 'IL': 500}


# ----------------------------- iterate -----------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(5)] == [0, 1, 2, 3, 4]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ----------------------------- sliding_window -----------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_n1():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ----------------------------- partition -----------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


def test_partition_n1():
    assert list(partition(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ----------------------------- partition_all -----------------------------
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_list_type():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


# ----------------------------- count -----------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ----------------------------- pluck -----------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, None)) == [None, 'Pies']


def test_pluck_list_default():
    data = [[1, 2], [3, 4, 5]]
    assert list(pluck([0, 2], data, None)) == [(1, None), (3, 5)]


# ----------------------------- getter -----------------------------
def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list_single():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([10, 20, 30]) == (10, 20)


def test_getter_list_empty():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ----------------------------- join -----------------------------
def test_join_inner():
    left = [1, 2, 3]
    right = [2, 3, 4]
    identity = lambda x: x
    result = list(join(identity, left, identity, right))
    assert sorted(result) == [(2, 2), (3, 3)]


def test_join_right():
    left = [1, 2]
    right = [2, 3]
    identity = lambda x: x
    result = list(join(identity, left, identity, right, left_default=None))
    assert sorted(result, key=lambda x: (x[1], x[0] == None)) == \
        [(2, 2), (None, 3)]


def test_join_left():
    left = [1, 2]
    right = [2, 3]
    identity = lambda x: x
    result = list(join(identity, left, identity, right, right_default=None))
    assert set(result) == {(2, 2), (1, None)}


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert set(result) == {(('Alice', 'Edith'), ('Edith', 'Paris')),
                           (('Alice', 'Zhao'), ('Alice', 'NYC'))}


def test_join_empty():
    identity = lambda x: x
    assert list(join(identity, [], identity, [1, 2])) == []


# ----------------------------- diff -----------------------------
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three():
    assert list(diff([1, 2], [1, 2], [1, 3])) == [(2, 2, 3)]


def test_diff_single_list_argument():
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ----------------------------- topk -----------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'x': 1}, {'x': 5}, {'x': 3}]
    result = topk(2, data, key='x')
    assert result == ({'x': 5}, {'x': 3})


def test_topk_more_than_length():
    assert set(topk(10, [1, 2, 3])) == {1, 2, 3}


# ----------------------------- peek / peekn -----------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([0, 1, 2])
    item, rest = peek(it)
    assert item == 0
    assert list(rest) == [0, 1, 2]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_fewer():
    peeked, rest = peekn(5, [0, 1])
    assert peeked == (0, 1)
    assert list(rest) == [0, 1]


# ----------------------------- random_sample -----------------------------
def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_randomobj():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_zero():
    seq = list(range(100))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_one():
    seq = list(range(10))
    # Deterministic Random(0) yields random() < 1 for all items
    assert list(random_sample(1.0, seq, random_state=0)) == seq
