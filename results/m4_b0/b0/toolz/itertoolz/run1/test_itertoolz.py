import itertools
import random
from operator import add, mul

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
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_all_true():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_basic():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


def test_accumulate_is_generator():
    result = accumulate(add, [1, 2, 3])
    assert next(result) == 1
    assert next(result) == 3


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_member():
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
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [], [])) == []
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 5, 9], [2, 3, 4, 10, 11])) == [
        1, 2, 3, 4, 5, 9, 10, 11
    ]


def test_merge_sorted_key_uneven():
    a = [(1, 'a'), (5, 'b')]
    b = [(2, 'c'), (3, 'd')]
    result = list(merge_sorted(a, b, key=lambda x: x[0]))
    assert result == [(1, 'a'), (2, 'c'), (3, 'd'), (5, 'b')]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_one_empty_inner():
    assert list(interleave([[], [1, 2, 3]])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == (
        'cat', 'mouse'
    )


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
def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / tail / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2, 3])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(10, [1, 2, 3])) == []


def test_tail_sequence():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


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
    assert second([1, 2, 3]) == 2


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'


def test_nth_negative_on_sequence():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_simple():
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


def test_get_single_list_index():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_missing_raises_keyerror():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_indexerror():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_indexerror_default():
    assert get(10, [1, 2, 3], 'x') == 'x'


def test_get_keyerror_default():
    assert get('missing', {'a': 1}, 'x') == 'x'


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    def gen():
        yield [1]
        yield [2, 3]
    assert list(concat(gen())) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------------------------------------------------------------------------
# interpose
# ---------------------------------------------------------------------------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


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
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}


def test_reduceby_with_constant_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data, 0) == {False: 9, True: 6}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_member_key():
    projects = [
        {'state': 'CA', 'cost': 1000000},
        {'state': 'IL', 'cost': 100000},
        {'state': 'IL', 'cost': 2000000},
        {'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [
        (1, 2), (2, 3), (3, 4)
    ]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    result = list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    assert result == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(5, [1, 2, 3])) == []


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)
    ]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_uneven_list():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [
        (1, 2), (3, 4), (5,)
    ]


def test_partition_all_uneven_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [
        (1, 2), (3, 4), (5,)
    ]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator_3():
    assert list(partition_all(3, iter(range(7)))) == [
        (0, 1, 2), (3, 4, 5), (6,)
    ]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_with_len():
    assert count([1, 2, 3]) == 3
    assert count('abc') == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, 'unknown')) == ['unknown', 'Pies']


def test_pluck_missing_raises():
    data = [{'id': 1}]
    with pytest.raises(KeyError):
        list(pluck('name', data))


def test_pluck_multi_with_default():
    data = [{'a': 1}, {'a': 2, 'b': 3}]
    assert list(pluck(['a', 'b'], data, None)) == [(1, None), (2, 3)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------
def test_getter_single():
    g = getter(0)
    assert g([10, 20, 30]) == 10


def test_getter_list():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_single_element_list():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_empty_list():
    g = getter([])
    assert g([10, 20, 30]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    assert result == [
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Zhao'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
    ]


def test_join_inner_member_keys():
    friends = [('Alice', 'Edith'), ('Bob', 'Alice')]
    cities = [('Alice', 'NYC')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Alice', 'NYC'))]


def test_join_right():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_empty():
    identity = lambda x: x
    assert list(join(identity, [], identity, [1, 2])) == []
    result = list(join(identity, [1, 2], identity, [],
                       right_default=None))
    assert result == [(1, None), (2, None)]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)
    ]


def test_diff_with_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 10]]))
    assert result == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
    assert result == [(3, 3, 4)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice'
    )


def test_topk_member_key():
    data = [{'id': 1}, {'id': 100}, {'id': 50}]
    assert topk(2, data, key='id') == ({'id': 100}, {'id': 50})


def test_topk_empty():
    assert topk(2, []) == ()


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, new_seq = peek(seq)
    assert item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, new_seq = peek(iter([5, 6, 7]))
    assert item == 5
    assert list(new_seq) == [5, 6, 7]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, new_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    items, new_seq = peekn(10, [1, 2, 3])
    assert items == (1, 2, 3)
    assert list(new_seq) == [1, 2, 3]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_seeded():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_with_random_obj():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_lazy():
    # random_sample returns a lazy filter object, not a list
    result = random_sample(0.5, [1, 2, 3], random_state=0)
    assert not isinstance(result, list)
    assert iter(result) is result


# ---------------------------------------------------------------------------
# no_default sentinel sanity
# ---------------------------------------------------------------------------
def test_no_default_sentinel_is_singleton():
    assert no_default is get.__defaults__[0]
