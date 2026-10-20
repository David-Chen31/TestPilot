import itertools
import operator
from functools import partial

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


# --------------------------------------------------------------------------
# remove
# --------------------------------------------------------------------------
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_returns_iterator():
    result = remove(lambda x: x > 0, [1, 2, 3])
    assert iter(result) is result
    assert list(result) == []


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# --------------------------------------------------------------------------
# accumulate
# --------------------------------------------------------------------------
def test_accumulate_default():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6


# --------------------------------------------------------------------------
# groupby
# --------------------------------------------------------------------------
def test_groupby_callable_key():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_noncallable_key():
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


def test_groupby_returns_plain_dict():
    result = groupby(len, ['a', 'bb'])
    assert type(result) is dict


# --------------------------------------------------------------------------
# merge_sorted
# --------------------------------------------------------------------------
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_no_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_arg():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_uneven_sizes():
    assert list(merge_sorted([1, 2], [3, 4, 5, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([3, 4, 5, 6], [1, 2])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


def test_merge_sorted_key_partial():
    data = [(1, 'a'), (3, 'b')]
    data2 = [(2, 'c'), (4, 'd')]
    assert list(merge_sorted(data, data2, key=lambda t: t[0])) == [
        (1, 'a'), (2, 'c'), (3, 'b'), (4, 'd')
    ]


# --------------------------------------------------------------------------
# interleave
# --------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


# --------------------------------------------------------------------------
# unique
# --------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == (
        'cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# --------------------------------------------------------------------------
# isiterable
# --------------------------------------------------------------------------
@pytest.mark.parametrize("val,expected", [
    ([1, 2, 3], True),
    ('abc', True),
    (5, False),
    (None, False),
    ((i for i in range(3)), True),
    ({'a': 1}, True),
])
def test_isiterable(val, expected):
    assert isiterable(val) is expected


# --------------------------------------------------------------------------
# isdistinct
# --------------------------------------------------------------------------
def test_isdistinct_lists():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# --------------------------------------------------------------------------
# take / drop / tail / take_nth
# --------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(5, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(5, [1, 2])) == []


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3, 4, 5])) == (4, 5)


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# --------------------------------------------------------------------------
# first / second / nth / last
# --------------------------------------------------------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth_seq():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# --------------------------------------------------------------------------
# get
# --------------------------------------------------------------------------
def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'unknown') == 'unknown'


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})


def test_get_list_missing_no_default_raises():
    with pytest.raises(KeyError):
        get(['Alice', 'Dennis'], {'Alice': 1})


def test_get_empty_list():
    assert get([], 'ABC') == ()
    assert get([], 'ABC', None) == ()


def test_get_single_list_no_default():
    assert get([1], 'ABC') == ('B',)


def test_get_noniterable_with_default():
    assert get(0, 5, 'default') == 'default'


# --------------------------------------------------------------------------
# concat / concatv / mapcat / cons / interpose
# --------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_input():
    assert list(concat(iter([[1, 2], [3]]))) == [1, 2, 3]


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


# --------------------------------------------------------------------------
# frequencies
# --------------------------------------------------------------------------
def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# --------------------------------------------------------------------------
# reduceby
# --------------------------------------------------------------------------
def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul_no_init():
    result = reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_with_init():
    data = [1, 2, 3, 4, 5]
    result = reduceby(lambda x: x % 2 == 0, operator.add, data, 0)
    assert result == {False: 9, True: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_noncallable_key():
    data = [{'state': 'CA', 'cost': 100},
            {'state': 'IL', 'cost': 200},
            {'state': 'CA', 'cost': 300}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], data, 0)
    assert result == {'CA': 400, 'IL': 200}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# --------------------------------------------------------------------------
# iterate
# --------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_powers_of_two():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# --------------------------------------------------------------------------
# sliding_window
# --------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    result = list(map(mean, sliding_window(2, [1, 2, 3, 4])))
    assert result == [1.5, 2.5, 3.5]


def test_sliding_window_shorter_than_n():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


def test_sliding_window_n_equal_length():
    assert list(sliding_window(3, [1, 2, 3])) == [(1, 2, 3)]


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_partial():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


def test_partition_pad_value_zero():
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == [
        (1, 2, 3), (4, 0, 0)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# --------------------------------------------------------------------------
# partition_all
# --------------------------------------------------------------------------
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_partial():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_with_len():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_iterator_input():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [
        (1, 2), (3, 4), (5,)]


def test_partition_all_shorter_than_n():
    assert list(partition_all(3, [1, 2])) == [(1, 2)]


# --------------------------------------------------------------------------
# count
# --------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# --------------------------------------------------------------------------
# pluck / getter
# --------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list_index():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, 'unknown')) == ['unknown', 'Pies']


def test_pluck_default_list():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, None)) == [(1, 2), (3, None)]


def test_getter_single():
    get_fn = getter(1)
    assert get_fn('ABC') == 'B'


def test_getter_list_one():
    get_fn = getter([1])
    assert get_fn('ABC') == ('B',)


def test_getter_list_multiple():
    get_fn = getter([0, 2])
    assert get_fn('ABC') == ('A', 'C')


def test_getter_empty_list():
    get_fn = getter([])
    assert get_fn('ABC') == ()


# --------------------------------------------------------------------------
# join
# --------------------------------------------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    expected = [
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
    ]
    assert sorted(result) == sorted(expected)


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result) == sorted([(2, 2), (3, 3), (None, 4), (1, None)])


def test_join_right():
    result = list(join(lambda x: x % 2, [1, 2], lambda x: x % 2, [2, 3, 4],
                       left_default=None))
    assert sorted(result) == sorted([(2, 2), (None, 3), (2, 4)])


def test_join_left():
    result = list(join(lambda x: x % 2, [1, 2], lambda x: x % 2, [2, 4, 5],
                       right_default=None))
    assert sorted(result) == sorted([(2, 2), (2, 4), (1, None)])


def test_join_noncallable_keys():
    result = list(join(0, [(1, 'a'), (2, 'b')], 0, [(2, 'c'), (3, 'd')]))
    assert result == [((2, 'b'), (2, 'c'))]


def test_join_empty():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_single_list_arg():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 1, 1], [1, 1, 2], [1, 1, 3]))
    assert result == [(1, 1, 2), (1, 1, 3)]


# --------------------------------------------------------------------------
# topk
# --------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_noncallable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_k_larger_than_seq():
    assert topk(5, [1, 2]) == (2, 1)


# --------------------------------------------------------------------------
# peek
# --------------------------------------------------------------------------
def test_peek_list():
    first_elem, seq = peek([0, 1, 2, 3, 4])
    assert first_elem == 0
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    first_elem, seq = peek(iter([1, 2, 3]))
    assert first_elem == 1
    assert list(seq) == [1, 2, 3]


# --------------------------------------------------------------------------
# peekn
# --------------------------------------------------------------------------
def test_peekn():
    first_two, seq = peekn(2, [0, 1, 2, 3, 4])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, seq = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(seq) == [1, 2]


def test_peekn_iterator():
    peeked, seq = peekn(2, iter([1, 2, 3, 4]))
    assert peeked == (1, 2)
    assert list(seq) == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# random_sample
# --------------------------------------------------------------------------
def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=2016)) == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=2016)) == []


def test_random_sample_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a  # non-empty for this seed/prob


def test_random_sample_with_random_obj():
    seq = list(range(100))
    import random
    randobj = random.Random(2016)
    a = list(random_sample(0.1, seq, random_state=randobj))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b


def test_random_sample_subset():
    seq = list(range(100))
    sampled = list(random_sample(0.1, seq, random_state=2016))
    for item in sampled:
        assert item in seq
