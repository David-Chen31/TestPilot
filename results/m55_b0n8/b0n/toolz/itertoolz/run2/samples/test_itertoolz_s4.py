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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_basic():
    def iseven(x):
        return x % 2 == 0

    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_lazy_and_infinite():
    result = remove(lambda x: x < 5, itertools.count())
    assert list(take(3, result)) == [5, 6, 7]


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


def test_accumulate_accepts_iterator():
    assert list(accumulate(operator.add, iter([1, 2, 3]))) == [1, 3, 6]


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable_key():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_string_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_non_callable_member_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert result['M'] == [{'name': 'Bob', 'gender': 'M'},
                           {'name': 'Charlie', 'gender': 'M'}]
    assert set(result) == {'F', 'M'}


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_returns_plain_dict():
    result = groupby(lambda x: x, [1, 1, 2])
    assert type(result) is dict
    assert result == {1: [1, 1], 2: [2]}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    result = merge_sorted([3, 2, 1])
    assert list(result) == [3, 2, 1]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_middle():
    assert list(merge_sorted([1, 2], [], [3, 4])) == [1, 2, 3, 4]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3], [4])) == [1, 2, 3, 4]
    assert list(merge_sorted([4], [1, 2, 3])) == [1, 2, 3, 4]


def test_merge_sorted_many():
    a = [1, 4, 7]
    b = [2, 5, 8]
    c = [3, 6, 9]
    assert list(merge_sorted(a, b, c)) == list(range(1, 10))


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_with_empty_inner():
    assert list(interleave([[1, 2], [], [3]])) == [1, 3, 2]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_no_key():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == (
        'cat', 'mouse')


def test_unique_lazy():
    result = unique(itertools.count())
    assert list(take(3, result)) == [0, 1, 2]


def test_unique_empty():
    assert list(unique([])) == []


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------

def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(iter([])) is True


def test_isiterable_false():
    assert isiterable(5) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------

def test_isdistinct_lists():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_empty():
    assert isdistinct([]) is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_len():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_lazy():
    assert list(take(4, take_nth(3, itertools.count()))) == [0, 3, 6, 9]


# ---------------------------------------------------------------------------
# tail / first / second / nth / last
# ---------------------------------------------------------------------------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_empty():
    assert tail(2, []) == []


def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_second_too_short_raises():
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(1, iter([10, 20, 30])) == 20
    assert nth(2, (1, 2, 3)) == 3


def test_nth_negative_on_sequence():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


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


def test_get_default_missing():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, None) is None


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_missing_raises_without_default():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})


def test_get_index_error_raises_without_default():
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_default_on_type_error():
    # int not indexable -> TypeError branch returns default
    assert get(0, 5, 'default') == 'default'


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_lazy_generator():
    def gen():
        yield [1]
        yield [2, 3]
    assert list(concat(gen())) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == list('ABCDE')


# ---------------------------------------------------------------------------
# cons
# ---------------------------------------------------------------------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_infinite():
    assert list(take(3, cons(0, itertools.count(1)))) == [0, 1, 2]


# ---------------------------------------------------------------------------
# interpose
# ---------------------------------------------------------------------------

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


def test_frequencies_returns_plain_dict():
    result = frequencies([1, 1, 2])
    assert type(result) is dict


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------

def test_reduceby_add():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul_with_init():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.mul, [1, 2, 3, 4, 5], 1)
    assert result == {False: 15, True: 8}


def test_reduceby_callable_init():
    iseven = lambda x: x % 2 == 0

    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_string_key():
    projects = [{'state': 'CA', 'cost': 100},
                {'state': 'IL', 'cost': 50},
                {'state': 'IL', 'cost': 25}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 100, 'IL': 75}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate_inc():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]


def test_iterate_double():
    result = list(take(5, iterate(lambda x: x * 2, 1)))
    assert result == [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(2, [1])) == []


def test_sliding_window_n_equal_len():
    assert list(sliding_window(3, [1, 2, 3])) == [(1, 2, 3)]


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------

def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder_list():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_remainder_iterator():
    # No __len__ -> binary-search branch
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [
        (1, 2), (3, 4), (5,)]


def test_partition_all_smaller_than_n():
    assert list(partition_all(3, [1])) == [(1,)]
    assert list(partition_all(3, iter([1, 2]))) == [(1, 2)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_exact_multiple_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4]))) == [(1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


def test_count_generator():
    assert count(x for x in range(5)) == 5


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------

def test_pluck_string():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_single_list_index():
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]


def test_pluck_default():
    data = [{'id': 1}, {'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_indices_default():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, 2), (3, None)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_multi():
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
    assert result == [(('a', 1), ('a', 10)), (('a', 2), ('a', 10))]


def test_join_inner_empty():
    assert list(join(first, [], first, [('a', 1)])) == []


def test_join_right():
    left = [('a', 1)]
    right = [('a', 10), ('c', 30)]
    result = list(join(first, left, first, right, left_default=None))
    assert (('a', 1), ('a', 10)) in result
    assert (None, ('c', 30)) in result
    assert len(result) == 2


def test_join_left():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10)]
    result = list(join(first, left, first, right, right_default=None))
    assert (('a', 1), ('a', 10)) in result
    assert (('b', 2), None) in result
    assert len(result) == 2


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}


def test_join_non_callable_keys():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10)]
    result = list(join(0, left, 0, right))
    assert result == [(('a', 1), ('a', 10))]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_with_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 2], [1, 5], [1, 6]))
    assert result == [(2, 5, 6)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 5}, {'v': 3}]
    assert topk(2, data, key='v') == ({'v': 5}, {'v': 3})


def test_topk_more_than_len():
    assert topk(10, [3, 1, 2]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    f, seq2 = peek(seq)
    assert f == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    f, seq2 = peek(iter([5, 6, 7]))
    assert f == 5
    assert list(seq2) == [5, 6, 7]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq2 = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_fewer_than_n():
    peeked, seq2 = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(seq2) == [1, 2]


def test_peekn_empty():
    peeked, seq2 = peekn(2, [])
    assert peeked == ()
    assert list(seq2) == []


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_seed_object():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=random.Random(2016)))
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, list(range(10)), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, list(range(10)), random_state=1)) == list(
        range(10))


def test_random_sample_lazy():
    result = random_sample(0.5, itertools.count(), random_state=0)
    assert list(result)[:3] is not None  # bounded consumed


def test_random_sample_uses_custom_random():
    class Always:
        def random(self):
            return 0.0

    assert list(random_sample(0.5, [1, 2, 3], random_state=Always())) == [
        1, 2, 3]
