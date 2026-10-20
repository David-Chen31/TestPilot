import itertools
import random
import operator
from functools import partial, reduce

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
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_all_removed():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_none_removed():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_is_lazy_iterator():
    result = remove(lambda x: x > 2, [1, 2, 3, 4])
    assert not isinstance(result, list)
    assert list(result) == [1, 2]


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


def test_accumulate_empty_without_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
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
    assert groupby(lambda x: x, []) == {}


def test_groupby_preserves_order():
    result = groupby(lambda x: x % 2, [1, 3, 2, 4])
    assert result[1] == [1, 3]
    assert result[0] == [2, 4]


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_with_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_empty_seqs():
    assert list(merge_sorted([], [1, 2, 3], [])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_outer():
    assert list(interleave([])) == []


def test_interleave_with_empty_inner():
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_is_lazy():
    result = unique([1, 1, 2, 2, 3])
    assert next(result) == 1
    assert next(result) == 2
    assert next(result) == 3
    with pytest.raises(StopIteration):
        next(result)


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(iter([])) is True


def test_isiterable_false():
    assert isiterable(5) is False
    assert isiterable(None) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2])) == []


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last / tail
# ---------------------------------------------------------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([10, 20]) == 10


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([10, 20, 30]) == 20


def test_second_too_short_raises():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_on_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30


def test_nth_on_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2, 3]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3, 4]) == 4


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'ABCDE') == 'DE'


def test_tail_on_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_length():
    assert tail(10, [1, 2]) == [1, 2]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_single():
    assert get(1, 'ABC') == 'B'
    assert get(0, [10, 20, 30]) == 10


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list_of_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_list_with_missing_and_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_single_with_default():
    assert get('Dennis', {'Alice': 1}, None) is None


def test_get_missing_single_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})


def test_get_index_error_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_unindexable_with_default():
    assert get(0, 5, 'default') == 'default'


def test_get_unindexable_without_default_raises():
    with pytest.raises(TypeError):
        get(0, 5)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_empty():
    assert list(concatv()) == []


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
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init_constant():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {False: {1, 3}, True: {2, 4}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


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
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(2, [1])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_on_iterator():
    # Iterator has no __len__, exercising the binary search path
    def gen():
        for i in range(1, 6):
            yield i

    assert list(partition_all(2, gen())) == [(1, 2), (3, 4), (5,)]


def test_partition_all_one():
    assert list(partition_all(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


def test_count_generator():
    assert count(i for i in range(10)) == 10


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2}]
    assert list(pluck('name', data, 'unknown')) == ['Cheese', 'unknown']


def test_pluck_multiple_with_default():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, None)) == [(1, 2), (3, None)]


def test_getter_single():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list_one():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    left = [('a', 1), ('b', 2), ('c', 3)]
    right = [('b', 20), ('c', 30), ('d', 40)]

    def lk(x):
        return x[0]

    def rk(x):
        return x[0]

    result = sorted(join(lk, left, rk, right))
    assert result == [(('b', 2), ('b', 20)), (('c', 3), ('c', 30))]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda p: (p[0] is None, p[0], p[1] is None,
                                          p[1] if p[1] is not None else 0)) \
        == [(1, None), (2, 2), (3, 3), (None, 4)]


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None))
    assert sorted(result, key=lambda p: (p[0] is None, p[0], p[1])) == \
        [(2, 2), (3, 3), (None, 4)]


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       right_default=None))
    assert sorted(result, key=lambda p: (p[0], p[1] is None, p[1] or 0)) == \
        [(1, None), (2, 2), (3, 3)]


def test_join_with_noncallable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert sorted(result) == sorted([
        (('Alice', 'Edith'), ('Edith', 'Paris')),
    ])


def test_join_empty_left():
    identity = lambda x: x
    assert list(join(identity, [], identity, [1, 2, 3])) == []


def test_join_empty_right():
    identity = lambda x: x
    assert list(join(identity, [1, 2, 3], identity, [])) == []


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


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    # When a single list argument is given, it's interpreted as a list of seqs
    result = list(diff([[1, 2, 3], [1, 2, 4]]))
    assert result == [(3, 4)]


def test_diff_all_equal():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_with_noncallable_key():
    data = [{'score': 1}, {'score': 5}, {'score': 3}]
    assert topk(1, data, key='score') == ({'score': 5},)


def test_topk_fewer_than_k():
    assert topk(5, [1, 2]) == (2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


def test_peekn_zero():
    peeked, rest = peekn(0, [1, 2, 3])
    assert peeked == ()
    assert list(rest) == [1, 2, 3]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_seeded_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_with_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    r = list(random_sample(0.1, seq, random_state=randobj))
    assert r == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    seq = list(range(20))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_is_lazy():
    result = random_sample(0.5, range(1000), random_state=0)
    # Result should be an iterator, not a materialized container
    assert not isinstance(result, list)
    assert hasattr(result, '__next__')
