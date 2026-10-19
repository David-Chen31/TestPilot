import itertools
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

def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


def test_remove_returns_iterator():
    result = remove(lambda x: False, [1, 2, 3])
    assert iter(result) is result


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_lazy():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6
    with pytest.raises(StopIteration):
        next(gen)


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_member_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert result['M'] == [
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_len():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([])) == []
    assert list(merge_sorted([], [])) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3, 4], [5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([5], [1, 2, 3, 4])) == [1, 2, 3, 4, 5]


def test_merge_sorted_multiple():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_key_multiple():
    # The implementation merges using the key only; the given inputs are not
    # individually sorted by the key ``-x`` (they are ascending), so the
    # resulting merged order is the one produced by the binary merge.
    result = list(merge_sorted([1, 4], [2, 5], [3, 6], key=lambda x: -x))
    assert result == [3, 6, 2, 5, 1, 4]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_uneven():
    assert list(interleave([[1, 2, 3], [4]])) == [1, 4, 2, 3]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique():
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


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'ABCDE') == 'DE'


def test_tail_iterator():
    result = tail(2, iter([1, 2, 3, 4, 5]))
    assert result == (4, 5)


def test_tail_n_greater_than_len():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


def test_tail_empty():
    assert tail(2, []) == []


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30])) == 30
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2, 3]))


def test_nth_sequence():
    assert nth(1, (1, 2, 3)) == 2


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


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


def test_get_default_single():
    assert get('missing', {'a': 1}, default='x') == 'x'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_empty_list_ind():
    assert get([], 'ABC') == ()
    assert get([], {}, default=None) == ()


def test_get_single_list():
    assert get([1], 'ABC') == ('B',)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    gen = (x for x in [[1, 2], [3, 4]])
    assert list(concat(gen)) == [1, 2, 3, 4]


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


# ---------------------------------------------------------------------------
# interpose
# ---------------------------------------------------------------------------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_value():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
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
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate():
    def inc(x):
        return x + 1

    counter = iterate(inc, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]


def test_iterate_double():
    double = lambda x: x * 2
    powers_of_two = iterate(double, 1)
    assert [next(powers_of_two) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_longer_than_seq():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_exact():
    assert list(sliding_window(3, [1, 2, 3])) == [(1, 2, 3)]


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]
    assert list(partition(2, [1, 2, 3], pad=0)) == [(1, 2), (3, 0)]


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_list_seq():
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_iterator():
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == [(1, 2, 3), (4, 5)]


def test_partition_all_exact_multiple():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(i for i in range(5)) == 5


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------

def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default='unknown')) == ['unknown', 'Pies']


def test_pluck_multiple_default():
    data = [[1, 2], [3]]
    result = list(pluck([0, 1], data, default=None))
    assert result == [(1, 2), (3, None)]


def test_pluck_single_list():
    assert list(pluck(['name'], [{'name': 'a'}, {'name': 'b'}])) == [('a',), ('b',)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    g = getter('a')
    assert g({'a': 1}) == 1


def test_getter_multiple():
    g = getter([0, 1])
    assert g([10, 20, 30]) == (10, 20)


def test_getter_single_list():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    # The join iterates the right sequence (cities) and for each city pairs
    # every friend whose second element matches with that city.
    assert result == [
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
    ]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_index_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    result = list(diff([[1, 2], [1, 3]]))
    assert result == [(2, 3)]


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 2, 4], [1, 5, 3]))
    assert result == [(2, 2, 5), (3, 4, 3)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    seq = iter([10, 20, 30])
    item, seq2 = peek(seq)
    assert item == 10
    assert list(seq2) == [10, 20, 30]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    items, seq2 = peekn(2, seq)
    assert items == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    items, seq2 = peekn(10, [1, 2])
    assert items == (1, 2)
    assert list(seq2) == [1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3, 4], random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3, 4], random_state=0)) == [1, 2, 3, 4]


def test_random_sample_lazy():
    result = random_sample(0.5, [1, 2, 3], random_state=0)
    assert iter(result) is result
