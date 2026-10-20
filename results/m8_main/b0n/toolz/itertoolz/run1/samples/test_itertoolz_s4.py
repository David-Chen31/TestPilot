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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: x > 0, [])) == []


def test_remove_all():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate_no_initial():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_lazy():
    result = accumulate(operator.add, [1, 2, 3])
    assert next(result) == 1
    assert next(result) == 3


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_string_key():
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


def test_groupby_len():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_single():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_none():
    assert list(merge_sorted()) == []


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 5], [2, 3, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([2, 3, 4, 6], [1, 5])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_empty_inner():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_three():
    assert list(interleave([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == \
        [1, 4, 7, 2, 5, 8, 3, 6, 9]


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
    assert tuple(unique(())) == ()


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

def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct("World") is True
    assert isdistinct([]) is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


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
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------

def test_first():
    assert first('ABC') == 'A'
    assert first([5]) == 5


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [1, 2, 3]) == 1
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(2, iter([1, 2, 3, 4])) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_basic():
    assert get(1, 'ABC') == 'B'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'x') == 'x'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})


def test_get_missing_index_raises():
    with pytest.raises(IndexError):
        get(5, 'ABC')


def test_get_empty_list_no_default():
    assert get([], 'ABC') == ()


def test_get_single_item_list():
    assert get([0], 'ABC') == ('A',)


def test_get_unhashable_ind_default():
    assert get({}, [1, 2], default=None) is None


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat([x] for x in range(3))) == [0, 1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


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


def test_reduceby_init_constant():
    data = [{'state': 'CA', 'cost': 1000000},
            {'state': 'IL', 'cost': 100000},
            {'state': 'IL', 'cost': 2000000},
            {'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], data, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, operator.add, [], 0) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    result = list(itertools.islice(iterate(lambda x: x * 2, 1), 5))
    assert result == [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator():
    assert list(partition_all(3, iter(range(7)))) == \
        [(0, 1, 2), (3, 4, 5), (6,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count('abc') == 3
    assert count([]) == 0


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


def test_pluck_multiple_default():
    data = [[1], [1, 2]]
    assert list(pluck([0, 1], data, None)) == [(1, None), (1, 2)]


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------

def test_getter_single():
    g = getter(0)
    assert g([1, 2, 3]) == 1


def test_getter_multiple():
    g = getter([0, 2])
    assert g([1, 2, 3]) == (1, 3)


def test_getter_single_list():
    g = getter([1])
    assert g([1, 2, 3]) == (2,)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4]))
    assert sorted(result) == [(2, 2), (3, 3)]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda t: (str(t[0]), str(t[1]))) == \
        sorted([(2, 2), (3, 3), (None, 4), (1, None)],
               key=lambda t: (str(t[0]), str(t[1])))


def test_join_right():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], left_default=None))
    assert sorted(result) == [(2, 2), (None, 3)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], right_default=None))
    assert sorted(result) == [(2, 2), (1, None)]


def test_join_key_returns_tuple_singleton():
    # groupby/getter semantics: getter returns tuples for list keys
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    assert ('Alice', 'Edith') in [(a, b) for (a, b), (c, d) in result] or True
    # more precise: check expected pairs
    expected = {((('Alice', 'Edith')), ('Alice', 'NYC')),
                (('Alice', 'Zhao'), ('Alice', 'NYC'))}
    assert set(result) == expected


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three_seqs():
    assert list(diff([1, 2], [2, 2], [3, 2])) == [(1, 2, 3)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_nested_list():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_string_key():
    data = [{'v': 1}, {'v': 10}, {'v': 5}]
    assert topk(2, data, key='v') == ({'v': 10}, {'v': 5})


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, seq2 = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(seq2) == [1, 2, 3]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, seq2 = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, seq2 = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(seq2) == [1, 2, 3]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

class _FixedRandom:
    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def test_random_sample_deterministic():
    seq = list(range(10))
    rand = _FixedRandom([0.05, 0.5, 0.1, 0.9, 0.0, 0.2, 0.7, 0.3, 0.15, 1.0])
    assert list(random_sample(0.2, seq, random_state=rand)) == [0, 2, 4, 8]


def test_random_sample_prob_zero():
    rand = _FixedRandom([0.5] * 5)
    assert list(random_sample(0.0, [1, 2, 3, 4, 5], random_state=rand)) == []


def test_random_sample_prob_one():
    rand = _FixedRandom([0.5] * 5)
    assert list(random_sample(1.0, [1, 2, 3, 4, 5], random_state=rand)) == \
        [1, 2, 3, 4, 5]


def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
