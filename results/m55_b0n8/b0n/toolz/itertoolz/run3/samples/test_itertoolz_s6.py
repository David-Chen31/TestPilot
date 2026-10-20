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

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_lazy():
    result = remove(lambda x: x > 2, itertools.count(0))
    assert list(take(3, result)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------

def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_is_lazy_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6
    with pytest.raises(StopIteration):
        next(gen)


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------

def test_groupby_callable_key():
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    result = groupby(lambda x: x % 2 == 0, data)
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_len_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_member_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert list(result.keys()) == ['F', 'M']
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert result['M'] == [
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]


def test_groupby_empty():
    assert groupby(len, []) == {}


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


def test_merge_sorted_zero_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_many_seqs():
    result = list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9]))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_with_key_many():
    seqs = [['apple', 'banana'], ['apricot', 'cherry']]
    result = list(merge_sorted(*seqs, key=len))
    assert result == ['apple', 'banana', 'apricot', 'cherry']


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


def test_interleave_with_empty_seqs():
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_infinite():
    result = interleave([itertools.count(0), itertools.count(100)])
    assert list(take(4, result)) == [0, 100, 1, 101]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_lazy():
    assert list(take(3, unique(itertools.count()))) == [0, 1, 2]


def test_unique_empty():
    assert list(unique([])) == []


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(iter([1, 2])) is True
    assert isiterable(5) is False
    assert isiterable(object()) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------

def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_string():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


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


def test_second_short_raises():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_index_error():
    with pytest.raises(IndexError):
        nth(5, [1, 2])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(3, 'abcdef') == 'def'


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)


def test_tail_n_larger_than_seq():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_basic():
    assert get(1, 'ABC') == 'B'


def test_get_multiple_list():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'unknown') == 'unknown'


def test_get_missing_default():
    phonebook = {'Alice': '555-1234'}
    assert get('Bob', phonebook, None) is None
    assert get(5, [1, 2, 3], 'X') == 'X'


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('Bob', {'Alice': 1})
    with pytest.raises(IndexError):
        get(5, [1, 2, 3])


def test_get_single_element_list():
    assert get(['a'], {'a': 1, 'b': 2}) == (1,)


def test_get_empty_list():
    assert get([], [1, 2, 3]) == ()


def test_get_single_element_list_default():
    assert get(['a', 'c'], {'a': 1, 'b': 2}, default=None) == (1, None)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1, 2], [3, 4]]))) == [1, 2, 3, 4]


def test_concat_empty():
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_mapcat_identity():
    assert list(mapcat(lambda x: x, [[1, 2], [3]])) == [1, 2, 3]


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_cons_infinite():
    it = cons(0, itertools.count(1))
    assert list(take(3, it)) == [0, 1, 2]


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
    data = ['cat', 'cat', 'ox', 'pig', 'pig', 'cat']
    result = frequencies(data)
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert isinstance(result, dict)


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------

def test_reduceby_sum():
    data = [1, 2, 3, 4, 5]
    result = reduceby(lambda x: x % 2 == 0, operator.add, data)
    assert result == {False: 9, True: 6}


def test_reduceby_product():
    data = [1, 2, 3, 4, 5]
    result = reduceby(lambda x: x % 2 == 0, operator.mul, data)
    assert result == {False: 15, True: 8}


def test_reduceby_with_init():
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

    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------

def test_iterate():
    inc = lambda x: x + 1
    counter = iterate(inc, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert list(take(5, powers)) == [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_exact():
    assert list(sliding_window(2, [1, 2])) == [(1, 2)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------

def test_partition_basic():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_partial_dropped():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)
    ]


def test_partition_pad_empty_seq():
    assert list(partition(2, [], pad=None)) == []


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_partial():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_short():
    assert list(partition_all(3, [1, 2])) == [(1, 2)]


def test_partition_all_iterator_no_len():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


def test_partition_all_nth():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == [
        (1, 2, 3), (4, 5, 6), (7,)
    ]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_string():
    assert count('abcd') == 4


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------

def test_pluck_name():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default='unknown')) == ['unknown', 'Pies']


def test_pluck_with_default_list():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck(['id', 'name'], data, default='unknown')) == [
        (1, 'unknown'), (2, 'Pies')
    ]


def test_getter_single():
    f = getter(0)
    assert f([10, 20]) == 10


def test_getter_list_multi():
    f = getter([1, 2])
    assert f([10, 20, 30]) == (20, 30)


def test_getter_list_single():
    f = getter([1])
    assert f([10, 20, 30]) == (20,)


def test_getter_list_empty():
    f = getter([])
    assert f([10, 20]) == ()


def test_getter_string_key():
    f = getter('a')
    assert f({'a': 1, 'b': 2}) == 1


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    assert set(result) == {
        ('Alice', 'Edith') and ('Alice', 'Edith'),
    } or ('Alice', 'Edith') not in result  # only sanity
    # Actual expected pairs:
    expected = []
    for (a, b) in friends:
        for (c, d) in cities:
            if b == c:
                expected.append(((a, b), (c, d)))
    assert set(result) == set(expected)


def test_join_inner_simple():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right))
    assert result == [(2, 2), (3, 3)]


def test_join_right_join():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right, left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_join():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (1, None)}


def test_join_full_join():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}


def test_join_with_index_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert set(result) == {
        (('Alice', 'Edith'), ('Edith', 'Paris')),
    }


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)
    ]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)) == [
        ('bananas', 'Oranges')
    ]


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3])) == [(2, 2, 5)]


def test_diff_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 10, 100]]))
    assert result == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_key_as_index():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_empty():
    assert topk(2, []) == ()


def test_topk_more_than_length():
    result = topk(5, [3, 1, 2])
    assert result == (3, 2, 1)


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, seq = peek(seq)
    assert first_item == 0
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    first_item, it2 = peek(it)
    assert first_item == 10
    assert list(it2) == [10, 20, 30]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn_iterator():
    it = iter([10, 20, 30])
    peeked, it2 = peekn(2, it)
    assert peeked == (10, 20)
    assert list(it2) == [10, 20, 30]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------

class _FixedRandom:
    """Deterministic mock random source cycling through given values."""
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def random(self):
        val = self.values[self.index % len(self.values)]
        self.index += 1
        return val


def test_random_sample_deterministic_with_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_none_selected():
    r = _FixedRandom([0.99])
    assert list(random_sample(0.1, [1, 2, 3], random_state=r)) == []


def test_random_sample_all_selected():
    r = _FixedRandom([0.0])
    assert list(random_sample(0.1, [1, 2, 3], random_state=r)) == [1, 2, 3]


def test_random_sample_prob_boundary():
    r = _FixedRandom([0.5])
    assert list(random_sample(0.5, [1, 2], random_state=r)) == []


def test_random_sample_custom_random_object():
    class R:
        def __init__(self):
            self._r = random.Random(2016)
        def random(self):
            return self._r.random()

    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=R()))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
