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


# ---------------------------------------------------------------- remove
def test_remove():
    def iseven(x):
        return x % 2 == 0

    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


# ---------------------------------------------------------------- accumulate
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_returns_iterator():
    result = accumulate(operator.add, [1, 2, 3])
    assert iter(result) is result


# ---------------------------------------------------------------- groupby
def test_groupby():
    assert groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']) == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_predicate():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }


def test_groupby_non_callable_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    assert groupby('gender', data) == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'},
        ],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------- merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([])) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1], [2, 3, 4])) == [1, 2, 3, 4]
    assert list(merge_sorted([1, 2, 3], [4])) == [1, 2, 3, 4]
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


def test_merge_sorted_key_uneven():
    key = lambda x: -x
    assert list(merge_sorted([3, 1], [4, 2], key=key)) == [4, 3, 2, 1]


# ---------------------------------------------------------------- interleave
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_single_seq():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ---------------------------------------------------------------- unique
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_lazy():
    it = unique([1, 1, 2, 2, 3])
    assert iter(it) is it
    assert next(it) == 1


# ---------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1, 2])) is True


# ---------------------------------------------------------------- isdistinct
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False
    assert isdistinct(iter([])) is True


# ---------------------------------------------------------------- take / drop
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
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6])) == [1, 4]


# ---------------------------------------------------------------- tail
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)
    assert tail(10, iter([1, 2])) == (1, 2)


# ---------------------------------------------------------------- first/second/nth/last
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


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [1, 2, 3]) == 1
    assert nth(2, (10, 20, 30)) == 30


def test_nth_iterator():
    assert nth(1, iter([1, 2, 3])) == 2
    assert nth(0, iter([5, 6])) == 5


def test_nth_negative_via_sequence():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# ---------------------------------------------------------------- get
def test_get_single_index():
    assert get(1, 'ABC') == 'B'
    assert get(0, [10, 20]) == 10


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list_of_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_list_of_indices_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_list_of_indices_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_default_missing():
    assert get('x', {}, None) is None
    assert get(5, [1, 2], 'default') == 'default'


def test_get_raises_without_default():
    with pytest.raises(KeyError):
        get('x', {})
    with pytest.raises(IndexError):
        get(5, [1, 2])


def test_get_type_error_default():
    assert get('a', 5, 'default') == 'default'


def test_get_multi_with_default_dict_missing():
    # non-list ind -> TypeError branch with default
    assert get([0, 1], [1], None) == (1, None)


# ---------------------------------------------------------------- concat / concatv
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv([1], [2], [3])) == [1, 2, 3]


# ---------------------------------------------------------------- mapcat
def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == ['A', 'B', 'C', 'D', 'E']
    assert list(mapcat(lambda x: [x, x], [1, 2])) == [1, 1, 2, 2]


# ---------------------------------------------------------------- cons
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------------------------------------------------------------- interpose
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2,
    }
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_no_initial():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_initial_constant():
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


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------- partition
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)
    ]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------------------------------------------------------- partition_all
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_iterator():
    # iterator -> no __len__ -> binary search branch
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == [(1, 2, 3), (4, 5)]
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_exact():
    assert list(partition_all(2, iter([1, 2, 3, 4]))) == [(1, 2), (3, 4)]


# ---------------------------------------------------------------- count
def test_count_sized():
    assert count([1, 2, 3]) == 3
    assert count((1, 2)) == 2
    assert count("abcd") == 4


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(iter([])) == 0


# ---------------------------------------------------------------- pluck
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_multiple_default():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, default=0)) == [(1, 2), (3, 0)]


# ---------------------------------------------------------------- getter
def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_single_list():
    g = getter([1])
    assert g('abc') == ('b',)


def test_getter_multi():
    g = getter([0, 2])
    assert g('abc') == ('a', 'c')


def test_getter_empty_list():
    g = getter([])
    assert g('abc') == ()


# ---------------------------------------------------------------- join
def test_join_inner():
    left = [('a', 1), ('b', 2), ('a', 3)]
    right = [('a', 'x'), ('b', 'y'), ('c', 'z')]
    result = list(join(first, left, first, right))
    assert sorted(result) == sorted([
        (('a', 1), ('a', 'x')),
        (('a', 3), ('a', 'x')),
        (('b', 2), ('b', 'y')),
    ])


def test_join_right():
    left = [('a', 1), ('b', 2)]
    right = [('a', 'x'), ('c', 'z')]
    result = list(join(first, left, first, right, left_default=None))
    assert result == [
        (('a', 1), ('a', 'x')),
        (None, ('c', 'z')),
    ]


def test_join_left():
    left = [('a', 1), ('b', 2)]
    right = [('a', 'x')]
    result = list(join(first, left, first, right, right_default=None))
    assert result == [
        (('a', 1), ('a', 'x')),
        (('b', 2), None),
    ]


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [
        (2, 2), (3, 3), (None, 4), (1, None),
    ]


def test_join_index_keys():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = join(1, friends, 0, cities)
    # leftseq grouped by second element: {'Edith': [('Alice','Edith')],
    #                                     'Alice': [('Zhao','Alice')]}
    # rightseq iterated in order: ('Alice','NYC') then ('Edith','Paris')
    assert list(result) == [
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Alice', 'Edith'), ('Edith', 'Paris')),
    ]


# ---------------------------------------------------------------- diff
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)
    ]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]


def test_diff_three():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice'
    )


def test_topk_non_callable_key():
    assert topk(2, [(1, 'a'), (5, 'b'), (3, 'c')], key=0) == ((5, 'b'), (3, 'c'))


def test_topk_empty():
    assert topk(2, []) == ()


def test_topk_more_than_available():
    assert topk(5, [1, 2]) == (2, 1)


# ---------------------------------------------------------------- peek / peekn
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, newseq = peek(seq)
    assert item == 0
    assert list(newseq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, newseq = peek(iter([0, 1, 2]))
    assert item == 0
    assert list(newseq) == [0, 1, 2]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    items, newseq = peekn(2, seq)
    assert items == (0, 1)
    assert list(newseq) == [0, 1, 2, 3, 4]


def test_peekn_iterator():
    items, newseq = peekn(2, iter([0, 1, 2, 3]))
    assert items == (0, 1)
    assert list(newseq) == [0, 1, 2, 3]


def test_peekn_more_than_available():
    items, newseq = peekn(5, iter([0, 1]))
    assert items == (0, 1)
    assert list(newseq) == [0, 1]


# ---------------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2


def test_random_sample_prob_zero():
    seq = list(range(100))
    assert list(random_sample(0.0, seq, random_state=1)) == []


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=1)) == seq


def test_random_sample_random_obj():
    class AlwaysTrue:
        def random(self):
            return 0.0

    class AlwaysFalse:
        def random(self):
            return 1.0

    seq = [1, 2, 3]
    assert list(random_sample(0.5, seq, random_state=AlwaysTrue())) == seq
    assert list(random_sample(0.5, seq, random_state=AlwaysFalse())) == []
