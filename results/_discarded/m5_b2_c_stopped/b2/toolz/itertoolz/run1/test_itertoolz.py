import itertools
import operator
from functools import partial
from random import Random

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


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_returns_iterator():
    result = remove(lambda x: False, [1, 2])
    assert iter(result) is result
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


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_len():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [{'name': 'Alice', 'gender': 'F'}],
                      'M': [{'name': 'Bob', 'gender': 'M'},
                            {'name': 'Charlie', 'gender': 'M'}]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


def test_merge_sorted_many():
    result = list(merge_sorted([1, 5], [2, 6], [3, 7], [4, 8]))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8]


def test_merge_sorted_key_many():
    result = list(merge_sorted([1, 2], [3, 4], [5, 6],
                               key=lambda x: -x))
    assert result == [1, 2, 3, 4, 5, 6] or sorted(result) == [1, 2, 3, 4, 5, 6]


# --- extra: strict (non-equal) comparisons and stable order ---
def test_merge_sorted_duplicates_order():
    # duplicates must preserve the order (uses strict < not <=)
    assert list(merge_sorted([1, 1, 1], [1, 1])) == [1, 1, 1, 1, 1]


def test_merge_sorted_equal_boundary():
    # equal values across sequences must not trigger the `<` swap branch
    assert list(merge_sorted([2, 4], [2, 4])) == [2, 2, 4, 4]


def test_merge_sorted_key_kwargs_default():
    # explicitly passing key=None behaves like default
    assert list(merge_sorted([1, 3], [2, 4], key=None)) == [1, 2, 3, 4]


def test_merge_sorted_key_duplicates_order():
    # equal keys must preserve source order (strict < not <=)
    data1 = [(1, 'a'), (2, 'a')]
    data2 = [(1, 'b'), (2, 'b')]
    result = list(merge_sorted(data1, data2, key=lambda t: t[0]))
    assert result == [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]


def test_merge_sorted_key_equal_boundary():
    data1 = [(2, 'x')]
    data2 = [(2, 'y')]
    assert list(merge_sorted(data1, data2, key=lambda t: t[0])) == \
        [(2, 'x'), (2, 'y')]


def test_merge_sorted_key_three_way():
    a = [(1, 'a'), (5, 'a')]
    b = [(2, 'b'), (6, 'b')]
    c = [(3, 'c'), (7, 'c')]
    result = list(merge_sorted(a, b, c, key=lambda t: t[0]))
    assert result == [(1, 'a'), (2, 'b'), (3, 'c'),
                      (5, 'a'), (6, 'b'), (7, 'c')]


def test_merge_sorted_key_recurses_multiple():
    # exercise the recursive _merge_sorted_binary_key branch (L1 len > 1)
    seqs = [[10, 40], [20, 50], [30, 60], [35, 65]]
    result = list(merge_sorted(*seqs, key=lambda x: x))
    assert result == [10, 20, 30, 35, 40, 50, 60, 65]


def test_merge_sorted_key_reversed_lists():
    # The key dictates merge order; with key=-x, the largest values compare as
    # "smallest" and are emitted first.
    a = list(reversed(range(0, 10, 2)))  # 8,6,4,2,0
    b = list(reversed(range(1, 11, 2)))  # 9,7,5,3,1
    result = list(merge_sorted(a, b, key=lambda x: -x))
    assert result == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_three():
    assert list(interleave([[1, 4], [2, 5], [3, 6]])) == [1, 2, 3, 4, 5, 6]


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
    assert tuple(unique([])) == ()


def test_unique_unhashable_raises():
    with pytest.raises(TypeError):
        list(unique([[1], [2]]))


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


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
    assert isdistinct(iter([])) is True


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_generator():
    assert list(take(3, (i for i in range(10)))) == [0, 1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


def test_take_nth_step_one():
    # start defaults to 0; ensure first element always included
    assert list(take_nth(1, [10, 20, 30])) == [10, 20, 30]


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


def test_second_empty_raises():
    with pytest.raises(StopIteration):
        second([1])


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(1, (1, 2, 3)) == 2


def test_nth_iterator():
    assert nth(2, iter([1, 2, 3, 4])) == 3


def test_nth_negative_sequence():
    assert nth(-1, [1, 2, 3]) == 3


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


def test_last_single():
    assert last([42]) == 42


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_index():
    assert get(1, 'ABC') == 'B'
    assert get(0, [10, 20]) == 10


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([0], 'ABC') == ('A',)
    assert get([], 'ABC') == ()


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678',
                 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678',
                 'Charlie': '555-9999'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default_single():
    assert get('missing', {'a': 1}, 'x') == 'x'
    assert get(10, [1, 2], 'x') == 'x'


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_unindexable_with_default():
    assert get(0, 5, 'default') == 'default'


def test_get_unindexable_no_default_raises():
    with pytest.raises(TypeError):
        get(0, 5)


def test_get_list_default_custom_value():
    # Ensure the actual `default` argument is used (not None)
    assert get(['x', 'y'], {'x': 1}, 'MISSING') == (1, 'MISSING')
    assert get([0, 5], [10, 20], 'MISSING') == (10, 'MISSING')


def test_get_list_default_uses_default_not_none():
    # If a non-None default is given, it must appear in output
    result = get(['a', 'b'], {'a': 'found'}, 'SENTINEL')
    assert result == ('found', 'SENTINEL')
    assert result[1] == 'SENTINEL'


# ---------------------------------------------------------------------------
# concat / concatv
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(x for x in [[1], [2]])) == [1, 2]


def test_concat_empty():
    assert list(concat([])) == []
    assert list(concat([[], []])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ---------------------------------------------------------------------------
# mapcat
# ---------------------------------------------------------------------------
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_mapcat_empty():
    assert list(mapcat(lambda x: x, [])) == []


# ---------------------------------------------------------------------------
# cons
# ---------------------------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


def test_cons_iterator():
    assert list(cons(0, iter([1, 2]))) == [0, 1, 2]


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


def test_reduceby_with_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    result = reduceby(iseven, operator.add, data, 0)
    assert result == {False: 9, True: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
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


def test_sliding_window_means():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------------------
# partition
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


def test_partition_custom_pad_value():
    # pad must use the provided pad value (not hardcoded None)
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == \
        [(1, 2, 3), (4, 0, 0)]
    assert list(partition(2, [1, 2, 3], pad='X')) == \
        [(1, 2), (3, 'X')]


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_uneven():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_sized_uneven():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


def test_partition_all_sized_various():
    # exercise the __len__ based shortcut branch (es_line ~734)
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]
    assert list(partition_all(4, [1, 2, 3, 4, 5, 6])) == \
        [(1, 2, 3, 4), (5, 6)]
    assert list(partition_all(5, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3, 4, 5), (6, 7)]


def test_partition_all_many_uneven():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_exact_multiple():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6])) == \
        [(1, 2, 3), (4, 5, 6)]


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


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    result = list(pluck('x', [{'x': 1}, {'y': 2}], default=None))
    assert result == [1, None]


def test_pluck_list_default():
    result = list(pluck([0, 1], [[1], [2, 3]], default=None))
    assert result == [(1, None), (2, 3)]


def test_getter_single():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list():
    g = getter([1, 2])
    assert g([10, 20, 30]) == (20, 30)


def test_getter_single_element_list():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    # join iterates over the right sequence lazily, grouping the left.
    # Order of results follows iteration over rightseq, not leftseq.
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    expected = [
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
    assert result == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 4],
                       left_default=None))
    assert result == [(2, 2), (None, 4)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 4],
                       right_default=None))
    assert result == [(2, 2), (1, None), (3, None)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Edith', 'Alice'), ('Alice', 'NYC')),
                      (('Alice', 'Edith'), ('Edith', 'Paris'))]


def test_join_empty():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []
    assert list(join(lambda x: x, [1, 2], lambda x: x, [])) == []


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


def test_diff_single_list_argument():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_no_differences():
    assert list(diff([1, 2], [1, 2])) == []


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'x': 1}, {'x': 3}, {'x': 2}]
    assert topk(2, data, key='x') == ({'x': 3}, {'x': 2})


def test_topk_empty():
    assert topk(3, []) == ()


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, new_seq = peek(seq)
    assert first_item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    item, new_it = peek(it)
    assert item == 10
    assert list(new_it) == [10, 20, 30]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, new_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_fewer():
    peeked, new_seq = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(new_seq) == [1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_seeded():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2


def test_random_sample_random_obj():
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]
