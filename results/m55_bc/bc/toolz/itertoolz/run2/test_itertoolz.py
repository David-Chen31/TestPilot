import itertools
import operator
from collections import OrderedDict
from functools import reduce

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
    no_default,
)


# remove -------------------------------------------------------------------

def test_remove():
    even = lambda x: x % 2 == 0
    assert list(remove(even, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(even, [])) == []


def test_remove_lazy():
    seen = []

    def pred(x):
        seen.append(x)
        return x < 0

    it = remove(pred, [1, 2, 3])
    assert seen == []
    assert next(it) == 1
    assert seen == [1]


# accumulate --------------------------------------------------------------

def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_without_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_lazy():
    it = accumulate(operator.add, [1, 2, 3, 4])
    assert next(it) == 1
    assert next(it) == 3


# groupby -----------------------------------------------------------------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result[False] == [1, 3, 5, 7]
    assert result[True] == [2, 4, 6, 8]


def test_groupby_member():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert [p['name'] for p in result['F']] == ['Alice']
    assert [p['name'] for p in result['M']] == ['Bob', 'Charlie']


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# merge_sorted ------------------------------------------------------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_and_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]
    assert list(merge_sorted([1], [])) == [1]


def test_merge_sorted_many():
    a = list(range(0, 10, 2))
    b = list(range(1, 10, 2))
    c = list(range(0, 10, 3))
    expected = sorted(a + b + c)
    assert list(merge_sorted(a, b, c)) == expected


def test_merge_sorted_recursive_binary():
    # Four sequences exercise the recursive _merge_sorted_binary path
    # (the branch where a partition has more than one sequence).
    assert list(merge_sorted([10], [1], [2], [3])) == [1, 2, 3, 10]
    # Right half is empty -> hits the `except StopIteration` on next(seq2).
    assert list(merge_sorted([1], [2], [], [])) == [1, 2]
    # Left half empty, right half non-empty.
    assert list(merge_sorted([], [], [5], [])) == [5]
    # Six sequences (deeper recursion).
    assert list(merge_sorted([1], [2], [3], [4], [5], [6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_key_recursive():
    # Four sequences -> recursion in _merge_sorted_binary_key.
    assert list(merge_sorted([10], [1], [2], [3], key=lambda x: x)) == \
        [1, 2, 3, 10]
    # Right half empty -> hits `except StopIteration` in the keyed variant.
    assert list(merge_sorted([1], [2], [], [], key=lambda x: x)) == [1, 2]
    # A key that changes ordering but not the merge algorithm.
    assert list(merge_sorted([10], [1], [2], [3], key=lambda x: -x)) == \
        [10, 3, 2, 1]


def test_merge_sorted_inner_else_break():
    # The entire remaining seq2 is smaller than the current val1, which
    # exhausts the inner loop and triggers the `for ... else: break` path.
    assert list(merge_sorted([10], [1, 2, 3])) == [1, 2, 3, 10]
    # Same behaviour, keyed variant.
    assert list(merge_sorted([10], [1, 2, 3], key=lambda x: x)) == [1, 2, 3, 10]
    # Partly smaller, then larger -> exercises the yielded val2 then break.
    assert list(merge_sorted([5], [1, 2, 7])) == [1, 2, 5, 7]


# interleave --------------------------------------------------------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_inner():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_uneven():
    assert list(interleave([[1], [], [2, 3]])) in (
        [1, 2, 3], [1, 3, 2])


# unique ------------------------------------------------------------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_lazy():
    it = unique([1, 1, 2, 2, 3])
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 3


# isiterable --------------------------------------------------------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# isdistinct --------------------------------------------------------------

def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# take / drop / tail ------------------------------------------------------

def test_take():
    assert list(take(2, [10, 20, 30])) == [10, 20]
    assert list(take(0, [10])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, "abcde") == "de"
    assert tail(2, iter([1, 2, 3])) == (2, 3)
    assert tail(2, [1]) == [1]


def test_tail_generator():
    assert tail(3, (x for x in range(5))) == (2, 3, 4)


# take_nth ----------------------------------------------------------------

def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# first / second / nth / last --------------------------------------------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2]) == 1
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second('A')


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(2, [1, 2, 3]) == 3
    assert nth(1, (1, 2, 3)) == 2
    assert nth(0, iter([10, 20, 30])) == 10
    with pytest.raises(IndexError):
        nth(10, [1, 2])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3
    with pytest.raises(IndexError):
        last([])


# get ---------------------------------------------------------------------

def test_get_basic():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': 5}) == 5


def test_get_list_ind():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([0], 'ABC') == ('A',)
    assert get([], 'ABC') == ()


def test_get_default():
    assert get([1, 2], 'AB', None) == ('B', None)
    assert get('missing', {'a': 1}, 0) == 0
    assert get(10, [1, 2], 'default') == 'default'


def test_get_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2])
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_typeerror_default():
    assert get(1.5, [1, 2], None) is None


def test_get_typeerror_no_default_raises():
    # A non-list, non-integer index with no default re-raises the TypeError.
    with pytest.raises(TypeError):
        get(1.5, [1, 2])


# concat / concatv --------------------------------------------------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []
    assert list(concat([(x for x in [1, 2]), [3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# mapcat ------------------------------------------------------------------

def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == \
        ['A', 'B', 'C', 'D', 'E']


def test_mapcat_empty():
    assert list(mapcat(lambda s: s, [])) == []


# cons --------------------------------------------------------------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_cons_infinite():
    assert list(take(3, cons(0, iterate(lambda x: x + 1, 1)))) == [0, 1, 2]


# interpose ---------------------------------------------------------------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# frequencies -------------------------------------------------------------

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# reduceby ----------------------------------------------------------------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_constant():
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
    assert result[True] == {2, 4}
    assert result[False] == {1, 3}


def test_reduceby_member_key():
    projects = [{'state': 'CA', 'cost': 100},
                {'state': 'IL', 'cost': 50},
                {'state': 'CA', 'cost': 200}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 300, 'IL': 50}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}
    assert reduceby(lambda x: x, operator.add, [], 0) == {}


# iterate -----------------------------------------------------------------

def test_iterate():
    assert list(take(4, iterate(lambda x: x + 1, 0))) == [0, 1, 2, 3]
    assert list(take(5, iterate(lambda x: x * 2, 1))) == [1, 2, 4, 8, 16]


# sliding_window ----------------------------------------------------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == \
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_short():
    assert list(sliding_window(2, [1])) == []
    assert list(sliding_window(2, [])) == []


# partition ---------------------------------------------------------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# partition_all -----------------------------------------------------------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    # no __len__ -> binary search path
    assert list(partition_all(3, iter(range(10)))) == \
        [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]


def test_partition_all_list_length_path():
    assert list(partition_all(3, list(range(10)))) == \
        [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]


def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_lookup_error_on_bad_len():
    # A sequence that lies about its __len__ causes the fast __len__ path to
    # detect the inconsistency and raise LookupError.
    class Liar(list):
        def __len__(self):
            return 4

    with pytest.raises(LookupError):
        list(partition_all(3, Liar([0, 1, 2, 3, 4])))


# count -------------------------------------------------------------------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3])) == 3
    assert count('abcd') == 4
    assert count([]) == 0


# pluck -------------------------------------------------------------------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']
    assert list(pluck(['id', 'name'], data, default=None)) == \
        [(1, None), (2, 'Pies')]


def test_pluck_missing_raises():
    with pytest.raises(KeyError):
        list(pluck('missing', [{'a': 1}]))


# getter ------------------------------------------------------------------

def test_getter_scalar():
    g = getter(1)
    assert g([10, 20, 30]) == 20
    assert g('ABC') == 'B'


def test_getter_single_list():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_multi_list():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# join --------------------------------------------------------------------

def test_join_inner():
    a = [1, 2, 3]
    b = [2, 3, 4]
    result = list(join(lambda x: x, a, lambda x: x, b))
    assert sorted(result) == [(2, 2), (3, 3)]


def test_join_right():
    a = [1, 2, 3]
    b = [2, 3, 4]
    result = list(join(lambda x: x, a, lambda x: x, b, left_default=None))
    assert (None, 4) in result
    assert sorted(x for x in result if x[0] is not None) == [(2, 2), (3, 3)]


def test_join_left():
    a = [1, 2, 3]
    b = [2, 3, 4]
    result = list(join(lambda x: x, a, lambda x: x, b, right_default=None))
    assert (1, None) in result
    assert (None, 4) not in result


def test_join_full():
    a = [1, 2, 3]
    b = [2, 3, 4]
    result = list(join(lambda x: x, a, lambda x: x, b,
                       left_default=None, right_default=None))
    # Order: matched pairs in rightseq order, then unmatched right items
    # (with left_default), then unmatched left keys (with right_default).
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_member_keys():
    friends = [('Alice', 'Edith'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert sorted(result) == [
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
    ]


def test_join_empty():
    assert list(join(lambda x: x, [], lambda x: x, [1, 2])) == []


# diff --------------------------------------------------------------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three():
    result = list(diff([1, 2], [1, 3], [1, 4]))
    assert result == [(2, 3, 4)]


def test_diff_accepts_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_identical():
    assert list(diff([1, 2], [1, 2])) == []


# topk --------------------------------------------------------------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key_callable():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_key_getter():
    data = [{'v': 1}, {'v': 5}, {'v': 3}]
    result = topk(2, data, key='v')
    assert [x['v'] for x in result] == [5, 3]


def test_topk_more_than_available():
    assert topk(10, [1, 2]) == (2, 1)


# peek / peekn ------------------------------------------------------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_generator():
    first_item, rest = peek(iter([5, 6, 7]))
    assert first_item == 5
    assert list(rest) == [5, 6, 7]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(rest) == [1, 2, 3]


# random_sample -----------------------------------------------------------

def test_random_sample_zero_prob():
    assert list(random_sample(0.0, range(100))) == []


def test_random_sample_one_prob():
    assert list(random_sample(1.0, range(10))) == list(range(10))


def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == \
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_obj():
    from random import Random
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=Random(2016))) == \
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_lazy():
    it = random_sample(1.0, iter(range(5)))
    assert next(it) == 0
    assert next(it) == 1
