import itertools
import operator
import pytest
from collections import OrderedDict

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------- remove ----------

def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------- accumulate ----------

def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_lazy():
    def gen():
        yield 1
        yield 2
        yield 3

    it = accumulate(operator.add, gen())
    assert next(it) == 1
    assert next(it) == 3
    assert next(it) == 6


# ---------- groupby ----------

def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, range(1, 9))
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_string_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result['F'] == [data[0]]
    assert result['M'] == [data[1], data[2]]


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------- merge_sorted ----------

def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2], [], [3, 4], [5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([1, 2, 3, 4, 5], [6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([6], [1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5, 6]


# ---------- interleave ----------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], [1, 2]])) == [1, 2]


def test_interleave_infinite():
    result = list(take(5, interleave([itertools.count(), itertools.count(10)])))
    assert result == [0, 10, 1, 11, 2]


# ---------- unique ----------

def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_lazy():
    it = unique(itertools.count())
    assert list(take(5, it)) == [0, 1, 2, 3, 4]


# ---------- isiterable ----------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------- isdistinct ----------

def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------- take ----------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


# ---------- tail ----------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


# ---------- drop ----------

def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2])) == []


# ---------- take_nth ----------

def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


# ---------- first / second / nth / last ----------

def test_first():
    assert first('ABC') == 'A'
    assert first([5]) == 5


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_second_too_short():
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(2, iter([1, 2, 3, 4])) == 3


def test_nth_list():
    assert nth(1, [1, 2, 3]) == 2


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------- get ----------

def test_get_single():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555-1234'}) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get(['Alice', 'Bob'], {'Alice': 1, 'Bob': 2}) == (1, 2)


def test_get_list_single():
    assert get([1], 'ABC') == ('B',)


def test_get_list_empty():
    assert get([], 'ABC') == ()


def test_get_default_missing():
    phonebook = {'Alice': '555-1234'}
    assert get('Dennis', phonebook, None) is None
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_index_error():
    with pytest.raises(IndexError):
        get(5, [1, 2, 3])


def test_get_index_error_default():
    assert get(5, [1, 2, 3], default='x') == 'x'


def test_get_key_error():
    with pytest.raises(KeyError):
        get('missing', {})


def test_get_key_error_default():
    assert get('missing', {}, default=42) == 42


# ---------- concat / concatv ----------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_empty():
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_no_args():
    assert list(concatv()) == []


# ---------- mapcat ----------

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------- cons ----------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_infinite():
    assert list(take(4, cons(0, itertools.count(1)))) == [0, 1, 2, 3]


# ---------- interpose ----------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


def test_interpose_iterator():
    assert list(interpose(0, iter([1, 2, 3]))) == [1, 0, 2, 0, 3]


# ---------- frequencies ----------

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------- reduceby ----------

def test_reduceby_no_init():
    data = [1, 2, 3, 4, 5]
    assert reduceby(lambda x: x % 2 == 0, operator.add, data) == {False: 9, True: 6}
    assert reduceby(lambda x: x % 2 == 0, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_constant_init():
    data = [1, 2, 3, 4, 5]
    assert reduceby(lambda x: x % 2 == 0, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_string_key():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------- iterate ----------

def test_iterate():
    it = iterate(lambda x: x + 1, 0)
    assert [next(it) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_n1():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------- partition ----------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------- partition_all ----------

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


# ---------- count ----------

def test_count_with_len():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3])) == 3


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------- pluck ----------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_default():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, 2), (3, None)]


# ---------- getter ----------

def test_getter_single():
    g = getter('name')
    assert g({'name': 'Alice'}) == 'Alice'


def test_getter_single_list():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_multi_list():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ---------- join ----------

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    pairs = {(a, d) for ((a, b), (c, d)) in result}
    assert pairs == {
        ('Alice', 'Berlin'), ('Alice', 'Paris'), ('Alice', 'Shanghai'),
        ('Edith', 'Chicago'), ('Edith', 'NYC'),
        ('Zhao', 'Chicago'), ('Zhao', 'NYC'),
        ('Zhao', 'Berlin'), ('Zhao', 'Paris'),
    }


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


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


def test_join_non_callable_keys():
    left = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
    right = [{'id': 2, 'w': 'c'}, {'id': 3, 'w': 'd'}]
    result = list(join('id', left, 'id', right))
    assert result == [({'id': 2, 'v': 'b'}, {'id': 2, 'w': 'c'})]


# ---------- diff ----------

def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three_sequences():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


def test_diff_from_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------- topk ----------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'x': 1}, {'x': 3}, {'x': 2}]
    result = topk(2, data, key='x')
    assert result == ({'x': 3}, {'x': 2})


def test_topk_empty():
    assert topk(2, []) == ()


# ---------- peek ----------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(rest) == [1, 2, 3]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------- peekn ----------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    head, rest = peekn(2, seq)
    assert head == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_zero():
    head, rest = peekn(0, [1, 2, 3])
    assert head == ()
    assert list(rest) == [1, 2, 3]


def test_peekn_more_than_length():
    head, rest = peekn(5, [1, 2])
    assert head == (1, 2)
    assert list(rest) == [1, 2]


# ---------- random_sample ----------

def test_random_sample_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_deterministic():
    seq = list(range(50))
    r1 = list(random_sample(0.5, seq, random_state=42))
    r2 = list(random_sample(0.5, seq, random_state=42))
    assert r1 == r2


def test_random_sample_prob_zero():
    assert list(random_sample(0, list(range(100)), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1, list(range(10)), random_state=1)) == \
        list(range(10))


def test_random_sample_custom_state():
    from random import Random
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=Random(2016)))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
