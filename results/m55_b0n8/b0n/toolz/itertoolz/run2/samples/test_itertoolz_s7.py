import itertools
import operator
import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second, nth,
    last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------- remove ----------

def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []
    assert list(remove(iseven, [2, 4])) == []


# ---------- accumulate ----------

def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_lazy_generator():
    gen = (x for x in [1, 2, 3])
    assert list(accumulate(operator.add, gen)) == [1, 3, 6]


# ---------- groupby ----------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4]) == {False: [1, 3], True: [2, 4]}


def test_groupby_key_index():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {5: ['Alice', 'Edith', 'Frank'],
                                   3: ['Bob', 'Dan'], 7: ['Charlie']}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [data[0]], 'M': [data[1], data[2]]}


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# ---------- merge_sorted ----------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2], [3, 4, 5, 6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([3, 4, 5, 6], [1, 2])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_many():
    seqs = [[1, 10], [2, 20], [3, 30], [4, 40]]
    assert list(merge_sorted(*seqs)) == [1, 2, 3, 4, 10, 20, 30, 40]


# ---------- interleave ----------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_with_empty():
    assert list(interleave([[1, 2], [], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_infinite_inner_terminates():
    # finite number of infinite-length sequences; islice to check
    seqs = [[1, 2, 3], [4, 5, 6]]
    assert list(interleave(seqs)) == [1, 4, 2, 5, 3, 6]


# ---------- unique ----------

def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


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


# ---------- take / drop / tail / take_nth ----------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(5, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(5, [1, 2])) == []


def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (10, 20, 30, 40, 50)) == (40, 50)
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)
    assert tail(10, [1, 2]) == [1, 2]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4, 5, 6, 7])) == [1, 4, 7]


# ---------- first / second / nth / last ----------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30, 40])) == 30
    assert nth(1, (1, 2, 3)) == 2


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(10, [1, 2])
    with pytest.raises(StopIteration):
        nth(10, iter([1, 2]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------- get ----------

def test_get_basic():
    assert get(1, 'ABC') == 'B'
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_default():
    assert get(10, [1, 2], default=None) is None
    assert get('x', {}, default=42) == 42


def test_get_list_single():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_typeerror_non_list():
    # non-indexable, non-list index, default given
    assert get(1, 5, default='d') == 'd'


# ---------- concat / concatv ----------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    def gen():
        yield [1, 2]
        yield [3]
    assert list(concat(gen())) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ---------- mapcat ----------

def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == ['A', 'B', 'C', 'D', 'E']


# ---------- cons ----------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]
    # works with infinite sequences lazily
    assert list(take(4, cons(0, iterate(lambda x: x + 1, 1)))) == [0, 1, 2, 3]


# ---------- interpose ----------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------- frequencies ----------

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------- reduceby ----------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_constant():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5], 0) == \
        {False: 9, True: 6}


def test_reduceby_init_callable():
    iseven = lambda x: x % 2 == 0

    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    data = [{'k': 'a', 'v': 1}, {'k': 'a', 'v': 2}, {'k': 'b', 'v': 5}]
    result = reduceby('k', lambda acc, x: acc + x['v'], data, 0)
    assert result == {'a': 3, 'b': 5}


# ---------- iterate ----------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]

    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ---------- partition ----------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------- partition_all ----------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_sized_seq():
    # list has __len__ -> fast path
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_unsized_seq():
    # generator has no __len__ -> binary-search path
    assert list(partition_all(3, iter([1, 2, 3, 4, 5, 6, 7]))) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


# ---------- count ----------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0
    assert count('abc') == 3


# ---------- pluck ----------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'x': 1}, {'y': 2}]
    assert list(pluck('x', data, default=None)) == [1, None]


def test_pluck_list_default():
    data = [[1], [2, 3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, None), (2, 3)]


# ---------- getter ----------

def test_getter_single():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list_single():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ---------- join ----------

def test_join_inner():
    a = [(1, 'a'), (2, 'b'), (3, 'c')]
    b = [('b', 20), ('c', 30), ('d', 40)]
    result = sorted(join(second, a, first, b))
    assert result == [((2, 'b'), ('b', 20)), ((3, 'c'), ('c', 30))]


def test_join_right_outer():
    a = [(1, 'a')]
    b = [('a', 10), ('b', 20)]
    result = list(join(second, a, first, b, left_default=None))
    assert result == [((1, 'a'), ('a', 10)), (None, ('b', 20))]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda x: (x[0] is None, x[1] is None, x)) == \
        sorted([(2, 2), (3, 3), (None, 4), (1, None)],
               key=lambda x: (x[0] is None, x[1] is None, x))


def test_join_left_outer():
    a = [(1, 'a'), (2, 'b')]
    b = [('a', 10)]
    result = list(join(second, a, first, b, right_default=None))
    assert result == [((1, 'a'), ('a', 10)), ((2, 'b'), None)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ---------- diff ----------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2]))


def test_diff_from_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 5, 3]])) == [(2, 5)]


def test_diff_three_seqs():
    assert list(diff([1, 2], [1, 3], [1, 4])) == [(2, 3, 4)]


def test_diff_no_differences():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ---------- topk ----------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 5}, {'v': 3}]
    assert topk(2, data, key='v') == ({'v': 5}, {'v': 3})


def test_topk_empty():
    assert topk(2, []) == ()


# ---------- peek ----------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator_consumption():
    it = iter([10, 20, 30])
    item, rest = peek(it)
    assert item == 10
    assert list(rest) == [10, 20, 30]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------- peekn ----------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    first_items, rest = peekn(10, [1, 2, 3])
    assert first_items == (1, 2, 3)
    assert list(rest) == [1, 2, 3]


def test_peekn_zero():
    items, rest = peekn(0, [1, 2, 3])
    assert items == ()
    assert list(rest) == [1, 2, 3]


# ---------- random_sample ----------

def test_random_sample_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    # same seed -> same result
    again = list(random_sample(0.1, seq, random_state=2016))
    assert again == result


def test_random_sample_prob_zero():
    seq = list(range(100))
    assert list(random_sample(0.0, seq, random_state=1)) == []


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=1)) == seq


def test_random_sample_random_object():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
