import pytest
from itertools import count as itertools_count

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---------------------------------------------------------------- remove
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda x: False, [1, 2])) == [1, 2]
    assert list(remove(lambda x: True, [1, 2])) == []


# ------------------------------------------------------------- accumulate
def test_accumulate_default():
    from operator import add, mul
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    from operator import add
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]
    assert list(accumulate(add, [], 0)) == [0]


def test_accumulate_empty():
    from operator import add
    assert list(accumulate(add, [])) == []


def test_accumulate_is_lazy_and_single_element():
    from operator import add
    assert list(accumulate(add, [5])) == [5]


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    def iseven(x):
        return x % 2 == 0
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }


def test_groupby_non_callable_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}


def test_groupby_member_index():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'}, {'name': 'Charlie', 'gender': 'M'}],
    }


def test_groupby_empty():
    assert groupby(lambda x: x, []) == {}


# ----------------------------------------------------------- merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_and_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_uneven_lengths():
    assert list(merge_sorted([1], [2, 3, 4])) == [1, 2, 3, 4]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


def test_merge_sorted_many():
    seqs = [[i] for i in range(10)]
    assert list(merge_sorted(*seqs)) == list(range(10))


# ------------------------------------------------------------- interleave
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


# ----------------------------------------------------------------- unique
def test_unique_default():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_unhashable_key():
    data = [{'a': 1}, {'a': 1}, {'a': 2}]
    result = list(unique(data, key=lambda d: d['a']))
    assert result == [{'a': 1}, {'a': 2}]


# ------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ------------------------------------------------------------- isdistinct
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# -------------------------------------------------------------------- take
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


# -------------------------------------------------------------------- tail
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(0, [1, 2, 3]) == []


def test_tail_generator():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)


# -------------------------------------------------------------------- drop
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---------------------------------------------------------------- take_nth
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ------------------------------------------------------------ first/second
def test_first():
    assert first('ABC') == 'A'
    assert first([10]) == 10
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second([1])


# -------------------------------------------------------------------- nth
def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([1, 2, 3])) == 2


# -------------------------------------------------------------------- last
def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# --------------------------------------------------------------------- get
def test_get_scalar():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_list_single():
    assert get([1], 'ABC') == ('B',)


def test_get_list_empty():
    assert get([], 'ABC') == ()


def test_get_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'unknown') == 'unknown'


def test_get_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(5, [1, 2, 3])


def test_get_missing_default_index_error():
    assert get(5, [1, 2, 3], default='x') == 'x'
    assert get('missing', {'a': 1}, default=0) == 0


# ------------------------------------------------------------------ concat
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    def gen():
        yield [1]
        yield [2, 3]
    assert list(concat(gen())) == [1, 2, 3]


# ----------------------------------------------------------------- concatv
def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ------------------------------------------------------------------ mapcat
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                          [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# -------------------------------------------------------------------- cons
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# --------------------------------------------------------------- interpose
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ------------------------------------------------------------- frequencies
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_no_init():
    from operator import add, mul

    def iseven(x):
        return x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    from operator import add
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data, 0) == {False: 9, True: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
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


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# --------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_means():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_short():
    assert list(sliding_window(3, [1, 2])) == []


# -------------------------------------------------------------- partition
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


# ---------------------------------------------------------- partition_all
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_with_len():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_generator():
    gen = (i for i in range(5))
    assert list(partition_all(2, gen)) == [(0, 1), (2, 3), (4,)]


# ------------------------------------------------------------------ count
def test_count():
    assert count([1, 2, 3]) == 3
    assert count([]) == 0
    assert count(iter([1, 2, 3, 4])) == 4


# ------------------------------------------------------------------ pluck
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']
    assert list(pluck(['id', 'name'], data, default=None)) == [
        (1, None), (2, 'Pies')]


# ----------------------------------------------------------------- getter
def test_getter_scalar():
    assert getter(1)('ABC') == 'B'


def test_getter_single_list():
    assert getter([1])('ABC') == ('B',)


def test_getter_multiple():
    assert getter([0, 2])('ABC') == ('A', 'C')


def test_getter_empty():
    assert getter([])('ABC') == ()


# ------------------------------------------------------------------- join
def test_join_inner():
    friends = [('Alice', 'Edith'),
               ('Alice', 'Zhao'),
               ('Edith', 'Alice'),
               ('Zhao', 'Alice'),
               ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'),
              ('Alice', 'Chicago'),
              ('Dan', 'Sydney'),
              ('Edith', 'Paris'),
              ('Edith', 'Berlin'),
              ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    got = sorted(unique(result))
    expected = sorted([
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
    ])
    assert got == expected


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_index_keys():
    friends = [('Alice', 'Edith')]
    cities = [('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ------------------------------------------------------------------- diff
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_no_differences():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ------------------------------------------------------------------ topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 50}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 50})


# ------------------------------------------------------------------ peek
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_generator():
    item, seq = peek(iter([10, 20, 30]))
    assert item == 10
    assert list(seq) == [10, 20, 30]


# ----------------------------------------------------------------- peekn
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, seq2 = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, seq = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(seq) == [1, 2, 3]


# ---------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    assert list(random_sample(0.1, seq, random_state=2016)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_obj():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    assert list(random_sample(0.1, seq, random_state=randobj)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero_and_one():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=0)) == []
    assert list(random_sample(1.0, seq, random_state=0)) == seq
