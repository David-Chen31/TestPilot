import itertools
import random
from operator import add, mul

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


# ---------- remove ----------

def test_remove():
    def iseven(x):
        return x % 2 == 0

    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


# ---------- accumulate ----------

def test_accumulate_basic():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


# ---------- groupby ----------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_key_member():
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


# ---------- merge_sorted ----------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_or_one():
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [])) == []


# ---------- interleave ----------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


# ---------- unique ----------

def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


# ---------- isiterable ----------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


# ---------- isdistinct ----------

def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------- take / drop / take_nth ----------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


# ---------- first / second / nth / last ----------

def test_first_second_last():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert last('ABC') == 'C'


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter([1, 2, 3])) == 2


def test_last_list():
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------- tail ----------

def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


# ---------- get ----------

def test_get_basic():
    assert get(1, 'ABC') == 'B'


def test_get_list_multiple():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'missing') == 'missing'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_index_error():
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_single_list_index():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list_index():
    assert get([], 'ABC') == ()


def test_get_type_error_no_default_raises():
    with pytest.raises(TypeError):
        get((1, 2), 'ABC')


def test_get_type_error_with_default():
    obj = 5
    assert get([0], obj, 'x') == ('x',)


# ---------- concat / concatv ----------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(x for x in [[1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


# ---------- mapcat ----------

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------- cons ----------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------- interpose ----------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------- frequencies ----------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------- reduceby ----------

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    result = reduceby(iseven, add, data, 10)
    assert result == {False: 19, True: 16}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_key_member():
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


def test_reduceby_empty():
    assert reduceby(len, add, []) == {}


# ---------- iterate ----------

def test_iterate():
    def inc(x):
        return x + 1

    counter = iterate(inc, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    powers = iterate(double, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_short_seq():
    assert list(sliding_window(3, [1, 2])) == []


# ---------- partition ----------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


# ---------- partition_all ----------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_exact():
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]


def test_partition_all_generator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [(1, 2), (3, 4), (5,)]


# ---------- count ----------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0


# ---------- pluck ----------

def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, None)) == [None, 'Pies']
    assert list(pluck(['id', 'name'], data, None)) == [(1, None), (2, 'Pies')]


# ---------- getter ----------

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_single_list():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_multi_list():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------- join ----------

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
    result = list(join(second, friends, first, cities))
    expected = [
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
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
    result = list(join(identity, [1, 2],
                       identity, [2, 3],
                       left_default='L'))
    assert result == [(2, 2), ('L', 3)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3],
                       right_default='R'))
    assert result == [(2, 2), (3, 3), (1, 'R')]


def test_join_key_indices():
    friends = [('Alice', 'Edith')]
    cities = [('Alice', 'NYC')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Alice', 'NYC'))]


# ---------- diff ----------

def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_single_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 4]]))
    assert result == [(3, 4)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------- topk ----------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_index_key():
    result = topk(1, [{'v': 1}, {'v': 5}], key='v')
    assert result == ({'v': 5},)


# ---------- peek ----------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, new_seq = peek(seq)
    assert item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, new_seq = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(new_seq) == [1, 2, 3]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


# ---------- peekn ----------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, new_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_short():
    peeked, new_seq = peekn(5, [1, 2])
    assert peeked == (1, 2)
    assert list(new_seq) == [1, 2]


# ---------- random_sample ----------

def test_random_sample_deterministic():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result == result2


def test_random_sample_random_obj():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, list(range(100)), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, list(range(10)), random_state=0)) == list(range(10))
