import itertools
import operator
import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)


# ---------- remove ----------

def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


# ---------- accumulate ----------

def test_accumulate_no_initial():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_is_lazy():
    it = accumulate(operator.add, [1, 2, 3])
    assert next(it) == 1
    assert next(it) == 3
    assert next(it) == 6


# ---------- groupby ----------

def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6])
    assert result == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_names_by_len():
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


# ---------- merge_sorted ----------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [])) == []
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [1, 2])) == [1, 2]


def test_merge_sorted_more_seqs():
    result = list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9]))
    assert result == list(range(1, 10))


def test_merge_sorted_key_more_seqs():
    a = [1, 4, 7]
    b = [2, 5, 8]
    c = [3, 6, 9]
    result = list(merge_sorted(a, b, c, key=lambda x: x))
    assert result == list(range(1, 10))


# ---------- interleave ----------

def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


# ---------- unique ----------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------- isiterable ----------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


# ---------- isdistinct ----------

def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 1])) is False


# ---------- take / drop ----------

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


# ---------- tail ----------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_short():
    assert tail(5, [1, 2]) == [1, 2]


# ---------- first / second / nth / last ----------

def test_first_second_last():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert last('ABC') == 'C'


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10


def test_nth_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


# ---------- get ----------

def test_get_basic():
    assert get(1, 'ABC') == 'B'
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678',
                 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'N/A') == 'N/A'


def test_get_single_list_index():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list_index():
    assert get([], 'ABC') == ()


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_type_error_no_default_raises():
    with pytest.raises(TypeError):
        get('a', 5)


def test_get_type_error_with_default():
    assert get('a', 5, 'x') == 'x'


# ---------- concat / concatv ----------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    gen = (range(i) for i in range(3))
    assert list(concat(gen)) == [0, 0, 1]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ---------- mapcat ----------

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------- cons / interpose ----------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose('a', [1])) == [1]


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


def test_reduceby_with_constant_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_with_callable_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 1, 2, 3]

    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(iseven, set_add, data, set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'state': 'CA', 'cost': 1000000},
                {'state': 'IL', 'cost': 100000},
                {'state': 'IL', 'cost': 2000000},
                {'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------- iterate ----------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(5, [1, 2])) == []


# ---------- partition ----------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


# ---------- partition_all ----------

def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == [(1, 2, 3), (4, 5)]
    assert list(partition_all(3, iter([1, 2, 3, 4, 5, 6, 7]))) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_with_sequence_len():
    assert list(partition_all(3, (1, 2, 3, 4))) == [(1, 2, 3), (4,)]


# ---------- count ----------

def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3])) == 3
    assert count(iter([])) == 0
    assert count("abcd") == 4


# ---------- pluck ----------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=0)) == [1, 0]


def test_pluck_default_multiple():
    data = [{'a': 1, 'b': 2}, {'a': 3}]
    assert list(pluck(['a', 'b'], data, default=None)) == \
        [(1, 2), (3, None)]


# ---------- getter ----------

def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list_single():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([10, 20, 30]) == (10, 20)


def test_getter_empty_list():
    g = getter([])
    assert g([10, 20]) == ()


# ---------- join ----------

def test_join_inner():
    left = [1, 2, 3]
    right = [2, 3, 4]
    identity = lambda x: x
    result = list(join(identity, left, identity, right))
    assert sorted(result) == [(2, 2), (3, 3)]


def test_join_right():
    left = [1, 2, 3]
    right = [2, 3, 4]
    identity = lambda x: x
    result = list(join(identity, left, identity, right, left_default=None))
    assert sorted(result) == [(2, 2), (3, 3), (None, 4)]


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda x: (x[0] is None, x[0])) == \
        [(1, None), (2, 2), (3, 3), (None, 4)]


def test_join_left():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert sorted(result, key=lambda x: x[0]) == [(1, None), (2, 2), (3, 3)]


def test_join_friends_cities():
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

    def first(x):
        return x[0]

    def second(x):
        return x[1]

    result = list(join(second, friends, first, cities))
    pairs = {(a, d) for ((a, b), (c, d)) in result}
    assert pairs == {
        ('Alice', 'Berlin'), ('Alice', 'Paris'), ('Alice', 'Shanghai'),
        ('Edith', 'Chicago'), ('Edith', 'NYC'),
        ('Zhao', 'Chicago'), ('Zhao', 'NYC'), ('Zhao', 'Berlin'),
        ('Zhao', 'Paris'),
    }


def test_join_non_callable_keys():
    left = [('a', 1), ('b', 2)]
    right = [('a', 10), ('c', 30)]
    result = list(join(0, left, 0, right))
    assert result == [(('a', 1), ('a', 10))]


# ---------- diff ----------

def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_with_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 5, 3], [1, 2, 3])) == [(2, 5, 2)]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ---------- topk ----------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_returns_tuple():
    result = topk(1, [3, 1, 2])
    assert isinstance(result, tuple)
    assert result == (3,)


# ---------- peek ----------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    item, rest = peek(it)
    assert item == 10
    assert list(rest) == [10, 20, 30]


# ---------- peekn ----------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# ---------- random_sample ----------

def test_random_sample_seeded_deterministic():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_with_random_object():
    from random import Random
    seq = list(range(100))
    randobj = Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    seq = list(range(100))
    assert list(random_sample(0.0, seq, random_state=42)) == []


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=42)) == list(range(10))
