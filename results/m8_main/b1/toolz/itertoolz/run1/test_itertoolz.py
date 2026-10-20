import itertools
import operator
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

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_all_removed():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_none_removed():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------- accumulate

def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_single_element():
    assert list(accumulate(operator.add, [5])) == [5]


# ---------------------------------------------------------------- groupby

def test_groupby_callable_key():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_len_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}


def test_groupby_member_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------- merge_sorted

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [], [])) == []


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3], [4, 5])) == [1, 2, 3, 4, 5]


def test_merge_sorted_with_key_empty_inputs():
    assert list(merge_sorted([], [1, 2], key=lambda x: x)) == [1, 2]


# ---------------------------------------------------------------- interleave

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ---------------------------------------------------------------- unique

def test_unique_no_key():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_all_same():
    assert tuple(unique([1, 1, 1, 1])) == (1,)


# ---------------------------------------------------------------- isiterable

def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(iter([1])) is True


def test_isiterable_false():
    assert isiterable(5) is False
    assert isiterable(None) is False


# ---------------------------------------------------------------- isdistinct

def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct("World") is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ---------------------------------------------------------------- take / drop

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2, 3])) == [1, 2, 3]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2, 3])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_1():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------- tail

def test_tail_sequence():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_length_short():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3])) == (2, 3)


# ---------------------------------------------------------------- first / second / nth / last

def test_first():
    assert first('ABC') == 'A'


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_nth_on_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(2, [10, 20, 30]) == 30
    assert nth(0, (1, 2, 3)) == 1


def test_nth_on_iterator():
    assert nth(1, iter([10, 20, 30])) == 20


def test_last_list():
    assert last('ABC') == 'C'


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------- get

def test_get_index():
    assert get(1, 'ABC') == 'B'


def test_get_list_of_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_dict_list():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default_missing():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_index_error_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_single_default():
    assert get('missing', {'a': 1}, 'default') == 'default'


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


# ---------------------------------------------------------------- concat / concatv

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concat_empty():
    assert list(concat([])) == []


# ---------------------------------------------------------------- mapcat

def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------- cons

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


# ---------------------------------------------------------------- interpose

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------- frequencies

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2,
    }


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby

def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    result = reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_constant_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4], 10)
    assert result == {False: 14, True: 16}


def test_reduceby_member_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------- iterate

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert [next(it) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------- sliding_window

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_means():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------- partition

def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_incomplete():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


# ---------------------------------------------------------------- partition_all

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [(1, 2), (3, 4), (5,)]


def test_partition_all_single_remainder():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


# ---------------------------------------------------------------- count

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------------------------------------------------------------- pluck

def test_pluck_string_key():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list_indices():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_single_index():
    assert list(pluck(0, [[1, 2], [3, 4]])) == [1, 3]


def test_pluck_with_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=None)) == [1, None]


def test_pluck_list_with_default():
    data = [[1, 2], [3]]
    result = list(pluck([0, 1], data, default=None))
    assert result == [(1, 2), (3, None)]


# ---------------------------------------------------------------- getter

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_single_element_list():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------------------------------------------------------------- join

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
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
    assert sorted(result) == sorted(expected)


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=repr) == sorted([(2, 2), (3, 3), (None, 4), (1, None)], key=repr)


def test_join_right_join():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [2, 3],
                       left_default=None))
    assert sorted(result, key=repr) == sorted([(2, 2), (None, 3)], key=repr)


def test_join_left_join():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [2, 3],
                       right_default=None))
    assert sorted(result, key=repr) == sorted([(2, 2), (1, None)], key=repr)


def test_join_member_keys():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai')]
    result = list(join(1, friends, 0, cities))
    expected = [
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
    ]
    assert sorted(result) == sorted(expected)


# ---------------------------------------------------------------- diff

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_with_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_no_differences():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ---------------------------------------------------------------- topk

def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_member_key():
    data = [{'v': 1}, {'v': 5}, {'v': 3}]
    assert topk(2, data, key='v') == ({'v': 5}, {'v': 3})


def test_topk_k_larger_than_seq():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------------------------------------------------------------- peek

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_item, rest = peek(seq)
    assert first_item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([1, 2, 3])
    first_item, rest = peek(it)
    assert first_item == 1
    assert list(rest) == [1, 2, 3]


# ---------------------------------------------------------------- peekn

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    first_three, rest = peekn(3, [1, 2])
    assert first_three == (1, 2)
    assert list(rest) == [1, 2]


def test_peekn_iterator():
    it = iter([1, 2, 3, 4])
    first_two, rest = peekn(2, it)
    assert first_two == (1, 2)
    assert list(rest) == [1, 2, 3, 4]


# ---------------------------------------------------------------- random_sample

def test_random_sample_seed_deterministic():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert result1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_zero_prob():
    assert list(random_sample(0.0, list(range(100)), random_state=42)) == []


def test_random_sample_one_prob():
    assert list(random_sample(1.0, list(range(10)), random_state=42)) == list(range(10))


class _FakeRandom:
    def __init__(self, values):
        self._values = iter(values)

    def random(self):
        return next(self._values)


def test_random_sample_custom_random_object():
    fake = _FakeRandom([0.05, 0.5, 0.05])
    result = list(random_sample(0.1, [1, 2, 3], random_state=fake))
    assert result == [1, 3]
