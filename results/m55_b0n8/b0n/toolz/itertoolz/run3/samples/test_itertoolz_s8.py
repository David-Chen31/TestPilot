import itertools
import operator
import random

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


# ---------------- remove ----------------

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_is_lazy():
    result = remove(lambda x: x < 3, iter([1, 2, 3, 4]))
    assert list(result) == [3, 4]


def test_remove_empty():
    assert list(remove(bool, [])) == []


# ---------------- accumulate ----------------

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


def test_accumulate_with_initial_zero():
    # initial == no_default uses == comparison, but 0 != no_default
    assert list(accumulate(operator.add, [1, 2], 0)) == [0, 1, 3]


# ---------------- groupby ----------------

def test_groupby_callable():
    result = groupby(lambda x: x % 2, [1, 2, 3, 4])
    assert result == {1: [1, 3], 0: [2, 4]}


def test_groupby_string_member():
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
    assert groupby(lambda x: x, []) == {}


# ---------------- merge_sorted ----------------

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_seqs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_key_multi():
    result = list(merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9]))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ---------------- interleave ----------------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], [1, 2]])) == [1, 2]


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ---------------- unique ----------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------- isiterable ----------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------- isdistinct ----------------

def test_isdistinct_iterable():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_empty():
    assert isdistinct([]) is True


# ---------------- take ----------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(5, [1, 2])) == [1, 2]


# ---------------- tail ----------------

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_short():
    assert tail(5, [1, 2]) == [1, 2]


# ---------------- drop ----------------

def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---------------- take_nth ----------------

def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, [1, 2, 3, 4])) == [1, 4]


# ---------------- first ----------------

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


# ---------------- second ----------------

def test_second():
    assert second('ABC') == 'B'


def test_second_short():
    with pytest.raises(StopIteration):
        second('A')


# ---------------- nth ----------------

def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'


def test_nth_out_of_range():
    with pytest.raises(StopIteration):
        nth(10, iter([1, 2, 3]))


# ---------------- last ----------------

def test_last():
    assert last('ABC') == 'C'
    assert last(iter([1, 2, 3])) == 3


def test_last_empty():
    with pytest.raises(IndexError):
        last([])


# ---------------- get ----------------

def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_list_indices_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678',
                 'Charlie': '555-9999'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '1'})


def test_get_single_list_default():
    assert get([5], [1, 2, 3], None) == (None,)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_empty_list_default():
    assert get([], 'ABC', None) == ()


def test_get_list_default_with_indexerror():
    assert get([0, 5], [1, 2, 3], 'zzz') == (1, 'zzz')


def test_get_type_error_with_default():
    assert get(1.5, [1, 2, 3], 'default') == 'default'


def test_get_type_error_no_default():
    with pytest.raises(TypeError):
        get(1.5, [1, 2, 3])


# ---------------- concat ----------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]


def test_concat_empty():
    assert list(concat([])) == []


# ---------------- concatv ----------------

def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_empty():
    assert list(concatv()) == []


# ---------------- mapcat ----------------

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_mapcat_empty():
    assert list(mapcat(list, [])) == []


# ---------------- cons ----------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


def test_cons_infinite():
    result = list(take(3, cons(0, iterate(lambda x: x + 1, 1))))
    assert result == [0, 1, 2]


# ---------------- interpose ----------------

def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------- frequencies ----------------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------- reduceby ----------------

def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2, operator.add, [1, 2, 3, 4, 5])
    assert result == {1: 9, 0: 6}


def test_reduceby_no_init_with_initial_value():
    result = reduceby(lambda x: x % 2, operator.mul, [1, 2, 3, 4, 5])
    assert result == {1: 15, 0: 8}


def test_reduceby_with_init():
    data = [1, 2, 3, 4, 5]
    result = reduceby(lambda x: x % 2, operator.add, data, 0)
    assert result == {1: 9, 0: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_key_index():
    projects = [
        {'name': 'a', 'state': 'CA', 'cost': 100},
        {'name': 'b', 'state': 'IL', 'cost': 200},
        {'name': 'c', 'state': 'CA', 'cost': 300},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 400, 'IL': 200}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------- iterate ----------------

def test_iterate():
    gen = iterate(lambda x: x + 1, 0)
    assert [next(gen) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    gen = iterate(lambda x: x * 2, 1)
    assert [next(gen) for _ in range(4)] == [1, 2, 4, 8]


# ---------------- sliding_window ----------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ---------------- partition ----------------

def test_partition_basic():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------- partition_all ----------------

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator():
    def gen():
        yield 1
        yield 2
        yield 3
    assert list(partition_all(2, gen())) == [(1, 2), (3,)]


def test_partition_all_sequence_type():
    # list has __len__
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_exact_multiple():
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]


# ---------------- count ----------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------------- pluck ----------------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, None)) == [None, 'Pies']


def test_pluck_default_multiple():
    data = [{'a': 1}, {'a': 2, 'b': 3}]
    assert list(pluck(['a', 'b'], data, None)) == [(1, None), (2, 3)]


# ---------------- getter ----------------

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_one():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multi():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_list_empty():
    g = getter([])
    assert g('ABC') == ()


# ---------------- join ----------------

def test_join_inner():
    left = [('a', 1), ('b', 2), ('a', 3)]
    right = [('a', 'x'), ('c', 'y')]
    result = list(join(first, left, first, right))
    assert sorted(result) == [(('a', 1), ('a', 'x')), (('a', 3), ('a', 'x'))]


def test_join_right():
    left = [('a', 1)]
    right = [('a', 'x'), ('c', 'y')]
    result = list(join(first, left, first, right, left_default=None))
    assert result == [(('a', 1), ('a', 'x')), (None, ('c', 'y'))]


def test_join_left():
    left = [('a', 1), ('b', 2)]
    right = [('a', 'x')]
    result = list(join(first, left, first, right, right_default=None))
    assert result == [(('a', 1), ('a', 'x')), (('b', 2), None)]


def test_join_full():
    result = list(join(lambda x: x, [1, 2, 3],
                       lambda x: x, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_index_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai')]
    result = list(join(1, friends, 0, cities))
    assert (('Alice', 'Edith'), ('Zhao', 'Shanghai')) not in result
    assert (('Alice', 'Zhao'), ('Zhao', 'Shanghai')) in result


# ---------------- diff ----------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_three():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
    assert result == [(3, 3, 4)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 4]]))
    assert result == [(3, 4)]


def test_diff_single_list_of_seqs():
    # A list of sequences is unpacked
    result = list(diff([[1, 2, 3], [1, 2, 4], [1, 2, 5]]))
    assert result == [(3, 4, 5)]


# ---------------- topk ----------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    result = topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    assert result == ('Charlie', 'Alice')


def test_topk_key_non_callable():
    data = [{'x': 1}, {'x': 5}, {'x': 3}]
    result = topk(2, data, key='x')
    assert result == ({'x': 5}, {'x': 3})


def test_topk_empty():
    assert topk(2, []) == ()


# ---------------- peek ----------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_el, rest = peek(seq)
    assert first_el == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(rest) == [1, 2, 3]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---------------- peekn ----------------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    peeked, rest = peekn(10, [0, 1])
    assert peeked == (0, 1)
    assert list(rest) == [0, 1]


def test_peekn_zero():
    peeked, rest = peekn(0, [0, 1])
    assert peeked == ()
    assert list(rest) == [0, 1]


# ---------------- random_sample ----------------

def test_random_sample_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_rand_obj():
    randobj = random.Random(2016)
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=1)) == [1, 2, 3]
