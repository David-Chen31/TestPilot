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
from toolz.utils import no_default


# ---------- remove ----------

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------- accumulate ----------

def test_accumulate_no_initial():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


# ---------- groupby ----------

def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert result == {False: [1, 3, 5], True: [2, 4, 6]}


def test_groupby_member():
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

def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_unequal_lengths():
    assert list(merge_sorted([1, 2, 3], [4, 5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([4, 5], [1, 2, 3])) == [1, 2, 3, 4, 5]


# ---------- interleave ----------

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_unequal():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_all_empty():
    assert list(interleave([[], []])) == []


# ---------- unique ----------

def test_unique_basic():
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

def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_str():
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
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


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
    assert nth(0, (1, 2, 3)) == 1
    assert nth(1, iter([10, 20, 30])) == 20


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last((1,)) == 1


# ---------- tail ----------

def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_tuple():
    assert tail(2, (1, 2, 3)) == (2, 3)


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3])) == (2, 3)


def test_tail_string():
    assert tail(2, 'abcd') == 'cd'


# ---------- get ----------

def test_get_simple():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555-1234'}) == '555-1234'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get(['Alice', 'Bob'], {'Alice': 1, 'Bob': 2}) == (1, 2)


def test_get_list_single():
    assert get(['A'], {'A': 1}) == (1,)


def test_get_list_empty():
    assert get([], {'A': 1}) == ()


def test_get_with_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'default') == 'default'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_list_missing_raises():
    with pytest.raises(KeyError):
        get(['Alice', 'Dennis'], {'Alice': 1})


# ---------- concat / concatv / mapcat ----------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_empty():
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------- cons / interpose ----------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------- frequencies ----------

def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------- reduceby ----------

def test_reduceby_no_init():
    result = reduceby(lambda x: x % 2 == 0, operator.add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    result = reduceby(lambda x: x % 2 == 0, operator.mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_with_init_constant():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# ---------- iterate ----------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------- sliding_window ----------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_size_1():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


# ---------- partition ----------

def test_partition_exact():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------- partition_all ----------

def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_single():
    assert list(partition_all(3, [1])) == [(1,)]


def test_partition_all_generator_no_len():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == [(1, 2), (3, 4), (5,)]


# ---------- count ----------

def test_count_with_len():
    assert count([1, 2, 3]) == 3
    assert count("abc") == 3


def test_count_lazy():
    assert count(iter([1, 2, 3, 4])) == 4
    assert count(x for x in range(5)) == 5


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------- pluck ----------

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_with_default():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2}]
    result = list(pluck(['id', 'name'], data, default=None))
    assert result == [(1, 'Cheese'), (2, None)]


# ---------- getter ----------

def test_getter_single_index():
    g = getter(0)
    assert g([10, 20]) == 10


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
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4]))
    assert result == [(2, 2), (3, 3)]


def test_join_inner_multiple_matches():
    left = [('a', 1), ('a', 2), ('b', 3)]
    right = [('a', 'x'), ('b', 'y')]
    result = list(join(0, left, 0, right))
    assert result == [(('a', 1), ('a', 'x')),
                      (('a', 2), ('a', 'x')),
                      (('b', 3), ('b', 'y'))]


def test_join_right_join():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_join():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 4],
                       right_default=None))
    assert result == [(2, 2), (1, None), (3, None)]


def test_join_full_join():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_index_keys():
    friends = [('Alice', 'Edith'), ('Edith', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris')),
                      (('Edith', 'Alice'), ('Alice', 'NYC'))]


# ---------- diff ----------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
    assert result == [(3, 3, 4)]


def test_diff_list_of_seqs():
    result = list(diff([[1, 2, 3], [1, 2, 10]]))
    assert result == [(3, 10)]


# ---------- topk ----------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_key_non_callable():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ---------- peek ----------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([1, 2, 3])
    item, it2 = peek(it)
    assert item == 1
    assert list(it2) == [1, 2, 3]


# ---------- peekn ----------

def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq2 = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    items, seq2 = peekn(10, [1, 2])
    assert items == (1, 2)
    assert list(seq2) == [1, 2]


def test_peekn_zero():
    items, seq2 = peekn(0, [1, 2, 3])
    assert items == ()
    assert list(seq2) == [1, 2, 3]


# ---------- random_sample ----------

def test_random_sample_seeded_is_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2
    assert r1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(10))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_custom_random_state():
    class FixedRandom:
        def random(self):
            return 0.05
    seq = list(range(5))
    # 0.05 < 0.1 for all -> all pass
    assert list(random_sample(0.1, seq, random_state=FixedRandom())) == seq
