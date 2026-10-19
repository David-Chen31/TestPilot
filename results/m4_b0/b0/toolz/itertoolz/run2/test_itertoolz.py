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


# ----------------------------- remove -------------------------------------
def test_remove_basic():
    def iseven(x):
        return x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []


# ---------------------------- accumulate ----------------------------------
def test_accumulate_no_initial():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_is_lazy_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6


# ----------------------------- groupby ------------------------------------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_key_non_callable():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {5: ['Alice', 'Edith', 'Frank'],
                      3: ['Bob', 'Dan'],
                      7: ['Charlie']}


def test_groupby_member():
    people = [{'name': 'Alice', 'gender': 'F'},
              {'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', people)
    assert result == {'F': [{'name': 'Alice', 'gender': 'F'}],
                      'M': [{'name': 'Bob', 'gender': 'M'},
                            {'name': 'Charlie', 'gender': 'M'}]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# --------------------------- merge_sorted ---------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_many_seqs():
    seqs = [[1, 4, 7], [2, 5, 8], [3, 6, 9], [0, 10]]
    assert list(merge_sorted(*seqs)) == list(range(11))


# ---------------------------- interleave ----------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_uneven():
    assert list(interleave([[1, 2, 3], [4]])) == [1, 4, 2, 3]


def test_interleave_empty():
    assert list(interleave([])) == []


# ------------------------------ unique ------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------- isiterable ----------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------------------- isdistinct ----------------------------------
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ------------------------- take / drop / take_nth -------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ----------------------- tail / first / second ----------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_less_than_n():
    assert tail(5, [1, 2]) == [1, 2]


def test_first():
    assert first('ABC') == 'A'
    assert first([10, 20]) == 10
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([10, 20, 30]) == 20
    with pytest.raises(StopIteration):
        second([1])


# ----------------------------- nth ----------------------------------------
def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(-1, [10, 20, 30]) == 30


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


# ----------------------------- last ---------------------------------------
def test_last():
    assert last('ABC') == 'C'
    assert last([10, 20, 30]) == 30
    assert last(iter([1, 2, 3])) == 3


# ----------------------------- get ----------------------------------------
def test_get_basic():
    assert get(1, 'ABC') == 'B'
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, None) is None


def test_get_single_list_no_default():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()
    assert get([], 'ABC', None) == ()


def test_get_missing_no_default():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': '555-1234'})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


# --------------------- concat / concatv / mapcat -------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    gen = (x for x in ([1], [2, 3]))
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---------------------------- cons ----------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---------------------------- interpose -----------------------------------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# --------------------------- frequencies ----------------------------------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------- reduceby ------------------------------------
def test_reduceby_basic():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init_value():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------- iterate -------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]

    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ------------------------- sliding_window ---------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_too_short():
    assert list(sliding_window(5, [1, 2, 3])) == []


# ---------------------------- partition -----------------------------------
def test_partition_exact():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


# -------------------------- partition_all ---------------------------------
def test_partition_all_exact():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == [(1, 2, 3), (4, 5)]


# ----------------------------- count --------------------------------------
def test_count_sequence():
    assert count([1, 2, 3]) == 3
    assert count("abc") == 3


def test_count_iterator():
    assert count(iter([1, 2, 3])) == 3
    assert count(iter([])) == 0


# ----------------------------- pluck --------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_single_element_list():
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]


# ----------------------------- getter -------------------------------------
def test_getter_simple():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_single_list():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ------------------------------ join --------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    assert len(result) == 9
    assert (('Alice', 'Zhao'), ('Zhao', 'Shanghai')) in result
    assert (('Edith', 'Alice'), ('Alice', 'NYC')) in result


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=str) == sorted([(2, 2), (3, 3), (None, 4), (1, None)], key=str)


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], left_default=None))
    assert sorted(result, key=str) == sorted([(2, 2), (None, 3)], key=str)


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3], right_default=None))
    assert sorted(result, key=str) == sorted([(2, 2), (3, 3), (1, None)], key=str)


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris')),
                      (('Alice', 'Zhao'), ('Edith', 'Paris'))]


# ------------------------------ diff --------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_with_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)) == [
        ('bananas', 'Oranges')]


def test_diff_single_list_arg():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3])) == [(2, 2, 5)]


# ------------------------------ topk --------------------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_with_non_callable_key():
    data = [{'v': 1}, {'v': 5}, {'v': 3}]
    assert topk(2, data, key='v') == ({'v': 5}, {'v': 3})


# ------------------------------ peek --------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([1, 2, 3])
    item, rest = peek(it)
    assert item == 1
    assert list(rest) == [1, 2, 3]


# ----------------------------- peekn --------------------------------------
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# --------------------------- random_sample --------------------------------
def test_random_sample_deterministic():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=0)) == list(range(10))


def test_random_sample_custom_random_state():
    from random import Random
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=Random(2016)))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
