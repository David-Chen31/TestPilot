import itertools
import operator
import random
from collections import OrderedDict

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


# ----------------------------- remove -----------------------------
def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_all_removed():
    assert list(remove(lambda x: True, [1, 2, 3])) == []


def test_remove_lazy():
    result = remove(lambda x: x < 3, [1, 2, 3, 4])
    assert not isinstance(result, list)
    assert list(result) == [3, 4]


# ----------------------------- accumulate -----------------------------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_initial_empty():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_single():
    assert list(accumulate(operator.add, [5])) == [5]


# ----------------------------- groupby -----------------------------
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_len():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_string_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [{'name': 'Alice', 'gender': 'F'}],
                      'M': [{'name': 'Bob', 'gender': 'M'},
                            {'name': 'Charlie', 'gender': 'M'}]}


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_returns_plain_dict():
    result = groupby(lambda x: x, [1, 2])
    assert type(result) is dict


# ----------------------------- merge_sorted -----------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3],
                             key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_three():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_unequal_lengths():
    assert list(merge_sorted([1, 2, 3], [4])) == [1, 2, 3, 4]
    assert list(merge_sorted([4], [1, 2, 3])) == [1, 2, 3, 4]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_key_empty():
    assert list(merge_sorted([], [], key=lambda x: x)) == []


# ----------------------------- interleave -----------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], []])) == []


def test_interleave_three():
    assert list(interleave([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == \
        [1, 4, 7, 2, 5, 8, 3, 6, 9]


# ----------------------------- unique -----------------------------
def test_unique_tuple():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)


def test_unique_dups():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_lazy():
    result = unique([1, 1, 2])
    assert not isinstance(result, list)
    assert list(result) == [1, 2]


# ----------------------------- isiterable -----------------------------
def test_isiterable_list():
    assert isiterable([1, 2, 3]) is True


def test_isiterable_string():
    assert isiterable('abc') is True


def test_isiterable_int():
    assert isiterable(5) is False


def test_isiterable_generator():
    assert isiterable(iter([1])) is True


# ----------------------------- isdistinct -----------------------------
def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_empty():
    assert isdistinct([]) is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ----------------------------- take -----------------------------
def test_take_basic():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_len():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2])) == []


# ----------------------------- tail -----------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_generator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_len():
    assert list(tail(10, [1, 2])) == [1, 2]


def test_tail_string():
    assert tail(2, 'abc') == 'bc'


# ----------------------------- drop -----------------------------
def test_drop_basic():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2])) == []


# ----------------------------- take_nth -----------------------------
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ----------------------------- first / second -----------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([10, 20, 30]) == 20


def test_second_too_short():
    with pytest.raises(StopIteration):
        second([1])


# ----------------------------- nth -----------------------------
def test_nth_basic():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [1, 2, 3]) == 1
    assert nth(2, (10, 20, 30)) == 30


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_nth_negative_list():
    assert nth(-1, [1, 2, 3]) == 3


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2])


# ----------------------------- last -----------------------------
def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


def test_last_generator():
    assert last(iter([1, 2, 3])) == 3


# ----------------------------- get -----------------------------
def test_get_index():
    assert get(1, 'ABC') == 'B'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678',
                 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, 'nope') == 'nope'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_single_list():
    assert get([1], 'ABC') == ('B',)


def test_get_unhashable_index_type():
    # seq doesn't support list indexing
    with pytest.raises(TypeError):
        get([1, 2], iter([1, 2, 3]))


# ----------------------------- concat / concatv -----------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator_of_seqs():
    assert list(concat(iter([head for head in ([1], [2, 3])]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_empty():
    assert list(concatv()) == []


# ----------------------------- mapcat -----------------------------
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ----------------------------- cons -----------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_infinite():
    result = take(4, cons(0, iterate(lambda x: x + 1, 1)))
    assert list(result) == [0, 1, 2, 3]


# ----------------------------- interpose -----------------------------
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ----------------------------- frequencies -----------------------------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


def test_frequencies_returns_plain_dict():
    result = frequencies([1, 1, 2])
    assert type(result) is dict


# ----------------------------- reduceby -----------------------------
def test_reduceby_add():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_init_constant():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state',
                      lambda acc, x: acc + x['cost'],
                      projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


# ----------------------------- iterate -----------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ----------------------------- sliding_window -----------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == \
        [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


def test_sliding_window_too_short():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == \
        [1.5, 2.5, 3.5]


# ----------------------------- partition -----------------------------
def test_partition_basic():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ----------------------------- partition_all -----------------------------
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator_odd():
    assert list(partition_all(3, iter(range(7)))) == \
        [(0, 1, 2), (3, 4, 5), (6,)]


def test_partition_all_generator_even():
    assert list(partition_all(3, iter(range(6)))) == \
        [(0, 1, 2), (3, 4, 5)]


# ----------------------------- count -----------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ----------------------------- pluck -----------------------------
def test_pluck_dict():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list_index():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == \
        [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_default():
    data = [[1, 2], [1]]
    assert list(pluck([0, 1], data, default=None)) == \
        [(1, 2), (1, None)]


# ----------------------------- getter -----------------------------
def test_getter_single():
    g = getter('name')
    assert g({'name': 'x'}) == 'x'


def test_getter_list_of_one():
    g = getter([0])
    assert g([5, 6]) == (5,)


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([5, 6, 7]) == (5, 6)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2]) == ()


# ----------------------------- join -----------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'),
              ('Dan', 'Sydney'), ('Edith', 'Paris'),
              ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    out = list(unique(result))
    assert len(out) == 10
    # Each pair should exist
    for ((a, b), (c, d)) in out:
        assert b == c
        assert (a, d) in {x for x in [(x[0], y[1])
                                       for x in friends for y in cities
                                       if x[1] == y[0]]}


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_right_default():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3],
                       right_default='R'))
    assert result == [(2, 2), (3, 3), (1, 'R')]


def test_join_left_default():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3, 4],
                       left_default='L'))
    assert result == [(2, 2), ('L', 3), ('L', 4)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ----------------------------- diff -----------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 5, 3], [1, 2, 3])) == [(2, 5, 2)]


def test_diff_single_seqs_list():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ----------------------------- topk -----------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    assert topk(2, [{'v': 1}, {'v': 5}, {'v': 3}], key='v') == \
        ({'v': 5}, {'v': 3})


def test_topk_k_gt_len():
    assert topk(10, [1, 2]) == (2, 1)


# ----------------------------- peek -----------------------------
def test_peek():
    first_item, seq = peek([0, 1, 2])
    assert first_item == 0
    assert list(seq) == [0, 1, 2]


def test_peek_generator():
    first_item, seq = peek(iter([5, 6, 7]))
    assert first_item == 5
    assert list(seq) == [5, 6, 7]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ----------------------------- peekn -----------------------------
def test_peekn():
    peeked, seq = peekn(2, [0, 1, 2, 3, 4])
    assert peeked == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, seq = peekn(10, [0, 1])
    assert peeked == (0, 1)
    assert list(seq) == [0, 1]


def test_peekn_zero():
    peeked, seq = peekn(0, [0, 1, 2])
    assert peeked == ()
    assert list(seq) == [0, 1, 2]


# ----------------------------- random_sample -----------------------------
def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(100), random_state=0)) == \
        list(range(100))


def test_random_sample_deterministic_int_seed():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_deterministic_randobj():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_is_lazy():
    result = random_sample(0.5, range(10), random_state=0)
    assert not isinstance(result, list)
    assert list(result) is not None
