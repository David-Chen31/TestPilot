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


# ----------------------------- remove -----------------------------
def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_returns_iterator():
    res = remove(lambda x: x > 1, [1, 2, 3])
    assert iter(res) is res
    assert list(res) == [1]


# ----------------------------- accumulate -----------------------------
def test_accumulate_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_empty():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_with_initial_empty():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_is_generator():
    g = accumulate(operator.add, [1, 2])
    assert next(g) == 1
    assert next(g) == 3


# ----------------------------- groupby -----------------------------
def test_groupby_callable():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_non_callable():
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


# ----------------------------- merge_sorted -----------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]


def test_merge_sorted_many():
    seqs = [[1, 4, 7], [2, 5, 8], [3, 6, 9], [0, 10]]
    assert list(merge_sorted(*seqs)) == sorted(sum(seqs, []))


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3, 4, 5], [6])) == [1, 2, 3, 4, 5, 6]
    assert list(merge_sorted([1], [2, 3, 4, 5, 6])) == [1, 2, 3, 4, 5, 6]


# ----------------------------- interleave -----------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_inner():
    assert list(interleave([[], []])) == []


# ----------------------------- unique -----------------------------
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ----------------------------- isiterable -----------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


# ----------------------------- isdistinct -----------------------------
def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_str():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ----------------------------- take / drop / tail -----------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(5, [1, 2])) == []


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_n_larger():
    assert tail(10, iter([1, 2])) == (1, 2)


# ----------------------------- take_nth -----------------------------
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ----------------------------- first / second / nth / last -----------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(1, iter([10, 20, 30])) == 20


def test_nth_index_error():
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# ----------------------------- get -----------------------------
def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default_dict():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default_missing_scalar():
    phonebook = {'Alice': '555-1234'}
    assert get('Dennis', phonebook, None) is None
    assert get('Dennis', phonebook, 'default') == 'default'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})


def test_get_index_error():
    with pytest.raises(IndexError):
        get(10, [1, 2])


def test_get_index_error_default():
    assert get(10, [1, 2], None) is None


def test_get_type_error_default():
    assert get(1, 5, 'default') == 'default'


# ----------------------------- concat / concatv / mapcat -----------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    gen = (x for x in [[1], [2, 3]])
    assert list(concat(gen)) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ----------------------------- cons / interpose -----------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_cons_infinite():
    res = take(5, cons(0, iterate(lambda x: x + 1, 1)))
    assert list(res) == [0, 1, 2, 3, 4]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ----------------------------- frequencies -----------------------------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ----------------------------- reduceby -----------------------------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5]) == \
        {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.mul, [1, 2, 3, 4, 5]) == \
        {False: 15, True: 8}


def test_reduceby_constant_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5], 0) == \
        {False: 9, True: 6}


def test_reduceby_callable_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5], lambda: 0) == \
        {False: 9, True: 6}


def test_reduceby_callable_init_set():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set) == \
        {True: {2, 4}, False: {1, 3}}


def test_reduceby_key_index():
    projects = [{'state': 'CA', 'cost': 10},
                {'state': 'IL', 'cost': 20},
                {'state': 'IL', 'cost': 5}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 10, 'IL': 25}


def test_reduceby_empty():
    assert reduceby(len, operator.add, []) == {}


# ----------------------------- iterate -----------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(5)] == [0, 1, 2, 3, 4]


def test_iterate_double():
    it = iterate(lambda x: x * 2, 1)
    assert list(take(5, it)) == [1, 2, 4, 8, 16]


# ----------------------------- sliding_window -----------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == \
        [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_short():
    assert list(sliding_window(2, [1])) == []


def test_sliding_window_size_1():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ----------------------------- partition -----------------------------
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_uneven():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ----------------------------- partition_all -----------------------------
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_uneven():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_single():
    assert list(partition_all(3, [1])) == [(1,)]


# ----------------------------- count -----------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ----------------------------- pluck / getter -----------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multi():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_multi_single():
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]


def test_pluck_default():
    data = [{'id': 1}, {'x': 2}]
    assert list(pluck('id', data, default=None)) == [1, None]


def test_pluck_default_list():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck(['a', 'b'], data, default=0)) == [(1, 0), (0, 2)]


def test_getter_single():
    g = getter(0)
    assert g(['a', 'b']) == 'a'


def test_getter_list_multi():
    g = getter([0, 1])
    assert g(['a', 'b', 'c']) == ('a', 'b')


def test_getter_list_single():
    g = getter([0])
    assert g(['a', 'b']) == ('a',)


def test_getter_list_empty():
    g = getter([])
    assert g(['a', 'b']) == ()


# ----------------------------- join -----------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Dan', 'Sydney'), ('Edith', 'Paris')]
    result = list(join(second, friends, first, cities))
    # The join is on the friend's name (second element of friends) with the
    # city's name (first element of cities).  Assert the exact result the
    # implementation produces.
    assert result == [
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Alice', 'Edith'), ('Edith', 'Paris')),
    ]


def test_join_inner_no_match():
    result = list(join(operator.itemgetter(0), [(1, 'a')],
                       operator.itemgetter(0), [(2, 'b')]))
    assert result == []


def test_join_right():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_full():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda t: (t[0] is None, t[1] is None,
                                          t[0] if t[0] is not None else -1)) \
        == sorted([(2, 2), (3, 3), (None, 4), (1, None)],
                  key=lambda t: (t[0] is None, t[1] is None,
                                 t[0] if t[0] is not None else -1))


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Edith', 'Paris'), ('Zhao', 'Shanghai')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris')),
                      (('Alice', 'Zhao'), ('Zhao', 'Shanghai'))]


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
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3])) == [(2, 2, 5)]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


# ----------------------------- topk -----------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    recs = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, recs, key='v') == ({'v': 100}, {'v': 10})


def test_topk_more_than_len():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ----------------------------- peek / peekn -----------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_el, rest = peek(seq)
    assert first_el == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    seq = iter([1, 2, 3])
    first_el, rest = peek(seq)
    assert first_el == 1
    assert list(rest) == [1, 2, 3]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_iterator():
    seq = iter([1, 2, 3, 4])
    peeked, rest = peekn(2, seq)
    assert peeked == (1, 2)
    assert list(rest) == [1, 2, 3, 4]


def test_peekn_more_than_available():
    peeked, rest = peekn(5, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(rest) == [1, 2, 3]


# ----------------------------- random_sample -----------------------------
def test_random_sample_seed_deterministic():
    seq = list(range(100))
    r1 = list(random_sample(0.1, seq, random_state=2016))
    r2 = list(random_sample(0.1, seq, random_state=2016))
    assert r1 == r2


def test_random_sample_prob_zero():
    seq = list(range(50))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_prob_one():
    seq = list(range(50))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_with_random_object():
    import random
    randobj = random.Random(2016)
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
