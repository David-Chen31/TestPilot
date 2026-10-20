import itertools
import pickle
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
from toolz.itertoolz import _merge_sorted_binary_key
from toolz.utils import no_default


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_basic():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


def test_accumulate_is_lazy_generator():
    result = accumulate(add, [1, 2, 3])
    assert iter(result) is result


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable_key():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_string_key():
    data = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    result = groupby('gender', data)
    assert result == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'},
        ],
    }


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_none():
    assert list(merge_sorted()) == []


def test_merge_sorted_with_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_with_key_extra():
    left = [1, 4, 7]
    right = [2, 5, 8]
    result = list(merge_sorted(left, right, key=lambda x: x))
    assert result == [1, 2, 4, 5, 7, 8]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3, 4], [5])) == [1, 2, 3, 4, 5]
    assert list(merge_sorted([5], [1, 2, 3, 4])) == [1, 2, 3, 4, 5]


def test_merge_sorted_many_seqs_deep_recursion():
    a = [1, 4, 7, 10]
    b = [2, 5, 8, 11]
    c = [3, 6, 9, 12]
    d = [0, 13]
    assert list(merge_sorted(a, b, c, d)) == list(range(14))


def test_merge_sorted_many_seqs_deep_recursion_key():
    a = [{'v': 1}, {'v': 4}]
    b = [{'v': 2}, {'v': 5}]
    c = [{'v': 3}, {'v': 6}]
    result = list(merge_sorted(a, b, c, key=lambda x: x['v']))
    assert [x['v'] for x in result] == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_empty_first_seq():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_empty_first_seq_key():
    assert list(merge_sorted([], [1, 2, 3], key=lambda x: x)) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [], key=lambda x: x)) == [1, 2, 3]


def test_merge_sorted_key_all_of_l2_consumed_first():
    left = [10, 20, 30]
    right = [1, 2, 3]
    result = list(merge_sorted(left, right, key=lambda x: x))
    assert result == [1, 2, 3, 10, 20, 30]


def test_merge_sorted_non_key_all_of_l2_consumed_first():
    left = [10, 20, 30]
    right = [1, 2, 3]
    result = list(merge_sorted(left, right))
    assert result == [1, 2, 3, 10, 20, 30]


def test_merge_sorted_key_alternating():
    left = [5, 15, 25]
    right = [0, 10, 20, 30]
    result = list(merge_sorted(left, right, key=lambda x: x))
    assert result == [0, 5, 10, 15, 20, 25, 30]


# Branch where L1 has >1 sequences causing recursion into _merge_sorted_binary_key.
# With 3 or 4 sequences and key, the split can produce len(L1) > 1.
def test_merge_sorted_key_recursive_left_half():
    # 4 sequences, mid=2, L1 has 2 -> _merge_sorted_binary_key(L1) recursion
    a = [{'v': 1}]
    b = [{'v': 2}]
    c = [{'v': 3}]
    d = [{'v': 4}]
    result = list(merge_sorted(a, b, c, d, key=lambda x: x['v']))
    assert [x['v'] for x in result] == [1, 2, 3, 4]


def test_merge_sorted_key_three_seqs_recursion():
    # 3 sequences, mid=1, L2 has 2 -> key recursion on L2
    a = [{'v': 1}]
    b = [{'v': 3}]
    c = [{'v': 2}]
    result = list(merge_sorted(a, b, c, key=lambda x: x['v']))
    assert [x['v'] for x in result] == [1, 2, 3]


def test_merge_sorted_binary_key_direct_empty_second():
    # Directly exercise the StopIteration branch (line 190-192) of the helper.
    gen = _merge_sorted_binary_key(([1, 2], [], [3, 4], []), lambda x: x)
    assert list(gen) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_strings():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_three():
    assert list(interleave([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == [
        1, 4, 7, 2, 5, 8, 3, 6, 9
    ]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([])) is True


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(5, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    result = tail(2, iter([1, 2, 3, 4]))
    assert result == (3, 4)


def test_tail_zero():
    assert tail(0, [1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second('A')


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


def test_last_empty_raises():
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_single():
    assert get(1, 'ABC') == 'B'


def test_get_list_no_default():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_list_single_element():
    assert get([1], 'ABC') == ('B',)


def test_get_list_empty():
    assert get([], 'ABC') == ()


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_missing_multiple_no_default_raises():
    with pytest.raises(KeyError):
        get(['Alice', 'Dennis'], {'Alice': 1})
    with pytest.raises(IndexError):
        get([10, 11], [1, 2, 3])


def test_get_non_indexable_with_default():
    assert get('x', 5, default=42) == 42


def test_get_non_indexable_without_default():
    with pytest.raises(TypeError):
        get('x', 5)


# Exercise the (KeyError, IndexError) branch with default != no_default (line 468)
def test_get_missing_index_with_default_returns_default():
    assert get(10, [1, 2, 3], None) is None
    assert get('Dennis', {'Alice': 1}, None) is None
    # non-default sentinel default value
    sentinel = object()
    assert get(10, [1, 2, 3], sentinel) is sentinel
    assert get('x', {'a': 1}, sentinel) is sentinel


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons / interpose
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------
def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_add():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, add, [1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_reduceby_mul():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, mul, [1, 2, 3, 4, 5])
    assert result == {False: 15, True: 8}


def test_reduceby_with_constant_init():
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

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_empty():
    assert reduceby(len, add, []) == {}


def test_reduceby_keyword_init():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, add, [1, 2, 3, 4, 5], init=0)
    assert result == {False: 9, True: 6}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(5)] == [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_size_3():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == [
        (1, 2, 3), (2, 3, 4), (3, 4, 5)
    ]


def test_sliding_window_short_sequence():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_odd_no_pad():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_odd_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)
    ]


def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_with_list_len():
    result = list(partition_all(3, [1, 2, 3, 4, 5, 6, 7]))
    assert result == [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_with_iterator():
    result = list(partition_all(3, iter([1, 2, 3, 4, 5, 6, 7])))
    assert result == [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_iterator_exact_multiple():
    result = list(partition_all(3, iter([1, 2, 3, 4, 5, 6])))
    assert result == [(1, 2, 3), (4, 5, 6)]


def test_partition_all_sequence_invalid_length_raises():
    class WeirdSubject:
        def __init__(self, data, fake_len):
            self._data = data
            self._fake_len = fake_len

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return self._fake_len

    with pytest.raises(LookupError):
        list(partition_all(3, WeirdSubject([1, 2, 3, 4], 5)))


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_multiple_with_default():
    data = [[1], [4, 5, 7]]
    assert list(pluck([0, 1], data, default=None)) == [(1, None), (4, 5)]


def test_pluck_missing_no_default_raises():
    data = [{'id': 1}]
    with pytest.raises(KeyError):
        list(pluck('name', data))


# ---------------------------------------------------------------------------
# getter
# ---------------------------------------------------------------------------
def test_getter_scalar():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multi():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_list_empty():
    g = getter([])
    assert g('ABC') == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    friends = [
        ('Alice', 'Edith'),
        ('Alice', 'Zhao'),
        ('Edith', 'Alice'),
        ('Zhao', 'Alice'),
        ('Zhao', 'Edith'),
    ]
    cities = [
        ('Alice', 'NYC'),
        ('Alice', 'Chicago'),
        ('Dan', 'Sydney'),
        ('Edith', 'Paris'),
        ('Edith', 'Berlin'),
        ('Zhao', 'Shanghai'),
    ]
    result = join(second, friends, first, cities)
    expected = {
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
    }
    assert set(result) == expected


def test_join_index_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai'), ('Dan', 'Sydney')]
    result = join(1, friends, 0, cities)
    assert set(result) == {
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
    }


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda t: (str(t[0]), str(t[1]))) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)],
        key=lambda t: (str(t[0]), str(t[1])),
    )


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4)}


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert set(result) == {(2, 2), (3, 3), (1, None)}


# Exercise the left-join exit path (line 900 -> function exit) where the
# right sequence is empty (nothing iterated) so unmatched left items are emitted.
def test_join_left_join_empty_right():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [],
                       right_default='R'))
    assert result == [(1, 'R'), (2, 'R')]


# Left join where all right keys match; only leftover left keys emitted.
def test_join_left_join_all_matched():
    result = list(join(lambda x: x, [1, 2], lambda x: x, [1, 2],
                       right_default=None))
    assert set(result) == {(1, 1), (2, 2)}


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)
    ]


def test_diff_with_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_non_list_single_seq_raises():
    # N == 1 but seqs[0] is not a list -> falls through to the N < 2 check.
    with pytest.raises(TypeError) as excinfo:
        list(diff((1, 2, 3)))
    assert str(excinfo.value) == 'Too few sequences given (min 2 required)'


def test_diff_zero_seqs_raises():
    with pytest.raises(TypeError) as excinfo:
        list(diff())
    assert str(excinfo.value) == 'Too few sequences given (min 2 required)'


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice'
    )


def test_topk_with_index_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(rest) == [1, 2, 3]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_short():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_seeded():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_reproducible():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == list(
        random_sample(0.1, seq, random_state=2016)
    )


def test_random_sample_with_random_obj():
    from random import Random
    randobj = Random(2016)
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=randobj)) == [
        7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98
    ]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(10), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=1)) == list(range(10))
