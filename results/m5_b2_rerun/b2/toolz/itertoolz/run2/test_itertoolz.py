import itertools
import operator
import collections
import pytest

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
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(iseven, [])) == []
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]
    assert list(remove(lambda x: True, [1, 2, 3])) == []


# ------------------------------------------------------------- accumulate
def test_accumulate_default():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_is_lazy_generator():
    gen = accumulate(operator.add, [1, 2, 3])
    assert iter(gen) is gen


# ---------------------------------------------------------------- groupby
def test_groupby_callable():
    iseven = lambda x: x % 2 == 0
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert list(result.keys()) == ['F', 'M']
    assert result['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert result['M'] == [{'name': 'Bob', 'gender': 'M'},
                           {'name': 'Charlie', 'gender': 'M'}]


def test_groupby_empty():
    assert groupby(len, []) == {}


# ------------------------------------------------------------ merge_sorted
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_many_seqs():
    result = list(merge_sorted([1, 4], [2, 5], [3, 6], [0, 7]))
    assert result == [0, 1, 2, 3, 4, 5, 6, 7]


def test_merge_sorted_equal_elements():
    # Strict `<` comparison: equal values from left come before right.
    assert list(merge_sorted([1, 1, 3], [1, 2, 2, 3])) == [1, 1, 1, 2, 2, 3, 3]


def test_merge_sorted_no_duplicates_strict():
    assert list(merge_sorted([2, 4, 6], [1, 3, 5])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_seqs_exhaust_first():
    # right sequence exhausts first -> "else" branch yields remaining seq1
    assert list(merge_sorted([10, 20, 30], [1, 2])) == [1, 2, 10, 20, 30]


def test_merge_sorted_right_longer():
    # left exhausts before right -> remaining seq2 appended
    assert list(merge_sorted([1, 2], [3, 4, 5])) == [1, 2, 3, 4, 5]


def test_merge_sorted_single_empty_seqs():
    assert list(merge_sorted([], [])) == []
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 2], [])) == [1, 2]


def test_merge_sorted_with_key_equal_keys():
    # equal keys: left comes first due to strict <
    data = [(0, 'a'), (0, 'b')]
    result = list(merge_sorted([data[0]], [data[1]], key=lambda x: x[0]))
    assert result == [(0, 'a'), (0, 'b')]


def test_merge_sorted_with_key_single_seq():
    # one seq given with a key in kwargs (returns iterator over seq as-is)
    assert list(merge_sorted([3, 1, 2], key=lambda x: -x)) == [3, 1, 2]


def test_merge_sorted_key_with_empty_and_multi():
    assert list(merge_sorted([], [1, 2], [3], key=lambda x: x)) == [1, 2, 3]


def test_merge_sorted_with_key_multiple_values():
    # ensure key applied at every comparison across recursion
    a = [10, 30]
    b = [20, 40]
    c = [15, 35]
    assert list(merge_sorted(a, b, c, key=lambda x: x)) == [10, 15, 20, 30, 35, 40]


def test_merge_sorted_kwargs_no_key():
    # kwargs with extra but no key should behave identically to no kwargs
    assert list(merge_sorted([1, 3], [2, 4], foo=1)) == [1, 2, 3, 4]


# ------------------------------------------------------------- interleave
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_one_seq():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_first_shorter():
    assert list(interleave([[1], [2, 3, 4]])) == [1, 2, 3, 4]


def test_interleave_middle_shorter():
    assert list(interleave([[1, 2, 3], [10], [20, 21]])) == [1, 10, 20, 2, 21, 3]


def test_interleave_last_shorter():
    assert list(interleave([[1, 2, 3], [10, 11]])) == [1, 10, 2, 11, 3]


# ----------------------------------------------------------------- unique
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert list(unique([])) == []


def test_unique_keeps_first():
    assert list(unique(['aa', 'b', 'cc', 'd'], key=len)) == ['aa', 'b']


# ------------------------------------------------------------- isiterable
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1, 2])) is True


# ------------------------------------------------------------- isdistinct
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True
    assert isdistinct(iter([])) is True


# ------------------------------------------------------------------- take
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(5, [1, 2])) == [1, 2]


# ------------------------------------------------------------------- tail
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(1, (1, 2, 3)) == (3,)


def test_tail_lazy():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


# ------------------------------------------------------------------- drop
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(10, [1, 2])) == []
    assert list(drop(0, [1, 2])) == [1, 2]


# --------------------------------------------------------------- take_nth
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


def test_take_nth_starts_at_zero():
    # explicit: starts at index 0, not None
    assert list(take_nth(4, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) == [0, 4, 8]


def test_take_nth_n_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ------------------------------------------------------------------ first
def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1
    with pytest.raises(StopIteration):
        first([])


# ----------------------------------------------------------------- second
def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second([1])


# -------------------------------------------------------------------- nth
def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30])) == 30


def test_nth_out_of_range():
    with pytest.raises(IndexError):
        nth(5, [1, 2])
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2]))


# ------------------------------------------------------------------- last
def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# -------------------------------------------------------------------- get
def test_get_index():
    assert get(1, 'ABC') == 'B'
    assert get(0, [10, 20]) == 10


def test_get_list_index():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, None) is None


def test_get_missing_no_default():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 1})
    with pytest.raises(IndexError):
        get(5, [1, 2])


def test_get_typeerror_non_list_no_default():
    with pytest.raises(TypeError):
        get(object(), [1, 2])


def test_get_typeerror_non_list_with_default():
    assert get(object(), [1, 2], 'default') == 'default'


def test_get_list_index_with_default():
    # list + default: uses _get per index
    assert get([0, 5], [10, 20, 30], 'x') == (10, 'x')


def test_get_empty_list_with_default():
    assert get([], [1, 2, 3], 'x') == ()


# ----------------------------------------------------------------- concat
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []
    assert list(concat([iter([1, 2]), iter([3])])) == [1, 2, 3]


# ---------------------------------------------------------------- concatv
def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']
    assert list(concatv()) == []


# ----------------------------------------------------------------- mapcat
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
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------- reduceby
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, [1, 2, 3, 4, 5]) == {False: 15, True: 8}


def test_reduceby_with_init_constant():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, [1, 2, 3, 4, 5], 0)
    assert result == {False: 9, True: 6}


def test_reduceby_with_init_callable():
    iseven = lambda x: x % 2 == 0

    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {False: {1, 3}, True: {2, 4}}


def test_reduceby_key_not_callable():
    projects = [{'state': 'CA', 'cost': 1000000},
                {'state': 'IL', 'cost': 100000},
                {'state': 'IL', 'cost': 2000000},
                {'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


def test_reduceby_no_init_single():
    assert reduceby(lambda x: 0, operator.add, [7]) == {0: 7}


# ---------------------------------------------------------------- iterate
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------- sliding_window
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_size_1():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# --------------------------------------------------------------- partition
def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_with_custom_pad():
    # Exercises the fillvalue=pad path with a non-None pad.
    assert list(partition(3, [1, 2, 3, 4, 5], pad='X')) == \
        [(1, 2, 3), (4, 5, 'X')]


def test_partition_with_pad_not_none():
    assert list(partition(2, [1, 2, 3], pad=0)) == [(1, 2), (3, 0)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ------------------------------------------------------------ partition_all
def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_shorter_than_n():
    assert list(partition_all(5, [1, 2, 3])) == [(1, 2, 3)]


def test_partition_all_exact():
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]


def test_partition_all_iterator_remainder():
    # exercises the binary-search (TypeError / no __len__) branch
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]


def test_partition_all_iterator_remainder_5_of_4():
    assert list(partition_all(4, iter(range(6)))) == [(0, 1, 2, 3), (4, 5)]


def test_partition_all_iterator_remainder_range():
    # range has __len__, so uses the (len % n) path.
    assert list(partition_all(4, range(6))) == [(0, 1, 2, 3), (4, 5)]


def test_partition_all_list_remainder():
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


# ------------------------------------------------------------------ count
def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0
    assert count("hello") == 5


def test_count_generator():
    # generator has no __len__, must use sum(1 for ...)
    assert count(x for x in range(7)) == 7


# ------------------------------------------------------------------ pluck
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_list():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'name': 'Pies'}]
    assert list(pluck('name', data, 'unknown')) == ['unknown', 'Pies']


def test_pluck_index():
    assert list(pluck(0, [[1, 2], [3, 4]])) == [1, 3]


def test_pluck_list_with_default():
    data = [[1, 2], [4, 5]]
    assert list(pluck([0, 5], data, 'x')) == [(1, 'x'), (4, 'x')]


# ----------------------------------------------------------------- getter
def test_getter_single():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list_single():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_list_empty():
    g = getter([])
    assert g([1, 2, 3]) == ()


# ------------------------------------------------------------------- join
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = list(join(second, friends, first, cities))
    # just check a couple of pairs
    assert (('Alice', 'Edith'), ('Edith', 'Paris')) in result
    assert (('Alice', 'Zhao'), ('Zhao', 'Shanghai')) in result


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert set(result) == {(2, 2), (3, 3), (None, 4), (1, None)}


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


def test_join_non_callable_key():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    # friends whose second element matches a city's first element:
    #   ('Alice', 'Edith') matches ('Edith', 'Paris')
    #   ('Zhao', 'Alice')  matches ('Alice', 'NYC')
    assert set(result) == {
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
    }


def test_join_inner_exact():
    identity = lambda x: x
    assert list(join(identity, [1, 2, 3], identity, [2, 3, 4])) == \
        [(2, 2), (3, 3)]


def test_join_left_join_exact_order():
    identity = lambda x: x
    # Right items yield first, then unmatched left keys at end.
    assert list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                     right_default=None)) == [(2, 2), (3, 3), (1, None)]


def test_join_with_dicts():
    left = [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
    right = [{'id': 2, 'w': 'x'}, {'id': 3, 'w': 'y'}]
    result = list(join('id', left, 'id', right))
    assert result == [({'id': 2, 'v': 'b'}, {'id': 2, 'w': 'x'})]


# ------------------------------------------------------------------- diff
def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 4]])) == [(3, 4)]


def test_diff_too_few_seqs():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3]))
    assert result == [(2, 2, 5)]


# ------------------------------------------------------------------- topk
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'x': 1}, {'x': 5}, {'x': 3}]
    assert topk(1, data, key='x') == ({'x': 5},)


def test_topk_k_larger():
    assert topk(5, [3, 1, 2]) == (3, 2, 1)


# ------------------------------------------------------------------- peek
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(rest) == [1, 2, 3]


# ------------------------------------------------------------------ peekn
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    peeked, rest = peekn(2, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, rest = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(rest) == [1, 2, 3]


# ----------------------------------------------------------- random_sample
def test_random_sample_deterministic():
    seq = list(range(100))
    assert list(random_sample(0.1, seq, random_state=2016)) == \
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    assert list(random_sample(0.1, seq, random_state=2016)) == \
        [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=1)) == []


def test_random_sample_prob_one():
    seq = list(range(10))
    assert list(random_sample(1.0, seq, random_state=1)) == seq


def test_random_sample_custom_state():
    class ConstRandom:
        def random(self):
            return 0.0

    seq = [1, 2, 3]
    assert list(random_sample(0.5, seq, random_state=ConstRandom())) == seq
