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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove_basic():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_remove_lazy():
    out = remove(lambda x: x < 3, range(5))
    # it's a lazy iterator
    assert not isinstance(out, list)
    assert list(out) == [3, 4]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_key_index():
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


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_many():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [1, 2], [])) == [1, 2]


def test_merge_sorted_empty_inputs_key():
    assert list(merge_sorted([], [1, 2], [], key=lambda x: x)) == [1, 2]


def test_merge_sorted_with_key_explicit_none():
    # Exercises kwargs.get('key', default) branch thoroughly
    assert list(merge_sorted([1, 3], [2, 4], key=None)) == [1, 2, 3, 4]


def test_merge_sorted_duplicates():
    # Equal values: `val2 < val1` is False, so val1 comes first.
    assert list(merge_sorted([1, 1], [1, 1])) == [1, 1, 1, 1]


def test_merge_sorted_duplicates_key():
    assert list(merge_sorted([1, 1], [1, 1], key=lambda x: x)) == [1, 1, 1, 1]


def test_merge_sorted_key_duplicates():
    # Equals under key: key2 < key1 is False, so left's val is emitted first.
    a = [('a', 1)]
    b = [('b', 1)]
    assert list(merge_sorted(a, b, key=lambda x: x[1])) == [('a', 1), ('b', 1)]


def test_merge_sorted_many_way_ordering():
    # Forces recursion into _merge_sorted_binary with >2 sequences so the
    # recursive `seq1 = _merge_sorted_binary(L1)` branch is exercised.
    seqs = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    assert list(merge_sorted(*seqs)) == list(range(1, 13))


def test_merge_sorted_many_way_ordering_key():
    # With the key, each input seq is assumed already sorted under that key.
    # Here inputs are ascending, but key=neg makes them descending relative to
    # key ordering, so output is NOT globally sorted.  Assert exact real output.
    seqs = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    neg = lambda x: -x
    assert list(merge_sorted(*seqs, key=neg)) == [4, 8, 12, 3, 7, 11,
                                                  2, 6, 10, 1, 5, 9]


def test_merge_sorted_first_exhausted():
    # A left element larger than all remaining right elements: run right to end.
    assert list(merge_sorted([10], [1, 2, 3])) == [1, 2, 3, 10]


def test_merge_sorted_key_first_exhausted():
    # key=neg: key(10) = -10, key(1) = -1, so merge order emits 10 first,
    # then since [1,2,3] is not key-sorted (key values -1 < -2 < -3 ascending
    # in-key? no), the actual output is [10, 1, 2, 3].
    assert list(merge_sorted([10], [1, 2, 3], key=lambda x: -x)) == [10, 1, 2, 3]


def test_merge_sorted_three_uneven():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_uneven_key():
    key = lambda x: -x
    assert list(merge_sorted([4, 1], [5, 2], [6, 3], key=key)) == [6, 5, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty_outer():
    assert list(interleave([])) == []


def test_interleave_inner_empty():
    assert list(interleave([[], [1, 2]])) == [1, 2]


def test_interleave_one_short_first():
    # First iterator exhausts immediately; others continue.
    assert list(interleave([[1], [2, 3], [4, 5]])) == [1, 2, 4, 3, 5]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


# ---------------------------------------------------------------------------
# isiterable / isdistinct
# ---------------------------------------------------------------------------
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


def test_isdistinct_sequence():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_starts_at_zero():
    # Distinguishes islice(seq, 0, None, n) from islice(seq, None, None, n)
    assert list(take_nth(3, [0, 1, 2, 3, 4, 5, 6])) == [0, 3, 6]


def test_take_nth_n_equals_one():
    assert list(take_nth(1, [10, 20, 30])) == [10, 20, 30]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first('ABC') == 'A'
    assert first(iter([1, 2, 3])) == 1


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second(iter([1, 2, 3])) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_nth_tuple():
    assert nth(1, (10, 20, 30)) == 20


# ---------------------------------------------------------------------------
# last
# ---------------------------------------------------------------------------
def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


def test_last_iterator():
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_int():
    assert get(1, 'ABC') == 'B'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, None) is None


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('Dennis', {'Alice': 'x'})


def test_get_single_list_no_default():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list_no_default():
    assert get([], 'ABC') == ()


def test_get_missing_index():
    with pytest.raises(IndexError):
        get(10, [1, 2])


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons / interpose
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2, 3]]))) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------
def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2,
    }


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}


def test_reduceby_with_init_constant():
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, [1, 2, 3, 4, 5], 100)
    assert result == {False: 109, True: 106}


def test_reduceby_with_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_key_index():
    projects = [
        {'state': 'CA', 'cost': 1000000},
        {'state': 'IL', 'cost': 100000},
        {'state': 'IL', 'cost': 2000000},
        {'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    inc = lambda x: x + 1
    counter = iterate(inc, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_iterate_double():
    double = lambda x: x * 2
    gen = iterate(double, 1)
    assert [next(gen) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_short():
    assert list(sliding_window(3, [1, 2])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_size_one():
    assert list(sliding_window(1, [1, 2, 3])) == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_generator():
    assert list(partition_all(3, iter(range(7)))) == [(0, 1, 2), (3, 4, 5), (6,)]


def test_partition_all_exact():
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]


def test_partition_all_list_one_leftover():
    # List input: __len__ branch computes end = len % n.
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_list_two_leftover():
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_tuple_partial():
    assert list(partition_all(3, (1, 2, 3, 4))) == [(1, 2, 3), (4,)]


def test_partition_all_single_item():
    assert list(partition_all(2, [1])) == [(1,)]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_len():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count(iter([])) == 0


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_multiple_default():
    data = [{'a': 1}, {'a': 2, 'b': 3}]
    assert list(pluck(['a', 'b'], data, default=0)) == [(1, 0), (2, 3)]


def test_getter_scalar():
    g = getter(1)
    assert g([10, 20, 30]) == 20


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([10, 20, 30]) == (10, 20)


def test_getter_list_single():
    g = getter([1])
    assert g([10, 20, 30]) == (20,)


def test_getter_empty_list():
    g = getter([])
    assert g([10, 20, 30]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    result = list(join(operator.itemgetter(1), [('a', 1), ('b', 2)],
                       operator.itemgetter(0), [(1, 'x'), (2, 'y'), (3, 'z')]))
    assert result == [(('a', 1), (1, 'x')), (('b', 2), (2, 'y'))]


def test_join_left_default():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right, left_default=None))
    # right join: unmatched right items paired with left_default
    assert (None, 4) in result
    assert (2, 2) in result
    assert (3, 3) in result


def test_join_right_default():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right, right_default=None))
    assert (1, None) in result
    assert (2, 2) in result
    assert (3, 3) in result


def test_join_full():
    left = [1, 2, 3]
    right = [2, 3, 4]
    result = list(join(lambda x: x, left, lambda x: x, right,
                       left_default=None, right_default=None))
    # The join yields matched pairs (rightseq order), then unmatched right
    # items paired with left_default, then unmatched left items at the end.
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_key_index():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Edith', 'Paris'))]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)) == [
        ('bananas', 'Oranges')
    ]


def test_diff_three():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4])) == [(3, 3, 4)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_list_of_seqs():
    # single list-of-lists argument gets unpacked
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_topk_key_index():
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


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_available():
    first_two, rest = peekn(5, [0, 1])
    assert first_two == (0, 1)
    assert list(rest) == [0, 1]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(100), random_state=0)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=0)) == list(range(10))


def test_random_sample_random_object():
    from random import Random
    randobj = Random(2016)
    result = list(random_sample(0.1, list(range(100)), random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
