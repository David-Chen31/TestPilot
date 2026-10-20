import itertools
import operator
from functools import reduce

import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, drop, take_nth, first, second, nth, last,
    get, concat, concatv, mapcat, cons, interpose, frequencies, reduceby,
    iterate, sliding_window, partition, partition_all, count, pluck, getter,
    join, tail, diff, topk, peek, peekn, random_sample,
)
from toolz.itertoolz import no_pad


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(bool, [])) == []


def test_remove_returns_iterator():
    result = remove(lambda x: False, [1, 2])
    assert not isinstance(result, list)
    assert list(result) == [1, 2]


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------
def test_accumulate_operator_add():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_operator_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_single_element():
    assert list(accumulate(operator.add, [5])) == [5]


def test_accumulate_initial_none_is_used():
    gen = accumulate(operator.add, [1, 2], None)
    assert next(gen) is None
    with pytest.raises(TypeError):
        next(gen)


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}


def test_groupby_iseven():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_member_key():
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
    assert groupby(lambda x: x, []) == {}


def test_groupby_preserves_order():
    result = groupby(lambda x: x, [1, 1, 2, 1])
    assert result == {1: [1, 1, 1], 2: [2]}


# ---------------------------------------------------------------------------
# merge_sorted
# ---------------------------------------------------------------------------
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single_arg():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_merge_sorted_with_empty_iterable():
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_three():
    assert list(merge_sorted([1, 4], [2, 5], [3, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_four_triggers_recursion():
    # 4 sequences -> mid=2, so both halves have len 2 and recurse into
    # _merge_sorted_binary (executes line 141 for L1 and L2).
    assert list(merge_sorted([1, 5], [2, 6], [3, 7], [4, 8])) == \
        [1, 2, 3, 4, 5, 6, 7, 8]


def test_merge_sorted_five():
    assert list(merge_sorted([1, 5], [2, 6], [3, 7], [4, 8], [0, 9])) == \
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_merge_sorted_inner_break_branch():
    # Craft a case where in the inner loop all remaining seq2 values are
    # smaller than val1, exhausting seq2 -> "else: break" path (lines 164,
    # 171-172).
    #
    # Take [100] and [1, 2, 3].  val1=100, val2=1 (1<100) -> yield 1.
    # Loop over seq2: 2<100 yield 2; 3<100 yield 3; seq2 exhausted ->
    # break.  Then yield val1=100, yield from seq1 (empty).
    assert list(merge_sorted([100], [1, 2, 3])) == [1, 2, 3, 100]


def test_merge_sorted_inner_break_branch_longer_seq1():
    # seq1 = [50, 100].  val1=50, val2=1 -> yield 1; 2<50 yield 2;
    # 3<50 yield 3; 4<50 yield 4; seq2 has 5 -> 5<50 yield 5; exhausted
    # -> break.  yield 50; yield from remaining seq1 -> 100.
    assert list(merge_sorted([50, 100], [1, 2, 3, 4, 5])) == \
        [1, 2, 3, 4, 5, 50, 100]


def test_merge_sorted_key_four_triggers_recursion():
    assert list(merge_sorted([1, 5], [2, 6], [3, 7], [4, 8],
                             key=lambda x: x)) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_merge_sorted_key_empty_second():
    # key path where second half is entirely empty -> except StopIteration
    # (lines 190-192) yields from seq1.
    assert list(merge_sorted([1, 2], [], key=lambda x: x)) == [1, 2]


def test_merge_sorted_key_inner_break():
    # key variant: [100], [1, 2, 3] with identity-ish key.
    assert list(merge_sorted([100], [1, 2, 3], key=lambda x: x)) == \
        [1, 2, 3, 100]


def test_merge_sorted_key_inner_break_longer():
    assert list(merge_sorted([50, 100], [1, 2, 3, 4, 5], key=lambda x: x)) == \
        [1, 2, 3, 4, 5, 50, 100]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_subsequences():
    assert list(interleave([[], []])) == []


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_with_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_preserves_first_occurrence():
    assert list(unique([3, 1, 3, 2, 1])) == [3, 1, 2]


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------
def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(range(3)) is True
    assert isiterable(iter([1])) is True


def test_isiterable_false():
    assert isiterable(5) is False
    assert isiterable(None) is False


# ---------------------------------------------------------------------------
# isdistinct
# ---------------------------------------------------------------------------
def test_isdistinct_lists():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_strings():
    assert isdistinct('Hello') is False
    assert isdistinct('World') is True


def test_isdistinct_empty():
    assert isdistinct([]) is True
    assert isdistinct('') is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---------------------------------------------------------------------------
# take / drop / take_nth
# ---------------------------------------------------------------------------
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_n_1():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


# ---------------------------------------------------------------------------
# first / second / nth / last
# ---------------------------------------------------------------------------
def test_first():
    assert first('ABC') == 'A'


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'


def test_second_short_raises():
    with pytest.raises(StopIteration):
        second('A')


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20]) == 10
    assert nth(1, (10, 20)) == 20


def test_nth_iterator():
    assert nth(2, iter([1, 2, 3, 4])) == 3


def test_nth_out_of_range_raises():
    with pytest.raises(IndexError):
        nth(10, [1, 2])


def test_last():
    assert last('ABC') == 'C'


def test_last_list():
    assert last([1, 2, 3]) == 3


def test_last_empty_raises():
    with pytest.raises(IndexError):
        last([])


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_single_index():
    assert get(1, 'ABC') == 'B'


def test_get_list_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_with_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_single_default():
    assert get('missing', {'a': 1}, None) is None


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_index_missing_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_list_single():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_unindexable_with_default():
    assert get(0, 5, 'default') == 'default'


def test_get_unindexable_no_default_raises():
    with pytest.raises(TypeError):
        get(0, 5)


# ---------------------------------------------------------------------------
# concat / concatv / mapcat / cons
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2]]))) == [1, 2]


def test_concat_empty():
    assert list(concat([])) == []


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_no_args():
    assert list(concatv()) == []


def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


# ---------------------------------------------------------------------------
# interpose
# ---------------------------------------------------------------------------
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
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_basic():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data, 0) == {False: 9, True: 6}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add,
                      [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_string_key():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_means():
    mean = lambda s: float(sum(s)) / len(s)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_sliding_window_larger_than_seq():
    assert list(sliding_window(3, [1, 2])) == []


# ---------------------------------------------------------------------------
# partition / partition_all
# ---------------------------------------------------------------------------
def test_partition_no_pad():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


def test_partition_with_pad_zero():
    assert list(partition(3, [1, 2, 3, 4], pad=0)) == [(1, 2, 3), (4, 0, 0)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3]))) == [(1, 2), (3,)]


def test_partition_all_sized_sequence_with_len():
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_iterator_fallback_path():
    # Exercise the TypeError fallback (binary search) path (lines 740-752).
    gen = (x for x in [1, 2, 3, 4, 5])
    assert list(partition_all(2, gen)) == [(1, 2), (3, 4), (5,)]


def test_partition_all_len_hint_mismatch_raises_lookuperror():
    # A sized sequence whose ``len`` lies about the number of elements,
    # together with padding, causes line 735's condition to be True and
    # raises LookupError (lines 735-738).
    class LyingSeq:
        def __init__(self, items, fake_len):
            self._items = list(items)
            self._fake_len = fake_len

        def __len__(self):
            return self._fake_len

        def __iter__(self):
            return iter(self._items)

    # items length 3, n=2 -> zip_longest produces (1, 2), (3, no_pad).
    # len() reports 4, so end = 4 % 2 = 0.  prev[0-1] is prev[-1] which is
    # no_pad -> first clause True -> raise LookupError.
    seq = LyingSeq([1, 2, 3], 4)
    with pytest.raises(LookupError):
        list(partition_all(2, seq))


def test_partition_all_len_hint_mismatch_second_clause():
    # Another mismatch where prev[end - 1] is not no_pad but prev[end]
    # is not no_pad either -> second clause raises.
    class LyingSeq:
        def __init__(self, items, fake_len):
            self._items = list(items)
            self._fake_len = fake_len

        def __len__(self):
            return self._fake_len

        def __iter__(self):
            return iter(self._items)

    # items length 3, n=3 -> zip_longest yields (1, 2, 3) only?  No: with
    # 3 items and n=3 the single tuple is (1, 2, 3); prev[-1] is not no_pad,
    # so no LookupError.  Use items length 5, n=3 -> tuples (1,2,3),(4,5,
    # no_pad).  fake_len=8 -> end = 8 % 3 = 2.  prev[1] = 5 (not no_pad);
    # prev[2] = no_pad, so `prev[end] is not no_pad` is False. First clause
    # False, second clause False -> no raise.  Need prev[end] not no_pad:
    # items length 4, n=3 -> (1,2,3),(4,no_pad,no_pad).  end with fake_len=5
    # -> 5 % 3 = 2. prev[1]=no_pad -> first clause True -> raises.
    # To reach second-clause True: choose fake_len s.t. prev[end-1] not
    # no_pad and prev[end] not no_pad.  With (4, no_pad, no_pad) as prev,
    # end=4 -> prev[3] IndexError.  end=1 -> prev[0]=4 not no_pad,
    # prev[1]=no_pad not "is not no_pad" -> False -> no raise.  Hard to
    # trigger the second clause in isolation; the first test suffices.
    seq = LyingSeq([1, 2, 3, 4], 5)
    with pytest.raises(LookupError):
        list(partition_all(3, seq))


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_generator():
    assert count(i for i in range(5)) == 5


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


def test_count_string():
    assert count('abc') == 3


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, default=None)) == [1, None]


def test_pluck_multiple_with_default():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, default=None)) == [(1, 2), (3, None)]


def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list():
    g = getter([0, 2])
    assert g([10, 20, 30]) == (10, 30)


def test_getter_single_element_list():
    g = getter([1])
    assert g([10, 20]) == (20,)


def test_getter_empty_list():
    g = getter([])
    assert g([10, 20]) == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
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
    result = list(join(identity, [1, 2, 3],
                       identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=repr) == sorted(
        [(2, 2), (3, 3), (None, 4), (1, None)], key=repr)


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [3, 4],
                       left_default=None))
    assert result == [(None, 3), (None, 4)]


def test_join_right_join_with_match():
    # Exercises the ``if key in d`` true branch of the right-join path
    # (lines 895-897).
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 3],
                       left_default=None))
    assert result == [(1, 1), (None, 3)]


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3],
                       right_default=None))
    assert sorted(result) == [(1, None), (2, 2)]


def test_join_inner_no_matches():
    assert list(join(lambda x: x, [1, 2], lambda x: x, [3, 4])) == []


def test_join_with_string_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai')]
    result = list(join(1, friends, 0, cities))
    expected = [
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
    ]
    assert result == expected


def test_join_right_default_only_no_unmatched_left():
    # Right default supplied, left default omitted.  When every left key
    # is matched on the right, the trailing loop over ``d.items()`` yields
    # nothing (covers the ``elif right_default != no_default`` entry and
    # falling through without yielding from the trailing loop).
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 2, 3],
                       right_default='R'))
    assert result == [(1, 1), (2, 2)]


def test_join_right_default_with_extra_right_items():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 2, 2],
                       right_default=None))
    assert result == [(1, 1), (2, 2), (2, 2)]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_three():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 5, 3], [1, 2, 4])) == [
        (2, 2, 5, 2), (3, 3, 3, 4)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2]))


def test_diff_too_few_single_nonlist_raises():
    # Single non-list arg -> N<2 -> TypeError (lines 950-951).
    with pytest.raises(TypeError, match='Too few sequences given'):
        list(diff(iter([[1, 2], [1, 3]])))


def test_diff_no_args_raises():
    with pytest.raises(TypeError, match='Too few sequences given'):
        list(diff())


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10, 100]])) == [(3, 10)]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_with_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_topk_with_index_key():
    data = [('a', 1), ('b', 3), ('c', 2)]
    assert topk(2, data, key=1) == (('b', 3), ('c', 2))


def test_topk_empty():
    assert topk(3, []) == ()


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_empty_raises():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    first_n, rest = peekn(10, [0, 1, 2])
    assert first_n == (0, 1, 2)
    assert list(rest) == [0, 1, 2]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_with_seed():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert result1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_with_random_object():
    from random import Random
    randobj = Random(2016)
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_length():
    assert tail(10, [1, 2]) == [1, 2]


def test_tail_empty():
    assert tail(2, []) == []
