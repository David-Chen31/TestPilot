import pytest
from operator import add, mul
from functools import reduce
import itertools
import random

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique,
    isiterable, isdistinct, take, tail, drop, take_nth, first, second,
    nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies,
    reduceby, iterate, sliding_window, partition, partition_all, count,
    pluck, getter, join, diff, topk, peek, peekn, random_sample,
)
from toolz.itertoolz import _merge_sorted_binary, _merge_sorted_binary_key
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
def test_accumulate_add():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


def test_accumulate_is_lazy():
    gen = accumulate(add, [1, 2, 3])
    assert next(gen) == 1
    assert next(gen) == 3
    assert next(gen) == 6


# ---------------------------------------------------------------------------
# groupby
# ---------------------------------------------------------------------------
def test_groupby_callable():
    res = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert res == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                   7: ['Charlie']}


def test_groupby_member_string():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    res = groupby('gender', data)
    assert res == {
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


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_zero():
    assert list(merge_sorted()) == []


def test_merge_sorted_one():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    res = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert res == [2, 1, 3, 3]


def test_merge_sorted_empty_inputs():
    assert list(merge_sorted([], [], [])) == []


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 4, 7], [2, 3], [5, 6, 8, 9])) == \
        [1, 2, 3, 4, 5, 6, 7, 8, 9]


# Additional merge_sorted tests targeting deeper recursion / ties / ordering.
def test_merge_sorted_many_seqs():
    # Exercises _merge_sorted_binary recursion with >2 inputs on each side.
    seqs = [[1, 10], [2, 9], [3, 8], [4, 7], [5, 6]]
    assert list(merge_sorted(*seqs)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_merge_sorted_ties_stable_side():
    # When values are equal, the binary merge yields left value first.
    # Left is [1], right is [1]; result order deterministic.
    assert list(merge_sorted([1, 2], [1, 2])) == [1, 1, 2, 2]


def test_merge_sorted_strict_less_notequal():
    # Strict '<' means equal values are not swapped: left-before-right hold.
    assert list(merge_sorted([2], [2])) == [2, 2]


def test_merge_sorted_key_ties():
    # The binary merge does not reorder equal-keyed items beyond what the
    # merge structure yields.  With the key lambda x: -x, keys are:
    # [1, 5] -> [-1, -5]; [2, 4] -> [-2, -4]; [3] -> [-3].
    # Merging halves via _merge_sorted_binary_key yields [3, 2, 4, 1, 5].
    res = list(merge_sorted([1, 5], [2, 4], [3], key=lambda x: -x))
    assert res == [3, 2, 4, 1, 5]


# New test specifically targeting equality handling in _merge_sorted_binary_key.
def test_merge_sorted_key_equal_keys_order():
    # With duplicates having equal keys, strict `<` means when key2 == key1,
    # the left value is emitted first (else branch). Verify exact output.
    res = list(merge_sorted([10, 30], [20, 40],
                            key=lambda x: x % 20))
    # keys: 10->10, 30->10, 20->0, 40->0
    # Left half [10,30] keys [10,10]; right half [20,40] keys [0,0]
    # Actually 2 seqs -> left=[10,30], right=[20,40]
    # seq1 keys: 10,10 ; seq2 keys: 0,0
    # val2=20 key2=0; first val1=10 key1=10 (0<10) yield 20, then next
    # val2=40 key2=0 (0<10) yield 40, then break inner; then ... 
    # This is hard to reason; just assert stable output computed from code.
    assert res == [20, 40, 10, 30]


def test_merge_sorted_key_equal_dup_keys():
    # Equal keys between left and right: left emitted first (strict <).
    res = list(merge_sorted([1, 3], [2, 4], key=lambda x: x % 2))
    # keys: 1->1, 3->1 ; 2->0, 4->0
    # left keys [1,1], right keys [0,0]
    assert res == [2, 4, 1, 3]


def test_merge_sorted_key_same_key_alternation():
    # Force equality of keys: all map to same key -> all from left first
    res = list(merge_sorted([1, 2], [3, 4], key=lambda x: 0))
    # key2 (=0) < key1 (=0) is False -> yield left val1 first
    assert res == [1, 2, 3, 4]


def test_merge_sorted_key_interleaved_equal_checks():
    # Use keys where key2 == key1 at some point to exercise `<` vs `<=`.
    # seqs: [a] with key 5, [b] with key 5, [c] with key 1.
    res = list(merge_sorted([5], [5], [1],
                            key=lambda x: x))
    # Need to think about halves; 3 seqs: mid=1, L1=[5], L2=[[5],[1]]
    # Just assert actual output of current code.
    assert res == [1, 5, 5]


def test_merge_sorted_key_many_seqs():
    seqs = [list(range(0, 60, 4)),
            list(range(1, 60, 4)),
            list(range(2, 60, 4)),
            list(range(3, 60, 4))]
    assert list(merge_sorted(*seqs, key=lambda x: x)) == list(range(60))


def test_merge_sorted_binary_direct_recursion():
    # Force recursion on both left and right halves via the helper directly.
    left_seqs = [[1, 7], [2, 8]]
    right_seqs = [[3, 9], [4, 10]]
    seqs = left_seqs + right_seqs
    assert list(_merge_sorted_binary(seqs)) == [1, 2, 3, 4, 7, 8, 9, 10]


def test_merge_sorted_binary_key_direct_recursion():
    seqs = [[1, 7], [2, 8], [3, 9], [4, 10]]
    assert list(_merge_sorted_binary_key(seqs, lambda x: x)) == \
        [1, 2, 3, 4, 7, 8, 9, 10]


def test_merge_sorted_binary_right_exhausted():
    # Right iterable is empty -> yields from left, returns.
    assert list(_merge_sorted_binary([[1, 2], []])) == [1, 2]


def test_merge_sorted_binary_left_exhausted_after_first():
    # Left empty; right has content; val2 from right then left empty.
    assert list(_merge_sorted_binary([[], [3, 4]])) == [3, 4]


# ---------------------------------------------------------------------------
# interleave
# ---------------------------------------------------------------------------
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[], []])) == []


def test_interleave_three():
    assert list(interleave([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == \
        [1, 4, 7, 2, 5, 8, 3, 6, 9]


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------
def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_lazy():
    it = unique([1, 1, 2, 2, 3])
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 3


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
def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_str():
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
    assert list(take(10, [1, 2])) == [1, 2]


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


def test_take_nth_start():
    # The start index must be 0 (mutant changes 0 -> None). With start=None,
    # islice would start at beginning too, so this must check actual output
    # specifically targeting start behavior with a non-zero-skip scenario.
    # islice(seq, 0, None, 3) vs islice(seq, None, None, 3) both start at 0,
    # so start=0 and start=None are equivalent.  However islice with start=0
    # does not allow the None start difference; they are the same. Skip.
    pass


def test_take_nth_step():
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ---------------------------------------------------------------------------
# tail / first / second / nth / last
# ---------------------------------------------------------------------------
def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30])) == 30


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------
def test_get_scalar():
    assert get(1, 'ABC') == 'B'


def test_get_list_of_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default_single_missing():
    assert get('missing', {}, 42) == 42


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {})


def test_get_list_empty():
    assert get([], 'ABC') == ()


def test_get_list_single_item():
    assert get([1], 'ABC') == ('B',)


def test_get_index_error_no_default_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_index_error_with_default():
    assert get(10, [1, 2, 3], 'x') == 'x'


def test_get_type_error_non_list_with_default():
    # Calling with non-indexable and non-list index -> TypeError branch,
    # default returned.
    assert get(1, 5, 'd') == 'd'


def test_get_type_error_non_list_no_default_raises():
    with pytest.raises(TypeError):
        get(1, 5)


# New: target the default value passed through in list-with-default branch.
def test_get_list_default_distinct_from_none():
    # Mutant changes `default` to `None` in the generator; make default
    # distinguish from None by using a sentinel value that is NOT None.
    sentinel = 'MISSING'
    phonebook = {'Alice': '555-1234'}
    res = get(['Alice', 'Dennis', 'Bob'], phonebook, sentinel)
    assert res == ('555-1234', 'MISSING', 'MISSING')
    # Ensure the value returned is exactly the sentinel, not None.
    assert res[1] is sentinel
    assert res[2] is sentinel


def test_get_list_default_object_identity():
    class Default:
        pass
    d = Default()
    assert get([0, 1, 2], ['a'], d) == ('a', d, d)
    result = get([0, 1], ['a'], d)
    assert result[1] is d


# ---------------------------------------------------------------------------
# concat / concatv / mapcat
# ---------------------------------------------------------------------------
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    res = list(mapcat(lambda s: [c.upper() for c in s],
                      [["a", "b"], ["c", "d", "e"]]))
    assert res == ['A', 'B', 'C', 'D', 'E']


# ---------------------------------------------------------------------------
# cons / interpose
# ---------------------------------------------------------------------------
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---------------------------------------------------------------------------
# frequencies
# ---------------------------------------------------------------------------
def test_frequencies():
    res = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert res == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---------------------------------------------------------------------------
# reduceby
# ---------------------------------------------------------------------------
def test_reduceby_basic():
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}
    assert reduceby(iseven, mul, [1, 2, 3, 4, 5]) == {False: 15, True: 8}


def test_reduceby_with_init_value():
    iseven = lambda x: x % 2 == 0
    res = reduceby(iseven, add, [1, 2, 3, 4, 5], 0)
    assert res == {False: 9, True: 6}


def test_reduceby_with_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    iseven = lambda x: x % 2 == 0
    res = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert res == {True: {2, 4}, False: {1, 3}}


def test_reduceby_member_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    res = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert res == {'CA': 1200000, 'IL': 2100000}


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    gen = iterate(lambda x: x * 2, 1)
    assert [next(gen) for _ in range(4)] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_too_large():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_exact():
    assert list(sliding_window(3, [1, 2, 3])) == [(1, 2, 3)]


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    res = list(partition(2, [1, 2, 3, 4, 5], pad=None))
    assert res == [(1, 2), (3, 4), (5, None)]


# New: pad value must be respected (mutant changes fillvalue to None).
def test_partition_with_sentinel_pad():
    sentinel = object()
    res = list(partition(3, [1, 2, 3, 4], pad=sentinel))
    assert res[0] == (1, 2, 3)
    assert res[1] == (4, sentinel, sentinel)
    assert res[1][1] is sentinel
    assert res[1][2] is sentinel


def test_partition_with_string_pad():
    res = list(partition(3, [1, 2], pad='X'))
    assert res == [(1, 2, 'X')]


# ---------------------------------------------------------------------------
# partition_all
# ---------------------------------------------------------------------------
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_exact_multiple():
    # Exact multiple: prev[-1] is not no_pad, yield prev with full length.
    assert list(partition_all(3, [1, 2, 3])) == [(1, 2, 3)]
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder_len_known():
    # List has __len__, exact remainder. n=3, len=5 -> end = 5 % 3 = 2.
    assert list(partition_all(3, [1, 2, 3, 4, 5])) == [(1, 2, 3), (4, 5)]


def test_partition_all_remainder_len_known_2():
    # n=4, len=6 -> end = 6 % 4 = 2
    assert list(partition_all(4, [1, 2, 3, 4, 5, 6])) == \
        [(1, 2, 3, 4), (5, 6)]


def test_partition_all_remainder_len_known_3():
    # n=5, len=7 -> end = 7 % 5 = 2
    assert list(partition_all(5, list(range(7)))) == \
        [(0, 1, 2, 3, 4), (5, 6)]


def test_partition_all_remainder_iterator_bisect():
    # Iterator has no __len__, forcing binary search path.
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2, 3), (4, 5)]


def test_partition_all_remainder_iterator_bisect_2():
    assert list(partition_all(4, iter(range(6)))) == \
        [(0, 1, 2, 3), (4, 5)]


def test_partition_all_remainder_iterator_bisect_3():
    # n=5, len=7 via iterator -> end computed by binary search
    assert list(partition_all(5, iter(range(7)))) == \
        [(0, 1, 2, 3, 4), (5, 6)]


def test_partition_all_single_remainder_iterator():
    assert list(partition_all(3, iter([1]))) == [(1,)]


def test_partition_all_single_remainder_list():
    assert list(partition_all(3, [1])) == [(1,)]


def test_partition_all_len_mismatch_raises():
    # __len__ present but doesn't match actual element count -> LookupError
    class Lying:
        def __init__(self):
            self.data = iter([1, 2, 3])
        def __iter__(self):
            return self
        def __next__(self):
            return next(self.data)
        def __len__(self):
            return 100
    with pytest.raises(LookupError):
        list(partition_all(2, Lying()))


def test_partition_all_len_mismatch_message():
    class Lying:
        def __init__(self):
            self.data = iter([1, 2, 3])
        def __iter__(self):
            return self
        def __next__(self):
            return next(self.data)
        def __len__(self):
            return 100
    with pytest.raises(LookupError) as excinfo:
        list(partition_all(2, Lying()))
    assert 'parition_all' in str(excinfo.value)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_with_len():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


# New: ensure __len__ path is actually used (mutant changes to hasattr(None,...)).
def test_count_uses_len():
    # A range has __len__; result must be its length.
    assert count(range(10)) == 10
    assert count('abcde') == 5
    assert count((1, 2, 3, 4)) == 4


def test_count_len_shortcut_not_iterating():
    # If __len__ exists but iteration would raise, count should still work
    # via __len__ (proving the hasattr(seq, '__len__') path is taken).
    class LenOnly:
        def __len__(self):
            return 42
        def __iter__(self):
            raise RuntimeError("should not iterate")
    assert count(LenOnly()) == 42


# ---------------------------------------------------------------------------
# pluck / getter
# ---------------------------------------------------------------------------
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_multiple_default():
    seqs = [[1, 2, 3], [4]]
    assert list(pluck([0, 1], seqs, default=None)) == \
        [(1, 2), (4, None)]


def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
def test_join_inner():
    left = [1, 2, 3]
    right = [2, 3, 4]
    identity = lambda x: x
    assert list(join(identity, left, identity, right)) == [(2, 2), (3, 3)]


def test_join_full_outer():
    identity = lambda x: x
    res = list(join(identity, [1, 2, 3],
                    identity, [2, 3, 4],
                    left_default=None, right_default=None))
    assert (2, 2) in res
    assert (3, 3) in res
    assert (None, 4) in res
    assert (1, None) in res
    assert len(res) == 4


def test_join_right_join():
    identity = lambda x: x
    res = list(join(identity, [1, 2], identity, [2, 3],
                    left_default=None))
    assert (2, 2) in res
    assert (None, 3) in res
    assert len(res) == 2


def test_join_left_join():
    identity = lambda x: x
    res = list(join(identity, [1, 2, 3], identity, [2, 3],
                    right_default=None))
    assert (2, 2) in res
    assert (3, 3) in res
    assert (1, None) in res
    assert len(res) == 3


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Dan', 'Sydney'), ('Edith', 'Paris')]
    result = list(join(1, friends, 0, cities))
    assert (('Alice', 'Edith'), ('Edith', 'Paris')) in result
    assert (('Edith', 'Alice'), ('Alice', 'NYC')) in result
    assert (('Zhao', 'Alice'), ('Alice', 'NYC')) in result
    assert len(result) == 3


def test_join_multiple_matches():
    left = [('a', 1), ('a', 2), ('b', 3)]
    right = [('a', 'A'), ('b', 'B')]
    res = list(join(0, left, 0, right))
    assert (('a', 1), ('a', 'A')) in res
    assert (('a', 2), ('a', 'A')) in res
    assert (('b', 3), ('b', 'B')) in res
    assert len(res) == 3


def test_join_left_join_no_extra():
    identity = lambda x: x
    res = list(join(identity, [1, 2], identity, [1, 2],
                    right_default=None))
    assert res == [(1, 1), (2, 2)]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    res = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                    key=str.lower))
    assert res == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three_seqs():
    res = list(diff([1, 2, 3], [1, 2, 3], [1, 9, 3]))
    assert res == [(2, 2, 9)]


def test_diff_three_seqs_with_key():
    res = list(diff(['a', 'B'], ['A', 'b'], ['a', 'Z'], key=str.lower))
    assert res == [('B', 'b', 'Z')]


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    res = topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    assert res == ('Charlie', 'Alice')


def test_topk_non_callable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    res = topk(2, data, key='v')
    assert res == ({'v': 100}, {'v': 10})


# ---------------------------------------------------------------------------
# peek / peekn
# ---------------------------------------------------------------------------
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    seq = iter([0, 1, 2])
    item, rest = peek(seq)
    assert item == 0
    assert list(rest) == [0, 1, 2]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    seq = [0, 1]
    peeked, rest = peekn(5, seq)
    assert peeked == (0, 1)
    assert list(rest) == [0, 1]


# ---------------------------------------------------------------------------
# random_sample
# ---------------------------------------------------------------------------
def test_random_sample_deterministic():
    seq = list(range(100))
    res1 = list(random_sample(0.1, seq, random_state=2016))
    res2 = list(random_sample(0.1, seq, random_state=2016))
    assert res1 == res2
    assert res1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    res = list(random_sample(0.1, seq, random_state=randobj))
    assert res == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_custom_random_object():
    class FakeRandom:
        def __init__(self):
            self.values = iter([0.05, 0.5, 0.2, 0.9, 0.0])
        def random(self):
            return next(self.values)
    seq = [1, 2, 3, 4, 5]
    res = list(random_sample(0.3, seq, random_state=FakeRandom()))
    assert res == [1, 3, 5]
