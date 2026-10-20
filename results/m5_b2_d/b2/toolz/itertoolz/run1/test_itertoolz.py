import itertools
import random
from operator import add, mul, itemgetter

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


# ----- remove -----

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_all():
    assert list(remove(lambda x: False, [1, 2, 3])) == [1, 2, 3]


def test_remove_is_lazy():
    result = remove(lambda x: x < 2, iter([1, 2, 3]))
    assert list(result) == [2, 3]


# ----- accumulate -----

def test_accumulate_add():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_no_initial():
    assert list(accumulate(add, [])) == []


def test_accumulate_empty_with_initial():
    assert list(accumulate(add, [], 1)) == [1]


def test_accumulate_single_element():
    assert list(accumulate(add, [5])) == [5]


def test_accumulate_lazy():
    result = accumulate(add, iter([1, 2, 3]))
    assert next(result) == 1
    assert next(result) == 3
    assert next(result) == 6


# ----- groupby -----

def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_non_callable_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {3: ['Bob', 'Dan'],
                                   5: ['Alice', 'Edith', 'Frank'],
                                   7: ['Charlie']}


def test_groupby_indexing_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [{'name': 'Alice', 'gender': 'F'}],
                      'M': [{'name': 'Bob', 'gender': 'M'},
                            {'name': 'Charlie', 'gender': 'M'}]}


def test_groupby_empty():
    assert groupby(len, []) == {}


def test_groupby_preserves_order():
    result = groupby(lambda x: x, [3, 1, 3, 2, 1])
    assert result == {3: [3, 3], 1: [1, 1], 2: [2]}


# ----- merge_sorted -----

def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_empty_args():
    assert list(merge_sorted()) == []


def test_merge_sorted_single_arg():
    assert list(merge_sorted([3, 1, 2])) == [3, 1, 2]


def test_merge_sorted_with_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]


def test_merge_sorted_one_empty():
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]


def test_merge_sorted_many():
    assert list(merge_sorted([1], [2], [3], [4], [5])) == [1, 2, 3, 4, 5]


def test_merge_sorted_with_key_empty():
    assert list(merge_sorted([], [1, 2, 3], key=lambda x: x)) == [1, 2, 3]


def test_merge_sorted_with_key_many():
    result = list(merge_sorted([10, 20], [11, 21], [12, 22],
                               key=lambda x: x % 10))
    assert result == [10, 20, 11, 21, 12, 22]


def test_merge_sorted_explicit_none_key():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6], key=None)) == \
        [1, 2, 3, 4, 5, 6]


def test_merge_sorted_no_kwargs_uses_plain_compare():
    assert list(merge_sorted([2, 3], [1, 4])) == [1, 2, 3, 4]


def test_merge_sorted_ties_prefer_first():
    assert list(merge_sorted([1, 2], [1, 2])) == [1, 1, 2, 2]
    assert list(merge_sorted([2, 3], [2, 3])) == [2, 2, 3, 3]


def test_merge_sorted_key_ties_prefer_first():
    a = [('a', 1), ('b', 2)]
    b = [('c', 1), ('d', 2)]
    result = list(merge_sorted(a, b, key=lambda t: t[1]))
    assert result == [('a', 1), ('c', 1), ('b', 2), ('d', 2)]


def test_merge_sorted_key_three_way():
    result = list(merge_sorted([3, 6], [1, 4], [2, 5], [0, 7],
                               key=lambda x: x))
    assert result == [0, 1, 2, 3, 4, 5, 6, 7]


def test_merge_sorted_key_three_way_ties():
    result = list(merge_sorted([1, 4], [2, 4], [3, 4], key=lambda x: x))
    assert result == [1, 2, 3, 4, 4, 4]


def test_merge_sorted_key_branch_break_behavior():
    result = list(merge_sorted([1, 5, 9], [2, 3, 4], key=lambda x: x))
    assert result == [1, 2, 3, 4, 5, 9]


def test_merge_sorted_key_branch_break_preserves_tail():
    result = list(merge_sorted([2, 4, 6, 8], [1, 3, 5, 7],
                               key=lambda x: x))
    assert result == [1, 2, 3, 4, 5, 6, 7, 8]


class _Cmp:
    """Comparable object with distinct identity but rich ordering helpers."""

    def __init__(self, value, tag):
        self.value = value
        self.tag = tag

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __eq__(self, other):
        return isinstance(other, _Cmp) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return '_Cmp(%r, %r)' % (self.value, self.tag)


def test_merge_sorted_outer_lt_strict_boundary():
    # If the outer `<` comparison (line 155) were `<=`, when val2 and val1
    # are equal (but distinguishable) the val2 element would be emitted
    # first.  With strict `<`, val1 must come first.
    a = _Cmp(5, 'a')
    b = _Cmp(5, 'b')
    result = list(merge_sorted([a], [b]))
    assert [o.tag for o in result] == ['a', 'b']


def test_merge_sorted_inner_lt_strict_boundary():
    # Exercises the inner `<` comparison (line 158).  val1==a (value 3),
    # seq2 = [b1 (value 1), b2 (value 3)]: b1<val1 yields, then b2 equals
    # val1.  With strict `<`, b2 falls into the else branch (yield val1,
    # break), so result is [b1, a, b2].  With `<=` the result is
    # [b1, b2, a].
    a = _Cmp(3, 'a')
    b1 = _Cmp(1, 'b1')
    b2 = _Cmp(3, 'b2')
    result = list(merge_sorted([a], [b1, b2]))
    assert [o.tag for o in result] == ['b1', 'a', 'b2']


def test_merge_sorted_inner_else_break_keeps_remaining_seq1():
    # seq2 is exhausted while seq1 still has items; the inner `for...else`
    # break (line 164) must not `return` — the current val1 and the rest of
    # seq1 must still be emitted.
    assert list(merge_sorted([3, 4], [1, 2])) == [1, 2, 3, 4]
    assert list(merge_sorted([5, 9], [1, 2])) == [1, 2, 5, 9]


# ----- interleave -----

def test_interleave_basic():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_single():
    assert list(interleave([[1, 2, 3]])) == [1, 2, 3]


# ----- unique -----

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_lazy():
    result = unique(iter([1, 1, 2, 2, 3]))
    assert next(result) == 1
    assert next(result) == 2
    assert next(result) == 3


# ----- isiterable -----

def test_isiterable_true():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable((1,)) is True
    assert isiterable(iter([1])) is True


def test_isiterable_false():
    assert isiterable(5) is False
    assert isiterable(None) is False


# ----- isdistinct -----

def test_isdistinct_true():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct("World") is True


def test_isdistinct_false():
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_empty():
    assert isdistinct([]) is True
    assert isdistinct(iter([])) is True


# ----- take -----

def test_take_basic():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_length():
    assert list(take(10, [1, 2, 3])) == [1, 2, 3]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


# ----- tail -----

def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_more_than_length():
    assert tail(10, [1, 2, 3]) == [1, 2, 3]


def test_tail_string():
    assert tail(2, "hello") == "lo"


# ----- drop -----

def test_drop_basic():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_drop_more_than_length():
    assert list(drop(10, [1, 2, 3])) == []


# ----- take_nth -----

def test_take_nth_basic():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_one():
    assert list(take_nth(1, [1, 2, 3])) == [1, 2, 3]


def test_take_nth_empty():
    assert list(take_nth(2, [])) == []


def test_take_nth_starts_at_zero():
    assert list(take_nth(3, [0, 1, 2, 3, 4, 5, 6])) == [0, 3, 6]


# ----- first / second / nth / last -----

def test_first():
    assert first('ABC') == 'A'
    assert first([1, 2, 3]) == 1


def test_first_empty():
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    assert second([1, 2, 3]) == 2


def test_second_short():
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(2, [10, 20, 30]) == 30
    assert nth(0, (1, 2, 3)) == 1


def test_nth_iterator():
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_nth_out_of_range():
    with pytest.raises(StopIteration):
        nth(5, iter([1, 2, 3]))
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3


# ----- get -----

def test_get_index():
    assert get(1, 'ABC') == 'B'


def test_get_multiple():
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get('Dennis', phonebook, None) is None


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_missing_index_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_single_list_no_default():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_non_indexable_with_default():
    assert get(0, 5, 'default') == 'default'


def test_get_non_indexable_no_default_raises():
    with pytest.raises(TypeError):
        get(0, 5)


def test_get_list_with_default_uses_provided_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis', 'Bob'], phonebook, 'MISSING') == \
        ('555-1234', 'MISSING', 'MISSING')


# ----- concat / concatv -----

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concat_generator():
    assert list(concat(iter([[1], [2]]))) == [1, 2]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_concatv_empty():
    assert list(concatv()) == []


# ----- mapcat -----

def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_mapcat_empty():
    assert list(mapcat(lambda s: s, [])) == []


# ----- cons -----

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_cons_empty():
    assert list(cons(1, [])) == [1]


# ----- interpose -----

def test_interpose_basic():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_single():
    assert list(interpose("a", [1])) == [1]


# ----- frequencies -----

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ----- reduceby -----

def test_reduceby_no_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}


def test_reduceby_with_init_value():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data, 0) == {False: 9, True: 6}


def test_reduceby_with_init_callable():
    def set_add(s, i):
        s.add(i)
        return s

    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3],
                      set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_string_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state',
                      lambda acc, x: acc + x['cost'],
                      projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, add, []) == {}


# ----- iterate -----

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ----- sliding_window -----

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == \
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_too_short():
    assert list(sliding_window(5, [1, 2, 3])) == []


def test_sliding_window_empty():
    assert list(sliding_window(2, [])) == []


# ----- partition -----

def test_partition_even():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


def test_partition_pad_non_none():
    assert list(partition(3, [1, 2, 3, 4], pad='X')) == \
        [(1, 2, 3), (4, 'X', 'X')]


def test_partition_pad_zero_marker():
    assert list(partition(2, [1, 2, 3], pad=0)) == [(1, 2), (3, 0)]


# ----- partition_all -----

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_single():
    assert list(partition_all(3, [1])) == [(1,)]


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_sequence_len_branch():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]


def test_partition_all_sequence_len_branch2():
    assert list(partition_all(4, [1, 2, 3, 4, 5])) == [(1, 2, 3, 4), (5,)]


def test_partition_all_sequence_multiple_remainder():
    # Final partial tuple has > 1 real element and trailing padding.
    # Exercises the `prev[-1] is no_pad` detection on the __len__ path.
    assert list(partition_all(4, [1, 2, 3, 4, 5, 6])) == \
        [(1, 2, 3, 4), (5, 6)]


def test_partition_all_iterator_binary_search_remainder_two():
    # No __len__ (iterator), so the binary-search fallback finds the first
    # padding slot for a partial tuple of more than one element.
    assert list(partition_all(3, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2, 3), (4, 5)]


def test_partition_all_iterator_binary_search_remainder_one():
    assert list(partition_all(3, iter([1, 2, 3, 4]))) == [(1, 2, 3), (4,)]


class _LyingLen:
    """Sequence whose reported length disagrees with what it yields."""

    def __init__(self, items, reported):
        self._items = items
        self._reported = reported

    def __len__(self):
        return self._reported

    def __iter__(self):
        return iter(self._items)


def test_partition_all_invalid_length_raises_lookup_error():
    # __len__ reports 7 but iteration yields 6 items for n=4; the modulo
    # computation lands on padding where a real element was expected, so
    # partition_all must raise a LookupError with the exact message.
    seq = _LyingLen([1, 2, 3, 4, 5, 6], 7)
    with pytest.raises(LookupError) as excinfo:
        list(partition_all(4, seq))
    assert str(excinfo.value) == \
        'The sequence passed to `parition_all` has invalid length'


# ----- count -----

def test_count_with_len():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0
    assert count(iter([])) == 0


def test_count_generator():
    assert count(x for x in range(5)) == 5


def test_count_prefers_len_dunder():
    # count must use __len__ when present, even if __iter__ yields a
    # different number of items.
    class LenDiffersFromIter:
        def __len__(self):
            return 10

        def __iter__(self):
            return iter([1, 2, 3])

    assert count(LenDiffersFromIter()) == 10


# ----- getter -----

def test_getter_single():
    g = getter(1)
    assert g('ABC') == 'B'


def test_getter_list_multi():
    g = getter([1, 2])
    assert g('ABC') == ('B', 'C')


def test_getter_list_single():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_empty_list():
    g = getter([])
    assert g('ABC') == ()


# ----- pluck -----

def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default=None)) == [None, 'Pies']


def test_pluck_list_with_default():
    data = [{'id': 1}, {'name': 'Pies'}]
    result = list(pluck(['id', 'name'], data, default=None))
    assert result == [(1, None), (None, 'Pies')]


def test_pluck_list_with_default_non_none():
    # The provided default (not None) must be forwarded for every missing
    # element across the list of indices.
    data = [{'id': 1}, {'name': 'Pies'}]
    result = list(pluck(['id', 'name'], data, default='MISSING'))
    assert result == [(1, 'MISSING'), ('MISSING', 'Pies')]


def test_pluck_empty():
    assert list(pluck(0, [])) == []


def test_pluck_default_non_none():
    data = [{'id': 1}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data, default='MISSING')) == \
        ['MISSING', 'Pies']


# ----- join -----

def test_join_inner():
    friends = [('Alice', 'Edith'),
               ('Alice', 'Zhao'),
               ('Edith', 'Alice'),
               ('Zhao', 'Alice'),
               ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'),
              ('Alice', 'Chicago'),
              ('Dan', 'Sydney'),
              ('Edith', 'Paris'),
              ('Edith', 'Berlin'),
              ('Zhao', 'Shanghai')]
    result = sorted(unique(join(second, friends, first, cities)))
    expected = sorted([
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Alice', 'Edith'), ('Edith', 'Berlin')),
        (('Alice', 'Zhao'), ('Zhao', 'Shanghai')),
        (('Edith', 'Alice'), ('Alice', 'NYC')),
        (('Edith', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
        (('Zhao', 'Alice'), ('Alice', 'Chicago')),
        (('Zhao', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Edith'), ('Edith', 'Berlin')),
    ])
    assert result == expected


def test_join_inner_simple():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4]))
    assert result == [(2, 2), (3, 3)]


def test_join_right_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert result == [(2, 2), (3, 3), (1, None)]


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert result == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_join_string_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC')]
    result = list(join(0, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Alice', 'NYC')),
                      (('Alice', 'Zhao'), ('Alice', 'NYC'))]


def test_join_left_outer_multiple_unmatched():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 9], identity, [1, 2],
                       right_default='R'))
    assert result == [(1, 1), (2, 2), (9, 'R')]


# ----- diff -----

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_default_non_none_used_as_fillvalue():
    # `default` must be forwarded as zip_longest's fillvalue; the padding
    # sentinel is the provided default (0 here), not None.
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=0)) == \
        [(3, 10), (0, 100)]
    assert list(diff([1, 2], [1, 2, 5], default='X')) == [('X', 5)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_key_with_default_non_none():
    assert list(diff(['a', 'b'], ['A', 'B', 'C'], key=str.lower,
                     default='?')) == [('?', 'C')]


def test_diff_list_of_sequences():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError) as excinfo:
        list(diff())
    assert str(excinfo.value) == 'Too few sequences given (min 2 required)'


def test_diff_one_element_list_raises_too_few():
    with pytest.raises(TypeError) as excinfo:
        list(diff([1]))
    assert str(excinfo.value) == 'Too few sequences given (min 2 required)'


def test_diff_single_list_of_non_iterables_raises():
    # diff([a, b, c]) expands to three sequences a, b, c; ints are not
    # iterable, so zip raises a TypeError.
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


def test_diff_three_seqs():
    assert list(diff([1, 2, 3], [1, 2, 3], [1, 2, 5])) == [(3, 3, 5)]


# ----- topk -----

def test_topk_basic():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_getter_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_empty():
    assert topk(2, []) == ()


def test_topk_more_than_length():
    assert topk(10, [1, 2, 3]) == (3, 2, 1)


# ----- peek / peekn -----

def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, new_seq = peek(seq)
    assert item == 0
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, new_seq = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(new_seq) == [1, 2, 3]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, new_seq = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq) == [0, 1, 2, 3, 4]


def test_peekn_more_than_length():
    peeked, new_seq = peekn(10, [0, 1, 2])
    assert peeked == (0, 1, 2)
    assert list(new_seq) == [0, 1, 2]


def test_peekn_zero():
    peeked, new_seq = peekn(0, [1, 2, 3])
    assert peeked == ()
    assert list(new_seq) == [1, 2, 3]


# ----- random_sample -----

def test_random_sample_deterministic():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert result1 == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    assert list(random_sample(1.0, [1, 2, 3], random_state=0)) == [1, 2, 3]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, [1, 2, 3], random_state=0)) == []


def test_random_sample_empty():
    assert list(random_sample(0.5, [], random_state=0)) == []


class _FixedRandom:
    """Deterministic random_state whose ``random`` returns a fixed value."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_random_sample_strict_less_than_prob():
    # The comparison is strict (`random() < prob`).  When random() equals
    # prob exactly, the element must be EXCLUDED.  A `<=` mutant would
    # include it.
    assert list(random_sample(0.5, [1, 2, 3],
                              random_state=_FixedRandom(0.5))) == []
    # Just below the threshold -> included.
    assert list(random_sample(0.5, [1, 2, 3],
                              random_state=_FixedRandom(0.4999))) == \
        [1, 2, 3]
    # Just above the threshold -> excluded.
    assert list(random_sample(0.5, [1, 2, 3],
                              random_state=_FixedRandom(0.5001))) == []
