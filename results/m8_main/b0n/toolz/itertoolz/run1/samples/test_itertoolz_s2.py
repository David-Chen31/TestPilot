import itertools
import operator
import random
import pytest

from toolz.itertoolz import (
    remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable,
    isdistinct, take, tail, drop, take_nth, first, second, nth, last, get,
    concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate,
    sliding_window, partition, partition_all, count, pluck, getter, join,
    diff, topk, peek, peekn, random_sample,
)
from toolz.utils import no_default


# ---- remove ----
def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(bool, [])) == []


# ---- accumulate ----
def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_lazy():
    it = accumulate(operator.add, iter([1, 2, 3]))
    assert next(it) == 1
    assert next(it) == 3
    assert next(it) == 6
    with pytest.raises(StopIteration):
        next(it)


# ---- groupby ----
def test_groupby_callable_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_non_callable_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [{'name': 'Alice', 'gender': 'F'}],
                      'M': [{'name': 'Bob', 'gender': 'M'},
                            {'name': 'Charlie', 'gender': 'M'}]}


def test_groupby_empty():
    assert groupby(len, []) == {}


# ---- merge_sorted ----
def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_no_seqs():
    assert list(merge_sorted()) == []


def test_merge_sorted_one_seq():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert result == [2, 1, 3, 3]


def test_merge_sorted_uneven():
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2])) == [1, 2]
    assert list(merge_sorted([1, 5], [2, 3, 4], [0])) == [0, 1, 2, 3, 4, 5]


# ---- interleave ----
def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []
    assert list(interleave([[]])) == []


# ---- unique ----
def test_unique():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert tuple(unique([])) == ()


def test_unique_with_key():
    result = tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len))
    assert result == ('cat', 'mouse')


def test_unique_lazy():
    it = unique([1, 1, 2, 2, 3])
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 3


# ---- isiterable ----
def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(iter([1])) is True


# ---- isdistinct ----
def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


# ---- take ----
def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(0, [1, 2])) == []
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_is_lazy():
    it = take(2, iter([1, 2, 3]))
    assert list(it) == [1, 2]


# ---- tail ----
def test_tail():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, (1, 2, 3)) == (2, 3)


def test_tail_iterator():
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)


# ---- drop ----
def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


# ---- take_nth ----
def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ---- first / second / nth / last ----
def test_first():
    assert first('ABC') == 'A'
    with pytest.raises(StopIteration):
        first([])


def test_second():
    assert second('ABC') == 'B'
    with pytest.raises(StopIteration):
        second([1])


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, (1, 2, 3)) == 3
    assert nth(1, iter([1, 2, 3])) == 2


def test_last():
    assert last('ABC') == 'C'
    assert last([1, 2, 3]) == 3
    assert last(iter([1, 2, 3])) == 3


# ---- get ----
def test_get_simple():
    assert get(1, 'ABC') == 'B'
    assert get('Alice', {'Alice': '555-1234'}) == '555-1234'


def test_get_list():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([1], 'ABC') == ('B',)
    assert get([], 'ABC') == ()


def test_get_list_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_default():
    assert get(10, [1, 2], None) is None
    assert get('missing', {}, 'default') == 'default'


def test_get_raises():
    with pytest.raises(IndexError):
        get(10, [1, 2])
    with pytest.raises(KeyError):
        get('missing', {})
    with pytest.raises(TypeError):
        get([1, 2], [1, 2, 3])  # list index on list would be slice, but returns


def test_get_typeerror():
    # Sequence that doesn't support list indexing triggers TypeError branch
    class NoListIndex:
        def __getitem__(self, i):
            if isinstance(i, list):
                raise TypeError("no list")
            return i

    obj = NoListIndex()
    assert get(5, obj) == 5
    assert get([1, 2], obj) == (1, 2)


# ---- concat / concatv ----
def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([])) == []


def test_concat_generator():
    def gen():
        yield [1, 2]
        yield [3]
    assert list(concat(gen())) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


# ---- mapcat ----
def test_mapcat():
    result = list(mapcat(lambda s: [c.upper() for c in s],
                         [["a", "b"], ["c", "d", "e"]]))
    assert result == ['A', 'B', 'C', 'D', 'E']


# ---- cons ----
def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(cons(1, [])) == [1]


# ---- interpose ----
def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose("a", [])) == []
    assert list(interpose("a", [1])) == [1]


# ---- frequencies ----
def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}


# ---- reduceby ----
def test_reduceby_simple():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_constant_init():
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4]
    assert reduceby(iseven, operator.add, data, 0) == {False: 4, True: 6}


def test_reduceby_with_callable_init():
    def set_add(s, i):
        s.add(i)
        return s

    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert result == {True: {2, 4}, False: {1, 3}}


def test_reduceby_non_callable_key():
    projects = [{'state': 'CA', 'cost': 100},
                {'state': 'IL', 'cost': 200},
                {'state': 'IL', 'cost': 300}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 100, 'IL': 500}


# ---- iterate ----
def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]

    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---- sliding_window ----
def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, [1, 2, 3, 4])) == [(1, 2, 3), (2, 3, 4)]


def test_sliding_window_too_short():
    assert list(sliding_window(3, [1, 2])) == []


# ---- partition ----
def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]


# ---- partition_all ----
def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(3, iter(range(7)))) == [
        (0, 1, 2), (3, 4, 5), (6,)]


def test_partition_all_with_len():
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == [
        (1, 2, 3), (4, 5, 6), (7,)]


# ---- count ----
def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0


# ---- pluck ----
def test_pluck_single():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    result = list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]]))
    assert result == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'name': 'a'}, {'name': 'b', 'id': 2}]
    result = list(pluck('id', data, default=None))
    assert result == [None, 2]


def test_pluck_list_with_default():
    data = [[1, 2], [3]]
    result = list(pluck([0, 1], data, default=None))
    assert result == [(1, 2), (3, None)]


# ---- getter ----
def test_getter_single():
    g = getter('a')
    assert g({'a': 1}) == 1


def test_getter_list_multiple():
    g = getter([0, 1])
    assert g([10, 20, 30]) == (10, 20)


def test_getter_single_list():
    g = getter([0])
    assert g([10, 20]) == (10,)


def test_getter_empty_list():
    g = getter([])
    assert g([1, 2]) == ()


# ---- join ----
def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris'), ('Zhao', 'Berlin')]
    result = join(first, friends, first, cities)
    assert list(result) == [(('Alice', 'Edith'), ('Alice', 'NYC')),
                            (('Alice', 'Zhao'), ('Alice', 'NYC'))]


def test_join_inner_simple():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4]))
    assert result == [(2, 2), (3, 3)]


def test_join_right_outer():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4],
                       left_default=None))
    assert result == [(2, 2), (3, 3), (None, 4)]


def test_join_left_outer():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4],
                       right_default=None))
    assert sorted(result) == [(1, None), (2, 2), (3, 3)]


def test_join_full_outer():
    result = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(result, key=lambda x: (x[0] is None, x[0] or 0,
                                         x[1] is None, x[1] or 0)) == [
        (1, None), (2, 2), (3, 3), (None, 4)]


def test_join_non_callable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris')]
    result = list(join(0, friends, 0, cities))
    assert result == [(('Alice', 'Edith'), ('Alice', 'NYC')),
                      (('Alice', 'Zhao'), ('Alice', 'NYC'))]


# ---- diff ----
def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]


def test_diff_with_key():
    result = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                       key=str.lower))
    assert result == [('bananas', 'Oranges')]


def test_diff_list_of_seqs():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))
    with pytest.raises(TypeError):
        list(diff([[1, 2, 3]]))


def test_diff_three_seqs():
    assert list(diff([1, 2], [1, 3], [1, 4])) == [(2, 3, 4)]


# ---- topk ----
def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    result = topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    assert result == ('Charlie', 'Alice')


def test_topk_non_callable_key():
    result = topk(2, [{'v': 1}, {'v': 5}, {'v': 3}], key='v')
    assert result == ({'v': 5}, {'v': 3})


# ---- peek ----
def test_peek():
    seq = [0, 1, 2, 3, 4]
    item, seq2 = peek(seq)
    assert item == 0
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    item, rest = peek(iter([1, 2, 3]))
    assert item == 1
    assert list(rest) == [1, 2, 3]


def test_peek_empty():
    with pytest.raises(StopIteration):
        peek([])


# ---- peekn ----
def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, seq2 = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq2) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(10, [1, 2, 3])
    assert peeked == (1, 2, 3)
    assert list(rest) == [1, 2, 3]


def test_peekn_zero():
    peeked, rest = peekn(0, [1, 2, 3])
    assert peeked == ()
    assert list(rest) == [1, 2, 3]


# ---- random_sample ----
def test_random_sample_deterministic_seed():
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=2016))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_reproducible():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b


def test_random_sample_random_object():
    seq = list(range(100))
    randobj = random.Random(2016)
    result = list(random_sample(0.1, seq, random_state=randobj))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_zero():
    assert list(random_sample(0.0, range(50), random_state=1)) == []


def test_random_sample_prob_one():
    assert list(random_sample(1.0, range(10), random_state=1)) == list(range(10))
