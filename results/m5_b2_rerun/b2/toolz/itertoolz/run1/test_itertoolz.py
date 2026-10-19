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


# ---------------- remove ----------------

def test_remove_basic():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_remove_empty():
    assert list(remove(lambda x: True, [])) == []


def test_remove_returns_iterator():
    # remove keeps elements for which predicate is False.
    r = remove(lambda x: x > 1, [1, 2, 3])
    assert next(r) == 1
    with pytest.raises(StopIteration):
        next(r)


# ---------------- accumulate ----------------

def test_accumulate_basic():
    assert list(accumulate(operator.add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]


def test_accumulate_mul():
    assert list(accumulate(operator.mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(operator.add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_accumulate_empty_with_initial():
    assert list(accumulate(operator.add, [], 1)) == [1]


def test_accumulate_empty_no_initial():
    assert list(accumulate(operator.add, [])) == []


def test_accumulate_initial_is_zero():
    # Ensure initial=0 (falsy) is distinct from the sentinel.
    assert list(accumulate(operator.add, [1, 2], 0)) == [0, 1, 3]


# ---------------- groupby ----------------

def test_groupby_callable():
    result = groupby(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_noncallable_key():
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
    assert groupby(lambda x: x, []) == {}


def test_groupby_len():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'],
                      7: ['Charlie']}


def test_groupby_returns_real_lists():
    result = groupby(lambda x: x % 2, [1, 2, 3])
    assert isinstance(result[1], list)
    assert result[1] == [1, 3]


# ---------------- merge_sorted ----------------

def test_merge_sorted_empty():
    assert list(merge_sorted()) == []


def test_merge_sorted_single():
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_two():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_three_strings():
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_with_empty_seq():
    assert list(merge_sorted([1, 2, 3], [])) == [1, 2, 3]
    assert list(merge_sorted([], [1, 2, 3])) == [1, 2, 3]


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]


def test_merge_sorted_many():
    seqs = [[1, 4, 7], [2, 5, 8], [0, 3, 6, 9]]
    assert list(merge_sorted(*seqs)) == list(range(10))


def test_merge_sorted_many_key():
    # Exercise recursive binary merge with key on >=4 sequences.
    seqs = [[3, 6, 9], [1, 4, 7], [2, 5, 8], [0, 6, 12]]
    result = list(merge_sorted(*seqs, key=lambda x: x // 3))
    # key(x) = x//3; merge stable-ish. Compute expected by stable merge.
    expected = sorted(itertools.chain(*seqs), key=lambda x: x // 3)
    assert list(map(lambda x: x // 3, result)) == \
        list(map(lambda x: x // 3, expected))
    # Also verify original elements are all present.
    assert sorted(result) == sorted(itertools.chain(*seqs))


def test_merge_sorted_equal_values_no_key():
    # Equal values: ensure both are emitted (not dropped) - distinguishes
    # strict `<` from `<=`.
    assert list(merge_sorted([1, 1], [1, 1])) == [1, 1, 1, 1]


def test_merge_sorted_equal_values_with_key():
    assert list(merge_sorted([1, 2], [3, 4], key=lambda x: 0)) == [1, 2, 3, 4]


def test_merge_sorted_key_uneven_branch():
    # 3 seqs forces left/right split where one side has 2 (recursive).
    a = [2, 8]
    b = [1, 4, 7]
    c = [3, 5, 9]
    result = list(merge_sorted(a, b, c))
    assert result == sorted(a + b + c)


def test_merge_sorted_key_uneven_branch_with_key():
    a = [20, 80]
    b = [10, 40, 70]
    c = [30, 50, 90]
    result = list(merge_sorted(a, b, c, key=lambda x: x))
    assert result == sorted(a + b + c)


# ---------------- interleave ----------------

def test_interleave_two():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_interleave_uneven():
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_interleave_empty():
    assert list(interleave([])) == []


def test_interleave_empty_inner():
    assert list(interleave([[], []])) == []


def test_interleave_three_mixed():
    assert list(interleave([[1, 2, 3], [4], [5, 6]])) == [1, 4, 5, 2, 6, 3]


# ---------------- unique ----------------

def test_unique_basic():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)


def test_unique_key():
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')


def test_unique_empty():
    assert tuple(unique([])) == ()


def test_unique_key_first_wins():
    assert list(unique(['a', 'A', 'b', 'B'], key=str.lower)) == ['a', 'b']


# ---------------- isiterable ----------------

def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


# ---------------- isdistinct ----------------

def test_isdistinct_list():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_str():
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_isdistinct_iterator():
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 1])) is False


def test_isdistinct_tuple():
    assert isdistinct((1, 2, 3)) is True
    assert isdistinct((1, 1)) is False


# ---------------- take / drop / tail / take_nth ----------------

def test_take():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]


def test_take_more_than_len():
    assert list(take(10, [1, 2])) == [1, 2]


def test_take_zero():
    assert list(take(0, [1, 2, 3])) == []


def test_drop():
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_drop_more_than_len():
    assert list(drop(10, [1, 2])) == []


def test_drop_zero():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]


def test_tail_list():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]


def test_tail_iterator():
    assert tail(2, iter([10, 20, 30, 40, 50])) == (40, 50)


def test_tail_string():
    assert tail(2, 'abcde') == 'de'


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_take_nth_three():
    assert list(take_nth(3, range(10))) == [0, 3, 6, 9]


# ---------------- first / second / nth / last ----------------

def test_first():
    assert first('ABC') == 'A'


def test_second():
    assert second('ABC') == 'B'


def test_nth_sequence():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10


def test_nth_iterator():
    assert nth(1, iter('ABC')) == 'B'


def test_last():
    assert last('ABC') == 'C'
    assert last(iter([1, 2, 3])) == 3


def test_first_empty_raises():
    with pytest.raises(StopIteration):
        first([])


# ---------------- get ----------------

def test_get_single():
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


def test_get_single_default():
    assert get('missing', {'a': 1}, None) is None
    assert get(5, [1, 2, 3], 'x') == 'x'


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get('missing', {'a': 1})


def test_get_single_element_list():
    assert get([1], 'ABC') == ('B',)


def test_get_empty_list():
    assert get([], 'ABC') == ()


def test_get_nonindexable_default():
    assert get(0, 5, 'dflt') == 'dflt'


def test_get_nonindexable_no_default_raises():
    with pytest.raises(TypeError):
        get(0, 5)


def test_get_list_missing_no_default_raises():
    with pytest.raises(KeyError):
        get(['a', 'z'], {'a': 1})


# ---------------- concat / concatv / mapcat ----------------

def test_concat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]


def test_concatv():
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [["a", "b"], ["c", "d", "e"]])) == \
        ['A', 'B', 'C', 'D', 'E']


# ---------------- cons / interpose ----------------

def test_cons():
    assert list(cons(1, [2, 3])) == [1, 2, 3]


def test_interpose():
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_interpose_empty():
    assert list(interpose("a", [])) == []


def test_interpose_one():
    assert list(interpose("a", [1])) == [1]


# ---------------- frequencies ----------------

def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}


def test_frequencies_empty():
    assert frequencies([]) == {}


# ---------------- reduceby ----------------

def test_reduceby_no_default():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    assert reduceby(iseven, operator.add, data) == {False: 9, True: 6}
    assert reduceby(iseven, operator.mul, data) == {False: 15, True: 8}


def test_reduceby_with_init():
    data = [1, 2, 3, 4, 5]
    iseven = lambda x: x % 2 == 0
    result = reduceby(iseven, operator.add, data, 0)
    assert result == {False: 9, True: 6}


def test_reduceby_with_init_nonzero():
    data = [1, 2, 3, 4]
    iseven = lambda x: x % 2 == 0
    # With init=100 each group starts at 100 then adds its members.
    assert reduceby(iseven, operator.add, data, 100) == \
        {False: 104, True: 106}


def test_reduceby_init_callable():
    def set_add(s, i):
        s.add(i)
        return s
    result = reduceby(lambda x: x % 2 == 0, set_add, [1, 2, 3, 4, 1, 2, 3],
                      set)
    assert result == {False: {1, 3}, True: {2, 4}}


def test_reduceby_noncallable_key():
    projects = [{'name': 'build roads', 'state': 'CA', 'cost': 1000000},
                {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
                {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
                {'name': 'help farmers', 'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_reduceby_empty():
    assert reduceby(lambda x: x, operator.add, []) == {}


def test_reduceby_single_item_no_init():
    # First item stored directly, no binop call.
    assert reduceby(lambda x: 0, operator.add, [42]) == {0: 42}


# ---------------- iterate ----------------

def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_iterate_double():
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


# ---------------- sliding_window ----------------

def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == \
        [(1, 2), (2, 3), (3, 4)]


def test_sliding_window_three():
    assert list(sliding_window(3, [1, 2, 3, 4, 5])) == \
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_sliding_window_too_small():
    assert list(sliding_window(2, [1])) == []


def test_sliding_window_mean():
    mean = lambda seq: float(sum(seq)) / len(seq)
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == \
        [1.5, 2.5, 3.5]


# ---------------- partition ----------------

def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_drop_tail():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_pad():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_empty():
    assert list(partition(2, [])) == []


# ---------------- partition_all ----------------

def test_partition_all_even():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]


def test_partition_all_odd():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_empty():
    assert list(partition_all(2, [])) == []


def test_partition_all_iterator():
    assert list(partition_all(2, iter([1, 2, 3, 4, 5]))) == \
        [(1, 2), (3, 4), (5,)]


def test_partition_all_three():
    assert list(partition_all(3, [1, 2, 3, 4])) == [(1, 2, 3), (4,)]


def test_partition_all_generator():
    def g():
        yield from [1, 2, 3, 4, 5]
    assert list(partition_all(2, g())) == [(1, 2), (3, 4), (5,)]


# ---------------- count ----------------

def test_count_list():
    assert count([1, 2, 3]) == 3


def test_count_iterator():
    assert count(iter([1, 2, 3, 4])) == 4


def test_count_empty():
    assert count([]) == 0


# ---------------- pluck ----------------

def test_pluck_name():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']


def test_pluck_multiple():
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == \
        [(1, 2), (4, 5)]


def test_pluck_default():
    data = [{'a': 1}, {'b': 2}]
    assert list(pluck('a', data, None)) == [1, None]


def test_pluck_default_list():
    data = [[1, 2], [3]]
    assert list(pluck([0, 1], data, 'X')) == [(1, 2), (3, 'X')]


def test_pluck_single_element_list():
    data = [{'a': 1}, {'a': 2}]
    assert list(pluck(['a'], data)) == [(1,), (2,)]


# ---------------- getter ----------------

def test_getter_single():
    g = getter(0)
    assert g([10, 20]) == 10


def test_getter_list_one():
    g = getter([1])
    assert g('ABC') == ('B',)


def test_getter_list_multiple():
    g = getter([0, 2])
    assert g('ABC') == ('A', 'C')


def test_getter_list_empty():
    g = getter([])
    assert g('ABC') == ()


# ---------------- join ----------------

def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'),
               ('Edith', 'Alice'), ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    expected_pairs = {
        ('Alice', 'Berlin'),
        ('Alice', 'Paris'),
        ('Alice', 'Shanghai'),
        ('Edith', 'Chicago'),
        ('Edith', 'NYC'),
        ('Zhao', 'Chicago'),
        ('Zhao', 'NYC'),
        ('Zhao', 'Berlin'),
        ('Zhao', 'Paris'),
    }
    got = {(a[0], b[1]) for (a, b) in set(result)}
    assert got == expected_pairs


def test_join_full_outer():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       left_default=None, right_default=None))
    assert sorted(map(str, result)) == sorted(map(str, [(2, 2), (3, 3), (None, 4), (1, None)]))


def test_join_right_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [2, 3], left_default=None))
    assert sorted(map(str, result)) == sorted(map(str, [(2, 2), (None, 3)]))


def test_join_left_join():
    identity = lambda x: x
    result = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                       right_default=None))
    assert sorted(map(str, result)) == sorted(map(str, [(2, 2), (3, 3), (1, None)]))


def test_join_noncallable_keys():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC')]
    result = list(join(1, friends, 0, cities))
    assert result == []


def test_join_noncallable_keys_with_match():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao')]
    cities = [('Alice', 'NYC'), ('Zhao', 'Shanghai'), ('Dan', 'Sydney')]
    result = list(join(0, friends, 0, cities))
    assert result == [
        (('Alice', 'Edith'), ('Alice', 'NYC')),
        (('Alice', 'Zhao'), ('Alice', 'NYC')),
    ]


def test_join_right_join_default_not_used():
    # No unmatched, left_default given only -> same as inner.
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 2], left_default='X'))
    assert sorted(result) == [(1, 1), (2, 2)]


def test_join_left_join_default_not_used():
    identity = lambda x: x
    result = list(join(identity, [1, 2], identity, [1, 2], right_default='X'))
    assert sorted(result) == [(1, 1), (2, 2)]


# ---------------- diff ----------------

def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]


def test_diff_key():
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]


def test_diff_single_list_arg():
    assert list(diff([[1, 2, 3], [1, 2, 10]])) == [(3, 10)]


def test_diff_too_few_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_diff_three():
    result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 8]))
    assert result == [(3, 3, 8)]


def test_diff_identical():
    assert list(diff([1, 2, 3], [1, 2, 3])) == []


# ---------------- topk ----------------

def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)


def test_topk_key():
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')


def test_topk_noncallable_key():
    data = [{'v': 1}, {'v': 100}, {'v': 10}]
    assert topk(2, data, key='v') == ({'v': 100}, {'v': 10})


def test_topk_more_than_len():
    assert topk(5, [3, 1, 2]) == (3, 2, 1)


# ---------------- peek / peekn ----------------

def test_peek():
    seq = [0, 1, 2, 3, 4]
    first_val, rest = peek(seq)
    assert first_val == 0
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peek_iterator():
    it = iter([10, 20, 30])
    first_val, rest = peek(it)
    assert first_val == 10
    assert list(rest) == [10, 20, 30]


def test_peekn():
    seq = [0, 1, 2, 3, 4]
    first_two, rest = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(rest) == [0, 1, 2, 3, 4]


def test_peekn_more_than_len():
    peeked, rest = peekn(10, [1, 2])
    assert peeked == (1, 2)
    assert list(rest) == [1, 2]


# ---------------- random_sample ----------------

def test_random_sample_seed_deterministic():
    seq = list(range(100))
    a = list(random_sample(0.1, seq, random_state=2016))
    b = list(random_sample(0.1, seq, random_state=2016))
    assert a == b
    assert a == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_prob_one():
    seq = list(range(5))
    assert list(random_sample(1.0, seq, random_state=0)) == seq


def test_random_sample_prob_zero():
    seq = list(range(5))
    assert list(random_sample(0.0, seq, random_state=0)) == []


def test_random_sample_random_object():
    from random import Random
    seq = list(range(100))
    result = list(random_sample(0.1, seq, random_state=Random(2016)))
    assert result == [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]


def test_random_sample_filters_lazily_one_step():
    seen = []

    def gen():
        for i in range(100):
            seen.append(i)
            yield i

    it = random_sample(0.0, gen(), random_state=0)
    assert next(it, None) is None
    assert len(seen) == 100


def test_random_sample_filters_lazily_finds_match():
    seen = []

    def gen():
        for i in range(100):
            seen.append(i)
            yield i

    it = random_sample(1.0, gen(), random_state=0)
    assert next(it) == 0
    assert seen == [0]
