import pytest
import itertools
import operator

from funcy.seqs import (
    repeatedly, iterate, take, drop, first, second, nth, last, rest, butlast,
    shuffled, ilen, lmap, lfilter, map, filter, lremove, remove, lkeep, keep,
    without, lwithout, lconcat, lcat, flatten, lflatten, lmapcat, mapcat,
    interleave, interpose, takewhile, dropwhile, ldistinct, distinct, split,
    lsplit, split_at, lsplit_at, split_by, lsplit_by, group_by, group_by_keys,
    group_values, count_by, count_reps, partition, lpartition, chunks, lchunks,
    partition_by, lpartition_by, with_prev, with_next, pairwise, reductions,
    lreductions, sums, lsums, lzip,
)


# repeatedly / iterate

def test_repeatedly_infinite():
    gen = repeatedly(lambda: 7)
    assert take(5, gen) == [7, 7, 7, 7, 7]


def test_repeatedly_n():
    assert take(10, repeatedly(lambda: 3, 3)) == [3, 3, 3]


def test_repeatedly_side_effects():
    calls = []
    def f():
        calls.append(1)
        return len(calls)
    assert take(4, repeatedly(f)) == [1, 2, 3, 4]


def test_iterate():
    gen = iterate(lambda x: x * 2, 1)
    assert take(5, gen) == [1, 2, 4, 8, 16]


# take / drop / first / second / nth / last / rest / butlast

def test_take():
    assert take(3, [1, 2, 3, 4, 5]) == [1, 2, 3]
    assert take(10, [1, 2]) == [1, 2]
    assert take(0, [1, 2]) == []
    assert take(3, iter([1, 2, 3, 4])) == [1, 2, 3]


def test_drop():
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]
    assert list(drop(0, [1, 2])) == [1, 2]
    assert list(drop(10, [1, 2])) == []


def test_first():
    assert first([1, 2, 3]) == 1
    assert first([]) is None
    assert first(iter([5, 6])) == 5


def test_second():
    assert second([1, 2, 3]) == 2
    assert second([1]) is None
    assert second([]) is None


def test_nth():
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, [10, 20, 30]) == 30
    assert nth(5, [10, 20]) is None
    # iterator path
    assert nth(2, iter([1, 2, 3, 4])) == 3
    assert nth(10, iter([1, 2])) is None


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([]) is None
    assert last(iter([1, 2, 3])) == 3
    assert last(iter([])) is None


def test_rest():
    assert list(rest([1, 2, 3])) == [2, 3]
    assert list(rest([1])) == []
    assert list(rest([])) == []


def test_butlast():
    assert list(butlast([1, 2, 3, 4])) == [1, 2, 3]
    assert list(butlast([1])) == []
    assert list(butlast([])) == []


def test_shuffled():
    # deterministic: single element and multiset comparison
    assert shuffled([1]) == [1]
    result = shuffled([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert result is not None


def test_ilen():
    assert ilen([1, 2, 3]) == 3
    assert ilen([]) == 0
    assert ilen(iter(range(10))) == 10
    assert ilen(x for x in range(7)) == 7


# map / filter / lmap / lfilter / remove / lremove

def test_map_basic():
    assert list(map(lambda x: x + 1, [1, 2, 3])) == [2, 3, 4]
    assert list(map(lambda a, b: a + b, [1, 2], [10, 20])) == [11, 22]


def test_map_string():
    # make_func: string becomes a regex finder
    result = list(map(r"\d+", ["a1", "b22", "c"]))
    assert result == ["1", "22", None]


def test_map_int_index():
    assert list(map(0, [(1, 2), (3, 4)])) == [1, 3]


def test_map_slice():
    assert list(map(slice(0, 2), [[1, 2, 3], [4, 5, 6]])) == [[1, 2], [4, 5]]


def test_map_dict():
    d = {"a": 1, "b": 2}
    assert list(map(d, ["a", "b"])) == [1, 2]


def test_map_none():
    assert list(map(None, [1, 2, 3])) == [1, 2, 3]


def test_filter_basic():
    assert list(filter(lambda x: x > 2, [1, 2, 3, 4])) == [3, 4]


def test_filter_string():
    assert list(filter(r"^a", ["abc", "bcd", "axe"])) == ["abc", "axe"]


def test_filter_none():
    assert list(filter(None, [0, 1, "", "x", None, 2])) == [1, "x", 2]


def test_filter_set():
    assert list(filter({1, 3}, [1, 2, 3, 4])) == [1, 3]


def test_lmap_lfilter():
    assert lmap(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert lfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [2, 4]


def test_map_invalid_type():
    with pytest.raises(TypeError):
        list(map(1.5, [1, 2]))


def test_remove():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_lremove():
    assert lremove(lambda x: x % 2 == 0, [1, 2, 3, 4]) == [1, 3]
    assert lremove({1, 2}, [1, 2, 3, 4]) == [3, 4]


def test_keep():
    assert list(keep(lambda x: x if x > 1 else None, [0, 1, 2, 3])) == [2, 3]
    # single-arg version
    assert list(keep([0, 1, None, 2, "", 3])) == [1, 2, 3]


def test_lkeep():
    assert lkeep(lambda x: x * 2 if x > 0 else 0, [-1, 0, 1, 2]) == [2, 4]
    assert lkeep([0, 1, None, 2]) == [1, 2]


# without / lwithout

def test_without():
    assert list(without([1, 2, 3, 4, 5], 2, 4)) == [1, 3, 5]
    assert list(without([1, 2], 1, 2)) == []


def test_lwithout():
    assert lwithout([1, 2, 3, 4], 2) == [1, 3, 4]
    assert lwithout([1, 2, 3], 5) == [1, 2, 3]


# concat / cat / flatten / mapcat

def test_lconcat():
    assert lconcat([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]
    assert lconcat() == []


def test_lcat():
    assert lcat([[1, 2], [3], []]) == [1, 2, 3]


def test_lmapcat():
    assert lmapcat(lambda x: [x, x], [1, 2, 3]) == [1, 1, 2, 2, 3, 3]


def test_mapcat():
    assert list(mapcat(lambda x: [x, x * 10], [1, 2])) == [1, 10, 2, 20]


def test_flatten():
    assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]


def test_flatten_custom_follow():
    # don't descend into ints by default; use custom follow for lists only
    seq = ["ab", ["cd", "ef"], "gh"]
    assert list(flatten(seq, follow=lambda x: isinstance(x, list))) == ["ab", "cd", "ef", "gh"]


def test_lflatten():
    assert lflatten([1, [2, 3], [[4]]]) == [1, 2, 3, 4]
    assert lflatten([]) == []


# interleave / interpose

def test_interleave():
    assert list(interleave([1, 2, 3], "abc")) == [1, "a", 2, "b", 3, "c"]
    assert list(interleave([1, 2], [3, 4], [5, 6])) == [1, 3, 5, 2, 4, 6]


def test_interpose():
    assert list(interpose(0, [1, 2, 3])) == [1, 0, 2, 0, 3]
    assert list(interpose(0, [1])) == [1]
    assert list(interpose(0, [])) == []


# takewhile / dropwhile

def test_takewhile():
    assert list(takewhile(lambda x: x < 3, [1, 2, 3, 4, 1])) == [1, 2]


def test_takewhile_single_arg():
    assert list(takewhile([1, 2, 0, 4])) == [1, 2]


def test_dropwhile():
    assert list(dropwhile(lambda x: x < 3, [1, 2, 3, 1, 4])) == [3, 1, 4]


def test_dropwhile_single_arg():
    # single-arg version drops leading truthy values, so leading 0 stops immediately
    assert list(dropwhile([0, 0, 1, 2, 0])) == [0, 0, 1, 2, 0]
    assert list(dropwhile([1, 2, 0, 4])) == [0, 4]


# distinct / ldistinct

def test_distinct():
    assert list(distinct([1, 2, 1, 3, 2, 4])) == [1, 2, 3, 4]
    assert list(distinct([])) == []


def test_distinct_key():
    data = [{"a": 1}, {"a": 1}, {"a": 2}]
    result = list(distinct(data, key=lambda x: x["a"]))
    assert result == [{"a": 1}, {"a": 2}]


def test_distinct_key_int():
    assert list(distinct([(1, "a"), (1, "b"), (2, "c")], key=0)) == [(1, "a"), (2, "c")]


def test_ldistinct():
    assert ldistinct([1, 1, 2, 3, 3, 1]) == [1, 2, 3]


# split / lsplit

def test_lsplit():
    passed, failed = lsplit(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
    assert passed == [2, 4]
    assert failed == [1, 3, 5]


def test_lsplit_string():
    passed, failed = lsplit(r"a", ["abc", "xyz", "cat"])
    assert passed == ["abc", "cat"]
    assert failed == ["xyz"]


def test_split():
    passed, failed = split(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6])
    assert list(passed) == [2, 4, 6]
    assert list(failed) == [1, 3, 5]


def test_split_interleaved_consumption():
    passed, failed = split(lambda x: x > 1, [1, 2, 3, 4])
    # consuming one of the pair drives the underlying iterator
    assert next(failed) == 1
    assert list(passed) == [2, 3, 4]
    assert list(failed) == []


# split_at / lsplit_at

def test_split_at():
    a, b = split_at(2, [1, 2, 3, 4])
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_split_at_beyond():
    a, b = split_at(10, [1, 2])
    assert list(a) == [1, 2]
    assert list(b) == []


def test_lsplit_at():
    assert lsplit_at(2, [1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert lsplit_at(0, [1, 2]) == ([], [1, 2])


# split_by / lsplit_by

def test_split_by():
    a, b = split_by(lambda x: x < 3, [1, 2, 3, 1])
    assert list(a) == [1, 2]
    assert list(b) == [3, 1]


def test_lsplit_by():
    assert lsplit_by(lambda x: x == 1, [1, 1, 2, 1]) == ([1, 1], [2, 1])
    assert lsplit_by(lambda x: x < 10, [1, 2, 3]) == ([1, 2, 3], [])


# group_by / group_by_keys / group_values / count_by / count_reps

def test_group_by():
    result = group_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: [1, 3, 5], 0: [2, 4]}


def test_group_by_string():
    result = group_by(r"\d", ["a1", "b2", "c"])
    assert dict(result) == {"1": ["a1"], "2": ["b2"], None: ["c"]}


def test_group_by_keys():
    data = [{"tags": ["a", "b"]}, {"tags": ["b"]}]
    result = group_by_keys(lambda x: x["tags"], data)
    assert dict(result) == {"a": [{"tags": ["a", "b"]}],
                            "b": [{"tags": ["a", "b"]}, {"tags": ["b"]}]}


def test_group_values():
    data = [("a", 1), ("b", 2), ("a", 3)]
    result = group_values(data)
    assert dict(result) == {"a": [1, 3], "b": [2]}


def test_count_by():
    result = count_by(lambda x: x % 2, [1, 2, 3, 4, 5])
    assert dict(result) == {1: 3, 0: 2}


def test_count_by_int():
    result = count_by(0, [(1, 2), (1, 3), (2, 4)])
    assert dict(result) == {1: 2, 2: 1}


def test_count_reps():
    result = count_reps("abca")
    assert dict(result) == {"a": 2, "b": 1, "c": 1}


# partition / lpartition / chunks / lchunks

def test_partition_list():
    assert lpartition(2, [1, 2, 3, 4, 5, 6]) == [[1, 2], [3, 4], [5, 6]]
    assert lpartition(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4]]


def test_partition_step():
    assert lpartition(2, 1, [1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4]]


def test_partition_iter():
    assert list(partition(2, iter([1, 2, 3, 4, 5, 6]))) == [[1, 2], [3, 4], [5, 6]]
    assert list(partition(2, iter([1, 2, 3]))) == [[1, 2]]


def test_partition_iter_step():
    assert list(partition(2, 1, iter([1, 2, 3, 4]))) == [[1, 2], [2, 3], [3, 4]]
    assert list(partition(2, 3, iter(range(10)))) == [[0, 1], [3, 4], [6, 7]]


def test_partition_zero():
    with pytest.raises(ValueError):
        list(partition(0, [1, 2, 3]))
    with pytest.raises(ValueError):
        list(partition(2, 0, [1, 2, 3]))


def test_chunks():
    assert lchunks(3, [1, 2, 3, 4, 5, 6, 7]) == [[1, 2, 3], [4, 5, 6], [7]]
    assert list(chunks(2, iter([1, 2, 3, 4, 5]))) == [[1, 2], [3, 4], [5]]


def test_chunks_step():
    assert lchunks(2, 1, [1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4], [4]]
    assert list(chunks(3, 2, iter(range(8)))) == [
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7]
    ]


# partition_by / lpartition_by

def test_partition_by():
    assert lpartition_by(lambda x: x % 2, [1, 3, 2, 4, 5]) == [[1, 3], [2, 4], [5]]
    assert [list(g) for g in partition_by(lambda x: x % 2, [1, 2, 3])] == [[1], [2], [3]]


def test_partition_by_int():
    data = [(1, "a"), (1, "b"), (2, "c")]
    assert lpartition_by(0, data) == [[(1, "a"), (1, "b")], [(2, "c")]]


# with_prev / with_next / pairwise

def test_with_prev():
    assert list(with_prev([1, 2, 3])) == [(1, None), (2, 1), (3, 2)]
    assert list(with_prev([1, 2, 3], fill=0)) == [(1, 0), (2, 1), (3, 2)]
    assert list(with_prev([])) == []


def test_with_next():
    assert list(with_next([1, 2, 3])) == [(1, 2), (2, 3), (3, None)]
    assert list(with_next([1, 2, 3], fill=0)) == [(1, 2), (2, 3), (3, 0)]
    assert list(with_next([])) == []


def test_pairwise():
    assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []


# reductions / lreductions / sums / lsums

def test_reductions():
    assert list(reductions(operator.add, [1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(reductions(lambda a, b: a * b, [1, 2, 3, 4])) == [1, 2, 6, 24]


def test_reductions_with_acc():
    assert list(reductions(operator.add, [1, 2, 3], 10)) == [11, 13, 16]


def test_lreductions():
    assert lreductions(operator.add, [1, 2, 3]) == [1, 3, 6]
    assert lreductions(operator.add, [1, 2, 3], acc=100) == [101, 103, 106]


def test_reductions_with_acc_operator_add():
    # f is exactly operator.add, acc supplied -> _reductions path
    assert list(reductions(operator.add, [1, 2, 3], 0)) == [1, 3, 6]


def test_sums():
    assert list(sums([1, 2, 3, 4])) == [1, 3, 6, 10]
    assert list(sums([1, 2, 3], acc=10)) == [11, 13, 16]


def test_lsums():
    assert lsums([1, 2, 3]) == [1, 3, 6]
    assert lsums([1, 2], acc=5) == [6, 8]


def test_sums_empty():
    assert list(sums([])) == []
    assert lsums([], acc=3) == []


# lzip / strict zip behavior

def test_lzip_basic():
    assert lzip([1, 2, 3], "abc") == [(1, "a"), (2, "b"), (3, "c")]


def test_lzip_zero_args():
    assert lzip() == []


def test_lzip_stops_at_shortest():
    assert lzip([1, 2, 3], [4, 5]) == [(1, 4), (2, 5)]


def test_lzip_strict_equal_lengths():
    assert lzip([1, 2], [3, 4], strict=True) == [(1, 3), (2, 4)]


def test_lzip_strict_single_seq():
    # strict with a single sequence just returns zipped single elements
    assert lzip([1, 2, 3], strict=True) == [(1,), (2,), (3,)]


def test_lzip_strict_first_too_long():
    with pytest.raises(ValueError) as excinfo:
        lzip([1, 2, 3], [4, 5], strict=True)
    assert "argument 2 is shorter than argument 1" in str(excinfo.value)


def test_lzip_strict_later_too_long():
    with pytest.raises(ValueError) as excinfo:
        lzip([1, 2], [3, 4, 5], strict=True)
    assert "argument 2 is longer than argument 1" in str(excinfo.value)


def test_lzip_strict_first_of_three_short():
    with pytest.raises(ValueError) as excinfo:
        lzip([1], [2, 3], [4, 5], strict=True)
    assert "argument 2 is longer than argument 1" in str(excinfo.value)


def test_lzip_strict_second_of_three_short():
    with pytest.raises(ValueError) as excinfo:
        lzip([1, 2, 3], [4], [5, 6, 7], strict=True)
    assert "argument 2 is shorter than argument 1" in str(excinfo.value)


def test_lzip_strict_third_of_three_short():
    with pytest.raises(ValueError) as excinfo:
        lzip([1, 2, 3], [4, 5, 6], [7], strict=True)
    # builtin zip() on Python 3.10+ says "arguments 1-2" (plural)
    assert "argument 3 is shorter than arguments 1-2" in str(excinfo.value)


def test_lzip_strict_iters():
    # iterators have no len(); exercise the _zip_strict_iters path
    assert lzip(iter([1, 2, 3]), iter("abc"), strict=True) == [
        (1, "a"), (2, "b"), (3, "c")
    ]


def test_lzip_strict_iters_too_long_first():
    with pytest.raises(ValueError) as excinfo:
        lzip(iter([1, 2, 3]), iter([4, 5]), strict=True)
    assert "argument 2 is shorter than argument 1" in str(excinfo.value)


def test_lzip_strict_iters_too_long_later():
    with pytest.raises(ValueError) as excinfo:
        lzip(iter([1, 2]), iter([3, 4, 5]), strict=True)
    assert "argument 2 is longer than argument 1" in str(excinfo.value)
