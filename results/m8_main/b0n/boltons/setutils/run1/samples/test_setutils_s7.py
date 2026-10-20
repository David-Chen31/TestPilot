"""Comprehensive pytest test suite for boltons.setutils."""

import copy
import pickle

import pytest

from boltons.setutils import IndexedSet, complement


# ---------------------------------------------------------------------------
# IndexedSet: construction, repr, equality
# ---------------------------------------------------------------------------

class TestIndexedSetConstruction:
    def test_empty(self):
        s = IndexedSet()
        assert len(s) == 0
        assert list(s) == []

    def test_from_list_dedup_preserves_order(self):
        s = IndexedSet([3, 1, 2, 1, 3])
        assert list(s) == [3, 1, 2]
        assert len(s) == 3

    def test_from_string(self):
        s = IndexedSet("freecreditreport.com")
        assert ''.join(s[:s.index('.')]) == 'frecditpo'

    def test_from_generator(self):
        s = IndexedSet(x for x in [1, 1, 2, 2, 3])
        assert list(s) == [1, 2, 3]

    def test_from_iterable_classmethod(self):
        s = IndexedSet.from_iterable([1, 2, 2, 3])
        assert isinstance(s, IndexedSet)
        assert list(s) == [1, 2, 3]

    def test_repr(self):
        s = IndexedSet([1, 2, 3])
        assert repr(s) == "IndexedSet([1, 2, 3])"

    def test_repr_empty(self):
        assert repr(IndexedSet()) == "IndexedSet([])"

    def test_eq_indexedset_same_order(self):
        assert IndexedSet([1, 2, 3]) == IndexedSet([1, 2, 3])

    def test_eq_indexedset_different_order(self):
        assert IndexedSet([1, 2, 3]) != IndexedSet([3, 2, 1])

    def test_eq_plain_set(self):
        assert IndexedSet([1, 2, 3]) == {3, 2, 1}
        assert {3, 2, 1} == IndexedSet([1, 2, 3])

    def test_eq_non_set_returns_false(self):
        assert (IndexedSet([1, 2]) == 42) is False

    def test_eq_unhashable_other(self):
        # set(self) == set(other) - other is a list, sets are unhashable inside
        assert (IndexedSet([1, 2]) == [1, [2]]) is False

    def test_eq_indexedset_with_list_other(self):
        assert IndexedSet([1, 2]) == [1, 2]


# ---------------------------------------------------------------------------
# IndexedSet: basic MutableSet API
# ---------------------------------------------------------------------------

class TestIndexedSetBasicOps:
    def test_contains(self):
        s = IndexedSet([1, 2, 3])
        assert 1 in s
        assert 5 not in s

    def test_iter(self):
        s = IndexedSet(['a', 'b', 'c'])
        assert list(s) == ['a', 'b', 'c']

    def test_reversed(self):
        s = IndexedSet([1, 2, 3])
        assert list(reversed(s)) == [3, 2, 1]

    def test_add(self):
        s = IndexedSet()
        s.add(5)
        s.add(3)
        s.add(5)
        assert list(s) == [5, 3]
        assert len(s) == 2

    def test_remove(self):
        s = IndexedSet([1, 2, 3])
        s.remove(2)
        assert list(s) == [1, 3]
        assert 2 not in s

    def test_remove_missing_raises(self):
        s = IndexedSet([1, 2, 3])
        with pytest.raises(KeyError):
            s.remove(99)

    def test_discard(self):
        s = IndexedSet([1, 2, 3])
        s.discard(2)
        assert list(s) == [1, 3]

    def test_discard_missing_no_raise(self):
        s = IndexedSet([1, 2, 3])
        s.discard(99)
        assert list(s) == [1, 2, 3]

    def test_clear(self):
        s = IndexedSet([1, 2, 3])
        s.clear()
        assert len(s) == 0
        assert list(s) == []

    def test_count(self):
        s = IndexedSet([1, 2, 3])
        assert s.count(1) == 1
        assert s.count(99) == 0

    def test_index(self):
        s = IndexedSet(['a', 'b', 'c'])
        assert s.index('a') == 0
        assert s.index('c') == 2

    def test_index_missing_raises(self):
        s = IndexedSet([1, 2, 3])
        with pytest.raises(ValueError):
            s.index(99)


# ---------------------------------------------------------------------------
# IndexedSet: indexing and slicing
# ---------------------------------------------------------------------------

class TestIndexedSetIndexing:
    def test_positive_index(self):
        s = IndexedSet([10, 20, 30])
        assert s[0] == 10
        assert s[1] == 20
        assert s[2] == 30

    def test_negative_index(self):
        s = IndexedSet([10, 20, 30])
        assert s[-1] == 30
        assert s[-2] == 20

    def test_index_out_of_range(self):
        s = IndexedSet([10, 20, 30])
        with pytest.raises(IndexError):
            s[5]
        with pytest.raises(IndexError):
            s[-5]

    def test_index_after_removal(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        s.remove(2)
        s.remove(4)
        assert list(s) == [1, 3, 5]
        assert s[0] == 1
        assert s[1] == 3
        assert s[2] == 5
        assert s[-1] == 5

    def test_slice_basic(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        assert list(s[1:4]) == [2, 3, 4]

    def test_slice_step(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        assert list(s[::2]) == [1, 3, 5]

    def test_slice_returns_indexedset(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        result = s[1:3]
        assert isinstance(result, IndexedSet)

    def test_slice_negative_bounds(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        assert list(s[-2:]) == [4, 5]
        assert list(s[:-2]) == [1, 2, 3]

    def test_slice_after_removal(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        s.remove(2)
        s.remove(4)
        assert list(s[0:2]) == [1, 3]
        assert list(s[-2:]) == [3, 5]


# ---------------------------------------------------------------------------
# IndexedSet: pop
# ---------------------------------------------------------------------------

class TestIndexedSetPop:
    def test_pop_default_removes_last(self):
        s = IndexedSet([1, 2, 3])
        assert s.pop() == 3
        assert list(s) == [1, 2]

    def test_pop_negative_one(self):
        s = IndexedSet([1, 2, 3])
        assert s.pop(-1) == 3
        assert list(s) == [1, 2]

    def test_pop_last_index(self):
        s = IndexedSet([1, 2, 3])
        assert s.pop(2) == 3
        assert list(s) == [1, 2]

    def test_pop_first(self):
        s = IndexedSet([1, 2, 3])
        assert s.pop(0) == 1
        assert list(s) == [2, 3]

    def test_pop_middle(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        assert s.pop(2) == 3
        assert list(s) == [1, 2, 4, 5]

    def test_pop_invalid_index(self):
        s = IndexedSet([1, 2, 3])
        with pytest.raises(IndexError):
            s.pop(10)


# ---------------------------------------------------------------------------
# IndexedSet: iteration helpers after removals
# ---------------------------------------------------------------------------

class TestIndexedSetIterAfterRemoval:
    def test_iter_skips_dead(self):
        s = IndexedSet(range(10))
        for i in [0, 3, 7, 9]:
            s.remove(i)
        expected = [i for i in range(10) if i not in (0, 3, 7, 9)]
        assert list(s) == expected

    def test_reversed_skips_dead(self):
        s = IndexedSet(range(10))
        for i in [0, 3, 7, 9]:
            s.remove(i)
        expected = list(reversed([i for i in range(10) if i not in (0, 3, 7, 9)]))
        assert list(reversed(s)) == expected


# ---------------------------------------------------------------------------
# IndexedSet: set operations (non-mutating)
# ---------------------------------------------------------------------------

class TestIndexedSetSetOps:
    def test_isdisjoint(self):
        assert IndexedSet([1, 2]).isdisjoint([3, 4])
        assert not IndexedSet([1, 2]).isdisjoint([2, 3])

    def test_issubset(self):
        assert IndexedSet([1, 2]).issubset([1, 2, 3])
        assert IndexedSet([1, 2]).issubset([1, 2])
        assert not IndexedSet([1, 2, 3]).issubset([1, 2])
        assert not IndexedSet([1, 5]).issubset([1, 2, 3])

    def test_issuperset(self):
        assert IndexedSet([1, 2, 3]).issuperset([1, 2])
        assert IndexedSet([1, 2, 3]).issuperset([1, 2, 3])
        assert not IndexedSet([1, 2]).issuperset([1, 2, 3])
        assert not IndexedSet([1, 2]).issuperset([1, 5])

    def test_union(self):
        s = IndexedSet([1, 2]).union([3, 4], [2, 5])
        assert isinstance(s, IndexedSet)
        assert list(s) == [1, 2, 3, 4, 5]

    def test_or_operator(self):
        s = IndexedSet([1, 2]) | {3}
        assert isinstance(s, IndexedSet)
        assert set(s) == {1, 2, 3}

    def test_ror_operator(self):
        s = {3} | IndexedSet([1, 2])
        assert isinstance(s, IndexedSet)
        assert set(s) == {1, 2, 3}

    def test_intersection_single(self):
        s = IndexedSet([1, 2, 3, 4]).intersection([2, 4, 5])
        assert list(s) == [2, 4]

    def test_intersection_multi(self):
        s = IndexedSet([1, 2, 3, 4]).intersection([2, 3, 4], [3, 4, 5])
        assert list(s) == [3, 4]

    def test_iter_intersection(self):
        s = IndexedSet([1, 2, 3, 4])
        assert list(s.iter_intersection([2, 3], [3, 4])) == [3]

    def test_and_operator(self):
        s = IndexedSet([1, 2, 3]) & {2, 3, 4}
        assert list(s) == [2, 3]

    def test_rand_operator(self):
        s = {1, 2, 3} & IndexedSet([2, 3, 4])
        assert isinstance(s, IndexedSet)
        assert list(s) == [2, 3]

    def test_difference_single(self):
        s = IndexedSet([1, 2, 3, 4]).difference([2, 4])
        assert list(s) == [1, 3]

    def test_difference_multi(self):
        s = IndexedSet([1, 2, 3, 4]).difference([2], [4])
        assert list(s) == [1, 3]

    def test_iter_difference(self):
        s = IndexedSet([1, 2, 3, 4])
        assert list(s.iter_difference([2], [4])) == [1, 3]

    def test_sub_operator(self):
        s = IndexedSet([1, 2, 3, 4]) - {2, 4}
        assert list(s) == [1, 3]

    def test_rsub_operator(self):
        # other - self, other is a list here
        result = [1, 2, 3, 4] - IndexedSet([2, 4])
        assert result == [1, 3]
        assert isinstance(result, list)

    def test_rsub_operator_set(self):
        result = {1, 2, 3, 4} - IndexedSet([2, 4])
        assert result == {1, 3}
        assert isinstance(result, set)

    def test_symmetric_difference(self):
        s = IndexedSet([1, 2, 3]).symmetric_difference([2, 3, 4])
        assert set(s) == {1, 4}

    def test_xor_operator(self):
        s = IndexedSet([1, 2, 3]) ^ {2, 3, 4}
        assert set(s) == {1, 4}

    def test_rxor_operator(self):
        s = {1, 2, 3} ^ IndexedSet([2, 3, 4])
        assert set(s) == {1, 4}


# ---------------------------------------------------------------------------
# IndexedSet: in-place set operations and dunder operators
# ---------------------------------------------------------------------------

class TestIndexedSetInPlaceOps:
    def test_update_single(self):
        s = IndexedSet([1, 2])
        s.update([2, 3])
        assert list(s) == [1, 2, 3]

    def test_update_multi(self):
        s = IndexedSet([1, 2])
        s.update([2, 3], [4])
        assert list(s) == [1, 2, 3, 4]

    def test_update_no_args(self):
        s = IndexedSet([1, 2])
        s.update()
        assert list(s) == [1, 2]

    def test_ior_operator(self):
        s = IndexedSet([1, 2])
        s |= {2, 3}
        assert list(s) == [1, 2, 3]

    def test_intersection_update(self):
        s = IndexedSet([1, 2, 3, 4])
        s.intersection_update([2, 3, 4], [3, 4, 5])
        assert list(s) == [3, 4]

    def test_intersection_update_bulk_path(self):
        s = IndexedSet(range(1000))
        s.intersection_update(range(500))
        assert list(s) == list(range(500))

    def test_iand_operator(self):
        s = IndexedSet([1, 2, 3, 4])
        s &= {2, 4}
        assert list(s) == [2, 4]

    def test_difference_update(self):
        s = IndexedSet([1, 2, 3, 4])
        s.difference_update([2, 4])
        assert list(s) == [1, 3]

    def test_difference_update_bulk_path(self):
        s = IndexedSet(range(1000))
        s.difference_update(range(500))
        assert list(s) == list(range(500, 1000))

    def test_difference_update_with_self(self):
        s = IndexedSet([1, 2, 3])
        s.difference_update(s)
        assert list(s) == []

    def test_isub_operator(self):
        s = IndexedSet([1, 2, 3, 4])
        s -= {2, 4}
        assert list(s) == [1, 3]

    def test_symmetric_difference_update(self):
        s = IndexedSet([1, 2, 3])
        s.symmetric_difference_update([2, 3, 4])
        assert set(s) == {1, 4}

    def test_symmetric_difference_update_self(self):
        s = IndexedSet([1, 2, 3])
        s.symmetric_difference_update(s)
        assert list(s) == []

    def test_ixor_operator(self):
        s = IndexedSet([1, 2, 3])
        s ^= {2, 3, 4}
        assert set(s) == {1, 4}


# ---------------------------------------------------------------------------
# IndexedSet: list ops (reverse, sort)
# ---------------------------------------------------------------------------

class TestIndexedSetListOps:
    def test_reverse(self):
        s = IndexedSet([1, 2, 3])
        s.reverse()
        assert list(s) == [3, 2, 1]
        assert s[0] == 3
        assert s.index(1) == 2

    def test_reverse_with_dead(self):
        s = IndexedSet([1, 2, 3, 4])
        s.remove(2)
        s.reverse()
        assert list(s) == [4, 3, 1]

    def test_sort(self):
        s = IndexedSet([3, 1, 2])
        s.sort()
        assert list(s) == [1, 2, 3]
        assert s.index(3) == 2

    def test_sort_reverse(self):
        s = IndexedSet([1, 2, 3])
        s.sort(reverse=True)
        assert list(s) == [3, 2, 1]

    def test_sort_with_key(self):
        s = IndexedSet(['aaa', 'b', 'cc'])
        s.sort(key=len)
        assert list(s) == ['b', 'cc', 'aaa']

    def test_sort_already_sorted_noop(self):
        s = IndexedSet([1, 2, 3])
        s.sort()
        assert list(s) == [1, 2, 3]


# ---------------------------------------------------------------------------
# IndexedSet: dead-index machinery / compaction via public API
# ---------------------------------------------------------------------------

class TestIndexedSetCompaction:
    def test_remove_all(self):
        s = IndexedSet(range(10))
        for i in range(10):
            s.remove(i)
        assert list(s) == []
        assert len(s) == 0

    def test_remove_many_then_add(self):
        s = IndexedSet(range(100))
        for i in range(0, 100, 2):
            s.remove(i)
        assert list(s) == list(range(1, 100, 2))
        for i in range(0, 100, 2):
            s.add(i)
        assert list(s) == list(range(1, 100, 2)) + list(range(0, 100, 2))

    def test_remove_from_end(self):
        s = IndexedSet([1, 2, 3, 4, 5])
        s.remove(5)
        s.remove(4)
        assert list(s) == [1, 2, 3]

    def test_reindex_after_many_removals(self):
        s = IndexedSet(range(50))
        for i in range(10, 40):
            s.remove(i)
        expected = list(range(10)) + list(range(40, 50))
        assert list(s) == expected
        for idx, val in enumerate(expected):
            assert s[idx] == val
            assert s.index(val) == idx

    def test_bulk_then_iter(self):
        s = IndexedSet(range(2000))
        s.difference_update(range(0, 2000, 3))
        expected = [i for i in range(2000) if i % 3 != 0]
        assert list(s) == expected


# ---------------------------------------------------------------------------
# IndexedSet: copy / pickle
# ---------------------------------------------------------------------------

class TestIndexedSetCopy:
    def test_copy_equal(self):
        s = IndexedSet([1, 2, 3])
        c = copy.copy(s)
        assert list(c) == list(s)

    def test_deepcopy(self):
        s = IndexedSet([[1], [2]])
        c = copy.deepcopy(s)
        assert list(c) == [[1], [2]]

    def test_pickle_roundtrip(self):
        s = IndexedSet([1, 2, 3])
        s.remove(2)
        s2 = pickle.loads(pickle.dumps(s))
        assert list(s2) == [1, 3]


# ---------------------------------------------------------------------------
# complement()
# ---------------------------------------------------------------------------

class TestComplement:
    def test_basic_complement(self):
        c = complement(set([1, 2, 3]))
        assert 1 not in c
        assert 4 in c
        assert 100 in c

    def test_repr_excluded(self):
        c = complement(set([1, 2]))
        assert repr(c) == "complement({1, 2})" or repr(c) == "complement({2, 1})"

    def test_complement_from_list(self):
        c = complement([1, 2])
        assert 1 not in c
        assert 3 in c

    def test_double_complement(self):
        s = set(range(5))
        c = complement(s)
        c2 = complement(c)
        assert isinstance(c2, _ComplementSet)
        assert set(c2) == s

    def test_invert_returns_complement(self):
        c = complement(set([1]))
        inv = ~c
        assert isinstance(inv, _ComplementSet)

    def test_complement_frozenset(self):
        c = complement(frozenset([1, 2]))
        assert 1 not in c
        assert 3 in c

    def test_length_of_complement_undefined(self):
        c = complement(set([1, 2]))
        with pytest.raises(NotImplementedError):
            len(c)

    def test_iter_of_complement_undefined(self):
        c = complement(set([1, 2]))
        with pytest.raises(NotImplementedError):
            iter(c)

    def test_bool_of_complement(self):
        c = complement(set([1]))
        assert bool(c) is True

    def test_complement_of_complemented_has_content(self):
        c = complement(set([1, 2]))
        inv = ~c
        assert list(inv) == [1, 2]
        assert len(inv) == 2
        assert bool(inv) is True


class TestComplementOps:
    def test_and_with_set(self):
        c = complement(set([2, 3]))
        result = c & set([1, 2, 3, 4])
        assert isinstance(result, _ComplementSet)
        assert 1 in result
        assert 4 in result
        assert 2 not in result
        assert 3 not in result

    def test_rand_with_set(self):
        c = complement(set([2, 3]))
        result = set([1, 2, 3, 4]) & c
        assert 1 in result
        assert 4 in result
        assert 2 not in result

    def test_and_two_complements(self):
        a = complement(set([1, 2]))
        b = complement(set([2, 3]))
        result = a & b
        # intersection of two complements = complement(union of exclusions)
        assert 1 not in result
        assert 2 not in result
        assert 3 not in result
        assert 4 in result

    def test_iand_with_set(self):
        c = complement(set([2, 3]))
        orig_id = id(c)
        c &= set([1, 2, 3, 4])
        assert id(c) == orig_id
        assert 1 in c
        assert 4 in c
        assert 2 not in c

    def test_or_with_set(self):
        c = complement(set([1, 2]))
        result = c | set([3])
        assert 3 not in result
        assert 1 not in result
        assert 4 in result

    def test_ror_with_set(self):
        c = complement(set([1, 2]))
        result = set([3]) | c
        assert 3 not in result
        assert 1 not in result
        assert 4 in result

    def test_or_two_complements(self):
        a = complement(set([1]))
        b = complement(set([2]))
        result = a | b
        assert 1 not in result
        assert 2 not in result
        assert 3 in result

    def test_ior_with_set(self):
        c = complement(set([1, 2, 3]))
        c |= set([2])
        assert 1 not in c
        assert 3 not in c
        assert 2 in c
        assert 4 in c

    def test_xor_with_set(self):
        c = complement(set([1, 2]))
        result = c ^ set([2, 3])
        assert 3 not in result
        assert 1 not in result
        assert 2 not in result

    def test_xor_two_complements(self):
        a = complement(set([1, 2]))
        b = complement(set([2, 3]))
        result = a ^ b
        # result is a normal (included) set
        assert isset(result)
        assert 1 in result
        assert 3 in result
        assert 2 not in result

    def test_rsub_set_minus_complement(self):
        c = complement(set([2, 3]))
        result = set([1, 2, 3, 4]) - c
        assert 2 in result
        assert 3 in result
        assert 1 not in result

    def test_sub_with_set(self):
        c = complement(set([1, 2]))
        result = c - set([3])
        # complement of {1,2} minus {3} = complement({1,2,3})
        assert 1 not in result
        assert 2 not in result
        assert 3 not in result
        assert 4 in result

    def test_isub(self):
        c = complement(set([1, 2]))
        c -= set([3])
        assert 1 not in c
        assert 3 not in c
        assert 4 in c

    def test_eq_complement_and_set(self):
        c = complement(set([1, 2]))
        assert c != set([1, 2])
        assert c == set([1, 2]) is False

    def test_eq_two_complements(self):
        a = complement(set([1, 2]))
        b = complement(set([2, 1]))
        assert a == b

    def test_hash(self):
        a = complement(set([1, 2]))
        # hash uses included ^ excluded; ensure it doesn't raise
        assert isinstance(hash(a), int)

    def test_isdisjoint(self):
        c = complement(set([1, 2]))
        # complement contains everything but 1,2 - so disjoint from nonempty set?
        assert c.isdisjoint(set([1, 2])) is False
        c2 = complement(set([1, 2]))
        assert c2.isdisjoint(set()) is False

    def test_issubset(self):
        # complement({1,2}) subset of complement({1,2,3})?
        a = complement(set([1, 2]))
        b = complement(set([1, 2, 3]))
        # everything missing from b is missing from a
        assert a.issubset(b) is True

    def test_le_operator(self):
        a = complement(set([1, 2, 3]))
        b = complement(set([1, 2]))
        # a excludes more -> a <= b
        assert a <= b

    def test_lt_operator(self):
        a = complement(set([1, 2, 3]))
        b = complement(set([1, 2]))
        assert a < b

    def test_ge_operator(self):
        a = complement(set([1, 2]))
        b = complement(set([1, 2, 3]))
        assert a >= b

    def test_gt_operator(self):
        a = complement(set([1, 2]))
        b = complement(set([1, 2, 3]))
        assert a > b

    def test_issuperset(self):
        a = complement(set([1, 2]))
        b = complement(set([1, 2, 3]))
        assert a.issuperset(b) is True


class TestComplementMutation:
    def test_add_to_complement(self):
        c = complement(set([1, 2]))
        c.add(1)  # remove 1 from exclusions
        assert 1 in c
        assert 2 not in c

    def test_remove_from_complement(self):
        c = complement(set([1, 2]))
        c.remove(3)  # add 3 to exclusions
        assert 3 not in c
        assert 1 not in c
        assert 4 in c

    def test_pop_complemented_raises(self):
        c = complement(set([1, 2]))
        with pytest.raises(NotImplementedError):
            c.pop()

    def test_pop_included(self):
        c = complement(set([1]))  # excluded={1}
        c.add(1)  # now excluded is empty
        c2 = ~c  # included = {} (empty set)
        # c2 is included set (empty)
        with pytest.raises(KeyError):
            c2.pop()

    def test_update_with_list(self):
        c = complement(set([1, 2, 3]))
        c.update([2])
        assert 1 not in c
        assert 3 not in c
        assert 2 in c

    def test_discard_from_complement(self):
        c = complement(set([1, 2]))
        c.discard(3)
        assert 3 not in c
        assert 1 not in c
        assert 4 in c

    def test_symmetric_difference_update(self):
        c = complement(set([1, 2]))
        c.symmetric_difference_update(set([2, 3]))
        # - + branch: self._excluded |= inc -> excluded = {1,2} | {2,3} = {1,2,3}
        assert 1 not in c
        assert 2 not in c
        assert 3 not in c
        assert 4 in c


class TestComplementErrors:
    def test_and_with_non_set_returns_notimplemented(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c & [1, 2]

    def test_or_with_non_set_returns_notimplemented(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c | [1, 2]

    def test_intersection_method_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.intersection([1, 2])

    def test_union_method_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.union([1, 2])

    def test_difference_method_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.difference([1, 2])

    def test_symmetric_difference_method_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.symmetric_difference([1, 2])

    def test_symmetric_difference_update_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.symmetric_difference_update([1, 2])

    def test_isdisjoint_with_non_set_raises_typeerror(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.isdisjoint([1, 2])

    def test_issubset_with_non_set(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.issubset([1, 2])

    def test_issuperset_with_non_set(self):
        c = complement(set([1]))
        with pytest.raises(TypeError):
            c.issuperset([1, 2])


class TestComplementIncludedVariant:
    """Operate on complements that hold an 'included' set (~complement)."""

    def _included(self, items):
        return ~complement(set(items))

    def test_included_iter_len(self):
        c = self._included([1, 2, 3])
        assert list(c) == [1, 2, 3]
        assert len(c) == 3

    def test_included_bool(self):
        assert bool(self._included([1])) is True
        assert bool(self._included([])) is False

    def test_included_contains(self):
        c = self._included([1, 2])
        assert 1 in c
        assert 3 not in c

    def test_included_repr(self):
        c = self._included([1, 2])
        assert repr(c) == "complement(complement({1, 2}))" or \
            repr(c) == "complement(complement({2, 1}))"

    def test_included_and_set(self):
        c = self._included([1, 2, 3])
        result = c & set([2, 3, 4])
        assert isinstance(result, _ComplementSet)
        assert list(result) == [2, 3]

    def test_included_and_complement(self):
        c = self._included([1, 2, 3])
        comp = complement(set([2]))
        result = c & comp
        # + - branch: exc - included = {2} - {1,2,3} = {}
        assert list(result) == []

    def test_included_or_set(self):
        c = self._included([1, 2])
        result = c | set([3, 4])
        assert list(result) == [1, 2, 3, 4]

    def test_included_xor_set(self):
        c = self._included([1, 2, 3])
        result = c ^ set([2, 3, 4])
        assert list(result) == [1, 4]

    def test_included_sub_set(self):
        c = self._included([1, 2, 3])
        result = c - set([2])
        assert list(result) == [1, 3]

    def test_included_rsub_set(self):
        c = self._included([1, 2, 3])
        result = set([2, 3, 4]) - c
        assert isinstance(result, _ComplementSet)
        assert 2 not in result
        assert 3 not in result
        assert 4 in result

    def test_included_pop(self):
        c = self._included([1, 2, 3])
        popped = c.pop()
        assert popped in (1, 2, 3)
        assert len(c) == 2

    def test_included_add(self):
        c = self._included([1])
        c.add(2)
        assert list(c) == [1, 2]

    def test_included_remove(self):
        c = self._included([1, 2])
        c.remove(1)
        assert list(c) == [2]

    def test_included_iand_set(self):
        c = self._included([1, 2, 3])
        c &= set([2, 3, 4])
        assert list(c) == [2, 3]

    def test_included_iand_complement(self):
        c = self._included([1, 2, 3])
        c &= complement(set([2]))
        # + - branch: included -= exc (2), then swap
        # {1,2,3} - {2} = {1,3}, then included=None, excluded={1,3}
        assert 1 not in c
        assert 3 not in c
        assert 2 in c

    def test_included_ior_set(self):
        c = self._included([1])
        c |= set([2])
        assert list(c) == [1, 2]

    def test_included_ior_complement(self):
        c = self._included([1, 2])
        c |= complement(set([2, 3]))
        # + - branch: excluded = exc - included = {2,3} - {1,2} = {3}
        assert 1 in c
        assert 2 in c
        assert 3 not in c

    def test_included_ixor_set(self):
        c = self._included([1, 2])
        c ^= set([2, 3])
        assert list(c) == [1, 3]

    def test_included_isub_set(self):
        c = self._included([1, 2, 3])
        c -= set([2])
        assert list(c) == [1, 3]

    def test_included_isub_complement(self):
        c = self._included([1, 2, 3])
        c -= complement(set([2]))
        # + - branch: included &= exc -> {2}
        assert list(c) == [2]

    def test_included_isdisjoint(self):
        c = self._included([1, 2, 3])
        assert c.isdisjoint(set([4, 5])) is True
        assert c.isdisjoint(set([2, 4])) is False
        assert c.isdisjoint(complement(set([1, 2, 3]))) is True

    def test_included_le_set(self):
        c = self._included([1, 2])
        assert c <= set([1, 2, 3]) is True
        assert c <= set([1]) is False

    def test_included_le_complement(self):
        c = self._included([1, 2])
        assert c <= complement(set([3])) is True
        assert c <= complement(set([1])) is False

    def test_included_lt_set(self):
        c = self._included([1, 2])
        assert c < set([1, 2, 3]) is True
        assert c < set([1, 2]) is False

    def test_included_ge_set(self):
        c = self._included([1, 2, 3])
        assert c >= set([1, 2]) is True
        assert c >= set([1, 2, 3]) is True
        assert c >= set([1, 2, 3, 4]) is False

    def test_included_ge_complement(self):
        c = self._included([1, 2])
        assert c >= complement(set([1])) is False

    def test_included_gt_set(self):
        c = self._included([1, 2, 3])
        assert c > set([1, 2]) is True
        assert c > set([1, 2, 3]) is False

    def test_included_gt_complement(self):
        c = self._included([1, 2])
        assert c > complement(set([1])) is False

    def test_included_update_with_set(self):
        c = self._included([1])
        c.update(set([2, 3]))
        assert list(c) == [1, 2, 3]

    def test_included_update_with_complement(self):
        c = self._included([1, 2, 3])
        c.update(complement(set([2])))
        # + - branch: included &= exc -> {2}, then swap to complement
        # c becomes complement of {2}
        assert 1 not in c
        assert 3 not in c
        assert 2 in c

    def test_included_discard_with_set(self):
        c = self._included([1, 2, 3])
        c.discard(set([2]))
        assert list(c) == [1, 3]

    def test_included_discard_with_complement(self):
        c = self._included([1, 2, 3])
        c.discard(complement(set([2])))
        # + - branch: included &= exc -> {2}
        assert list(c) == [2]

    def test_included_complement_in_place(self):
        c = self._included([1, 2])
        c.complement()
        # now excluded = {1, 2}
        assert 1 not in c
        assert 3 in c

    def test_complemented_with_frozenset(self):
        c = complement(frozenset([1, 2]))
        inv = c.complemented()
        assert isinstance(inv, _ComplementSet)
        assert inv._included is not None


# helper used above (declared here to keep test file self-contained)
def isset(obj):
    return isinstance(obj, set)


# A few sanity checks using explicit private-class import
from boltons.setutils import _ComplementSet  # noqa: E402
