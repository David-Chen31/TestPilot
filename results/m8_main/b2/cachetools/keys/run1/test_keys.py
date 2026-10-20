"""Tests for cachetools.keys."""

import pickle

import pytest

from cachetools.keys import (
    hashkey,
    methodkey,
    typedkey,
    typedmethodkey,
    _HashedTuple,
    _kwmark,
)


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

def test_hashkey_no_args_no_kwargs():
    key = hashkey()
    assert isinstance(key, _HashedTuple)
    assert key == ()


def test_hashkey_args_only():
    key = hashkey(1, 2, 3)
    assert isinstance(key, _HashedTuple)
    assert key == (1, 2, 3)


def test_hashkey_kwargs_only():
    key = hashkey(a=1, b=2)
    assert isinstance(key, _HashedTuple)
    assert key == _kwmark + (("a", 1), ("b", 2))


def test_hashkey_args_and_kwargs():
    key = hashkey(1, 2, a=3, b=4)
    assert key == (1, 2) + _kwmark + (("a", 3), ("b", 4))


def test_hashkey_kwargs_sorted():
    key1 = hashkey(a=1, b=2, c=3)
    key2 = hashkey(c=3, b=2, a=1)
    assert key1 == key2


def test_hashkey_kwargs_sorted_into_kwmark():
    key = hashkey(b=2, a=1)
    # kwargs must be sorted by key name
    idx = key.index(_kwmark[0])
    assert key[idx + 1:] == (("a", 1), ("b", 2))


def test_hashkey_sentinel_separates_args_from_kwargs():
    key = hashkey((_HashedTuple,))
    key_with_kwargs = hashkey(_HashedTuple=1)
    assert key != key_with_kwargs


def test_hashkey_args_equal_to_no_kwargs_key():
    # args (1, 2) with no kwargs differs from args (1,) with kwarg 2=...
    assert hashkey(1, 2) != hashkey(1, x=2)


def test_hashkey_unhashable_raises():
    with pytest.raises(TypeError):
        hash(hashkey([1, 2]))


def test_hashkey_returns_hashed_tuple_type():
    assert isinstance(hashkey(1), _HashedTuple)
    assert isinstance(hashkey(a=1), _HashedTuple)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

def test_methodkey_ignores_self():
    class C:
        def m(self, *args, **kwargs):
            return methodkey(self, *args, **kwargs)

    obj = C()
    assert obj.m(1, 2) == hashkey(1, 2)
    assert obj.m(a=1) == hashkey(a=1)
    assert obj.m(1, a=2) == hashkey(1, a=2)


def test_methodkey_two_instances_same_key():
    class C:
        pass

    c1, c2 = C(), C()
    assert methodkey(c1, 1) == methodkey(c2, 1)


def test_methodkey_no_extra_args():
    assert methodkey(object()) == hashkey()


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

def test_typedkey_no_args_no_kwargs():
    key = typedkey()
    assert isinstance(key, _HashedTuple)
    assert key == ()


def test_typedkey_args_only():
    key = typedkey(1, 2)
    assert key == (1, 2, int, int)


def test_typedkey_args_distinct_types():
    key = typedkey(1, 1.0)
    assert key == (1, 1.0, int, float)


def test_typedkey_kwargs_only():
    key = typedkey(a=1, b=2.0)
    assert key == _kwmark + (("a", 1), ("b", 2.0)) + (int, float)


def test_typedkey_args_and_kwargs():
    key = typedkey(1, a=2.0)
    assert key == (1,) + _kwmark + (("a", 2.0),) + (float,) + (int,)


def test_typedkey_kwargs_sorted():
    key1 = typedkey(a=1, b=2)
    key2 = typedkey(b=2, a=1)
    assert key1 == key2


def test_typedkey_distinguishes_types():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey(True) != typedkey(1)
    assert typedkey("1") != typedkey(1)


def test_typedkey_same_typed_args_equal():
    assert typedkey(1, 2) == typedkey(1, 2)


def test_typedkey_unhashable_raises():
    with pytest.raises(TypeError):
        hash(typedkey([1]))


def test_typedkey_returns_hashed_tuple_type():
    assert isinstance(typedkey(1), _HashedTuple)
    assert isinstance(typedkey(a=1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

def test_typedmethodkey_ignores_self():
    class C:
        def m(self, *args, **kwargs):
            return typedmethodkey(self, *args, **kwargs)

    obj = C()
    assert obj.m(1, 2) == typedkey(1, 2)
    assert obj.m(a=1) == typedkey(a=1)
    assert obj.m(1, a=2.0) == typedkey(1, a=2.0)


def test_typedmethodkey_distinguishes_types():
    class C:
        pass

    c = C()
    assert typedmethodkey(c, 1) != typedmethodkey(c, 1.0)


def test_typedmethodkey_two_instances_same_key():
    class C:
        pass

    c1, c2 = C(), C()
    assert typedmethodkey(c1, 1) == typedmethodkey(c2, 1)


def test_typedmethodkey_no_extra_args():
    assert typedmethodkey(object()) == typedkey()


# ---------------------------------------------------------------------------
# _HashedTuple behavior
# ---------------------------------------------------------------------------

def test_hashed_tuple_hash_is_cached():
    ht = _HashedTuple((1, 2, 3))
    h1 = hash(ht)
    h2 = hash(ht)
    assert h1 == h2
    assert h1 == hash((1, 2, 3))


def test_hashed_tuple_hash_cached_after_first_call():
    # The hash value must be memoized in the instance so that repeated
    # calls do not recompute (and so that the stored value is not None).
    ht = _HashedTuple((1, 2, 3))
    first = hash(ht)
    # The private attribute should hold the cached hash after hashing.
    cached = ht._HashedTuple__hashvalue
    assert cached is not None
    assert cached == first
    # A second hash call must return the same value.
    assert hash(ht) == cached


def test_hashed_tuple_hash_cached_value_used():
    # Verify that the cached hash is actually used on subsequent calls,
    # by checking the stored value matches the returned hash.
    ht = _HashedTuple((10, 20, 30))
    h1 = hash(ht)
    stored = ht._HashedTuple__hashvalue
    assert stored == h1
    # stored must be an int (tuple.__hash__ returns an int)
    assert isinstance(stored, int)
    h2 = hash(ht)
    assert h2 == stored


def test_hashed_tuple_hash_uses_cached_value_on_second_call():
    # Distinguish the cached-hash path from the recompute path: pre-set
    # the cached hash to a sentinel and verify it is returned verbatim,
    # without recomputing tuple.__hash__.
    ht = _HashedTuple((1, 2, 3))
    real = hash(ht)  # prime the cache
    sentinel = real + 12345
    ht._HashedTuple__hashvalue = sentinel
    assert hash(ht) == sentinel
    # Sanity: restore would return the real value.
    ht._HashedTuple__hashvalue = real
    assert hash(ht) == real


def test_hashed_tuple_hash_cached_on_empty_tuple():
    # Empty _HashedTuple hash must be memoized too.
    ht = _HashedTuple(())
    assert hash(ht) == hash(())
    assert ht._HashedTuple__hashvalue == hash(())


def test_hashed_tuple_add_returns_hashed_tuple():
    ht = _HashedTuple((1, 2))
    result = ht + (3,)
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_radd_returns_hashed_tuple():
    ht = _HashedTuple((1, 2))
    result = (0,) + ht
    assert isinstance(result, _HashedTuple)
    assert result == (0, 1, 2)


def test_hashed_tuple_getstate_empty():
    ht = _HashedTuple((1, 2))
    assert ht.__getstate__() == {}


def test_hashed_tuple_pickle_roundtrip():
    ht = hashkey(1, 2, a="x")
    restored = pickle.loads(pickle.dumps(ht))
    assert isinstance(restored, _HashedTuple)
    assert restored == ht
    assert hash(restored) == hash(ht)


def test_kwmark_is_hashed_tuple_sentinel():
    assert isinstance(_kwmark, tuple)
    assert _kwmark[0] is _HashedTuple
    assert len(_kwmark) == 1
