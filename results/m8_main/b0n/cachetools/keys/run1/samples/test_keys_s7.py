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
# _HashedTuple behavior (used by public functions, worth covering directly)
# ---------------------------------------------------------------------------

class TestHashedTuple:
    def test_is_tuple_subclass(self):
        assert isinstance(_HashedTuple((1, 2)), tuple)

    def test_hashvalue_cached(self):
        t = _HashedTuple((1, 2, 3))
        # Access the mangled name to verify the cached value semantics
        assert t._HashedTuple__hashvalue is None
        h1 = hash(t)
        assert t._HashedTuple__hashvalue == h1
        h2 = hash(t)
        assert h1 == h2

    def test_hash_equals_tuple_hash(self):
        data = (1, "a", (2, 3))
        assert hash(_HashedTuple(data)) == hash(data)

    def test_add_returns_hashedtuple(self):
        t = _HashedTuple((1, 2))
        result = t + (3, 4)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3, 4)

    def test_radd_returns_hashedtuple(self):
        t = _HashedTuple((3, 4))
        result = (1, 2) + t
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3, 4)

    def test_getstate_empty(self):
        t = _HashedTuple((1, 2))
        assert t.__getstate__() == {}

    def test_pickle_roundtrip(self):
        t = _HashedTuple((1, 2, 3))
        loaded = pickle.loads(pickle.dumps(t))
        assert isinstance(loaded, _HashedTuple)
        assert loaded == t

    def test_pickle_resets_hashvalue(self):
        t = _HashedTuple((1, 2))
        hash(t)  # populate cache
        loaded = pickle.loads(pickle.dumps(t))
        assert loaded._HashedTuple__hashvalue is None


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

class TestHashkey:
    def test_empty(self):
        assert hashkey() == ()

    def test_positional(self):
        key = hashkey(1, 2, 3)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, 3)

    def test_kwargs_sorted(self):
        key = hashkey(a=1, b=2)
        assert key == (_kwmark, ("a", 1), ("b", 2))

    def test_kwargs_order_independent(self):
        assert hashkey(a=1, b=2) == hashkey(b=2, a=1)

    def test_args_and_kwargs(self):
        key = hashkey(1, 2, x=3, y=4)
        assert key == (1, 2, _kwmark, ("x", 3), ("y", 4))

    def test_kwmark_separates_positional_from_kwargs(self):
        # A positional arg equal to _kwmark must still produce a different key
        key = hashkey(_kwmark, a=1)
        assert key == (_kwmark, _kwmark, ("a", 1))

    def test_returns_hashedtuple(self):
        assert isinstance(hashkey(1, 2), _HashedTuple)
        assert isinstance(hashkey(a=1), _HashedTuple)

    def test_hashable(self):
        key = hashkey(1, [])  # tuple with list: unhashable element
        with pytest.raises(TypeError):
            hash(key)

    def test_equal_keys_hash_equal(self):
        k1 = hashkey(1, 2, a=3)
        k2 = hashkey(1, 2, a=3)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_distinct_keys_differ(self):
        assert hashkey(1) != hashkey(2)
        assert hashkey(a=1) != hashkey(a=2)
        assert hashkey(1) != hashkey(1, 2)

    def test_pickle_roundtrip(self):
        key = hashkey(1, 2, a=3)
        loaded = pickle.loads(pickle.dumps(key))
        assert loaded == key
        assert isinstance(loaded, _HashedTuple)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

class TestMethodkey:
    def test_ignores_self(self):
        # methodkey takes self as first parameter and drops it
        assert methodkey("self-val", 1, 2) == hashkey(1, 2)

    def test_no_args(self):
        assert methodkey("self-val") == hashkey()

    def test_kwargs(self):
        assert methodkey("self-val", a=1, b=2) == hashkey(a=1, b=2)

    def test_args_and_kwargs(self):
        assert methodkey("self-val", 10, x=1) == hashkey(10, x=1)

    def test_returns_hashedtuple(self):
        assert isinstance(methodkey("self-val", 1), _HashedTuple)

    def test_different_self_same_key(self):
        assert methodkey("a", 1) == methodkey("b", 1)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

class TestTypedkey:
    def test_empty(self):
        assert typedkey() == ()

    def test_positional_includes_types(self):
        key = typedkey(1, 2)
        assert key == (1, 2, int, int)

    def test_kwargs_includes_types(self):
        key = typedkey(a=1, b=2)
        assert key == (_kwmark, ("a", 1), ("b", 2), int, int)

    def test_args_and_kwargs(self):
        key = typedkey(1, x="s")
        assert key == (1, _kwmark, ("x", "s"), str, int)

    def test_kwargs_sorted(self):
        k1 = typedkey(a=1, b=2)
        k2 = typedkey(b=2, a=1)
        assert k1 == k2

    def test_type_distinguishes(self):
        assert typedkey(1) != typedkey(1.0)
        assert typedkey(1) != typedkey(True)

    def test_value_distinguishes(self):
        assert typedkey(1) != typedkey(2)

    def test_kwargs_order_types(self):
        # types follow the sorted kwargs order
        key = typedkey(b="s", a=1)
        assert key == (_kwmark, ("a", 1), ("b", "s"), int, str)

    def test_returns_hashedtuple(self):
        assert isinstance(typedkey(1), _HashedTuple)
        assert isinstance(typedkey(a=1), _HashedTuple)

    def test_hashable(self):
        key = typedkey(1, 2)
        hash(key)

    def test_unhashable_value_raises(self):
        key = typedkey([])
        with pytest.raises(TypeError):
            hash(key)

    def test_equal_keys_hash_equal(self):
        k1 = typedkey(1, a=2)
        k2 = typedkey(1, a=2)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_pickle_roundtrip(self):
        key = typedkey(1, a="s")
        loaded = pickle.loads(pickle.dumps(key))
        assert loaded == key


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

class TestTypedmethodkey:
    def test_ignores_self(self):
        assert typedmethodkey("self-val", 1, 2) == typedkey(1, 2)

    def test_no_args(self):
        assert typedmethodkey("self-val") == typedkey()

    def test_kwargs(self):
        assert typedmethodkey("self-val", a=1) == typedkey(a=1)

    def test_args_and_kwargs(self):
        assert typedmethodkey("self-val", 1, x="s") == typedkey(1, x="s")

    def test_returns_hashedtuple(self):
        assert isinstance(typedmethodkey("self-val", 1), _HashedTuple)

    def test_different_self_same_key(self):
        assert typedmethodkey("a", 1) == typedmethodkey("b", 1)


# ---------------------------------------------------------------------------
# Cross-function sanity
# ---------------------------------------------------------------------------

class TestCrossFunction:
    def test_hashkey_vs_typedkey(self):
        assert hashkey(1) != typedkey(1)
        # hashkey(1) == (1,), typedkey(1) == (1, int)
        assert hashkey(1) == (1,)
        assert typedkey(1) == (1, int)

    def test_method_equivalence(self):
        assert methodkey("s", 1, a=2) == hashkey(1, a=2)
        assert typedmethodkey("s", 1, a=2) == typedkey(1, a=2)
