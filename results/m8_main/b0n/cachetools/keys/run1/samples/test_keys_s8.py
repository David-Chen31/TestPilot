"""Unit tests for cachetools.keys."""

import pickle

import pytest

from cachetools.keys import (
    _HashedTuple,
    _kwmark,
    hashkey,
    methodkey,
    typedkey,
    typedmethodkey,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class HashCounter:
    """A hashable object that counts how many times it has been hashed."""

    def __init__(self, value):
        self.value = value
        self.hash_count = 0

    def __hash__(self):
        self.hash_count += 1
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, HashCounter) and self.value == other.value


# ---------------------------------------------------------------------------
# _HashedTuple
# ---------------------------------------------------------------------------

class TestHashedTuple:
    def test_is_tuple_subclass(self):
        t = _HashedTuple((1, 2, 3))
        assert isinstance(t, tuple)
        assert t == (1, 2, 3)

    def test_hash_cached(self):
        item = HashCounter("x")
        t = _HashedTuple((item,))
        h1 = hash(t)
        assert item.hash_count == 1
        h2 = hash(t)
        # The cached hash value is reused; the element is not re-hashed.
        assert item.hash_count == 1
        assert h1 == h2

    def test_hash_matches_tuple_hash(self):
        t = _HashedTuple((1, 2, 3))
        assert hash(t) == hash((1, 2, 3))

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

    def test_add_is_truple_equivalent(self):
        t = _HashedTuple((1,))
        assert (t + (2,)) == ((1,) + (2,))

    def test_getstate_empty(self):
        t = _HashedTuple((1, 2, 3))
        assert t.__getstate__() == {}

    def test_pickle_roundtrip(self):
        t = _HashedTuple((1, 2, 3))
        restored = pickle.loads(pickle.dumps(t))
        assert isinstance(restored, _HashedTuple)
        assert restored == t
        # Hash still works after unpickling (cached value reset to None).
        assert hash(restored) == hash(t)

    def test_hash_updates_hashvalue_after_unpickle(self):
        t = _HashedTuple((10,))
        restored = pickle.loads(pickle.dumps(t))
        h = hash(restored)
        assert h == hash((10,))


# ---------------------------------------------------------------------------
# _kwmark sentinel
# ---------------------------------------------------------------------------

class TestKwmark:
    def test_kwmark_is_tuple_containing_hashedtuple_class(self):
        assert _kwmark == (_HashedTuple,)
        assert _kwmark[0] is _HashedTuple

    def test_kwmark_pickle_identity(self):
        restored = pickle.loads(pickle.dumps(_kwmark))
        assert restored[0] is _HashedTuple


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

class TestHashkey:
    def test_no_kwargs_returns_hashedtuple(self):
        key = hashkey(1, 2, 3)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, 3)

    def test_no_args_no_kwargs(self):
        key = hashkey()
        assert isinstance(key, _HashedTuple)
        assert key == ()

    def test_with_kwargs(self):
        key = hashkey(1, 2, a=3, b=4)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, _kwmark, ("a", 3), ("b", 4))

    def test_kwargs_sorted(self):
        key1 = hashkey(b=4, a=3)
        key2 = hashkey(a=3, b=4)
        assert key1 == key2
        assert key1 == (_kwmark, ("a", 3), ("b", 4))

    def test_kwargs_only(self):
        key = hashkey(x=1)
        assert key == (_kwmark, ("x", 1))

    def test_args_and_kwargs_distinct(self):
        # Different argument/kwarg arrangement must produce different keys.
        assert hashkey(1, x=1) != hashkey(1, 1)

    def test_same_args_produce_equal_keys(self):
        assert hashkey(1, 2) == hashkey(1, 2)

    def test_different_args_produce_different_keys(self):
        assert hashkey(1, 2) != hashkey(1, 3)

    def test_type_sensitive_for_hash_equality(self):
        # hashkey is not typed; but different values should differ.
        assert hashkey(1) != hashkey("1")

    def test_hash_is_cached_via_hashedtuple(self):
        item = HashCounter("y")
        key = hashkey(item)
        hash(key)
        assert item.hash_count == 1
        hash(key)
        assert item.hash_count == 1

    def test_kwarg_values_hashed(self):
        item = HashCounter("z")
        key = hashkey(a=item)
        hash(key)
        assert item.hash_count == 1
        hash(key)
        assert item.hash_count == 1

    def test_kwmark_prevents_collision(self):
        # A single tuple argument equal to _kwmark-ish must not collide with
        # kwargs-only key.
        key_with_kwmark = hashkey("a", 1)
        key_kwargs = hashkey(a=1)
        assert key_with_kwmark != key_kwargs

    def test_unhashable_argument_raises(self):
        with pytest.raises(TypeError):
            hash(hashkey([1, 2, 3]))

    def test_unhashable_kwarg_raises(self):
        with pytest.raises(TypeError):
            hash(hashkey(a=[1, 2, 3]))

    def test_pickle_roundtrip(self):
        key = hashkey(1, 2, a=3)
        restored = pickle.loads(pickle.dumps(key))
        assert restored == key
        assert hash(restored) == hash(key)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

class TestMethodkey:
    def test_ignores_self(self):
        class Dummy:
            pass

        obj = Dummy()
        # methodkey's first positional is bound to `self` and excluded.
        assert methodkey(obj, 1, 2) == hashkey(1, 2)

    def test_kwargs(self):
        class Dummy:
            pass

        obj = Dummy()
        assert methodkey(obj, 1, a=2) == hashkey(1, a=2)

    def test_returns_hashedtuple(self):
        class Dummy:
            pass

        obj = Dummy()
        key = methodkey(obj, 5)
        assert isinstance(key, _HashedTuple)
        assert key == (5,)

    def test_self_not_in_key(self):
        class Dummy:
            pass

        obj1, obj2 = Dummy(), Dummy()
        assert methodkey(obj1, 1) == methodkey(obj2, 1)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

class TestTypedkey:
    def test_no_kwargs(self):
        key = typedkey(1, 2)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, int, int)

    def test_no_args_no_kwargs(self):
        key = typedkey()
        assert key == ()

    def test_with_kwargs(self):
        key = typedkey(1, a=2)
        assert isinstance(key, _HashedTuple)
        # args, _kwmark, sorted kwargs items, types of kwarg values, types of args
        assert key == (1, _kwmark, ("a", 2), int, int)

    def test_kwargs_sorted(self):
        key1 = typedkey(b=4, a=3)
        key2 = typedkey(a=3, b=4)
        assert key1 == key2

    def test_type_included_in_key(self):
        assert typedkey(1) != typedkey(True)

    def test_int_vs_float(self):
        assert typedkey(1) != typedkey(1.0)

    def test_typed_matches_for_same_types(self):
        assert typedkey(1, 2) == typedkey(1, 2)

    def test_type_difference_only(self):
        # Same value, different types, must yield different keys.
        assert typedkey(0) != typedkey(0.0)
        assert typedkey(0) != typedkey(False)

    def test_kwargs_type_matters(self):
        assert typedkey(a=1) != typedkey(a=1.0)

    def test_multiple_kwarg_types_order(self):
        key = typedkey(a=1, b="x")
        assert key == (_kwmark, ("a", 1), ("b", "x"), int, str)

    def test_args_types_at_end(self):
        key = typedkey("s", 1)
        assert key == ("s", 1, str, int)

    def test_hash_cached(self):
        item = HashCounter(1)
        key = typedkey(item)
        hash(key)
        count = item.hash_count
        hash(key)
        assert item.hash_count == count

    def test_unhashable_raises(self):
        with pytest.raises(TypeError):
            hash(typedkey([1]))

    def test_pickle_roundtrip(self):
        key = typedkey(1, a=2)
        restored = pickle.loads(pickle.dumps(key))
        assert restored == key
        assert hash(restored) == hash(key)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

class TestTypedmethodkey:
    def test_ignores_self(self):
        class Dummy:
            pass

        obj = Dummy()
        assert typedmethodkey(obj, 1) == typedkey(1)

    def test_kwargs(self):
        class Dummy:
            pass

        obj = Dummy()
        assert typedmethodkey(obj, a=1) == typedkey(a=1)

    def test_returns_hashedtuple(self):
        class Dummy:
            pass

        obj = Dummy()
        key = typedmethodkey(obj, 5)
        assert isinstance(key, _HashedTuple)
        assert key == (5, int)

    def test_self_not_in_key(self):
        class Dummy:
            pass

        obj1, obj2 = Dummy(), Dummy()
        assert typedmethodkey(obj1, 1) == typedmethodkey(obj2, 1)


# ---------------------------------------------------------------------------
# Cross-function behavior
# ---------------------------------------------------------------------------

class TestCrossFunction:
    def test_hashkey_vs_typedkey_different_for_bound_methods(self):
        # hashkey and typedkey differ when types are involved.
        assert hashkey(1) != typedkey(1)

    def test_all_keys_hashable(self):
        for factory in (hashkey, typedkey):
            for key in (
                factory(),
                factory(1, 2),
                factory(1, a=2),
                factory(a=1),
            ):
                assert isinstance(hash(key), int)

    def test_methodkey_equivalent_to_hashkey(self):
        class Dummy:
            pass

        obj = Dummy()
        assert methodkey(obj, 1, 2, a=3) == hashkey(1, 2, a=3)

    def test_typedmethodkey_equivalent_to_typedkey(self):
        class Dummy:
            pass

        obj = Dummy()
        assert typedmethodkey(obj, 1, 2, a=3) == typedkey(1, 2, a=3)
