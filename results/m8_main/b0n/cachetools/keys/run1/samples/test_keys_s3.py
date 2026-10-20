"""Tests for cachetools.keys."""

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
# _HashedTuple
# ---------------------------------------------------------------------------

class TestHashedTuple:
    def test_is_tuple_subclass(self):
        key = _HashedTuple((1, 2, 3))
        assert isinstance(key, tuple)
        assert key == (1, 2, 3)

    def test_hash_is_cached(self, monkeypatch):
        key = _HashedTuple((1, 2, 3))
        calls = []
        real_hash = tuple.__hash__

        def counting_hash(self):
            calls.append(1)
            return real_hash(self)

        monkeypatch.setattr(tuple, "__hash__", counting_hash)
        first = key.__hash__()
        second = key.__hash__()
        assert first == second == real_hash((1, 2, 3))
        assert len(calls) == 1

    def test_add_returns_hashed_tuple(self):
        key = _HashedTuple((1, 2))
        result = key + (3, 4)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3, 4)

    def test_radd_returns_hashed_tuple(self):
        key = _HashedTuple((3, 4))
        result = (1, 2) + key
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3, 4)

    def test_getstate(self):
        key = _HashedTuple((1, 2))
        key.__hash__()  # populate cache
        assert key.__getstate__() == {}

    def test_pickle_roundtrip(self):
        key = _HashedTuple((1, 2, 3))
        key.__hash__()
        restored = pickle.loads(pickle.dumps(key))
        assert restored == key
        assert isinstance(restored, _HashedTuple)


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

class TestHashkey:
    def test_no_args_no_kwargs(self):
        key = hashkey()
        assert isinstance(key, _HashedTuple)
        assert key == ()

    def test_positional_args_only(self):
        key = hashkey(1, 2, 3)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, 3)
        assert _kwmark not in key

    def test_kwargs_only(self):
        key = hashkey(a=1, b=2)
        assert key == (_kwmark, ("a", 1), ("b", 2))

    def test_kwargs_sorted(self):
        key1 = hashkey(b=2, a=1)
        key2 = hashkey(a=1, b=2)
        assert key1 == key2
        assert key1 == (_kwmark,) + (("a", 1), ("b", 2))

    def test_args_and_kwargs(self):
        key = hashkey(10, b=2, a=1)
        assert key == (10, _kwmark, ("a", 1), ("b", 2))

    def test_kwmark_separates_args_from_kwargs(self):
        assert hashkey("a", 1) != hashkey("a", a=1)
        assert hashkey(("a", 1)) != hashkey(a=1)

    def test_hashable(self):
        key = hashkey(1, 2, a=3)
        assert isinstance(hash(key), int)

    def test_equal_keys_equal_hashes(self):
        assert hashkey(1, a=2) == hashkey(1, a=2)
        assert hash(hashkey(1, a=2)) == hash(hashkey(1, a=2))

    def test_unhashable_raises(self):
        key = hashkey([1, 2, 3])
        with pytest.raises(TypeError):
            hash(key)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

class TestMethodkey:
    def test_ignores_self(self):
        class Obj:
            pass

        obj = Obj()
        other = Obj()
        assert methodkey(obj, 1, 2) == methodkey(other, 1, 2)
        assert methodkey(obj) == ()
        assert methodkey(obj, 1, 2) == hashkey(1, 2)

    def test_with_kwargs(self):
        obj = object()
        assert methodkey(obj, a=1) == hashkey(a=1)

    def test_returns_hashed_tuple(self):
        assert isinstance(methodkey(object(), 1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

class TestTypedkey:
    def test_no_args_no_kwargs(self):
        key = typedkey()
        assert isinstance(key, _HashedTuple)
        assert key == ()

    def test_positional_args_only(self):
        key = typedkey(1, 2)
        assert key == (1, 2, int, int)

    def test_kwargs_only(self):
        key = typedkey(a=1)
        assert key == (_kwmark, ("a", 1), int)

    def test_args_and_kwargs(self):
        key = typedkey(1, a="x")
        assert key == (1, _kwmark, ("a", "x"), str, int)

    def test_kwargs_sorted(self):
        assert typedkey(b=2, a=1) == typedkey(a=1, b=2)
        assert typedkey(b=2, a=1) == (_kwmark, ("a", 1), ("b", 2), int, int)

    def test_distinguishes_types(self):
        assert typedkey(1) != typedkey(1.0)
        assert typedkey(1) == typedkey(1)
        assert typedkey(a=1) != typedkey(a=1.0)

    def test_distinguishes_bool_and_int(self):
        assert typedkey(True) != typedkey(1)
        assert typedkey(True) == typedkey(True)

    def test_no_type_of_str(self):
        # types are only appended for argument values, not the keys of kwargs
        key = typedkey(a=1)
        assert key == (_kwmark, ("a", 1), int)

    def test_hashable(self):
        key = typedkey(1, a="x")
        assert isinstance(hash(key), int)

    def test_returns_hashed_tuple(self):
        assert isinstance(typedkey(1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

class TestTypedmethodkey:
    def test_ignores_self(self):
        obj1 = object()
        obj2 = object()
        assert typedmethodkey(obj1, 1) == typedmethodkey(obj2, 1)
        assert typedmethodkey(obj1, 1) == typedkey(1)

    def test_with_kwargs(self):
        obj = object()
        assert typedmethodkey(obj, a=1) == typedkey(a=1)

    def test_returns_hashed_tuple(self):
        assert isinstance(typedmethodkey(object(), 1), _HashedTuple)
