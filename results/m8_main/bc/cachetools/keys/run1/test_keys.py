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
# hashkey
# ---------------------------------------------------------------------------

def test_hashkey_no_kwargs_returns_hashed_tuple():
    key = hashkey(1, 2, 3)
    assert isinstance(key, _HashedTuple)
    assert key == (1, 2, 3)


def test_hashkey_empty():
    key = hashkey()
    assert isinstance(key, _HashedTuple)
    assert key == ()


def test_hashkey_with_kwargs_includes_marker():
    key = hashkey(1, b="x", a="y")
    # kwargs are sorted, marker separates args from kwargs
    assert key == (1,) + _kwmark + (("a", "y"), ("b", "x"))


def test_hashkey_kwargs_sorted():
    key1 = hashkey(a=1, b=2)
    key2 = hashkey(b=2, a=1)
    assert key1 == key2
    assert key1 == _kwmark + (("a", 1), ("b", 2))


def test_hashkey_only_kwargs():
    key = hashkey(x=5)
    assert key == _kwmark + (("x", 5),)


def test_hashkey_different_args_differ():
    assert hashkey(1, 2) != hashkey(2, 1)


def test_hashkey_kwargs_marker_not_equal_to_plain_args():
    # Positional-only call must differ from one with kwargs
    assert hashkey(1) != hashkey(a=1)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

def test_methodkey_ignores_self():
    class Obj:
        pass

    obj = Obj()
    assert methodkey(obj, 1, 2) == methodkey(object(), 1, 2)
    assert methodkey(obj, 1, 2) == hashkey(1, 2)


def test_methodkey_with_kwargs():
    assert methodkey(None, 1, a=2) == hashkey(1, a=2)


def test_methodkey_no_args():
    assert methodkey("self") == hashkey()


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

def test_typedkey_no_kwargs_includes_types():
    key = typedkey(1, 2)
    assert key == (1, 2, int, int)


def test_typedkey_empty():
    key = typedkey()
    assert key == ()


def test_typedkey_with_kwargs_sorts_kwargs_and_types():
    key = typedkey(1, b="x", a=2)
    # args, marker, sorted kwargs, then kwarg value types, then arg types
    assert key == (1,) + _kwmark + (("a", 2), ("b", "x")) + (int, str) + (int,)


def test_typedkey_distinguishes_types():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey(1) == typedkey(1)


def test_typedkey_kwargs_only():
    key = typedkey(a=1)
    assert key == _kwmark + (("a", 1),) + (int,)


def test_typedkey_kwargs_sorted():
    assert typedkey(a=1, b=2) == typedkey(b=2, a=1)


def test_typedkey_returns_hashed_tuple():
    assert isinstance(typedkey(1), _HashedTuple)
    assert isinstance(typedkey(a=1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

def test_typedmethodkey_ignores_self():
    class Obj:
        pass

    obj = Obj()
    assert typedmethodkey(obj, 1) == typedkey(1)
    assert typedmethodkey(obj, 1) == (1, int)


def test_typedmethodkey_with_kwargs():
    assert typedmethodkey(None, 1, a="x") == typedkey(1, a="x")


# ---------------------------------------------------------------------------
# _HashedTuple caching behavior
# ---------------------------------------------------------------------------

def test_hashedtuple_hash_cached():
    calls = []

    class Counting:
        def __hash__(self):
            calls.append(1)
            return 42

    ht = _HashedTuple((Counting(),))
    h1 = hash(ht)
    h2 = hash(ht)
    assert h1 == h2
    assert len(calls) == 1


def test_hashedtuple_add_returns_hashedtuple():
    ht = _HashedTuple((1, 2))
    result = ht + (3,)
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashedtuple_radd_returns_hashedtuple():
    ht = _HashedTuple((2, 3))
    result = (1,) + ht
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashedtuple_getstate_empty():
    ht = _HashedTuple((1, 2, 3))
    assert ht.__getstate__() == {}


# ---------------------------------------------------------------------------
# pickling
# ---------------------------------------------------------------------------

def test_kwmark_survives_pickle():
    data = pickle.dumps(_kwmark)
    restored = pickle.loads(data)
    assert restored == _kwmark


def test_hashkey_pickle_roundtrip():
    key = hashkey(1, a=2)
    restored = pickle.loads(pickle.dumps(key))
    assert restored == key
    assert hash(restored) == hash(key)


# ---------------------------------------------------------------------------
# hashability / error handling
# ---------------------------------------------------------------------------

def test_hashkey_unhashable_raises_on_hash():
    # Building the key is fine for some unhashable parts, but hashing fails.
    key = hashkey([1, 2])
    with pytest.raises(TypeError):
        hash(key)


def test_typedkey_unhashable_raises_on_hash():
    key = typedkey([1, 2])
    with pytest.raises(TypeError):
        hash(key)
