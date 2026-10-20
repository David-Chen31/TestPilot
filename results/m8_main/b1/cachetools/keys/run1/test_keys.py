import pickle

import pytest

from cachetools.keys import hashkey, methodkey, typedkey, typedmethodkey
from cachetools.keys import _HashedTuple


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

def test_hashkey_no_args():
    key = hashkey()
    assert key == ()
    assert isinstance(key, _HashedTuple)


def test_hashkey_args_only():
    key = hashkey(1, 2, 3)
    assert key == (1, 2, 3)
    assert isinstance(key, _HashedTuple)


def test_hashkey_empty_kwargs_ignored():
    key = hashkey(1, 2)
    assert key == (1, 2)


def test_hashkey_kwargs_sorted():
    key = hashkey(b=2, a=1)
    assert key == (_HashedTuple,) + (("a", 1), ("b", 2))


def test_hashkey_kwargs_order_independent():
    assert hashkey(a=1, b=2) == hashkey(b=2, a=1)


def test_hashkey_args_and_kwargs():
    key = hashkey(1, 2, x=10, y=20)
    # args, then sentinel (the _HashedTuple class itself), then sorted kwargs
    assert key[0] == 1
    assert key[1] == 2
    assert key[2] is _HashedTuple
    assert key[3:] == (("x", 10), ("y", 20))


def test_hashkey_mixed_order_equality():
    assert hashkey(1, a=2, b=3) == hashkey(1, b=3, a=2)


def test_hashkey_returns_hashed_tuple():
    key = hashkey(1, 2)
    assert isinstance(key, tuple)
    assert isinstance(key, _HashedTuple)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

class _Sample:
    def method(self, *args, **kwargs):
        return methodkey(self, *args, **kwargs)


def test_methodkey_ignores_self():
    obj = _Sample()
    key = obj.method()
    assert key == ()
    assert isinstance(key, _HashedTuple)


def test_methodkey_ignores_self_with_args():
    obj = _Sample()
    key = obj.method(1, 2)
    assert key == (1, 2)


def test_methodkey_ignores_self_with_kwargs():
    obj = _Sample()
    key = obj.method(a=1)
    assert key == (_HashedTuple,) + (("a", 1),)


def test_methodkey_matches_hashkey_without_self():
    obj = _Sample()
    assert obj.method(1, b=2) == hashkey(1, b=2)


def test_methodkey_returns_hashed_tuple():
    obj = _Sample()
    assert isinstance(obj.method(1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

def test_typedkey_no_args():
    key = typedkey()
    assert key == ()
    assert isinstance(key, _HashedTuple)


def test_typedkey_args_only_adds_types():
    key = typedkey(1, 2.0)
    assert key == (1, 2.0, int, float)


def test_typedkey_empty_kwargs_adds_arg_types():
    key = typedkey(1)
    assert key == (1, int)


def test_typedkey_kwargs_adds_types():
    key = typedkey(a=1, b=2.0)
    assert key == (
        _HashedTuple,
        ("a", 1),
        ("b", 2.0),
        int,
        float,
    )


def test_typedkey_args_and_kwargs():
    key = typedkey(1, "x", a=2.0)
    assert key == (
        1,
        "x",
        _HashedTuple,
        ("a", 2.0),
        float,   # kwarg type
        int,     # arg type for 1
        str,     # arg type for "x"
    )


def test_typedkey_distinguishes_types():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey(True) != typedkey(1)


def test_typedkey_kwargs_order_independent():
    assert typedkey(a=1, b=2) == typedkey(b=2, a=1)


def test_typedkey_kwarg_type_distinguishes():
    assert typedkey(a=1) != typedkey(a=1.0)


def test_typedkey_returns_hashed_tuple():
    assert isinstance(typedkey(1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

class _TypedSample:
    def method(self, *args, **kwargs):
        return typedmethodkey(self, *args, **kwargs)


def test_typedmethodkey_ignores_self():
    obj = _TypedSample()
    key = obj.method()
    assert key == ()


def test_typedmethodkey_ignores_self_with_args():
    obj = _TypedSample()
    key = obj.method(1, 2.0)
    assert key == (1, 2.0, int, float)


def test_typedmethodkey_matches_typedkey():
    obj = _TypedSample()
    assert obj.method(1, a=2) == typedkey(1, a=2)


def test_typedmethodkey_returns_hashed_tuple():
    obj = _TypedSample()
    assert isinstance(obj.method(1), _HashedTuple)


# ---------------------------------------------------------------------------
# _HashedTuple behavior
# ---------------------------------------------------------------------------

def test_hashed_tuple_hash_consistency():
    t = _HashedTuple((1, 2, 3))
    assert hash(t) == hash(t)
    assert hash(t) == hash((1, 2, 3))


def test_hashed_tuple_hash_cached():
    t = _HashedTuple((1, 2, 3))
    first = hash(t)
    assert t._HashedTuple__hashvalue == first
    assert hash(t) == first


def test_hashed_tuple_add():
    t = _HashedTuple((1, 2))
    result = t + (3, 4)
    assert result == (1, 2, 3, 4)
    assert isinstance(result, _HashedTuple)


def test_hashed_tuple_radd():
    t = _HashedTuple((3, 4))
    result = (1, 2) + t
    assert result == (1, 2, 3, 4)
    assert isinstance(result, _HashedTuple)


def test_hashed_tuple_add_returns_new_instance():
    t = _HashedTuple((1, 2))
    result = t + (3,)
    assert result is not t


def test_hashed_tuple_getstate_empty():
    t = _HashedTuple((1, 2, 3))
    assert t.__getstate__() == {}


def test_hashed_tuple_pickle_roundtrip():
    t = _HashedTuple((1, 2, 3))
    assert hash(t) == hash(tuple((1, 2, 3)))
    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, _HashedTuple)
    assert restored == (1, 2, 3)


def test_hashed_tuple_unhashable_raises():
    t = _HashedTuple(([1, 2],))
    with pytest.raises(TypeError):
        hash(t)


# ---------------------------------------------------------------------------
# keys hashable / unhashable behavior
# ---------------------------------------------------------------------------

def test_hashkey_unhashable_arg_raises_on_hash():
    key = hashkey([1, 2])
    with pytest.raises(TypeError):
        hash(key)


def test_typedkey_unhashable_arg_raises_on_hash():
    key = typedkey([1, 2])
    with pytest.raises(TypeError):
        hash(key)
