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

def test_hashkey_no_args():
    key = hashkey()
    assert key == ()
    assert isinstance(key, tuple)


def test_hashkey_positional_only():
    key = hashkey(1, 2, 3)
    assert key == (1, 2, 3)
    assert isinstance(key, tuple)


def test_hashkey_kwargs_sorted():
    key = hashkey(b=2, a=1)
    assert key == (() + _kwmark + (("a", 1), ("b", 2)))
    # sentinel separates args from kwargs
    assert key[len(_kwmark) - 1] is _kwmark[0]


def test_hashkey_args_and_kwargs():
    key = hashkey(1, 2, c=3, b=4)
    assert key == ((1, 2) + _kwmark + (("b", 4), ("c", 3)))


def test_hashkey_kwarg_order_independent():
    assert hashkey(a=1, b=2) == hashkey(b=2, a=1)


def test_hashkey_args_vs_kwargs_differ():
    assert hashkey(1) != hashkey(a=1)
    assert hashkey(b=2, a=1) != hashkey(1, 2)


def test_hashkey_returns_hashed_tuple():
    key = hashkey(1, 2, 3)
    assert isinstance(key, _HashedTuple)


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

def test_methodkey_ignores_self():
    class Obj:
        pass

    obj = Obj()
    key = methodkey(obj, 1, 2)
    assert key == (1, 2)
    assert isinstance(key, _HashedTuple)


def test_methodkey_ignores_self_with_kwargs():
    key = methodkey("self", 1, b=2, a=3)
    assert key == ((1,) + _kwmark + (("a", 3), ("b", 2)))


def test_methodkey_same_as_hashkey_without_self():
    assert methodkey(object(), 1, 2) == hashkey(1, 2)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

def test_typedkey_no_args():
    key = typedkey()
    assert key == ()
    assert isinstance(key, _HashedTuple)


def test_typedkey_positional_only():
    key = typedkey(1, 2)
    assert key == (1, 2, int, int)


def test_typedkey_distinguishes_types_when_no_kwargs():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey("1") != typedkey(1)


def test_typedkey_kwargs_sorted():
    key = typedkey(b=2, a=1)
    assert key == (() + _kwmark + (("a", 1), ("b", 2)) + (int, int))


def test_typedkey_args_and_kwargs():
    key = typedkey(1, "x", c=3, b=2.0)
    sorted_kwargs = (("b", 2.0), ("c", 3))
    expected = (
        (1, "x")
        + _kwmark
        + sorted_kwargs
        + (float, int)   # types of kwargs values in sorted order
        + (int, str)     # types of positional args
    )
    assert key == expected


def test_typedkey_kwarg_order_independent():
    assert typedkey(a=1, b=2) == typedkey(b=2, a=1)


def test_typedkey_kwargs_types_only_for_kwargs_when_no_args():
    key = typedkey(x=1, y="a")
    assert key == (() + _kwmark + (("x", 1), ("y", "a")) + (int, str))


def test_typedkey_distinguishes_kwarg_types():
    assert typedkey(a=1) != typedkey(a=1.0)


def test_typedkey_returns_hashed_tuple():
    assert isinstance(typedkey(1, a=2), _HashedTuple)


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

def test_typedmethodkey_ignores_self():
    class Obj:
        pass

    obj = Obj()
    key = typedmethodkey(obj, 1, "a")
    assert key == (1, "a", int, str)


def test_typedmethodkey_with_kwargs():
    key = typedmethodkey("self", 1, b=2.0, a=3)
    assert key == ((1,) + _kwmark + (("a", 3), ("b", 2.0)) + (int, float) + (int,))


def test_typedmethodkey_same_as_typedkey_without_self():
    assert typedmethodkey(object(), 1, 2) == typedkey(1, 2)


# ---------------------------------------------------------------------------
# _HashedTuple behavior
# ---------------------------------------------------------------------------

def test_hashed_tuple_hash_cached():
    ht = _HashedTuple((1, 2, 3))
    first = hash(ht)
    assert hash(ht) == first
    # second call returns stored value
    assert hash(ht) == first


def test_hashed_tuple_equals_tuple():
    assert _HashedTuple((1, 2)) == (1, 2)
    assert hash(_HashedTuple((1, 2))) == hash((1, 2))


def test_hashed_tuple_add_returns_hashed_tuple():
    ht = _HashedTuple((1, 2))
    result = ht + (3,)
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_radd_returns_hashed_tuple():
    ht = _HashedTuple((2, 3))
    result = (1,) + ht
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_getstate_empty():
    ht = _HashedTuple((1, 2))
    assert ht.__getstate__() == {}


def test_hashed_tuple_pickle_roundtrip():
    ht = _HashedTuple((1, "a", 2.5))
    restored = pickle.loads(pickle.dumps(ht))
    assert isinstance(restored, _HashedTuple)
    assert restored == ht
    assert hash(restored) == hash(ht)


def test_kwmark_identity_preserved_by_pickle():
    mark = pickle.loads(pickle.dumps(_kwmark))
    assert mark == _kwmark


# ---------------------------------------------------------------------------
# error handling / unhashable
# ---------------------------------------------------------------------------

def test_hashkey_unhashable_arg_raises():
    with pytest.raises(TypeError):
        hashkey([1, 2])


def test_hashkey_unhashable_kwarg_raises():
    with pytest.raises(TypeError):
        hashkey(a=[1, 2])


def test_typedkey_unhashable_raises():
    with pytest.raises(TypeError):
        typedkey([1, 2])


def test_typedkey_unhashable_kwarg_raises():
    with pytest.raises(TypeError):
        typedkey(a=object())
