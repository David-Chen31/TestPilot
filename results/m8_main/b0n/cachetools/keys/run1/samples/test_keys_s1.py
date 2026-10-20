"""Tests for cachetools.keys."""

import pickle

import pytest

import cachetools.keys as keys
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


def test_hashed_tuple_is_tuple():
    t = _HashedTuple((1, 2, 3))
    assert isinstance(t, tuple)
    assert tuple.__eq__(t, (1, 2, 3))
    assert t == (1, 2, 3)


def test_hashed_tuple_hash_cached():
    t = _HashedTuple((1, 2, 3))
    first = t.__hashvalue
    assert first is None
    h = hash(t)
    assert h == hash((1, 2, 3))
    assert t.__hashvalue == h
    # Second call returns the same cached value.
    assert hash(t) == h
    assert t.__hashvalue == h


def test_hashed_tuple_hash_equal_to_tuple():
    t = _HashedTuple(("a", "b"))
    assert hash(t) == hash(("a", "b"))


def test_hashed_tuple_add_returns_hashed_tuple():
    t = _HashedTuple((1, 2))
    result = t + (3, 4)
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3, 4)


def test_hashed_tuple_radd_returns_hashed_tuple():
    t = _HashedTuple((3, 4))
    result = (1, 2) + t
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3, 4)


def test_hashed_tuple_radd_list():
    t = _HashedTuple((3, 4))
    result = [1, 2] + t
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3, 4)


def test_hashed_tuple_pickle_roundtrip():
    t = _HashedTuple((1, "two", 3.0))
    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, _HashedTuple)
    assert restored == t
    assert hash(restored) == hash(t)
    # getstate returns empty dict
    assert t.__getstate__() == {}


def test_hashed_tuple_hash_unhashable():
    t = _HashedTuple(([1],))
    with pytest.raises(TypeError):
        hash(t)


def test_kwmark_identity_preserved_through_pickle():
    restored = pickle.loads(pickle.dumps(_kwmark))
    assert restored[0] is _HashedTuple


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------


def test_hashkey_no_args():
    k = hashkey()
    assert isinstance(k, _HashedTuple)
    assert k == ()
    assert _kwmark not in k


def test_hashkey_positional_only():
    k = hashkey(1, 2, 3)
    assert isinstance(k, _HashedTuple)
    assert k == (1, 2, 3)


def test_hashkey_with_kwargs():
    k = hashkey(1, b=2, a=3)
    # kwargs sorted by name
    assert k == (1, _kwmark, ("a", 3), ("b", 2))


def test_hashkey_kwargs_only():
    k = hashkey(x=1, y=2)
    assert k == (_kwmark, ("x", 1), ("y", 2))


def test_hashkey_kwargs_sorted_order():
    k1 = hashkey(a=1, b=2, c=3)
    k2 = hashkey(c=3, b=2, a=1)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_hashkey_differs_with_positional():
    assert hashkey(1) != hashkey(2)
    assert hashkey(1, 2) != hashkey(2, 1)


def test_hashkey_positional_vs_kwarg():
    # (1, 2) positional should not equal kwargs form
    assert hashkey(1, 2) != hashkey(a=1, b=2)


def test_hashkey_empty_kwargs_same_as_no_kwargs():
    # explicit empty kwargs means branch is not taken
    assert hashkey(1, 2) == hashkey(1, 2, **{})


def test_hashkey_hash_consistent():
    k = hashkey(1, 2, a=3)
    assert hash(k) == hash(k)


def test_hashkey_unhashable_arg():
    with pytest.raises(TypeError):
        hash(hashkey([1, 2]))


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------


def test_methodkey_ignores_self():
    class C:
        pass

    obj = C()
    k = methodkey(obj, 1, 2)
    # methodkey drops self
    assert k == (1, 2)
    assert isinstance(k, _HashedTuple)


def test_methodkey_with_kwargs_ignores_self():
    class C:
        pass

    obj = C()
    k = methodkey(obj, 1, a=2)
    assert k == (1, _kwmark, ("a", 2))


def test_methodkey_two_instances_same_key():
    class C:
        pass

    k1 = methodkey(C(), 1)
    k2 = methodkey(C(), 1)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_methodkey_no_args():
    k = methodkey(object())
    assert k == ()


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------


def test_typedkey_no_args():
    k = typedkey()
    assert isinstance(k, _HashedTuple)
    assert k == ()


def test_typedkey_positional_only():
    # No kwargs: key = args + types of args
    k = typedkey(1, 2)
    assert k == (1, 2, int, int)


def test_typedkey_positional_differs_by_type():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey(1) != typedkey(True)


def test_typedkey_same_value_same_type_equal():
    assert typedkey(1) == typedkey(1)
    assert hash(typedkey(1)) == hash(typedkey(1))


def test_typedkey_with_kwargs():
    k = typedkey(1, a=2)
    # args + _kwmark + sorted_kwargs + types of sorted kwarg values + types of args
    assert k == (1, _kwmark, ("a", 2), int, int)


def test_typedkey_kwargs_only():
    k = typedkey(a=2)
    assert k == (_kwmark, ("a", 2), int)


def test_typedkey_kwargs_sorted_order():
    k1 = typedkey(a=1, b=2)
    k2 = typedkey(b=2, a=1)
    assert k1 == k2


def test_typedkey_kwargs_differs_by_type():
    assert typedkey(a=1) != typedkey(a=1.0)


def test_typedkey_mixed_types():
    k = typedkey("x", 1, b=True, c=2.0)
    assert k == (
        "x",
        1,
        _kwmark,
        ("b", True),
        ("c", 2.0),
        bool,
        float,
        str,
        int,
    )


def test_typedkey_is_hashed_tuple_after_add():
    k = typedkey(1, 2)
    assert isinstance(k, _HashedTuple)


def test_typedkey_unhashable_arg():
    with pytest.raises(TypeError):
        hash(typedkey([1]))


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------


def test_typedmethodkey_ignores_self():
    class C:
        pass

    obj = C()
    k = typedmethodkey(obj, 1)
    assert k == (1, int)


def test_typedmethodkey_with_kwargs_ignores_self():
    class C:
        pass

    obj = C()
    k = typedmethodkey(obj, 1, a=2)
    assert k == (1, _kwmark, ("a", 2), int, int)


def test_typedmethodkey_two_instances_same_key():
    class C:
        pass

    k1 = typedmethodkey(C(), 1)
    k2 = typedmethodkey(C(), 1)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_typedmethodkey_differs_by_type():
    class C:
        pass

    assert typedmethodkey(C(), 1) != typedmethodkey(C(), 1.0)


def test_typedmethodkey_no_args():
    k = typedmethodkey(object())
    assert k == ()


# ---------------------------------------------------------------------------
# Cross-function sanity
# ---------------------------------------------------------------------------


def test_hashkey_vs_typedkey():
    # Differ when types distinguish equal-hashing values
    assert hashkey(1) != typedkey(1)
    assert hashkey(1) == (1,)
    assert typedkey(1) == (1, int)


def test_kwmark_in_key_only_with_kwargs():
    assert _kwmark not in hashkey(1, 2)
    assert _kwmark in hashkey(a=1)
    assert _kwmark not in typedkey(1, 2)
    assert _kwmark in typedkey(a=1)


def test_all_exports():
    assert set(keys.__all__) == {
        "hashkey",
        "methodkey",
        "typedkey",
        "typedmethodkey",
    }
