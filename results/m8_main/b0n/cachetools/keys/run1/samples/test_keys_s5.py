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


# --- basic return types ---------------------------------------------------


def test_hashkey_returns_hashed_tuple():
    key = hashkey(1, 2, 3)
    assert isinstance(key, _HashedTuple)
    assert key == (1, 2, 3)
    assert isinstance(key, tuple)


def test_hashkey_no_args():
    key = hashkey()
    assert isinstance(key, _HashedTuple)
    assert key == ()


def test_hashkey_single_arg():
    assert hashkey("a") == ("a",)


def test_hashkey_positional_only():
    assert hashkey(1, "x", None) == (1, "x", None)


# --- kwargs handling ------------------------------------------------------


def test_hashkey_with_kwargs():
    key = hashkey(1, b=2, a=3)
    # kwargs are sorted and separated from args by _kwmark
    assert key == (1, _kwmark, ("a", 3), ("b", 2))


def test_hashkey_kwargs_sorted():
    k1 = hashkey(a=1, b=2, c=3)
    k2 = hashkey(c=3, b=2, a=1)
    assert k1 == k2


def test_hashkey_no_kwargs_has_no_marker():
    assert _kwmark not in hashkey(1, 2, 3)
    assert _kwmark not in hashkey()


def test_hashkey_empty_kwargs_treated_as_no_kwargs():
    # explicit empty kwargs -> falsy -> no marker
    assert hashkey(1, 2, **{}) == (1, 2)


def test_hashkey_args_and_kwargs_distinct():
    assert hashkey(1, 2) != hashkey(1, b=2)
    assert hashkey(1, 2) == (1, 2)
    assert hashkey(1, b=2) == (1, _kwmark, ("b", 2))


def test_hashkey_kwarg_value_ordering_preserved():
    # values are not sorted, only items by key
    key = hashkey(z=1, a=2)
    assert key == (_kwmark, ("a", 2), ("z", 1))


# --- methodkey ------------------------------------------------------------


def test_methodkey_ignores_self():
    class C:
        pass

    obj = C()
    assert methodkey(obj, 1, 2) == hashkey(1, 2)
    assert methodkey(obj, 1, 2) == (1, 2)


def test_methodkey_with_kwargs():
    class C:
        pass

    obj = C()
    assert methodkey(obj, x=1) == hashkey(x=1)
    assert methodkey(obj, x=1) == (_kwmark, ("x", 1))


def test_methodkey_self_not_in_key():
    class C:
        pass

    a, b = C(), C()
    assert methodkey(a, 1) == methodkey(b, 1)


# --- typedkey -------------------------------------------------------------


def test_typedkey_no_args():
    assert typedkey() == ()


def test_typedkey_no_kwargs():
    assert typedkey(1, "x") == (1, "x", int, str)


def test_typedkey_distinguishes_types():
    assert typedkey(1) != typedkey("1")
    assert typedkey(1) == (1, int)
    assert typedkey("1") == ("1", str)


def test_typedkey_with_kwargs():
    key = typedkey(1, a=2)
    # args + marker + sorted kwargs, then types of kwarg values, then types of args
    assert key == (1, _kwmark, ("a", 2), int, int)


def test_typedkey_kwargs_sorted():
    assert typedkey(a=1, b=2) == typedkey(b=2, a=1)


def test_typedkey_kwargs_types_included():
    key = typedkey(a=1)
    assert key == (_kwmark, ("a", 1), int)


def test_typedkey_kwarg_key_type_not_included():
    # only kwarg *values* types are included; keys are strings
    key = typedkey(a=1)
    # result length: _kwmark, ("a", 1), int
    assert len(key) == 3


def test_typedkey_args_and_kwargs_types_order():
    key = typedkey(1, a="x")
    # args=(1), marker, ("a","x"), types of kwarg values=(str,), types of args=(int,)
    assert key == (1, _kwmark, ("a", "x"), str, int)


def test_typedkey_bool_vs_int():
    # bool is subclass of int but type() differs
    assert typedkey(True) == (True, bool)
    assert typedkey(1) == (1, int)
    assert typedkey(True) != typedkey(1)


# --- typedmethodkey -------------------------------------------------------


def test_typedmethodkey_ignores_self():
    class C:
        pass

    obj = C()
    assert typedmethodkey(obj, 1) == typedkey(1)
    assert typedmethodkey(obj, 1) == (1, int)


def test_typedmethodkey_with_kwargs():
    class C:
        pass

    obj = C()
    assert typedmethodkey(obj, a=1) == typedkey(a=1)


# --- _HashedTuple behavior ------------------------------------------------


def test_hashed_tuple_caches_hash():
    calls = []

    class HashCounter:
        def __init__(self, value):
            self.value = value
            self.count = 0

        def __hash__(self):
            self.count += 1
            return hash(self.value)

        def __eq__(self, other):
            return self.value == other.value

    hc = HashCounter(42)
    t = _HashedTuple((hc,))
    h1 = hash(t)
    h2 = hash(t)
    assert h1 == h2
    assert hc.count == 1


def test_hashed_tuple_add_returns_hashed_tuple():
    t = _HashedTuple((1, 2))
    result = t + (3,)
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_radd_returns_hashed_tuple():
    t = _HashedTuple((2, 3))
    result = (1,) + t
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_add_with_list():
    t = _HashedTuple((1,))
    result = t + [2, 3]
    assert isinstance(result, _HashedTuple)
    assert result == (1, 2, 3)


def test_hashed_tuple_getstate():
    t = _HashedTuple((1, 2))
    assert t.__getstate__() == {}


def test_hashed_tuple_pickle():
    t = _HashedTuple((1, 2, 3))
    restored = pickle.loads(pickle.dumps(t))
    assert isinstance(restored, _HashedTuple)
    assert restored == t


def test_kwmark_is_hashed_tuple_class():
    assert _kwmark == (_HashedTuple,)
    assert _kwmark[0] is _HashedTuple


def test_hashed_tuple_is_tuple():
    assert issubclass(_HashedTuple, tuple)


# --- equality across functions -------------------------------------------


def test_hashkey_vs_typedkey_differ():
    assert hashkey(1) != typedkey(1)
    assert hashkey() != typedkey()


def test_typedkey_chaining_uses_hashed_tuple():
    key = typedkey(1, a=2)
    assert isinstance(key, _HashedTuple)


def test_hashkey_and_methodkey_equivalent():
    class C:
        pass

    obj = C()
    assert hashkey(1, 2) == methodkey(obj, 1, 2)
    assert hashkey(a=1) == methodkey(obj, a=1)


def test_typedkey_and_typedmethodkey_equivalent():
    class C:
        pass

    obj = C()
    assert typedkey(1, 2) == typedmethodkey(obj, 1, 2)
    assert typedkey(a=1) == typedmethodkey(obj, a=1)


# --- hashing the keys themselves works -----------------------------------


def test_keys_are_hashable():
    assert hash(hashkey(1, a=2)) == hash(hashkey(1, a=2))
    assert hash(typedkey(1, a=2)) == hash(typedkey(1, a=2))
    assert isinstance(hash(hashkey()), int)
    assert isinstance(hash(typedkey()), int)


def test_keys_usable_as_dict_keys():
    d = {}
    d[hashkey(1, a=2)] = "value"
    assert d[hashkey(1, a=2)] == "value"


def test_unhashable_arg_raises():
    with pytest.raises(TypeError):
        hash(hashkey([1, 2]))


def test_unhashable_kwarg_value_raises():
    with pytest.raises(TypeError):
        hash(hashkey(a=[1, 2]))


def test_unhashable_arg_typedkey_raises():
    with pytest.raises(TypeError):
        hash(typedkey([1, 2]))
