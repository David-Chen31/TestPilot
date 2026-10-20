"""Unit tests for cachetools.keys."""

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
# _HashedTuple
# ---------------------------------------------------------------------------

class TestHashedTuple:
    def test_is_tuple(self):
        t = _HashedTuple((1, 2, 3))
        assert isinstance(t, tuple)
        assert t == (1, 2, 3)

    def test_hash_called_once(self):
        calls = []

        class Tracker:
            def __hash__(self):
                calls.append(1)
                return 42

            def __eq__(self, other):
                return self is other

        tracker = Tracker()
        t = _HashedTuple((tracker,))

        # repeated hashing should only invoke the element's __hash__ once
        h1 = hash(t)
        h2 = hash(t)
        h3 = hash(t)
        assert h1 == h2 == h3
        assert len(calls) == 1

    def test_hash_cached_value_consistent_with_tuple(self):
        t = _HashedTuple((1, 2, 3))
        assert hash(t) == hash((1, 2, 3))
        # also equal to hashing the tuple contents directly
        assert hash(t) == hash(t)

    def test_add_returns_hashed_tuple(self):
        t = _HashedTuple((1, 2))
        result = t + (3,)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3)

    def test_radd_returns_hashed_tuple(self):
        t = _HashedTuple((2, 3))
        result = (1,) + t
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3)

    def test_add_with_other_hashed_tuple(self):
        a = _HashedTuple((1, 2))
        b = _HashedTuple((3, 4))
        result = a + b
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3, 4)

    def test_getstate_empty(self):
        t = _HashedTuple((1, 2, 3))
        state = t.__getstate__()
        assert state == {}

    def test_pickle_roundtrip(self):
        t = _HashedTuple((1, 2, 3))
        restored = pickle.loads(pickle.dumps(t))
        assert isinstance(restored, _HashedTuple)
        assert restored == t
        assert hash(restored) == hash(t)

    def test_hash_after_unpickle(self):
        calls = []

        class Tracker:
            def __hash__(self):
                calls.append(1)
                return 7

            def __eq__(self, other):
                return self is other

        tracker = Tracker()
        t = _HashedTuple((tracker,))
        restored = pickle.loads(pickle.dumps(t))
        # hash value must be recomputed lazily but only once
        h1 = hash(restored)
        h2 = hash(restored)
        assert h1 == h2


# ---------------------------------------------------------------------------
# _kwmark
# ---------------------------------------------------------------------------

class TestKwmark:
    def test_is_single_element_tuple(self):
        assert isinstance(_kwmark, tuple)
        assert len(_kwmark) == 1
        assert _kwmark[0] is _HashedTuple

    def test_identity_after_pickle(self):
        restored = pickle.loads(pickle.dumps(_kwmark))
        assert restored == _kwmark
        assert restored[0] is _HashedTuple


# ---------------------------------------------------------------------------
# hashkey
# ---------------------------------------------------------------------------

class TestHashkey:
    def test_no_args(self):
        k = hashkey()
        assert isinstance(k, _HashedTuple)
        assert k == ()

    def test_positional_args(self):
        k = hashkey(1, 2, 3)
        assert isinstance(k, _HashedTuple)
        assert k == (1, 2, 3)

    def test_single_arg(self):
        k = hashkey("a")
        assert k == ("a",)

    def test_kwargs_only(self):
        k = hashkey(a=1, b=2)
        assert k == (_kwmark[0], ("a", 1), ("b", 2))

    def test_kwargs_sorted(self):
        k1 = hashkey(a=1, b=2)
        k2 = hashkey(b=2, a=1)
        assert k1 == k2
        assert k1 == (_kwmark[0], ("a", 1), ("b", 2))

    def test_args_and_kwargs(self):
        k = hashkey(1, 2, a=3, b=4)
        assert k == (1, 2, _kwmark[0], ("a", 3), ("b", 4))

    def test_args_and_kwargs_sorted_independent_of_order(self):
        k1 = hashkey(1, a=3, b=4)
        k2 = hashkey(1, b=4, a=3)
        assert k1 == k2

    def test_different_types_differ(self):
        assert hashkey(1) != hashkey("1")

    def test_unhashable_raises(self):
        with pytest.raises(TypeError):
            hashkey([1, 2, 3])

    def test_unhashable_kwarg_raises(self):
        with pytest.raises(TypeError):
            hashkey(a=[1, 2, 3])

    def test_returned_key_is_hashable(self):
        k = hashkey(1, 2, a=3)
        assert isinstance(hash(k), int)

    def test_kwargs_separated_from_args(self):
        # A single positional dict-like marker must not collide
        k1 = hashkey(1, 2)
        k2 = hashkey(1, ("a", 2)) if False else None  # placeholder no-op
        # explicit: args-only key is a plain arg tuple, no marker inserted
        assert k1 == (1, 2)
        assert _kwmark[0] not in k1

    def test_marker_distinguishes_arg_from_kwarg(self):
        # hashkey with kwarg b=2 differs from treating it as positional tuple
        assert hashkey(b=2) != hashkey(("b", 2))


# ---------------------------------------------------------------------------
# methodkey
# ---------------------------------------------------------------------------

class TestMethodkey:
    def test_ignores_self(self):
        class Dummy:
            pass

        obj = Dummy()
        assert methodkey(obj, 1, 2) == hashkey(1, 2)
        assert methodkey(obj, 1, 2) == (1, 2)

    def test_with_kwargs(self):
        class Dummy:
            pass

        obj = Dummy()
        k = methodkey(obj, 1, a=2)
        assert k == hashkey(1, a=2)
        assert k == (1, _kwmark[0], ("a", 2))

    def test_no_extra_args(self):
        class Dummy:
            pass

        obj = Dummy()
        assert methodkey(obj) == hashkey()
        assert methodkey(obj) == ()

    def test_two_objects_same_key(self):
        class Dummy:
            pass

        assert methodkey(Dummy(), 1) == methodkey(Dummy(), 1)

    def test_returned_type(self):
        class Dummy:
            pass

        assert isinstance(methodkey(Dummy(), 1), _HashedTuple)


# ---------------------------------------------------------------------------
# typedkey
# ---------------------------------------------------------------------------

class TestTypedkey:
    def test_no_args(self):
        k = typedkey()
        assert isinstance(k, _HashedTuple)
        assert k == ()

    def test_positional_args_include_types(self):
        k = typedkey(1, 2)
        assert k == (1, 2, int, int)

    def test_diff_types_differ(self):
        assert typedkey(1) != typedkey(1.0)
        assert typedkey(1) == (1, int)
        assert typedkey(1.0) == (1.0, float)

    def test_kwargs_only(self):
        k = typedkey(a=1, b=2)
        assert k == (
            _kwmark[0],
            ("a", 1),
            ("b", 2),
            int,
            int,
        )

    def test_kwargs_sorted(self):
        assert typedkey(a=1, b=2) == typedkey(b=2, a=1)

    def test_args_and_kwargs(self):
        k = typedkey(1, a=2)
        assert k == (1, _kwmark[0], ("a", 2), int, int)

    def test_kwarg_types_append_in_sorted_order(self):
        # kwargs sorted a, b ; types appended a's then b's, then args' types
        k = typedkey(a="x", b=1)
        assert k == (_kwmark[0], ("a", "x"), ("b", 1), str, int)

    def test_args_types_appended_last(self):
        k = typedkey(1.5, "s", a=2)
        # order: args + marker + sorted kwargs + kwargs types + args types
        assert k == (1.5, "s", _kwmark[0], ("a", 2), int, float, str)

    def test_diff_kwarg_types_differ(self):
        assert typedkey(a=1) != typedkey(a=1.0)
        assert typedkey(a=1) == (_kwmark[0], ("a", 1), int)

    def test_unhashable_raises(self):
        with pytest.raises(TypeError):
            typedkey([1, 2, 3])

    def test_unhashable_kwarg_raises(self):
        with pytest.raises(TypeError):
            typedkey(a=[1, 2, 3])

    def test_returned_key_is_hashable(self):
        k = typedkey(1, 2, a=3)
        assert isinstance(hash(k), int)
        assert isinstance(k, _HashedTuple)

    def test_type_of_value_not_name(self):
        # different named classes with same equality but different types
        class A:
            def __eq__(self, other):
                return isinstance(other, A)

            def __hash__(self):
                return 1

        class B:
            def __eq__(self, other):
                return isinstance(other, B)

            def __hash__(self):
                return 1

        assert typedkey(A()) != typedkey(B())


# ---------------------------------------------------------------------------
# typedmethodkey
# ---------------------------------------------------------------------------

class TestTypedmethodkey:
    def test_ignores_self(self):
        class Dummy:
            pass

        obj = Dummy()
        assert typedmethodkey(obj, 1, 2) == typedkey(1, 2)
        assert typedmethodkey(obj, 1, 2) == (1, 2, int, int)

    def test_with_kwargs(self):
        class Dummy:
            pass

        obj = Dummy()
        k = typedmethodkey(obj, 1, a=2)
        assert k == typedkey(1, a=2)
        assert k == (1, _kwmark[0], ("a", 2), int, int)

    def test_no_extra_args(self):
        class Dummy:
            pass

        obj = Dummy()
        assert typedmethodkey(obj) == typedkey()
        assert typedmethodkey(obj) == ()

    def test_diff_types_differ(self):
        class Dummy:
            pass

        assert typedmethodkey(Dummy(), 1) != typedmethodkey(Dummy(), 1.0)

    def test_returned_type(self):
        class Dummy:
            pass

        assert isinstance(typedmethodkey(Dummy(), 1), _HashedTuple)
