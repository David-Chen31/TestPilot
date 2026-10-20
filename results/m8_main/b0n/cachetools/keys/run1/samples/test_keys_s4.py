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


class TestHashedTuple:
    def test_is_tuple(self):
        ht = _HashedTuple((1, 2, 3))
        assert isinstance(ht, tuple)
        assert ht == (1, 2, 3)
        assert tuple(ht) == (1, 2, 3)

    def test_hash_caching(self):
        ht = _HashedTuple((1, 2, 3))
        first = hash(ht)
        assert first == hash((1, 2, 3))
        # After first hash, __hashvalue is set
        assert ht._HashedTuple__hashvalue == first
        # Second call returns cached value
        assert ht._HashedTuple__hashvalue == first

    def test_hash_cached_even_if_objects_change_hash(self):
        class H:
            def __init__(self):
                self.calls = 0
                self.value = 1

            def __hash__(self):
                self.calls += 1
                return self.value

            def __eq__(self, other):
                return self is other

        h = H()
        ht = _HashedTuple((h,))
        hash(ht)
        assert h.calls == 1
        hash(ht)
        assert h.calls == 1

    def test_add_returns_hashed_tuple(self):
        ht = _HashedTuple((1, 2))
        result = ht + (3,)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3)

    def test_radd_returns_hashed_tuple(self):
        ht = _HashedTuple((2, 3))
        result = (1,) + ht
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2, 3)

    def test_getstate(self):
        ht = _HashedTuple((1, 2))
        assert ht.__getstate__() == {}

    def test_pickle_roundtrip(self):
        ht = _HashedTuple((1, 2, 3))
        restored = pickle.loads(pickle.dumps(ht))
        assert isinstance(restored, _HashedTuple)
        assert restored == ht
        assert hash(restored) == hash(ht)


class TestHashkey:
    def test_no_args(self):
        key = hashkey()
        assert isinstance(key, _HashedTuple)
        assert key == ()

    def test_positional_args(self):
        key = hashkey(1, 2, 3)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, 3)

    def test_kwargs_only(self):
        key = hashkey(a=1, b=2)
        assert isinstance(key, _HashedTuple)
        assert key == _kwmark + (("a", 1), ("b", 2))

    def test_args_and_kwargs(self):
        key = hashkey(1, 2, a=3, b=4)
        assert key == (1, 2) + _kwmark + (("a", 3), ("b", 4))

    def test_kwargs_sorted(self):
        key1 = hashkey(b=2, a=1)
        key2 = hashkey(a=1, b=2)
        assert key1 == key2

    def test_kwargs_sorted_multiple(self):
        key = hashkey(c=3, a=1, b=2)
        assert key == _kwmark + (("a", 1), ("b", 2), ("c", 3))

    def test_no_args_no_kwargs_equality(self):
        assert hashkey() == hashkey()

    def test_equal_keys_equal_hash(self):
        assert hash(hashkey(1, 2, a=3)) == hash(hashkey(1, 2, a=3))

    def test_args_and_kwargs_differ(self):
        # (1, 2) with empty kwargs vs. no kwargs
        assert hashkey(1, 2) != hashkey(1, 2, a=3)

    def test_positional_kwmark_collision_avoided(self):
        # A positional argument equal to _kwmark should not be confused
        key = hashkey(_kwmark)
        assert key == (_kwmark,)

    def test_unhashable_raises(self):
        with pytest.raises(TypeError):
            hash(hashkey([1, 2]))

    def test_kwmark_identity_preserved_on_pickle(self):
        key = hashkey(1, a=2)
        restored = pickle.loads(pickle.dumps(key))
        assert restored == key

    def test_only_one_hash_per_element(self):
        class H:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 42

            def __eq__(self, other):
                return self is other

        h = H()
        key = hashkey(h)
        hash(key)
        hash(key)
        hash(key)
        assert h.calls == 1


class TestMethodkey:
    def test_drops_self(self):
        class C:
            pass

        obj = C()
        assert methodkey(obj, 1, 2) == hashkey(1, 2)
        assert methodkey(obj) == hashkey()

    def test_with_kwargs(self):
        class C:
            pass

        obj = C()
        assert methodkey(obj, a=1, b=2) == hashkey(a=1, b=2)

    def test_with_args_and_kwargs(self):
        class C:
            pass

        obj = C()
        assert methodkey(obj, 1, a=2) == hashkey(1, a=2)

    def test_returns_hashed_tuple(self):
        assert isinstance(methodkey(object(), 1), _HashedTuple)

    def test_different_self_same_key(self):
        a = object()
        b = object()
        assert methodkey(a, 1) == methodkey(b, 1)


class TestTypedkey:
    def test_no_args(self):
        key = typedkey()
        assert isinstance(key, _HashedTuple)
        assert key == ()

    def test_positional_args(self):
        key = typedkey(1, 2)
        assert isinstance(key, _HashedTuple)
        assert key == (1, 2, int, int)

    def test_kwargs_only(self):
        key = typedkey(a=1, b=2)
        assert key == _kwmark + (("a", 1), ("b", 2)) + (int, int)

    def test_args_and_kwargs(self):
        key = typedkey(1, a=2)
        assert key == (1,) + _kwmark + (("a", 2),) + (int,) + (int,)

    def test_kwargs_sorted(self):
        assert typedkey(b=2, a=1) == typedkey(a=1, b=2)

    def test_type_included_for_equal_values(self):
        assert typedkey(1) != typedkey(1.0)
        assert typedkey(True) != typedkey(1)

    def test_type_included_for_kwargs(self):
        assert typedkey(a=1) != typedkey(a=1.0)

    def test_same_values_same_types_equal(self):
        assert typedkey(1, 2, a=3) == typedkey(1, 2, a=3)

    def test_different_kwargs_value_types(self):
        k1 = typedkey(x=1)
        k2 = typedkey(x=1.0)
        assert k1 != k2

    def test_type_order_follows_args(self):
        key = typedkey(1, "a", 2.0, x=1, y="b")
        expected = (
            (1, "a", 2.0)
            + _kwmark
            + (("x", 1), ("y", "b"))
            + (int, str)  # kwarg value types sorted by key: x->int, y->str
            + (int, str, float)  # positional arg types
        )
        assert key == expected

    def test_unhashable_raises(self):
        with pytest.raises(TypeError):
            hash(typedkey([1, 2]))

    def test_only_one_hash_per_element(self):
        class H:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 7

            def __eq__(self, other):
                return self is other

        h = H()
        key = typedkey(h)
        hash(key)
        hash(key)
        assert h.calls == 1

    def test_type_only_attached_for_args(self):
        # No kwargs: only arg types appended
        key = typedkey(1, 2)
        assert key[-2:] == (int, int)
        assert len(key) == 4


class TestTypedmethodkey:
    def test_drops_self(self):
        class C:
            pass

        obj = C()
        assert typedmethodkey(obj, 1, 2) == typedkey(1, 2)
        assert typedmethodkey(obj) == typedkey()

    def test_with_kwargs(self):
        class C:
            pass

        obj = C()
        assert typedmethodkey(obj, a=1, b=2) == typedkey(a=1, b=2)

    def test_with_args_and_kwargs(self):
        class C:
            pass

        obj = C()
        assert typedmethodkey(obj, 1, a=2) == typedkey(1, a=2)

    def test_returns_hashed_tuple(self):
        assert isinstance(typedmethodkey(object(), 1), _HashedTuple)

    def test_types_differ(self):
        class C:
            pass

        obj = C()
        assert typedmethodkey(obj, 1) != typedmethodkey(obj, 1.0)


class TestKwmark:
    def test_is_single_element_tuple(self):
        assert isinstance(_kwmark, tuple)
        assert len(_kwmark) == 1
        assert _kwmark[0] is _HashedTuple

    def test_pickle_preserves_identity(self):
        restored = pickle.loads(pickle.dumps(_kwmark))
        assert restored is _kwmark

    def test_hashkey_uses_kwmark_only_with_kwargs(self):
        key_no_kwargs = hashkey(1, 2)
        key_kwargs = hashkey(1, 2, a=3)
        assert _kwmark not in key_no_kwargs
        assert _kwmark in key_kwargs
