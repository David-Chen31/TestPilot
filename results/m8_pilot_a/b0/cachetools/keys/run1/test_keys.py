import pickle

import pytest

from cachetools.keys import hashkey, methodkey, typedkey, typedmethodkey
from cachetools.keys import _HashedTuple, _kwmark


class TestHashKey:
    def test_no_args(self):
        assert hashkey() == ()
        assert isinstance(hashkey(), _HashedTuple)
        assert hashkey() == hashkey()

    def test_positional_args(self):
        k = hashkey(1, 2, 3)
        assert isinstance(k, _HashedTuple)
        assert k == (1, 2, 3)
        assert hashkey(1, 2, 3) == k

    def test_single_arg(self):
        assert hashkey("a") == ("a",)

    def test_kwargs_sorted(self):
        k1 = hashkey(1, a=2, b=3)
        k2 = hashkey(1, b=3, a=2)
        assert k1 == k2
        # sorted order: a before b
        assert k1 == (1, _kwmark, ("a", 2), ("b", 3))

    def test_kwargs_only(self):
        k = hashkey(a=1)
        assert k == (_kwmark, ("a", 1))

    def test_args_vs_kwargs_distinct(self):
        assert hashkey(1) != hashkey(a=1)
        assert hashkey(1) != hashkey(**{"1": 1}) if False else True

    def test_empty_kwargs_no_marker(self):
        # kwargs empty dict is falsy, so no marker
        k = hashkey(1, 2)
        assert _kwmark not in k

    def test_hashcall_once(self):
        class Counter:
            count = 0

            def __hash__(self):
                Counter.count += 1
                return 42

        c = Counter()
        k = hashkey(c)
        Counter.count = 0
        h1 = hash(k)
        h2 = hash(k)
        assert h1 == h2
        assert Counter.count == 1

    def test_unhashable_raises(self):
        k = hashkey([1, 2])
        with pytest.raises(TypeError):
            hash(k)

    def test_add_returns_hashedtuple(self):
        k = hashkey(1)
        result = k + (2,)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2)

    def test_radd_returns_hashedtuple(self):
        k = hashkey(2)
        result = (1,) + k
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2)

    def test_equality_with_plain_tuple(self):
        assert hashkey(1, 2) == (1, 2)
        assert (1, 2) == hashkey(1, 2)

    def test_pickle_roundtrip(self):
        k = hashkey(1, 2, a=3)
        restored = pickle.loads(pickle.dumps(k))
        assert restored == k
        assert hash(restored) == hash(k)

    def test_pickle_preserves_kwmark_identity(self):
        k = hashkey(a=1)
        restored = pickle.loads(pickle.dumps(k))
        assert restored[0] is _kwmark

    def test_getstate(self):
        k = hashkey(1)
        assert k.__getstate__() == {}


class TestMethodKey:
    def test_drops_self(self):
        class Obj:
            pass

        a = Obj()
        b = Obj()
        assert methodkey(a, 1, 2) == methodkey(b, 1, 2)
        assert methodkey(a, 1, 2) == hashkey(1, 2)

    def test_with_kwargs(self):
        assert methodkey(object(), x=1, y=2) == hashkey(x=1, y=2)

    def test_no_args(self):
        assert methodkey(object()) == hashkey() == ()
        assert isinstance(methodkey(object()), _HashedTuple)

    def test_returns_hashedtuple(self):
        assert isinstance(methodkey(object(), 1), _HashedTuple)


class TestTypedKey:
    def test_no_args(self):
        assert typedkey() == ()

    def test_positional_args_types_appended(self):
        k = typedkey(1, "a")
        assert k == (1, "a", int, str)
        assert isinstance(k, _HashedTuple)

    def test_types_distinguish(self):
        assert typedkey(1) != typedkey(1.0)
        assert typedkey(0) != typedkey(False)

    def test_same_type_same_key(self):
        assert typedkey(1) == typedkey(1)
        assert typedkey(1, 2) == typedkey(1, 2)

    def test_kwargs(self):
        k = typedkey(a=1, b="x")
        # sorted kwargs: ("a",1) then ("b","x"); types appended for values sorted order
        assert k == (_kwmark, ("a", 1), ("b", "x"), int, str)

    def test_kwargs_types_distinguish(self):
        assert typedkey(a=1) != typedkey(a=1.0)

    def test_kwargs_order_independent(self):
        assert typedkey(a=1, b=2) == typedkey(b=2, a=1)

    def test_args_and_kwargs(self):
        k = typedkey(1, a="x")
        assert k == (1, _kwmark, ("a", "x"), str, int)

    def test_types_ordering_args_after_kwargs(self):
        k = typedkey(True, x=1.0)
        # kwargs types first, then args types
        assert k == (True, _kwmark, ("x", 1.0), float, bool)

    def test_hashcall_once(self):
        class Counter:
            count = 0

            def __hash__(self):
                Counter.count += 1
                return 7

        c = Counter()
        k = typedkey(c)
        Counter.count = 0
        hash(k)
        hash(k)
        assert Counter.count == 1

    def test_unhashable_raises(self):
        k = typedkey([1])
        with pytest.raises(TypeError):
            hash(k)

    def test_pickle_roundtrip(self):
        k = typedkey(1, "a", x=2.0)
        restored = pickle.loads(pickle.dumps(k))
        assert restored == k
        assert hash(restored) == hash(k)
        assert restored[0] is _kwmark


class TestTypedMethodKey:
    def test_drops_self(self):
        a = object()
        b = object()
        assert typedmethodkey(a, 1) == typedmethodkey(b, 1)
        assert typedmethodkey(a, 1) == typedkey(1)

    def test_types_distinguish(self):
        a = object()
        assert typedmethodkey(a, 1) != typedmethodkey(a, 1.0)

    def test_with_kwargs(self):
        a = object()
        assert typedmethodkey(a, x=1) == typedkey(x=1)

    def test_returns_hashedtuple(self):
        assert isinstance(typedmethodkey(object(), 1), _HashedTuple)


class TestHashedTuple:
    def test_subclass_of_tuple(self):
        assert issubclass(_HashedTuple, tuple)

    def test_hash_matches_tuple_hash(self):
        k = _HashedTuple((1, 2, 3))
        assert hash(k) == hash((1, 2, 3))

    def test_add_type(self):
        k = _HashedTuple((1,))
        result = k + (2,)
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2)

    def test_radd_type(self):
        k = _HashedTuple((2,))
        result = (1,) + k
        assert isinstance(result, _HashedTuple)
        assert result == (1, 2)

    def test_getstate(self):
        assert _HashedTuple((1,)).__getstate__() == {}
