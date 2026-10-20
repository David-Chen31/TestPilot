import pickle
import time
import types

import pytest

import cachetools
from cachetools import (
    Cache,
    FIFOCache,
    LFUCache,
    LRUCache,
    RRCache,
    TTLCache,
    TLRUCache,
    cached,
    cachedmethod,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


class CountingCache(Cache):
    """Cache subclass implementing __missing__ by storing computed values."""

    def __init__(self, maxsize, getsizeof=None):
        Cache.__init__(self, maxsize, getsizeof)
        self.misses = 0

    def __missing__(self, key):
        self.misses += 1
        value = key * 2
        self[key] = value
        return value


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_init_negative_maxsize(self):
        with pytest.raises(ValueError):
            Cache(-1)

    def test_maxsize_zero(self):
        c = Cache(0)
        assert c.maxsize == 0
        assert c.currsize == 0
        with pytest.raises(ValueError):
            c["a"] = 1

    def test_set_get_contains(self):
        c = Cache(10)
        c["a"] = 1
        assert c["a"] == 1
        assert "a" in c
        assert "b" not in c
        assert len(c) == 1
        assert c.currsize == 1
        assert c.maxsize == 10

    def test_missing_key(self):
        c = Cache(1)
        with pytest.raises(KeyError):
            c["missing"]

    def test_getsizeof_default(self):
        assert Cache.getsizeof(object()) == 1

    def test_custom_getsizeof(self):
        c = Cache(10, getsizeof=lambda v: len(v))
        c["a"] = "xx"
        assert c.currsize == 2
        c["b"] = "yyy"
        assert c.currsize == 5
        assert len(c) == 2

    def test_getsizeof_negative(self):
        c = Cache(10, getsizeof=lambda v: -1)
        with pytest.raises(ValueError):
            c["a"] = 1

    def test_value_too_large(self):
        c = Cache(5)
        with pytest.raises(ValueError):
            c["a"] = 1
        c2 = Cache(5, getsizeof=lambda v: v)
        with pytest.raises(ValueError):
            c2["a"] = 10

    def test_overwrite(self):
        c = Cache(10)
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        assert len(c) == 1
        assert c.currsize == 1

    def test_eviction(self):
        c = Cache(2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert len(c) == 2
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_delitem(self):
        c = Cache(10)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert c.currsize == 0
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_missing(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            del c["x"]

    def test_iter(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        assert set(iter(c)) == {"a", "b"}

    def test_get(self):
        c = Cache(10)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("b", 42) == 42

    def test_pop(self):
        c = Cache(10)
        c["a"] = 1
        assert c.pop("a") == 1
        assert "a" not in c
        with pytest.raises(KeyError):
            c.pop("a")
        assert c.pop("a", 5) == 5

    def test_pop_no_default(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            c.pop("x")

    def test_setdefault(self):
        c = Cache(10)
        assert c.setdefault("a", 1) == 1
        assert c["a"] == 1
        assert c.setdefault("a", 2) == 1

    def test_clear(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0

    def test_repr(self):
        c = Cache(10)
        c["a"] = 1
        r = repr(c)
        assert "Cache" in r
        assert "maxsize=10" in r
        assert "currsize=1" in r

    def test_missing_subclass(self):
        c = CountingCache(10)
        assert c[3] == 6
        assert c.misses == 1
        assert c[3] == 6
        assert c.misses == 1

    def test_subclass_get_does_not_trigger_missing(self):
        c = CountingCache(10)
        assert c.get(1) is None
        assert c.misses == 0

    def test_getsizeof_is_static_getsizeof_with_none(self):
        c = Cache(10, getsizeof=None)
        # default size behavior
        c["x"] = "abc"
        assert c.currsize == 1


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_eviction_order(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["d"] = 4
        assert "a" not in c
        assert set(c) == {"b", "c", "d"}

    def test_overwrite_moves_to_end(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["a"] = 4  # a moves to end
        c["d"] = 5
        assert "b" not in c
        assert set(c) == {"a", "c", "d"}

    def test_popitem(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert set(c) == {"b"}
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = FIFOCache(3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        del c["a"]
        c["c"] = 3
        c["d"] = 4
        assert "b" not in c
        assert set(c) == {"c", "d"}


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_eviction_least_frequent(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]  # a used twice
        c["c"] = 3
        assert "b" not in c
        assert set(c) == {"a", "c"}

    def test_overwrite_increments(self):
        c = LFUCache(2)
        c["a"] = 1
        c["a"] = 2
        c["b"] = 3
        # a has higher count so b evicted on insertion of c
        c["c"] = 4
        assert set(c) == {"a", "c"}
        assert c["a"] == 2

    def test_popitem(self):
        c = LFUCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        _ = c["b"]
        _ = c["c"]
        _ = c["c"]
        k, v = c.popitem()
        assert k == "a"
        assert v == 1
        k, v = c.popitem()
        assert k == "b"
        k, v = c.popitem()
        assert k == "c"
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LFUCache(3)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LFUCache(3)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert len(c) == 0

    def test_delitem_missing(self):
        c = LFUCache(3)
        with pytest.raises(KeyError):
            del c["x"]

    def test_getsize_eviction(self):
        c = LFUCache(4, getsizeof=lambda v: v)
        c["a"] = 2
        c["b"] = 1
        c["c"] = 3  # total 6 -> evict a or b
        assert c.currsize <= 4


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_eviction_least_recently_used(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        c["c"] = 3
        assert "b" not in c
        assert set(c) == {"a", "c"}

    def test_overwrite_touches(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 3
        c["c"] = 4
        assert "b" not in c
        assert set(c) == {"a", "c"}

    def test_popitem(self):
        c = LRUCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        _ = c["a"]
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("c", 3)
        assert c.popitem() == ("a", 1)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LRUCache(3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LRUCache(3)
        c["a"] = 1
        c["b"] = 2
        del c["a"]
        c["c"] = 3
        c["d"] = 4
        assert "b" not in c
        assert set(c) == {"c", "d"}


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_choice_property(self):
        def choice(seq):
            return seq[0]

        c = RRCache(3, choice=choice)
        assert c.choice is choice

    def test_popitem(self):
        keys_used = []

        def choice(seq):
            keys_used.append(list(seq))
            return seq[0]

        c = RRCache(10, choice=choice)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "a" not in c
        assert len(c) == 1

    def test_popitem_empty(self):
        c = RRCache(3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_eviction(self):
        c = RRCache(2, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert set(c) == {"b", "c"}

    def test_delitem_updates_index(self):
        def choice(seq):
            return seq[-1] if seq else None

        c = RRCache(10, choice=choice)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["b"]
        # a and c remain, b removed; pop from end yields c
        assert c.popitem() == ("c", 3)
        assert c.popitem() == ("a", 1)

    def test_clear(self):
        c = RRCache(3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_overwrite(self):
        c = RRCache(3, choice=lambda seq: seq[0])
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        assert len(c) == 1


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

class TestTTLCache:
    def test_init(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        assert c.ttl == 5
        assert c.maxsize == 10
        assert c.currsize == 0

    def test_getitem_not_expired(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        assert c["a"] == 1

    def test_expiry(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_getitem_expired_calls_missing(self):
        calls = []

        class TC(TTLCache):
            def __missing__(self, key):
                calls.append(key)
                return "MISS"

        timer = Clock()
        c = TC(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert c["a"] == "MISS"
        assert calls == ["a"]

    def test_contains_expired(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        assert "a" in c
        timer.advance(5)
        assert "a" not in c

    def test_expire(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(2)
        c["b"] = 2
        timer.advance(3)
        expired = c.expire()
        assert expired == [("a", 1)]
        assert "a" not in c
        assert "b" in c
        assert c.currsize == 1

    def test_expire_explicit_time(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        expired = c.expire(0)
        assert expired == []
        expired = c.expire(10)
        assert expired == [("a", 1)]

    def test_popitem(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(1)
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem_skips_expired(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_clear(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_expired(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        with pytest.raises(KeyError):
            del c["a"]

    def test_len_expires(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.advance(5)
        assert len(c) == 0

    def test_currsize_expires(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert c.currsize == 0

    def test_iter_expires(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(2)
        c["b"] = 2
        timer.advance(3)
        assert list(c) == ["b"]

    def test_get_and_pop_with_timer(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.get("x") is None
        assert c.pop("a") == 1

    def test_setdefault(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        assert c.setdefault("a", 1) == 1
        assert c.setdefault("a", 2) == 1

    def test_repr_expires(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        r = repr(c)
        assert "TTLCache" in r
        assert "currsize=0" in r

    def test_eviction_oldest_first(self):
        timer = Clock()
        c = TTLCache(2, 100, timer=timer)
        c["a"] = 1
        timer.advance(1)
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert set(c) == {"b", "c"}

    def test_timer_nesting(self):
        timer = Clock()

        with c_timer(timer):
            pass

        c = TTLCache(10, 5, timer=timer)
        with c.timer as t1:
            timer.advance(10)
            with c.timer as t2:
                assert t1 == t2
        assert t1 == 0

    def test_timer_getattr(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        # _Timer forwards attribute lookups
        assert hasattr(c.timer, "t")
        assert c.timer.t == 0


def c_timer(timer):
    class _Ctx:
        def __enter__(self):
            return timer()

        def __exit__(self, *a):
            return False

    return _Ctx()


class TestTTLCachePickle:
    def test_pickle_roundtrip(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        data = pickle.dumps(c)
        c2 = pickle.loads(data)
        assert set(c2) == {"a", "b"}
        assert c2["a"] == 1
        assert c2.ttl == 5

    def test_pickle_expired_removed(self):
        timer = Clock()
        c = TTLCache(10, 5, timer=timer)
        c["a"] = 1
        timer.advance(2)
        c["b"] = 2
        timer.advance(5)
        data = pickle.dumps(c)
        c2 = pickle.loads(data)
        assert list(c2) == []


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

class TestTLRUCache:
    def test_init(self):
        timer = Clock()
        ttu = lambda k, v, now: now + 10
        c = TLRUCache(10, ttu, timer=timer)
        assert c.ttu is ttu

    def test_getitem(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        assert c["a"] == 1

    def test_expiry(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_per_key_ttu(self):
        timer = Clock()
        c = TLRUCache(
            10, lambda k, v, now: now + (1 if k == "short" else 100), timer=timer
        )
        c["short"] = 1
        c["long"] = 2
        timer.advance(1)
        assert "short" not in c
        assert "long" in c

    def test_already_expired_setitem_new_key(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now - 1, timer=timer)
        c["a"] = 1
        assert "a" not in c

    def test_already_expired_setitem_existing_key(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        assert c["a"] == 1

        def ttu(k, v, now):
            return now - 1

        c2 = TLRUCache(10, ttu, timer=timer)
        c2["a"] = 1
        assert "a" not in c2

    def test_overwrite_existing(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        assert len(c) == 1

    def test_expire(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        timer.advance(2)
        c["b"] = 2
        timer.advance(3)
        expired = c.expire()
        assert expired == [("a", 1)]
        assert "b" in c

    def test_expire_explicit_time(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        assert c.expire(0) == []
        assert c.expire(100) == [("a", 1)]

    def test_popitem(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        timer.advance(1)
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_expired(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        with pytest.raises(KeyError):
            del c["a"]

    def test_iter(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert set(c) == {"a", "b"}

    def test_iter_skips_removed(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 3  # marks old item removed
        assert set(c) == {"a", "b"}

    def test_len_expires(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert len(c) == 0

    def test_currsize_expires(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 5, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert c.currsize == 0

    def test_heap_cleanup(self):
        timer = Clock()
        c = TLRUCache(10, lambda k, v, now: now + 100, timer=timer)
        for i in range(10):
            c["a"] = i  # repeatedly mark removed items
        # after many overwrites, expire should clean up heap
        expired = c.expire()
        assert expired == []
        assert list(c) == ["a"]


# ---------------------------------------------------------------------------
# cached
# ---------------------------------------------------------------------------

class TestCached:
    def test_basic(self):
        c = Cache(10)
        calls = []

        @cached(c)
        def f(x):
            calls.append(x)
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls == [3]

    def test_info(self):
        c = Cache(10)

        @cached(c, info=True)
        def f(x):
            return x * 2

        info = f.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        f.cache_clear()
        info = f.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert info.currsize == 0

    def test_info_mapping(self):
        d = {}

        @cached(d, info=True)
        def f(x):
            return x

        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize is None
        assert info.currsize == 1

    def test_info_plain_dict_no_len(self):
        # a non-mapping, non-Cache object results in 0/0 maxsize/currsize;
        # use a mapping-like object without being a Mapping instance.
        class FakeCache:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __contains__(self, key):
                return key in self.data

            def __delitem__(self, key):
                del self.data[key]

        fc = FakeCache()

        @cached(fc, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.maxsize == 0
        assert info.currsize == 0

    def test_custom_key(self):
        c = Cache(10)

        @cached(c, key=lambda args, kwargs: args[0] % 2)
        def f(x):
            return x

        assert f(2) == 2
        assert f(4) == 2
        assert f(3) == 3

    def test_typed(self):
        c = Cache(10)

        @cached(c, key=cachetools.keys.hashkey, info=True)
        def f(x):
            return x

        f(1)
        f(1.0)
        info = f.cache_info()
        # hashkey(1) == 1, hashkey(1.0) == 1.0 -> different keys
        assert info.misses == 2

    def test_decorator_returns_wrapper(self):
        c = Cache(10)

        def f(x):
            return x * 2

        wrapped = cached(c)(f)
        assert wrapped(5) == 10
        assert wrapped(5) == 10


# ---------------------------------------------------------------------------
# cachedmethod
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        c = Cache(10)
        calls = []

        class Obj:
            @cachedmethod(lambda self: c)
            def method(self, x):
                calls.append(x)
                return x * 2

        o = Obj()
        assert o.method(3) == 6
        assert o.method(3) == 6
        assert calls == [3]

    def test_info(self):
        c = Cache(10)

        class Obj:
            @cachedmethod(lambda self: c, info=True)
            def method(self, x):
                return x

        o = Obj()
        o.method(1)
        o.method(1)
        info = o.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10

    def test_info_mapping(self):
        cache_map = {}

        class Obj:
            @cachedmethod(lambda self: cache_map, info=True)
            def method(self, x):
                return x

        o = Obj()
        o.method(1)
        o.method(1)
        info = o.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize is None

    def test_info_invalid_cache(self):
        class BadCache:
            def __getitem__(self, key):
                raise KeyError(key)

            def __setitem__(self, key, value):
                pass

        class Obj:
            @cachedmethod(lambda self: BadCache(), info=True)
            def method(self, x):
                return x

        o = Obj()
        with pytest.raises(TypeError):
            o.method.cache_info()

    def test_custom_key(self):
        c = Cache(10)

        class Obj:
            @cachedmethod(lambda self: c, key=lambda self, x: x % 2)
            def method(self, x):
                return x

        o = Obj()
        assert o.method(2) == 2
        assert o.method(4) == 2


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

class TestModule:
    def test_version(self):
        assert isinstance(cachetools.__version__, str)
        assert cachetools.__version__

    def test_all(self):
        for name in cachetools.__all__:
            assert hasattr(cachetools, name)
        assert set(cachetools.__all__) == {
            "Cache",
            "FIFOCache",
            "LFUCache",
            "LRUCache",
            "RRCache",
            "TLRUCache",
            "TTLCache",
            "cached",
            "cachedmethod",
        }
