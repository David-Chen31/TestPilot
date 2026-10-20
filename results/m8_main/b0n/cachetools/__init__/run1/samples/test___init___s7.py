"""Unit tests for cachetools public API: Cache, FIFOCache, LFUCache,
LRUCache, RRCache, TTLCache, TLRUCache, cached, cachedmethod.
"""

import collections
import pickle

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
# Simple cache base class
# ---------------------------------------------------------------------------

class TestCache:
    def test_maxsize_validation(self):
        with pytest.raises(ValueError):
            Cache(-1)
        Cache(0)  # zero is allowed

    def test_basic_setitem_getitem(self):
        c = Cache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        assert c["a"] == 1
        assert c["b"] == 2
        assert len(c) == 2
        assert "a" in c
        assert "z" not in c

    def test_missing_key(self):
        c = Cache(maxsize=2)
        with pytest.raises(KeyError):
            c["missing"]

    def test_get(self):
        c = Cache(maxsize=2)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.get("b") is None
        assert c.get("b", 99) == 99

    def test_pop(self):
        c = Cache(maxsize=2)
        c["a"] = 1
        assert c.pop("a") == 1
        assert "a" not in c
        with pytest.raises(KeyError):
            c.pop("a")
        assert c.pop("a", 42) == 42

    def test_setdefault(self):
        c = Cache(maxsize=2)
        assert c.setdefault("a", 5) == 5
        assert c["a"] == 5
        assert c.setdefault("a", 7) == 5

    def test_currsize_maxsize_properties(self):
        c = Cache(maxsize=10)
        assert c.maxsize == 10
        assert c.currsize == 0
        c["a"] = 1
        assert c.currsize == 1

    def test_getsizeof_default(self):
        assert Cache.getsizeof(object()) == 1

    def test_custom_getsizeof(self):
        c = Cache(maxsize=10, getsizeof=len)
        c["a"] = "hello"
        assert c.currsize == 5
        c["b"] = "xy"
        assert c.currsize == 7

    def test_value_too_large(self):
        c = Cache(maxsize=2)
        with pytest.raises(ValueError):
            c["big"] = "value"  # size 1, but maxsize 2, so this is allowed...

    def test_value_too_large_err(self):
        c = Cache(maxsize=2, getsizeof=len)
        with pytest.raises(ValueError):
            c["x"] = "abc"  # size 3 > 2

    def test_negative_size(self):
        c = Cache(maxsize=2, getsizeof=lambda v: -1)
        with pytest.raises(ValueError):
            c["x"] = 1

    def test_eviction(self):
        c = Cache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert len(c) == 2
        # 'a' was first/smallest by insertion-order override in base (popitem default)
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_repr(self):
        c = Cache(maxsize=5)
        c["a"] = 1
        r = repr(c)
        assert r.startswith("Cache(")
        assert "maxsize=5" in r
        assert "currsize=1" in r

    def test_clear(self):
        c = Cache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0

    def test_delitem(self):
        c = Cache(maxsize=5)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert c.currsize == 0
        with pytest.raises(KeyError):
            del c["a"]

    def test_iter(self):
        c = Cache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        assert list(c) == ["a", "b"]

    def test_updating_existing_key_size(self):
        c = Cache(maxsize=10, getsizeof=len)
        c["a"] = "hello"  # 5
        c["a"] = "hi"     # 2
        assert c.currsize == 2


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_fifo_eviction(self):
        c = FIFOCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_update_moves_to_end(self):
        c = FIFOCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10  # moves a to end
        c["c"] = 3   # should evict b now
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_popitem(self):
        c = FIFOCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = FIFOCache(maxsize=5)
        c["a"] = 1
        del c["a"]
        assert "a" not in c

    def test_clear(self):
        c = FIFOCache(maxsize=5)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_getsizeof(self):
        c = FIFOCache(maxsize=10, getsizeof=len)
        c["a"] = "hello"
        assert c.currsize == 5


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_lru_eviction(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]  # touch a
        c["c"] = 3  # should evict b
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_getitem_touches(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        assert c["a"] == 1
        c["c"] = 3
        assert "a" in c
        assert "b" not in c

    def test_popitem(self):
        c = LRUCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        c["a"]  # touch a
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("a", 1)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LRUCache(maxsize=5)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LRUCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        del c["a"]
        assert "a" not in c
        assert list(c) == ["b"]

    def test_missing_returns_via_missing(self):
        class MyCache(LRUCache):
            def __missing__(self, key):
                return "miss"

        c = MyCache(maxsize=2)
        assert c["x"] == "miss"
        # missing should not be added to order
        assert len(c) == 0


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_lfu_eviction(self):
        c = LFUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]  # a count 2
        c["c"] = 3  # should evict b (count 1)
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_getitem_touches(self):
        c = LFUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]
        c["a"]
        c["c"] = 3
        assert "a" in c
        assert "b" not in c

    def test_popitem(self):
        c = LFUCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        c["a"]
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("a", 1)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LFUCache(maxsize=5)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LFUCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        del c["a"]
        assert "a" not in c

    def test_missing_no_touch(self):
        class MyCache(LFUCache):
            def __missing__(self, key):
                return "miss"

        c = MyCache(maxsize=2)
        assert c["x"] == "miss"
        assert len(c) == 0


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_choice_property(self):
        def chooser(seq):
            return seq[0]

        c = RRCache(maxsize=2, choice=chooser)
        assert c.choice is chooser

    def test_random_replacement(self):
        c = RRCache(maxsize=2, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # evicts seq[0] == "a"
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_popitem(self):
        c = RRCache(maxsize=5, choice=lambda seq: seq[-1])
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("a", 1)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem_index_rebalance(self):
        c = RRCache(maxsize=5, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["a"]
        # b or c should still be reachable
        assert "b" in c
        assert "c" in c
        assert "a" not in c

    def test_clear(self):
        c = RRCache(maxsize=5)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_getsizeof(self):
        c = RRCache(maxsize=10, getsizeof=len, choice=lambda seq: seq[0])
        c["a"] = "hello"
        assert c.currsize == 5


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

class FakeTimer:
    def __init__(self, t=0):
        self.time = t

    def __call__(self):
        return self.time

    def advance(self, dt):
        self.time += dt

    def __getattr__(self, name):
        raise AttributeError(name)


class TestTTLCache:
    def test_basic(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        assert c["a"] == 1
        assert c.ttl == 10

    def test_expiry(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert c["a"] == 1
        timer.advance(6)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_len_expires(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        timer.advance(6)
        assert len(c) == 0

    def test_currsize_expires(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        assert c.currsize == 1
        timer.advance(6)
        assert c.currsize == 0

    def test_expire_returns_pairs(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.advance(6)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_popitem_skips_expired(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        timer.advance(6)
        c["b"] = 2
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_lru_ordering(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=3, ttl=100, timer=timer)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert c["a"] == 1
        c["d"] = 4  # evict b (LRU)
        assert "b" not in c
        assert "a" in c
        assert "c" in c
        assert "d" in c

    def test_clear(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        timer.advance(10)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem_expired_raises(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        timer.advance(10)
        # delitem still works for expired item but raises KeyError afterwards
        timer2 = FakeTimer(0)
        c2 = TTLCache(maxsize=5, ttl=5, timer=timer2)
        c2["a"] = 1
        timer2.advance(10)
        with pytest.raises(KeyError):
            del c2["a"]

    def test_get_and_pop(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.pop("a") == 1
        assert c.get("a", 42) == 42

    def test_pickle(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        # Reduction of timer strips callable? Just check link reduce exists
        link = TTLCache._Link("k", 5.0)
        r = link.__reduce__()
        assert r[0] is TTLCache._Link

    def test_setstate_rebuilds(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=100, timer=timer)
        c["a"] = 1
        c["b"] = 2
        state = c.__dict__.copy()
        c2 = TTLCache(maxsize=5, ttl=100, timer=timer)
        c2.__setstate__(state)
        assert c2["a"] == 1
        assert c2["b"] == 2

    def test_repr(self):
        timer = FakeTimer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        assert "maxsize=5" in repr(c)


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

class TestTLRUCache:
    @staticmethod
    def ttu(key, value, now):
        return now + 10

    def test_basic(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        assert c["a"] == 1
        assert c.ttu is self.ttu

    def test_expiry(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        timer.advance(5)
        assert c["a"] == 1
        timer.advance(6)
        assert "a" not in c

    def test_len_expires(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        timer.advance(11)
        assert len(c) == 0

    def test_expire_returns_pairs(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.advance(11)
        assert set(c.expire()) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_already_expired_item_not_stored(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, now: now, timer=timer)
        c["a"] = 1
        assert "a" not in c
        assert len(c) == 0

    def test_lru_ordering(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=3, ttu=self.ttu, timer=timer)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert c["a"] == 1
        c["d"] = 4
        assert "b" not in c
        assert "a" in c
        assert "c" in c
        assert "d" in c

    def test_popitem(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_ttu_varies(self):
        timer = FakeTimer(0)

        def ttu(key, value, now):
            return now + key

        c = TLRUCache(maxsize=5, ttu=ttu, timer=timer)
        c[1] = "one"
        c[10] = "ten"
        timer.advance(2)
        assert 1 not in c
        assert 10 in c
        timer.advance(10)
        assert 10 not in c

    def test_get_and_pop(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.pop("a") == 1
        assert c.get("a", 42) == 42

    def test_heap_cleanup(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=100, ttu=lambda k, v, now: now + 1000, timer=timer)
        for i in range(50):
            c[i] = i
        for i in range(50):
            c[i] = i * 2  # mark old items removed, triggering cleanup
        assert len(c) == 50
        assert c[0] == 0
        assert c[49] == 98

    def test_repr(self):
        timer = FakeTimer(0)
        c = TLRUCache(maxsize=5, ttu=self.ttu, timer=timer)
        c["a"] = 1
        assert "maxsize=5" in repr(c)


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------

class TestCached:
    def test_basic(self):
        cache = {}
        calls = []

        @cached(cache)
        def f(x):
            calls.append(x)
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls == [3]

    def test_cache_eviction(self):
        cache = LRUCache(maxsize=2)

        @cached(cache)
        def f(x):
            return x * 2

        f(1)
        f(2)
        f(3)
        assert 1 not in cache
        assert 2 in cache
        assert 3 in cache

    def test_info(self):
        cache = Cache(maxsize=10)

        @cached(cache, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.hits == 0
        assert info.misses == 1
        assert info.maxsize == 10
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        f.cache_clear()
        assert len(cache) == 0

    def test_info_mapping(self):
        cache = {}

        @cached(cache, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.maxsize is None
        assert info.currsize == 1

    def test_info_other(self):
        class NotACache:
            def __getitem__(self, k):
                raise KeyError(k)

            def __setitem__(self, k, v):
                pass

            def __contains__(self, k):
                return False

            def __delitem__(self, k):
                pass

            def __len__(self):
                return 0

        cache = NotACache()

        @cached(cache, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.maxsize == 0
        assert info.currsize == 0

    def test_custom_key(self):
        cache = {}
        calls = []

        @cached(cache, key=lambda args, kwargs: args[0] % 2)
        def f(x):
            calls.append(x)
            return x

        assert f(1) == 1
        assert f(3) == 1  # same key
        assert f(2) == 2
        assert calls == [1, 2]


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        class C:
            def __init__(self):
                self.calls = []
                self.cache = {}

            @cachedmethod(lambda self: self.cache)
            def f(self, x):
                self.calls.append(x)
                return x * 2

        c = C()
        assert c.f(3) == 6
        assert c.f(3) == 6
        assert c.calls == [3]

    def test_cache_eviction(self):
        class C:
            def __init__(self):
                self.cache = LRUCache(maxsize=2)

            @cachedmethod(lambda self: self.cache)
            def f(self, x):
                return x * 2

        c = C()
        c.f(1)
        c.f(2)
        c.f(3)
        assert 1 not in c.cache
        assert 3 in c.cache

    def test_info(self):
        class C:
            def __init__(self):
                self.cache = Cache(maxsize=10)

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        c = C()
        c.f(1)
        info = c.f.cache_info()
        assert info.hits == 0
        assert info.misses == 1
        assert info.maxsize == 10
        c.f(1)
        info = c.f.cache_info()
        assert info.hits == 1

    def test_info_mapping(self):
        class C:
            def __init__(self):
                self.cache = {}

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        c = C()
        c.f(1)
        info = c.f.cache_info()
        assert info.maxsize is None
        assert info.currsize == 1

    def test_info_invalid(self):
        class C:
            def __init__(self):
                self.cache = []

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        c = C()
        with pytest.raises(TypeError):
            c.f.cache_info()

    def test_custom_key(self):
        class C:
            def __init__(self):
                self.cache = {}
                self.calls = []

            @cachedmethod(lambda self: self.cache, key=lambda self, x: x % 2)
            def f(self, x):
                self.calls.append(x)
                return x

        c = C()
        assert c.f(1) == 1
        assert c.f(3) == 1
        assert c.f(2) == 2
        assert c.calls == [1, 2]


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

def test_all_exports():
    assert "Cache" in cachetools.__all__
    assert "cached" in cachetools.__all__
    assert "cachedmethod" in cachetools.__all__


def test_version():
    assert cachetools.__version__ == "7.2.4"
