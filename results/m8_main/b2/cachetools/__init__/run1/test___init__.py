import pickle
import random
import threading
import time

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


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_maxsize_negative(self):
        with pytest.raises(ValueError):
            Cache(-1)

    def test_maxsize_zero(self):
        c = Cache(0)
        assert c.maxsize == 0
        assert c.currsize == 0

    def test_basic_set_get(self):
        c = Cache(10)
        c["a"] = 1
        assert c["a"] == 1
        assert "a" in c
        assert len(c) == 1
        assert c.currsize == 1
        assert c.maxsize == 10

    def test_missing_raises(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            c["nope"]

    def test_getsizeof_default(self):
        c = Cache(10)
        assert c.getsizeof(object()) == 1
        assert Cache.getsizeof(object()) == 1

    def test_custom_getsizeof(self):
        c = Cache(10, getsizeof=len)
        c["a"] = "hello"
        assert c.currsize == 5

    def test_value_larger_than_maxsize_raises(self):
        # NOTE: base Cache.getsizeof returns 1 for any value, so we need a
        # custom getsizeof to exercise the "value too large" branch.
        c = Cache(2, getsizeof=len)
        with pytest.raises(ValueError):
            c["a"] = "abc"

    def test_value_equal_to_maxsize(self):
        c = Cache(3, getsizeof=len)
        c["a"] = "abc"
        assert c.currsize == 3
        assert c["a"] == "abc"

    def test_negative_size(self):
        c = Cache(10, getsizeof=lambda v: -1)
        with pytest.raises(ValueError):
            c["a"] = 1

    def test_overwrite_replaces_size(self):
        c = Cache(10, getsizeof=len)
        c["a"] = "hi"
        assert c.currsize == 2
        c["a"] = "hello"
        assert c.currsize == 5

    def test_overwrite_shrinking(self):
        c = Cache(10, getsizeof=len)
        c["a"] = "hello"
        c["a"] = "hi"
        assert c.currsize == 2

    def test_eviction(self):
        c = Cache(2)
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        c["c"] = 3
        assert "c" in c
        assert len(c) == 2

    def test_delitem(self):
        c = Cache(10)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert c.currsize == 0

    def test_delitem_missing(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            del c["missing"]

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
        assert c.pop("b", 99) == 99
        with pytest.raises(KeyError):
            c.pop("b")

    def test_setdefault(self):
        c = Cache(10)
        assert c.setdefault("a", 1) == 1
        assert c.setdefault("a", 5) == 1
        assert c["a"] == 1

    def test_setdefault_none(self):
        c = Cache(10)
        assert c.setdefault("a") is None
        assert c["a"] is None

    def test_clear(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0

    def test_iter(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        assert set(iter(c)) == {"a", "b"}

    def test_repr(self):
        c = Cache(10)
        c["a"] = 1
        r = repr(c)
        assert r.startswith("Cache(")
        assert "maxsize=10" in r
        assert "currsize=1" in r

    def test_currsize_tracking(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        assert c.currsize == 2
        c["a"] = 5
        assert c.currsize == 2


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_fifo_order(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # evicts "a"
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_overwrite_moves_to_end(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10  # "a" moves to end
        c["c"] = 3  # should evict "b"
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_popitem(self):
        c = FIFOCache(10)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = FIFOCache(10)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = FIFOCache(10)
        c["a"] = 1
        del c["a"]
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_lfu_eviction(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        # access "a" once -> count 2, "b" stays count 1
        _ = c["a"]
        c["c"] = 3  # evicts "b"
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_lfu_among_ties(self):
        # popitem removes an arbitrary element from the least-frequently-used
        # group; with a single tie group, exactly one of the two keys is gone.
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        item = c.popitem()
        assert item[0] in ("a", "b")
        assert len(c) == 1
        remaining = next(iter(c))
        assert remaining != item[0]

    def test_popitem_empty(self):
        c = LFUCache(10)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LFUCache(10)
        c["a"] = 1
        del c["a"]
        with pytest.raises(KeyError):
            c.popitem()

    def test_overwrite_touches(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        c["b"] = 20  # touch b
        c["c"] = 3  # evicts a (count 1)
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_clear(self):
        c = LFUCache(10)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_missing_not_touched(self):
        class MyLFU(LFUCache):
            def __missing__(self, key):
                return "miss"

        c = MyLFU(10)
        assert c["x"] == "miss"
        assert len(c) == 0


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_lru_eviction(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]  # make "a" most recent
        c["c"] = 3  # evicts "b"
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_popitem(self):
        c = LRUCache(10)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("a", 1)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LRUCache(10)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LRUCache(10)
        c["a"] = 1
        del c["a"]
        with pytest.raises(KeyError):
            c.popitem()

    def test_missing_not_touched(self):
        class MyLRU(LRUCache):
            def __missing__(self, key):
                return "miss"

        c = MyLRU(10)
        assert c["x"] == "miss"
        assert len(c) == 0


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_choice_property(self):
        def first(seq):
            return seq[0]

        c = RRCache(10, choice=first)
        assert c.choice is first

    def test_default_choice(self):
        c = RRCache(10)
        assert c.choice is random.choice

    def test_popitem_uses_choice(self):
        def first(seq):
            return seq[0]

        c = RRCache(10, choice=first)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "b" in c

    def test_popitem_empty(self):
        c = RRCache(10)
        with pytest.raises(KeyError):
            c.popitem()

    def test_eviction(self):
        def first(seq):
            return seq[0]

        c = RRCache(2, choice=first)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # will popitem until fits (evicts a)
        assert "c" in c
        assert len(c) == 2

    def test_delitem_keeps_keys_consistent(self):
        c = RRCache(10)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["a"]
        assert sorted(c) == ["b", "c"]
        assert c.pop("b") == 2

    def test_overwrite_keeps_key_single(self):
        c = RRCache(10)
        c["a"] = 1
        c["a"] = 2
        assert len(c) == 1
        assert c["a"] == 2

    def test_clear(self):
        c = RRCache(10)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

class TestTTLCache:
    def test_ttl_property(self):
        c = TTLCache(10, ttl=5, timer=_Clock())
        assert c.ttl == 5

    def test_expires(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        assert c["a"] == 1
        clock.advance(4)
        assert c["a"] == 1
        clock.advance(1)
        with pytest.raises(KeyError):
            _ = c["a"]

    def test_contains_respects_expiry(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        assert "a" in c
        clock.advance(5)
        assert "a" not in c

    def test_len_expires(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        c["b"] = 2
        clock.advance(5)
        assert len(c) == 0

    def test_expire_returns_items(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        expired = c.expire()
        assert expired == [("a", 1)]
        assert len(c) == 0

    def test_expire_with_explicit_time(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        assert c.expire(10) == [("a", 1)]
        assert len(c) == 0

    def test_lru_eviction(self):
        clock = _Clock()
        c = TTLCache(2, ttl=100, timer=clock)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        c["c"] = 3
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_popitem(self):
        clock = _Clock()
        c = TTLCache(10, ttl=100, timer=clock)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem_skips_expired(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_iter(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(3)
        c["b"] = 2
        assert list(c) == ["a", "b"]

    def test_iter_skips_expired(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        c["b"] = 2
        assert list(c) == ["b"]

    def test_clear(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        assert list(c) == []

    def test_delitem_expired_raises(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        with pytest.raises(KeyError):
            del c["a"]
        assert len(c) == 0

    def test_currsize_expires(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        assert c.currsize == 0

    def test_repr_expires(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        clock.advance(5)
        r = repr(c)
        assert "currsize=0" in r

    def test_pickle(self):
        c = TTLCache(10, ttl=1000, timer=time.monotonic)
        c["a"] = 1
        restored = pickle.loads(pickle.dumps(c))
        assert restored["a"] == 1

    def test_get_with_expiry(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        c["a"] = 1
        assert c.get("a") == 1
        clock.advance(5)
        assert c.get("a") is None

    def test_setdefault_with_expiry(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        assert c.setdefault("a", 1) == 1
        assert c.setdefault("a", 2) == 1

    def test_missing_hook(self):
        clock = _Clock()

        class MyTTC(TTLCache):
            def __missing__(self, key):
                return "miss"

        c = MyTTC(10, ttl=5, timer=clock)
        assert c["x"] == "miss"

    def test_timer_property(self):
        clock = _Clock()
        c = TTLCache(10, ttl=5, timer=clock)
        assert c.timer() == 0.0
        clock.advance(7)
        assert c.timer() == 7.0


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

class TestTLRUCache:
    def test_ttu_property(self):
        def ttu(key, value, time):
            return time + 10

        c = TLRUCache(10, ttu, timer=_Clock())
        assert c.ttu is ttu

    def test_basic(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 10

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        assert c["a"] == 1
        assert "a" in c
        clock.advance(9)
        assert c["a"] == 1
        clock.advance(1)
        assert "a" not in c
        with pytest.raises(KeyError):
            _ = c["a"]

    def test_already_expired_not_stored(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time  # already expired

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        assert "a" not in c
        assert len(c) == 0

    def test_updating_with_expired_removes(self):
        clock = _Clock()
        expire_keys = set()

        def ttu(key, value, time):
            if key in expire_keys:
                return time
            return time + 10

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        assert "a" in c
        expire_keys.add("a")
        c["a"] = 2
        assert "a" not in c

    def test_expire_returns_items(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 5

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        clock.advance(5)
        assert c.expire() == [("a", 1)]
        assert len(c) == 0

    def test_lru_eviction(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 1000

        c = TLRUCache(2, ttu, timer=clock)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        c["c"] = 3
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_popitem(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 1000

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_iter_skips_expired_and_removed(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 5

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        clock.advance(5)
        c["b"] = 2
        assert list(c) == ["b"]

    def test_clear(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 100

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        assert list(c) == []

    def test_delitem_expired_raises(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 5

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        clock.advance(5)
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_fresh(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 5

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert len(c) == 0

    def test_heap_cleanup(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 1000

        c = TLRUCache(100, ttu, timer=clock)
        for i in range(20):
            c[i] = i
        for i in range(20):
            c[i] = i  # mark old items removed
        expired = c.expire()
        assert expired == []
        assert len(c) == 20

    def test_currsize_expires(self):
        clock = _Clock()

        def ttu(key, value, time):
            return time + 5

        c = TLRUCache(10, ttu, timer=clock)
        c["a"] = 1
        clock.advance(5)
        assert c.currsize == 0

    def test_missing_hook(self):
        clock = _Clock()

        class MyTLRU(TLRUCache):
            def __missing__(self, key):
                return "miss"

        def ttu(key, value, time):
            return time + 10

        c = MyTLRU(10, ttu, timer=clock)
        assert c["x"] == "miss"


# ---------------------------------------------------------------------------
# cached
# ---------------------------------------------------------------------------

class TestCached:
    def test_basic_cached(self):
        calls = []

        @cached(Cache(10))
        def f(x):
            calls.append(x)
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls == [3]

    def test_cached_with_lock(self):
        lock = threading.RLock()

        @cached(Cache(10), lock=lock)
        def f(x):
            return x + 1

        assert f(1) == 2
        assert f(1) == 2

    def test_cached_with_condition(self):
        cond = threading.Condition()

        @cached(Cache(10), condition=cond)
        def f(x):
            return x * 3

        assert f(2) == 6
        assert f(2) == 6

    def test_cached_info_cache(self):
        cache = Cache(10)

        @cached(cache, info=True)
        def f(x):
            return x

        assert f(1) == 1
        assert f(1) == 1
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        assert info.currsize == 1

    def test_cached_info_mapping(self):
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

    def test_cached_info_other(self):
        # non-Mapping, non-Cache -> maxsize=0, currsize=0. Must still be a
        # mutable mapping for the cache itself, so use a dict subclass that
        # is not a Cache instance (already a Mapping though).  To exercise the
        # "else" branch, use an object that supports item assignment but is
        # not a Mapping/Cache, e.g. a custom dict-like object.
        class WeirdCache:
            def __init__(self):
                self._d = {}

            def __getitem__(self, k):
                return self._d[k]

            def __setitem__(self, k, v):
                self._d[k] = v

            def __contains__(self, k):
                return k in self._d

            def __len__(self):
                return len(self._d)

        obj = WeirdCache()

        @cached(obj, info=True)
        def f(x):
            return x

        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 0
        assert info.currsize == 0

    def test_cached_custom_key(self):
        cache = Cache(10)

        @cached(cache, key=lambda *args, **kwargs: args[0])
        def f(x):
            return x

        assert f("a") == "a"
        assert f("a") == "a"
        assert "a" in cache
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# cachedmethod
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        cache = Cache(10)
        calls = []

        class Obj:
            @cachedmethod(lambda self: cache)
            def method(self, x):
                calls.append(x)
                return x * 2

        o = Obj()
        assert o.method(2) == 4
        assert o.method(2) == 4
        assert calls == [2]

    def test_info_cache(self):
        cache = Cache(10)

        class Obj:
            @cachedmethod(lambda self: cache, info=True)
            def method(self, x):
                return x

        o = Obj()
        o.method(1)
        o.method(1)
        # cache_info is bound to the instance, taking no extra arguments
        info = o.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        assert info.currsize == 1

    def test_info_mapping(self):
        d = {}

        class Obj:
            @cachedmethod(lambda self: d, info=True)
            def method(self, x):
                return x

        o = Obj()
        o.method(1)
        o.method(1)
        info = o.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize is None
        assert info.currsize == 1

    def test_info_non_mapping_raises(self):
        class Obj:
            @cachedmethod(lambda self: object(), info=True)
            def method(self, x):
                return x

        o = Obj()
        with pytest.raises(TypeError):
            o.method(1)


# ---------------------------------------------------------------------------
# __all__ and __version__
# ---------------------------------------------------------------------------

class TestModuleAttributes:
    def test_all(self):
        expected = {
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
        assert set(cachetools.__all__) == expected

    def test_version(self):
        assert isinstance(cachetools.__version__, str)
        assert cachetools.__version__.count(".") == 2
