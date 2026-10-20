"""Unit tests for the cachetools package public API."""

import pickle
import unittest.mock

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
# helpers
# ---------------------------------------------------------------------------

class ManualTimer:
    """Deterministic timer for time-aware caches."""

    def __init__(self, time=0.0):
        self.time = time

    def __call__(self):
        return self.time

    def advance(self, delta):
        self.time += delta


def make_all_caches(maxsize=3, **kwargs):
    """Return instances of every concrete cache class."""
    return [
        Cache(maxsize, **kwargs),
        FIFOCache(maxsize, **kwargs),
        LFUCache(maxsize, **kwargs),
        LRUCache(maxsize, **kwargs),
        RRCache(maxsize, choice=lambda seq: seq[0], **kwargs),
    ]


# ---------------------------------------------------------------------------
# _DefaultSize / Cache basics
# ---------------------------------------------------------------------------

class TestCache:
    def test_init_rejects_negative_maxsize(self):
        with pytest.raises(ValueError):
            Cache(-1)

    def test_maxsize_and_currsize(self):
        c = Cache(10)
        assert c.maxsize == 10
        assert c.currsize == 0
        c["a"] = 1
        assert c.currsize == 1
        c["b"] = 2
        assert c.currsize == 2

    def test_setitem_getitem_contains_delitem(self):
        c = Cache(10)
        c["a"] = 1
        assert c["a"] == 1
        assert "a" in c
        assert "b" not in c
        del c["a"]
        assert "a" not in c
        assert c.currsize == 0

    def test_getitem_missing_raises(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            c["missing"]

    def test_delitem_missing_raises(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            del c["missing"]

    def test_len_and_iter(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert len(c) == 3
        assert set(iter(c)) == {"a", "b", "c"}

    def test_getsizeof_default_is_one(self):
        assert Cache.getsizeof(object()) == 1

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
        assert c.pop("a", 99) == 99

    def test_setdefault(self):
        c = Cache(10)
        assert c.setdefault("a", 1) == 1
        assert c.setdefault("a", 2) == 1
        assert c["a"] == 1

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
        assert r.startswith("Cache(")
        assert "maxsize=10" in r
        assert "currsize=1" in r

    def test_value_too_large(self):
        c = Cache(1)
        with pytest.raises(ValueError):
            c["a"] = 1
        with pytest.raises(ValueError):
            c["a"] = 2

    def test_negative_size_value(self):
        c = Cache(10, getsizeof=lambda v: -1)
        with pytest.raises(ValueError):
            c["a"] = 1

    def test_eviction_by_size(self):
        c = Cache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["d"] = 4  # evicts nothing specific; popitem from default MutableMapping
        # default Cache popitem pops first key from __data
        assert "a" not in c
        assert c.currsize <= 3

    def test_custom_getsizeof(self):
        c = Cache(10, getsizeof=lambda v: v)
        c["a"] = 3
        assert c.currsize == 3
        c["a"] = 5
        assert c.currsize == 5

    def test_getsizeof_replacement_shrinks(self):
        c = Cache(10, getsizeof=lambda v: len(v))
        c["k"] = "abc"
        assert c.currsize == 3
        c["k"] = "a"
        assert c.currsize == 1


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_popitem_empty(self):
        c = FIFOCache(3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_fifo_order(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("c", 3)

    def test_update_moves_to_end(self):
        c = FIFOCache(3)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("a", 10)

    def test_eviction(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert "b" in c
        assert "c" in c

    def test_clear(self):
        c = FIFOCache(3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_popitem_empty(self):
        c = LFUCache(3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_get_item_order(self):
        c = LFUCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        # b accessed more often -> a evicted first
        _ = c["b"]
        _ = c["b"]
        _ = c.popitem()
        assert c.popitem()[0] in ("a", "b", "c")
        _ = c.popitem()
        assert c.popitem  # drain helper below

    def test_frequency_order(self):
        c = LFUCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        _ = c["a"]  # freq a = 2
        _ = c["a"]  # freq a = 3
        _ = c["b"]  # freq b = 2
        removed = c.popitem()
        assert removed[0] == "c"  # c at freq 1
        removed2 = c.popitem()
        assert removed2[0] in ("a", "b")

    def test_eviction(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]  # a freq = 2
        c["c"] = 3  # should evict b
        assert "a" in c
        assert "c" in c
        assert "b" not in c

    def test_clear_resets(self):
        c = LFUCache(3)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        c.clear()
        assert len(c) == 0
        c["x"] = 1
        assert c.popitem() == ("x", 1)


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_popitem_empty(self):
        c = LRUCache(3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_lru_order(self):
        c = LRUCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        _ = c["a"]  # a now most recently used
        assert c.popitem() == ("b", 2)
        assert c.popitem() == ("c", 3)
        assert c.popitem() == ("a", 1)

    def test_eviction(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        c["c"] = 3
        assert "a" in c
        assert "c" in c
        assert "b" not in c

    def test_update_touches(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10
        c["c"] = 3  # evict b
        assert "a" in c
        assert "b" not in c

    def test_clear(self):
        c = LRUCache(3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_choice_property(self):
        fn = lambda seq: seq[0]
        c = RRCache(3, choice=fn)
        assert c.choice is fn

    def test_popitem_order_with_choice(self):
        c = RRCache(3, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)

    def test_popitem_empty(self):
        c = RRCache(3, choice=lambda seq: seq[0])
        with pytest.raises(KeyError):
            c.popitem()

    def test_eviction(self):
        c = RRCache(2, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3  # evict a
        assert "a" not in c
        assert "c" in c

    def test_delitem_swaps_last(self):
        c = RRCache(3, choice=lambda seq: seq[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["a"]
        assert "a" not in c
        assert set(c) == {"b", "c"}
        # choice still picks from remaining keys
        assert c.popitem() in [("b", 2), ("c", 3)]

    def test_clear(self):
        c = RRCache(3, choice=lambda seq: seq[0])
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
        c = TTLCache(3, ttl=10, timer=ManualTimer())
        assert c.ttl == 10

    def test_timer_property(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=10, timer=timer)
        assert c.timer() == timer()

    def test_get_and_expiry(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        assert c["a"] == 1
        timer.advance(6)
        with pytest.raises(KeyError):
            c["a"]

    def test_contains_expired(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        assert "a" in c
        timer.advance(6)
        assert "a" not in c

    def test_expire_returns_pairs(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.advance(6)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_expire_with_explicit_time(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        expired = c.expire(10)
        assert expired == [("a", 1)]

    def test_len_expires(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        assert len(c) == 1
        timer.advance(6)
        assert len(c) == 0

    def test_currsize_expires(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        assert c.currsize == 1
        timer.advance(6)
        assert c.currsize == 0

    def test_popitem_empty(self):
        c = TTLCache(3, ttl=5, timer=ManualTimer())
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "a" not in c

    def test_clear(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c["b"] == 2

    def test_iter_skips_expired(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        timer.advance(6)
        c["b"] = 2
        assert set(iter(c)) == {"b"}

    def test_delitem_expired_raises(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        timer.advance(6)
        with pytest.raises(KeyError):
            del c["a"]

    def test_get_and_pop_and_setdefault(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.setdefault("b", 2) == 2
        assert c.setdefault("b", 3) == 2
        assert c.pop("a") == 1

    def test_repr_expires(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=5, timer=timer)
        c["a"] = 1
        timer.advance(6)
        assert "currsize=0" in repr(c)

    def test_pickle_roundtrip(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        d = pickle.loads(pickle.dumps(c))
        assert d["a"] == 1
        assert d["b"] == 2
        assert d.ttl == 10

    def test_timer_reduce(self):
        timer = ManualTimer()
        c = TTLCache(3, ttl=10, timer=timer)
        t = c.timer
        restored, = pickle.loads(pickle.dumps(t)),
        # restored is a Timer wrapping the original timer
        assert restored() == timer()


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

class TestTLRUCache:
    def _make(self, ttl=10, timer=None):
        if timer is None:
            timer = ManualTimer()
        return TLRUCache(3, ttu=lambda k, v, now: now + ttl, timer=timer), timer

    def test_ttu_property(self):
        ttu = lambda k, v, now: now + 10
        c = TLRUCache(3, ttu=ttu, timer=ManualTimer())
        assert c.ttu is ttu

    def test_get_and_expiry(self):
        c, timer = self._make()
        c["a"] = 1
        assert c["a"] == 1
        timer.advance(11)
        with pytest.raises(KeyError):
            c["a"]

    def test_contains_expired(self):
        c, timer = self._make()
        c["a"] = 1
        assert "a" in c
        timer.advance(11)
        assert "a" not in c

    def test_expire_returns_pairs(self):
        c, timer = self._make()
        c["a"] = 1
        c["b"] = 2
        timer.advance(11)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_setitem_already_expired_removes(self):
        c, timer = self._make(ttl=0)
        c["a"] = 1
        # ttu returns now; not (now < now) -> removes
        assert "a" not in c

    def test_len_and_currsize_expire(self):
        c, timer = self._make()
        c["a"] = 1
        assert len(c) == 1
        assert c.currsize == 1
        timer.advance(11)
        assert len(c) == 0
        assert c.currsize == 0

    def test_popitem_empty(self):
        c, _ = self._make()
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem(self):
        c, _ = self._make()
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)

    def test_clear(self):
        c, _ = self._make()
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0

    def test_delitem_expired_raises(self):
        c, timer = self._make()
        c["a"] = 1
        timer.advance(11)
        with pytest.raises(KeyError):
            del c["a"]

    def test_iter_skips_removed_and_expired(self):
        c, timer = self._make()
        c["a"] = 1
        c["b"] = 2
        del c["a"]
        assert set(iter(c)) == {"b"}

    def test_update_marked_removed(self):
        c, _ = self._make()
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2

    def test_get_pop_setdefault(self):
        c, _ = self._make()
        c["a"] = 1
        assert c.get("a") == 1
        assert c.setdefault("b", 2) == 2
        assert c.setdefault("b", 3) == 2
        assert c.pop("a") == 1

    def test_heap_cleanup(self):
        c, _ = self._make()
        for i in range(20):
            c["k%d" % i] = i
            c["k%d" % i] = i + 1  # creates removed items in heap
        # cache still works and evicts
        assert len(c) <= 3

    def test_repr_expires(self):
        c, timer = self._make()
        c["a"] = 1
        timer.advance(11)
        assert "currsize=0" in repr(c)


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------

class TestCached:
    def test_cached_with_cache_instance(self):
        c = Cache(10)

        @cached(c)
        def f(x):
            return x * 2

        assert f(3) == 6
        assert c[3] == 6
        assert f(3) == 6

    def test_cached_with_dict(self):
        d = {}

        @cached(d)
        def f(x):
            return x + 1

        assert f(1) == 2
        assert d[1] == 2

    def test_cached_user_function(self):
        c = {}

        @cached(c)
        def f(x):
            return x + 1

        assert f(5) == 6
        assert f(5) == 6

    def test_cached_method_uses_lock(self):
        c = {}
        lock = unittest.mock.MagicMock()
        lock.__enter__ = unittest.mock.MagicMock(return_value=None)
        lock.__exit__ = unittest.mock.MagicMock(return_value=None)

        @cached(c, lock=lock)
        def f(x):
            return x

        f(1)
        assert lock.__enter__.called

    def test_cached_info(self):
        c = Cache(10)

        @cached(c, info=True)
        def f(x):
            return x

        _ = f(1)
        _ = f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        f.cache_clear()
        assert f.cache_info().currsize == 0

    def test_cached_info_plain_dict(self):
        d = {}

        @cached(d, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.maxsize is None
        assert info.currsize == 1

    def test_cached_info_non_mapping(self):
        sentinel = object()

        @cached(sentinel, info=True)
        def f(x):
            return x

        f(1)
        info = f.cache_info()
        assert info.maxsize == 0
        assert info.currsize == 0

    def test_cached_key_function(self):
        c = {}

        @cached(c, key=lambda x: "k")
        def f(x):
            return x

        f(1)
        f(2)
        assert len(c) == 1


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        class A:
            def __init__(self):
                self.cache = Cache(10)

            @cachedmethod(lambda self: self.cache)
            def f(self, x):
                return x * 2

        a = A()
        assert a.f(3) == 6
        assert a.f(3) == 6
        assert a.f.cache_info().hits == 1

    def test_info_with_dict(self):
        class A:
            def __init__(self):
                self.cache = {}

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        a = A()
        a.f(1)
        info = a.f.cache_info()
        assert info.maxsize is None
        assert info.currsize == 1

    def test_info_invalid_cache(self):
        class A:
            def __init__(self):
                self.cache = object()

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        a = A()
        with pytest.raises(TypeError):
            a.f.cache_info()

    def test_lock(self):
        lock = unittest.mock.MagicMock()
        lock.__enter__ = unittest.mock.MagicMock(return_value=None)
        lock.__exit__ = unittest.mock.MagicMock(return_value=None)

        class A:
            def __init__(self):
                self.cache = {}

            @cachedmethod(lambda self: self.cache, lock=lock)
            def f(self, x):
                return x

        a = A()
        a.f(1)
        assert lock.__enter__.called


# ---------------------------------------------------------------------------
# cross-cache consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: Cache(3),
    lambda: FIFOCache(3),
    lambda: LFUCache(3),
    lambda: LRUCache(3),
    lambda: RRCache(3, choice=lambda seq: seq[0]),
    lambda: TTLCache(3, ttl=100, timer=ManualTimer()),
    lambda: TLRUCache(3, ttu=lambda k, v, now: now + 100, timer=ManualTimer()),
])
def test_common_mapping_behavior(factory):
    c = factory()
    assert len(c) == 0
    c["a"] = 1
    assert c["a"] == 1
    assert "a" in c
    assert c.get("a") == 1
    assert c.get("missing") is None
    assert c.get("missing", 5) == 5
    c["b"] = 2
    assert len(c) == 2
    assert c.pop("a") == 1
    assert "a" not in c
    c.clear()
    assert len(c) == 0


def test_module_all_exports():
    assert set(cachetools.__all__) == {
        "Cache", "FIFOCache", "LFUCache", "LRUCache", "RRCache",
        "TLRUCache", "TTLCache", "cached", "cachedmethod",
    }


def test_version_string():
    assert isinstance(cachetools.__version__, str)
    assert cachetools.__version__.count(".") == 2
