import unittest.mock as mock
import pytest

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

class Clock:
    """A deterministic, manually advanced clock."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_caches(maxsize):
    return [
        Cache(maxsize),
        FIFOCache(maxsize),
        LFUCache(maxsize),
        LRUCache(maxsize),
        RRCache(maxsize, choice=lambda keys: keys[0]),
    ]


def check_cache(cache, maxsize):
    assert cache.maxsize == maxsize
    assert cache.currsize == 0
    assert len(cache) == 0

    for i in range(maxsize + 1):
        cache[i] = i
    assert len(cache) == maxsize
    assert cache.currsize == maxsize
    # after inserting 0..maxsize, key 0 was evicted; remaining keys are
    # 1..maxsize in insertion order.
    remaining = list(range(1, maxsize + 1))
    assert sorted(cache) == remaining
    assert all(cache[i] == i for i in remaining)

    del cache[1]
    assert len(cache) == maxsize - 1
    assert cache.currsize == maxsize - 1
    assert 1 not in cache

    cache[0] = 0
    assert len(cache) == maxsize
    assert cache.currsize == maxsize
    # after deletion only maxsize-1 slots are used, inserting 0 does not
    # trigger eviction; the cache now holds {0} ∪ {2..maxsize}
    expected = {0} | set(range(2, maxsize + 1))
    assert set(cache) == expected
    assert all(cache[i] == i for i in cache)

    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    assert repr(cache) == "%s(%s, maxsize=%r, currsize=%r)" % (
        type(cache).__name__, repr({}), maxsize, 0
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_negative_maxsize():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_maxsize_zero():
    cache = Cache(0)
    with pytest.raises(ValueError):
        cache[1] = 1
    assert len(cache) == 0
    assert cache.currsize == 0


def test_cache_getsizeof_default():
    assert Cache.getsizeof(42) == 1


def test_cache_getsizeof_custom():
    cache = Cache(10, getsizeof=lambda v: len(v))
    cache["a"] = "12345"
    assert cache.currsize == 5
    cache["b"] = "12"
    assert cache.currsize == 7
    cache["a"] = "1"
    assert cache.currsize == 3


def test_cache_value_too_large():
    # default getsizeof is 1, so maxsize=5 never triggers "value too large"
    cache = Cache(5)
    cache["a"] = 1
    assert cache["a"] == 1
    # with custom getsizeof a too-large value raises
    cache = Cache(5, getsizeof=lambda v: v)
    with pytest.raises(ValueError):
        cache["a"] = 6


def test_cache_negative_size_value():
    cache = Cache(5, getsizeof=lambda v: -1)
    with pytest.raises(ValueError):
        cache["a"] = 1


def test_cache_get_pop_setdefault():
    cache = Cache(5)
    assert cache.get("a") is None
    assert cache.get("a", 42) == 42
    with pytest.raises(KeyError):
        cache.pop("a")
    assert cache.pop("a", 42) == 42

    cache["a"] = 1
    assert cache.get("a") == 1
    # pop still uses __getitem__
    assert cache.pop("a") == 1
    assert "a" not in cache

    assert cache.setdefault("b", 2) == 2
    assert cache["b"] == 2
    assert cache.setdefault("b", 3) == 2


def test_cache_missing():
    cache = Cache(5)
    with pytest.raises(KeyError):
        cache["missing"]


def test_cache_popitem_empty():
    # Mixin popitem uses iteration; should raise KeyError on empty
    cache = Cache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_cache_clear():
    cache = Cache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0


def test_cache_repr():
    cache = Cache(5)
    assert repr(cache) == "Cache({}, maxsize=5, currsize=0)"
    cache["a"] = 1
    assert repr(cache) == "Cache({'a': 1}, maxsize=5, currsize=1)"


def test_cache_subclass_evict_functional():
    cache = Cache(2)
    cache[1] = 1
    cache[2] = 2
    # Cache base popitem pops in iteration order (first inserted)
    assert cache.popitem() == (1, 1)
    assert cache.popitem() == (2, 2)
    with pytest.raises(KeyError):
        cache.popitem()


def test_cache_eviction():
    cache = Cache(2)
    cache[1] = 1
    cache[2] = 2
    cache[3] = 3
    assert len(cache) == 2
    assert 1 not in cache


def test_cache_evict_by_size():
    cache = Cache(10, getsizeof=lambda v: len(v))
    cache["a"] = "12345"
    cache["b"] = "12345"
    assert cache.currsize == 10
    cache["c"] = "12345"
    assert cache.currsize <= 10
    assert "a" not in cache


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

def test_fifo_eviction():
    cache = FIFOCache(2)
    cache[1] = 1
    cache[2] = 2
    cache[1] = 1  # moves to end (re-inserted)
    cache[3] = 3
    assert 2 not in cache
    cache[4] = 4
    assert 1 not in cache
    assert sorted(cache) == [3, 4]


def test_fifo_popitem_empty():
    cache = FIFOCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_fifo_clear_resets_order():
    cache = FIFOCache(5)
    cache["a"] = 1
    cache.clear()
    cache["b"] = 2
    assert list(cache) == ["b"]
    assert cache.popitem() == ("b", 2)


def test_fifo_popitem():
    cache = FIFOCache(5)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.popitem() == ("a", 1)
    assert cache.popitem() == ("b", 2)


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

def test_lfu_eviction():
    cache = LFUCache(2)
    cache[1] = 1
    cache[2] = 2
    _ = cache[1]
    cache[3] = 3
    assert 1 in cache
    assert 2 not in cache


def test_lfu_popitem_empty():
    cache = LFUCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lfu_popitem():
    cache = LFUCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # bump a
    key, value = cache.popitem()
    assert key == "b"
    assert value == 2


def test_lfu_touch_same_count():
    cache = LFUCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]
    cache["a"]
    cache["a"]
    assert cache.popitem() == ("b", 2)


def test_lfu_update_keeps_entry_single():
    cache = LFUCache(5)
    cache["a"] = 1
    cache["a"] = 2
    assert cache["a"] == 2
    assert len(cache) == 1


def test_lfu_clear():
    cache = LFUCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0


def test_lfu_getitem_missing():
    cache = LFUCache(5)
    with pytest.raises(KeyError):
        cache["x"]


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

def test_lru_eviction():
    cache = LRUCache(2)
    cache[1] = 1
    cache[2] = 2
    _ = cache[1]
    cache[3] = 3
    assert 1 in cache
    assert 2 not in cache


def test_lru_popitem_empty():
    cache = LRUCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lru_popitem():
    cache = LRUCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]
    assert cache.popitem() == ("b", 2)


def test_lru_clear():
    cache = LRUCache(5)
    cache["a"] = 1
    cache.clear()
    cache["b"] = 2
    assert list(cache) == ["b"]


def test_lru_getitem_missing():
    cache = LRUCache(5)
    with pytest.raises(KeyError):
        cache["x"]


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

def test_rr_choice_property():
    choice = lambda keys: keys[0]
    cache = RRCache(5, choice=choice)
    assert cache.choice is choice


def test_rr_eviction():
    cache = RRCache(2, choice=lambda keys: keys[0])
    cache[1] = 1
    cache[2] = 2
    cache[3] = 3
    assert len(cache) == 2
    assert 1 not in cache


def test_rr_popitem_empty():
    cache = RRCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_rr_popitem():
    cache = RRCache(5, choice=lambda keys: keys[0])
    cache["a"] = 1
    cache["b"] = 2
    assert cache.popitem() == ("a", 1)


def test_rr_del_middle():
    cache = RRCache(5, choice=lambda keys: keys[0])
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    del cache["b"]
    assert "b" not in cache
    assert set(cache) == {"a", "c"}
    # index mapping maintained
    assert cache.popitem()[0] in ("a", "c")


def test_rr_clear():
    cache = RRCache(5)
    cache["a"] = 1
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(KeyError):
        cache.popitem()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

def test_ttl_expire():
    clock = Clock(0)
    cache = TTLCache(maxsize=2, ttl=2, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    assert len(cache) == 2
    clock.advance(1.5)
    assert len(cache) == 1
    assert "a" not in cache
    assert "b" in cache
    clock.advance(1)
    assert len(cache) == 0


def test_ttl_ttl_property():
    cache = TTLCache(5, ttl=10)
    assert cache.ttl == 10


def test_ttl_getitem_expired():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    clock.advance(2)
    with pytest.raises(KeyError):
        cache["a"]


def test_ttl_contains():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    assert "a" in cache
    clock.advance(2)
    assert "a" not in cache


def test_ttl_popitem_empty():
    cache = TTLCache(5, ttl=10)
    with pytest.raises(KeyError):
        cache.popitem()


def test_ttl_popitem_skips_expired():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=2, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    clock.advance(1.5)
    assert cache.popitem() == ("b", 2)


def test_ttl_refresh():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    clock.advance(0.5)
    cache["a"] = 2
    clock.advance(0.75)
    assert "a" in cache
    assert cache["a"] == 2


def test_ttl_expire_returns_expired():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    cache["b"] = 2
    clock.advance(2)
    expired = cache.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(cache) == 0


def test_ttl_expire_explicit_time():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    expired = cache.expire(time=5)
    assert expired == [("a", 1)]
    clock.advance(5)
    assert "a" not in cache


def test_ttl_clear():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0


def test_ttl_iter_expired():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    clock.advance(2)
    assert list(cache) == []


def test_ttl_delitem():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache


def test_ttl_timer_property():
    clock = Clock(0)
    cache = TTLCache(maxsize=5, ttl=1, timer=clock)
    assert cache.timer() == 0


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

def ttu(key, value, time):
    return time + (value if isinstance(value, (int, float)) else 100)


def test_tlru_ttu_property():
    cache = TLRUCache(5, ttu=ttu)
    assert cache.ttu is ttu


def test_tlru_expire():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 1
    cache["b"] = 5
    assert len(cache) == 2
    clock.advance(2)
    assert len(cache) == 1
    assert "a" not in cache
    assert "b" in cache


def test_tlru_contains():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 1
    assert "a" in cache
    clock.advance(2)
    assert "a" not in cache


def test_tlru_getitem_expired():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 1
    clock.advance(2)
    with pytest.raises(KeyError):
        cache["a"]


def test_tlru_popitem_empty():
    cache = TLRUCache(5, ttu=ttu)
    with pytest.raises(KeyError):
        cache.popitem()


def test_tlru_popitem():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 100
    cache["b"] = 100
    assert cache.popitem() == ("a", 100)


def test_tlru_setitem_expired_removes_entry():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 100
    cache["a"] = -1
    assert "a" not in cache


def test_tlru_expire_returns_pairs():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 1
    cache["b"] = 2
    clock.advance(100)
    expired = cache.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]


def test_tlru_iter():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 100
    cache["b"] = 100
    assert set(cache) == {"a", "b"}
    cache["a"] = 1
    clock.advance(2)
    assert list(cache) == ["b"]


def test_tlru_clear():
    clock = Clock(0)
    cache = TLRUCache(maxsize=5, ttu=ttu, timer=clock)
    cache["a"] = 100
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0


def test_tlru_heap_cleanup():
    clock = Clock(0)
    cache = TLRUCache(maxsize=10, ttu=ttu, timer=clock)
    for i in range(5):
        cache[i] = 100
    for i in range(5):
        cache[i] = 100
    cache.expire(clock())
    assert len(cache) == 5


# ---------------------------------------------------------------------------
# generic Cache semantics for subclasses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Cache, FIFOCache, LFUCache, LRUCache, RRCache])
def test_check_cache_basic(cls):
    if cls is RRCache:
        cache = RRCache(3, choice=lambda keys: keys[0])
    else:
        cache = cls(3)
    check_cache(cache, 3)


# ---------------------------------------------------------------------------
# cached / cachedmethod
# ---------------------------------------------------------------------------

def test_cached_basic():
    calls = []

    @cached(cache=Cache(5))
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


def test_cached_info():
    @cached(cache=Cache(5), info=True)
    def f(x):
        return x

    f(1)
    f(1)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 5
    assert info.currsize == 1


def test_cached_info_mapping():
    c = {}

    @cached(cache=c, info=True)
    def f(x):
        return x

    f(1)
    info = f.cache_info()
    assert info.maxsize is None
    assert info.currsize == 1


def test_cached_key():
    @cached(cache=Cache(5), key=lambda *a, **kw: a)
    def f(x):
        return x

    f(1)
    f(1)
    assert len(f.cache) == 1


def test_cachedmethod_basic():
    class A:
        def __init__(self):
            self.calls = 0

        @cachedmethod(cache=lambda self: self._cache)
        def f(self, x):
            self.calls += 1
            return x * 2

    a = A()
    a._cache = Cache(5)
    assert a.f(1) == 2
    assert a.f(1) == 2
    assert a.calls == 1


def test_cachedmethod_info():
    class A:
        def __init__(self):
            self._cache = Cache(5)

        @cachedmethod(cache=lambda self: self._cache, info=True)
        def f(self, x):
            return x

    a = A()
    a.f(1)
    a.f(1)
    info = a.f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 5
