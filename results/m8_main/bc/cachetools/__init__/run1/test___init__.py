"""Tests for the public API of cachetools."""

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
from cachetools import _CacheInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingTimer:
    """Deterministic timer whose value only changes when explicitly told to."""

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, amount):
        self.value += amount


class FixedChoice:
    def __init__(self, idx=0):
        self.idx = idx

    def __call__(self, seq):
        return seq[self.idx]


# ---------------------------------------------------------------------------
# _DefaultSize
# ---------------------------------------------------------------------------


def test_default_size():
    from cachetools import _DefaultSize

    d = _DefaultSize()
    assert d["x"] == 1
    assert d.pop("x") == 1
    d["x"] = 5
    d.clear()


# ---------------------------------------------------------------------------
# Cache base class
# ---------------------------------------------------------------------------


def test_cache_maxsize_negative():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_basic():
    c = Cache(2)
    c["a"] = 1
    c["b"] = 2
    assert c.maxsize == 2
    assert c.currsize == 2
    assert len(c) == 2
    assert "a" in c
    assert "x" not in c
    assert c["a"] == 1
    assert sorted(c) == ["a", "b"]


def test_cache_missing():
    c = Cache(2)
    with pytest.raises(KeyError):
        c["x"]


def test_cache_size_exceeded_evicts():
    c = Cache(2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert len(c) == 2
    # values fit (getsizeof returns 1), so no eviction can occur here since
    # each non-negative size equals maxsize? Actually with default size 1 and
    # maxsize 2 only 2 items can fit; the third triggers eviction.
    assert "c" in c


def test_cache_value_too_large():
    # With a getsizeof that reports a size greater than maxsize, __setitem__
    # raises ValueError.
    c = Cache(2, getsizeof=lambda v: 5)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_getsizeof():
    def getsizeof(value):
        return value

    c = Cache(10, getsizeof=getsizeof)
    c["a"] = 3
    c["b"] = 4
    assert c.currsize == 7
    # adding more that overflows should evict
    c["c"] = 4
    assert c.currsize <= 10


def test_cache_negative_size():
    c = Cache(10, getsizeof=lambda v: -1)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_size_too_large_with_getsizeof():
    c = Cache(5, getsizeof=lambda v: v)
    with pytest.raises(ValueError):
        c["a"] = 10


def test_cache_update_size():
    c = Cache(10, getsizeof=lambda v: v)
    c["a"] = 3
    c["a"] = 5
    assert c.currsize == 5
    assert c["a"] == 5


def test_cache_update_shrinks():
    c = Cache(10, getsizeof=lambda v: v)
    c["a"] = 5
    c["a"] = 2
    assert c.currsize == 2


def test_cache_get_default():
    c = Cache(2)
    assert c.get("x") is None
    assert c.get("x", 5) == 5
    c["a"] = 1
    assert c.get("a") == 1


def test_cache_pop():
    c = Cache(2)
    c["a"] = 1
    assert c.pop("a") == 1
    assert "a" not in c
    with pytest.raises(KeyError):
        c.pop("a")
    assert c.pop("a", 99) == 99


def test_cache_setdefault():
    c = Cache(2)
    assert c.setdefault("a", 5) == 5
    assert c["a"] == 5
    assert c.setdefault("a", 10) == 5


def test_cache_delitem():
    c = Cache(2)
    c["a"] = 1
    del c["a"]
    assert "a" not in c
    assert c.currsize == 0
    with pytest.raises(KeyError):
        del c["a"]


def test_cache_clear():
    c = Cache(10)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_cache_repr():
    c = Cache(5)
    c["a"] = 1
    r = repr(c)
    assert "Cache(" in r
    assert "maxsize=5" in r
    assert "currsize=1" in r


def test_cache_getsizeof_static():
    assert Cache.getsizeof(object()) == 1


def test_cache_update_via_lru_eviction():
    # Exercises Cache.__setitem__ update branch where popitem evicts the
    # same key being updated, forcing diffsize to be recomputed.
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    # 'a' is now least recently used; updating it needs to grow its size,
    # but a size for 'a' has to be set. Use getsizeof overrides via subclass
    # approach: use a plain Cache with getsizeof and popitem behavior.
    c2 = FIFOCache(3, getsizeof=lambda v: v)
    c2["a"] = 1
    c2["b"] = 1
    c2["c"] = 1
    # updating 'a' to a bigger size forces evictions; if 'a' itself is the
    # one popped first, diffsize must fall back to the new size.
    c2["a"] = 3
    assert c2.currsize <= 3
    assert c2["a"] == 3


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------


def test_fifo_eviction():
    c = FIFOCache(2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert sorted(c) == ["b", "c"]


def test_fifo_update_moves_to_end():
    c = FIFOCache(2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10  # refresh a
    c["c"] = 3
    # b should have been evicted
    assert sorted(c) == ["a", "c"]
    assert c["a"] == 10


def test_fifo_popitem():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert "a" not in c


def test_fifo_popitem_empty():
    c = FIFOCache(3)
    with pytest.raises(KeyError):
        c.popitem()


def test_fifo_delitem():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    assert c.popitem() == ("b", 2)


def test_fifo_clear():
    c = FIFOCache(3)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_eviction():
    c = LRUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]  # touch a
    c["c"] = 3
    assert sorted(c) == ["a", "c"]


def test_lru_popitem():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    assert c.popitem() == ("b", 2)


def test_lru_popitem_empty():
    c = LRUCache(3)
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_delitem():
    c = LRUCache(3)
    c["a"] = 1
    del c["a"]
    assert "a" not in c


def test_lru_clear():
    c = LRUCache(3)
    c["a"] = 1
    c.clear()
    assert len(c) == 0


def test_lru_missing_not_touched():
    # Exercise LRUCache.__getitem__ branch where __missing__ does not store.
    class MyLRU(LRUCache):
        def __missing__(self, key):
            return "missing:" + key

    c = MyLRU(2)
    assert c["x"] == "missing:x"
    assert len(c) == 0


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------


def test_lfu_eviction():
    c = LFUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    _ = c["a"]
    c["c"] = 3
    assert sorted(c) == ["a", "c"]


def test_lfu_only_one_sort():
    c = LFUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert sorted(c) == ["a", "c"]


def test_lfu_update():
    c = LFUCache(2)
    c["a"] = 1
    c["a"] = 2
    assert c["a"] == 2
    assert c.currsize == 1


def test_lfu_popitem():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    _ = c["a"]
    assert c.popitem() == ("b", 2)


def test_lfu_popitem_empty():
    c = LFUCache(3)
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_delitem():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    assert c.popitem() == ("b", 2)


def test_lfu_clear():
    c = LFUCache(3)
    c["a"] = 1
    c.clear()
    assert len(c) == 0


def test_lfu_missing_not_touched():
    class MyLFU(LFUCache):
        def __missing__(self, key):
            return None

    c = MyLFU(2)
    assert c["x"] is None
    assert len(c) == 0


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------


def test_rr_choice_property():
    choice = FixedChoice()
    c = RRCache(2, choice=choice)
    assert c.choice is choice


def test_rr_eviction():
    c = RRCache(2, choice=FixedChoice(0))
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert sorted(c) == ["b", "c"]


def test_rr_update():
    c = RRCache(2, choice=FixedChoice(0))
    c["a"] = 1
    c["a"] = 2
    assert c["a"] == 2


def test_rr_popitem():
    c = RRCache(3, choice=FixedChoice(1))
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    key, value = c.popitem()
    assert (key, value) == ("b", 2)
    assert sorted(c) == ["a", "c"]


def test_rr_popitem_empty():
    c = RRCache(3)
    with pytest.raises(KeyError):
        c.popitem()


def test_rr_delitem_swap():
    c = RRCache(3, choice=FixedChoice(0))
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    del c["b"]
    assert sorted(c) == ["a", "c"]


def test_rr_delitem_last():
    c = RRCache(3, choice=FixedChoice(0))
    c["a"] = 1
    c["b"] = 2
    del c["b"]
    assert sorted(c) == ["a"]


def test_rr_clear():
    c = RRCache(3)
    c["a"] = 1
    c.clear()
    assert len(c) == 0


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------


def test_ttl_basic():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    assert c["a"] == 1
    assert c.ttl == 5
    assert timer is not None


def test_ttl_expire():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    assert "a" in c
    timer.advance(6)
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_ttl_len_expires():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    timer.advance(6)
    assert len(c) == 0


def test_ttl_currsize_expires():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    assert c.currsize == 0


def test_ttl_repr_expires():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    r = repr(c)
    assert "currsize=0" in r


def test_ttl_expire_method_returns_pairs():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    assert c.expire(10) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_ttl_expire_partial():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(3)
    c["b"] = 2
    timer.advance(3)  # a expired, b not
    expired = c.expire()
    assert expired == [("a", 1)]
    assert "b" in c


def test_ttl_expire_no_time_arg():
    # expire(None) defaults to self.timer() (line 549-550).
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    assert c.expire() == [("a", 1)]


def test_ttl_popitem():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)


def test_ttl_popitem_empty():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_popitem_skips_expired():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    c["b"] = 2
    assert c.popitem() == ("b", 2)


def test_ttl_delitem_expired_raises():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    with pytest.raises(KeyError):
        del c["a"]
    assert "a" not in c


def test_ttl_get_pop_setdefault():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("x", 5) == 5
    assert c.pop("a") == 1
    assert c.setdefault("b", 2) == 2
    assert c["b"] == 2


def test_ttl_get_expired_default():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(6)
    assert c.get("a") is None
    assert c.get("a", 42) == 42


def test_ttl_iter_only_unexpired():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(3)
    c["b"] = 2
    timer.advance(3)
    assert list(c) == ["b"]


def test_ttl_iter_all_expired():
    # Covers the branch where the iterator sees expired links (line 525->527).
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    timer.advance(6)
    assert list(c) == []


def test_ttl_contains_missing():
    # Covers TTLCache.__contains__ KeyError branch (lines 487-488).
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    assert "b" not in c


def test_ttl_getitem_missing():
    # __getitem__ where __getlink raises KeyError -> expired=False -> cache miss.
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    with pytest.raises(KeyError):
        c["missing"]


def test_ttl_setitem_update_existing():
    # Updating an existing key hits the link.unlink() branch (line 506).
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    timer.advance(1)
    c["a"] = 2
    assert c["a"] == 2
    assert len(c) == 1


def test_ttl_lru_ordering():
    timer = CountingTimer()
    c = TTLCache(2, ttl=100, timer=timer)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert sorted(c) == ["a", "c"]


def test_ttl_clear():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_ttl_timer_property():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    assert c.timer() == 0
    timer.advance(1)
    assert c.timer() == 1


def test_ttl_timer_getattr():
    # _Timer.__getattr__ delegates to the wrapped timer (line 401).
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    assert c.timer.value == 0
    timer.advance(2)
    assert c.timer.value == 2


def test_ttl_pickle_roundtrip():
    timer = CountingTimer()
    c = TTLCache(10, ttl=5, timer=timer)
    c["a"] = 1
    data = pickle.dumps(c)
    c2 = pickle.loads(data)
    assert c2["a"] == 1
    assert c2.ttl == 5


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------


def ttu(key, value, now):
    return now + 5


def test_tlru_basic():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    assert c["a"] == 1
    assert c.ttu is ttu


def test_tlru_expire():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    timer.advance(6)
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_tlru_expire_method():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    c["b"] = 2
    expired = c.expire(10)
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_tlru_expire_no_time_arg():
    # expire(None) uses self.timer() (lines 679-680).
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    timer.advance(6)
    assert c.expire() == [("a", 1)]


def test_tlru_setitem_expired_removes():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=lambda k, v, now: now, timer=timer)
    c["a"] = 1
    assert "a" not in c


def test_tlru_setitem_expired_existing_removes():
    # Existing key updated with already-expired ttu -> __delitem helper path.
    timer = CountingTimer()
    c = TLRUCache(10, ttu=lambda k, v, now: now + 100, timer=timer)
    c["a"] = 1
    assert "a" in c
    # Now change ttu behavior by using a mutable function
    def ttu_expired(k, v, now):
        return now if v == 2 else now + 100
    c2 = TLRUCache(10, ttu=ttu_expired, timer=timer)
    c2["a"] = 1
    c2["a"] = 2
    assert "a" not in c2


def test_tlru_update_existing():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    c["a"] = 2
    assert c["a"] == 2
    assert len(c) == 1


def test_tlru_popitem():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)


def test_tlru_popitem_empty():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    with pytest.raises(KeyError):
        c.popitem()


def test_tlru_delitem_expired():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    timer.advance(6)
    with pytest.raises(KeyError):
        del c["a"]
    assert "a" not in c


def test_tlru_delitem_missing():
    # __delitem with unknown key: ignored in helper, inner delitem raises.
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    with pytest.raises(KeyError):
        del c["missing"]


def test_tlru_contains_missing():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    assert "x" not in c


def test_tlru_getitem_missing():
    # __getitem__ where __getitem raises KeyError -> cache miss (lines 626-627).
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    with pytest.raises(KeyError):
        c["missing"]


def test_tlru_lru_ordering():
    timer = CountingTimer()
    c = TLRUCache(2, ttu=lambda k, v, now: now + 100, timer=timer)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert sorted(c) == ["a", "c"]


def test_tlru_iter():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    timer.advance(3)
    c["b"] = 2
    timer.advance(3)
    assert list(c) == ["b"]


def test_tlru_clear():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_tlru_heap_cleanup():
    timer = CountingTimer()

    def ttu2(key, value, now):
        return now + 100

    c = TLRUCache(100, ttu=ttu2, timer=timer)
    for i in range(10):
        c[i] = i
    for i in range(5):
        del c[i]
    # some operations to trigger heap cleanup
    c.expire(0)
    assert len(c) == 5


def test_tlru_get_pop_setdefault():
    timer = CountingTimer()
    c = TLRUCache(10, ttu=ttu, timer=timer)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("x", 0) == 0
    assert c.pop("a") == 1
    assert c.setdefault("b", 2) == 2


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------


def test_cached_basic():
    calls = []

    @cached(Cache(10))
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


def test_cached_info():
    @cached(Cache(10), info=True)
    def f(x):
        return x * 2

    assert f(1) == 2
    info = f.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 10
    f(1)
    assert f.cache_info().hits == 1


def test_cached_info_mapping():
    cache = {}

    @cached(cache, info=True)
    def f(x):
        return x

    f(1)
    info = f.cache_info()
    assert info.maxsize is None
    assert info.currsize == 1


def test_cached_info_non_mapping():
    # info=True with a non-Mapping, non-Cache cache yields (0, 0) sizes.
    store = {}

    class FakeCache:
        pass

    fake = FakeCache()

    # Use a callable cache-like object via a plain object that isn't a Mapping.
    # We still need _wrapper to work, so pass a dict-like; but to exercise
    # the else branch we pass something that behaves like a mapping through
    # __getitem__/__setitem__/__contains__ but isn't registered as Mapping.
    class NotMapping:
        def __init__(self):
            self.data = {}

        def __contains__(self, k):
            return k in self.data

        def __getitem__(self, k):
            return self.data[k]

        def __setitem__(self, k, v):
            self.data[k] = v

    nm = NotMapping()

    @cached(nm, info=True)
    def f(x):
        return x

    assert f(5) == 5
    info = f.cache_info()
    assert info.maxsize == 0
    assert info.currsize == 0


def test_cached_clear():
    @cached(Cache(10))
    def f(x):
        return x

    f(1)
    f.cache_clear()
    assert len(f.cache) == 0


def test_cached_unhashable_error():
    @cached(Cache(10))
    def f(x):
        return x

    with pytest.raises(TypeError):
        f([1, 2, 3])


def test_cached_custom_key():
    @cached(Cache(10), key=lambda *a, **k: "fixed")
    def f(x):
        return x

    assert f(1) == 1
    assert f(2) == 1


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------


class _Ob:
    def __init__(self):
        self.cache = LRUCache(10)
        self.calls = 0

    @cachedmethod(lambda self: self.cache)
    def compute(self, x):
        self.calls += 1
        return x * 3


def test_cachedmethod_basic():
    ob = _Ob()
    assert ob.compute(2) == 6
    assert ob.compute(2) == 6
    assert ob.calls == 1


def test_cachedmethod_clear():
    ob = _Ob()
    ob.compute(1)
    ob.compute.cache_clear(ob)
    assert len(ob.cache) == 0


def test_cachedmethod_info():
    class Ob:
        def __init__(self):
            self.cache = LRUCache(10)

        @cachedmethod(lambda self: self.cache, info=True)
        def compute(self, x):
            return x

    ob = Ob()
    ob.compute(1)
    info = ob.compute.cache_info()
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 10


def test_cachedmethod_info_mapping():
    # cachedmethod info=True with a Mapping cache yields maxsize=None.
    class Ob:
        def __init__(self):
            self.cache = {"seed": 0}

        @cachedmethod(lambda self: self.cache, info=True)
        def compute(self, x):
            return x

    ob = Ob()
    ob.compute(1)
    info = ob.compute.cache_info()
    assert info.maxsize is None
    assert info.currsize == 2


def test_cachedmethod_info_type_error():
    class Ob:
        def __init__(self):
            self.cache = 42  # not a mapping

        @cachedmethod(lambda self: self.cache, info=True)
        def compute(self, x):
            return x

    ob = Ob()
    with pytest.raises(TypeError):
        ob.compute(1)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_public_names():
    for name in (
        "Cache",
        "FIFOCache",
        "LFUCache",
        "LRUCache",
        "RRCache",
        "TLRUCache",
        "TTLCache",
        "cached",
        "cachedmethod",
    ):
        assert hasattr(cachetools, name)
        assert name in cachetools.__all__


def test_version_string():
    assert isinstance(cachetools.__version__, str)
    assert cachetools.__version__
