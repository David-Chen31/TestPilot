"""Unit tests for the public API of :mod:`cachetools`."""

import collections
import collections.abc
import heapq
import pickle
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
    keys,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class PicklableTimer:
    """A small deterministic timer object usable by TTLCache/TLRUCache."""

    def __init__(self, time=0.0):
        self.time = time

    def __call__(self):
        return self.time

    def advance(self, delta):
        self.time += delta


def make_ttl_clock():
    return PicklableTimer()


# ---------------------------------------------------------------------------
# module level
# ---------------------------------------------------------------------------


def test_dunder_all():
    assert cachetools.__all__ == (
        "Cache",
        "FIFOCache",
        "LFUCache",
        "LRUCache",
        "RRCache",
        "TLRUCache",
        "TTLCache",
        "cached",
        "cachedmethod",
    )


def test_version():
    assert isinstance(cachetools.__version__, str)
    assert cachetools.__version__.count(".") == 2


# ---------------------------------------------------------------------------
# Cache (base class)
# ---------------------------------------------------------------------------


def test_cache_default_getsizeof():
    cache = Cache(10)
    assert cache.maxsize == 10
    assert cache.currsize == 0
    assert Cache.getsizeof(object()) == 1


def test_cache_negative_maxsize():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_setitem_getitem_len_contains():
    cache = Cache(10)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert len(cache) == 2
    assert "a" in cache
    assert "missing" not in cache
    assert cache.currsize == 2
    assert list(iter(cache)) == ["a", "b"]


def test_cache_missing_raises():
    cache = Cache(10)
    with pytest.raises(KeyError):
        cache["missing"]


def test_cache_delitem():
    cache = Cache(10)
    cache["a"] = 1
    assert cache.currsize == 1
    del cache["a"]
    assert "a" not in cache
    assert cache.currsize == 0
    assert len(cache) == 0
    with pytest.raises(KeyError):
        del cache["a"]


def test_cache_update_existing():
    cache = Cache(10)
    cache["a"] = 1
    cache["a"] = 2
    assert cache["a"] == 2
    assert cache.currsize == 1


def test_cache_value_too_large():
    cache = Cache(2)
    with pytest.raises(ValueError):
        cache["a"] = object()
    # value size 1 fits; assignment succeeds
    cache["a"] = 1
    assert cache["a"] == 1


def test_cache_eviction_simple():
    cache = Cache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    # Default Cache.popitem is from MutableMapping: pops arbitrary item.
    # Since Dict insertion order, MutableMapping.popitem pops last item.
    assert len(cache) == 0
    assert cache.currsize == 0


def test_cache_setitem_uses_custom_getsizeof():
    def getsizeof(value):
        return value

    cache = Cache(10, getsizeof=getsizeof)
    cache["a"] = 3
    assert cache.currsize == 3
    cache["a"] = 5
    assert cache.currsize == 5
    cache["b"] = 2
    assert cache.currsize == 7


def test_cache_getsizeof_negative_value():
    def getsizeof(value):
        return value

    cache = Cache(10, getsizeof=getsizeof)
    with pytest.raises(ValueError):
        cache["x"] = -1


def test_cache_get():
    cache = Cache(10)
    cache["a"] = 1
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("b", 42) == 42


def test_cache_pop():
    cache = Cache(10)
    cache["a"] = 1
    assert cache.pop("a") == 1
    assert "a" not in cache
    assert cache.currsize == 0
    with pytest.raises(KeyError):
        cache.pop("a")
    assert cache.pop("a", 5) == 5


def test_cache_setdefault():
    cache = Cache(10)
    assert cache.setdefault("a", 1) == 1
    assert cache["a"] == 1
    assert cache.setdefault("a", 2) == 1
    assert cache.currsize == 1


def test_cache_clear():
    cache = Cache(10)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    assert not cache


def test_cache_repr():
    cache = Cache(3)
    cache["a"] = 1
    assert repr(cache) == "Cache({'a': 1}, maxsize=3, currsize=1)"


def test_cache_mutable_mapping_protocol():
    cache = Cache(10)
    assert isinstance(cache, collections.abc.MutableMapping)


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------


def test_fifo_eviction_order():
    cache = FIFOCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["d"] = 4
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert "d" in cache


def test_fifo_move_on_update():
    cache = FIFOCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"] = 10  # updates move to end
    cache["d"] = 4
    # b is now the oldest
    assert "b" not in cache
    assert "a" in cache
    assert cache["a"] == 10


def test_fifo_popitem_order():
    cache = FIFOCache(5)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.popitem() == ("a", 1)
    assert cache.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        cache.popitem()


def test_fifo_clear():
    cache = FIFOCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    # Can still insert after clear
    cache["x"] = 1
    assert cache["x"] == 1


def test_fifo_delitem():
    cache = FIFOCache(5)
    cache["a"] = 1
    cache["b"] = 2
    del cache["a"]
    assert "a" not in cache
    assert cache.currsize == 1
    cache["c"] = 3
    cache["d"] = 4
    cache["e"] = 5
    cache["f"] = 6
    # b should have been evicted
    assert "b" not in cache


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------


def test_lfu_eviction_least_frequent():
    cache = LFUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"]  # touch a
    cache["b"]  # touch b
    cache["d"] = 4  # should evict c
    assert "c" not in cache
    assert "a" in cache
    assert "b" in cache
    assert "d" in cache


def test_lfu_popitem():
    cache = LFUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # a has higher count
    key, value = cache.popitem()
    assert key == "b"
    assert value == 2


def test_lfu_popitem_empty():
    cache = LFUCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lfu_getitem_updates_count():
    cache = LFUCache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # touch a
    cache["c"] = 3  # evicts b
    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache


def test_lfu_missing_raises():
    cache = LFUCache(2)
    with pytest.raises(KeyError):
        cache["missing"]


def test_lfu_clear():
    cache = LFUCache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["x"] = 1
    assert cache["x"] == 1


def test_lfu_delitem():
    cache = LFUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    del cache["a"]
    assert "a" not in cache
    assert cache.currsize == 1
    cache["c"] = 3
    cache["d"] = 4
    cache["e"] = 5
    assert "b" not in cache


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_eviction():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"]  # touch a
    cache["d"] = 4  # evicts b
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache
    assert "d" in cache


def test_lru_setitem_updates_order():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"] = 10  # move a to end
    cache["d"] = 4  # evicts b
    assert "b" not in cache
    assert cache["a"] == 10


def test_lru_get_with_missing_does_not_touch():
    cache = LRUCache(2)
    cache["a"] = 1
    cache["b"] = 2
    with pytest.raises(KeyError):
        cache["x"]
    cache["c"] = 3  # should evict a
    assert "a" not in cache
    assert "b" in cache


def test_lru_popitem_order():
    cache = LRUCache(5)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # move a to end
    assert cache.popitem() == ("b", 2)
    assert cache.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lru_clear():
    cache = LRUCache(2)
    cache["a"] = 1
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["x"] = 1
    cache["y"] = 2
    assert len(cache) == 2


def test_lru_delitem():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    del cache["a"]
    assert "a" not in cache
    cache["c"] = 3
    cache["d"] = 4
    cache["e"] = 5
    assert "b" not in cache


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------


def test_rr_choice_property():
    def dummy_choice(seq):
        return seq[0]

    cache = RRCache(5, choice=dummy_choice)
    assert cache.choice is dummy_choice


def test_rr_popitem_uses_choice():
    calls = []

    def choice(seq):
        calls.append(seq)
        return seq[0]

    cache = RRCache(5, choice=choice)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache.popitem() == ("a", 1)
    assert "a" not in cache
    # after evicting, index swaps in last item
    assert set(cache) == {"b", "c"}


def test_rr_popitem_empty():
    cache = RRCache(5)
    with pytest.raises(KeyError):
        cache.popitem()


def test_rr_eviction_deterministic():
    # custom choice that always picks the first key
    cache = RRCache(2, choice=lambda seq: seq[0])
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3  # evicts first key
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


def test_rr_delitem_middle():
    cache = RRCache(5, choice=lambda seq: seq[0])
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    del cache["b"]
    assert "b" not in cache
    assert set(cache) == {"a", "c"}


def test_rr_clear():
    cache = RRCache(5, choice=lambda seq: seq[0])
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["x"] = 1
    assert cache["x"] == 1


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------


def test_ttl_basic():
    clock = PicklableTimer(0.0)
    cache = TTLCache(3, ttl=10, timer=clock)
    assert cache.ttl == 10
    assert cache.timer() == 0.0
    cache["a"] = 1
    assert cache["a"] == 1
    assert cache.currsize == 1


def test_ttl_expiration():
    clock = PicklableTimer(0.0)
    cache = TTLCache(3, ttl=10, timer=clock)
    cache["a"] = 1
    clock.advance(5)
    assert cache["a"] == 1
    clock.advance(6)  # now at 11 > 10
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_ttl_expire_returns_expired():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    cache["b"] = 2
    clock.advance(11)
    expired = cache.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(cache) == 0


def test_ttl_contains():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    assert "a" in cache
    assert "b" not in cache
    clock.advance(11)
    assert "a" not in cache


def test_ttl_lru_eviction():
    clock = PicklableTimer(0.0)
    cache = TTLCache(2, ttl=100, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    clock.advance(1)
    cache["a"]  # touch a
    clock.advance(1)
    cache["c"] = 3  # evicts b
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_ttl_popitem():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=100, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    clock.advance(1)
    cache["a"]  # touch a
    assert cache.popitem() == ("b", 2)


def test_ttl_popitem_empty():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=100, timer=clock)
    with pytest.raises(KeyError):
        cache.popitem()


def test_ttl_popitem_skips_expired():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    clock.advance(5)
    cache["b"] = 2
    clock.advance(6)  # a expired, b not
    key, value = cache.popitem()
    assert key == "b"
    assert value == 2
    assert len(cache) == 0


def test_ttl_delitem_expired():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    clock.advance(11)
    with pytest.raises(KeyError):
        del cache["a"]
    # underlying entries removed regardless
    assert len(cache) == 0


def test_ttl_delitem_not_expired():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache


def test_ttl_iteration_filters_expired():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    clock.advance(5)
    cache["b"] = 2
    clock.advance(6)  # a expired, b still valid
    assert list(cache) == ["b"]


def test_ttl_clear():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    cache.clear()
    assert len(cache) == 0
    cache["x"] = 1
    assert cache["x"] == 1


def test_ttl_get_pop_setdefault_with_timer():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=10, timer=clock)
    cache["a"] = 1
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.setdefault("c", 3) == 3
    assert cache.pop("a") == 1


def test_ttl_getsizeof_with_maxsize():
    clock = PicklableTimer(0.0)

    def getsizeof(v):
        return v

    cache = TTLCache(5, ttl=10, timer=clock, getsizeof=getsizeof)
    cache["a"] = 2
    cache["b"] = 3
    assert cache.currsize == 5
    cache["c"] = 4  # evict some to fit
    assert cache.currsize <= 5


def test_ttl_setstate_via_pickle():
    clock = PicklableTimer(0.0)
    cache = TTLCache(5, ttl=100, timer=clock)
    cache["a"] = 1
    cache["b"] = 2
    data = pickle.dumps(cache)
    restored = pickle.loads(data)
    assert restored["a"] == 1
    assert restored["b"] == 2
    assert restored.ttl == 100


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------


def make_ttu():
    def ttu(key, value, time):
        return time + value.get("ttl", 10)

    return ttu


def test_tlru_basic():
    clock = PicklableTimer(0.0)
    ttu = make_ttu()
    cache = TLRUCache(5, ttu=ttu, timer=clock)
    assert cache.ttu is ttu
    cache["a"] = {"ttl": 10, "v": 1}
    assert cache["a"] == {"ttl": 10, "v": 1}


def test_tlru_expiration():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 10}
    clock.advance(11)
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_tlru_expire_returns_expired():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 5}
    clock.advance(1)
    cache["b"] = {"ttl": 10}
    clock.advance(6)
    expired = cache.expire()
    assert [k for k, _ in expired] == ["a"]


def test_tlru_lru_eviction():
    clock = PicklableTimer(0.0)
    ttu = make_ttu()

    def ttu_long(k, v, t):
        return t + 100

    cache = TLRUCache(2, ttu=ttu_long, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    clock.advance(1)
    cache["a"]  # touch a
    clock.advance(1)
    cache["c"] = 3  # evict b
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_tlru_setitem_already_expired_removes():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 10}
    cache["a"] = {"ttl": -1}  # replacement expires immediately
    assert "a" not in cache
    assert len(cache) == 0


def test_tlru_setitem_already_expired_when_missing():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": -5}
    assert "a" not in cache
    assert len(cache) == 0


def test_tlru_update_replaces_item():
    clock = PicklableTimer(0.0)

    def ttu(key, value, time):
        return time + 1000

    cache = TLRUCache(5, ttu=ttu, timer=clock)
    cache["a"] = 1
    cache["a"] = 2
    assert cache["a"] == 2
    # heap contains old removed item, but len counts only live items
    assert len(cache) == 1


def test_tlru_delitem():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 10}
    del cache["a"]
    assert "a" not in cache


def test_tlru_delitem_expired_raises():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 10}
    clock.advance(11)
    with pytest.raises(KeyError):
        del cache["a"]


def test_tlru_popitem():
    clock = PicklableTimer(0.0)

    def ttu(k, v, t):
        return t + 1000

    cache = TLRUCache(5, ttu=ttu, timer=clock)
    cache["a"] = 1
    clock.advance(1)
    cache["b"] = 2
    clock.advance(1)
    cache["a"]  # touch a
    assert cache.popitem() == ("b", 2)


def test_tlru_popitem_empty():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    with pytest.raises(KeyError):
        cache.popitem()


def test_tlru_iteration_filters_expired_and_removed():
    clock = PicklableTimer(0.0)

    def ttu(k, v, t):
        return t + v

    cache = TLRUCache(5, ttu=ttu, timer=clock)
    cache["a"] = 1
    cache["b"] = 100
    clock.advance(2)  # a expired, b valid
    assert list(cache) == ["b"]


def test_tlru_heap_cleanup_branch():
    clock = PicklableTimer(0.0)

    def ttu(k, v, t):
        return t + 10000

    cache = TLRUCache(100, ttu=ttu, timer=clock)
    for i in range(10):
        cache[str(i)] = i
        cache[str(i)] = i  # replaces, leaving removed entries in heap
    # trigger cleanup via expire
    cache.expire(clock.time)
    assert len(cache) == 10


def test_tlru_clear():
    clock = PicklableTimer(0.0)
    cache = TLRUCache(5, ttu=make_ttu(), timer=clock)
    cache["a"] = {"ttl": 10}
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["x"] = {"ttl": 10}
    assert cache["x"] == {"ttl": 10}


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------


def test_cached_basic():
    cache = {}
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]
    assert f(3) == 6
    assert calls == [2, 3]


def test_cached_with_cache_object():
    cache = Cache(10)

    @cached(cache)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert cache["(1, (), frozenset())"] == 2 or len(cache) == 1


def test_cached_info_mapping():
    cache = {}

    @cached(cache, info=True)
    def f(x):
        return x

    info = f.cache_info()
    assert info == cachetools._CacheInfo(0, 0, None, 0)
    assert f(1) == 1
    assert f(1) == 1
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize is None
    assert info.currsize == 1


def test_cached_info_cache():
    cache = Cache(10)

    @cached(cache, info=True)
    def f(x):
        return x

    assert f(1) == 1
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 10
    assert info.currsize == 1


def test_cached_info_non_mapping():
    class CustomCache:
        def __init__(self):
            self.data = {}

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def __contains__(self, key):
            return key in self.data

        def __len__(self):
            return len(self.data)

    # Non-Mapping, non-Cache object -> maxsize=0, currsize=0 in info
    cache = CustomCache()

    @cached(cache, info=True)
    def f(x):
        return x

    assert f(1) == 1
    info = f.cache_info()
    assert info.maxsize == 0
    assert info.currsize == 0


def test_cached_clear():
    cache = Cache(10)

    @cached(cache)
    def f(x):
        return x

    f(1)
    f.cache_clear()
    assert len(cache) == 0


def test_cached_key_function():
    cache = {}
    calls = []

    def key(*args, **kwargs):
        return "static"

    @cached(cache, key=key)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    assert calls == [1]
    assert cache["static"] == 1


def test_cached_lock():
    import threading

    cache = {}
    lock = threading.RLock()

    @cached(cache, lock=lock)
    def f(x):
        return x * 3

    assert f(2) == 6
    assert f(2) == 6


def test_cached_unhashable_args():
    cache = {}

    @cached(cache, key=lambda *args, **kwargs: "constant")
    def f(x):
        return x

    assert f([1]) == [1]


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------


def test_cachedmethod_basic():
    class A:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            self.calls += 1
            return x * 2

    a = A()
    assert a.f(2) == 4
    assert a.f(2) == 4
    assert a.calls == 1
    assert len(a.cache) == 1


def test_cachedmethod_info():
    class A:
        def __init__(self):
            self.cache = Cache(10)

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x

    a = A()
    assert a.f(1) == 1
    assert a.f(1) == 1
    info = a.f.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    a.f.cache_clear()


def test_cachedmethod_info_invalid_cache():
    class A:
        def __init__(self):
            self.cache = 42

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x

    a = A()
    assert a.f(1) == 1
    with pytest.raises(TypeError):
        a.f.cache_info()


def test_cachedmethod_separate_instances():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            return x

    a1 = A()
    a2 = A()
    a1.f(1)
    a2.f(2)
    assert len(a1.cache) == 1
    assert len(a2.cache) == 1


def test_cachedmethod_shared_cache():
    shared = {}

    class A:
        @cachedmethod(lambda self: shared)
        def f(self, x):
            return x

    a1 = A()
    a2 = A()
    a1.f(1)
    a2.f(2)
    assert len(shared) == 2
