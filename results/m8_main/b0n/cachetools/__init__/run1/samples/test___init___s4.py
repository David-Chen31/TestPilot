import collections
import pickle
import pytest

from cachetools import (
    Cache,
    FIFOCache,
    LFUCache,
    LRUCache,
    RRCache,
    TLRUCache,
    TTLCache,
    cached,
    cachedmethod,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class CountingTimer:
    def __init__(self, time=0.0):
        self.time = time

    def __call__(self):
        return self.time


class DefaultCache(Cache):
    def __missing__(self, key):
        return key * 2


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_basic():
    c = Cache(maxsize=2)
    assert c.maxsize == 2
    assert c.currsize == 0
    assert len(c) == 0
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    assert c.currsize == 2
    assert sorted(c) == ["a", "b"]
    assert "a" in c
    assert "z" not in c


def test_cache_getsizeof_default():
    assert Cache.getsizeof(object()) == 1


def test_cache_size_accounting():
    sizes = {"a": 1, "b": 2, "c": 3}
    c = Cache(maxsize=10, getsizeof=sizes.__getitem__)
    c["a"] = "a"
    c["b"] = "b"
    assert c.currsize == 3
    del c["a"]
    assert c.currsize == 2
    c["c"] = "c"
    assert c.currsize == 5


def test_cache_eviction():
    c = Cache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert c.currsize == 2


def test_cache_update_existing_size():
    sizes = {"a": 1, "b": 2}
    c = Cache(maxsize=5, getsizeof=sizes.__getitem__)
    c["a"] = "a"
    c["b"] = "b"
    c["a"] = "a"  # no size change
    assert c.currsize == 3


def test_cache_update_size_increases():
    sizes = {"a": 2, "b": 3}
    c = Cache(maxsize=5, getsizeof=sizes.__getitem__)
    c["a"] = "a"
    c["b"] = "b"
    assert c.currsize == 5
    c["a"] = "a"  # diffsize = 0, no eviction needed
    assert c.currsize == 5


def test_cache_negative_maxsize():
    with pytest.raises(ValueError):
        Cache(maxsize=-1)


def test_cache_value_too_large():
    c = Cache(maxsize=1)
    sizes = {1: 2}
    c.getsizeof = sizes.__getitem__
    with pytest.raises(ValueError):
        c["x"] = 1


def test_cache_negative_size_value():
    c = Cache(maxsize=10)
    c.getsizeof = lambda v: -1
    with pytest.raises(ValueError):
        c["x"] = 1


def test_cache_missing():
    c = Cache(maxsize=2)
    with pytest.raises(KeyError):
        c["missing"]


def test_cache_subclass_missing():
    c = DefaultCache(maxsize=2)
    assert c[5] == 10
    assert len(c) == 0


def test_cache_get():
    c = Cache(maxsize=2)
    assert c.get("x") is None
    assert c.get("x", 42) == 42
    c["x"] = 1
    assert c.get("x") == 1
    assert c.get("x", 42) == 1


def test_cache_get_subclass_missing():
    c = DefaultCache(maxsize=2)
    assert c.get(3) == 6
    assert len(c) == 0  # missing did not store


def test_cache_pop():
    c = Cache(maxsize=2)
    with pytest.raises(KeyError):
        c.pop("x")
    assert c.pop("x", 7) == 7
    c["x"] = 1
    assert c.pop("x") == 1
    assert len(c) == 0


def test_cache_pop_subclass_missing():
    c = DefaultCache(maxsize=2)
    assert c.pop(3) == 6
    assert len(c) == 0


def test_cache_setdefault():
    c = Cache(maxsize=2)
    assert c.setdefault("x", 1) == 1
    assert c["x"] == 1
    assert c.setdefault("x", 2) == 1
    assert c.setdefault("y") is None
    assert c["y"] is None


def test_cache_clear():
    c = Cache(maxsize=4)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_cache_repr():
    c = Cache(maxsize=4)
    c["a"] = 1
    r = repr(c)
    assert "Cache" in r
    assert "maxsize=4" in r
    assert "currsize=1" in r


def test_cache_getsizeof_override():
    c = Cache(maxsize=10, getsizeof=lambda v: len(v))
    c["x"] = "abc"
    assert c.currsize == 3


def test_cache_empty_popitem():
    with pytest.raises(KeyError):
        Cache(maxsize=1).popitem()


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

def test_fifo_basic():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert c["b"] == 2
    assert c["c"] == 3


def test_fifo_update_moves_to_end():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10  # now b is oldest
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 10
    assert c["c"] == 3


def test_fifo_popitem():
    c = FIFOCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_fifo_clear():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_fifo_delete():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    c["c"] = 3
    assert c["b"] == 2
    assert c["c"] == 3


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

def test_lfu_basic_eviction():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"]  # a used twice
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 1
    assert c["c"] == 3


def test_lfu_least_frequent():
    c = LFUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"]
    c["b"]
    c["a"]
    c["d"] = 4
    assert "c" not in c


def test_lfu_popitem_empty():
    with pytest.raises(KeyError):
        LFUCache(maxsize=1).popitem()


def test_lfu_popitem():
    c = LFUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["a"]
    k, v = c.popitem()
    assert k == "b" and v == 2
    assert "b" not in c


def test_lfu_clear():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    c["c"] = 3
    assert c["c"] == 3
    assert c.popitem() == ("c", 3)


def test_lfu_delete():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    c["c"] = 3
    assert c["b"] == 2
    assert c["c"] == 3


def test_lfu_missing_does_not_touch():
    class C(LFUCache):
        def __missing__(self, key):
            return key

    c = C(maxsize=2)
    assert c[5] == 5
    assert len(c) == 0


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

def test_lru_basic():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"]
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 1
    assert c["c"] == 3


def test_lru_popitem():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        c.popitem()
        c.popitem()


def test_lru_empty_popitem():
    with pytest.raises(KeyError):
        LRUCache(maxsize=1).popitem()


def test_lru_clear():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    c["x"] = 5
    assert c.popitem() == ("x", 5)


def test_lru_delete():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    c["c"] = 3
    assert c["b"] == 2
    assert c["c"] == 3


def test_lru_missing_does_not_touch():
    class C(LRUCache):
        def __missing__(self, key):
            return key + 1

    c = C(maxsize=2)
    assert c[1] == 2
    assert len(c) == 0


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

def test_rr_choice_property():
    def choice(seq):
        return seq[0]

    c = RRCache(maxsize=2, choice=choice)
    assert c.choice is choice


def test_rr_deterministic_choice():
    c = RRCache(maxsize=2, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c


def test_rr_popitem():
    c = RRCache(maxsize=3, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    k, v = c.popitem()
    assert k == "a" and v == 1


def test_rr_empty_popitem():
    with pytest.raises(KeyError):
        RRCache(maxsize=1, choice=lambda seq: seq[0]).popitem()


def test_rr_delete_index_maintenance():
    c = RRCache(maxsize=3, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    del c["a"]
    # remaining keys still reachable
    assert c["b"] == 2
    assert c["c"] == 3
    c["d"] = 4
    assert len(c) == 3


def test_rr_clear():
    c = RRCache(maxsize=3, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

def make_ttl(ttl, maxsize=10, timer=None):
    timer = timer or CountingTimer()
    return TTLCache(maxsize=maxsize, ttl=ttl, timer=timer), timer


def test_ttl_basic():
    c, t = make_ttl(10)
    assert c.ttl == 10
    c["a"] = 1
    assert c["a"] == 1
    t.time = 5
    assert c["a"] == 1
    t.time = 10
    assert "a" not in c


def test_ttl_expire():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 2
    c["b"] = 2
    t.time = 6
    expired = c.expire()
    assert expired == [("a", 1)]
    assert "a" not in c
    assert "b" in c


def test_ttl_getitem_after_expiry():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 10
    with pytest.raises(KeyError):
        c["a"]


def test_ttl_getitem_missing_returns_missing():
    class C(TTLCache):
        def __missing__(self, key):
            return "missing"

    timer = CountingTimer()
    c = C(maxsize=2, ttl=5, timer=timer)
    assert c["x"] == "missing"
    assert len(c) == 0


def test_ttl_contains():
    c, t = make_ttl(5)
    c["a"] = 1
    assert "a" in c
    t.time = 5
    assert "a" not in c


def test_ttl_len_expires():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 3
    c["b"] = 2
    t.time = 6
    assert len(c) == 1


def test_ttl_currsize_expires():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 6
    assert c.currsize == 0


def test_ttl_popitem():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)


def test_ttl_popitem_empty():
    c, t = make_ttl(5)
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_popitem_after_expiry():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 10
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_pop_and_get_expire():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 2
    assert c.get("a") == 1
    assert c.pop("a") == 1
    c["b"] = 2
    t.time = 10
    assert c.get("b") is None
    assert c.pop("b", "x") == "x"


def test_ttl_setdefault():
    c, t = make_ttl(5)
    assert c.setdefault("a", 1) == 1
    assert c["a"] == 1
    assert c.setdefault("a", 2) == 1


def test_ttl_delete_expired_raises():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 10
    with pytest.raises(KeyError):
        del c["a"]


def test_ttl_iter():
    c, t = make_ttl(10)
    c["a"] = 1
    t.time = 2
    c["b"] = 2
    t.time = 5
    assert set(c) == {"a", "b"}
    t.time = 20
    assert set(c) == set()


def test_ttl_clear():
    c, t = make_ttl(5)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    c["b"] = 2
    assert c["b"] == 2


def test_ttl_repr():
    c, t = make_ttl(5)
    c["a"] = 1
    assert "TTLCache" in repr(c)


def test_ttl_update_ttl():
    c, t = make_ttl(5)
    c["a"] = 1
    t.time = 3
    c["a"] = 2
    t.time = 6
    assert c["a"] == 2
    t.time = 8
    assert "a" not in c


def test_ttl_timer_property():
    timer = CountingTimer()
    c = TTLCache(maxsize=2, ttl=5, timer=timer)
    assert c.timer() == 0.0


def test_ttl_pickle():
    c, t = make_ttl(5)
    c["a"] = 1
    data = pickle.dumps(c)
    c2 = pickle.loads(data)
    assert c2["a"] == 1


def test_ttl_timer_nesting():
    timer = CountingTimer(0)
    tt = TTLCache._Timer(timer)
    with tt as time1:
        timer.time = 5
        with tt as time2:
            assert time1 == time2 == 0
        assert tt() == 0
    assert tt() == 5


def test_ttl_timer_reduce():
    timer = CountingTimer()
    tt = TTLCache._Timer(timer)
    cls, args = tt.__reduce__()
    assert args[0] is timer


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

def make_tlru(ttu, maxsize=10, timer=None):
    timer = timer or CountingTimer()
    return TLRUCache(maxsize=maxsize, ttu=ttu, timer=timer), timer


def test_tlru_basic():
    c, t = make_tlru(lambda k, v, now: now + 5)
    assert c.ttu is not None
    c["a"] = 1
    assert c["a"] == 1
    t.time = 5
    assert "a" not in c


def test_tlru_expire():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    t.time = 2
    c["b"] = 2
    t.time = 6
    expired = c.expire()
    assert expired == [("a", 1)]
    assert "b" in c


def test_tlru_already_expired_set_removes():
    c, t = make_tlru(lambda k, v, now: now - 1)
    c["a"] = 1
    assert "a" not in c
    assert len(c) == 0


def test_tlru_update_replaces():
    times = {"a": 5}
    c, t = make_tlru(lambda k, v, now: now + times.get(k, 10))
    c["a"] = 1
    t.time = 3
    c["a"] = 2
    assert c["a"] == 2


def test_tlru_delete_expired():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    t.time = 10
    with pytest.raises(KeyError):
        del c["a"]


def test_tlru_delete_ok():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    del c["a"]
    assert "a" not in c


def test_tlru_contains():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    assert "a" in c
    t.time = 5
    assert "a" not in c


def test_tlru_iter():
    c, t = make_tlru(lambda k, v, now: now + 10)
    c["a"] = 1
    t.time = 3
    c["b"] = 2
    assert set(c) == {"a", "b"}
    t.time = 100
    assert set(c) == set()


def test_tlru_popitem():
    c, t = make_tlru(lambda k, v, now: now + 10)
    c["a"] = 1
    t.time = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)


def test_tlru_popitem_empty():
    c, t = make_tlru(lambda k, v, now: now + 10)
    with pytest.raises(KeyError):
        c.popitem()


def test_tlru_clear():
    c, t = make_tlru(lambda k, v, now: now + 10)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    c["b"] = 2
    assert c["b"] == 2


def test_tlru_len_currsize_expire():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    t.time = 6
    assert len(c) == 0
    c["b"] = 2
    assert c.currsize == 1


def test_tlru_get_pop_setdefault():
    c, t = make_tlru(lambda k, v, now: now + 5)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.pop("a") == 1
    c["b"] = 2
    t.time = 6
    assert c.get("b") is None
    assert c.setdefault("c", 3) == 3


def test_tlru_missing():
    class C(TLRUCache):
        def __missing__(self, key):
            return "m"

    timer = CountingTimer()
    c = C(maxsize=2, ttu=lambda k, v, now: now + 5, timer=timer)
    assert c["x"] == "m"


def test_tlru_heap_cleanup():
    c, t = make_tlru(lambda k, v, now: now + 100, maxsize=100)
    for i in range(10):
        c[i] = i
    # cause many removals to trigger heap cleanup
    for i in range(10):
        del c[i]
    c[99] = 99
    assert c[99] == 99


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------

def test_cached_basic():
    cache = {}

    @cached(cache)
    def f(x):
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert len(cache) == 1


def test_cached_with_cache_object():
    cache = LRUCache(maxsize=2)

    @cached(cache)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(2) == 3
    assert f(3) == 4
    assert len(cache) == 2


def test_cached_info():
    cache = Cache(maxsize=10)

    @cached(cache, info=True)
    def f(x):
        return x

    assert f(1) == 1
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 1
    f(1)
    info = f.cache_info()
    assert info.hits == 1
    assert info.maxsize == 10


def test_cached_info_mapping():
    cache = {}

    @cached(cache, info=True)
    def f(x):
        return x

    f(1)
    f(1)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize is None


def test_cached_clear_cache():
    cache = {}

    @cached(cache)
    def f(x):
        return x

    f(1)
    f.cache_clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------

def test_cachedmethod_basic():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            return x * 3

    a = A()
    assert a.f(2) == 6
    assert a.f(2) == 6
    assert len(a.cache) == 1


def test_cachedmethod_info():
    class A:
        def __init__(self):
            self.cache = Cache(maxsize=10)

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x

    a = A()
    a.f(1)
    info = a.f.cache_info()
    assert info.misses == 1
    assert info.maxsize == 10
    a.f(1)
    info = a.f.cache_info()
    assert info.hits == 1


def test_cachedmethod_info_mapping():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x

    a = A()
    a.f(1)
    info = a.f.cache_info()
    assert info.misses == 1
    assert info.maxsize is None


def test_cachedmethod_clear():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            return x

    a = A()
    a.f(1)
    a.f.cache_clear()
    assert len(a.cache) == 0
