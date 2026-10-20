import collections
import pickle
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


class _Clock:
    """Deterministic fake clock."""

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, delta):
        self.value += delta


def _name(key):
    return key


# ---------------------------------------------------------------------------
# Cache (base class)
# ---------------------------------------------------------------------------

def test_cache_basic_set_get_len():
    c = Cache(10)
    assert len(c) == 0
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    assert c["b"] == 2
    assert len(c) == 2
    assert c.currsize == 2
    assert c.maxsize == 10
    assert "a" in c
    assert "x" not in c


def test_cache_missing_key():
    c = Cache(10)
    with pytest.raises(KeyError):
        c["nope"]


def test_cache_negative_maxsize():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_getsizeof():
    c = Cache(10, getsizeof=lambda v: v)
    c["a"] = 3
    c["b"] = 4
    assert c.currsize == 7
    c["a"] = 5
    assert c.currsize == 9


def test_cache_value_too_large():
    c = Cache(5)
    with pytest.raises(ValueError):
        c["a"] = object()  # default getsizeof returns 1 -> fine
    c2 = Cache(5, getsizeof=lambda v: 10)
    with pytest.raises(ValueError):
        c2["a"] = 1


def test_cache_negative_size():
    c = Cache(5, getsizeof=lambda v: -1)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_eviction():
    c = Cache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["d"] = 4  # evicts oldest insertion (dict order)
    assert "a" not in c
    assert set(c) == {"b", "c", "d"}


def test_cache_replace_existing():
    c = Cache(3, getsizeof=lambda v: v)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"] = 2
    assert c.currsize == 7
    assert "b" not in c  # evicted to make room


def test_cache_delitem():
    c = Cache(10)
    c["a"] = 1
    del c["a"]
    assert "a" not in c
    assert c.currsize == 0
    with pytest.raises(KeyError):
        del c["a"]


def test_cache_get():
    c = Cache(10)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("x") is None
    assert c.get("x", 42) == 42


def test_cache_pop():
    c = Cache(10)
    c["a"] = 1
    assert c.pop("a") == 1
    assert "a" not in c
    with pytest.raises(KeyError):
        c.pop("a")
    assert c.pop("a", "dflt") == "dflt"


def test_cache_setdefault():
    c = Cache(10)
    assert c.setdefault("a", 1) == 1
    assert c["a"] == 1
    assert c.setdefault("a", 2) == 1


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
    assert r.startswith("Cache(")
    assert "maxsize=5" in r
    assert "currsize=1" in r


def test_cache_getsizeof_static():
    assert Cache.getsizeof(object()) == 1


def test_cache_missing_custom_subclass():
    class MyCache(Cache):
        def __missing__(self, key):
            return "missing"

    c = MyCache(10)
    # __missing__ doesn't store
    assert c["x"] == "missing"
    assert "x" not in c
    # get() should not return __missing__ result
    assert c.get("x", "default") == "default"


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

def test_fifo_eviction():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["d"] = 4
    assert "a" not in c
    assert set(c) == {"b", "c", "d"}


def test_fifo_reinsert_moves_to_end():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10  # reinsert a -> b becomes first
    c["c"] = 3
    c["d"] = 4  # evict b
    assert "b" not in c
    assert set(c) == {"a", "c", "d"}


def test_fifo_popitem():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_fifo_delete_and_clear():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

def test_lfu_eviction_least_frequent():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    _ = c["a"]
    _ = c["b"]  # a,b used twice, c once
    c["d"] = 4
    assert "c" not in c
    assert set(c) == {"a", "b", "d"}


def test_lfu_popitem():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]  # a used twice
    key, value = c.popitem()
    assert key == "b"
    assert value == 2
    with pytest.raises(KeyError):
        LFUCache(0).popitem()


def test_lfu_delete_and_clear():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_count_increment_same_link():
    c = LFUCache(10)
    c["a"] = 1
    _ = c["a"]
    _ = c["a"]
    assert c.popitem() == ("a", 1)


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

def test_lru_eviction():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    _ = c["a"]  # a most recently used
    c["d"] = 4  # evict b
    assert "b" not in c
    assert set(c) == {"a", "c", "d"}


def test_lru_setitem_touches():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10  # touch a
    c["c"] = 3
    c["d"] = 4  # evict b
    assert "b" not in c


def test_lru_popitem():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_delete_clear():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

def test_rr_choice_property():
    def choose(keys):
        return keys[0]

    c = RRCache(3, choice=choose)
    assert c.choice is choose


def test_rr_popitem_deterministic():
    def choose(keys):
        return keys[0]

    c = RRCache(3, choice=choose)
    c["a"] = 1
    c["b"] = 2
    key, value = c.popitem()
    assert (key, value) == ("a", 1)
    with pytest.raises(KeyError):
        RRCache(0).popitem()


def test_rr_eviction():
    def choose(keys):
        return keys[0]

    c = RRCache(2, choice=choose)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3  # popitem selects "a"
    assert "a" not in c
    assert set(c) == {"b", "c"}


def test_rr_delete_clear():
    c = RRCache(3)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    c.clear()
    assert len(c) == 0


 # ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

def test_ttl_basic_and_expiry():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(5)
    assert "a" in c
    assert c["a"] == 1
    clock.advance(5)
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_ttl_ttl_property():
    c = TTLCache(3, ttl=10, timer=_Clock())
    assert c.ttl == 10


def test_ttl_expire_returns_pairs():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    clock.advance(10)
    expired = c.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_ttl_eviction_lru():
    clock = _Clock()
    c = TTLCache(2, ttl=100, timer=clock)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_ttl_popitem_no_expired():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        TTLCache(0, ttl=10, timer=clock).popitem()


def test_ttl_delete_expired_raises():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(10)
    with pytest.raises(KeyError):
        del c["a"]
    assert "a" not in c


def test_ttl_iter_skips_expired():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(5)
    c["b"] = 2
    clock.advance(5)
    assert list(c) == ["b"]


def test_ttl_clear_and_len():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    assert len(c) == 1
    c.clear()
    assert len(c) == 0


def test_ttl_pickle_roundtrip():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    data = pickle.dumps(c)
    c2 = pickle.loads(data)
    assert list(c2) == ["a"]
    assert c2["a"] == 1


def test_ttl_repr_expires():
    clock = _Clock()
    c = TTLCache(3, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(20)
    r = repr(c)
    assert "currsize=0" in r


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

def test_tlru_ttu_property():
    def ttu(key, value, time):
        return time + 5

    c = TLRUCache(3, ttu=ttu, timer=_Clock())
    assert c.ttu is ttu


def test_tlru_expiry():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    clock.advance(5)
    assert c["a"] == 1
    clock.advance(5)
    assert "a" not in c


def test_tlru_expire_returns_pairs():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    clock.advance(10)
    expired = c.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_tlru_lru_eviction():
    clock = _Clock()
    c = TLRUCache(2, ttu=lambda k, v, t: t + 100, timer=clock)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_tlru_popitem():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 100, timer=clock)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        TLRUCache(0, ttu=lambda k, v, t: t + 1, timer=clock).popitem()


def test_tlru_delete_expired_raises():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    clock.advance(10)
    with pytest.raises(KeyError):
        del c["a"]
    assert "a" not in c


def test_tlru_set_expired_removes_existing():
    clock = _Clock()
    expiry = {"a": 100}
    c = TLRUCache(3, ttu=lambda k, v, t: expiry[k], timer=clock)
    c["a"] = 1
    expiry["a"] = 0
    c["a"] = 2  # already expired -> removes
    assert "a" not in c


def test_tlru_iter_skips_expired_and_removed():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    clock.advance(5)
    c["b"] = 2
    clock.advance(5)
    assert list(c) == ["b"]


def test_tlru_clear():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    c.clear()
    assert len(c) == 0


def test_tlru_heap_cleanup():
    clock = _Clock()
    c = TLRUCache(100, ttu=lambda k, v, t: t + 1000, timer=clock)
    for i in range(10):
        c[i] = i
    for i in range(10):
        c[i] = i + 100  # mark old items removed
    assert len(c) == 10
    # trigger cleanup via expire
    c.expire()
    assert len(c) == 10


def test_tlru_repr():
    clock = _Clock()
    c = TLRUCache(3, ttu=lambda k, v, t: t + 10, timer=clock)
    c["a"] = 1
    assert "currsize=1" in repr(c)


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------

def test_cached_basic():
    c = Cache(10)
    calls = []

    @cached(c)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls == [3]


def test_cached_with_lock():
    import threading

    c = Cache(10)
    lock = threading.Lock()

    @cached(c, lock=lock)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(1) == 2


def test_cached_with_condition():
    import threading

    c = Cache(10)
    lock = threading.Lock()
    cond = threading.Condition(lock)

    @cached(c, lock=lock, condition=cond)
    def f(x):
        return x + 1

    assert f(1) == 2


def test_cached_info():
    c = Cache(10)

    @cached(c, info=True)
    def f(x):
        return x

    f(1)
    f(1)
    f(2)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 10
    assert info.currsize == 2


def test_cached_info_mapping():
    c = {}

    @cached(c, info=True)
    def f(x):
        return x

    f(1)
    info = f.cache_info()
    assert info.maxsize is None
    assert info.currsize == 1


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------

def test_cachedmethod_basic():
    calls = []

    class Foo:
        def __init__(self):
            self._cache = Cache(10)

        @cachedmethod(lambda self: self._cache)
        def compute(self, x):
            calls.append(x)
            return x * 3

    foo = Foo()
    assert foo.compute(2) == 6
    assert foo.compute(2) == 6
    assert calls == [2]


def test_cachedmethod_info():
    class Foo:
        def __init__(self):
            self._cache = Cache(10)

        @cachedmethod(lambda self: self._cache, info=True)
        def compute(self, x):
            return x

    foo = Foo()
    foo.compute(1)
    foo.compute(1)
    info = foo.compute.cache_info(foo)
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 10


def test_cachedmethod_info_mapping():
    class Foo:
        def __init__(self):
            self._cache = {}

        @cachedmethod(lambda self: self._cache, info=True)
        def compute(self, x):
            return x

    foo = Foo()
    foo.compute(1)
    info = foo.compute.cache_info(foo)
    assert info.maxsize is None
    assert info.currsize == 1


def test_cachedmethod_info_invalid_cache():
    class Foo:
        def __init__(self):
            self._cache = object()

        @cachedmethod(lambda self: self._cache, info=True)
        def compute(self, x):
            return x

    foo = Foo()
    with pytest.raises(TypeError):
        foo.compute.cache_info(foo)
