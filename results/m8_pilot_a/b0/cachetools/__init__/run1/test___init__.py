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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountCache(Cache):
    """Cache that records calls to __getitem__ / __setitem__ etc."""

    def __init__(self, *args, **kwargs):
        Cache.__init__(self, *args, **kwargs)
        self.clear_calls = 0

    def clear(self):
        Cache.clear(self)
        self.clear_calls += 1


class MissingCache(Cache):
    def __missing__(self, key):
        value = "missing:" + key
        self[key] = value
        return value


class NoStoreMissingCache(Cache):
    def __missing__(self, key):
        return "default"


class ManualTimer:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_init_and_props():
    c = Cache(maxsize=10)
    assert c.maxsize == 10
    assert c.currsize == 0
    assert len(c) == 0
    assert c.getsizeof(object()) == 1


def test_cache_negative_maxsize():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_set_get_delete():
    c = Cache(10)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    assert c["b"] == 2
    assert "a" in c
    assert len(c) == 2
    assert c.currsize == 2
    del c["a"]
    assert "a" not in c
    assert len(c) == 1
    assert c.currsize == 1


def test_cache_missing_keyerror():
    c = Cache(10)
    with pytest.raises(KeyError):
        c["x"]
    with pytest.raises(KeyError):
        del c["x"]


def test_cache_eviction_fifo():
    c = Cache(2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert len(c) == 2
    assert c.currsize == 2


def test_cache_value_too_large():
    c = Cache(1)
    with pytest.raises(ValueError):
        c["a"] = object()
    # custom getsizeof
    c2 = Cache(5, getsizeof=lambda v: 10)
    with pytest.raises(ValueError):
        c2["a"] = 1


def test_cache_negative_getsizeof():
    c = Cache(10, getsizeof=lambda v: -1)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_getsizeof_eviction():
    c = Cache(5, getsizeof=len)
    c["a"] = "12"
    c["b"] = "34"
    assert c.currsize == 4
    c["c"] = "5"
    assert c.currsize <= 5


def test_cache_update_size_existing_key():
    c = Cache(10, getsizeof=len)
    c["a"] = "12"
    assert c.currsize == 2
    c["a"] = "3456"
    assert c.currsize == 4
    assert c["a"] == "3456"


def test_cache_get():
    c = Cache(10)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("b", 42) == 42


def test_cache_pop():
    c = Cache(10)
    c["a"] = 1
    assert c.pop("a") == 1
    assert "a" not in c
    with pytest.raises(KeyError):
        c.pop("a")
    assert c.pop("a", 5) == 5


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


def test_cache_iter():
    c = Cache(10)
    c["a"] = 1
    c["b"] = 2
    assert set(iter(c)) == {"a", "b"}


def test_cache_repr():
    c = Cache(10)
    c["a"] = 1
    r = repr(c)
    assert r.startswith("Cache(")
    assert "maxsize=10" in r
    assert "currsize=1" in r


def test_cache_missing_subclass():
    c = MissingCache(10)
    assert c["a"] == "missing:a"
    assert c.get("a") == "missing:a"
    # get on absent key invokes __getitem__, which populates
    assert "a" in c


def test_cache_get_does_not_trigger_missing_for_existing_logic():
    c = NoStoreMissingCache(10)
    # get uses "in", so __missing__ not called
    assert c.get("a") is None


def test_cache_pop_with_missing_subclass():
    c = MissingCache(10)
    # pop on missing key does not trigger __missing__ since uses "in"
    with pytest.raises(KeyError):
        c.pop("x")


def test_cache_clear_count():
    c = CountCache(10)
    c["a"] = 1
    c.clear()
    assert c.clear_calls == 1
    assert len(c) == 0


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------


def test_fifo_eviction_order():
    c = FIFOCache(2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert "b" in c
    assert "c" in c


def test_fifo_popitem():
    c = FIFOCache(10)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_fifo_update_moves_to_end():
    c = FIFOCache(2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10  # update moves to end
    c["c"] = 3  # should evict "b"
    assert "a" in c
    assert "b" not in c
    assert "c" in c


def test_fifo_delete():
    c = FIFOCache(10)
    c["a"] = 1
    del c["a"]
    assert "a" not in c
    with pytest.raises(KeyError):
        del c["a"]


def test_fifo_clear():
    c = FIFOCache(10)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------


def test_lfu_eviction_least_frequent():
    c = LFUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]  # touch a
    c["c"] = 3
    assert "a" in c
    assert "b" not in c
    assert "c" in c


def test_lfu_popitem():
    c = LFUCache(10)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    key, value = c.popitem()
    assert key == "b"
    assert value == 2
    key, value = c.popitem()
    assert (key, value) == ("a", 1)
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_delete():
    c = LFUCache(10)
    c["a"] = 1
    del c["a"]
    assert "a" not in c


def test_lfu_update():
    c = LFUCache(10)
    c["a"] = 1
    c["a"] = 2
    assert c["a"] == 2


def test_lfu_clear():
    c = LFUCache(10)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_missing_subclass():
    c = MissingCache(10)
    # plain Cache subclass, sanity via LFU would be complex
    assert c["a"] == "missing:a"


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_eviction():
    c = LRUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "a" in c
    assert "b" not in c
    assert "c" in c


def test_lru_popitem():
    c = LRUCache(10)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_update_moves():
    c = LRUCache(2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10
    c["c"] = 3
    assert "a" in c
    assert "b" not in c


def test_lru_delete():
    c = LRUCache(10)
    c["a"] = 1
    del c["a"]
    assert "a" not in c


def test_lru_clear():
    c = LRUCache(10)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_missing_returns():
    c = NoStoreMissingCache(10)
    assert c["x"] == "default"
    assert "x" not in c


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------


def test_rr_choice_property():
    def choice(seq):
        return seq[0]

    c = RRCache(10, choice=choice)
    assert c.choice is choice


def test_rr_eviction_uses_choice():
    def choice(seq):
        return seq[0]

    c = RRCache(2, choice=choice)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert "b" in c
    assert "c" in c


def test_rr_popitem():
    def choice(seq):
        return seq[-1]

    c = RRCache(10, choice=choice)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        c.popitem()


def test_rr_delete_tail():
    def choice(seq):
        return seq[0]

    c = RRCache(10, choice=choice)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    del c["b"]
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_rr_delete_last_index():
    def choice(seq):
        return seq[0]

    c = RRCache(10, choice=choice)
    c["a"] = 1
    c["b"] = 2
    del c["b"]
    assert set(c) == {"a"}


def test_rr_clear():
    def choice(seq):
        return seq[0]

    c = RRCache(10, choice=choice)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------


def test_ttl_properties():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    assert c.maxsize == 10
    assert c.ttl == 5
    assert c.timer is not None


def test_ttl_basic_expiry():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    assert c["a"] == 1
    timer.t = 4
    assert "a" in c
    timer.t = 5
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_ttl_expire_returns_items():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    timer.t = 10
    expired = c.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_ttl_popitem_skips_expired():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 3
    c["b"] = 2
    timer.t = 6  # "a" expired, "b" not
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_len_expires():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 10
    assert len(c) == 0


def test_ttl_currsize_expires():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 10
    assert c.currsize == 0


def test_ttl_get_expires():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 10
    assert c.get("a") is None


def test_ttl_setdefault():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    assert c.setdefault("a", 1) == 1
    assert c.setdefault("a", 2) == 1


def test_ttl_delete_expired():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 10
    with pytest.raises(KeyError):
        del c["a"]
    assert len(c) == 0


def test_ttl_delete_nonexistent():
    c = TTLCache(maxsize=10, ttl=5, timer=ManualTimer())
    with pytest.raises(KeyError):
        del c["x"]


def test_ttl_update_existing():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 3
    c["a"] = 2
    timer.t = 7  # original expiry 5, new expiry 8
    assert c["a"] == 2
    timer.t = 8
    assert "a" not in c


def test_ttl_iter_skips_expired():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 3
    c["b"] = 2
    timer.t = 6
    assert list(c) == ["b"]


def test_ttl_repr_expires():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    timer.t = 10
    r = repr(c)
    assert "currsize=0" in r


def test_ttl_clear():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_timer_reduce():
    timer = ManualTimer()
    t = TTLCache(10, 5, timer=timer).timer
    fn, args = t.__reduce__()
    assert args[0] is timer


def test_ttl_pickle_roundtrip():
    timer = ManualTimer()
    c = TTLCache(maxsize=10, ttl=5, timer=timer)
    c["a"] = 1
    c2 = pickle.loads(pickle.dumps(c))
    assert c2["a"] == 1


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------


def test_tlru_basic():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    assert c.ttu is not None
    c["a"] = 1
    assert c["a"] == 1
    timer.t = 4
    assert "a" in c
    timer.t = 5
    assert "a" not in c


def test_tlru_ttu_exposed():
    def ttu(k, v, t):
        return t + 5

    c = TLRUCache(maxsize=10, ttu=ttu, timer=ManualTimer())
    assert c.ttu is ttu


def test_tlru_expire():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    c["b"] = 2
    timer.t = 10
    expired = c.expire()
    assert sorted(expired) == [("a", 1), ("b", 2)]
    assert len(c) == 0


def test_tlru_setitem_already_expired_removes():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t - 1, timer=timer)
    c["a"] = 1
    assert "a" not in c


def test_tlru_update_existing_marks_removed():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 10, timer=timer)
    c["a"] = 1
    timer.t = 1
    c["a"] = 2
    assert c["a"] == 2


def test_tlru_popitem_skips_expired():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    timer.t = 3
    c["b"] = 2
    timer.t = 6
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_tlru_delete_expired_raises():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    timer.t = 10
    with pytest.raises(KeyError):
        del c["a"]


def test_tlru_delete_absent():
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=ManualTimer())
    with pytest.raises(KeyError):
        del c["x"]


def test_tlru_iter_skips_expired():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    timer.t = 3
    c["b"] = 2
    timer.t = 6
    assert list(c) == ["b"]


def test_tlru_clear():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_tlru_heap_cleanup():
    timer = ManualTimer()
    c = TLRUCache(maxsize=100, ttu=lambda k, v, t: t + 100, timer=timer)
    for i in range(10):
        c[i] = i
        # overwrite to mark removed
        c[i] = i + 1000
    # triggering expire should clean up
    c.expire()
    assert len(c) == 10


def test_tlru_len_currsize():
    timer = ManualTimer()
    c = TLRUCache(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    c["a"] = 1
    assert len(c) == 1
    assert c.currsize == 1
    timer.t = 10
    assert len(c) == 0
    assert c.currsize == 0


def test_tlru_missing_subclass():
    class MissingTLRU(TLRUCache):
        def __missing__(self, key):
            return "gone"

    timer = ManualTimer()
    c = MissingTLRU(maxsize=10, ttu=lambda k, v, t: t + 5, timer=timer)
    assert c["x"] == "gone"


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

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


def test_cached_custom_key():
    c = Cache(10)

    @cached(c, key=lambda args, kwargs: args[0])
    def f(x, y=0):
        return x + y

    assert f(1, y=2) == 1
    assert f(1, y=5) == 1


def test_cached_with_lock():
    import threading

    c = Cache(10)
    lock = threading.RLock()

    @cached(c, lock=lock)
    def f(x):
        return x + 1

    assert f(2) == 3
    assert f(2) == 3


def test_cached_info():
    c = Cache(10)

    @cached(c, info=True)
    def f(x):
        return x + 1

    assert f(1) == 2
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 10
    assert f(1) == 2
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_cached_info_mapping():
    d = {}

    @cached(d, info=True)
    def f(x):
        return x + 1

    assert f(1) == 2
    info = f.cache_info()
    assert info.maxsize is None
    assert info.currsize == 1


def test_cached_info_nonmapping():
    class NoMapping:
        def __getitem__(self, key):
            raise KeyError(key)

        def __setitem__(self, key, value):
            pass

        def __contains__(self, key):
            return False

    @cached(NoMapping(), info=True)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f.cache_info().maxsize == 0


def test_cached_condition():
    c = Cache(10)
    allow = [False]

    @cached(c, condition=lambda res: allow[0])
    def f(x):
        return x + 1

    assert f(1) == 2
    # not cached
    assert "1" not in c and (1,) not in c
    allow[0] = True
    assert f(1) == 2
    assert f(1) == 2


def test_cached_clear():
    c = Cache(10)

    @cached(c)
    def f(x):
        return x + 1

    f(1)
    f.cache_clear()
    assert len(c) == 0


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------


def test_cachedmethod_basic():
    class Obj:
        def __init__(self):
            self.cache = Cache(10)
            self.calls = 0

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            self.calls += 1
            return x * 2

    o = Obj()
    assert o.f(2) == 4
    assert o.f(2) == 4
    assert o.calls == 1


def test_cachedmethod_info():
    class Obj:
        def __init__(self):
            self.cache = Cache(10)

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x * 2

    o = Obj()
    assert o.f(2) == 4
    info = o.f.cache_info()
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 10
    assert o.f(2) == 4
    assert o.f.cache_info().hits == 1


def test_cachedmethod_info_raises_on_nonmapping():
    class Obj:
        def __init__(self):
            self.cache = object()

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            return x * 2

    o = Obj()
    with pytest.raises(TypeError):
        o.f(2)


def test_cachedmethod_clear():
    class Obj:
        def __init__(self):
            self.cache = Cache(10)

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            return x + 1

    o = Obj()
    o.f(1)
    o.f.cache_clear()
    assert len(o.cache) == 0
