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
# Helper: deterministic timer
# ---------------------------------------------------------------------------

class Time:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, delta):
        self.value += delta
        return self.value


# ---------------------------------------------------------------------------
# Cache base class
# ---------------------------------------------------------------------------

class TestCache:
    def test_maxsize_and_currsize(self):
        c = Cache(10)
        assert c.maxsize == 10
        assert c.currsize == 0
        c["a"] = 1
        assert c.currsize == 1
        c["b"] = 2
        assert c.currsize == 2
        del c["a"]
        assert c.currsize == 1

    def test_negative_maxsize(self):
        with pytest.raises(ValueError):
            Cache(-1)

    def test_len_iter_contains(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        assert set(iter(c)) == {"a", "b"}
        assert "a" in c
        assert "z" not in c

    def test_missing_raises(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            c["missing"]

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
        assert c.pop("a", 7) == 7

    def test_setdefault(self):
        c = Cache(10)
        assert c.setdefault("a", 1) == 1
        assert c["a"] == 1
        assert c.setdefault("a", 2) == 1
        assert c.setdefault("b") is None

    def test_clear(self):
        c = Cache(10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0

    def test_delete_missing(self):
        c = Cache(10)
        with pytest.raises(KeyError):
            del c["missing"]

    def test_repr(self):
        c = Cache(10)
        r = repr(c)
        assert r.startswith("Cache(")
        assert "maxsize=10" in r
        assert "currsize=0" in r

    def test_value_too_large(self):
        c = Cache(1)
        with pytest.raises(ValueError):
            c["a"] = object()

    def test_negative_value_size(self):
        c = Cache(10, getsizeof=lambda v: -1)
        with pytest.raises(ValueError):
            c["a"] = 1

    def test_custom_getsizeof(self):
        c = Cache(10, getsizeof=len)
        c["a"] = "abc"
        c["b"] = "de"
        assert c.currsize == 5

    def test_eviction(self):
        c = Cache(2)
        c["a"] = 1
        c["b"] = 2
        # base Cache.popitem is inherited from MutableMapping? actually
        # Cache doesn't define popitem, so MutableMapping's uses iter.
        c["c"] = 3
        assert len(c) == 2
        assert c.currsize == 2

    def test_setitem_larger_than_maxsize_with_custom(self):
        c = Cache(5, getsizeof=len)
        c["a"] = "abc"
        assert c.currsize == 3
        with pytest.raises(ValueError):
            c["b"] = "abcdef"

    def test_resize_existing(self):
        c = Cache(10, getsizeof=len)
        c["a"] = "abc"
        c["a"] = "ab"
        assert c.currsize == 2


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_is_cache(self):
        c = FIFOCache(2)
        assert isinstance(c, Cache)

    def test_order(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert set(c) == {"b", "c"}

    def test_update_moves_to_end(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10  # update moves to end
        c["c"] = 3
        assert "b" not in c
        assert "a" in c
        assert "c" in c

    def test_popitem_empty(self):
        c = FIFOCache(2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem(self):
        c = FIFOCache(2)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = FIFOCache(2)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_delitem(self):
        c = FIFOCache(2)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert c.currsize == 0


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_is_cache(self):
        assert isinstance(LFUCache(2), Cache)

    def test_eviction_least_frequent(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]  # bump a
        c["c"] = 3  # evict b
        assert "a" in c
        assert "c" in c
        assert "b" not in c

    def test_popitem_empty(self):
        with pytest.raises(KeyError):
            LFUCache(2).popitem()

    def test_popitem(self):
        c = LFUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        assert c.popitem() == ("b", 2)

    def test_clear(self):
        c = LFUCache(2)
        c["a"] = 1
        _ = c["a"]
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_update(self):
        c = LFUCache(2)
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        assert c.currsize == 1

    def test_delitem(self):
        c = LFUCache(2)
        c["a"] = 1
        del c["a"]
        assert "a" not in c


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_is_cache(self):
        assert isinstance(LRUCache(2), Cache)

    def test_eviction(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]  # bump a
        c["c"] = 3  # evict b
        assert set(c) == {"a", "c"}

    def test_popitem_empty(self):
        with pytest.raises(KeyError):
            LRUCache(2).popitem()

    def test_popitem(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        _ = c["a"]
        assert c.popitem() == ("b", 2)

    def test_clear(self):
        c = LRUCache(2)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_update_moves_to_end(self):
        c = LRUCache(2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10
        c["c"] = 3
        assert set(c) == {"a", "c"}

    def test_delitem(self):
        c = LRUCache(2)
        c["a"] = 1
        del c["a"]
        assert "a" not in c


# ---------------------------------------------------------------------------
# RR
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_is_cache(self):
        assert isinstance(RRCache(2), Cache)

    def test_choice_property(self):
        def choice(seq):
            return seq[0]
        c = RRCache(2, choice=choice)
        assert c.choice is choice

    def test_eviction_with_deterministic_choice(self):
        def choice(seq):
            return seq[0]
        c = RRCache(2, choice=choice)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        # a evicted since choice picks first key
        assert "a" not in c
        assert set(c) == {"b", "c"}

    def test_popitem_empty(self):
        with pytest.raises(KeyError):
            RRCache(2).popitem()

    def test_popitem(self):
        def choice(seq):
            return seq[-1]
        c = RRCache(2, choice=choice)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("b", 2)

    def test_clear(self):
        c = RRCache(2)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert "b" in c

    def test_delitem_swap(self):
        c = RRCache(3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["a"]
        assert set(c) == {"b", "c"}
        assert c.currsize == 2


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

class TestTTLCache:
    def test_is_cache(self):
        assert isinstance(TTLCache(2, 10), Cache)

    def test_ttl_property(self):
        c = TTLCache(2, 10)
        assert c.ttl == 10

    def test_basic(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        assert c["a"] == 1
        assert c.currsize == 1

    def test_expiry(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_expire_returns_pairs(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["b"] = 2
        timer.advance(10)
        expired = c.expire()
        assert ("a", 1) in expired
        assert "a" not in c

    def test_expire_empty(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        assert c.expire() == []

    def test_len_triggers_expire(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert len(c) == 0

    def test_currsize_triggers_expire(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert c.currsize == 0

    def test_popitem_empty(self):
        c = TTLCache(2, 10)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem_skips_expired(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["b"] = 2
        timer.advance(6)  # a expired, b not
        assert c.popitem() == ("b", 2)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c["b"] == 2

    def test_delitem_expired_raises(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_valid(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c

    def test_iteration_skips_expired(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["b"] = 2
        timer.advance(6)
        assert list(c) == ["b"]

    def test_get_triggers_expire(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert c.get("a") is None

    def test_update_extends_expiry(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        timer.advance(8)
        c["a"] = 2
        timer.advance(5)
        assert c["a"] == 2

    def test_pickle_roundtrip(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        # pickling a _Link exercises __reduce__
        link = c._TTLCache__links["a"]
        restored = pickle.loads(pickle.dumps(link))
        assert restored.key == "a"
        assert restored.expires == link.expires

    def test_setstate(self):
        timer = Time(0)
        c = TTLCache(2, 10, timer=timer)
        c["a"] = 1
        state = c.__dict__.copy()
        c2 = TTLCache(2, 10, timer=timer)
        c2.__setstate__(state)
        assert c2["a"] == 1


# ---------------------------------------------------------------------------
# TLRU
# ---------------------------------------------------------------------------

class TestTLRUCache:
    def test_is_cache(self):
        c = TLRUCache(2, lambda k, v, t: t + 10)
        assert isinstance(c, Cache)

    def test_ttu_property(self):
        ttu = lambda k, v, t: t + 10
        c = TLRUCache(2, ttu)
        assert c.ttu is ttu

    def test_basic(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        assert c["a"] == 1

    def test_expiry(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_setitem_already_expired(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t, timer=timer)
        c["a"] = 1
        assert "a" not in c

    def test_setitem_already_expired_existing(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["a"] = 2
        assert c["a"] == 2

    def test_expire_returns_pairs(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(5)
        c["b"] = 2
        timer.advance(10)
        expired = c.expire()
        assert ("a", 1) in expired

    def test_expire_empty(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        assert c.expire() == []

    def test_heap_cleanup(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 100, timer=timer)
        for i in range(6):
            c[f"k{i}"] = i
            c[f"k{i}"] = i
        # trigger cleanup
        c.expire()
        assert len(list(c)) == len(c.__dict__["_TLRUCache__items"])

    def test_popitem_empty(self):
        c = TLRUCache(2, lambda k, v, t: t + 10)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)

    def test_clear(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        c["b"] = 2
        assert c["b"] == 2

    def test_delitem_expired_raises(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_valid(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c

    def test_iteration_skips_expired(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        c["b"] = 2
        assert list(c) == ["b"]

    def test_get_triggers(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert c.get("a") is None

    def test_len_triggers_expire(self):
        timer = Time(0)
        c = TLRUCache(2, lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.advance(10)
        assert len(c) == 0


# ---------------------------------------------------------------------------
# cached decorator
# ---------------------------------------------------------------------------

class TestCached:
    def test_basic(self):
        cache = Cache(10)
        calls = []

        @cached(cache)
        def f(x):
            calls.append(x)
            return x * 2

        assert f(1) == 2
        assert f(1) == 2
        assert calls == [1]

    def test_with_dict(self):
        cache = {}
        calls = []

        @cached(cache)
        def f(x):
            calls.append(x)
            return x + 1

        assert f(1) == 2
        assert f(1) == 2
        assert len(calls) == 1

    def test_info(self):
        cache = Cache(10)
        calls = []

        @cached(cache, info=True)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(1)
        f(2)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 10

    def test_info_mapping(self):
        cache = {}
        calls = []

        @cached(cache, info=True)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize is None

    def test_clear(self):
        cache = Cache(10)

        @cached(cache)
        def f(x):
            return x

        f(1)
        assert "1" in list(f.cache.keys()) or len(f.cache) == 1
        f.cache_clear()
        assert len(f.cache) == 0

    def test_custom_key(self):
        cache = Cache(10)

        @cached(cache, key=lambda x: x)
        def f(x):
            return x

        f("a")
        assert "a" in f.cache

    def test_lock(self):
        import threading
        cache = Cache(10)
        lock = threading.RLock()

        @cached(cache, lock=lock)
        def f(x):
            return x

        assert f(1) == 1


class TestCachedMethod:
    def test_basic(self):
        class A:
            def __init__(self):
                self.cache = Cache(10)
                self.calls = 0

            @cachedmethod(lambda self: self.cache)
            def f(self, x):
                self.calls += 1
                return x * 2

        a = A()
        assert a.f(1) == 2
        assert a.f(1) == 2
        assert a.calls == 1

    def test_info(self):
        class A:
            def __init__(self):
                self.cache = Cache(10)

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        a = A()
        a.f(1)
        a.f(1)
        a.f(2)
        info = a.f.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 10
        assert info.currsize == 2

    def test_info_mapping(self):
        class A:
            def __init__(self):
                self.cache = {}

            @cachedmethod(lambda self: self.cache, info=True)
            def f(self, x):
                return x

        a = A()
        a.f(1)
        a.f(1)
        info = a.f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize is None

    def test_clear(self):
        class A:
            def __init__(self):
                self.cache = Cache(10)

            @cachedmethod(lambda self: self.cache)
            def f(self, x):
                return x

        a = A()
        a.f(1)
        assert len(a.cache) == 1
        a.f.cache_clear()
        assert len(a.cache) == 0
