"""Unit tests for cachetools public API."""

import pickle
import unittest.mock

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
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_init_invalid_maxsize(self):
        with pytest.raises(ValueError):
            Cache(-1)

    def test_getsizeof_negative(self):
        c = Cache(maxsize=10)
        c.getsizeof = lambda v: -1
        with pytest.raises(ValueError):
            c["k"] = 1

    def test_value_too_large(self):
        c = Cache(maxsize=5)
        with pytest.raises(ValueError):
            c["k"] = "abc"
        c.getsizeof = lambda v: 10
        with pytest.raises(ValueError):
            c["k"] = 1

    def test_basic_set_get_len_contains(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        c["b"] = 2
        assert c["a"] == 1
        assert c["b"] == 2
        assert len(c) == 2
        assert "a" in c
        assert "z" not in c

    def test_missing_raises(self):
        c = Cache(maxsize=10)
        with pytest.raises(KeyError):
            c["x"]

    def test_maxsize_currsize(self):
        c = Cache(maxsize=10)
        assert c.maxsize == 10
        assert c.currsize == 0
        c["a"] = 1
        assert c.currsize == 1

    def test_custom_getsizeof(self):
        c = Cache(maxsize=10, getsizeof=len)
        c["a"] = "abc"
        assert c.currsize == 3
        c["a"] = "x"
        assert c.currsize == 1

    def test_eviction_uses_popitem(self):
        class MyCache(Cache):
            def popitem(self):
                key = next(iter(self))
                return (key, self.pop(key))

        c = MyCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert c["b"] == 2
        assert c["c"] == 3

    def test_eviction_when_updating_existing_too_large(self):
        c = Cache(maxsize=3, getsizeof=len)
        c["a"] = "ab"  # size 2
        c["b"] = "c"   # size 1 -> total 3
        # update "a" to size 3; the adding must evict "b" first.
        c["a"] = "xyz"
        assert c.currsize == 3
        assert "b" not in c
        assert c["a"] == "xyz"

    def test_eviction_updating_existing_evicts_self(self):
        c = Cache(maxsize=2, getsizeof=len)
        c["a"] = "a"  # size 1
        c["b"] = "b"  # size 1
        # trigger an eviction where the key being updated could be popped
        class SelfPopCache(Cache):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.popped = []

            def popitem(self):
                # Simulate eviction of the key being updated
                if "a" in self:
                    v = self.pop("a")
                    self.popped.append("a")
                    return ("a", v)
                for k in list(self):
                    return (k, self.pop(k))
                raise KeyError

        c = SelfPopCache(maxsize=2, getsizeof=len)
        c["a"] = "a"
        c["b"] = "b"
        c["a"] = "ab"  # size increase by 1 -> total 3 -> triggers eviction
        assert c.popped == ["a"]
        assert c["a"] == "ab"
        assert c.currsize == 2

    def test_delitem(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert len(c) == 0
        with pytest.raises(KeyError):
            del c["a"]

    def test_get(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.get("missing") is None
        assert c.get("missing", 42) == 42

    def test_pop(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        assert c.pop("a") == 1
        assert "a" not in c
        with pytest.raises(KeyError):
            c.pop("a")
        assert c.pop("a", 5) == 5

    def test_setdefault(self):
        c = Cache(maxsize=10)
        assert c.setdefault("a", 1) == 1
        assert c["a"] == 1
        assert c.setdefault("a", 2) == 1
        assert c["a"] == 1

    def test_clear(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        assert c.currsize == 0
        assert "a" not in c

    def test_repr(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        r = repr(c)
        assert "Cache(" in r
        assert "maxsize=10" in r
        assert "currsize=1" in r

    def test_getsizeof_static(self):
        assert Cache.getsizeof(any_value) == 1

    def test_iter(self):
        c = Cache(maxsize=10)
        c["a"] = 1
        c["b"] = 2
        assert set(c) == {"a", "b"}


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------

class TestFIFOCache:
    def test_eviction_order(self):
        c = FIFOCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert c["b"] == 2
        assert c["c"] == 3

    def test_update_moves_to_end(self):
        c = FIFOCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10  # move to end
        c["c"] = 3   # should evict "b"
        assert "b" not in c
        assert c["a"] == 10
        assert c["c"] == 3

    def test_popitem(self):
        c = FIFOCache(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "a" not in c
        assert c["b"] == 2

    def test_popitem_empty(self):
        c = FIFOCache(maxsize=3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = FIFOCache(maxsize=3)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        assert c.popitem.__self__ is c  # basic sanity
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = FIFOCache(maxsize=3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------

class TestLFUCache:
    def test_eviction_least_frequent(self):
        c = LFUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]  # touch a
        c["c"] = 3
        assert "b" not in c
        assert c["a"] == 1
        assert c["c"] == 3

    def test_getitem_increments_count(self):
        c = LFUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["b"]
        c["b"]
        c["c"] = 3
        assert "a" not in c
        assert c["b"] == 2
        assert c["c"] == 3

    def test_setitem_existing_touches(self):
        c = LFUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10
        c["c"] = 3
        assert "b" not in c
        assert c["a"] == 10

    def test_popitem(self):
        c = LFUCache(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["b"]
        c["c"]
        c["c"]
        k, v = c.popitem()
        assert k == "a"
        assert v == 1
        assert "a" not in c

    def test_popitem_empty(self):
        c = LFUCache(maxsize=3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LFUCache(maxsize=3)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LFUCache(maxsize=3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_link_reuse(self):
        c = LFUCache(maxsize=5)
        c["a"] = 1
        c["b"] = 2
        c["a"]
        c["b"]
        c["b"]
        c["a"]
        c["c"] = 3
        c["c"]
        # Just verify nothing weird
        assert len(c) == 3


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_eviction_least_recently_used(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]  # recently used
        c["c"] = 3
        assert "b" not in c
        assert c["a"] == 1
        assert c["c"] == 3

    def test_getitem_touches(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"]
        c["c"] = 3
        assert "b" not in c

    def test_setitem_existing_touches(self):
        c = LRUCache(maxsize=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 10
        c["c"] = 3
        assert "b" not in c
        assert c["a"] == 10

    def test_popitem(self):
        c = LRUCache(maxsize=3)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        c["a"]
        assert c.popitem() == ("b", 2)
        assert "b" not in c

    def test_popitem_empty(self):
        c = LRUCache(maxsize=3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem(self):
        c = LRUCache(maxsize=3)
        c["a"] = 1
        del c["a"]
        assert "a" not in c
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = LRUCache(maxsize=3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_missing_does_not_touch(self):
        class MyLRU(LRUCache):
            def __missing__(self, key):
                return None

        c = MyLRU(maxsize=2)
        assert c["missing"] is None
        assert len(c) == 0


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------

class TestRRCache:
    def test_choice_default(self):
        c = RRCache(maxsize=2)
        import random
        assert c.choice is random.choice

    def test_choice_custom(self):
        def first(seq):
            return seq[0]

        c = RRCache(maxsize=2, choice=first)
        assert c.choice is first
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        assert "a" not in c
        assert c["b"] == 2
        assert c["c"] == 3

    def test_setitem_existing(self):
        c = RRCache(maxsize=2, choice=lambda s: s[0])
        c["a"] = 1
        c["a"] = 10
        assert c["a"] == 10
        assert len(c) == 1

    def test_delitem_middle(self):
        c = RRCache(maxsize=5, choice=lambda s: s[0])
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3
        del c["b"]
        assert "b" not in c
        assert c["a"] == 1
        assert c["c"] == 3
        assert len(c) == 2

    def test_popitem(self):
        c = RRCache(maxsize=3, choice=lambda s: s[-1])
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("b", 2)
        assert "b" not in c

    def test_popitem_empty(self):
        c = RRCache(maxsize=3)
        with pytest.raises(KeyError):
            c.popitem()

    def test_clear(self):
        c = RRCache(maxsize=3)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------

def make_timer(start=0.0):
    state = {"time": start}

    def timer():
        return state["time"]

    def set_time(t):
        state["time"] = t

    timer.set_time = set_time
    return timer


class TestTTLCache:
    def test_basic_ttl(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=3, ttl=10, timer=timer)
        assert c.ttl == 10
        c["a"] = 1
        assert c["a"] == 1
        timer.set_time(9)
        assert c["a"] == 1
        timer.set_time(10)
        assert "a" not in c
        with pytest.raises(KeyError):
            c["a"]

    def test_contains_expired(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=3, ttl=5, timer=timer)
        c["a"] = 1
        assert "a" in c
        timer.set_time(5)
        assert "a" not in c

    def test_expire_returns_pairs(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.set_time(5)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_expire_none_time(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=5, timer=timer)
        c["a"] = 1
        timer.set_time(10)
        expired = c.expire()
        assert expired == [("a", 1)]

    def test_expire_with_time_arg(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=100, timer=timer)
        c["a"] = 1
        expired = c.expire(50)
        assert expired == []

    def test_popitem(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=100, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "a" not in c

    def test_popitem_expired_removed(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem_empty(self):
        c = TTLCache(maxsize=5, ttl=10, timer=make_timer(0))
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem_expired_raises(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_valid(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c

    def test_iter(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.set_time(5)
        assert set(c) == {"a", "b"}
        timer.set_time(10)
        assert list(c) == []

    def test_len_expires(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        assert len(c) == 0

    def test_currsize_expires(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        assert c.currsize == 0

    def test_clear(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        with pytest.raises(KeyError):
            c.popitem()

    def test_get_pop_setdefault_expire(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.setdefault("b", 2) == 2
        timer.set_time(20)
        assert c.get("a") is None
        assert c.get("b") is None
        assert len(c) == 0

    def test_repr_expires(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        r = repr(c)
        assert "currsize=0" in r

    def test_update_existing(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        timer.set_time(4)
        c["a"] = 2
        assert c["a"] == 2
        timer.set_time(10)  # original would expire at 10, but updated at 4 expires at 14
        assert c["a"] == 2
        timer.set_time(14)
        assert "a" not in c

    def test_timer_property(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        assert c.timer() == 0

    def test_setstate_rebuild(self):
        timer = make_timer(0)
        c = TTLCache(maxsize=5, ttl=10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.set_time(5)
        c2 = TTLCache(maxsize=5, ttl=10, timer=timer)
        c2.__setstate__(c.__dict__)
        assert "a" in c2
        assert "b" in c2
        timer.set_time(20)
        # Force expire via public method
        assert c2.expire() != []


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------

class TestTLRUCache:
    def test_basic(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=3, ttu=lambda k, v, t: t + 10, timer=timer)
        assert c.ttu is not None
        c["a"] = 1
        assert c["a"] == 1
        timer.set_time(9)
        assert c["a"] == 1
        timer.set_time(10)
        assert "a" not in c

    def test_contains(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=3, ttu=lambda k, v, t: t + 5, timer=timer)
        c["a"] = 1
        assert "a" in c
        timer.set_time(5)
        assert "a" not in c

    def test_expire_returns_pairs(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 5, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.set_time(5)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0

    def test_expire_with_time(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 100, timer=timer)
        c["a"] = 1
        assert c.expire(50) == []

    def test_setitem_already_expired_removes(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t, timer=timer)
        c["a"] = 1
        assert "a" not in c
        assert len(c) == 0

    def test_setitem_already_expired_removes_existing(self):
        timer = make_timer(0)
        ttu = {"mode": "valid"}
        def f(k, v, t):
            if ttu["mode"] == "valid":
                return t + 100
            return t
        c = TLRUCache(maxsize=5, ttu=f, timer=timer)
        c["a"] = 1
        assert "a" in c
        ttu["mode"] = "invalid"
        c["a"] = 2
        assert "a" not in c

    def test_popitem(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 100, timer=timer)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        assert "a" not in c

    def test_popitem_expired(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        with pytest.raises(KeyError):
            c.popitem()

    def test_popitem_empty(self):
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=make_timer(0))
        with pytest.raises(KeyError):
            c.popitem()

    def test_delitem_expired_raises(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        with pytest.raises(KeyError):
            del c["a"]

    def test_delitem_valid(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        del c["a"]
        assert "a" not in c

    def test_iter(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        c["b"] = 2
        timer.set_time(5)
        assert set(c) == {"a", "b"}
        timer.set_time(10)
        assert list(c) == []

    def test_iter_skips_removed(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 100, timer=timer)
        c["a"] = 1
        c["b"] = 2
        # Updating "a" marks old item removed; new item pushed
        c["a"] = 10
        assert set(c) == {"a", "b"}

    def test_len_expires(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        timer.set_time(20)
        assert len(c) == 0

    def test_clear(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        c.clear()
        assert len(c) == 0
        assert c.popitem.__self__ is c
        with pytest.raises(KeyError):
            c.popitem()

    def test_get_pop_setdefault_expire(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 10, timer=timer)
        c["a"] = 1
        assert c.get("a") == 1
        assert c.setdefault("b", 2) == 2
        timer.set_time(20)
        assert len(c) == 0

    def test_heap_cleanup(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=100, ttu=lambda k, v, t: t + 1000, timer=timer)
        for i in range(30):
            c[i] = i
        # Update every key a bunch to mark old items removed
        for _ in range(3):
            for i in range(30):
                c[i] = i * 2
        # After enough updates, expire() triggers heap cleanup (private)
        assert c.expire() == []
        assert len(c) == 30

    def test_update_marks_removed(self):
        timer = make_timer(0)
        c = TLRUCache(maxsize=5, ttu=lambda k, v, t: t + 100, timer=timer)
        c["a"] = 1
        c["a"] = 2
        assert c["a"] == 2
        # Old item should be marked removed
        removed_count = sum(1 for item in c._TLRUCache__order if item.removed)
        assert removed_count >= 1


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

        assert f(1) == 2
        assert f(1) == 2
        assert calls == [1]
        assert f(2) == 4
        assert calls == [1, 2]

    def test_with_cachetools_cache(self):
        cache = LRUCache(maxsize=2)
        calls = []

        @cached(cache)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(2)
        f(3)  # evicts 1
        f(1)  # miss
        assert calls == [1, 2, 3, 1]

    def test_custom_key(self):
        cache = {}

        @cached(cache, key=lambda x, y: x)
        def f(x, y):
            return x + y

        assert f(1, 10) == 11
        assert f(1, 100) == 11  # cached by x only

    def test_info_cachetools(self):
        cache = Cache(maxsize=10)

        @cached(cache, info=True)
        def f(x):
            return x

        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        assert info.currsize == 1

    def test_info_mapping(self):
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
        assert info.currsize == 1

    def test_info_non_mapping(self):
        # Wrapper accepts any object; info reports (0, 0) when not Mapping
        class DummyCache:
            def __init__(self):
                self.d = {}

            def __getitem__(self, k):
                return self.d[k]

            def __setitem__(self, k, v):
                self.d[k] = v

            def __contains__(self, k):
                return k in self.d

        cache = DummyCache()

        @cached(cache, info=True)
        def f(x):
            return x

        f(1)
        f(1)
        info = f.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 0
        assert info.currsize == 0

    def test_clear(self):
        cache = {}

        @cached(cache)
        def f(x):
            return x

        f(1)
        assert len(cache) == 1
        f.cache_clear()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# cachedmethod decorator
# ---------------------------------------------------------------------------

class TestCachedMethod:
    def test_basic(self):
        cache = {}

        class A:
            def __init__(self):
                self.calls = []

            @cachedmethod(cache)
            def method(self, x):
                self.calls.append(x)
                return x * 2

        a = A()
        assert a.method(1) == 2
        assert a.method(1) == 2
        assert a.calls == [1]

    def test_info(self):
        cache = {}

        class A:
            @cachedmethod(cache, info=True)
            def method(self, x):
                return x

        a = A()
        a.method(1)
        a.method(1)
        info = a.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_info_cachetools(self):
        cache = LRUCache(maxsize=10)

        class A:
            @cachedmethod(lambda self: cache, info=True)
            def method(self, x):
                return x

        a = A()
        a.method(1)
        a.method(1)
        info = a.method.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert info.maxsize == 10
        assert info.currsize == 1

    def test_info_invalid_cache_type(self):
        class A:
            @cachedmethod(lambda self: self.weird, info=True)
            def method(self, x):
                return x

            weird = object()

        # Should raise TypeError at call time when info is requested
        a = A()
        with pytest.raises(TypeError):
            a.method(1)

    def test_clear(self):
        cache = {}

        class A:
            @cachedmethod(cache)
            def method(self, x):
                return x

        a = A()
        a.method(1)
        assert len(cache) == 1
        a.method.cache_clear()
        assert len(cache) == 0
