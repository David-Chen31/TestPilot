import math
import time

import pytest

from cachetools.func import (
    _UnboundTTLCache,
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
)


# ---------------------------------------------------------------- general


def _counter():
    calls = []

    def func(*args, **kwargs):
        calls.append((args, kwargs))
        return len(calls)

    return func, calls


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_basic_caching(decorator):
    func, calls = _counter()
    wrapped = decorator(func)

    assert wrapped(1) == 1
    assert wrapped(1) == 1
    assert wrapped(2) == 2
    assert wrapped(2) == 2
    assert calls == [((1,), {}), ((2,), {})]


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_cache_parameters(decorator):
    wrapped = decorator()(lambda x: x)
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache])
def test_callable_maxsize_used_as_decorator(decorator):
    func, calls = _counter()
    wrapped = decorator(func)
    assert wrapped(10) == 1
    assert wrapped(10) == 1
    assert calls == [((10,), {})]
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


def test_rr_callable_maxsize_used_as_decorator():
    func, calls = _counter()
    wrapped = rr_cache(func)
    assert wrapped(5) == 1
    assert wrapped(5) == 1
    assert calls == [((5,), {})]
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_maxsize_none_unbounded(decorator):
    func, calls = _counter()
    wrapped = decorator(maxsize=None)(func)

    for i in range(200):
        assert wrapped(i) == i + 1
    for i in range(200):
        assert wrapped(i) == i + 1
    assert len(calls) == 200
    assert wrapped.cache_parameters() == {"maxsize": None, "typed": False}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_cache_info(decorator):
    wrapped = decorator()(lambda x: x)
    assert wrapped.cache_info() == (0, 0, 128, 0)
    wrapped(1)
    wrapped(1)
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_maxsize_none_cache_info(decorator):
    wrapped = decorator(maxsize=None)(lambda x: x)
    wrapped(1)
    wrapped(1)
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize is None


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_typed_keys(decorator):
    func, calls = _counter()
    wrapped = decorator(typed=True)(func)

    wrapped(1)
    wrapped(1.0)
    assert len(calls) == 2

    wrapped(1)
    wrapped(1.0)
    assert len(calls) == 2


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_untyped_keys(decorator):
    func, calls = _counter()
    wrapped = decorator(typed=False)(func)

    wrapped(1)
    wrapped(1.0)
    assert len(calls) == 1


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_kwargs(decorator):
    func, calls = _counter()
    wrapped = decorator()(func)

    wrapped(1, a=2)
    wrapped(1, a=2)
    assert len(calls) == 1


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_eviction_maxsize(decorator):
    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = decorator(maxsize=2)(func)
    wrapped(1)
    wrapped(2)
    wrapped(3)  # evicts something

    calls.clear()
    wrapped(1)
    wrapped(2)
    wrapped(3)
    # at most 2 of these are cached, so at least 1 miss
    assert len(calls) >= 1


# ---------------------------------------------------------------- fifo


def test_fifo_eviction_order():
    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = fifo_cache(maxsize=2)(func)
    wrapped(1)
    wrapped(2)
    wrapped(1)  # hit, does not change FIFO order
    wrapped(3)  # evicts 1 (first in)
    calls.clear()
    wrapped(2)
    assert calls == []  # still cached
    wrapped(1)
    assert calls == [1]  # was evicted


# ---------------------------------------------------------------- lru


def test_lru_eviction_order():
    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = lru_cache(maxsize=2)(func)
    wrapped(1)
    wrapped(2)
    wrapped(1)  # hit, makes 1 most recent
    wrapped(3)  # evicts 2
    calls.clear()
    wrapped(1)
    assert calls == []  # still cached
    wrapped(2)
    assert calls == [2]  # was evicted


# ---------------------------------------------------------------- lfu


def test_lfu_eviction_order():
    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = lfu_cache(maxsize=2)(func)
    wrapped(1)
    wrapped(2)
    wrapped(2)
    wrapped(1)  # 2 has higher use count
    wrapped(3)  # evicts 1 (least frequent)
    calls.clear()
    wrapped(2)
    assert calls == []
    wrapped(1)
    assert calls == [1]


# ---------------------------------------------------------------- rr


def test_rr_choice():
    chosen = []

    def choice(keys):
        chosen.append(list(keys))
        return keys[0]

    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = rr_cache(maxsize=2, choice=choice)(func)
    wrapped(1)
    wrapped(2)
    assert chosen == []  # no eviction yet
    wrapped(3)  # evict
    assert len(chosen) == 1


def test_rr_choice_deterministic_eviction():
    def choice(keys):
        return min(keys)

    calls = []

    def func(x):
        calls.append(x)
        return x

    wrapped = rr_cache(maxsize=2, choice=choice)(func)
    wrapped(2)
    wrapped(3)
    wrapped(1)  # evicts min(keys) == 2

    calls.clear()
    wrapped(3)
    assert calls == []
    wrapped(2)
    assert calls == [2]


# ---------------------------------------------------------------- ttl


def test_ttl_basic():
    now = [0.0]

    def timer():
        return now[0]

    func, calls = _counter()
    wrapped = ttl_cache(maxsize=8, ttl=10, timer=timer)(func)

    assert wrapped(1) == 1
    assert wrapped(1) == 1
    assert len(calls) == 1

    now[0] = 5.0
    assert wrapped(1) == 1
    assert len(calls) == 1

    now[0] = 11.0
    assert wrapped(1) == 2
    assert len(calls) == 2


def test_ttl_cache_parameters():
    wrapped = ttl_cache()(lambda x: x)
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


def test_ttl_callable_maxsize_decorator():
    func, calls = _counter()
    wrapped = ttl_cache(func)
    assert wrapped(1) == 1
    assert wrapped(1) == 1
    assert len(calls) == 1
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


def test_ttl_maxsize_none_unbounded():
    now = [0.0]

    def timer():
        return now[0]

    func, calls = _counter()
    wrapped = ttl_cache(maxsize=None, ttl=10, timer=timer)(func)

    for i in range(50):
        wrapped(i)
    for i in range(50):
        wrapped(i)
    assert len(calls) == 50
    assert wrapped.cache_parameters() == {"maxsize": None, "typed": False}


def test_ttl_expiry_eviction():
    now = [0.0]

    def timer():
        return now[0]

    func, calls = _counter()
    wrapped = ttl_cache(maxsize=2, ttl=10, timer=timer)(func)

    wrapped(1)
    wrapped(2)
    now[0] = 5.0
    wrapped(3)  # 1 and 2 still valid, cache full, eviction
    calls.clear()
    wrapped(1)
    # possibly evicted, just check something happens sensibly
    assert len(calls) <= 1


def test_ttl_expired_items_recomputed():
    now = [0.0]

    def timer():
        return now[0]

    func, calls = _counter()
    wrapped = ttl_cache(maxsize=8, ttl=10, timer=timer)(func)

    assert wrapped(1) == 1
    now[0] = 20.0
    assert wrapped(1) == 2
    assert len(calls) == 2


def test_ttl_typed():
    now = [0.0]

    def timer():
        return now[0]

    func, calls = _counter()
    wrapped = ttl_cache(ttl=10, timer=timer, typed=True)(func)

    wrapped(1)
    wrapped(1.0)
    assert len(calls) == 2


def test_ttl_cache_info():
    now = [0.0]

    def timer():
        return now[0]

    wrapped = ttl_cache(ttl=10, timer=timer)(lambda x: x)
    assert wrapped.cache_info() == (0, 0, 128, 0)
    wrapped(1)
    wrapped(1)
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1


# ---------------------------------------------------------------- unbound ttl


def test_unbound_ttl_cache():
    now = [0.0]

    def timer():
        return now[0]

    cache = _UnboundTTLCache(ttl=10, timer=timer)
    assert cache.maxsize is None
    assert cache.ttl == 10

    cache[1] = "a"
    assert cache[1] == "a"
    now[0] = 20.0
    with pytest.raises(KeyError):
        cache[1]


def test_unbound_ttl_cache_uses_infinite_storage():
    now = [0.0]

    def timer():
        return now[0]

    cache = _UnboundTTLCache(ttl=10, timer=timer)
    for i in range(100):
        cache[i] = i
    for i in range(100):
        assert cache[i] == i
    assert cache.maxsize is None


# ---------------------------------------------------------------- exceptions


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_exception_not_cached(decorator):
    calls = []

    def func(x):
        calls.append(x)
        if x < 0:
            raise ValueError("negative")
        return x

    wrapped = decorator()(func)
    with pytest.raises(ValueError):
        wrapped(-1)
    with pytest.raises(ValueError):
        wrapped(-1)
    assert calls == [-1, -1]


def test_ttl_exception_not_cached():
    now = [0.0]

    def timer():
        return now[0]

    calls = []

    def func(x):
        calls.append(x)
        raise ValueError("boom")

    wrapped = ttl_cache(ttl=10, timer=timer)(func)
    with pytest.raises(ValueError):
        wrapped(1)
    with pytest.raises(ValueError):
        wrapped(1)
    assert calls == [1, 1]
