"""Tests for cachetools.func."""

import threading
import time

import pytest

from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
)
from cachetools import _CacheInfo


DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]


@pytest.fixture(params=DECORATORS, ids=lambda d: d.__name__)
def decorator(request):
    return request.param


def test_cache_parameters(decorator):
    @decorator()
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}


def test_cache_parameters_maxsize(decorator):
    @decorator(maxsize=42)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 42, "typed": False}


def test_cache_parameters_typed(decorator):
    @decorator(typed=True)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 128, "typed": True}


def test_cache_parameters_unbound(decorator):
    @decorator(maxsize=None)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": None, "typed": False}


def test_direct_decoration(decorator):
    @decorator
    def func(x):
        return x

    assert func(1) == 1
    assert func(2) == 2
    assert func.cache_parameters() == {"maxsize": 128, "typed": False}


def test_cache_info_and_hits(decorator):
    calls = []

    @decorator()
    def func(x):
        calls.append(x)
        return x * 2

    assert func(1) == 2
    assert func(1) == 2
    assert func(2) == 4
    assert calls == [1, 2]

    info = func.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 128


def test_cache_clear(decorator):
    calls = []

    @decorator()
    def func(x):
        calls.append(x)
        return x

    func(1)
    func(1)
    func.cache_clear()
    func(1)
    assert calls == [1, 1]


def test_cache_typed():
    @lru_cache(typed=True)
    def func(x):
        return x

    func(1)
    func(1.0)
    assert func.cache_info().misses == 2
    func(1)
    assert func.cache_info().hits == 1


def test_cache_not_typed():
    @lru_cache(typed=False)
    def func(x):
        return x

    func(1)
    func(1.0)
    # 1 and 1.0 hash equal without typed distinction
    assert func.cache_info().misses == 1


def test_exceptions_not_cached():
    calls = []

    @lru_cache()
    def func(x):
        calls.append(x)
        if x < 0:
            raise ValueError(x)
        return x

    with pytest.raises(ValueError):
        func(-1)
    with pytest.raises(ValueError):
        func(-1)
    assert calls == [-1, -1]


def test_maxsize_eviction_lru():
    @lru_cache(maxsize=2)
    def func(x):
        return x

    func(1)
    func(2)
    func(1)  # hit -> 1 most recently used
    func(3)  # evicts 2
    assert func(1) == 1
    assert func(3) == 3
    info = func.cache_info()
    assert info.maxsize == 2
    assert info.currsize == 2


def test_maxsize_eviction_fifo():
    @fifo_cache(maxsize=2)
    def func(x):
        return x

    func(1)
    func(2)
    func(1)  # hit, but does not change FIFO order
    func(3)  # evicts 1 (oldest inserted)
    func(1)  # miss
    assert func.cache_info().misses == 4


def test_maxsize_eviction_lfu():
    @lfu_cache(maxsize=2)
    def func(x):
        return x

    func(1)
    func(2)
    func(1)  # 1 used more frequently
    func(3)  # evicts 2
    func(2)  # miss, would be evicted if 1 was not protected
    assert func.cache_info().misses == 4


def test_rr_cache_custom_choice():
    chosen = []

    def choice(seq):
        chosen.append(list(seq))
        return seq[0]

    @rr_cache(maxsize=2, choice=choice)
    def func(x):
        return x

    func(1)
    func(2)
    func(3)  # triggers eviction using choice
    assert chosen  # choice was called
    assert func.cache_info().maxsize == 2


def test_rr_cache_default():
    @rr_cache(maxsize=2)
    def func(x):
        return x

    func(1)
    func(2)
    func(3)
    assert func.cache_info().currsize == 2


def test_currsize_eviction(decorator):
    @decorator(maxsize=3)
    def func(x):
        return x

    for i in range(10):
        func(i)

    info = func.cache_info()
    assert info.currsize == 3
    assert info.maxsize == 3


def test_unbound_cache(decorator):
    calls = []

    @decorator(maxsize=None)
    def func(x):
        calls.append(x)
        return x

    for i in range(100):
        func(i)
    for i in range(100):
        func(i)

    assert len(calls) == 100
    info = func.cache_info()
    assert info.hits == 100
    assert info.misses == 100


def test_ttl_cache_basic():
    time_val = [0.0]

    def timer():
        return time_val[0]

    calls = []

    @ttl_cache(maxsize=128, ttl=10, timer=timer)
    def func(x):
        calls.append(x)
        return x

    assert func(1) == 1
    assert func(1) == 1
    assert calls == [1]

    time_val[0] = 5.0
    assert func(1) == 1
    assert calls == [1]

    time_val[0] = 10.0  # expired
    assert func(1) == 1
    assert calls == [1, 1]


def test_ttl_cache_maxsize():
    time_val = [0.0]

    def timer():
        return time_val[0]

    @ttl_cache(maxsize=2, ttl=100, timer=timer)
    def func(x):
        return x

    func(1)
    func(2)
    func(3)
    info = func.cache_info()
    assert info.maxsize == 2
    assert info.currsize == 2


def test_ttl_cache_unbound():
    time_val = [0.0]

    def timer():
        return time_val[0]

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def func(x):
        return x

    for i in range(50):
        func(i)

    info = func.cache_info()
    assert info.maxsize is None

    time_val[0] = 100.0
    # after expiry, cache should empty out on access
    func(0)
    assert func.cache_info().currsize == 1


def test_ttl_cache_direct_decoration():
    time_val = [0.0]

    def timer():
        return time_val[0]

    @ttl_cache
    def func(x):
        return x

    assert func(1) == 1
    assert func.cache_parameters() == {"maxsize": 128, "typed": False}


def test_unhashable_raises(decorator):
    @decorator()
    def func(x):
        return x

    with pytest.raises(TypeError):
        func([1, 2, 3])


def test_keyword_arguments(decorator):
    calls = []

    @decorator()
    def func(a, b=2):
        calls.append((a, b))
        return a + b

    assert func(1) == 3
    assert func(1, b=2) == 3
    assert func(1) == 3  # same hashkey
    assert calls == [(1, 2)]


def test_thread_safety_stampede():
    calls = []
    lock = threading.Lock()

    @lru_cache(maxsize=10)
    def func(x):
        with lock:
            calls.append(x)
        time.sleep(0.01)
        return x

    results = []

    def worker():
        results.append(func(1))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [1, 1, 1, 1, 1]
    assert calls == [1]
