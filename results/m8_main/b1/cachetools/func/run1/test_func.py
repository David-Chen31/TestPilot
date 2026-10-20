import math
import threading
import time

import pytest

from cachetools import _CacheInfo
from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counting(fn):
    """Wrap a function so each call increments a counter."""
    calls = []

    def inner(*args, **kwargs):
        calls.append((args, kwargs))
        return fn(*args, **kwargs)

    inner.calls = calls
    return inner


class FakeTimer:
    """Deterministic timer implementing the required context manager API."""

    def __init__(self, start=0.0):
        self.time = start

    def __call__(self):
        return self.time

    def __enter__(self):
        return self.time

    def __exit__(self, *exc):
        return False

    def advance(self, delta):
        self.time += delta


# ---------------------------------------------------------------------------
# _cache / cache_parameters shared across decorators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_parameters_default(decorator):
    @decorator()
    def fn(x):
        return x

    assert fn.cache_parameters() == {"maxsize": 128, "typed": False}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_parameters_typed(decorator):
    @decorator(typed=True)
    def fn(x):
        return x

    assert fn.cache_parameters() == {"maxsize": 128, "typed": True}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_parameters_maxsize(decorator):
    @decorator(maxsize=42)
    def fn(x):
        return x

    assert fn.cache_parameters() == {"maxsize": 42, "typed": False}


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_parameters_unbounded(decorator):
    @decorator(maxsize=None)
    def fn(x):
        return x

    assert fn.cache_parameters() == {"maxsize": None, "typed": False}


# ---------------------------------------------------------------------------
# Direct decoration without parentheses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_direct_decoration(decorator):
    @decorator
    def fn(x):
        return x * 2

    assert fn.cache_parameters() == {"maxsize": 128, "typed": False}
    assert fn(3) == 6
    assert fn(3) == 6
    assert fn.cache_info().hits == 1
    assert fn.cache_info().misses == 1


# ---------------------------------------------------------------------------
# Basic caching behaviour for all decorators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_basic_caching(decorator):
    def compute(x):
        return x + 1

    fn = _counting(compute)
    wrapped = decorator(maxsize=10)(fn)

    assert wrapped(1) == 2
    assert wrapped(1) == 2
    assert wrapped(2) == 3
    assert len(fn.calls) == 2

    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 10
    assert info.currsize == 2


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_eviction(decorator):
    fn = _counting(lambda x: x)
    wrapped = decorator(maxsize=2)(fn)

    wrapped(1)
    wrapped(2)
    wrapped(3)  # evicts one
    assert wrapped.cache_info().currsize == 2

    info = wrapped.cache_info()
    assert info.misses == 3


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_typed_key(decorator):
    fn = _counting(lambda x: x)
    wrapped = decorator(typed=True)(fn)

    wrapped(1)
    wrapped(1.0)
    # 1 and 1.0 are different keys under typed key
    assert len(fn.calls) == 2


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_untyped_key(decorator):
    fn = _counting(lambda x: x)
    wrapped = decorator(typed=False)(fn)

    wrapped(1)
    wrapped(1.0)
    # 1 == 1.0 -> same hashkey -> cached
    assert len(fn.calls) == 1
    assert wrapped.cache_info().hits == 1


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_unbounded_maxsize(decorator):
    fn = _counting(lambda x: x)
    wrapped = decorator(maxsize=None)(fn)

    for i in range(1000):
        wrapped(i)
    assert len(fn.calls) == 1000
    assert wrapped.cache_info().currsize == 1000

    # second loop hits all
    for i in range(1000):
        wrapped(i)
    assert len(fn.calls) == 1000
    assert wrapped.cache_info().hits == 1000


# ---------------------------------------------------------------------------
# Algorithm specific behaviour: FIFO
# ---------------------------------------------------------------------------

def test_fifo_order():
    fn = _counting(lambda x: x)
    wrapped = fifo_cache(maxsize=2)(fn)

    wrapped(1)
    wrapped(2)
    # accessing 1 does not change FIFO order
    wrapped(1)
    wrapped(3)  # should evict 1 (first in)
    assert len(fn.calls) == 3

    wrapped(1)  # miss
    assert len(fn.calls) == 4
    # 2 was evicted too? First in was 1 (evicted), then 2, then 3; now inserting 1 -> evicts 2
    wrapped(2)  # miss
    assert len(fn.calls) == 5


# ---------------------------------------------------------------------------
# Algorithm specific behaviour: LRU
# ---------------------------------------------------------------------------

def test_lru_order():
    fn = _counting(lambda x: x)
    wrapped = lru_cache(maxsize=2)(fn)

    wrapped(1)
    wrapped(2)
    wrapped(1)  # touch 1
    wrapped(3)  # should evict 2
    assert len(fn.calls) == 3

    wrapped(1)  # hit
    assert len(fn.calls) == 3
    wrapped(2)  # miss
    assert len(fn.calls) == 4


# ---------------------------------------------------------------------------
# Algorithm specific behaviour: LFU
# ---------------------------------------------------------------------------

def test_lfu_order():
    fn = _counting(lambda x: x)
    wrapped = lfu_cache(maxsize=2)(fn)

    wrapped(1)
    wrapped(2)
    wrapped(1)  # 1 used twice
    wrapped(3)  # should evict 2 (less frequently used)
    assert len(fn.calls) == 3

    wrapped(1)  # hit
    assert len(fn.calls) == 3
    wrapped(2)  # miss
    assert len(fn.calls) == 4


# ---------------------------------------------------------------------------
# Algorithm specific behaviour: RR with deterministic choice
# ---------------------------------------------------------------------------

def test_rr_choice_property():
    def my_choice(seq):
        return seq[0]

    wrapped = rr_cache(maxsize=5, choice=my_choice)(lambda x: x)
    # Access the cache through closure is not possible; use popitem side effect
    # via eviction instead.
    wrapped.cache_info()  # touch to ensure wrapper exists


def test_rr_deterministic_eviction():
    chosen = []

    def my_choice(seq):
        chosen.append(list(seq))
        return seq[0]

    fn = _counting(lambda x: x)
    wrapped = rr_cache(maxsize=2, choice=my_choice)(fn)

    wrapped(1)
    wrapped(2)
    wrapped(3)  # popitem chooses first of keys; keys order after inserts: [1, 2]
    # The cached key is a typed key tuple when using typed=False? No: hashkey
    # returns a _HashedTuple, which is a tuple subclass; comparing to 1 fails.
    # Just assert the first element of the last choice sequence is present, and
    # that the chosen key corresponds to argument 1 (hashkey(1)).
    assert chosen[-1][0] == (1,)
    # 1 was evicted
    wrapped(1)
    assert len(fn.calls) == 4


# ---------------------------------------------------------------------------
# Algorithm specific behaviour: TTL
# ---------------------------------------------------------------------------

def test_ttl_expiry():
    timer = FakeTimer()
    fn = _counting(lambda x: x)
    wrapped = ttl_cache(maxsize=10, ttl=5, timer=timer)(fn)

    wrapped(1)
    assert len(fn.calls) == 1
    timer.advance(4)
    wrapped(1)  # still valid
    assert len(fn.calls) == 1
    timer.advance(2)  # now expired
    wrapped(1)
    assert len(fn.calls) == 2


def test_ttl_unbounded():
    timer = FakeTimer()
    fn = _counting(lambda x: x)
    wrapped = ttl_cache(maxsize=None, ttl=10, timer=timer)(fn)

    assert wrapped.cache_parameters() == {"maxsize": None, "typed": False}
    for i in range(100):
        wrapped(i)
    assert wrapped.cache_info().currsize == 100
    timer.advance(11)
    wrapped(0)
    assert len(fn.calls) == 101


def test_ttl_direct_decoration():
    timer = FakeTimer()

    @ttl_cache
    def fn(x):
        return x

    # Direct decoration uses default timer/time.monotonic, so just check caching works
    assert fn(1) == 1
    assert fn(1) == 1
    assert fn.cache_info().hits == 1
    assert fn.cache_parameters() == {"maxsize": 128, "typed": False}


# ---------------------------------------------------------------------------
# Cache stampede / thread-safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_thread_safety_single_call(decorator):
    barrier = threading.Barrier(10)
    calls = []
    lock = threading.Lock()

    @decorator(maxsize=10)
    def fn(x):
        with lock:
            calls.append(x)
        return x

    results = []

    def worker():
        barrier.wait()
        results.append(fn(5))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [5] * 10
    assert len(calls) == 1
    assert fn.cache_info().misses == 1
    assert fn.cache_info().hits == 9


# ---------------------------------------------------------------------------
# cache_clear / cache_info semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_clear(decorator):
    fn = _counting(lambda x: x)
    wrapped = decorator(maxsize=10)(fn)

    wrapped(1)
    wrapped(2)
    assert wrapped.cache_info().currsize == 2

    wrapped.cache_clear()
    info = wrapped.cache_info()
    assert info.currsize == 0
    assert info.hits == 0
    assert info.misses == 0

    wrapped(1)
    assert len(fn.calls) == 3


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_cache_info_type(decorator):
    @decorator(maxsize=5)
    def fn(x):
        return x

    info = fn.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.maxsize == 5
    assert info.currsize == 0
    assert info.hits == 0
    assert info.misses == 0


@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache])
def test_cache_info_currsize_after_eviction(decorator):
    @decorator(maxsize=1)
    def fn(x):
        return x

    fn(1)
    fn(2)  # evicts 1
    info = fn.cache_info()
    assert info.currsize == 1
    assert info.misses == 2


# ---------------------------------------------------------------------------
# Exceptions should not poison the cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_exception_propagates(decorator):
    @decorator(maxsize=10)
    def fn(x):
        if x == 0:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        fn(0)

    assert fn(1) == 1
    # A failing first call must not increment hits for the key
    assert fn.cache_info().hits == 0


# ---------------------------------------------------------------------------
# ttl_cache edge: expiry on set triggers cleanup
# ---------------------------------------------------------------------------

def test_ttl_expire_during_set():
    timer = FakeTimer()
    fn = _counting(lambda x: x)
    wrapped = ttl_cache(maxsize=10, ttl=5, timer=timer)(fn)

    wrapped(1)
    wrapped(2)
    assert wrapped.cache_info().currsize == 2
    timer.advance(6)
    wrapped(3)  # inserting should expire previous entries
    info = wrapped.cache_info()
    # currsize counts live entries: 1 and 2 expired, only 3 remains
    assert info.currsize == 1


# ---------------------------------------------------------------------------
# Key computation with kwargs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache])
def test_kwargs_caching(decorator):
    fn = _counting(lambda *a, **kw: (a, tuple(sorted(kw.items()))))
    wrapped = decorator(maxsize=10)(fn)

    wrapped(1, x=2)
    wrapped(1, x=2)
    assert len(fn.calls) == 1

    wrapped(1, x=3)
    assert len(fn.calls) == 2
