import math
import random

import pytest

from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
    _UnboundTTLCache,
)


ALL_DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]


# ---------------------------------------------------------------- basic reuse

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_basic_memoization(decorator):
    calls = []

    @decorator()
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]
    assert f(2) == 4
    assert f(2) == 4
    assert calls == [1, 2]


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_info(decorator):
    @decorator()
    def f(x):
        return x

    f(1)
    f(1)
    f(2)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_clear(decorator):
    calls = []

    @decorator()
    def f(x):
        calls.append(x)
        return x

    f(1)
    f.cache_clear()
    f(1)
    assert calls == [1, 1]


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_parameters_default(decorator):
    @decorator()
    def f(x):
        return x

    params = f.cache_parameters()
    assert params["maxsize"] == 128
    assert params["typed"] is False


# -------------------------------------------------------------- bare decorator

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_callable_maxsize_acts_as_decorator(decorator):
    calls = []

    @decorator
    def f(x):
        calls.append(x)
        return x + 1

    params = f.cache_parameters()
    assert params["maxsize"] == 128
    assert params["typed"] is False

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


# --------------------------------------------------------------- maxsize None

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_maxsize_none(decorator):
    calls = []

    @decorator(maxsize=None)
    def f(x):
        calls.append(x)
        return x

    params = f.cache_parameters()
    assert params["maxsize"] is None

    for i in range(300):
        assert f(i) == i
    assert calls == list(range(300))
    # all cached, no eviction
    assert f(0) == 0
    assert calls == list(range(300))


# --------------------------------------------------------------- typed option

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_typed_true_distinguishes_types(decorator):
    calls = []

    @decorator(typed=True)
    def f(x):
        calls.append(x)
        return id(x)

    f(1)
    f(1.0)
    # int 1 and float 1.0 must be distinct cache keys when typed
    assert len(calls) == 2
    f(1)
    assert len(calls) == 2


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_untyped_merges_equal_values(decorator):
    calls = []

    @decorator(typed=False)
    def f(x):
        calls.append(x)
        return "v"

    # 1 and 1.0 hash equal => same key in untyped mode
    f(1)
    f(1.0)
    assert len(calls) == 1


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_typed_cache_parameters(decorator):
    @decorator(maxsize=5, typed=True)
    def f(x):
        return x

    params = f.cache_parameters()
    assert params == {"maxsize": 5, "typed": True}


# ------------------------------------------------------------------ maxsize=1

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_maxsize_one_evicts(decorator):
    calls = []

    @decorator(maxsize=1)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)  # 2 should have evicted 1
    assert calls == [1, 2, 1]


# --------------------------------------------------------------- invalid size

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_invalid_maxsize_raises(decorator):
    with pytest.raises(ValueError):
        decorator(maxsize=-1)


# ---------------------------------------------------------- eviction behavior

def test_fifo_eviction_order():
    calls = []

    @fifo_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)   # hit; FIFO does not reorder
    f(3)   # evicts 1 (first in)
    f(1)   # miss -> recompute
    assert calls == [1, 2, 3, 1]


def test_lru_eviction_order():
    calls = []

    @lru_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)   # hit; LRU reorders -> 2 is LRU
    f(3)   # evicts 2
    f(2)   # miss
    f(1)   # hit
    assert calls == [1, 2, 3, 2]


def test_lfu_eviction_order():
    calls = []

    @lfu_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)   # 1 now more frequent
    f(1)
    f(3)   # evicts 2 (least frequent)
    assert 2 not in f.cache
    f(2)   # miss -> recompute
    assert 1 in f.cache
    assert calls == [1, 2, 3, 2]


def test_rr_eviction_uses_choice():
    # deterministic choice: always pick the first key in the cache
    def choice(seq):
        return seq[0]

    calls = []

    @rr_cache(maxsize=2, choice=choice)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)   # evicts first key (1)
    assert 1 not in f.cache
    assert 2 in f.cache
    assert 3 in f.cache


def test_rr_choice_property():
    cache = {"miss": True}  # placeholder to make sure we use the API properly

    def choice(seq):
        return seq[-1]

    @rr_cache(maxsize=2, choice=choice)
    def f(x):
        return x

    # ensure the underlying cache uses the provided choice function
    # by forcing an eviction and verifying last key survives
    f(1)
    f(2)
    f(3)  # evicts last of [1, 2] -> 2
    assert 2 not in f.cache
    assert 1 in f.cache
    assert 3 in f.cache


# ----------------------------------------------------------------- TTL tests

def test_ttl_cache_expiry():
    time_value = [0.0]

    def timer():
        return time_value[0]

    calls = []

    @ttl_cache(maxsize=10, ttl=10, timer=timer)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    time_value[0] = 5.0
    assert f(1) == 1  # still cached
    assert calls == [1]

    time_value[0] = 10.0
    assert f(1) == 1  # expired -> recompute
    assert calls == [1, 1]


def test_ttl_cache_maxsize_none():
    time_value = [0.0]

    def timer():
        return time_value[0]

    calls = []

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def f(x):
        calls.append(x)
        return x

    params = f.cache_parameters()
    assert params["maxsize"] is None

    for i in range(50):
        f(i)
    assert calls == list(range(50))

    # items expire over time even with unbounded size
    time_value[0] = 100.0
    f(0)
    assert calls[-1] == 0
    assert len(calls) == 51


def test_ttl_cache_evicts_lru_within_maxsize():
    time_value = [0.0]

    def timer():
        return time_value[0]

    calls = []

    @ttl_cache(maxsize=2, ttl=100, timer=timer)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)  # reorder
    f(3)  # evicts 2
    assert 2 not in f.cache
    assert 1 in f.cache
    assert 3 in f.cache


def test_ttl_cache_callable_maxsize():
    time_value = [0.0]

    def timer():
        return time_value[0]

    calls = []

    @ttl_cache(ttl=5, timer=timer)
    def f(x):
        calls.append(x)
        return x

    params = f.cache_parameters()
    assert params["maxsize"] == 128
    assert params["typed"] is False

    f(1)
    f(1)
    assert calls == [1]


# ---------------------------------------------------------- unbound TTL cache

def test_unbound_ttl_cache_maxsize_property():
    time_value = [0.0]

    def timer():
        return time_value[0]

    cache = _UnboundTTLCache(ttl=10, timer=timer)
    assert cache.maxsize is None
    assert cache.ttl == 10
    cache[1] = "one"
    assert cache[1] == "one"


# --------------------------------------------------------------- exceptions

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_exceptions_not_cached(decorator):
    calls = []

    @decorator()
    def f(x):
        calls.append(x)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        f(1)
    with pytest.raises(ValueError):
        f(1)
    assert calls == [1, 1]


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_multiple_args_memoized(decorator):
    calls = []

    @decorator()
    def f(a, b, c=0):
        calls.append((a, b, c))
        return a + b + c

    assert f(1, 2, c=3) == 6
    assert f(1, 2, c=3) == 6
    assert f(1, 2) == 3
    assert calls == [(1, 2, 3), (1, 2, 0)]


# ------------------------------------------------ cache_parameters thread safety

@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_maxsize_stored_in_parameters(decorator):
    @decorator(maxsize=7)
    def f(x):
        return x

    assert f.cache_parameters()["maxsize"] == 7
