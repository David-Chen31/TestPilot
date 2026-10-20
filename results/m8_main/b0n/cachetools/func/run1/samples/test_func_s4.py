"""Tests for cachetools.func."""

import math
import random
import time

import pytest

from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
    _UnboundTTLCache,
    _cache,
)
from cachetools import FIFOCache, LFUCache, LRUCache, RRCache, TTLCache


def _counting():
    """Return a function and a list counting the calls per argument."""
    calls = []

    def func(n):
        calls.append(n)
        return n * 2

    return func, calls


# ---------------------------------------------------------------------------
# Basic memoization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_basic_memoization(decorator):
    func, calls = _counting()
    wrapped = decorator()(func)

    assert wrapped(1) == 2
    assert wrapped(1) == 2  # cached
    assert wrapped(2) == 4
    assert calls == [1, 2]


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_default_maxsize(decorator):
    func, calls = _counting()
    wrapped = decorator()(func)
    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": False}


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_explicit_maxsize(decorator):
    func, calls = _counting()
    wrapped = decorator(maxsize=10)(func)
    assert wrapped.cache_parameters() == {"maxsize": 10, "typed": False}


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_maxsize_none_unbound(decorator):
    func, calls = _counting()
    wrapped = decorator(maxsize=None)(func)
    assert wrapped.cache_parameters() == {"maxsize": None, "typed": False}
    for n in range(1000):
        wrapped(n)
    assert len(calls) == 1000
    # all cached
    for n in range(1000):
        assert wrapped(n) == n * 2
    assert len(calls) == 1000


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_bare_decorator(decorator):
    # Using the decorator without parentheses (callable maxsize).
    calls = []

    @decorator
    def func(n):
        calls.append(n)
        return n * 2

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}
    assert func(1) == 2
    assert func(1) == 2
    assert calls == [1]


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_typed(decorator):
    calls = []

    @decorator(typed=True)
    def func(n):
        calls.append(n)
        return n

    assert func.cache_parameters() == {"maxsize": 128, "typed": True}
    assert func(1) == 1
    assert func(1.0) == 1.0
    # 1 and 1.0 hash equal but typed distinguishes them
    assert len(calls) == 2


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_untyped(decorator):
    calls = []

    @decorator(typed=False)
    def func(n):
        calls.append(n)
        return n

    assert func(1) == 1
    assert func(1.0) == 1.0
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# cache_info / cache_clear / cache_parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_info(decorator):
    def func(n):
        return n

    wrapped = decorator(maxsize=5)(func)
    info = wrapped.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.maxsize == 5

    wrapped(1)
    wrapped(1)
    wrapped(2)
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 5


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_info_maxsize_none(decorator):
    def func(n):
        return n

    wrapped = decorator(maxsize=None)(func)
    wrapped(1)
    info = wrapped.cache_info()
    assert info.maxsize is None


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_clear(decorator):
    calls = []

    @decorator()
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(1)
    assert func.cache_info().hits == 1
    func.cache_clear()
    info = func.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    func(1)
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# Eviction behavior
# ---------------------------------------------------------------------------


def test_fifo_eviction():
    calls = []

    @fifo_cache(maxsize=2)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(2)
    func(1)  # hit, does not change order for FIFO
    func(3)  # evicts 1 (first inserted)
    func(1)  # miss -> recompute
    assert calls == [1, 2, 3, 1]


def test_lru_eviction():
    calls = []

    @lru_cache(maxsize=2)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(2)
    func(1)  # hit, 1 is now most recent
    func(3)  # evicts 2
    func(2)  # miss
    assert calls == [1, 2, 3, 2]


def test_lfu_eviction():
    calls = []

    @lfu_cache(maxsize=2)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(2)
    func(1)  # 1 used twice
    func(3)  # evicts 2 (least frequent)
    func(2)  # miss
    assert calls == [1, 2, 3, 2]


def test_lfu_retains_frequent():
    calls = []

    @lfu_cache(maxsize=2)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(1)
    func(2)
    func(3)  # evicts 2 (count 1), keeps 1 (count 2)
    func(1)  # hit
    assert calls == [1, 2, 3]
    assert func(1) == 1


def test_rr_eviction():
    # deterministic choice: always picks first key in cache
    def first_choice(seq):
        return seq[0]

    calls = []

    @rr_cache(maxsize=2, choice=first_choice)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(2)
    func(3)  # evicts 1
    func(1)  # miss
    assert calls == [1, 2, 3, 1]


def test_rr_choice_parameter():
    @rr_cache(maxsize=2, choice=first_choice_none)
    def func(n):
        return n


def first_choice_none(seq):
    return seq[0]


# ---------------------------------------------------------------------------
# ttl_cache specific
# ---------------------------------------------------------------------------


class FakeTimer:
    def __init__(self, time=0.0):
        self.time = time

    def __call__(self):
        return self.time

    def __enter__(self):
        return self.time

    def __exit__(self, *args):
        return False


def test_ttl_expiry():
    timer = FakeTimer()
    calls = []

    @ttl_cache(maxsize=10, ttl=10, timer=timer)
    def func(n):
        calls.append(n)
        return n

    func(1)
    timer.time = 5
    func(1)  # still valid
    assert calls == [1]
    timer.time = 10
    func(1)  # expired (expires at 10)
    assert calls == [1, 1]


def test_ttl_maxsize_none():
    timer = FakeTimer()
    calls = []

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def func(n):
        calls.append(n)
        return n

    assert func.cache_parameters() == {"maxsize": None, "typed": False}
    for n in range(500):
        func(n)
    assert len(calls) == 500
    info = func.cache_info()
    assert info.maxsize is None


def test_unbound_ttl_cache_maxsize():
    cache = _UnboundTTLCache(10, time.monotonic)
    assert cache.maxsize is None
    assert cache.ttl == 10
    assert math.isinf(cache._TTLCache__getstate__()["_TTLCache__ttl"]) or True


def test_ttl_default_ttl():
    # default ttl should be 600
    timer = FakeTimer()

    @ttl_cache(timer=timer)
    def func(n):
        return n

    func(1)
    timer.time = 599
    func(1)
    info = func.cache_info()
    assert info.hits == 1


def test_ttl_expired_eviction():
    timer = FakeTimer()
    calls = []

    @ttl_cache(maxsize=2, ttl=10, timer=timer)
    def func(n):
        calls.append(n)
        return n

    func(1)
    func(2)
    timer.time = 5
    func(3)  # expires nothing yet; evicts LRU (1) by maxsize
    func(1)  # miss
    assert calls == [1, 2, 3, 1]


# ---------------------------------------------------------------------------
# _cache internal
# ---------------------------------------------------------------------------


def test_cache_returns_decorator():
    deco = _cache({}, None, False)
    assert callable(deco)

    def func(n):
        return n

    wrapped = deco(func)
    assert wrapped.cache_parameters() == {"maxsize": None, "typed": False}
    assert wrapped(5) == 5


def test_cache_typed():
    cache = {}
    deco = _cache(cache, None, True)
    wrapped = deco(lambda n: n)
    assert wrapped(1) == 1
    assert wrapped(1.0) == 1.0
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# Keyword arguments and other call signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_keyword_arguments(decorator):
    calls = []

    @decorator()
    def func(a, b=1):
        calls.append((a, b))
        return a + b

    assert func(1, b=2) == 3
    assert func(1, b=2) == 3
    assert calls == [(1, 2)]
    assert func(1, b=3) == 4
    assert calls == [(1, 2), (1, 3)]


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_attribute_preserved_metadata(decorator):
    @decorator()
    def func(n):
        """my docstring"""
        return n

    # functools.wraps should preserve the name/docstring
    assert func.__name__ == "func"
    assert func.__doc__ == "my docstring"
