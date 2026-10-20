import math
import random

import pytest

from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
)
from cachetools import _CacheInfo


CACHE_DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]


# ---------------------------------------------------------------------------
# Basic memoization behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_basic_memoization(decorator):
    calls = []

    @decorator(maxsize=128)
    def foo(x):
        calls.append(x)
        return x * 2

    assert foo(1) == 2
    assert foo(2) == 4
    assert foo(1) == 2
    assert foo(2) == 4
    assert calls == [1, 2]


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_cache_info_hits_misses(decorator):
    @decorator(maxsize=128)
    def foo(x):
        return x

    foo(1)
    foo(1)
    foo(2)
    info = foo.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 128


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_cache_parameters(decorator):
    @decorator(maxsize=64, typed=True)
    def foo(x):
        return x

    assert foo.cache_parameters() == {"maxsize": 64, "typed": True}


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_cache_parameters_untyped(decorator):
    @decorator(maxsize=8)
    def foo(x):
        return x

    assert foo.cache_parameters() == {"maxsize": 8, "typed": False}


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_cache_clear(decorator):
    calls = []

    @decorator(maxsize=128)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1)
    foo.cache_clear()
    foo(1)
    assert calls == [1, 1]
    info = foo.cache_info()
    assert info.hits == 0
    assert info.misses == 1


# ---------------------------------------------------------------------------
# maxsize=None (unbounded) mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_maxsize_none_unbounded(decorator):
    calls = []

    @decorator(maxsize=None)
    def foo(x):
        calls.append(x)
        return x

    for i in range(100):
        foo(i)
    for i in range(100):
        foo(i)
    assert len(calls) == 100
    info = foo.cache_info()
    assert info.maxsize is None
    assert info.hits == 100
    assert info.misses == 100


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_maxsize_none_cache_parameters(decorator):
    @decorator(maxsize=None)
    def foo(x):
        return x

    assert foo.cache_parameters() == {"maxsize": None, "typed": False}


# ---------------------------------------------------------------------------
# Bare-decorator (callable maxsize) usage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_bare_decorator(decorator):
    calls = []

    @decorator
    def foo(x):
        calls.append(x)
        return x

    assert foo(1) == 1
    assert foo(1) == 1
    assert calls == [1]
    assert foo.cache_parameters() == {"maxsize": 128, "typed": False}
    info = foo.cache_info()
    assert info.hits == 1
    assert info.misses == 1


# ---------------------------------------------------------------------------
# Eviction behavior
# ---------------------------------------------------------------------------

def test_fifo_eviction():
    @fifo_cache(maxsize=2)
    def foo(x):
        return x

    foo(1)
    foo(2)
    foo(1)  # hit, does not affect FIFO order
    foo(3)  # evicts 1
    assert foo.cache_info().hits == 1
    foo(2)  # hit
    foo(3)  # hit
    foo(1)  # miss (re-added)
    foo(4)  # evicts 2
    foo(4)
    foo(2)  # miss (evicted)


def test_lru_eviction():
    @lru_cache(maxsize=2)
    def foo(x):
        return x

    foo(1)
    foo(2)
    foo(1)  # hit, marks 1 most-recently used
    foo(3)  # evicts 2
    calls = []
    original = foo

    # Check 2 was evicted: calling it counts as a miss
    before = foo.cache_info().misses
    foo(2)
    after = foo.cache_info().misses
    assert after == before + 1


def test_lfu_eviction():
    @lfu_cache(maxsize=2)
    def foo(x):
        return x

    foo(1)
    foo(2)
    foo(1)  # 1 has count 2, 2 has count 1
    foo(3)  # evicts 2 (least frequently used)
    before = foo.cache_info().misses
    foo(1)  # hit
    assert foo.cache_info().misses == before
    before = foo.cache_info().misses
    foo(2)  # miss (was evicted)
    assert foo.cache_info().misses == before + 1


def test_rr_eviction_deterministic_choice():
    def choice(keys):
        return keys[0]

    @rr_cache(maxsize=2, choice=choice)
    def foo(x):
        return x

    foo(1)
    foo(2)
    foo(3)  # evicts 1 (choice picks first)
    before = foo.cache_info().misses
    foo(2)  # hit
    assert foo.cache_info().misses == before
    before = foo.cache_info().misses
    foo(1)  # miss (evicted)
    assert foo.cache_info().misses == before + 1


# ---------------------------------------------------------------------------
# typed key behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_typed_true_distinguishes_types(decorator):
    calls = []

    @decorator(maxsize=128, typed=True)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1.0)
    assert len(calls) == 2


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_untyped_collapses_equal_keys(decorator):
    calls = []

    @decorator(maxsize=128, typed=False)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1.0)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# ttl_cache specific behavior
# ---------------------------------------------------------------------------

class FakeTimer:
    def __init__(self, time=0.0):
        self._time = time

    def __call__(self):
        return self._time

    def advance(self, delta):
        self._time += delta


def test_ttl_cache_expires_items():
    timer = FakeTimer()

    calls = []

    @ttl_cache(maxsize=128, ttl=10, timer=timer)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1)
    assert calls == [1]

    timer.advance(11)
    foo(1)  # expired, miss
    assert calls == [1, 1]


def test_ttl_cache_not_expired_before_ttl():
    timer = FakeTimer()

    calls = []

    @ttl_cache(maxsize=128, ttl=10, timer=timer)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    timer.advance(5)
    foo(1)
    assert calls == [1]


def test_ttl_cache_unbounded_has_maxsize_none():
    timer = FakeTimer()

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def foo(x):
        return x

    foo(1)
    info = foo.cache_info()
    assert info.maxsize is None


def test_ttl_cache_bare_decorator():
    timer = FakeTimer()

    calls = []

    @ttl_cache(ttl=10, timer=timer)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1)
    assert calls == [1]
    assert foo.cache_parameters()["maxsize"] == 128


# ---------------------------------------------------------------------------
# Return values of cached maxsize=None factories
# ---------------------------------------------------------------------------

def test_maxsize_none_stores_in_dict():
    @lru_cache(maxsize=None)
    def foo(x):
        return x

    for i in range(50):
        foo(i)
    info = foo.cache_info()
    assert info.currsize == 50
    assert info.maxsize is None


# ---------------------------------------------------------------------------
# Exceptions are not cached
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_exceptions_not_cached(decorator):
    calls = []

    @decorator(maxsize=128)
    def foo(x):
        calls.append(x)
        if x == 0:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError):
        foo(0)
    with pytest.raises(ValueError):
        foo(0)
    assert calls == [0, 0]

    assert foo(1) == 1
    assert foo(1) == 1
    assert calls == [0, 0, 1]


# ---------------------------------------------------------------------------
# Wrapper attributes / functools compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_wrapper_has_cache_methods(decorator):
    @decorator(maxsize=4)
    def foo(x):
        return x

    assert hasattr(foo, "cache_info")
    assert hasattr(foo, "cache_clear")
    assert hasattr(foo, "cache_parameters")
    assert foo.cache_parameters() == {"maxsize": 4, "typed": False}


@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_wrapper_keyword_arguments(decorator):
    calls = []

    @decorator(maxsize=128)
    def foo(x, y):
        calls.append((x, y))
        return x + y

    assert foo(1, 2) == 3
    assert foo(1, 2) == 3
    assert foo(x=1, y=2) == 3
    assert foo(x=1, y=2) == 3
    # positional and keyword forms are distinct keys
    assert foo(1, y=2) == 3
    assert foo(1, y=2) == 3
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Updating a cached value with the same key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_hit_does_not_call_wrapped(decorator):
    calls = []

    @decorator(maxsize=2)
    def foo(x):
        calls.append(x)
        return x

    for _ in range(5):
        foo(7)
    assert calls == [7]
    info = foo.cache_info()
    assert info.hits == 4
    assert info.misses == 1


# ---------------------------------------------------------------------------
# rr_cache default choice (random) sanity
# ---------------------------------------------------------------------------

def test_rr_cache_default_choice_attribute():
    @rr_cache(maxsize=1)
    def foo(x):
        return x

    # only one key allowed; second call evicts first
    foo(1)
    foo(2)
    before = foo.cache_info().misses
    foo(1)  # was evicted
    assert foo.cache_info().misses == before + 1


# ---------------------------------------------------------------------------
# maxsize edge case: maxsize=0 never caches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decorator", CACHE_DECORATORS)
def test_maxsize_zero_never_caches(decorator):
    calls = []

    @decorator(maxsize=0)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1)
    assert calls == [1, 1]
    info = foo.cache_info()
    assert info.hits == 0
    assert info.misses == 2
    assert info.maxsize == 0
