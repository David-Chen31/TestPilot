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
)
from cachetools import _CacheInfo


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def counting():
    """Return a factory building functions that count their invocations."""

    def make(func):
        calls = []

        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return func(*args, **kwargs)

        wrapper.calls = calls
        return wrapper

    return make


class FakeTimer:
    """Deterministic timer for the TTL cache."""

    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t

    def __enter__(self):
        return self.t

    def __exit__(self, *exc):
        return False

    def advance(self, delta):
        self.t += delta


# ---------------------------------------------------------------- basic memoization


DECORATORS = [
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
]


@pytest.mark.parametrize("decorator", DECORATORS)
def test_basic_memoization(decorator, counting):
    func = counting(lambda x: x * 2)
    cached = decorator(maxsize=10)(func)

    assert cached(3) == 6
    assert cached(3) == 6
    assert cached(4) == 8

    assert func.calls == [((3,), {}), ((4,), {})]


@pytest.mark.parametrize("decorator", DECORATORS)
def test_cache_info(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=10)(func)

    cached(1)
    cached(1)
    cached(1)
    cached(2)

    info = cached.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 2
    assert info.misses == 2
    assert info.maxsize == 10
    assert info.currsize == 2


@pytest.mark.parametrize("decorator", DECORATORS)
def test_cache_parameters(decorator):
    cached = decorator(maxsize=10, typed=True)(lambda x: x)
    assert cached.cache_parameters() == {"maxsize": 10, "typed": True}

    cached = decorator(maxsize=5)(lambda x: x)
    assert cached.cache_parameters() == {"maxsize": 5, "typed": False}


@pytest.mark.parametrize("decorator", DECORATORS)
def test_cache_clear(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=10)(func)

    cached(1)
    cached(1)
    assert cached.cache_info().hits == 1

    cached.cache_clear()

    info = cached.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0

    cached(1)
    assert func.calls == [((1,), {}), ((1,), {})]


# ---------------------------------------------------------------- typed keys


@pytest.mark.parametrize("decorator", DECORATORS)
def test_typed_keys(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=10, typed=True)(func)

    assert cached(1) == 1
    assert cached(1.0) == 1.0
    assert cached(True) is True

    assert len(func.calls) == 3


@pytest.mark.parametrize("decorator", DECORATORS)
def test_untyped_keys(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=10)(func)

    assert cached(1) == 1
    assert cached(1.0) == 1
    assert cached(True) == 1

    assert len(func.calls) == 1


# ---------------------------------------------------------------- maxsize=None


@pytest.mark.parametrize("decorator", DECORATORS)
def test_maxsize_none_unbounded(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=None)(func)

    for i in range(100):
        assert cached(i) == i

    assert len(func.calls) == 100

    info = cached.cache_info()
    assert info.maxsize is None
    assert info.currsize == 100
    assert cached.cache_parameters() == {"maxsize": None, "typed": False}


# ---------------------------------------------------------------- bare decorator (callable maxsize)


@pytest.mark.parametrize("decorator", DECORATORS)
def test_bare_decorator_uses_default_maxsize(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(func)

    assert cached.cache_parameters() == {"maxsize": 128, "typed": False}
    assert cached(1) == 1
    assert cached(1) == 1
    assert len(func.calls) == 1


# ---------------------------------------------------------------- eviction


def test_fifo_eviction(counting):
    func = counting(lambda x: x)
    cached = fifo_cache(maxsize=2)(func)

    cached(1)
    cached(2)
    cached(1)  # does not change FIFO order
    cached(3)  # evicts 1 (first inserted)

    assert cached(1) == 1  # miss
    assert cached(2) == 2  # cache hit
    assert cached(3) == 3  # cache hit

    assert len(func.calls) == 4


def test_lru_eviction(counting):
    func = counting(lambda x: x)
    cached = lru_cache(maxsize=2)(func)

    cached(1)
    cached(2)
    cached(1)  # 1 becomes most recently used
    cached(3)  # evicts 2

    assert cached(1) == 1  # hit
    assert cached(2) == 2  # miss (was evicted)
    assert cached(3) == 3  # hit

    assert len(func.calls) == 4


def test_lfu_eviction(counting):
    func = counting(lambda x: x)
    cached = lfu_cache(maxsize=2)(func)

    cached(1)
    cached(1)  # 1 used more often
    cached(2)
    cached(3)  # evicts 2 (least used)

    assert cached(1) == 1  # hit
    assert cached(3) == 3  # hit
    assert cached(2) == 2  # miss (evicted)

    calls = len(func.calls)
    cached(1)
    cached(3)
    assert len(func.calls) == calls


def test_rr_eviction_with_fixed_choice(counting):
    func = counting(lambda x: x)
    # always pick the first key in the current key list
    cached = rr_cache(maxsize=2, choice=lambda ks: ks[0])(func)

    cached(1)
    cached(2)
    cached(3)  # evicts 1 (first key)

    assert cached(2) == 2  # hit
    assert cached(3) == 3  # hit
    assert cached(1) == 1  # miss

    assert len(func.calls) == 4


# ---------------------------------------------------------------- ttl specific


def test_ttl_expiration(counting):
    timer = FakeTimer(0)
    func = counting(lambda x: x)
    cached = ttl_cache(maxsize=10, ttl=10, timer=timer)(func)

    cached(1)
    cached(2)
    assert len(func.calls) == 2

    timer.advance(5)
    assert cached(1) == 1
    assert cached(2) == 2
    assert len(func.calls) == 2

    timer.advance(6)  # now 11: both expired
    assert cached(1) == 1
    assert cached(2) == 2
    assert len(func.calls) == 4


def test_ttl_partial_expiration(counting):
    timer = FakeTimer(0)
    func = counting(lambda x: x)
    cached = ttl_cache(maxsize=10, ttl=10, timer=timer)(func)

    cached(1)
    timer.advance(5)
    cached(2)

    timer.advance(6)  # t=11: item 1 expired, item 2 (t=15) still valid

    assert cached(2) == 2  # hit
    assert cached(1) == 1  # miss
    assert len(func.calls) == 3


def test_ttl_unbounded(counting):
    timer = FakeTimer(0)
    func = counting(lambda x: x)
    cached = ttl_cache(maxsize=None, ttl=10, timer=timer)(func)

    for i in range(20):
        cached(i)
    assert len(func.calls) == 20

    info = cached.cache_info()
    assert info.maxsize is None
    assert info.currsize == 20

    timer.advance(11)
    assert cached.cache_info().currsize == 0  # expired on access via info? not necessarily


def test_ttl_maxsize_none_keeps_all_unexpired(counting):
    timer = FakeTimer(0)
    func = counting(lambda x: x)
    cached = ttl_cache(maxsize=None, ttl=100, timer=timer)(func)

    for i in range(50):
        cached(i)

    assert len(func.calls) == 50
    assert cached.cache_info().currsize == 50


def test_ttl_eviction(counting):
    timer = FakeTimer(0)
    func = counting(lambda x: x)
    cached = ttl_cache(maxsize=2, ttl=100, timer=timer)(func)

    cached(1)
    cached(2)
    cached(1)  # 1 most recently used
    cached(3)  # evicts 2 (least recently used)

    assert cached(1) == 1  # hit
    assert cached(3) == 3  # hit
    assert cached(2) == 2  # miss
    assert len(func.calls) == 4


# ---------------------------------------------------------------- exception handling


@pytest.mark.parametrize("decorator", DECORATORS)
def test_exceptions_not_cached(decorator, counting):
    state = {"fail": True}

    def inner(x):
        if state["fail"]:
            raise ValueError("boom")
        return x

    func = counting(inner)
    cached = decorator(maxsize=10)(func)

    with pytest.raises(ValueError):
        cached(1)

    state["fail"] = False
    assert cached(1) == 1
    assert cached(1) == 1

    assert len(func.calls) == 2


# ---------------------------------------------------------------- keyword args


@pytest.mark.parametrize("decorator", DECORATORS)
def test_kwargs(decorator, counting):
    func = counting(lambda a, b=0: a + b)
    cached = decorator(maxsize=10)(func)

    assert cached(1, b=2) == 3
    assert cached(1, b=2) == 3
    assert len(func.calls) == 1

    assert cached(a=1, b=2) == 3
    assert cached(1, b=2) == 3  # hashkey normalizes
    assert len(func.calls) == 1


# ---------------------------------------------------------------- cache_info currsize


@pytest.mark.parametrize("decorator", DECORATORS)
def test_currsize_respects_maxsize(decorator, counting):
    func = counting(lambda x: x)
    cached = decorator(maxsize=3)(func)

    for i in range(10):
        cached(i)

    assert cached.cache_info().currsize == 3


# ---------------------------------------------------------------- module __all__


def test_all_exports():
    import cachetools.func as mod

    assert set(mod.__all__) == {
        "fifo_cache",
        "lfu_cache",
        "lru_cache",
        "rr_cache",
        "ttl_cache",
    }


# ---------------------------------------------------------------- ttl cache parameters


def test_ttl_cache_parameters():
    cached = ttl_cache(maxsize=3, ttl=42, typed=True)(lambda x: x)
    assert cached.cache_parameters() == {"maxsize": 3, "typed": True}

    cached = ttl_cache(maxsize=None)(lambda x: x)
    assert cached.cache_parameters() == {"maxsize": None, "typed": False}
