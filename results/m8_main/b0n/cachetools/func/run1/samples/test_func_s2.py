import pytest

from cachetools import FIFOCache, LFUCache, LRUCache, RRCache, TTLCache
from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
    _UnboundTTLCache,
)


ALL_DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]


# ---------------------------------------------------------------------------
# basic caching / info
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_basic_caching(decorator):
    calls = []

    @decorator(maxsize=8)
    def foo(x):
        calls.append(x)
        return x * 2

    assert foo(1) == 2
    assert foo(1) == 2
    assert foo(2) == 4
    assert calls == [1, 2]

    info = foo.cache_info()
    assert info.hits == 1
    assert info.misses == 2


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_parameters_default(decorator):
    @decorator()
    def foo(x):
        return x

    params = foo.cache_parameters()
    assert params["maxsize"] == 128
    assert params["typed"] is False


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_parameters_custom(decorator):
    @decorator(maxsize=5, typed=True)
    def foo(x):
        return x

    params = foo.cache_parameters()
    assert params["maxsize"] == 5
    assert params["typed"] is True


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_bare_decorator_usage(decorator):
    """Calling the decorator directly on a function uses default maxsize 128."""

    @decorator
    def foo(x):
        return x + 1

    assert foo(1) == 2
    assert foo(1) == 2
    params = foo.cache_parameters()
    assert params["maxsize"] == 128
    assert params["typed"] is False
    info = foo.cache_info()
    assert info.hits == 1
    assert info.misses == 1


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_maxsize_none_unbounded(decorator):
    calls = []

    @decorator(maxsize=None)
    def foo(x):
        calls.append(x)
        return x

    for _ in range(3):
        assert foo(0) == 0
    info = foo.cache_info()
    assert info.maxsize is None
    assert info.hits == 2
    assert info.misses == 1
    assert calls == [0]
    assert foo.cache_parameters()["maxsize"] is None


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_typed_keys(decorator):
    @decorator(maxsize=8, typed=True)
    def foo(x):
        return x

    assert foo(1) == 1
    assert foo(1.0) == 1.0
    assert foo(True) is True
    info = foo.cache_info()
    # 1, 1.0 and True are distinct keys under typedkey
    assert info.misses == 3
    assert info.hits == 0
    assert foo(1) == 1
    assert foo(1).__class__ is int
    info = foo.cache_info()
    assert info.hits == 1


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_untyped_keys(decorator):
    @decorator(maxsize=8, typed=False)
    def foo(x):
        return x

    assert foo(1) == 1
    assert foo(1.0) == 1
    info = foo.cache_info()
    assert info.misses == 1
    assert info.hits == 1


# ---------------------------------------------------------------------------
# cache class checks
# ---------------------------------------------------------------------------


def test_fifo_uses_fifo_cache():
    @fifo_cache(maxsize=8)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize == 8


def test_lfu_uses_lfu_cache():
    @lfu_cache(maxsize=8)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize == 8


def test_lru_uses_lru_cache():
    @lru_cache(maxsize=8)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize == 8


def test_rr_uses_rr_cache():
    @rr_cache(maxsize=8)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize == 8


def test_ttl_uses_ttl_cache():
    @ttl_cache(maxsize=8, ttl=100)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize == 8


# ---------------------------------------------------------------------------
# eviction behavior
# ---------------------------------------------------------------------------


def test_fifo_eviction_order():
    calls = []

    @fifo_cache(maxsize=2)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(2)
    foo(1)  # still cached, FIFO order unchanged
    foo(3)  # evicts 1 (first in)
    calls.clear()
    foo(1)  # miss -> recomputed
    assert calls == [1]
    calls.clear()
    foo(2)  # hit
    assert calls == []


def test_lru_eviction_order():
    calls = []

    @lru_cache(maxsize=2)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(2)
    foo(1)  # 1 becomes most recent
    foo(3)  # evicts 2
    calls.clear()
    foo(2)  # miss
    assert calls == [2]
    calls.clear()
    foo(1)  # hit
    assert calls == []


def test_lfu_eviction_order():
    calls = []

    @lfu_cache(maxsize=2)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(2)
    foo(1)  # 1 becomes more frequent
    foo(3)  # evicts 2
    calls.clear()
    foo(2)  # miss
    assert calls == [2]
    calls.clear()
    foo(1)  # hit
    assert calls == []


def test_rr_eviction_with_fixed_choice():
    calls = []

    @rr_cache(maxsize=2, choice=lambda keys: keys[0])
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(2)
    foo(3)  # evicts keys[0] == 1
    calls.clear()
    foo(1)  # miss
    assert calls == [1]
    calls.clear()
    foo(2)  # hit
    assert calls == []


# ---------------------------------------------------------------------------
# ttl behaviour with deterministic timer
# ---------------------------------------------------------------------------


class _Timer:
    def __init__(self):
        self._now = 0.0

    def __call__(self):
        return self._now

    def advance(self, dt):
        self._now += dt


def test_ttl_expiration():
    timer = _Timer()
    calls = []

    @ttl_cache(maxsize=8, ttl=10, timer=timer)
    def foo(x):
        calls.append(x)
        return x

    assert foo(1) == 1
    assert foo(1) == 1
    assert calls == [1]

    timer.advance(10)  # expires (not strictly less than)
    assert foo(1) == 1
    assert calls == [1, 1]


def test_ttl_not_expired_before_ttl():
    timer = _Timer()
    calls = []

    @ttl_cache(maxsize=8, ttl=10, timer=timer)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    timer.advance(9.9)
    foo(1)
    assert calls == [1]


def test_ttl_maxsize_none_unbounded():
    timer = _Timer()

    @ttl_cache(maxsize=None, ttl=5, timer=timer)
    def foo(x):
        return x

    info = foo.cache_info()
    assert info.maxsize is None


def test_unbound_ttl_cache_maxsize_property():
    cache = _UnboundTTLCache(10, time_source := lambda: 0.0)
    assert cache.maxsize is None
    assert cache.ttl == 10


def test_ttl_bare_decorator_uses_defaults():
    @ttl_cache
    def foo(x):
        return x

    assert foo(1) == 1
    params = foo.cache_parameters()
    assert params["maxsize"] == 128


def test_rr_bare_decorator_uses_defaults():
    @rr_cache
    def foo(x):
        return x

    assert foo(1) == 1
    assert foo.cache_parameters()["maxsize"] == 128


# ---------------------------------------------------------------------------
# error propagation and not-caching of exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_exceptions_propagate_and_are_not_cached(decorator):
    calls = []

    @decorator(maxsize=8)
    def foo(x):
        calls.append(x)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        foo(1)
    with pytest.raises(ValueError, match="boom"):
        foo(1)
    assert calls == [1, 1]


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_none_return_value_is_cached(decorator):
    calls = []

    @decorator(maxsize=8)
    def foo(x):
        calls.append(x)
        return None

    assert foo(1) is None
    assert foo(1) is None
    assert calls == [1]


# ---------------------------------------------------------------------------
# cache_clear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_clear(decorator):
    calls = []

    @decorator(maxsize=8)
    def foo(x):
        calls.append(x)
        return x

    foo(1)
    foo(1)
    foo.cache_clear()
    info = foo.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    foo(1)
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# cachetools classes instantiated correctly
# ---------------------------------------------------------------------------


def test_ttl_kwargs_used():
    timer = _Timer()

    @ttl_cache(maxsize=4, ttl=50, timer=timer)
    def foo(x):
        return x

    foo(1)
    timer.advance(49)
    foo(1)  # hit -> misses stays 1
    assert foo.cache_info().misses == 1
    timer.advance(1)
    foo(1)  # expired -> miss
    assert foo.cache_info().misses == 2
