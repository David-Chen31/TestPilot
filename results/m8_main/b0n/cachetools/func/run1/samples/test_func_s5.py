import math
import random

import pytest

from cachetools import FIFOCache, LFUCache, LRUCache, RRCache, TTLCache
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


class FakeTimer:
    """A deterministic timer usable as TTLCache timer argument."""

    def __init__(self, time=0.0):
        self._time = time

    def __call__(self):
        return self._time

    def advance(self, delta):
        self._time += delta


def counting_func(return_value=None):
    calls = []

    def func(*args, **kwargs):
        calls.append((args, kwargs))
        if return_value is not None:
            return return_value
        return (args, tuple(sorted(kwargs.items())))

    func.calls = calls
    return func


# ---------------------------------------------------------------------------
# Decorator callable / parameter forms
# ---------------------------------------------------------------------------


ALL_DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_bare_decorator_maxsize_default(decorator):
    func = counting_func(42)

    wrapped = decorator(func)

    params = wrapped.cache_parameters()
    assert params == {"maxsize": 128, "typed": False}
    assert wrapped() == 42
    assert wrapped() == 42
    assert len(func.calls) == 1


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_decorator_explicit_maxsize(decorator):
    func = counting_func(1)

    wrapped = decorator(maxsize=3)(func)

    assert wrapped.cache_parameters() == {"maxsize": 3, "typed": False}

    for i in range(5):
        assert wrapped(i) == 1
    # all identical cached values but distinct keys -> 5 stored into size 3
    info = wrapped.cache_info()
    assert info.hits == 0
    assert info.misses == 5
    assert info.maxsize == 3


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_decorator_maxsize_none_unbounded(decorator):
    func = counting_func()

    wrapped = decorator(maxsize=None)(func)

    params = wrapped.cache_parameters()
    assert params == {"maxsize": None, "typed": False}

    for i in range(5):
        assert wrapped(i) == ((i,), ())

    info = wrapped.cache_info()
    assert info.hits == 0
    assert info.misses == 5
    assert info.maxsize is None
    assert info.currsize == 5


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_typed_flag(decorator):
    func = counting_func()

    wrapped = decorator(maxsize=128, typed=True)(func)

    assert wrapped.cache_parameters() == {"maxsize": 128, "typed": True}

    assert wrapped(1) == ((1,), ())
    assert wrapped(1.0) == ((1.0,), ())
    # same hash/eq: typed key should distinguish 1 and 1.0 -> 2 misses
    info = wrapped.cache_info()
    assert info.misses == 2
    assert info.hits == 0

    # repeat calls -> hits
    assert wrapped(1) == ((1,), ())
    assert wrapped(1.0) == ((1.0,), ())
    info = wrapped.cache_info()
    assert info.hits == 2


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_untyped_treats_equal_keys_same(decorator):
    func = counting_func()

    wrapped = decorator(maxsize=128)(func)

    assert wrapped(1) == ((1,), ())
    # 1.0 untyped key collides with 1
    assert wrapped(1.0) == ((1,), ())
    info = wrapped.cache_info()
    assert info.misses == 1
    assert info.hits == 1


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_kwargs_are_part_of_key(decorator):
    func = counting_func()

    wrapped = decorator(maxsize=128)(func)

    assert wrapped(a=1, b=2) == ((), (("a", 1), ("b", 2)))
    # different kwarg order -> distinct hashkey? hashkey uses sorted items
    assert wrapped(b=2, a=1) == ((), (("a", 1), ("b", 2)))
    info = wrapped.cache_info()
    # hashkey sorts kwargs so these should be the same key
    assert info.misses == 1
    assert info.hits == 1


# ---------------------------------------------------------------------------
# cache functions behaviors
# ---------------------------------------------------------------------------


def test_fifo_cache_eviction_order():
    func = counting_func()

    wrapped = fifo_cache(maxsize=2)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # access 1 doesn't change FIFO order
    assert wrapped(1) == ((1,), ())
    # insert 3 evicts 1 (oldest)
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    # 1 was evicted
    assert wrapped(1) == ((1,), ())
    assert len(func.calls) == n_calls + 1
    # 3 is still cached
    n_calls = len(func.calls)
    assert wrapped(3) == ((3,), ())
    assert len(func.calls) == n_calls


def test_lru_cache_eviction_order():
    func = counting_func()

    wrapped = lru_cache(maxsize=2)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # access 1 -> now 2 is least recently used
    assert wrapped(1) == ((1,), ())
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    # 2 was evicted
    assert wrapped(2) == ((2,), ())
    assert len(func.calls) == n_calls + 1
    # 1 is still cached
    n_calls = len(func.calls)
    assert wrapped(1) == ((1,), ())
    assert len(func.calls) == n_calls


def test_lfu_cache_eviction_order():
    func = counting_func()

    wrapped = lfu_cache(maxsize=2)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # use 1 so its frequency is higher
    assert wrapped(1) == ((1,), ())
    # insert 3, should evict 2 (least frequent)
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    assert wrapped(1) == ((1,), ())
    assert len(func.calls) == n_calls
    n_calls = len(func.calls)
    assert wrapped(2) == ((2,), ())
    assert len(func.calls) == n_calls + 1


def test_rr_cache_uses_deterministic_choice():
    func = counting_func()

    def choice(keys):
        return keys[0]

    wrapped = rr_cache(maxsize=2, choice=choice)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # choice picks first key -> evicts 1
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    assert wrapped(1) == ((1,), ())
    assert len(func.calls) == n_calls + 1
    n_calls = len(func.calls)
    assert wrapped(2) == ((2,), ())
    assert len(func.calls) == n_calls


def test_ttl_cache_expiry():
    timer = FakeTimer()
    func = counting_func()

    wrapped = ttl_cache(maxsize=128, ttl=10, timer=timer)(func)

    assert wrapped(1) == ((1,), ())
    timer.advance(5)
    # still valid
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.hits == 1

    timer.advance(10)
    # expired -> recompute
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.misses == 2
    assert info.hits == 1


def test_ttl_cache_eviction_by_maxsize():
    timer = FakeTimer()
    func = counting_func()

    wrapped = ttl_cache(maxsize=2, ttl=100, timer=timer)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # access 1 -> LRU order updates
    assert wrapped(1) == ((1,), ())
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    assert wrapped(2) == ((2,), ())
    assert len(func.calls) == n_calls + 1


# ---------------------------------------------------------------------------
# Specialized checks per decorator for cache types under the hood
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decorator,expected_type",
    [
        (fifo_cache, FIFOCache),
        (lfu_cache, LFUCache),
        (lru_cache, LRUCache),
        (rr_cache, RRCache),
    ],
)
def test_cache_type_maxsize_none_is_dict(decorator, expected_type):
    func = counting_func()
    wrapped = decorator(maxsize=None)(func)
    info = wrapped.cache_info()
    assert info.maxsize is None
    assert wrapped(1) == ((1,), ())
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_ttl_cache_maxsize_none_unbound():
    timer = FakeTimer()
    func = counting_func()
    wrapped = ttl_cache(maxsize=None, ttl=10, timer=timer)(func)

    info = wrapped.cache_info()
    assert info.maxsize is None

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    timer.advance(5)
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.hits == 1

    timer.advance(10)
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.misses == 3
    assert info.hits == 1


# ---------------------------------------------------------------------------
# Cache info / parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_parameters_callable(decorator):
    func = counting_func()
    wrapped = decorator(maxsize=5, typed=True)(func)
    params = wrapped.cache_parameters()
    assert params == {"maxsize": 5, "typed": True}
    # calling again yields same result
    assert wrapped.cache_parameters() == params


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_clear_resets_info(decorator):
    func = counting_func()
    wrapped = decorator(maxsize=128)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    wrapped.cache_clear()
    info = wrapped.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0

    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.misses == 1
    assert info.hits == 0


# ---------------------------------------------------------------------------
# Exceptions should not be cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_exceptions_not_cached(decorator):
    calls = []

    def boom(x):
        calls.append(x)
        raise ValueError("boom")

    wrapped = decorator(maxsize=128)(boom)

    with pytest.raises(ValueError):
        wrapped(1)
    with pytest.raises(ValueError):
        wrapped(1)
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# Threads / cache stampede: ensure the condition deduplicates concurrent calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_concurrent_calls_are_deduplicated(decorator):
    import threading

    call_started = threading.Event()
    release = threading.Event()
    calls = []

    def slow(x):
        calls.append(x)
        call_started.set()
        release.wait(5)
        return x * 2

    wrapped = decorator(maxsize=128)(slow)

    results = []

    def run():
        results.append(wrapped(21))

    t = threading.Thread(target=run)
    t.start()
    assert call_started.wait(5)
    # while the first call is in-flight, another thread should block
    # on the condition and reuse the result rather than call func again.
    t2 = threading.Thread(target=run)
    t2.start()
    release.set()
    t.join(5)
    t2.join(5)

    assert calls == [21]
    assert results == [42, 42]


# ---------------------------------------------------------------------------
# Specific to rr_cache: custom choice used for RR strategy
# ---------------------------------------------------------------------------


def test_rr_cache_choice_property_effect():
    timer_choice_calls = []

    def choice(keys):
        timer_choice_calls.append(list(keys))
        return keys[-1]

    func = counting_func()
    wrapped = rr_cache(maxsize=2, choice=choice)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    # choice returns last key -> evicts 2
    assert wrapped(3) == ((3,), ())
    assert timer_choice_calls == [[1, 2]]

    n_calls = len(func.calls)
    assert wrapped(2) == ((2,), ())
    assert len(func.calls) == n_calls + 1


# ---------------------------------------------------------------------------
# ttl_cache with default timer (uses monotonic but we only exercise caching)
# ---------------------------------------------------------------------------


def test_ttl_cache_default_timer_caches():
    func = counting_func()
    wrapped = ttl_cache(maxsize=2, ttl=600)(func)

    assert wrapped(1) == ((1,), ())
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 2


# ---------------------------------------------------------------------------
# test various maxsize argument types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_maxsize_positional(decorator):
    func = counting_func(7)
    wrapped = decorator(2)(func)
    assert wrapped.cache_parameters() == {"maxsize": 2, "typed": False}


def test_rr_cache_positional_args():
    func = counting_func()

    def choice(keys):
        return keys[0]

    wrapped = rr_cache(2, choice)(func)
    assert wrapped.cache_parameters() == {"maxsize": 2, "typed": False}

    assert wrapped(1) == ((1,), ())
    assert wrapped(2) == ((2,), ())
    assert wrapped(3) == ((3,), ())

    n_calls = len(func.calls)
    assert wrapped(1) == ((1,), ())
    assert len(func.calls) == n_calls + 1


def test_ttl_cache_positional_args():
    timer = FakeTimer()
    func = counting_func()

    wrapped = ttl_cache(2, 10, timer)(func)
    assert wrapped.cache_parameters() == {"maxsize": 2, "typed": False}

    assert wrapped(1) == ((1,), ())
    timer.advance(11)
    assert wrapped(1) == ((1,), ())
    info = wrapped.cache_info()
    assert info.misses == 2


# ---------------------------------------------------------------------------
# none of the decorators break on tuple/list-hashable keys vs unhashable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_unhashable_argument_raises(decorator):
    func = counting_func()
    wrapped = decorator(maxsize=2)(func)

    with pytest.raises(TypeError):
        wrapped([1, 2, 3])


# ---------------------------------------------------------------------------
# cache_info structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decorator", ALL_DECORATORS)
def test_cache_info_hit_miss_accounting(decorator):
    func = counting_func()
    wrapped = decorator(maxsize=4)(func)

    wrapped(1)
    wrapped(2)
    wrapped(1)
    wrapped(3)
    wrapped(3)
    wrapped(3)

    info = wrapped.cache_info()
    assert info.misses == 3
    assert info.hits == 3
    # current number of distinct keys stored
    assert info.currsize == 3
