"""Tests for cachetools.func."""

import gc
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


class CountedCall:
    """Helper that counts calls and returns args."""

    def __init__(self):
        self.count = 0

    def __call__(self, *args):
        self.count += 1
        return args


def _run_basic_checks(decorator_fn):
    """Common behavior checks for a cache decorator factory."""
    calls = CountedCall()

    @decorator_fn(maxsize=2)
    def func(n):
        return calls(n)

    info = func.cache_info()
    assert info == _CacheInfo(0, 0, 2, 0)

    assert func(1) == (1,)
    assert calls.count == 1
    assert func(1) == (1,)
    assert calls.count == 1

    assert func(2) == (2,)
    assert calls.count == 2

    info = func.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 2
    assert info.currsize == 2

    assert func.cache_parameters() == {"maxsize": 2, "typed": False}

    func.cache_clear()
    info = func.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0


def test_fifo_cache_basic():
    _run_basic_checks(fifo_cache)


def test_lfu_cache_basic():
    _run_basic_checks(lfu_cache)


def test_lru_cache_basic():
    _run_basic_checks(lru_cache)


def test_ttl_cache_basic():
    _run_basic_checks(ttl_cache)


def test_rr_cache_basic():
    _run_basic_checks(rr_cache)


def test_maxsize_defaults():
    @fifo_cache()
    def fifo(n):
        return n

    @lfu_cache()
    def lfu(n):
        return n

    @lru_cache()
    def lru(n):
        return n

    @rr_cache()
    def rr(n):
        return n

    @ttl_cache()
    def ttl(n):
        return n

    for func in (fifo, lfu, lru, rr, ttl):
        assert func.cache_parameters() == {"maxsize": 128, "typed": False}
        assert func.cache_info().maxsize == 128


@pytest.mark.parametrize(
    "decorator_fn", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]
)
def test_maxsize_none_unbounded(decorator_fn):
    calls = CountedCall()

    @decorator_fn(maxsize=None)
    def func(n):
        return calls(n)

    assert func.cache_parameters() == {"maxsize": None, "typed": False}
    assert func.cache_info().maxsize is None

    for i in range(30):
        assert func(i) == (i,)
    assert calls.count == 30

    for i in range(30):
        assert func(i) == (i,)
    assert calls.count == 30

    info = func.cache_info()
    assert info.hits == 30
    assert info.misses == 30
    assert info.maxsize is None
    assert info.currsize == 30


@pytest.mark.parametrize(
    "decorator_fn", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]
)
def test_bare_decorator_usage(decorator_fn):
    """Using @decorator_fn directly on a function picks maxsize=128."""

    @decorator_fn
    def func(n):
        return n * 2

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}
    assert func(3) == 6
    assert func(3) == 6
    info = func.cache_info()
    assert info.misses == 1
    assert info.hits == 1


@pytest.mark.parametrize(
    "decorator_fn", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]
)
def test_typed_parameter(decorator_fn):
    calls = CountedCall()

    @decorator_fn(maxsize=8, typed=True)
    def func(n):
        return calls(n)

    assert func.cache_parameters() == {"maxsize": 8, "typed": True}

    # 1 and 1.0 must be treated as different keys with typed=True
    assert func(1) == (1,)
    assert func(1.0) == (1.0,)
    assert calls.count == 2

    assert func(1) == (1,)
    assert func(1.0) == (1.0,)
    assert calls.count == 2


@pytest.mark.parametrize(
    "decorator_fn", [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache]
)
def test_untyped_parameter(decorator_fn):
    calls = CountedCall()

    @decorator_fn(maxsize=8, typed=False)
    def func(n):
        return calls(n)

    assert func(1) == (1,)
    assert func(1.0) == (1,)
    assert calls.count == 1


def test_fifo_eviction_order():
    calls = CountedCall()

    @fifo_cache(maxsize=2)
    def func(n):
        return calls(n)

    func(1)
    func(2)
    func(1)  # hit, does not change FIFO order
    func(3)  # evicts 1
    assert calls.count == 3

    func(1)  # miss, evicts 2
    assert calls.count == 4
    func(2)  # miss, evicts 3
    assert calls.count == 5


def test_lru_eviction_order():
    calls = CountedCall()

    @lru_cache(maxsize=2)
    def func(n):
        return calls(n)

    func(1)
    func(2)
    func(1)  # hit, makes 1 most recently used
    func(3)  # evicts 2, not 1
    assert calls.count == 3

    func(1)  # hit
    assert calls.count == 3
    func(2)  # miss, evicts 3
    assert calls.count == 4


def test_lfu_eviction_order():
    calls = CountedCall()

    @lfu_cache(maxsize=2)
    def func(n):
        return calls(n)

    func(1)
    func(2)
    func(1)  # 1 used twice, 2 used once
    func(3)  # evicts 2 (least frequently used)
    assert calls.count == 3

    func(1)  # still cached
    assert calls.count == 3
    func(2)  # miss, evicts based on frequency
    assert calls.count == 4


def test_rr_cache_with_fixed_choice():
    calls = CountedCall()

    # deterministic choice: always pick the first key in the list
    @rr_cache(maxsize=2, choice=lambda keys: keys[0])
    def func(n):
        return calls(n)

    assert func.cache_parameters() == {"maxsize": 2, "typed": False}

    func(1)
    func(2)
    func(3)  # evicts 1 (first key)
    assert calls.count == 3

    func(2)  # still cached
    assert calls.count == 3

    func(1)  # miss
    assert calls.count == 4


def test_ttl_cache_expiration_no_time_dependency():
    """Use a controllable timer to check TTL behavior deterministically."""
    current = [0.0]

    def timer():
        return current[0]

    calls = CountedCall()

    @ttl_cache(maxsize=8, ttl=10, timer=timer)
    def func(n):
        return calls(n)

    assert func(1) == (1,)
    assert calls.count == 1
    assert func(1) == (1,)
    assert calls.count == 1

    current[0] = 5.0
    assert func(1) == (1,)
    assert calls.count == 1

    current[0] = 10.0  # expired (timer() < expires is False)
    assert func(1) == (1,)
    assert calls.count == 2


def test_ttl_cache_maxsize_none_unbounded():
    current = [0.0]

    def timer():
        return current[0]

    calls = CountedCall()

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def func(n):
        return calls(n)

    assert func.cache_parameters() == {"maxsize": None, "typed": False}
    assert func.cache_info().maxsize is None

    for i in range(20):
        assert func(i) == (i,)
    assert calls.count == 20

    for i in range(20):
        assert func(i) == (i,)
    assert calls.count == 20


def test_ttl_cache_ttl_parameter():
    current = [0.0]
    calls = CountedCall()

    @ttl_cache(maxsize=4, ttl=100, timer=lambda: current[0])
    def func(n):
        return calls(n)

    func(1)
    current[0] = 50.0
    func(1)  # hit
    assert calls.count == 1
    current[0] = 100.0
    func(1)  # expired
    assert calls.count == 2


def test_cache_parameters_returns_new_dict():
    @lru_cache(maxsize=5, typed=True)
    def func(n):
        return n

    params = func.cache_parameters()
    assert params == {"maxsize": 5, "typed": True}
    params["maxsize"] = 99
    assert func.cache_parameters() == {"maxsize": 5, "typed": True}


def test_lru_cache_clear():
    calls = CountedCall()

    @lru_cache(maxsize=4)
    def func(n):
        return calls(n)

    func(1)
    func(2)
    assert func.cache_info().currsize == 2
    func.cache_clear()
    assert func.cache_info().currsize == 0
    func(1)
    assert calls.count == 3
    assert func.cache_info().misses == 1


def test_lru_cache_info_tracking():
    @lru_cache(maxsize=3)
    def func(n):
        return n

    for _ in range(3):
        func(1)  # 1 miss, 2 hits
    info = func.cache_info()
    assert info.hits == 2
    assert info.misses == 1
    assert info.currsize == 1
    assert info.maxsize == 3


def test_lfu_cache_info_currsize():
    @lfu_cache(maxsize=10)
    def func(n):
        return n

    for i in range(5):
        func(i)
    info = func.cache_info()
    assert info.misses == 5
    assert info.currsize == 5
    assert info.maxsize == 10


def test_fifo_cache_info_currsize_after_eviction():
    @fifo_cache(maxsize=2)
    def func(n):
        return n

    for i in range(10):
        func(i)
    info = func.cache_info()
    assert info.misses == 10
    assert info.currsize == 2
    assert info.maxsize == 2


def test_ttl_cache_info_currsize():
    current = [0.0]

    @ttl_cache(maxsize=10, ttl=5, timer=lambda: current[0])
    def func(n):
        return n

    for i in range(3):
        func(i)
    info = func.cache_info()
    assert info.currsize == 3
    assert info.maxsize == 10

    current[0] = 100.0
    # trigger expiry
    func(100)
    info = func.cache_info()
    assert info.currsize == 1


def test_thread_safety_with_reentrant_calls():
    """Many threads requesting same key should only call func once."""
    calls = CountedCall()
    lock = threading.Lock()

    @lru_cache(maxsize=16)
    def func(n):
        with lock:
            calls.count += 1
        time.sleep(0.01)
        return n * 2

    results = []
    threads = []

    def worker():
        results.append(func(42))

    for _ in range(10):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert results == [84] * 10
    assert calls.count == 1


def test_fifo_cache_bare_decorated_function_object():
    def func(n):
        return n + 1

    decorated = fifo_cache(func)
    assert decorated(1) == 2
    assert decorated.cache_parameters() == {"maxsize": 128, "typed": False}


def test_rr_cache_bare_decorated_with_choice():
    def func(n):
        return n + 1

    decorated = rr_cache(maxsize=func)
    assert decorated(1) == 2
    assert decorated.cache_parameters() == {"maxsize": 128, "typed": False}


def test_ttl_cache_bare_decorated_function_object():
    def func(n):
        return n + 1

    decorated = ttl_cache(func)
    assert decorated(1) == 2
    assert decorated(2) == 3
    assert decorated.cache_parameters() == {"maxsize": 128, "typed": False}


def test_lru_and_lfu_bare_decorator_return_wrapper():
    @lru_cache
    def a(n):
        return n

    @lfu_cache
    def b(n):
        return n

    assert a(1) == 1
    assert b(1) == 1
    assert a.cache_parameters() == {"maxsize": 128, "typed": False}
    assert b.cache_parameters() == {"maxsize": 128, "typed": False}


def test_cache_info_type():
    @lru_cache(maxsize=2)
    def func(n):
        return n

    func(1)
    info = func.cache_info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 0
    assert info.misses == 1
    assert info.maxsize == 2
    assert info.currsize == 1


def test_ttl_cache_expire_during_get():
    current = [0.0]

    @ttl_cache(maxsize=4, ttl=5, timer=lambda: current[0])
    def func(n):
        return n

    func(1)
    func(2)
    current[0] = 10.0
    # accesses should re-execute
    assert func(1) == 1
    assert func(2) == 2
    info = func.cache_info()
    assert info.misses == 4


def test_maxsize_zero_behaves():
    calls = CountedCall()

    @lru_cache(maxsize=0)
    def func(n):
        return calls(n)

    func(1)
    func(1)
    assert calls.count == 2
    assert func.cache_info().currsize == 0


def test_kwargs_and_various_args():
    calls = CountedCall()

    @lru_cache(maxsize=8)
    def func(a, b=0):
        return calls(a, b)

    func(1)
    func(1, b=0)  # distinct key from func(1)
    func(1, 0)  # distinct key as well
    info = func.cache_info()
    assert info.misses == 3
    func(1)
    assert func.cache_info().hits == 1


def test_multiple_instances_independent():
    @lru_cache(maxsize=2)
    def f1(n):
        return n

    @lru_cache(maxsize=2)
    def f2(n):
        return n

    f1(1)
    assert f1.cache_info().currsize == 1
    assert f2.cache_info().currsize == 0


def test_ttl_cache_with_typed():
    calls = CountedCall()

    @ttl_cache(maxsize=4, ttl=1000, timer=lambda: 0.0, typed=True)
    def func(n):
        return calls(n)

    func(1)
    func(1.0)
    assert calls.count == 2
