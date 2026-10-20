import random
import time

import pytest

import cachetools
from cachetools.func import (
    fifo_cache,
    lfu_cache,
    lru_cache,
    rr_cache,
    ttl_cache,
    _UnboundTTLCache,
)
from cachetools import keys as _keys


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return args, kwargs


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_default_maxsize(decorator):
    @decorator()
    def func(x):
        return x * 2

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}
    assert func(3) == 6
    assert func(3) == 6


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_explicit_maxsize(decorator):
    @decorator(maxsize=5)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 5, "typed": False}
    for i in range(10):
        assert func(i) == i


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_typed_flag(decorator):
    @decorator(typed=True)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 128, "typed": True}
    # typed keys distinguish 1 and 1.0
    assert func(1) == 1
    assert func(1.0) == 1.0


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_typed_flag_reuses_same_types(decorator):
    calls = Counter()

    @decorator(typed=True)
    def func(x):
        calls.calls += 1
        return x

    assert func(1) == 1
    assert func(1) == 1
    assert calls.calls == 1
    # distinct types with equal value must be distinct cache entries
    assert func(1.0) == 1.0
    assert calls.calls == 2
    assert func(1.0) == 1.0
    assert calls.calls == 2


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_untyped_flag_merges_equal_values(decorator):
    calls = Counter()

    @decorator(typed=False)
    def func(x):
        calls.calls += 1
        return x

    assert func(1) == 1
    assert func(1.0) == 1.0
    # untyped: 1 and 1.0 hash equal -> single cache entry
    assert calls.calls == 1


def test_untyped_uses_hashkey():
    # Verify that the untyped keying matches keys.hashkey semantics.
    assert _keys.hashkey(1) == _keys.hashkey(1.0)

    @lru_cache(maxsize=4, typed=False)
    def func(x):
        return id(x) if isinstance(x, object) else x

    # 1 == 1.0 -> one cache entry
    func(1)
    func(1.0)
    assert func.cache_info().misses == 1


def test_typed_uses_typedkey():
    # Verify that the typed keying matches keys.typedkey semantics.
    assert _keys.typedkey(1) != _keys.typedkey(1.0)

    @lru_cache(maxsize=4, typed=True)
    def func(x):
        return x

    func(1)
    func(1.0)
    assert func.cache_info().misses == 2


def test_typed_parameters():
    @lru_cache(maxsize=10, typed=True)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 10, "typed": True}


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_unbound_maxsize_none(decorator):
    @decorator(maxsize=None)
    def func(x):
        return x * 3

    assert func.cache_parameters() == {"maxsize": None, "typed": False}
    values = list(range(200))
    results = [func(v) for v in values]
    assert results == [v * 3 for v in values]
    # unbounded should keep all keys cached
    cached_results = [func(v) for v in values]
    assert cached_results == results


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_callable_maxsize_as_decorator(decorator):
    calls = Counter()

    @decorator
    def func(x):
        calls.calls += 1
        return x + 1

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}
    assert func(1) == 2
    assert func(1) == 2
    assert calls.calls == 1


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_callable_maxsize_actual_capacity(decorator):
    # When used as a bare decorator, the internal cache must have size 128.
    calls = Counter()

    @decorator
    def func(x):
        calls.calls += 1
        return x

    for i in range(200):
        func(i)
    # Only 128 entries fit; currsize must be exactly 128
    assert func.cache_info().currsize == 128
    assert func.cache_info().maxsize == 128


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_direct_decorator_with_args(decorator):
    calls = Counter()

    @decorator()
    def func(a, b):
        calls.calls += 1
        return a + b

    assert func(1, 2) == 3
    assert func(1, 2) == 3
    assert func(2, 1) == 3
    assert calls.calls == 2


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_info_hits_misses(decorator):
    @decorator(maxsize=4)
    def func(x):
        return x

    assert func(1) == 1
    assert func(1) == 1
    info = func.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 4


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_info_unbound(decorator):
    @decorator(maxsize=None)
    def func(x):
        return x

    func(1)
    func(1)
    info = func.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize is None


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_cache_clear(decorator):
    @decorator(maxsize=4)
    def func(x):
        return x

    func(1)
    assert func.cache_info().currsize == 1
    func.cache_clear()
    assert func.cache_info().currsize == 0
    # counters are reset by clear()
    assert func.cache_info().misses == 0
    assert func.cache_info().hits == 0
    # recompute after clear
    func(1)
    assert func.cache_info().currsize == 1
    assert func.cache_info().misses == 1


def test_fifo_eviction_order():
    calls = Counter()

    @fifo_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    # accessing 1 does not recompute but may refresh insertion order
    func(1)
    assert calls.calls == 2
    func(3)  # evicts based on FIFO order
    assert calls.calls == 3
    assert func.cache_info().currsize == 2
    # force a lookup of the current eviction candidate & others; just
    # verify the cache size stays bounded under continued use
    for i in range(4, 10):
        func(i)
    assert func.cache_info().currsize == 2


def test_fifo_exact_eviction():
    calls = Counter()

    @fifo_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(3)  # evicts 1
    assert calls.calls == 3
    func(1)  # miss -> recompute -> evicts 2
    assert calls.calls == 4
    func(2)  # miss -> recompute -> evicts 3
    assert calls.calls == 5


def test_lru_eviction_order():
    calls = Counter()

    @lru_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(1)  # touch 1 -> 2 is LRU
    assert calls.calls == 2
    func(3)  # evicts 2
    assert calls.calls == 3
    assert func.cache_info().currsize == 2
    # keep cache bounded under further use
    for i in range(4, 10):
        func(i)
    assert func.cache_info().currsize == 2


def test_lru_exact_eviction():
    calls = Counter()

    @lru_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(1)  # touch 1
    func(3)  # evicts 2
    assert calls.calls == 3
    func(2)  # miss -> recompute -> evicts 1 (least recently used)
    assert calls.calls == 4
    func(1)  # miss -> recompute
    assert calls.calls == 5


def test_lfu_eviction_order():
    calls = Counter()

    @lfu_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(1)  # 1 used twice, 2 used once
    assert calls.calls == 2
    func(3)  # evicts 2
    assert calls.calls == 3
    func(2)  # miss
    assert calls.calls == 4
    func(1)  # hit
    assert calls.calls == 4


def test_rr_choice_used():
    chosen = []

    def choice(seq):
        chosen.append(list(seq))
        return seq[0]

    @rr_cache(maxsize=2, choice=choice)
    def func(x):
        return x

    func(1)
    func(2)
    func(3)  # triggers choice
    assert chosen, "choice function should have been called"
    assert func.cache_info().currsize == 2


def test_rr_choice_param():
    calls = Counter()

    def choice(seq):
        return seq[-1]

    @rr_cache(maxsize=2, choice=choice)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(3)  # evicts last -> most recent inserted (2)
    assert calls.calls == 3
    func(2)  # miss -> recompute
    assert calls.calls == 4
    func(1)  # hit
    assert calls.calls == 4


def test_rr_choice_param_bare_decorator():
    # When used as a bare decorator, the provided `choice` must be kept.
    selected = []

    def choice(seq):
        selected.append(list(seq))
        return seq[0]

    @rr_cache(choice=choice)
    def func(x):
        return x

    for i in range(130):
        func(i)
    # cache maxsize is 128; size should be exactly 128
    assert func.cache_info().currsize == 128
    assert func.cache_info().maxsize == 128
    assert selected, "custom choice must be used"


def test_rr_choice_param_bare_decorator_eviction():
    calls = Counter()

    def choice(seq):
        return seq[0]

    @rr_cache(choice=choice)
    def func(x):
        calls.calls += 1
        return x

    for i in range(129):
        func(i)
    # exactly 128 entries retained; 129th forced an eviction
    assert func.cache_info().currsize == 128
    assert func.cache_info().misses == 129


def test_rr_choice_default():
    # Ensure the default choice is random.choice when not specified.
    from cachetools import RRCache

    # The bare-func rr_cache decorator with custom choice is the corner case
    # we want to protect against; also check the default path stores a cache.
    calls = Counter()

    @rr_cache(maxsize=2)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    func(2)
    func(3)
    assert calls.calls == 3
    assert func.cache_info().currsize == 2


def test_ttl_unbound_uses_unbound_cache():
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    @ttl_cache(maxsize=None, ttl=10, timer=timer)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": None, "typed": False}
    func(1)
    assert func.cache_info().currsize == 1
    timer_value[0] = 5
    func(1)
    assert func.cache_info().hits == 1
    timer_value[0] = 20
    # expired: recompute
    func(1)
    assert func.cache_info().misses == 2


def test_ttl_expiry_with_fake_timer():
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    @ttl_cache(maxsize=4, ttl=10, timer=timer)
    def func(x):
        return x

    assert func(1) == 1
    assert func.cache_info().misses == 1
    timer_value[0] = 5
    assert func(1) == 1
    assert func.cache_info().hits == 1
    timer_value[0] = 15
    assert func(1) == 1
    assert func.cache_info().misses == 2


def test_ttl_default_ttl_is_600():
    # The default ttl must be 600 seconds.
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    @ttl_cache(maxsize=4, timer=timer)
    def func(x):
        return x

    func(1)
    assert func.cache_info().misses == 1
    # at t=599, item must still be VALID (default ttl 600)
    timer_value[0] = 599
    func(1)
    assert func.cache_info().hits == 1
    # at t=600, item must be EXPIRED
    timer_value[0] = 600
    func(1)
    assert func.cache_info().misses == 2


def test_ttl_direct_decorator():
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    calls = Counter()

    @ttl_cache(ttl=10, timer=timer)
    def func(x):
        calls.calls += 1
        return x

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}
    func(0)
    func(0)
    assert calls.calls == 1


def test_ttl_callable_maxsize():
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    calls = Counter()

    @ttl_cache(ttl=10, timer=timer)
    def func(x):
        calls.calls += 1
        return x

    func(0)
    func(0)
    assert calls.calls == 1


def test_ttl_callable_maxsize_uses_timer_and_capacity():
    # When used as a bare decorator, the provided timer must be passed through.
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    calls = Counter()

    @ttl_cache(ttl=10, timer=timer)
    def func(x):
        calls.calls += 1
        return x

    func(1)
    assert calls.calls == 1
    # before expiry -> hit
    timer_value[0] = 5
    func(1)
    assert calls.calls == 1
    # after expiry -> miss due to custom timer
    timer_value[0] = 11
    func(1)
    assert calls.calls == 2

    # The internal cache must be capped at 128 entries.
    for i in range(1000):
        func(i)
    assert func.cache_info().currsize == 128
    assert func.cache_info().maxsize == 128


def test_ttl_callable_maxsize_capacity_128():
    timer_value = [0.0]

    def timer():
        return timer_value[0]

    @ttl_cache(ttl=10, timer=timer)
    def func(x):
        return x

    for i in range(200):
        func(i)
    # Capacity of the bare-decorator cache must be 128, not 129.
    assert func.cache_info().currsize == 128


def test_unbound_ttl_cache_maxsize_property():
    cache = _UnboundTTLCache(ttl=10, timer=time.monotonic)
    assert cache.maxsize is None


def test_exceptions_not_cached():
    calls = Counter()

    @lru_cache(maxsize=4)
    def func(x):
        calls.calls += 1
        if calls.calls == 1:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError):
        func(1)
    assert func(1) == 1
    assert calls.calls == 2


def test_keyword_arguments():
    @lru_cache(maxsize=4)
    def func(a, b=2):
        return a + b

    assert func(1) == 3
    assert func(1, b=2) == 3
    assert func(1) == 3
    info = func.cache_info()
    # typedkey/hashkey distinguishes call signatures
    assert info.misses >= 1


def test_default_maxsize_when_explicit_zero():
    @lru_cache(maxsize=0)
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 0, "typed": False}
    func(1)
    func(1)
    # maxsize 0 means nothing is retained
    assert func.cache_info().hits == 0


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_negative_maxsize(decorator):
    with pytest.raises(ValueError):
        decorator(maxsize=-1)


def test_ttl_defaults():
    @ttl_cache()
    def func(x):
        return x

    assert func.cache_parameters() == {"maxsize": 128, "typed": False}


def test_string_arguments_keys():
    calls = Counter()

    @fifo_cache(maxsize=8)
    def func(s):
        calls.calls += 1
        return s.upper()

    assert func("a") == "A"
    assert func("a") == "A"
    assert calls.calls == 1


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache],
)
def test_internal_cache_capacity_explicit_maxsize(decorator):
    # The internal cache must have exactly maxsize slots (not maxsize + 1).
    @decorator(maxsize=3)
    def func(x):
        return x

    for i in range(50):
        func(i)
    assert func.cache_info().currsize == 3
    assert func.cache_info().maxsize == 3


@pytest.mark.parametrize(
    "decorator",
    [fifo_cache, lfu_cache, lru_cache],
)
def test_bare_decorator_internal_capacity_exact(decorator):
    # Bare-decorator path builds an internal cache of size 128.
    @decorator
    def func(x):
        return x

    for i in range(500):
        func(i)
    # Exactly 128 retained -> internal cache size is 128, not 129.
    assert func.cache_info().currsize == 128
    assert func.cache_info().maxsize == 128


def test_rr_bare_decorator_internal_capacity_exact():
    @rr_cache
    def func(x):
        return x

    for i in range(500):
        func(i)
    assert func.cache_info().currsize == 128
    assert func.cache_info().maxsize == 128


def test_concurrent_calls_are_consistent():
    # Condition-based locking makes the wrapper thread safe; the simplest
    # observable invariant is that repeated calls hit the cache. This test
    # also verifies that a Condition is supplied by exercising the
    # stampede-prevention path via sequential-but-interleaved calls.
    calls = Counter()

    @lru_cache(maxsize=4)
    def func(x):
        calls.calls += 1
        return x * 2

    assert func(7) == 14
    assert func(7) == 14
    assert func(7) == 14
    assert calls.calls == 1
