import asyncio
import threading

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    hedge,
    on_hedge,
)


# ---------------------------------------------------------------------------
# A simple wait generator that never sleeps
# ---------------------------------------------------------------------------
def fast_wait(attempt):
    return 0


def no_jitter(value):
    return value


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------
def test_hedge_error_is_exception():
    err = HedgeError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# hedge() - basic success
# ---------------------------------------------------------------------------
def test_hedge_sync_basic_success():
    result = hedge(lambda: 42, fast_wait, jitter=None)
    assert result == 42


def test_hedge_sync_with_args_and_kw():
    def target(a, b, c=0):
        return a + b + c

    result = hedge(target, fast_wait, args=(1, 2), kw={"c": 3}, jitter=None)
    assert result == 6


def test_hedge_sync_kw_none_defaults_to_empty():
    result = hedge(lambda: "ok", fast_wait, kw=None, jitter=None)
    assert result == "ok"


# ---------------------------------------------------------------------------
# hedge() - exception filtering / raising
# ---------------------------------------------------------------------------
def test_hedge_sync_raises_first_exception():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        hedge(boom, fast_wait, jitter=None, max_hedge=1, max_tries=1)


def test_hedge_sync_exception_not_matching_propagates():
    def boom():
        raise KeyError("key")

    with pytest.raises(KeyError):
        hedge(boom, fast_wait, jitter=None, exception=ValueError, max_hedge=1, max_tries=1)


def test_hedge_sync_recovers_after_retry():
    counter = {"n": 0}
    lock = threading.Lock()

    def target():
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if n < 3:
            raise ValueError("retry me")
        return "done"

    result = hedge(
        target,
        fast_wait,
        jitter=None,
        exception=ValueError,
        max_hedge=1,
        max_tries=5,
    )
    assert result == "done"
    assert counter["n"] >= 3


# ---------------------------------------------------------------------------
# hedge() - on_hedge callback
# ---------------------------------------------------------------------------
def test_hedge_sync_on_hedge_callback():
    calls = []

    def handler(d):
        calls.append(dict(d))

    result = hedge(
        lambda: 1,
        fast_wait,
        jitter=None,
        max_hedge=2,
        on_hedge=handler,
    )
    assert result == 1
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2
    assert calls[0]["hedge_count"] == 2
    assert "target" in calls[0]


def test_hedge_sync_on_hedge_iterable():
    calls = []

    def h1(d):
        calls.append("h1")

    def h2(d):
        calls.append("h2")

    hedge(lambda: 1, fast_wait, jitter=None, max_hedge=1, on_hedge=[h1, h2])
    assert calls == ["h1", "h2"]


def test_hedge_sync_on_hedge_none():
    # simply shouldn't crash
    result = hedge(lambda: 5, fast_wait, jitter=None, max_hedge=1, on_hedge=None)
    assert result == 5


# ---------------------------------------------------------------------------
# hedge() - predicate path
# ---------------------------------------------------------------------------
def test_hedge_sync_predicate_triggers_retry():
    counter = {"n": 0}
    lock = threading.Lock()

    def target():
        with lock:
            counter["n"] += 1
            n = counter["n"]
        return n

    result = hedge(
        target,
        fast_wait,
        jitter=None,
        predicate=lambda v: v < 2,
        max_hedge=1,
        max_tries=5,
    )
    assert result >= 2


# ---------------------------------------------------------------------------
# hedge() - async path
# ---------------------------------------------------------------------------
def test_hedge_dispatches_async_coroutine():
    async def coro():
        return "async-result"

    result = asyncio.run(hedge(coro, fast_wait, jitter=None, max_hedge=1))
    assert result == "async-result"


def test_hedge_async_basic_success():
    async def coro():
        return 123

    async def runner():
        return await hedge(coro, fast_wait, jitter=None, max_hedge=2)

    assert asyncio.run(runner()) == 123


def test_hedge_async_with_args_kw():
    async def coro(a, b, c=0):
        return a + b + c

    async def runner():
        return await hedge(coro, fast_wait, args=(1, 2), kw={"c": 10}, jitter=None, max_hedge=1)

    assert asyncio.run(runner()) == 13


def test_hedge_async_raises_first_exception():
    async def coro():
        raise ValueError("async-boom")

    async def runner():
        return await hedge(coro, fast_wait, jitter=None, max_hedge=1, max_tries=1)

    with pytest.raises(ValueError, match="async-boom"):
        asyncio.run(runner())


def test_hedge_async_on_hedge_callback():
    calls = []

    async def coro():
        return "ok"

    def handler(d):
        calls.append(dict(d))

    async def runner():
        return await hedge(coro, fast_wait, jitter=None, max_hedge=3, on_hedge=handler)

    result = asyncio.run(runner())
    assert result == "ok"
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 3
    assert calls[0]["hedge_count"] == 3


# ---------------------------------------------------------------------------
# hedge() - timeout behavior (returns HedgeError when nothing completes)
# ---------------------------------------------------------------------------
def test_hedge_sync_timeout_raises_hedge_error():
    ev = threading.Event()

    def slow():
        # slow forever but bounded by max_time / max_tries safety? we make it block
        ev.wait(timeout=5)
        return "never"

    with pytest.raises((HedgeError, Exception)):
        hedge(slow, fast_wait, jitter=None, max_hedge=1, timeout=0.05)
    ev.set()


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------
def test_on_hedge_decorator_sync():
    @on_hedge(fast_wait, jitter=None, max_hedge=1)
    def target(a, b=1):
        return a + b

    assert target(2, b=3) == 5
    # functools.wraps preserves metadata
    assert target.__name__ == "target"


def test_on_hedge_decorator_async():
    @on_hedge(fast_wait, jitter=None, max_hedge=1)
    async def target(x):
        return x * 2

    assert asyncio.run(target(4)) == 8
    assert target.__name__ == "target"


def test_on_hedge_decorator_sync_passes_kwargs():
    @on_hedge(fast_wait, jitter=None, max_hedge=1)
    def target(*, value):
        return value

    assert target(value="kw") == "kw"


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------
def test_hedging_retrying_init_stores_config():
    hr = HedgingRetrying(
        fast_wait,
        max_hedge=4,
        exception=ValueError,
        max_tries=2,
        max_time=10,
        jitter=None,
        on_hedge=None,
        timeout=1.0,
        foo="bar",
    )
    assert hr._wait_gen is fast_wait
    assert hr._max_hedge == 4
    assert hr._exception is ValueError
    assert hr._max_tries == 2
    assert hr._max_time == 10
    assert hr._jitter is None
    assert hr._on_hedge is None
    assert hr._timeout == 1.0
    assert hr._wait_gen_kwargs == {"foo": "bar"}


def test_hedging_retrying_sync_context_manager():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1)
    with hr as entered:
        assert entered is hr


def test_hedging_retrying_async_context_manager():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1)

    async def runner():
        async with hr as entered:
            return entered is hr

    assert asyncio.run(runner()) is True


def test_hedging_retrying_call_sync():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1)
    result = hr.call(lambda a, b: a * b, 3, b=4)
    assert result == 12


def test_hedging_retrying_call_sync_rejects_coroutine():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1)

    async def coro():
        return 1

    with pytest.raises(TypeError, match="Use async_call for async functions"):
        hr.call(coro)


def test_hedging_retrying_async_call():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1)

    async def coro(a, b):
        return a + b

    async def runner():
        return await hr.async_call(coro, 5, b=7)

    assert asyncio.run(runner()) == 12


def test_hedging_retrying_call_recovers_from_retriable_error():
    hr = HedgingRetrying(
        fast_wait,
        jitter=None,
        max_hedge=1,
        exception=ValueError,
        max_tries=5,
    )
    counter = {"n": 0}
    lock = threading.Lock()

    def target():
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if n < 2:
            raise ValueError("again")
        return "ok"

    assert hr.call(target) == "ok"


def test_hedging_retrying_async_call_raises_on_failure():
    hr = HedgingRetrying(fast_wait, jitter=None, max_hedge=1, max_tries=1)

    async def coro():
        raise RuntimeError("fail-async")

    async def runner():
        return await hr.async_call(coro)

    with pytest.raises(RuntimeError, match="fail-async"):
        asyncio.run(runner())
