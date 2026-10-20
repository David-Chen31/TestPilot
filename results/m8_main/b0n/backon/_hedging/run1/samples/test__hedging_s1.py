from __future__ import annotations

import asyncio
import operator
import threading
import time

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    hedge,
    on_hedge,
)


# A fast deterministic wait generator: yields 0 always.
def zero_wait():
    while True:
        yield 0


def _wait_gen_that_is_callable(*_, **__):
    # wait_gen expected to be a generator function; but the retry inner
    # likely calls wait_gen(**kwargs). We'll provide a proper generator fn.
    def gen():
        while True:
            yield 0

    return gen


# ---------------------------------------------------------------------------
# hedge()
# ---------------------------------------------------------------------------


def test_hedge_sync_returns_first_success():
    calls = []

    def target():
        calls.append(1)
        return "ok"

    result = hedge(target, zero_wait, max_hedge=2)
    assert result == "ok"
    assert len(calls) == 2  # both submitted immediately


def test_hedge_sync_both_fail_raises_one_of_them():
    def target():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        hedge(target, zero_wait, max_hedge=2)


def test_hedge_sync_first_exc_when_no_success():
    # ONE path succeeds fast, other raises -> success returned
    def target():
        return 42

    result = hedge(target, zero_wait, max_hedge=3)
    assert result == 42


def test_hedge_sync_timeout_zero_raises_hedge_error():
    # With timeout=0, futures_wait returns immediately with (almost certainly)
    # nothing done -> no first_exc -> HedgeError.
    def target():
        time.sleep(0.5)
        return "late"

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        hedge(target, zero_wait, max_hedge=2, timeout=0.0)


def test_hedge_sync_with_args_and_kw():
    seen = []

    def target(a, b=None):
        seen.append((a, b))
        return a + (b or 0)

    result = hedge(target, zero_wait, args=(3,), kw={"b": 4}, max_hedge=2)
    assert result == 7
    assert all(call == (3, 4) for call in seen)
    assert len(seen) == 2


def test_hedge_sync_kw_none_ok():
    def target():
        return "x"

    assert hedge(target, zero_wait, max_hedge=1, kw=None) == "x"


def test_hedge_sync_exception_predicate_filtering():
    # Only ValueError should be retried. Since it always raises, retry is
    # attempted but eventually giveup -> propagates.
    def target():
        raise ValueError("v")

    with pytest.raises(ValueError):
        hedge(target, zero_wait, exception=ValueError, max_hedge=1, max_tries=1)


def test_hedge_sync_non_matching_exception_propagates():
    def target():
        raise KeyError("k")

    with pytest.raises(KeyError):
        hedge(target, zero_wait, exception=ValueError, max_hedge=1, max_tries=1)


def test_hedge_sync_on_hedge_single_handler():
    events = []

    def target():
        return "ok"

    def handler(ctx):
        events.append(ctx)

    hedge(target, zero_wait, max_hedge=2, on_hedge=handler)
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["hedge_count"] == 2
    assert events[0]["target"] is target


def test_hedge_sync_on_hedge_iterable_handlers():
    events = []

    def handler1(ctx):
        events.append(("h1", ctx["hedge_count"]))

    def handler2(ctx):
        events.append(("h2", ctx["hedge_count"]))

    def target():
        return "ok"

    hedge(target, zero_wait, max_hedge=3, on_hedge=[handler1, handler2])
    assert events == [("h1", 3), ("h2", 3)]


def test_hedge_sync_on_hedge_none_no_error():
    def target():
        return "ok"

    assert hedge(target, zero_wait, max_hedge=1, on_hedge=None) == "ok"


def test_hedge_dispatches_async_coroutine_function():
    async def target():
        return "async-ok"

    result = hedge(target, zero_wait, max_hedge=2)
    # hedge() returns an awaitable from _hedge_async
    assert asyncio.iscoroutine(result)
    assert asyncio.run(result) == "async-ok"


def test_hedge_async_returns_first_success():
    async def target():
        return "ok"

    result = asyncio.run(hedge(target, zero_wait, max_hedge=2))
    assert result == "ok"


def test_hedge_async_failures_propagate():
    async def target():
        raise ValueError("async-boom")

    with pytest.raises(ValueError, match="async-boom"):
        asyncio.run(hedge(target, zero_wait, max_hedge=2))


def test_hedge_async_timeout_zero_raises_hedge_error():
    async def target():
        await asyncio.sleep(0.5)
        return "late"

    with pytest.raises(HedgeError):
        asyncio.run(hedge(target, zero_wait, max_hedge=2, timeout=0.0))


def test_hedge_async_on_hedge_handler():
    events = []

    async def target():
        return "ok"

    def handler(ctx):
        events.append(ctx)

    asyncio.run(hedge(target, zero_wait, max_hedge=2, on_hedge=handler))
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["hedge_count"] == 2


def test_hedge_async_handler_iterable():
    events = []

    async def target():
        return "ok"

    def h1(ctx):
        events.append(("h1", ctx["max_hedge"]))

    def h2(ctx):
        events.append(("h2", ctx["max_hedge"]))

    asyncio.run(hedge(target, zero_wait, max_hedge=1, on_hedge=[h1, h2]))
    assert events == [("h1", 1), ("h2", 1)]


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------


def test_on_hedge_decorator_sync():
    calls = []

    @on_hedge(zero_wait, max_hedge=2)
    def target(x, y=0):
        calls.append((x, y))
        return x + y

    assert target(2, y=3) == 5
    assert len(calls) == 2
    assert all(c == (2, 3) for c in calls)


def test_on_hedge_decorator_sync_handler():
    events = []

    @on_hedge(zero_wait, max_hedge=2, on_hedge=lambda ctx: events.append(ctx))
    def target():
        return "ok"

    assert target() == "ok"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2


def test_on_hedge_decorator_preserves_wraps():
    @on_hedge(zero_wait, max_hedge=1)
    def myfunc():
        """my doc"""
        return 1

    assert myfunc.__name__ == "myfunc"
    assert myfunc.__doc__ == "my doc"


def test_on_hedge_decorator_async():
    calls = []

    @on_hedge(zero_wait, max_hedge=2)
    async def target(x):
        calls.append(x)
        return x * 2

    assert asyncio.run(target(5)) == 10
    assert len(calls) == 2


def test_on_hedge_decorator_async_wraps():
    @on_hedge(zero_wait, max_hedge=1)
    async def acoro():
        """async doc"""
        return 1

    assert acoro.__name__ == "acoro"
    assert acoro.__doc__ == "async doc"


def test_on_hedge_decorator_async_error():
    @on_hedge(zero_wait, max_hedge=2)
    async def target():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        asyncio.run(target())


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------


def test_hedging_retrying_call_sync():
    seen = []

    def target(a, b=0):
        seen.append((a, b))
        return a * b

    hr = HedgingRetrying(zero_wait, max_hedge=2)
    result = hr.call(target, 3, b=4)
    assert result == 12
    assert all(c == (3, 4) for c in seen)
    assert len(seen) == 2


def test_hedging_retrying_call_rejects_async():
    async def target():
        return 1

    hr = HedgingRetrying(zero_wait, max_hedge=1)
    with pytest.raises(TypeError, match="Use async_call"):
        hr.call(target)


def test_hedging_retrying_call_error_propagates():
    def target():
        raise ValueError("bad")

    hr = HedgingRetrying(zero_wait, max_hedge=2)
    with pytest.raises(ValueError, match="bad"):
        hr.call(target)


def test_hedging_retrying_async_call():
    seen = []

    async def target(a, b=0):
        seen.append((a, b))
        return a + b

    hr = HedgingRetrying(zero_wait, max_hedge=2)
    result = asyncio.run(hr.async_call(target, 1, b=2))
    assert result == 3
    assert len(seen) == 2


def test_hedging_retrying_async_call_error():
    async def target():
        raise RuntimeError("fail")

    hr = HedgingRetrying(zero_wait, max_hedge=2)
    with pytest.raises(RuntimeError, match="fail"):
        asyncio.run(hr.async_call(target))


def test_hedging_retrying_context_managers():
    hr = HedgingRetrying(zero_wait, max_hedge=1)
    with hr as entered:
        assert entered is hr
    # exits cleanly


def test_hedging_retrying_async_context_managers():
    async def main():
        hr = HedgingRetrying(zero_wait, max_hedge=1)
        async with hr as entered:
            assert entered is hr

    asyncio.run(main())


def test_hedging_retrying_timeout_zero_sync():
    def target():
        time.sleep(0.3)
        return "late"

    hr = HedgingRetrying(zero_wait, max_hedge=2, timeout=0.0)
    with pytest.raises(HedgeError):
        hr.call(target)


def test_hedging_retrying_on_hedge_events():
    events = []

    def handler(ctx):
        events.append(ctx)

    def target():
        return "ok"

    hr = HedgingRetrying(zero_wait, max_hedge=3, on_hedge=handler)
    assert hr.call(target) == "ok"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 3
    assert events[0]["hedge_count"] == 3


def test_hedging_retrying_exception_filter_sync():
    def target():
        raise KeyError("wrong type")

    hr = HedgingRetrying(zero_wait, max_hedge=1, exception=ValueError, max_tries=1)
    with pytest.raises(KeyError):
        hr.call(target)


# ---------------------------------------------------------------------------
# HedgeError basic
# ---------------------------------------------------------------------------


def test_hedge_error_is_exception():
    err = HedgeError("msg")
    assert isinstance(err, Exception)
    assert str(err) == "msg"
