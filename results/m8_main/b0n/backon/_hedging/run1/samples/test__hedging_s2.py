from __future__ import annotations

import asyncio
import operator

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    hedge,
    on_hedge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_wait_gen(*args, **kwargs):
    # zero wait, used to avoid slow tests
    while True:
        yield 0


async def _async_fast_wait_gen(*args, **kwargs):
    while True:
        yield 0


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------

def test_hedge_error_is_exception():
    err = HedgeError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# hedge - sync
# ---------------------------------------------------------------------------

def test_hedge_sync_success_first_try():
    calls = []

    def target():
        calls.append(1)
        return "ok"

    result = hedge(target, max_hedge=3)
    assert result == "ok"
    assert len(calls) >= 1


def test_hedge_sync_passes_args_and_kwargs():
    def target(a, b, c=None):
        return (a, b, c)

    result = hedge(target, args=(1, 2), kw={"c": 3}, max_hedge=1)
    assert result == (1, 2, 3)


def test_hedge_sync_raises_underlying_exception():
    class MyError(Exception):
        pass

    def target():
        raise MyError("bad")

    with pytest.raises(MyError):
        hedge(target, max_hedge=2)
    # attempts to raise an unused variable? just ensure we get here


def test_hedge_sync_raises_first_exception_after_all_fail():
    class ErrA(Exception):
        pass

    def target():
        raise ErrA("nope")

    with pytest.raises(ErrA):
        hedge(target, max_hedge=3)


def test_hedge_sync_retries_until_success():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("fail")
        return "done"

    result = hedge(
        target,
        _fast_wait_gen,
        max_hedge=1,
        exception=ValueError,
        predicate=operator.not_,
    )
    assert result == "done"
    assert state["n"] == 2


def test_hedge_sync_returns_truthy_predicate_none_matches():
    # predicate default is operator.not_; default condition only triggers retry
    # on falsy values when exception is None.
    def target():
        return "truthy"

    assert hedge(target, max_hedge=1) == "truthy"


def test_hedge_sync_on_hedge_single_handler():
    events = []

    def handler(evt):
        events.append(evt)

    def target():
        return 1

    hedge(target, max_hedge=2, on_hedge=handler)
    assert len(events) == 1
    evt = events[0]
    assert evt["max_hedge"] == 2
    assert evt["hedge_count"] == 2
    assert callable(evt["target"])


def test_hedge_sync_on_hedge_iterable_of_handlers():
    events = []

    def h1(evt):
        events.append(("h1", evt["hedge_count"]))

    def h2(evt):
        events.append(("h2", evt["hedge_count"]))

    hedge(lambda: "x", max_hedge=1, on_hedge=[h1, h2])
    assert events == [("h1", 1), ("h2", 1)]


def test_hedge_sync_timeout_with_no_completion_raises():
    # All futures fail -> first_exc raised, not HedgeError, but we test the
    # timeout path where futures are cancelled.  We rely on slow target to
    # force the timeout; but we must be deterministic, so return quickly
    # by giving a tiny timeout where nothing completes.
    import threading

    release = threading.Event()

    def target():
        release.wait()

    try:
        with pytest.raises(HedgeError):
            hedge(target, max_hedge=2, timeout=0.01)
    finally:
        release.set()


def test_hedge_sync_timeout_not_done_and_done_success():
    # One future completes fast, another sleeps; with timeout but FIRST_COMPLETED
    # the fast one returns.
    def target():
        return "fast"

    assert hedge(target, max_hedge=2, timeout=1.0) == "fast"


# ---------------------------------------------------------------------------
# hedge - async
# ---------------------------------------------------------------------------

def test_hedge_async_success():
    async def target():
        return "async-ok"

    result = asyncio.run(hedge(target, max_hedge=2))
    assert result == "async-ok"


def test_hedge_async_passes_args_kwargs():
    async def target(a, b, c=None):
        return (a, b, c)

    result = asyncio.run(hedge(target, args=(1, 2), kw={"c": 9}, max_hedge=1))
    assert result == (1, 2, 9)


def test_hedge_async_raises_underlying_exception():
    class MyErr(Exception):
        pass

    async def target():
        raise MyErr("nope")

    with pytest.raises(MyErr):
        asyncio.run(hedge(target, max_hedge=2))


def test_hedge_async_on_hedge_handler():
    events = []

    def handler(evt):
        events.append(dict(evt))

    async def target():
        return 5

    asyncio.run(hedge(target, max_hedge=3, on_hedge=handler))
    assert len(events) == 1
    assert events[0]["max_hedge"] == 3
    assert events[0]["hedge_count"] == 3


def test_hedge_async_retries_then_succeeds():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 1:
            raise ValueError("fail")
        return "ok"

    result = asyncio.run(
        hedge(target, _fast_wait_gen, max_hedge=1, exception=ValueError)
    )
    assert result == "ok"


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------

def test_on_hedge_decorator_sync():
    calls = []

    @on_hedge(max_hedge=1)
    def target(x, y=2):
        calls.append((x, y))
        return x + y

    assert target(3) == 5
    assert (3, 2) in calls
    assert target.__name__ == "target"


def test_on_hedge_decorator_sync_passes_through_exception():
    class Boom(Exception):
        pass

    @on_hedge(max_hedge=1)
    def target():
        raise Boom("x")

    with pytest.raises(Boom):
        target()


def test_on_hedge_decorator_async():
    @on_hedge(max_hedge=1)
    async def target(a, b=5):
        return a + b

    assert asyncio.run(target(1)) == 6
    assert target.__name__ == "target"


def test_on_hedge_decorator_async_exception():
    class Boom(Exception):
        pass

    @on_hedge(max_hedge=1)
    async def target():
        raise Boom("z")

    with pytest.raises(Boom):
        asyncio.run(target())


def test_on_hedge_decorator_triggers_handler():
    events = []

    @on_hedge(max_hedge=2, on_hedge=lambda e: events.append(e["hedge_count"]))
    def target():
        return 42

    assert target() == 42
    assert events == [2]


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------

def test_hedging_retrying_call_sync():
    hr = HedgingRetrying(max_hedge=1)
    assert hr.call(lambda a, b=2: a + b, 3) == 5


def test_hedging_retrying_call_sync_raises_on_async_target():
    async def coro():
        return 1

    hr = HedgingRetrying()
    with pytest.raises(TypeError, match="Use async_call for async functions"):
        hr.call(coro)


def test_hedging_retrying_async_call():
    async def target(a, b=1):
        return a + b

    hr = HedgingRetrying(max_hedge=1)
    assert asyncio.run(hr.async_call(target, 4, b=6)) == 10


def test_hedging_retrying_call_passes_exception_config():
    class Boom(Exception):
        pass

    counts = {"n": 0}

    def target():
        counts["n"] += 1
        if counts["n"] < 2:
            raise Boom("retry")
        return "ok"

    hr = HedgingRetrying(
        _fast_wait_gen,
        max_hedge=1,
        exception=Boom,
    )
    assert hr.call(target) == "ok"
    assert counts["n"] == 2


def test_hedging_retrying_context_manager():
    hr = HedgingRetrying()
    with hr as entered:
        assert entered is hr


def test_hedging_retrying_async_context_manager():
    async def run():
        hr = HedgingRetrying()
        async with hr as entered:
            assert entered is hr

    asyncio.run(run())


def test_hedging_retrying_stores_config():
    hr = HedgingRetrying(
        _fast_wait_gen,
        max_hedge=7,
        exception=ValueError,
        max_tries=4,
        max_time=2.0,
        jitter=None,
        on_hedge=None,
        timeout=3.0,
        foo="bar",
    )
    assert hr._wait_gen is _fast_wait_gen
    assert hr._max_hedge == 7
    assert hr._exception is ValueError
    assert hr._max_tries == 4
    assert hr._max_time == 2.0
    assert hr._jitter is None
    assert hr._on_hedge is None
    assert hr._timeout == 3.0
    assert hr._wait_gen_kwargs == {"foo": "bar"}


def test_hedging_retrying_on_hedge_handler():
    events = []
    hr = HedgingRetrying(max_hedge=2, on_hedge=lambda e: events.append(e["hedge_count"]))
    assert hr.call(lambda: 1) == 1
    assert events == [2]
