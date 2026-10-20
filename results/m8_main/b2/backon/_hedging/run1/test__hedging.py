from __future__ import annotations

import asyncio
import operator
from typing import Any
from unittest.mock import MagicMock

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    _hedge_async,
    _hedge_sync,
    _make_hedge_target,
    hedge,
    on_hedge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_wait(*a: Any, **k: Any) -> float:
    return 0.0


async def _azero_wait(*a: Any, **k: Any) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# _make_hedge_target
# ---------------------------------------------------------------------------

def test_make_hedge_target_binds_args_and_kwargs():
    calls = []

    def target(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    wrapped = _make_hedge_target(target, (1, 2), {"x": 3})
    assert wrapped() == "result"
    assert calls == [((1, 2), {"x": 3})]


# ---------------------------------------------------------------------------
# hedge() -> sync path
# ---------------------------------------------------------------------------

def test_hedge_sync_success():
    def target():
        return 42

    assert hedge(target, _zero_wait, max_hedge=1) == 42


def test_hedge_sync_passes_args_and_kw():
    captured = {}

    def target(a, b, *, c):
        captured["a"] = a
        captured["b"] = b
        captured["c"] = c
        return a + b + c

    result = hedge(target, _zero_wait, max_hedge=1, args=(1, 2), kw={"c": 3})
    assert result == 6
    assert captured == {"a": 1, "b": 2, "c": 3}


def test_hedge_sync_raises_first_exception():
    def target():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        hedge(target, _zero_wait, max_hedge=1)


def test_hedge_sync_exception_filtered_then_predicate():
    """Non-matching exception types are not retried; predicate decides."""
    def target():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        hedge(target, _zero_wait, max_hedge=1, exception=KeyError)


def test_hedge_sync_predicate_controls_retry():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        return attempts["n"]

    # predicate not_(value) -> retries while result is falsy. 1 is truthy,
    # so it should succeed on the first try.
    result = hedge(target, _zero_wait, max_hedge=1, predicate=operator.not_)
    assert result == 1


def test_hedge_sync_on_hedge_handler_called():
    calls = []

    def handler(state):
        calls.append(dict(state))

    def target():
        return "ok"

    hedge(target, _zero_wait, max_hedge=2, on_hedge=handler)
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2
    assert calls[0]["target"] is target
    assert calls[0]["hedge_count"] == 2


def test_hedge_sync_on_hedge_iterable_of_handlers():
    calls_a = []
    calls_b = []

    def target():
        return "ok"

    hedge(
        target,
        _zero_wait,
        max_hedge=1,
        on_hedge=[calls_a.append, calls_b.append],
    )
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert calls_a[0]["max_hedge"] == 1


def test_hedge_sync_returns_a_fast_success_result():
    # max_hedge > 1: the executor races; at least one succeeds so the
    # returned value is "ok".
    def target():
        return "ok"

    assert hedge(target, _zero_wait, max_hedge=3) == "ok"


def test_hedge_sync_timeout_zero_returns_result():
    # timeout=0.0 with a trivially fast function still completes; futures_wait
    # with FIRST_COMPLETED returns the done future since it finishes immediately.
    def target():
        return "ok"

    assert hedge(target, _zero_wait, max_hedge=1, timeout=0.0) == "ok"


# ---------------------------------------------------------------------------
# hedge() -> async path
# ---------------------------------------------------------------------------

def test_hedge_async_is_returned_for_coroutine():
    async def target():
        return 7

    coro = hedge(target, _azero_wait, max_hedge=1)
    assert asyncio.iscoroutine(coro)
    result = asyncio.get_event_loop().run_until_complete(coro)
    assert result == 7


def test_hedge_dispatch_returns_non_coroutine_for_sync():
    def target():
        return 1

    result = hedge(target, _zero_wait, max_hedge=1)
    assert result == 1


def test_hedge_async_success():
    async def target():
        return "async-ok"

    async def go():
        return await _hedge_async(target, _azero_wait, max_hedge=1)

    assert asyncio.get_event_loop().run_until_complete(go()) == "async-ok"


def test_hedge_async_raises_exception():
    async def target():
        raise RuntimeError("async boom")

    async def go():
        await _hedge_async(target, _azero_wait, max_hedge=1)

    with pytest.raises(RuntimeError, match="async boom"):
        asyncio.get_event_loop().run_until_complete(go())


def test_hedge_async_on_hedge_handler_called():
    calls = []

    def handler(state):
        calls.append(dict(state))

    async def target():
        return 1

    async def go():
        await _hedge_async(target, _azero_wait, max_hedge=2, on_hedge=handler)

    asyncio.get_event_loop().run_until_complete(go())
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2
    assert calls[0]["hedge_count"] == 2


def test_hedge_async_timeout_zero_returns_result():
    # Timeout 0 with an instantly-completing coroutine still yields a result.
    async def target():
        return 1

    async def go():
        return await _hedge_async(target, _azero_wait, max_hedge=1, timeout=0.0)

    assert asyncio.get_event_loop().run_until_complete(go()) == 1


def test_hedge_async_args_and_kw():
    captured = {}

    async def target(a, b, *, c):
        captured.update(a=a, b=b, c=c)
        return a + b + c

    async def go():
        return await _hedge_async(
            target, _azero_wait, max_hedge=1, args=(1, 2), kw={"c": 3}
        )

    assert asyncio.get_event_loop().run_until_complete(go()) == 6
    assert captured == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------

def test_on_hedge_decorator_sync():
    calls = []

    @on_hedge(_zero_wait, max_hedge=1, on_hedge=calls.append)
    def target(a, b):
        return a + b

    assert target(2, 3) == 5
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 1
    assert target.__name__ == "target"


def test_on_hedge_decorator_async():
    calls = []

    @on_hedge(_azero_wait, max_hedge=1, on_hedge=calls.append)
    async def target(a, b):
        return a + b

    result = asyncio.get_event_loop().run_until_complete(target(2, 3))
    assert result == 5
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 1
    assert target.__name__ == "target"


def test_on_hedge_decorator_sync_error():
    @on_hedge(_zero_wait, max_hedge=1)
    def target():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        target()


def test_on_hedge_decorator_async_error():
    @on_hedge(_azero_wait, max_hedge=1)
    async def target():
        raise ValueError("async nope")

    with pytest.raises(ValueError, match="async nope"):
        asyncio.get_event_loop().run_until_complete(target())


def test_on_hedge_preserves_wraps():
    @on_hedge(_zero_wait, max_hedge=1)
    def my_func():
        """docstring here"""

    assert my_func.__doc__ == "docstring here"
    assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------

def test_hedging_retrying_sync_call():
    retrying = HedgingRetrying(_zero_wait, max_hedge=1)
    assert retrying.call(lambda a, b: a + b, 2, 3) == 5


def test_hedging_retrying_sync_call_rejects_async():
    retrying = HedgingRetrying(_zero_wait, max_hedge=1)

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call"):
        retrying.call(target)


def test_hedging_retrying_sync_call_propagates_error():
    retrying = HedgingRetrying(_zero_wait, max_hedge=1)

    def target():
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        retrying.call(target)


def test_hedging_retrying_async_call():
    retrying = HedgingRetrying(_azero_wait, max_hedge=1)

    async def target(a, b):
        return a * b

    result = asyncio.get_event_loop().run_until_complete(
        retrying.async_call(target, 2, 3)
    )
    assert result == 6


def test_hedging_retrying_async_call_propagates_error():
    retrying = HedgingRetrying(_azero_wait, max_hedge=1)

    async def target():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.get_event_loop().run_until_complete(retrying.async_call(target))


def test_hedging_retrying_sync_context_manager():
    with HedgingRetrying(_zero_wait, max_hedge=1) as retrying:
        assert retrying.call(lambda: 1) == 1


def test_hedging_retrying_async_context_manager():
    async def go():
        async with HedgingRetrying(_azero_wait, max_hedge=1) as retrying:
            async def target():
                return 9
            return await retrying.async_call(target)

    assert asyncio.get_event_loop().run_until_complete(go()) == 9


def test_hedging_retrying_stores_configuration():
    handler = MagicMock()
    r = HedgingRetrying(
        _zero_wait,
        max_hedge=5,
        exception=ValueError,
        max_tries=2,
        max_time=1.0,
        jitter=None,
        on_hedge=handler,
        timeout=3.0,
        foo="bar",
    )
    assert r._max_hedge == 5
    assert r._exception is ValueError
    assert r._max_tries == 2
    assert r._max_time == 1.0
    assert r._jitter is None
    assert r._on_hedge is handler
    assert r._timeout == 3.0
    assert r._wait_gen_kwargs == {"foo": "bar"}


def test_hedging_retrying_calls_on_hedge_handler():
    # MagicMock defines __iter__, so it is treated as an iterable of handlers
    # (iterating yields nothing), hence it is never called directly.  Use a
    # plain function so it is recognized as a single handler.
    calls = []

    def handler(state):
        calls.append(dict(state))

    retrying = HedgingRetrying(_zero_wait, max_hedge=1, on_hedge=handler)
    assert retrying.call(lambda: "ok") == "ok"
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 1
    assert calls[0]["hedge_count"] == 1


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------

def test_hedge_error_is_exception():
    assert issubclass(HedgeError, Exception)
    err = HedgeError("msg")
    assert str(err) == "msg"
