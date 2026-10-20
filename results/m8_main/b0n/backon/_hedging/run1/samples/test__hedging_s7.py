from __future__ import annotations

import asyncio
import operator
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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

def _no_jitter(value: float) -> float:
    return 0.0


async def _no_sleep(_: float) -> None:
    return None


def _fast_wait_gen():
    # A wait generator that never actually blocks.
    yield 0.0
    while True:
        yield 0.0


async def _fast_wait_gen_async():
    yield 0.0
    while True:
        yield 0.0


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------

def test_hedge_error_is_exception():
    err = HedgeError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# hedge: sync
# ---------------------------------------------------------------------------

def test_hedge_sync_success_first_try():
    calls = []

    def target(x):
        calls.append(x)
        return x * 2

    result = hedge(
        target,
        _fast_wait_gen,
        args=(5,),
        max_hedge=2,
        jitter=_no_jitter,
    )
    assert result == 10
    # At least one call happened.
    assert len(calls) >= 1
    assert calls[0] == 5


def test_hedge_sync_with_kwargs():
    def target(a, b=None):
        return (a, b)

    result = hedge(
        target,
        _fast_wait_gen,
        args=(1,),
        kw={"b": 2},
        max_hedge=1,
        jitter=_no_jitter,
    )
    assert result == (1, 2)


def test_hedge_sync_raises_first_exception():
    def target():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        hedge(
            target,
            _fast_wait_gen,
            max_hedge=1,
            exception=(ValueError,),
            max_tries=1,
            jitter=_no_jitter,
        )


def test_hedge_sync_retries_until_success():
    counter = {"n": 0}
    lock = threading.Lock()

    def target():
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if n < 3:
            raise RuntimeError("fail")
        return "ok"

    result = hedge(
        target,
        _fast_wait_gen,
        max_hedge=1,
        exception=RuntimeError,
        max_tries=5,
        jitter=_no_jitter,
    )
    assert result == "ok"
    assert counter["n"] == 3


def test_hedge_sync_on_hedge_handler():
    events = []

    def handler(state):
        events.append(state)

    def target():
        return 1

    hedge(
        target,
        _fast_wait_gen,
        max_hedge=2,
        jitter=_no_jitter,
        on_hedge=handler,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev["max_hedge"] == 2
    assert ev["hedge_count"] == 2
    assert ev["target"] is target


def test_hedge_sync_on_hedge_multiple_handlers():
    events_a = []
    events_b = []

    def target():
        return 1

    hedge(
        target,
        _fast_wait_gen,
        max_hedge=1,
        jitter=_no_jitter,
        on_hedge=[lambda s: events_a.append(s), lambda s: events_b.append(s)],
    )
    assert len(events_a) == 1
    assert len(events_b) == 1


def test_hedge_sync_predicate_success():
    # default predicate operator.not_ => truthy return values are "done"
    def target():
        return 42

    result = hedge(target, _fast_wait_gen, max_hedge=1, jitter=_no_jitter)
    assert result == 42


def test_hedge_sync_predicate_retries():
    counter = {"n": 0}

    def target():
        counter["n"] += 1
        # return predicate: predicate receives value; predicate operator.not_
        # means retry while value is truthy (not_ value => False => retry).
        return counter["n"] < 3

    result = hedge(
        target,
        _fast_wait_gen,
        max_hedge=1,
        max_tries=5,
        jitter=_no_jitter,
        predicate=lambda v: v is True,  # retry while result is True
    )
    # target returns True for n=1,2 then False for n=3 -> success
    assert result is False
    assert counter["n"] == 3


def test_hedge_sync_timeout_leads_to_hedge_error():
    def target():
        time.sleep(0.2)
        return "late"

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        hedge(
            target,
            _fast_wait_gen,
            max_hedge=1,
            jitter=_no_jitter,
            timeout=0.01,
        )


def test_hedge_sync_passes_through_exception_from_done():
    def target():
        raise KeyError("k")

    with pytest.raises(KeyError):
        hedge(
            target,
            _fast_wait_gen,
            max_hedge=1,
            exception=KeyError,
            max_tries=1,
            jitter=_no_jitter,
        )


# ---------------------------------------------------------------------------
# hedge: async
# ---------------------------------------------------------------------------

def test_hedge_dispatches_async_coroutine():
    calls = []

    async def target(x):
        calls.append(x)
        return x + 1

    result = asyncio.run(
        hedge(
            target,
            _fast_wait_gen_async,
            args=(1,),
            max_hedge=1,
            jitter=_no_jitter,
        )
    )
    assert result == 2
    assert calls[0] == 1


def test_hedge_async_on_hedge():
    events = []

    async def target():
        return "ok"

    def handler(state):
        events.append(state)

    result = asyncio.run(
        hedge(
            target,
            _fast_wait_gen_async,
            max_hedge=2,
            jitter=_no_jitter,
            on_hedge=handler,
        )
    )
    assert result == "ok"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["hedge_count"] == 2


def test_hedge_async_raises_first_exception():
    async def target():
        raise ValueError("async bad")

    with pytest.raises(ValueError, match="async bad"):
        asyncio.run(
            hedge(
                target,
                _fast_wait_gen_async,
                max_hedge=1,
                exception=ValueError,
                max_tries=1,
                jitter=_no_jitter,
            )
        )


def test_hedge_async_retries_until_success():
    counter = {"n": 0}

    async def target():
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("nope")
        return "done"

    result = asyncio.run(
        hedge(
            target,
            _fast_wait_gen_async,
            max_hedge=1,
            exception=RuntimeError,
            max_tries=5,
            jitter=_no_jitter,
        )
    )
    assert result == "done"
    assert counter["n"] == 3


def test_hedge_async_timeout_hedge_error():
    async def target():
        await asyncio.sleep(0.2)
        return "late"

    with pytest.raises(HedgeError):
        asyncio.run(
            hedge(
                target,
                _fast_wait_gen_async,
                max_hedge=1,
                jitter=_no_jitter,
                timeout=0.01,
            )
        )


def test_hedge_async_with_kwargs():
    async def target(a, b=None):
        return (a, b)

    result = asyncio.run(
        hedge(
            target,
            _fast_wait_gen_async,
            args=(1,),
            kw={"b": 2},
            max_hedge=1,
            jitter=_no_jitter,
        )
    )
    assert result == (1, 2)


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------

def test_on_hedge_decorator_sync():
    calls = []

    @on_hedge(_fast_wait_gen, max_hedge=2, jitter=_no_jitter)
    def target(x):
        calls.append(x)
        return x * 3

    result = target(3)
    assert result == 9
    assert calls[0] == 3


def test_on_hedge_decorator_sync_preserves_metadata():
    @on_hedge(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)
    def target():
        """docstring"""
        return 1

    assert target.__name__ == "target"
    assert target.__doc__ == "docstring"


def test_on_hedge_decorator_async():
    calls = []

    @on_hedge(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter)
    async def target(x):
        calls.append(x)
        return x * 4

    result = asyncio.run(target(2))
    assert result == 8
    assert calls[0] == 2


def test_on_hedge_decorator_async_preserves_metadata():
    @on_hedge(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter)
    async def target():
        """async doc"""
        return 1

    assert target.__name__ == "target"
    assert target.__doc__ == "async doc"


def test_on_hedge_decorator_forwards_kwargs_and_args():
    @on_hedge(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)
    def target(a, b):
        return a + b

    assert target(1, b=2) == 3
    assert target(a=1, b=2) == 3


def test_on_hedge_decorator_surfaces_errors():
    @on_hedge(_fast_wait_gen, max_hedge=1, jitter=_no_jitter, max_tries=1,
              exception=ValueError)
    def target():
        raise ValueError("deco")

    with pytest.raises(ValueError, match="deco"):
        target()


def test_on_hedge_decorator_async_surfaces_errors():
    @on_hedge(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter,
              max_tries=1, exception=ValueError)
    async def target():
        raise ValueError("async deco")

    with pytest.raises(ValueError, match="async deco"):
        asyncio.run(target())


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------

def test_hedging_retrying_call_sync():
    hr = HedgingRetrying(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)
    assert hr.call(lambda x: x + 1, 1) == 2


def test_hedging_retrying_call_sync_kwargs():
    hr = HedgingRetrying(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)

    def target(a, b=None):
        return (a, b)

    assert hr.call(target, 1, b=2) == (1, 2)


def test_hedging_retrying_call_async_target_raises():
    hr = HedgingRetrying(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call for async functions"):
        hr.call(target)


def test_hedging_retrying_async_call():
    hr = HedgingRetrying(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter)

    async def target(x):
        return x * 2

    assert asyncio.run(hr.async_call(target, 3)) == 6


def test_hedging_retrying_async_call_retries():
    counter = {"n": 0}

    async def target():
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("again")
        return "ok"

    hr = HedgingRetrying(
        _fast_wait_gen_async,
        max_hedge=1,
        exception=RuntimeError,
        max_tries=5,
        jitter=_no_jitter,
    )
    assert asyncio.run(hr.async_call(target)) == "ok"
    assert counter["n"] == 3


def test_hedging_retrying_sync_retries():
    counter = {"n": 0}

    def target():
        counter["n"] += 1
        if counter["n"] < 2:
            raise RuntimeError("once")
        return "value"

    hr = HedgingRetrying(
        _fast_wait_gen,
        max_hedge=1,
        exception=RuntimeError,
        max_tries=3,
        jitter=_no_jitter,
    )
    assert hr.call(target) == "value"
    assert counter["n"] == 2


def test_hedging_retrying_on_hedge():
    events = []

    hr = HedgingRetrying(
        _fast_wait_gen,
        max_hedge=2,
        jitter=_no_jitter,
        on_hedge=lambda s: events.append(s),
    )
    assert hr.call(lambda: "x") == "x"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2


def test_hedging_retrying_sync_context_manager():
    hr = HedgingRetrying(_fast_wait_gen, max_hedge=1, jitter=_no_jitter)
    with hr as entered:
        assert entered is hr


def test_hedging_retrying_async_context_manager():
    hr = HedgingRetrying(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter)

    async def runner():
        async with hr as entered:
            assert entered is hr
            return await entered.async_call(_acoro_result)

    async def _acoro_result():
        return "ctx"

    assert asyncio.run(runner()) == "ctx"


def test_hedging_retrying_async_call_timeout():
    hr = HedgingRetrying(_fast_wait_gen_async, max_hedge=1, jitter=_no_jitter,
                         timeout=0.01)

    async def target():
        await asyncio.sleep(0.2)
        return "late"

    with pytest.raises(HedgeError):
        asyncio.run(hr.async_call(target))


def test_hedging_retrying_stores_config():
    def wg():
        yield 0.0
        while True:
            yield 0.0

    handler = lambda s: None  # noqa: E731
    hr = HedgingRetrying(
        wg,
        max_hedge=7,
        exception=ValueError,
        max_tries=4,
        max_time=5.0,
        jitter=_no_jitter,
        on_hedge=handler,
        timeout=2.0,
        foo="bar",
    )
    assert hr._max_hedge == 7
    assert hr._exception is ValueError
    assert hr._max_tries == 4
    assert hr._max_time == 5.0
    assert hr._jitter is _no_jitter
    assert hr._on_hedge is handler
    assert hr._timeout == 2.0
    assert hr._wait_gen_kwargs == {"foo": "bar"}


# ---------------------------------------------------------------------------
# _make_hedge_target indirectly via hedge using args/kw
# ---------------------------------------------------------------------------

def test_hedge_sync_passes_args_and_kwargs_to_target():
    received = {}

    def target(a, b=None):
        received["a"] = a
        received["b"] = b
        return "ok"

    hedge(target, _fast_wait_gen, args=("A",), kw={"b": "B"},
          max_hedge=1, jitter=_no_jitter)
    assert received == {"a": "A", "b": "B"}
