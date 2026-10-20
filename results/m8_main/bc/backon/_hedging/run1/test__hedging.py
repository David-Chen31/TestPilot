from __future__ import annotations

import asyncio
import operator

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    _hedge_async,
    _hedge_sync,
    hedge,
    on_hedge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant(value):
    return value


async def _async_constant(value):
    return value


def _failing(exc_type=ValueError, msg="boom"):
    raise exc_type(msg)


async def _async_failing(exc_type=ValueError, msg="boom"):
    raise exc_type(msg)


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls


class AsyncCounter:
    def __init__(self):
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        return self.calls


def _collect_handler(sink):
    def handler(state):
        sink.append(dict(state))

    return handler


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------

def test_hedge_error_is_exception():
    assert issubclass(HedgeError, Exception)
    with pytest.raises(HedgeError):
        raise HedgeError("x")


# ---------------------------------------------------------------------------
# hedge (sync)
# ---------------------------------------------------------------------------

def test_hedge_sync_success():
    assert hedge(_constant, args=(42,)) == 42


def test_hedge_sync_with_kw():
    def target(a, b):
        return a + b

    assert hedge(target, args=(1,), kw={"b": 2}) == 3


def test_hedge_sync_returns_value():
    c = Counter()
    # first call succeeds
    assert hedge(c, max_hedge=1) == 1
    assert c.calls == 1


def test_hedge_sync_all_fail_raises_original_exception():
    def target():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        hedge(target, max_hedge=1)


def test_hedge_sync_exception_filtered_success():
    # exception only for KeyError; ValueError not retried -> but it's still an
    # exception raised by the target, so it propagates.
    def target():
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        hedge(target, max_hedge=1, exception=KeyError)


def test_hedge_sync_predicate_false_success():
    # predicate not_ applied to return value; returning truthy -> predicate False
    # means retry condition is False, so a "failure" is never retried and
    # the value is returned straight away.
    assert hedge(_constant, args=(5,), max_hedge=1, predicate=operator.not_) == 5


def test_hedge_sync_on_hedge_called():
    sink = []
    hedge(_constant, args=(1,), max_hedge=2, on_hedge=_collect_handler(sink))
    assert len(sink) == 1
    assert sink[0]["max_hedge"] == 2
    assert sink[0]["hedge_count"] == 2
    assert sink[0]["target"] is _constant


def test_hedge_sync_multiple_on_hedge_handlers():
    sink_a = []
    sink_b = []
    handlers = [_collect_handler(sink_a), _collect_handler(sink_b)]
    hedge(_constant, args=(0,), max_hedge=1, on_hedge=handlers)
    assert len(sink_a) == 1
    assert len(sink_b) == 1


def test_hedge_sync_timeout_raises_hedge_error():
    import threading

    event = threading.Event()

    def slow():
        event.wait(2)
        return 1

    try:
        with pytest.raises(HedgeError, match="all hedged requests failed"):
            hedge(slow, max_hedge=1, timeout=0.05)
    finally:
        event.set()


# ---------------------------------------------------------------------------
# hedge (async)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hedge_async_success():
    result = await hedge(_async_constant, args=(7,))
    assert result == 7


@pytest.mark.asyncio
async def test_hedge_async_kw():
    async def target(a, b):
        return a - b

    assert await hedge(target, args=(5,), kw={"b": 3}) == 2


@pytest.mark.asyncio
async def test_hedge_async_all_fail_raises_original():
    async def target():
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await hedge(target, max_hedge=1)


@pytest.mark.asyncio
async def test_hedge_async_on_hedge_called():
    sink = []
    await hedge(_async_constant, args=(1,), max_hedge=3, on_hedge=_collect_handler(sink))
    assert len(sink) == 1
    assert sink[0]["max_hedge"] == 3
    assert sink[0]["hedge_count"] == 3
    assert sink[0]["target"] is _async_constant


@pytest.mark.asyncio
async def test_hedge_async_multiple_handlers():
    a = []
    b = []
    await hedge(
        _async_constant, args=(0,), max_hedge=1,
        on_hedge=[_collect_handler(a), _collect_handler(b)],
    )
    assert len(a) == 1
    assert len(b) == 1


@pytest.mark.asyncio
async def test_hedge_async_timeout_raises_hedge_error():
    async def slow():
        await asyncio.sleep(5)
        return 1

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        await hedge(slow, max_hedge=1, timeout=0.05)


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------

def test_on_hedge_decorator_sync():
    @on_hedge(max_hedge=1)
    def target(x, y=0):
        return x + y

    assert target(3, y=4) == 7
    assert target.__name__ == "target"


def test_on_hedge_decorator_sync_raises():
    @on_hedge(max_hedge=1)
    def target():
        raise KeyError("k")

    with pytest.raises(KeyError, match="k"):
        target()


def test_on_hedge_decorator_on_hedge_handlers():
    sink = []

    @on_hedge(max_hedge=2, on_hedge=_collect_handler(sink))
    def target():
        return "ok"

    assert target() == "ok"
    assert len(sink) == 1
    assert sink[0]["max_hedge"] == 2
    assert sink[0]["hedge_count"] == 2


@pytest.mark.asyncio
async def test_on_hedge_decorator_async():
    @on_hedge(max_hedge=1)
    async def target(x):
        return x * 2

    assert await target(5) == 10
    assert target.__name__ == "target"


@pytest.mark.asyncio
async def test_on_hedge_decorator_async_raises():
    @on_hedge(max_hedge=1)
    async def target():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await target()


@pytest.mark.asyncio
async def test_on_hedge_decorator_async_on_hedge():
    sink = []

    @on_hedge(max_hedge=3, on_hedge=_collect_handler(sink))
    async def target():
        return 1

    assert await target() == 1
    assert len(sink) == 1
    assert sink[0]["hedge_count"] == 3


def test_on_hedge_decorator_preserves_metadata():
    @on_hedge()
    def documented(x):
        """some doc"""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "some doc"
    assert documented(3) == 3


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------

def test_hedging_retrying_sync_context_manager():
    hr = HedgingRetrying(max_hedge=1)
    with hr as entered:
        assert entered is hr


def test_hedging_retrying_sync_call():
    hr = HedgingRetrying(max_hedge=1)
    assert hr.call(_constant, 99) == 99


def test_hedging_retrying_sync_call_kwargs():
    hr = HedgingRetrying(max_hedge=1)

    def target(a, b):
        return a * b

    assert hr.call(target, 3, b=4) == 12


def test_hedging_retrying_sync_call_raises():
    hr = HedgingRetrying(max_hedge=1)

    def target():
        raise TypeError("t")

    with pytest.raises(TypeError, match="t"):
        hr.call(target)


def test_hedging_retrying_sync_call_rejects_async():
    hr = HedgingRetrying()

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call for async functions"):
        hr.call(target)


def test_hedging_retrying_on_hedge_sync():
    sink = []
    hr = HedgingRetrying(max_hedge=2, on_hedge=_collect_handler(sink))
    assert hr.call(_constant, 1) == 1
    assert len(sink) == 1
    assert sink[0]["hedge_count"] == 2


@pytest.mark.asyncio
async def test_hedging_retrying_async_context_manager():
    hr = HedgingRetrying(max_hedge=1)
    async with hr as entered:
        assert entered is hr


@pytest.mark.asyncio
async def test_hedging_retrying_async_call():
    hr = HedgingRetrying(max_hedge=1)
    assert await hr.async_call(_async_constant, 55) == 55


@pytest.mark.asyncio
async def test_hedging_retrying_async_call_raises():
    hr = HedgingRetrying(max_hedge=1)

    async def target():
        raise RuntimeError("r")

    with pytest.raises(RuntimeError, match="r"):
        await hr.async_call(target)


@pytest.mark.asyncio
async def test_hedging_retrying_async_on_hedge():
    sink = []
    hr = HedgingRetrying(max_hedge=2, on_hedge=_collect_handler(sink))
    assert await hr.async_call(_async_constant, 0) == 0
    assert len(sink) == 1
    assert sink[0]["hedge_count"] == 2


# ---------------------------------------------------------------------------
# Lower-level functions _hedge_sync / _hedge_async
# ---------------------------------------------------------------------------

def test_hedge_sync_inner_direct():
    assert _hedge_sync(_constant, args=(11,)) == 11


def test_hedge_sync_inner_raises():
    with pytest.raises(ValueError):
        _hedge_sync(lambda: _failing(ValueError, "v"), max_hedge=1)


@pytest.mark.asyncio
async def test_hedge_async_inner_direct():
    assert await _hedge_async(_async_constant, args=(12,)) == 12


@pytest.mark.asyncio
async def test_hedge_async_inner_raises():
    async def bad():
        raise ValueError("v")

    with pytest.raises(ValueError, match="v"):
        await _hedge_async(bad, max_hedge=1)


# ---------------------------------------------------------------------------
# max_hedge edge cases
# ---------------------------------------------------------------------------

def test_max_hedge_zero_raises_hedge_error():
    # With zero hedge workers, no futures are submitted and none complete.
    with pytest.raises(HedgeError, match="all hedged requests failed"):
        _hedge_sync(_constant, max_hedge=0, args=(1,))


@pytest.mark.asyncio
async def test_max_hedge_zero_async_raises_hedge_error():
    with pytest.raises(HedgeError, match="all hedged requests failed"):
        await _hedge_async(_async_constant, max_hedge=0, args=(1,))
