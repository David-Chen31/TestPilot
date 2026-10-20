import asyncio
import time

import pytest

from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    hedge,
    on_hedge,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _no_jitter(value):
    return 0.0


def _zero_wait():
    while True:
        yield 0


# --------------------------------------------------------------------------
# HedgeError
# --------------------------------------------------------------------------
def test_hedge_error_is_exception():
    assert issubclass(HedgeError, Exception)
    with pytest.raises(HedgeError):
        raise HedgeError("boom")


# --------------------------------------------------------------------------
# hedge: sync
# --------------------------------------------------------------------------
def test_hedge_sync_returns_value():
    calls = []

    def target(x):
        calls.append(x)
        return x * 2

    result = hedge(target, _zero_wait, args=(21,), max_hedge=1, jitter=_no_jitter)
    assert result == 42
    assert calls == [21]


def test_hedge_sync_passes_kwargs():
    def target(a, b, key=None):
        return (a, b, key)

    result = hedge(
        target,
        _zero_wait,
        args=(1, 2),
        kw={"key": "val"},
        max_hedge=1,
        jitter=_no_jitter,
    )
    assert result == (1, 2, "val")


def test_hedge_sync_kw_none_defaults_empty():
    def target():
        return "ok"

    assert hedge(target, _zero_wait, max_hedge=1, jitter=_no_jitter) == "ok"


def test_hedge_sync_retries_within_single_hedge():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("nope")
        return "success"

    result = hedge(
        target,
        _zero_wait,
        max_hedge=1,
        max_tries=5,
        exception=ValueError,
        jitter=_no_jitter,
    )
    assert result == "success"
    assert attempts["n"] == 3


def test_hedge_sync_raises_after_giveup():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        hedge(
            target,
            _zero_wait,
            max_hedge=1,
            max_tries=3,
            exception=RuntimeError,
            jitter=_no_jitter,
        )
    assert attempts["n"] == 3


def test_hedge_sync_on_hedge_called():
    events = []

    def target():
        return 1

    hedge(
        target,
        _zero_wait,
        max_hedge=2,
        jitter=_no_jitter,
        on_hedge=lambda evt: events.append(evt),
    )
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["target"] is target
    assert events[0]["hedge_count"] == 2


def test_hedge_sync_on_hedge_iterable():
    events = []

    def target():
        return 1

    def h1(evt):
        events.append(("h1", evt["hedge_count"]))

    def h2(evt):
        events.append(("h2", evt["hedge_count"]))

    hedge(target, _zero_wait, max_hedge=1, jitter=_no_jitter, on_hedge=[h1, h2])
    assert events == [("h1", 1), ("h2", 1)]


def test_hedge_sync_returns_first_success():
    def target():
        return "ok"

    result = hedge(target, _zero_wait, max_hedge=3, jitter=_no_jitter)
    assert result == "ok"


# --------------------------------------------------------------------------
# hedge: async
# --------------------------------------------------------------------------
def test_hedge_dispatch_to_async():
    async def target(x):
        return x + 1

    result = asyncio.run(
        hedge(target, _zero_wait, args=(1,), max_hedge=1, jitter=_no_jitter)
    )
    assert result == 2


def test_hedge_async_retries():
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("nope")
        return "done"

    result = asyncio.run(
        hedge(
            target,
            _zero_wait,
            max_hedge=1,
            max_tries=5,
            exception=ValueError,
            jitter=_no_jitter,
        )
    )
    assert result == "done"
    assert attempts["n"] == 3


def test_hedge_async_raises_after_giveup():
    async def target():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(
            hedge(
                target,
                _zero_wait,
                max_hedge=1,
                max_tries=2,
                exception=RuntimeError,
                jitter=_no_jitter,
            )
        )


def test_hedge_async_on_hedge_called():
    events = []

    async def target():
        return 5

    result = asyncio.run(
        hedge(
            target,
            _zero_wait,
            max_hedge=2,
            jitter=_no_jitter,
            on_hedge=lambda evt: events.append(evt),
        )
    )
    assert result == 5
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["hedge_count"] == 2


# --------------------------------------------------------------------------
# on_hedge decorator
# --------------------------------------------------------------------------
def test_on_hedge_decorator_sync():
    @on_hedge(_zero_wait, max_hedge=1, jitter=_no_jitter)
    def target(x):
        return x * 3

    assert target(4) == 12
    assert target.__name__ == "target"


def test_on_hedge_decorator_async():
    @on_hedge(_zero_wait, max_hedge=1, jitter=_no_jitter)
    async def target(x):
        return x + 5

    assert asyncio.run(target(3)) == 8
    assert target.__name__ == "target"


def test_on_hedge_decorator_exceptions():
    @on_hedge(
        _zero_wait,
        max_hedge=1,
        max_tries=2,
        exception=ValueError,
        jitter=_no_jitter,
    )
    def target():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        target()


def test_on_hedge_decorator_kwargs_passed():
    @on_hedge(_zero_wait, max_hedge=1, jitter=_no_jitter)
    def target(a, b, c=0):
        return a + b + c

    assert target(1, 2, c=3) == 6


# --------------------------------------------------------------------------
# HedgingRetrying
# --------------------------------------------------------------------------
def test_hedging_retrying_sync_context():
    attempts = {"n": 0}

    def target(x):
        attempts["n"] += 1
        return x + 1

    hedger = HedgingRetrying(_zero_wait, max_hedge=1, jitter=_no_jitter)
    with hedger as h:
        assert h is hedger
        result = h.call(target, 5)
    assert result == 6
    assert attempts["n"] == 1


def test_hedging_retrying_async_context():
    async def target(x):
        return x * 2

    hedger = HedgingRetrying(_zero_wait, max_hedge=1, jitter=_no_jitter)

    async def runner():
        async with hedger as h:
            assert h is hedger
            return await h.async_call(target, 7)

    assert asyncio.run(runner()) == 14


def test_hedging_retrying_call_rejects_async():
    async def target():
        return 1

    hedger = HedgingRetrying(_zero_wait, max_hedge=1, jitter=_no_jitter)

    with pytest.raises(TypeError):
        hedger.call(target)


def test_hedging_retrying_call_retries():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("nope")
        return "done"

    hedger = HedgingRetrying(
        _zero_wait,
        max_hedge=1,
        max_tries=3,
        exception=ValueError,
        jitter=_no_jitter,
    )
    assert hedger.call(target) == "done"
    assert attempts["n"] == 2


def test_hedging_retrying_async_call_retries():
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("nope")
        return "done"

    hedger = HedgingRetrying(
        _zero_wait,
        max_hedge=1,
        max_tries=3,
        exception=ValueError,
        jitter=_no_jitter,
    )
    assert asyncio.run(hedger.async_call(target)) == "done"
    assert attempts["n"] == 2


def test_hedging_retrying_stores_attributes():
    def wg():
        yield 0

    h = HedgingRetrying(
        wg,
        max_hedge=5,
        exception=ValueError,
        max_tries=4,
        max_time=10.0,
        jitter=_no_jitter,
        timeout=3.0,
        foo="bar",
    )
    assert h._wait_gen is wg
    assert h._max_hedge == 5
    assert h._exception is ValueError
    assert h._max_tries == 4
    assert h._max_time == 10.0
    assert h._timeout == 3.0
    assert h._wait_gen_kwargs == {"foo": "bar"}


def test_hedging_retrying_on_hedge_events():
    events = []

    def target():
        return 1

    h = HedgingRetrying(
        _zero_wait,
        max_hedge=2,
        jitter=_no_jitter,
        on_hedge=lambda e: events.append(e),
    )
    assert h.call(target) == 1
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["hedge_count"] == 2
