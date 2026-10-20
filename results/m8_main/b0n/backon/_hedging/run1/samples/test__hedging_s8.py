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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_wait_gen(monkeypatch):
    # make sure expo sleeps are not the source of slowness; reduce expo base
    yield


def _zero_wait_gen(**kwargs):
    # a wait generator that yields zeros forever-ish
    while True:
        yield 0


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------


def test_hedge_error_is_exception_subclass():
    assert issubclass(HedgeError, Exception)


# ---------------------------------------------------------------------------
# hedge: dispatch
# ---------------------------------------------------------------------------


def test_hedge_sync_returns_value():
    calls = []

    def target(x, y):
        calls.append((x, y))
        return x + y

    result = hedge(
        target,
        wait_gen=_zero_wait_gen,
        args=(2, 3),
        max_hedge=1,
    )
    assert result == 5
    assert calls == [(2, 3)]


def test_hedge_sync_kw_passed():
    def target(*, a, b):
        return a * b

    result = hedge(
        target,
        wait_gen=_zero_wait_gen,
        kw={"a": 4, "b": 5},
        max_hedge=1,
    )
    assert result == 20


@pytest.mark.asyncio
async def test_hedge_async_returns_value():
    calls = []

    async def target(x):
        calls.append(x)
        return x * 2

    result = await hedge(
        target,
        wait_gen=_zero_wait_gen,
        args=(7,),
        max_hedge=1,
    )
    assert result == 14
    assert calls == [7]


def test_hedge_sync_propagates_first_success():
    # first attempt fails, retries should eventually return value
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("boom")
        return "ok"

    result = hedge(
        target,
        wait_gen=_zero_wait_gen,
        max_tries=5,
        max_hedge=1,
    )
    assert result == "ok"
    assert attempts["n"] == 3


def test_hedge_sync_raises_after_giveup():
    def target():
        raise ValueError("always")

    with pytest.raises(ValueError, match="always"):
        hedge(
            target,
            wait_gen=_zero_wait_gen,
            max_tries=2,
            max_hedge=1,
        )


def test_hedge_sync_raises_value_error_with_exception_filter():
    def target():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        hedge(
            target,
            wait_gen=_zero_wait_gen,
            max_tries=1,
            max_hedge=1,
            exception=RuntimeError,
        )


# ---------------------------------------------------------------------------
# hedge: on_hedge handler(s)
# ---------------------------------------------------------------------------


def test_hedge_sync_on_hedge_single_handler_called():
    events = []

    def handler(info):
        events.append(info)

    def target():
        return 1

    hedge(
        target,
        wait_gen=_zero_wait_gen,
        max_hedge=2,
        on_hedge=handler,
    )
    assert len(events) == 1
    assert events[0]["max_hedge"] == 2
    assert events[0]["target"] is target
    assert events[0]["hedge_count"] == 2


def test_hedge_sync_on_hedge_multiple_handlers():
    events_a = []
    events_b = []

    def handler_a(info):
        events_a.append(info)

    def handler_b(info):
        events_b.append(info)

    def target():
        return 1

    hedge(
        target,
        wait_gen=_zero_wait_gen,
        max_hedge=1,
        on_hedge=[handler_a, handler_b],
    )
    assert len(events_a) == 1
    assert len(events_b) == 1
    assert events_a[0]["hedge_count"] == 1


@pytest.mark.asyncio
async def test_hedge_async_on_hedge_handler_called():
    events = []

    def handler(info):
        events.append(info)

    async def target():
        return 1

    result = await hedge(
        target,
        wait_gen=_zero_wait_gen,
        max_hedge=3,
        on_hedge=handler,
    )
    assert result == 1
    assert len(events) == 1
    assert events[0]["max_hedge"] == 3
    assert events[0]["hedge_count"] == 3


# ---------------------------------------------------------------------------
# hedge: timeout
# ---------------------------------------------------------------------------


def test_hedge_sync_timeout_cancels_and_returns_first():
    # With a timeout and a target that is slow on first call and fast on second
    # hedge count > 1 means multiple threads run concurrently; but FIRST_COMPLETED
    # with timeout triggers. We want to test the timeout path raising.
    def target():
        time.sleep(1.0)
        return "done"

    with pytest.raises(HedgeError):
        hedge(
            target,
            wait_gen=_zero_wait_gen,
            max_hedge=1,
            timeout=0.01,
        )


@pytest.mark.asyncio
async def test_hedge_async_timeout_raises_hedge_error():
    async def target():
        await asyncio.sleep(1.0)
        return "done"

    with pytest.raises(HedgeError):
        await hedge(
            target,
            wait_gen=_zero_wait_gen,
            max_hedge=1,
            timeout=0.01,
        )


# ---------------------------------------------------------------------------
# hedge: async error aggregation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hedge_async_raises_first_exception():
    async def target():
        raise ValueError("async boom")

    with pytest.raises(ValueError, match="async boom"):
        await hedge(
            target,
            wait_gen=_zero_wait_gen,
            max_tries=1,
            max_hedge=1,
        )


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------


def test_on_hedge_decorator_sync():
    calls = {"n": 0}

    @on_hedge(wait_gen=_zero_wait_gen, max_hedge=1)
    def target(x, y):
        calls["n"] += 1
        return x * y

    assert target(3, 4) == 12
    assert calls["n"] == 1
    assert target.__name__ == "target"


def test_on_hedge_decorator_sync_handles_exception():
    @on_hedge(wait_gen=_zero_wait_gen, max_hedge=1, max_tries=1)
    def target():
        raise KeyError("k")

    with pytest.raises(KeyError, match="k"):
        target()


def test_on_hedge_decorator_sync_on_hedge_handler():
    events = []

    def handler(info):
        events.append(info)

    @on_hedge(wait_gen=_zero_wait_gen, max_hedge=2, on_hedge=handler)
    def target():
        return 1

    assert target() == 1
    assert len(events) == 1
    assert events[0]["hedge_count"] == 2


@pytest.mark.asyncio
async def test_on_hedge_decorator_async():
    calls = {"n": 0}

    @on_hedge(wait_gen=_zero_wait_gen, max_hedge=1)
    async def target(a):
        calls["n"] += 1
        return a + 1

    result = await target(9)
    assert result == 10
    assert calls["n"] == 1
    assert target.__name__ == "target"


@pytest.mark.asyncio
async def test_on_hedge_decorator_async_raises():
    @on_hedge(wait_gen=_zero_wait_gen, max_hedge=1, max_tries=1)
    async def target():
        raise RuntimeError("async decorator")

    with pytest.raises(RuntimeError, match="async decorator"):
        await target()


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------


def test_hedging_retrying_sync_context_manager():
    with HedgingRetrying(wait_gen=_zero_wait_gen) as retrying:
        assert isinstance(retrying, HedgingRetrying)
        result = retrying.call(lambda a, b: a + b, 2, 3)
        assert result == 5


@pytest.mark.asyncio
async def test_hedging_retrying_async_context_manager():
    async with HedgingRetrying(wait_gen=_zero_wait_gen) as retrying:
        assert isinstance(retrying, HedgingRetrying)

        async def target(x):
            return x * 2

        result = await retrying.async_call(target, 5)
        assert result == 10


def test_hedging_retrying_call_sync_rejects_coroutine():
    async def target():
        return 1

    retrying = HedgingRetrying(wait_gen=_zero_wait_gen)
    with pytest.raises(TypeError, match="Use async_call"):
        retrying.call(target)


def test_hedging_retrying_call_sync_success_kwargs():
    retrying = HedgingRetrying(wait_gen=_zero_wait_gen, max_hedge=1)

    def target(*, a, b):
        return a - b

    assert retrying.call(target, a=10, b=3) == 7


def test_hedging_retrying_call_sync_exception():
    retrying = HedgingRetrying(wait_gen=_zero_wait_gen, max_hedge=1, max_tries=1)

    def target():
        raise ValueError("sync call")

    with pytest.raises(ValueError, match="sync call"):
        retrying.call(target)


def test_hedging_retrying_call_sync_on_hedge_handler():
    events = []

    def handler(info):
        events.append(info)

    retrying = HedgingRetrying(
        wait_gen=_zero_wait_gen, max_hedge=2, on_hedge=handler
    )

    def target():
        return 1

    assert retrying.call(target) == 1
    assert len(events) == 1
    assert events[0]["hedge_count"] == 2


@pytest.mark.asyncio
async def test_hedging_retrying_async_call_success():
    retrying = HedgingRetrying(wait_gen=_zero_wait_gen, max_hedge=1)

    async def target(a, b):
        return a * b

    result = await retrying.async_call(target, 3, 4)
    assert result == 12


@pytest.mark.asyncio
async def test_hedging_retrying_async_call_exception():
    retrying = HedgingRetrying(
        wait_gen=_zero_wait_gen, max_hedge=1, max_tries=1
    )

    async def target():
        raise ValueError("async call")

    with pytest.raises(ValueError, match="async call"):
        await retrying.async_call(target)


# ---------------------------------------------------------------------------
# retry across multiple tries / hedging counts
# ---------------------------------------------------------------------------


def test_hedge_sync_retry_until_success_with_jitter_none():
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("tmp")
        return "value"

    result = hedge(
        target,
        wait_gen=_zero_wait_gen,
        jitter=None,
        max_tries=3,
        max_hedge=1,
    )
    assert result == "value"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_hedge_async_retry_until_success():
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("tmp")
        return "value"

    result = await hedge(
        target,
        wait_gen=_zero_wait_gen,
        jitter=None,
        max_tries=3,
        max_hedge=1,
    )
    assert result == "value"
    assert attempts["n"] == 2


def test_hedge_sync_predicate_returns_value_path():
    # predicate is applied to the successful value; with operator.not_ a truthy
    # return value stops. If target returns falsy, retries continue until giveup.
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            return 0  # falsy -> not_ == True -> retry
        return 5

    result = hedge(
        target,
        wait_gen=_zero_wait_gen,
        max_tries=3,
        max_hedge=1,
        jitter=None,
    )
    assert result == 5
    assert attempts["n"] == 2
