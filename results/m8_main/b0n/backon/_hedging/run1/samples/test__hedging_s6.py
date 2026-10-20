import asyncio
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
# HedgeError
# ---------------------------------------------------------------------------


def test_hedge_error_is_exception():
    err = HedgeError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _no_jitter(value):
    return 0.0


# ---------------------------------------------------------------------------
# hedge: sync
# ---------------------------------------------------------------------------


def test_hedge_sync_returns_value():
    result = hedge(lambda: 42, max_hedge=2)
    assert result == 42


def test_hedge_sync_passes_args_and_kw():
    def target(a, b, c=None):
        return (a, b, c)

    result = hedge(target, args=(1, 2), kw={"c": 3}, max_hedge=1)
    assert result == (1, 2, 3)


def test_hedge_sync_kw_none_defaults():
    def target():
        return "ok"

    assert hedge(target, max_hedge=1) == "ok"


def test_hedge_sync_raises_first_exception():
    def target():
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        hedge(target, max_hedge=2, exception=ValueError)


def test_hedge_sync_raises_first_exception_without_exception_arg():
    def target():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        hedge(target, max_hedge=1, predicate=lambda v: False)


def test_hedge_sync_exception_retries_then_succeeds():
    calls = {"n": 0}
    lock = threading.Lock()

    def target():
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n < 2:
            raise ValueError("transient")
        return "ok"

    result = hedge(
        target,
        max_hedge=1,
        exception=ValueError,
        max_tries=5,
        jitter=_no_jitter,
        base=0.0,
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_hedge_sync_on_hedge_single_handler():
    events = []

    def handler(payload):
        events.append(dict(payload))

    target = lambda: "done"

    result = hedge(target, max_hedge=3, on_hedge=handler, timeout=5.0)
    assert result == "done"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 3
    assert events[0]["hedge_count"] == 3
    assert events[0]["target"] is target


def test_hedge_sync_on_hedge_iterable_handlers():
    events = []
    handler_a = lambda payload: events.append(("a", payload["hedge_count"]))
    handler_b = lambda payload: events.append(("b", payload["hedge_count"]))

    result = hedge(lambda: 7, max_hedge=2, on_hedge=[handler_a, handler_b])
    assert result == 7
    assert [x[0] for x in events] == ["a", "b"]
    assert all(x[1] == 2 for x in events)


def test_hedge_sync_timeout_raises_hedge_error():
    def slow():
        time.sleep(1.0)
        return "never"

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        hedge(slow, max_hedge=1, timeout=0.01)


def test_hedge_sync_predicate_branch():
    # predicate path (exception=None) uses predicate on value
    def target():
        return 0

    with pytest.raises(ValueError):  # predicate raises -> propagated
        hedge(
            target,
            max_hedge=1,
            predicate=lambda v: (_ for _ in ()).throw(ValueError("pred")),
        )


# ---------------------------------------------------------------------------
# hedge: async
# ---------------------------------------------------------------------------


def test_hedge_async_returns_value():
    async def target():
        return "async"

    assert asyncio.run(hedge(target, max_hedge=2)) == "async"


def test_hedge_async_passes_args_and_kw():
    async def target(a, b, c=None):
        return (a, b, c)

    result = asyncio.run(
        hedge(target, args=(1, 2), kw={"c": 3}, max_hedge=1)
    )
    assert result == (1, 2, 3)


def test_hedge_async_raises_first_exception():
    async def target():
        raise ValueError("async-fail")

    with pytest.raises(ValueError, match="async-fail"):
        asyncio.run(hedge(target, max_hedge=2, exception=ValueError))


def test_hedge_async_on_hedge_handlers():
    events = []

    async def target():
        return "x"

    def handler(payload):
        events.append(dict(payload))

    result = asyncio.run(
        hedge(target, max_hedge=3, on_hedge=handler, timeout=5.0)
    )
    assert result == "x"
    assert len(events) == 1
    assert events[0]["max_hedge"] == 3
    assert events[0]["hedge_count"] == 3
    assert events[0]["target"] is target


def test_hedge_async_timeout_raises_hedge_error():
    async def slow():
        await asyncio.sleep(1.0)
        return "never"

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        asyncio.run(hedge(slow, max_hedge=1, timeout=0.01))


# ---------------------------------------------------------------------------
# on_hedge decorator: sync
# ---------------------------------------------------------------------------


def test_on_hedge_decorator_sync():
    @on_hedge(max_hedge=2)
    def target(x, y=1):
        return x + y

    assert target(5, y=2) == 7


def test_on_hedge_decorator_sync_preserves_metadata():
    @on_hedge(max_hedge=1)
    def target():
        """docstring"""
        return 1

    assert target.__name__ == "target"
    assert target.__doc__ == "docstring"


def test_on_hedge_decorator_sync_raises():
    @on_hedge(max_hedge=1, exception=ValueError)
    def target():
        raise ValueError("decofail")

    with pytest.raises(ValueError, match="decofail"):
        target()


def test_on_hedge_decorator_sync_on_hedge_called():
    events = []

    @on_hedge(max_hedge=2, on_hedge=lambda p: events.append(p["hedge_count"]))
    def target():
        return "ok"

    assert target() == "ok"
    assert events == [2]


# ---------------------------------------------------------------------------
# on_hedge decorator: async
# ---------------------------------------------------------------------------


def test_on_hedge_decorator_async():
    @on_hedge(max_hedge=2)
    async def target(x, y=1):
        return x + y

    assert asyncio.run(target(5, y=2)) == 7


def test_on_hedge_decorator_async_preserves_metadata():
    @on_hedge(max_hedge=1)
    async def target():
        """async-doc"""
        return 1

    assert target.__name__ == "target"
    assert target.__doc__ == "async-doc"


def test_on_hedge_decorator_async_raises():
    @on_hedge(max_hedge=1, exception=ValueError)
    async def target():
        raise ValueError("asyncdeco-fail")

    with pytest.raises(ValueError, match="asyncdeco-fail"):
        asyncio.run(target())


def test_on_hedge_decorator_async_on_hedge_called():
    events = []

    @on_hedge(max_hedge=2, on_hedge=lambda p: events.append(p["hedge_count"]))
    async def target():
        return "ok"

    assert asyncio.run(target()) == "ok"
    assert events == [2]


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------


def test_hedging_retrying_attrs():
    def tg():
        return 1

    instance = HedgingRetrying(
        max_hedge=5,
        exception=ValueError,
        max_tries=4,
        max_time=10.0,
        jitter=_no_jitter,
        on_hedge=None,
        timeout=1.5,
    )
    assert instance._max_hedge == 5
    assert instance._exception is ValueError
    assert instance._max_tries == 4
    assert instance._max_time == 10.0
    assert instance._jitter is _no_jitter
    assert instance._timeout == 1.5
    assert instance._on_hedge is None


def test_hedging_retrying_sync_context_manager():
    with HedgingRetrying(max_hedge=1) as hr:
        assert isinstance(hr, HedgingRetrying)
        assert hr.call(lambda: 3) == 3


def test_hedging_retrying_async_context_manager():
    async def run():
        async with HedgingRetrying(max_hedge=1) as hr:
            assert isinstance(hr, HedgingRetrying)
            return await hr.async_call(_async_return("value"))

    assert asyncio.run(run()) == "value"


async def _async_return(value):
    return value


def test_hedging_retrying_call_sync():
    hr = HedgingRetrying(max_hedge=2)

    def target(a, b):
        return a * b

    assert hr.call(target, 3, 4) == 12


def test_hedging_retrying_call_rejects_async():
    hr = HedgingRetrying(max_hedge=1)

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call for async functions"):
        hr.call(target)


def test_hedging_retrying_call_raises():
    hr = HedgingRetrying(max_hedge=1, exception=ValueError)

    def target():
        raise ValueError("callfail")

    with pytest.raises(ValueError, match="callfail"):
        hr.call(target)


def test_hedging_retrying_async_call():
    hr = HedgingRetrying(max_hedge=2)

    async def target(a, b):
        return a + b

    assert asyncio.run(hr.async_call(target, 5, 6)) == 11


def test_hedging_retrying_async_call_raises():
    hr = HedgingRetrying(max_hedge=1, exception=ValueError)

    async def target():
        raise ValueError("asynccall-fail")

    with pytest.raises(ValueError, match="asynccall-fail"):
        asyncio.run(hr.async_call(target))


def test_hedging_retrying_call_on_hedge_called():
    events = []

    hr = HedgingRetrying(
        max_hedge=3,
        on_hedge=lambda p: events.append(p["hedge_count"]),
    )

    assert hr.call(lambda: "ok") == "ok"
    assert events == [3]


def test_hedging_retrying_async_call_on_hedge_called():
    events = []

    hr = HedgingRetrying(
        max_hedge=2,
        on_hedge=lambda p: events.append(p["hedge_count"]),
    )

    async def target():
        return "ok"

    assert asyncio.run(hr.async_call(target)) == "ok"
    assert events == [2]
