import asyncio
import sys

import pytest

import backon._hedging as hedging_mod
from backon._hedging import (
    HedgeError,
    HedgingRetrying,
    hedge,
    on_hedge,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_wait_gen(monkeypatch):
    """Make the retry wait generator not incur real delays."""

    def _zero(_state):
        yield 0

    # Patch the wait generators used by the hedging module.
    monkeypatch.setattr(hedging_mod, "expo", _zero)
    return _zero


def _make_call_counter(succeed_on_call=1, result="ok", delay=0.0):
    """Return (callable, counter dict).

    Callable returns `result` on the `succeed_on_call`-th call and raises
    RuntimeError otherwise. The counter dict has key 'calls'.
    """
    state = {"calls": 0}

    def fn(*args, **kwargs):
        state["calls"] += 1
        n = state["calls"]
        if delay:
            import time

            time.sleep(delay)
        if n >= succeed_on_call:
            return (result, args, kwargs)
        raise RuntimeError(f"fail #{n}")

    return fn, state


async def _async_counter(succeed_on_call=1, result="ok", delay=0.0):
    state = {"calls": 0}

    async def fn(*args, **kwargs):
        state["calls"] += 1
        n = state["calls"]
        if delay:
            await asyncio.sleep(delay)
        if n >= succeed_on_call:
            return (result, args, kwargs)
        raise RuntimeError(f"fail #{n}")

    return fn, state


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------


def test_hedge_error_is_exception():
    assert issubclass(HedgeError, Exception)
    err = HedgeError("boom")
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# hedge (sync)
# ---------------------------------------------------------------------------


def test_hedge_successful_call_returns_result():
    def target(a, b, c=0):
        return a + b + c

    result = hedge(target, args=(1, 2), kw={"c": 3}, max_hedge=1)
    assert result == 6


def test_hedge_passes_args_and_kwargs():
    received = {}

    def target(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return "done"

    result = hedge(target, args=(1,), kw={"x": 2}, max_hedge=1)
    assert result == "done"
    assert received["args"] == (1,)
    assert received["kwargs"] == {"x": 2}


def test_hedge_constructs_thread_pool_with_max_hedge(monkeypatch):
    """Verify hedge submits exactly `max_hedge` jobs."""
    captured = {}

    from concurrent.futures import ThreadPoolExecutor as RealTPE

    class RecordingTPE(RealTPE):
        def __init__(self, *args, **kwargs):
            captured["max_workers"] = kwargs.get(
                "max_workers", args[0] if args else None
            )
            super().__init__(*args, **kwargs)

        def submit(self, fn, *args, **kwargs):
            captured.setdefault("submits", 0)
            captured["submits"] += 1
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(hedging_mod, "ThreadPoolExecutor", RecordingTPE)

    def target():
        return 1

    hedge(target, max_hedge=2)
    assert captured["max_workers"] == 2
    assert captured["submits"] == 2


def test_hedge_propagates_first_exception_when_all_fail():
    def target():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        hedge(target, max_hedge=1)


def test_hedge_on_hedge_single_handler_called():
    """Sync on_hedge handler is invoked once per hedge with a dict."""
    calls = []

    def handler(info):
        calls.append(info)

    def target():
        return "ok"

    hedge(target, max_hedge=2, on_hedge=handler)
    assert len(calls) == 1
    info = calls[0]
    assert info["max_hedge"] == 2
    assert info["target"] is target
    # hedge_count is `_i + 1` where `_i` is the last loop index -> == max_hedge
    assert info["hedge_count"] == 2


def test_hedge_on_hedge_iterable_of_handlers():
    calls_a = []
    calls_b = []

    def a(info):
        calls_a.append(info)

    def b(info):
        calls_b.append(info)

    def target():
        return "ok"

    hedge(target, max_hedge=1, on_hedge=[a, b])
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert calls_a[0]["max_hedge"] == 1
    assert calls_b[0]["hedge_count"] == 1


def test_hedge_async_is_awaited_by_hedge():
    """hedge detects coroutine functions and uses _hedge_async."""

    async def target():
        return "async-result"

    async def run():
        return await hedge(target, max_hedge=1)

    result = asyncio.run(run())
    assert result == "async-result"


# ---------------------------------------------------------------------------
# hedge (async)
# ---------------------------------------------------------------------------


def test_hedge_async_success():
    async def target(a, b):
        return a + b

    async def run():
        return await hedging_mod._hedge_async(
            target, args=(1, 2), kw={}, max_hedge=1
        )

    assert asyncio.run(run()) == 3


def test_hedge_async_all_fail_raises_first_exception():
    async def target():
        raise RuntimeError("async-boom")

    async def run():
        await hedging_mod._hedge_async(target, max_hedge=1)

    with pytest.raises(RuntimeError, match="async-boom"):
        asyncio.run(run())


def test_hedge_async_on_hedge_handler_called():
    calls = []

    def handler(info):
        calls.append(info)

    async def target():
        return "ok"

    async def run():
        return await hedging_mod._hedge_async(
            target, max_hedge=3, on_hedge=handler
        )

    assert asyncio.run(run()) == "ok"
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 3
    assert calls[0]["hedge_count"] == 3
    assert calls[0]["target"] is target


def test_hedge_async_on_hedge_iterable_handlers():
    calls = []

    def h(info):
        calls.append(info["hedge_count"])

    async def target():
        return "ok"

    async def run():
        return await hedging_mod._hedge_async(
            target, max_hedge=2, on_hedge=[h, h]
        )

    assert asyncio.run(run()) == "ok"
    assert calls == [2, 2]


# ---------------------------------------------------------------------------
# on_hedge decorator
# ---------------------------------------------------------------------------


def test_on_hedge_decorator_sync():
    calls = []

    def handler(info):
        calls.append(info)

    @on_hedge(max_hedge=2, on_hedge=handler)
    def target(x, y=0):
        return x + y

    assert target(3, y=4) == 7
    assert target.__name__ == "target"
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2


def test_on_hedge_decorator_async():
    calls = []

    def handler(info):
        calls.append(info)

    @on_hedge(max_hedge=1, on_hedge=handler)
    async def target(x):
        return x * 2

    async def run():
        return await target(5)

    assert asyncio.run(run()) == 10
    assert len(calls) == 1
    assert calls[0]["hedge_count"] == 1


def test_on_hedge_decorator_propagates_exception():
    @on_hedge(max_hedge=1)
    def target():
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        target()


def test_on_hedge_decorator_preserves_wraps():
    @on_hedge(max_hedge=1)
    def documented(a, b):
        """docstring."""
        return a - b

    assert documented(10, 3) == 7
    assert documented.__name__ == "documented"
    assert documented.__doc__ == "docstring."


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------


def test_hedging_retrying_call_sync():
    retrier = HedgingRetrying(max_hedge=1)

    def target(a, b=0):
        return a * b

    assert retrier.call(target, 3, b=4) == 12


def test_hedging_retrying_call_rejects_async():
    retrier = HedgingRetrying()

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call"):
        retrier.call(target)


def test_hedging_retrying_async_call():
    retrier = HedgingRetrying(max_hedge=1)

    async def target(a, b=0):
        return a + b

    async def run():
        return await retrier.async_call(target, 2, b=5)

    assert asyncio.run(run()) == 7


def test_hedging_retrying_stores_settings():
    handler = lambda info: None
    jitter = lambda v: 0.0

    def wg():
        while True:
            yield 0

    retrier = HedgingRetrying(
        wait_gen=wg,
        max_hedge=5,
        exception=ValueError,
        max_tries=3,
        max_time=1.0,
        jitter=jitter,
        on_hedge=handler,
        timeout=2.0,
        extra="kwarg",
    )
    assert retrier._wait_gen is wg
    assert retrier._max_hedge == 5
    assert retrier._exception is ValueError
    assert retrier._max_tries == 3
    assert retrier._max_time == 1.0
    assert retrier._jitter is jitter
    assert retrier._on_hedge is handler
    assert retrier._timeout == 2.0
    assert retrier._wait_gen_kwargs == {"extra": "kwarg"}


def test_hedging_retrying_context_managers():
    retrier = HedgingRetrying()

    with retrier as r1:
        assert r1 is retrier

    async def run():
        async with retrier as r2:
            assert r2 is retrier

    asyncio.run(run())


def test_hedging_retrying_call_invokes_on_hedge():
    seen = []

    retrier = HedgingRetrying(max_hedge=2, on_hedge=lambda info: seen.append(info))

    def target():
        return "res"

    assert retrier.call(target) == "res"
    assert len(seen) == 1
    assert seen[0]["max_hedge"] == 2


def test_hedging_retrying_async_call_invokes_on_hedge():
    seen = []

    retrier = HedgingRetrying(max_hedge=2, on_hedge=lambda info: seen.append(info))

    async def target():
        return "res"

    async def run():
        return await retrier.async_call(target)

    assert asyncio.run(run()) == "res"
    assert len(seen) == 1
    assert seen[0]["hedge_count"] == 2


def test_hedging_retrying_call_propagates_exception():
    retrier = HedgingRetrying(max_hedge=1)

    def target():
        raise OSError("disk")

    with pytest.raises(OSError, match="disk"):
        retrier.call(target)


# ---------------------------------------------------------------------------
# Misc internal helpers
# ---------------------------------------------------------------------------


def test_make_hedge_target_binds_args():
    def fn(a, b=None):
        return (a, b)

    bound = hedging_mod._make_hedge_target(fn, (1,), {"b": 2})
    assert bound() == (1, 2)


def test_hedge_none_kw_defaults_to_empty_dict():
    captured = {}

    def target(**kwargs):
        captured.update(kwargs)
        return "ok"

    # kw=None used internally; no kwargs passed
    assert hedge(target, kw=None, max_hedge=1) == "ok"
    assert captured == {}
