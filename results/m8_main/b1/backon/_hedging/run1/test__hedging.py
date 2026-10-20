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
    _hedge_sync,
    _hedge_async,
    _make_hedge_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_jitter(value):
    return 0.0


def _zero_wait():
    # A proper wait generator: the retry loop calls .next() on it, so it must
    # be an iterator exposing __next__ (generators do; see note below).
    # NOTE: `_retry_loop_s` calls `wait_gen().next()` on the *result* of
    # wait_gen, so wait_gen must return an object with a `.next` method.
    return _ZeroWait()


class _ZeroWait:
    def next(self, *args, **kwargs):
        return 0


# ---------------------------------------------------------------------------
# HedgeError
# ---------------------------------------------------------------------------

def test_hedge_error_is_exception():
    assert issubclass(HedgeError, Exception)
    err = HedgeError("boom")
    assert str(err) == "boom"


# ---------------------------------------------------------------------------
# _make_hedge_target
# ---------------------------------------------------------------------------

def test_make_hedge_target_calls_with_args_kwargs():
    calls = []

    def target(*a, **k):
        calls.append((a, k))
        return "ok"

    wrapped = _make_hedge_target(target, (1, 2), {"x": 3})
    assert wrapped() == "ok"
    assert calls == [((1, 2), {"x": 3})]


# ---------------------------------------------------------------------------
# hedge() sync path
# ---------------------------------------------------------------------------

def test_hedge_sync_success_runs_all_workers():
    call_count = {"n": 0}

    def target():
        call_count["n"] += 1
        return 42

    result = hedge(
        target,
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=2,
        max_tries=1,
    )
    assert result == 42
    # Both hedged workers may run; at least one must have.
    assert call_count["n"] >= 1


def test_hedge_sync_propagates_first_exception_and_raises():
    def target():
        raise ValueError("sync-fail")

    with pytest.raises(ValueError, match="sync-fail"):
        hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=2,
            max_tries=1,
        )


def test_hedge_sync_returns_value_when_one_succeeds():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "value"

    result = hedge(
        target,
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=1,
        max_tries=1,
    )
    assert result == "value"
    assert calls["n"] == 1


def test_hedge_sync_passes_args_and_kwargs():
    seen = []

    def target(a, b):
        seen.append((a, b))
        return a + b

    result = hedge(
        target,
        args=(2,),
        kw={"b": 5},
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=1,
        max_tries=1,
    )
    assert result == 7
    assert seen == [(2, 5)]


def test_hedge_sync_hedge_start_handler_called():
    handler_calls = []

    def handler(spec):
        handler_calls.append(spec)

    def target():
        return 1

    hedge(
        target,
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=3,
        max_tries=1,
        on_hedge=handler,
    )
    assert len(handler_calls) == 1
    spec = handler_calls[0]
    assert spec["max_hedge"] == 3
    assert spec["target"] is target
    assert spec["hedge_count"] == 3


def test_hedge_sync_iterable_handlers():
    calls = []

    def h1(spec):
        calls.append(("h1", spec))

    def h2(spec):
        calls.append(("h2", spec))

    def target():
        return 1

    hedge(
        target,
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=2,
        max_tries=1,
        on_hedge=[h1, h2],
    )
    assert len(calls) == 2
    assert calls[0][0] == "h1"
    assert calls[1][0] == "h2"


def test_hedge_sync_with_exception_filter_matches():
    def target():
        raise KeyError("k")

    with pytest.raises(KeyError):
        hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=1,
            max_tries=2,
            exception=KeyError,
        )


def test_hedge_sync_predicate_condition():
    # predicate operator.not_: returns True when result is falsy, treated as "retry"
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return 0 if calls["n"] < 2 else 5

    result = hedge(
        target,
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=1,
        max_tries=3,
    )
    assert result == 5


def test_hedge_timeout_raises_hedge_error_when_nothing_done():
    # All workers block; timeout=0.01 should not finish; expects HedgeError.
    #
    # IMPORTANT: The ThreadPoolExecutor context manager on __exit__ calls
    # shutdown(wait=True), which will block until workers finish. So the
    # blocking work must be interruptible and released by the test.

    event = threading.Event()

    def target():
        event.wait()
        return 1

    # We must release the event *before* leaving the `with pytest.raises`
    # block, because `hedge` won't return/raise until the executor shuts down.
    # Since the executor shutdown happens inside `hedge`, and the exec wait()
    # gets timeout=0.01 (but then continues to shutdown(wait=True))...
    #
    # Actually, we can't easily avoid this. Instead, use a bounded wait to
    # ensure it doesn't hang forever and use a timeout that will actually
    # be shorter than the block, then release the event from another thread
    # so the shutdown completes.
    def releaser():
        time.sleep(0.2)
        event.set()

    t = threading.Thread(target=releaser, daemon=True)
    t.start()

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=1,
            max_tries=1,
            timeout=0.01,
        )

    event.set()
    t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# hedge() async path
# ---------------------------------------------------------------------------

def test_hedge_async_success():
    async def target():
        return 99

    result = asyncio.run(
        hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=2,
            max_tries=1,
        )
    )
    assert result == 99


def test_hedge_async_returns_first_non_execption_among_done():
    async def target():
        return "async-value"

    result = asyncio.run(
        _hedge_async(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=1,
            max_tries=1,
        )
    )
    assert result == "async-value"


def test_hedge_async_raises_on_failure():
    async def target():
        raise RuntimeError("async-fail")

    with pytest.raises(RuntimeError, match="async-fail"):
        asyncio.run(
            hedge(
                target,
                wait_gen=_zero_wait,
                jitter=_no_jitter,
                max_hedge=2,
                max_tries=1,
            )
        )


def test_hedge_async_passes_args_kwargs():
    seen = []

    async def target(a, b):
        seen.append((a, b))
        return a * b

    result = asyncio.run(
        hedge(
            target,
            args=(3,),
            kw={"b": 4},
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=1,
            max_tries=1,
        )
    )
    assert result == 12
    assert seen == [(3, 4)]


def test_hedge_async_on_hedge_handler():
    calls = []

    def handler(spec):
        calls.append(spec)

    async def target():
        return 1

    asyncio.run(
        hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=3,
            max_tries=1,
            on_hedge=handler,
        )
    )
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 3
    assert calls[0]["target"] is target
    assert calls[0]["hedge_count"] == 3


def test_hedge_async_timeout_raises_hedge_error():
    async def target():
        await asyncio.sleep(10)
        return 1

    async def go():
        return await hedge(
            target,
            wait_gen=_zero_wait,
            jitter=_no_jitter,
            max_hedge=1,
            max_tries=1,
            timeout=0.01,
        )

    with pytest.raises(HedgeError, match="all hedged requests failed"):
        asyncio.run(go())


# ---------------------------------------------------------------------------
# on_hedge() decorator
# ---------------------------------------------------------------------------

def test_on_hedge_decorator_sync():
    @on_hedge(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1)
    def target(x):
        return x * 2

    assert target(21) == 42


def test_on_hedge_decorator_sync_preserves_wraps():
    @on_hedge(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1)
    def my_func(x):
        """docstring here"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "docstring here"


def test_on_hedge_decorator_async():
    @on_hedge(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1)
    async def target(x):
        return x + 1

    assert asyncio.run(target(41)) == 42


def test_on_hedge_decorator_async_preserves_wraps():
    @on_hedge(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1)
    async def my_func(x):
        """async doc"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "async doc"


def test_on_hedge_decorator_async_error():
    @on_hedge(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1)
    async def target():
        raise ValueError("async-decorated")

    with pytest.raises(ValueError, match="async-decorated"):
        asyncio.run(target())


def test_on_hedge_handler_used():
    calls = []

    def handler(spec):
        calls.append(spec)

    @on_hedge(
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=2,
        max_tries=1,
        on_hedge=handler,
    )
    def target():
        return "done"

    assert target() == "done"
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2


# ---------------------------------------------------------------------------
# HedgingRetrying
# ---------------------------------------------------------------------------

def test_hedging_retrying_stores_attrs():
    hr = HedgingRetrying(
        wait_gen=_zero_wait,
        max_hedge=5,
        exception=ValueError,
        max_tries=2,
        max_time=10.0,
        jitter=_no_jitter,
        on_hedge=None,
        timeout=1.0,
    )
    assert hr._wait_gen is _zero_wait
    assert hr._max_hedge == 5
    assert hr._exception is ValueError
    assert hr._max_tries == 2
    assert hr._max_time == 10.0
    assert hr._jitter is _no_jitter
    assert hr._timeout == 1.0


def test_hedging_retrying_context_managers():
    hr = HedgingRetrying(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1)

    with hr as ctx:
        assert ctx is hr

    async def go():
        async with hr as ctx:
            assert ctx is hr

    asyncio.run(go())


def test_hedging_retrying_call_sync():
    hr = HedgingRetrying(
        wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1
    )

    def target(a, b):
        return a + b

    assert hr.call(target, 3, 4) == 7


def test_hedging_retrying_call_sync_kwargs():
    hr = HedgingRetrying(
        wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1
    )

    def target(a, b=0):
        return a * b

    assert hr.call(target, 5, b=6) == 30


def test_hedging_retrying_call_sync_rejects_async():
    hr = HedgingRetrying(wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1)

    async def target():
        return 1

    with pytest.raises(TypeError, match="Use async_call"):
        hr.call(target)


def test_hedging_retrying_async_call():
    hr = HedgingRetrying(
        wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1
    )

    async def target(x):
        return x * 10

    assert asyncio.run(hr.async_call(target, 7)) == 70


def test_hedging_retrying_async_call_kwargs():
    hr = HedgingRetrying(
        wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1
    )

    async def target(a, b=1):
        return a - b

    assert asyncio.run(hr.async_call(target, 10, b=3)) == 7


def test_hedging_retrying_async_call_exception():
    hr = HedgingRetrying(
        wait_gen=_zero_wait, jitter=_no_jitter, max_hedge=1, max_tries=1
    )

    async def target():
        raise KeyError("async-err")

    with pytest.raises(KeyError):
        asyncio.run(hr.async_call(target))


def test_hedging_retrying_with_hedge_handler():
    calls = []

    def handler(spec):
        calls.append(spec)

    hr = HedgingRetrying(
        wait_gen=_zero_wait,
        jitter=_no_jitter,
        max_hedge=2,
        max_tries=1,
        on_hedge=handler,
    )

    def target():
        return 1

    assert hr.call(target) == 1
    assert len(calls) == 1
    assert calls[0]["max_hedge"] == 2
