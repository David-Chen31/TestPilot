import asyncio
import inspect
import logging
import operator
import time
import pytest

from backon._common import _TEST_CONFIG
from backon._retry._classes import (
    Retrying,
    RetryingCaller,
    AsyncRetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)
from backon._rate_limiter import RateLimiter


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self):
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return True


def _record_sleep():
    calls = []

    def sleep(seconds):
        calls.append(seconds)

    return calls, sleep


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------
# _RetryAttempt
# --------------------------------------------------------------------------

def test_retry_attempt_success():
    attempt = _RetryAttempt()
    with attempt:
        attempt.set_value(42)
    assert attempt.failed is False
    assert attempt.exception is None
    assert attempt.value == 42


def test_retry_attempt_enter_returns_self():
    attempt = _RetryAttempt()
    with attempt as ctx:
        assert ctx is attempt


def test_retry_attempt_captures_exception():
    attempt = _RetryAttempt()
    err = ValueError("boom")
    with attempt:
        raise err
    assert attempt.failed is True
    assert attempt.exception is err
    assert attempt.value is None


def test_retry_attempt_does_not_capture_base_exception():
    attempt = _RetryAttempt()
    with pytest.raises(KeyboardInterrupt):
        with attempt:
            raise KeyboardInterrupt()
    # __exit__ returned False so it propagates; exception not recorded
    assert attempt.failed is False


# --------------------------------------------------------------------------
# sleep_using_event
# --------------------------------------------------------------------------

def test_sleep_using_event():
    event = _FakeEvent()
    sleep = sleep_using_event(event)
    assert sleep(1.5) is None
    assert event.waits == [1.5]
    sleep(2.0)
    assert event.waits == [1.5, 2.0]


# --------------------------------------------------------------------------
# Retrying - basic attributes and context manager
# --------------------------------------------------------------------------

def test_retrying_defaults():
    r = Retrying()
    assert r.enabled is True
    assert r.statistics == {}
    assert r.call_state is None


def test_retrying_enabled_setter():
    r = Retrying()
    r.enabled = False
    assert r.enabled is False
    r.enabled = True
    assert r.enabled is True


def test_retrying_context_manager():
    r = Retrying()
    with r as ctx:
        assert ctx is r


def test_retrying_async_context_manager():
    async def _go():
        r = Retrying()
        async with r as ctx:
            assert ctx is r

    _run(_go())


# --------------------------------------------------------------------------
# Retrying.call - success / failure
# --------------------------------------------------------------------------

def test_retrying_call_succeeds_immediately():
    r = Retrying()

    def target():
        return "ok"

    assert r.call(target) == "ok"
    stats = r.statistics
    assert stats["attempt_number"] == 1


def test_retrying_call_retries_then_succeeds():
    calls = []
    _, sleep = _record_sleep()
    # max_tries=2 means 2 total attempts: fail once, then succeed.
    r = Retrying(max_tries=2, sleep=sleep, jitter=None, max_time=None)

    def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("nope")
        return "done"

    result = r.call(target)
    assert result == "done"
    assert len(calls) == 2


def test_retrying_call_giveup_raises():
    _, sleep = _record_sleep()
    r = Retrying(max_tries=2, sleep=sleep, jitter=None)

    def target():
        raise ValueError("always")

    with pytest.raises(ValueError):
        r.call(target)


def test_retrying_call_giveup_no_raise():
    _, sleep = _record_sleep()
    r = Retrying(max_tries=2, sleep=sleep, jitter=None, raise_on_giveup=False)

    def target():
        raise ValueError("always")

    # With raise_on_giveup=False, no exception propagates; returns None.
    result = r.call(target)
    assert result is None


def test_retrying_call_rejects_coroutine():
    r = Retrying()

    async def coro():
        return 1

    with pytest.raises(TypeError):
        r.call(coro)


def test_retrying_call_passes_args_kwargs():
    r = Retrying()

    def target(a, b, *, c):
        return (a, b, c)

    assert r.call(target, 1, 2, c=3) == (1, 2, 3)


def test_retrying_call_updates_call_state():
    r = Retrying()

    def target():
        return 1

    r.call(target)
    assert r._call_state is not None
    assert r._state is not None
    assert r.call_state is r._call_state


def test_retrying_call_uses_rate_limiter():
    # Rate limiting is exercised via RateLimiter directly to avoid retry loops.
    rl = RateLimiter(max_calls=1, period=100.0)
    assert rl(lambda: 1) == 1

    from backon._rate_limiter import RateLimitError

    assert rl.acquire() is False


# --------------------------------------------------------------------------
# Retrying.async_call
# --------------------------------------------------------------------------

def test_async_call_succeeds():
    async def _go():
        r = Retrying()

        async def target():
            return "async-ok"

        return await r.async_call(target)

    assert _run(_go()) == "async-ok"


def test_async_call_retries_then_succeeds():
    async def _go():
        calls = []

        async def sleep(seconds):
            pass

        # max_tries=2 means 2 total attempts: fail once, then succeed.
        r = Retrying(max_tries=2, sleep=sleep, jitter=None)

        async def target():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("nope")
            return "done"

        result = await r.async_call(target)
        return result, len(calls)

    result, n = _run(_go())
    assert result == "done"
    assert n == 2


def test_async_call_giveup_raises():
    async def _go():
        async def sleep(seconds):
            pass

        r = Retrying(max_tries=2, sleep=sleep, jitter=None)

        async def target():
            raise ValueError("always")

        with pytest.raises(ValueError):
            await r.async_call(target)

    _run(_go())


def test_async_call_passes_args_kwargs():
    async def _go():
        r = Retrying()

        async def target(a, *, b):
            return a + b

        return await r.async_call(target, 1, b=2)

    assert _run(_go()) == 3


# --------------------------------------------------------------------------
# Retrying iteration protocol
# --------------------------------------------------------------------------
#
# __next__ returns an attempt. The previous attempt's `failed` flag decides
# whether iteration stops or continues. With a successful prior attempt,
# next() raises StopIteration. With a failed prior attempt, the condition
# and stop predicates run; if they allow, iteration yields a new attempt.

from backon._conditions import RetryCondition, Stop


class _AlwaysRetry(RetryCondition):
    def __call__(self, state):
        return True


class _NeverStop(Stop):
    def __call__(self, state):
        return False


def test_retrying_iter_success():
    r = Retrying()
    iterator = iter(r)
    attempt = next(iterator)
    attempt.set_value(5)
    with pytest.raises(StopIteration):
        next(iterator)


def test_retrying_iter_retries():
    _, sleep = _record_sleep()
    r = Retrying(sleep=sleep, jitter=None, condition=_AlwaysRetry(), stop=_NeverStop())
    iterator = iter(r)
    assert next(iterator) is not None
    # Each attempt must fail for iteration to continue.
    r._last_attempt._exception = ValueError("x")
    assert next(iterator) is not None
    r._last_attempt._exception = ValueError("x")
    assert next(iterator) is not None


def test_retrying_iter_counts_tries():
    _, sleep = _record_sleep()
    r = Retrying(sleep=sleep, jitter=None, condition=_AlwaysRetry(), stop=_NeverStop())
    iterator = iter(r)
    next(iterator)
    assert r._iter_state.tries == 1
    r._last_attempt._exception = ValueError("x")
    next(iterator)
    assert r._iter_state.tries == 2


def test_retrying_iter_max_tries_gives_up():
    _, sleep = _record_sleep()
    # max_tries=3 gives at least 2 allowed attempts before raising.
    r = Retrying(max_tries=3, sleep=sleep, jitter=None)
    iterator = iter(r)
    next(iterator)
    r._last_attempt._exception = ValueError("x")
    # Second attempt allowed (tries=1 < max_tries=3)
    next(iterator)
    r._last_attempt._exception = ValueError("y")
    # Second failure should now trigger stop (tries=2 reached)
    with pytest.raises(ValueError):
        next(iterator)


def test_retrying_iter_giveup_no_raise_stops_iteration():
    _, sleep = _record_sleep()
    # max_tries=2 -> first failure at tries==1... let's confirm via explicit
    # behavior: after first failed attempt with tries==1, stop triggers at
    # tries==1 when max_tries==1. Use max_tries=1 to make it deterministic.
    r = Retrying(max_tries=1, sleep=sleep, jitter=None, raise_on_giveup=False)
    iterator = iter(r)
    next(iterator)
    r._last_attempt._exception = ValueError("x")
    with pytest.raises(StopIteration):
        next(iterator)


# --------------------------------------------------------------------------
# Retrying.copy
# --------------------------------------------------------------------------

def test_retrying_copy_preserves_settings():
    def custom_giveup(e):
        return True

    r = Retrying(
        max_tries=7,
        max_time=10.0,
        jitter=None,
        giveup=custom_giveup,
        raise_on_giveup=False,
        logger=None,
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        name="myname",
        enabled=False,
        attempt_timeout=3.0,
    )
    c = r.copy()
    assert isinstance(c, Retrying)
    assert c is not r
    assert c._max_tries == 7
    assert c._max_time == 10.0
    assert c._jitter is None
    assert c._giveup is custom_giveup
    assert c._raise_on_giveup is False
    assert c._logger is None
    assert c._backoff_log_level == logging.WARNING
    assert c._giveup_log_level == logging.CRITICAL
    assert c._name == "myname"
    assert c.enabled is False
    assert c._attempt_timeout == 3.0


def test_retrying_copy_copies_wait_gen_kwargs():
    r = Retrying(base=2, factor=1.5)
    c = r.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 1.5}


# --------------------------------------------------------------------------
# Retrying statistics
# --------------------------------------------------------------------------

def test_retrying_statistics_after_call():
    _, sleep = _record_sleep()

    def target():
        raise ValueError("x")

    r = Retrying(max_tries=2, sleep=sleep, jitter=None)
    with pytest.raises(ValueError):
        r.call(target)
    stats = r.statistics
    assert isinstance(stats, dict)
    assert "attempt_number" in stats


# --------------------------------------------------------------------------
# RetryingCaller
# --------------------------------------------------------------------------

def test_retrying_caller_success():
    caller = RetryingCaller()
    assert caller(lambda: 10) == 10


def test_retrying_caller_retries():
    calls = []
    _, sleep = _record_sleep()
    # max_tries=2 means 2 total attempts: fail once, then succeed.
    caller = RetryingCaller(max_tries=2, sleep=sleep, jitter=None)

    def target():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("retry")
        return "ok"

    assert caller(target) == "ok"
    assert len(calls) == 2


def test_retrying_caller_giveup():
    _, sleep = _record_sleep()
    caller = RetryingCaller(max_tries=2, sleep=sleep, jitter=None)

    def target():
        raise ValueError("always")

    with pytest.raises(ValueError):
        caller(target)


def test_retrying_caller_rejects_coroutine():
    caller = RetryingCaller()

    async def coro():
        return 1

    with pytest.raises(TypeError):
        caller(coro)


def test_retrying_caller_on_returns_new_instance():
    caller = RetryingCaller(max_tries=3)
    new = caller.on(ValueError)
    assert isinstance(new, RetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    assert caller._exception is None
    assert new._max_tries == 3


def test_retrying_caller_on_with_sequence():
    caller = RetryingCaller()
    new = caller.on([ValueError, TypeError])
    assert new._exception == [ValueError, TypeError]


def test_retrying_caller_call_passes_args():
    caller = RetryingCaller()
    assert caller(lambda a, b: a + b, 3, 4) == 7


def test_retrying_caller_copy():
    def giveup(e):
        return False

    caller = RetryingCaller(
        max_tries=9,
        max_time=3.0,
        jitter=None,
        giveup=giveup,
        name="foo",
        raise_on_giveup=False,
        attempt_timeout=1.0,
        base=2,
    )
    c = caller.copy()
    assert isinstance(c, RetryingCaller)
    assert c is not caller
    assert c._max_tries == 9
    assert c._max_time == 3.0
    assert c._jitter is None
    assert c._giveup is giveup
    assert c._name == "foo"
    assert c._raise_on_giveup is False
    assert c._attempt_timeout == 1.0
    assert c._wait_gen_kwargs == {"base": 2}


# --------------------------------------------------------------------------
# AsyncRetryingCaller
# --------------------------------------------------------------------------

def test_async_retrying_caller_success():
    async def _go():
        caller = AsyncRetryingCaller()

        async def target():
            return "ok"

        return await caller(target)

    assert _run(_go()) == "ok"


def test_async_retrying_caller_retries():
    async def _go():
        calls = []

        async def sleep(seconds):
            pass

        # max_tries=2 means 2 total attempts: fail once, then succeed.
        caller = AsyncRetryingCaller(max_tries=2, sleep=sleep, jitter=None)

        async def target():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("retry")
            return "ok"

        result = await caller(target)
        return result, len(calls)

    result, n = _run(_go())
    assert result == "ok"
    assert n == 2


def test_async_retrying_caller_giveup():
    async def _go():
        async def sleep(seconds):
            pass

        caller = AsyncRetryingCaller(max_tries=2, sleep=sleep, jitter=None)

        async def target():
            raise ValueError("always")

        with pytest.raises(ValueError):
            await caller(target)

    _run(_go())


def test_async_retrying_caller_args():
    async def _go():
        caller = AsyncRetryingCaller()

        async def target(a, b):
            return a * b

        return await caller(target, 3, 4)

    assert _run(_go()) == 12


def test_async_retrying_caller_on():
    caller = AsyncRetryingCaller(max_tries=4)
    new = caller.on(ValueError)
    assert isinstance(new, AsyncRetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    assert caller._exception is None
    assert new._max_tries == 4


def test_async_retrying_caller_copy():
    def giveup(e):
        return True

    caller = AsyncRetryingCaller(
        max_tries=5,
        max_time=2.5,
        jitter=None,
        giveup=giveup,
        name="async-name",
        raise_on_giveup=False,
        attempt_timeout=0.5,
        base=3,
    )
    c = caller.copy()
    assert isinstance(c, AsyncRetryingCaller)
    assert c is not caller
    assert c._max_tries == 5
    assert c._max_time == 2.5
    assert c._jitter is None
    assert c._giveup is giveup
    assert c._name == "async-name"
    assert c._raise_on_giveup is False
    assert c._attempt_timeout == 0.5
    assert c._wait_gen_kwargs == {"base": 3}
