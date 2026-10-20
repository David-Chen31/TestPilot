import asyncio
import inspect
import logging
import operator
import sys

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------

def test_retry_attempt_enter_returns_self():
    attempt = _RetryAttempt()
    with attempt as a:
        assert a is attempt


def test_retry_attempt_success_path():
    attempt = _RetryAttempt()
    with attempt:
        attempt.set_value(42)
    assert attempt.failed is False
    assert attempt.exception is None
    assert attempt.value == 42


def test_retry_attempt_captures_exception():
    attempt = _RetryAttempt()
    with attempt:
        raise ValueError("boom")
    assert attempt.failed is True
    assert isinstance(attempt.exception, ValueError)
    assert str(attempt.exception) == "boom"


def test_retry_attempt_does_not_swallow_base_exception():
    attempt = _RetryAttempt()
    with pytest.raises(KeyboardInterrupt):
        with attempt:
            raise KeyboardInterrupt()
    assert attempt.failed is False
    assert attempt.exception is None


def test_retry_attempt_value_default_none():
    attempt = _RetryAttempt()
    assert attempt.value is None


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self):
        self.calls = []

    def wait(self, timeout=None):
        self.calls.append(timeout)
        return False


def test_sleep_using_event_calls_wait_with_timeout():
    event = _FakeEvent()
    sleep = sleep_using_event(event)
    sleep(1.5)
    assert event.calls == [1.5]


def test_sleep_using_event_returns_callable():
    event = _FakeEvent()
    sleep = sleep_using_event(event)
    assert callable(sleep)


# ---------------------------------------------------------------------------
# Retrying - construction / properties
# ---------------------------------------------------------------------------

def test_retrying_statistics_empty_by_default():
    r = Retrying()
    assert r.statistics == {}


def test_retrying_call_state_none_by_default():
    r = Retrying()
    assert r.call_state is None


def test_retrying_enabled_default_true():
    r = Retrying()
    assert r.enabled is True


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


@pytest.mark.asyncio
async def test_retrying_async_context_manager():
    r = Retrying()
    async with r as ctx:
        assert ctx is r


# ---------------------------------------------------------------------------
# Retrying - iteration protocol
# ---------------------------------------------------------------------------

def test_retrying_iteration_stops_after_success():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    attempts = 0
    for attempt in r:
        attempts += 1
        attempt.set_value("ok")
    assert attempts == 1


def test_retrying_iteration_retries_on_failure():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    attempts = 0
    with pytest.raises(ValueError):
        for attempt in r:
            attempts += 1
            raise ValueError("fail")
    assert attempts == 3


def test_retrying_iteration_giveup_raise_on_giveup_false():
    r = Retrying(max_tries=2, sleep=lambda s: None, jitter=None, raise_on_giveup=False)
    attempts = 0
    for attempt in r:
        attempts += 1
        raise ValueError("fail")
    assert attempts == 2


def test_retrying_iteration_giveup_raise_on_giveup_true():
    r = Retrying(max_tries=2, sleep=lambda s: None, jitter=None, raise_on_giveup=True)
    with pytest.raises(ValueError):
        for attempt in r:
            raise ValueError("fail")


def test_retrying_iteration_giveup_predicate():
    giveup_calls = []

    def giveup(exc):
        giveup_calls.append(exc)
        return True

    r = Retrying(sleep=lambda s: None, jitter=None, giveup=giveup, raise_on_giveup=False)
    attempts = 0
    for attempt in r:
        attempts += 1
        raise ValueError("fail")
    assert attempts == 1
    assert len(giveup_calls) == 1


def test_retrying_iteration_sleep_called_with_jitter_none():
    slept = []
    r = Retrying(max_tries=3, sleep=slept.append, jitter=None, wait_gen=_fixed_wait(1.0))
    attempts = 0
    with pytest.raises(ValueError):
        for attempt in r:
            attempts += 1
            raise ValueError("fail")
    assert attempts == 3
    # before attempt 2, then before attempt 3 -> two sleeps
    assert slept == [1.0, 1.0]


def test_retrying_iteration_no_sleep_on_zero_seconds():
    slept = []
    r = Retrying(max_tries=2, sleep=slept.append, jitter=None, wait_gen=_fixed_wait(0.0))
    attempts = 0
    with pytest.raises(ValueError):
        for attempt in r:
            attempts += 1
            raise ValueError("fail")
    assert attempts == 2
    assert slept == []


def test_retrying_predicate_based_retry():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    attempts = 0
    for attempt in r:
        attempts += 1
        attempt.set_value(False if attempts < 3 else True)
    assert attempts == 3


def _fixed_wait(value):
    def gen(**kwargs):
        while True:
            yield value
    return gen


# ---------------------------------------------------------------------------
# Retrying - copy
# ---------------------------------------------------------------------------

def test_retrying_copy_preserves_config():
    sleep_fn = lambda s: None
    r = Retrying(
        max_tries=5,
        max_time=10,
        sleep=sleep_fn,
        enabled=False,
        name="myname",
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        raise_on_giveup=False,
    )
    c = r.copy()
    assert isinstance(c, Retrying)
    assert c is not r
    assert c._max_tries == 5
    assert c._max_time == 10
    assert c._sleep is sleep_fn
    assert c.enabled is False
    assert c._name == "myname"
    assert c._backoff_log_level == logging.WARNING
    assert c._giveup_log_level == logging.CRITICAL
    assert c._raise_on_giveup is False


# ---------------------------------------------------------------------------
# Retrying - call
# ---------------------------------------------------------------------------

def test_retrying_call_success():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    result = r.call(lambda a, b: a + b, 1, 2)
    assert result == 3


def test_retrying_call_retries_then_succeeds():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "done"

    r = Retrying(max_tries=5, sleep=lambda s: None, jitter=None)
    result = r.call(target)
    assert result == "done"
    assert state["n"] == 3


def test_retrying_call_giveup_raises():
    def target():
        raise ValueError("always")

    r = Retrying(max_tries=2, sleep=lambda s: None, jitter=None)
    with pytest.raises(ValueError):
        r.call(target)


def test_retrying_call_giveup_no_raise():
    def target():
        raise ValueError("always")

    r = Retrying(
        max_tries=2, sleep=lambda s: None, jitter=None, raise_on_giveup=False
    )
    result = r.call(target)
    assert result is None


def test_retrying_call_async_target_raises_typeerror():
    async def coro():
        return 1

    r = Retrying()
    with pytest.raises(TypeError):
        r.call(coro)


def test_retrying_call_sets_statistics_after_completion():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    r.call(lambda: "ok")
    stats = r.statistics
    # After the call completes, statistics comes from _state (or call_state)
    # but at minimum it should be a dict.
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Retrying - async_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrying_async_call_success():
    async def target(a, b):
        return a * b

    r = Retrying(max_tries=3, sleep=_async_no_sleep, jitter=None)
    result = await r.async_call(target, 3, 4)
    assert result == 12


@pytest.mark.asyncio
async def test_retrying_async_call_retries_then_succeeds():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "done"

    r = Retrying(max_tries=5, sleep=_async_no_sleep, jitter=None)
    result = await r.async_call(target)
    assert result == "done"
    assert state["n"] == 3


@pytest.mark.asyncio
async def test_retrying_async_call_giveup_raises():
    async def target():
        raise ValueError("always")

    r = Retrying(max_tries=2, sleep=_async_no_sleep, jitter=None)
    with pytest.raises(ValueError):
        await r.async_call(target)


async def _async_no_sleep(seconds):
    return None


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------

def test_retrying_caller_call_success():
    caller = RetryingCaller(max_tries=3, sleep=lambda s: None, jitter=None)
    assert caller(lambda a, b: a + b, 2, 3) == 5


def test_retrying_caller_retries():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("fail")
        return "ok"

    caller = RetryingCaller(max_tries=5, sleep=lambda s: None, jitter=None)
    assert caller(target) == "ok"
    assert state["n"] == 2


def test_retrying_caller_giveup_raises():
    def target():
        raise ValueError("always")

    caller = RetryingCaller(max_tries=2, sleep=lambda s: None, jitter=None)
    with pytest.raises(ValueError):
        caller(target)


def test_retrying_caller_async_target_raises_typeerror():
    async def coro():
        return 1

    caller = RetryingCaller()
    with pytest.raises(TypeError):
        caller(coro)


def test_retrying_caller_on_returns_new_with_exception():
    caller = RetryingCaller(max_tries=3)
    new = caller.on(ValueError)
    assert isinstance(new, RetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    # Original unchanged
    assert caller._exception is None


def test_retrying_caller_on_sequence():
    caller = RetryingCaller()
    new = caller.on((ValueError, TypeError))
    assert new._exception == (ValueError, TypeError)


def test_retrying_caller_copy_preserves_config():
    sleep_fn = lambda s: None
    caller = RetryingCaller(
        max_tries=5,
        sleep=sleep_fn,
        name="n",
        raise_on_giveup=False,
        backoff_log_level=logging.WARNING,
    )
    c = caller.copy()
    assert isinstance(c, RetryingCaller)
    assert c is not caller
    assert c._max_tries == 5
    assert c._sleep is sleep_fn
    assert c._name == "n"
    assert c._raise_on_giveup is False
    assert c._backoff_log_level == logging.WARNING


def test_retrying_caller_on_exception_limits_retries():
    state = {"n": 0}

    def target():
        state["n"] += 1
        raise TypeError("nope")

    # Only retry on ValueError -> TypeError should propagate immediately.
    caller = RetryingCaller(max_tries=5, sleep=lambda s: None, jitter=None).on(ValueError)
    with pytest.raises(TypeError):
        caller(target)
    assert state["n"] == 1


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_retrying_caller_success():
    async def target(a, b):
        return a + b

    caller = AsyncRetryingCaller(max_tries=3, sleep=_async_no_sleep, jitter=None)
    assert await caller(target, 1, 2) == 3


@pytest.mark.asyncio
async def test_async_retrying_caller_retries():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("fail")
        return "ok"

    caller = AsyncRetryingCaller(max_tries=5, sleep=_async_no_sleep, jitter=None)
    assert await caller(target) == "ok"
    assert state["n"] == 2


@pytest.mark.asyncio
async def test_async_retrying_caller_giveup_raises():
    async def target():
        raise ValueError("always")

    caller = AsyncRetryingCaller(max_tries=2, sleep=_async_no_sleep, jitter=None)
    with pytest.raises(ValueError):
        await caller(target)


def test_async_retrying_caller_on_returns_new():
    caller = AsyncRetryingCaller()
    new = caller.on(ValueError)
    assert isinstance(new, AsyncRetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    assert caller._exception is None


def test_async_retrying_caller_copy_preserves_config():
    sleep_fn = lambda s: None
    caller = AsyncRetryingCaller(
        max_tries=4,
        sleep=sleep_fn,
        name="x",
        raise_on_giveup=False,
    )
    c = caller.copy()
    assert isinstance(c, AsyncRetryingCaller)
    assert c is not caller
    assert c._max_tries == 4
    assert c._sleep is sleep_fn
    assert c._name == "x"
    assert c._raise_on_giveup is False


# ---------------------------------------------------------------------------
# Signature checks
# ---------------------------------------------------------------------------

def test_retrying_call_uses_wait_gen_kwargs():
    # Ensure wait_gen kwargs are forwarded; create a custom wait_gen that
    # records kwargs.
    recorded = {}

    def custom_wait(**kwargs):
        recorded.update(kwargs)
        while True:
            yield 0.0

    r = Retrying(wait_gen=custom_wait, sleep=lambda s: None, jitter=None, custom_kw=7)
    r.call(lambda: "ok")
    assert recorded == {"custom_kw": 7}
