import asyncio
import logging
import operator
import threading

import pytest

from backon._retry._classes import (
    _RetryAttempt,
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
)
from backon._wait_gen import expo


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------

def test_retry_attempt_context_success():
    attempt = _RetryAttempt()
    with attempt as a:
        assert a is attempt
        a.set_value(42)
    assert attempt.failed is False
    assert attempt.exception is None
    assert attempt.value == 42


def test_retry_attempt_context_swallows_exception():
    attempt = _RetryAttempt()
    with attempt:
        raise ValueError("boom")
    assert attempt.failed is True
    assert isinstance(attempt.exception, ValueError)
    assert attempt.value is None


def test_retry_attempt_does_not_swallow_base_exception():
    attempt = _RetryAttempt()
    with pytest.raises(KeyboardInterrupt):
        with attempt:
            raise KeyboardInterrupt()
    assert attempt.failed is False


# ---------------------------------------------------------------------------
# Retrying: defaults and basic attributes
# ---------------------------------------------------------------------------

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


def test_retrying_context_manager_sync():
    r = Retrying()
    with r as entered:
        assert entered is r


@pytest.mark.asyncio
async def test_retrying_context_manager_async():
    r = Retrying()
    async with r as entered:
        assert entered is r


# ---------------------------------------------------------------------------
# Retrying: successful call
# ---------------------------------------------------------------------------

def test_retrying_call_success():
    calls = []

    def target(x):
        calls.append(x)
        return x * 2

    r = Retrying()
    assert r.call(target, 5) == 10
    assert calls == [5]
    assert r.call_state is not None


def test_retrying_call_uses_kwargs():
    def target(a, b=0):
        return a + b

    r = Retrying()
    assert r.call(target, 1, b=2) == 3


def test_retrying_call_rejects_coroutine_function():
    async def target():
        return 1

    r = Retrying()
    with pytest.raises(TypeError):
        r.call(target)


# ---------------------------------------------------------------------------
# Retrying: retries until success
# ---------------------------------------------------------------------------

def test_retrying_call_retries_until_success():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "ok"

    r = Retrying(exception=ValueError, max_tries=5, sleep=lambda _s: None, jitter=None)
    assert r.call(target) == "ok"
    assert state["n"] == 3
    assert r.statistics["tries"] >= 1


def test_retrying_call_gives_up_raises():
    def target():
        raise ValueError("always")

    r = Retrying(exception=ValueError, max_tries=2, sleep=lambda _s: None, jitter=None)
    with pytest.raises(ValueError):
        r.call(target)


def test_retrying_call_gives_up_no_raise():
    def target():
        raise ValueError("always")

    r = Retrying(
        exception=ValueError,
        max_tries=2,
        sleep=lambda _s: None,
        jitter=None,
        raise_on_giveup=False,
    )
    with pytest.raises(StopIteration):
        r.call(target)


def test_retrying_call_retry_error_callback():
    def target():
        raise ValueError("always")

    def cb(details):
        return "recovered"

    r = Retrying(
        exception=ValueError,
        max_tries=2,
        sleep=lambda _s: None,
        jitter=None,
        retry_error_callback=cb,
    )
    assert r.call(target) == "recovered"


def test_retrying_call_records_call_state_after_failure():
    def target():
        raise ValueError("x")

    r = Retrying(
        exception=ValueError,
        max_tries=1,
        sleep=lambda _s: None,
        jitter=None,
        raise_on_giveup=False,
        retry_error_callback=lambda d: None,
    )
    r.call(target)
    assert r.call_state is not None


# ---------------------------------------------------------------------------
# Retrying: async_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_call_success():
    async def target(x):
        return x + 1

    r = Retrying()
    assert await r.async_call(target, 1) == 2
    assert r.call_state is not None


@pytest.mark.asyncio
async def test_async_call_retries_until_success():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "ok"

    r = Retrying(
        exception=ValueError,
        max_tries=5,
        sleep=lambda _s: None,
        jitter=None,
    )
    # async_call uses the provided sleep; make sure it's awaited
    async def async_sleep(s):
        return None

    r = Retrying(
        exception=ValueError,
        max_tries=5,
        sleep=async_sleep,
        jitter=None,
    )
    assert await r.async_call(target) == "ok"
    assert state["n"] == 3


@pytest.mark.asyncio
async def test_async_call_gives_up_raises():
    async def target():
        raise ValueError("always")

    async def async_sleep(s):
        return None

    r = Retrying(
        exception=ValueError,
        max_tries=2,
        sleep=async_sleep,
        jitter=None,
    )
    with pytest.raises(ValueError):
        await r.async_call(target)


# ---------------------------------------------------------------------------
# Retrying: copy
# ---------------------------------------------------------------------------

def test_retrying_copy_preserves_config():
    def pred(v):
        return v == 0

    r = Retrying(
        exception=ValueError,
        max_tries=3,
        max_time=10,
        predicate=pred,
        raise_on_giveup=False,
        enabled=False,
        name="myname",
        logger="test.logger",
        attempt_timeout=1.5,
    )
    c = r.copy()
    assert isinstance(c, Retrying)
    assert c is not r
    assert c._exception is ValueError
    assert c._max_tries == 3
    assert c._max_time == 10
    assert c._predicate is pred
    assert c._raise_on_giveup is False
    assert c.enabled is False
    assert c._name == "myname"
    assert c._logger == "test.logger"
    assert c._attempt_timeout == 1.5


def test_retrying_copy_passes_wait_gen_kwargs():
    r = Retrying(expo, base=2, factor=3)
    c = r.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 3}


# ---------------------------------------------------------------------------
# Retrying: iteration protocol
# ---------------------------------------------------------------------------

def test_retrying_iter_success_first_attempt():
    r = Retrying()
    it = iter(r)
    attempt = next(it)
    assert isinstance(attempt, _RetryAttempt)
    attempt.set_value("done")
    with pytest.raises(StopIteration):
        next(it)


def test_retrying_iter_retries_after_failure():
    r = Retrying(exception=ValueError, max_tries=3, sleep=lambda _s: None, jitter=None)
    it = iter(r)
    with it as attempt:
        raise ValueError("fail")
    assert attempt.failed

    attempt2 = next(it)
    assert isinstance(attempt2, _RetryAttempt)
    attempt2.set_value("ok")
    with pytest.raises(StopIteration):
        next(it)


def test_retrying_iter_giveup_raises():
    r = Retrying(exception=ValueError, max_tries=1, sleep=lambda _s: None, jitter=None)
    it = iter(r)
    with it:
        raise ValueError("fail")
    with pytest.raises(ValueError):
        next(it)


def test_retrying_iter_giveup_no_raise():
    r = Retrying(
        exception=ValueError,
        max_tries=1,
        sleep=lambda _s: None,
        jitter=None,
        raise_on_giveup=False,
    )
    it = iter(r)
    with it:
        raise ValueError("fail")
    with pytest.raises(StopIteration):
        next(it)


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------

def test_sleep_using_event_calls_wait_with_timeout():
    class FakeEvent:
        def __init__(self):
            self.calls = []

        def wait(self, timeout=None):
            self.calls.append(timeout)
            return True

    ev = FakeEvent()
    sleep = sleep_using_event(ev)
    sleep(2.5)
    assert ev.calls == [2.5]


def test_sleep_using_event_real_event():
    ev = threading.Event()
    sleep = sleep_using_event(ev)
    # timeout=0 should return immediately without setting the event
    sleep(0)
    assert not ev.is_set()


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------

def test_retrying_caller_success():
    calls = []

    def target(x):
        calls.append(x)
        return x + 1

    caller = RetryingCaller()
    assert caller(target, 4) == 5
    assert calls == [4]


def test_retrying_caller_retries_until_success():
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("fail")
        return "ok"

    caller = RetryingCaller(
        exception=ValueError, max_tries=5, sleep=lambda _s: None, jitter=None
    )
    assert caller(target) == "ok"
    assert state["n"] == 3


def test_retrying_caller_gives_up_raises():
    def target():
        raise ValueError("always")

    caller = RetryingCaller(
        exception=ValueError, max_tries=2, sleep=lambda _s: None, jitter=None
    )
    with pytest.raises(ValueError):
        caller(target)


def test_retrying_caller_rejects_coroutine_function():
    async def target():
        return 1

    caller = RetryingCaller()
    with pytest.raises(TypeError):
        caller(target)


def test_retrying_caller_on_returns_new_with_exception():
    caller = RetryingCaller()
    new = caller.on(ValueError)
    assert new is not caller
    assert isinstance(new, RetryingCaller)
    assert new._exception is ValueError
    # Original unchanged
    assert caller._exception is None


def test_retrying_caller_on_accepts_sequence():
    caller = RetryingCaller()
    new = caller.on((ValueError, KeyError))
    assert new._exception == (ValueError, KeyError)


def test_retrying_caller_copy_preserves_config():
    caller = RetryingCaller(
        exception=ValueError,
        max_tries=3,
        max_time=10,
        raise_on_giveup=False,
        name="abc",
        logger="my.logger",
        attempt_timeout=0.5,
    )
    c = caller.copy()
    assert isinstance(c, RetryingCaller)
    assert c is not caller
    assert c._exception is ValueError
    assert c._max_tries == 3
    assert c._max_time == 10
    assert c._raise_on_giveup is False
    assert c._name == "abc"
    assert c._logger == "my.logger"
    assert c._attempt_timeout == 0.5


def test_retrying_caller_copy_passes_wait_gen_kwargs():
    caller = RetryingCaller(expo, base=2, factor=3)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 3}


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_retrying_caller_success():
    async def target(x):
        return x * 3

    caller = AsyncRetryingCaller()
    assert await caller(target, 4) == 12


@pytest.mark.asyncio
async def test_async_retrying_caller_retries_until_success():
    state = {"n": 0}

    async def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("fail")
        return "ok"

    async def async_sleep(s):
        return None

    caller = AsyncRetryingCaller(
        exception=ValueError, max_tries=5, sleep=async_sleep, jitter=None
    )
    assert await caller(target) == "ok"
    assert state["n"] == 2


@pytest.mark.asyncio
async def test_async_retrying_caller_gives_up_raises():
    async def target():
        raise ValueError("always")

    async def async_sleep(s):
        return None

    caller = AsyncRetryingCaller(
        exception=ValueError, max_tries=2, sleep=async_sleep, jitter=None
    )
    with pytest.raises(ValueError):
        await caller(target)


def test_async_retrying_caller_on_returns_new():
    caller = AsyncRetryingCaller()
    new = caller.on(ValueError)
    assert new is not caller
    assert isinstance(new, AsyncRetryingCaller)
    assert new._exception is ValueError
    assert caller._exception is None


def test_async_retrying_caller_copy_preserves_config():
    caller = AsyncRetryingCaller(
        exception=KeyError,
        max_tries=4,
        max_time=5,
        raise_on_giveup=False,
        name="x",
        logger="l",
        attempt_timeout=0.25,
    )
    c = caller.copy()
    assert isinstance(c, AsyncRetryingCaller)
    assert c is not caller
    assert c._exception is KeyError
    assert c._max_tries == 4
    assert c._max_time == 5
    assert c._raise_on_giveup is False
    assert c._name == "x"
    assert c._logger == "l"
    assert c._attempt_timeout == 0.25


def test_async_retrying_caller_copy_passes_wait_gen_kwargs():
    caller = AsyncRetryingCaller(expo, base=5)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 5}


# ---------------------------------------------------------------------------
# Disabled retrying / caller
# ---------------------------------------------------------------------------

def test_retrying_call_single_attempt_success_returns_value():
    r = Retrying(max_tries=1)
    assert r.call(lambda: "direct") == "direct"


def test_retrying_condition_and_stop_override():
    from backon._conditions import stop_after_attempt

    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("x")

    r = Retrying(
        exception=ValueError,
        stop=stop_after_attempt(3),
        sleep=lambda _s: None,
        jitter=None,
        raise_on_giveup=False,
        retry_error_callback=lambda d: "cb",
    )
    assert r.call(target) == "cb"
    assert calls["n"] == 3
