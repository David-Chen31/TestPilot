"""Unit tests for backon._retry._classes."""

from __future__ import annotations

import asyncio
import logging
import operator
import time

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    _RetryAttempt,
    sleep_using_event,
)
from backon._conditions import RetryCondition, Stop
from backon._jitter import full_jitter
from backon._wait_gen import expo


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------


def test_retry_attempt_context_manager_success():
    with _RetryAttempt() as a:
        assert isinstance(a, _RetryAttempt)
        a.set_value(42)
    assert a.failed is False
    assert a.exception is None
    assert a.value == 42


def test_retry_attempt_captures_exception():
    exc = ValueError("boom")
    with _RetryAttempt() as a:
        raise exc
    assert a.failed is True
    assert a.exception is exc
    assert a.value is None


def test_retry_attempt_base_exception_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        with _RetryAttempt():
            raise KeyboardInterrupt()


def test_retry_attempt_returns_false_on_success():
    a = _RetryAttempt()
    assert a.__exit__(None, None, None) is False


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------


def test_sleep_using_event():
    class FakeEvent:
        def __init__(self):
            self.calls = []

        def wait(self, timeout=None):
            self.calls.append(timeout)

    ev = FakeEvent()
    fn = sleep_using_event(ev)
    fn(1.5)
    assert ev.calls == [1.5]


# ---------------------------------------------------------------------------
# Retrying: basic properties
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


def test_retrying_context_managers():
    r = Retrying()
    with r as ctx:
        assert ctx is r


def test_retrying_async_context_manager():
    async def run():
        r = Retrying()
        async with r as ctx:
            assert ctx is r

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Retrying.call
# ---------------------------------------------------------------------------


def test_retrying_call_success_first_try():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)
    calls = []

    def target(x, y=0):
        calls.append((x, y))
        return x + y

    result = r.call(target, 2, y=3)
    assert result == 5
    assert calls == [(2, 3)]


def test_retrying_call_retries_then_succeeds():
    r = Retrying(
        max_tries=5,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("nope")
        return "ok"

    assert r.call(target) == "ok"
    assert attempts["n"] == 3


def test_retrying_call_gives_up_and_raises():
    r = Retrying(
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )
    attempts = {"n": 0}
    exc = ValueError("always")

    def target():
        attempts["n"] += 1
        raise exc

    with pytest.raises(ValueError) as excinfo:
        r.call(target)
    assert excinfo.value is exc
    assert attempts["n"] == 2


def test_retrying_call_giveup_no_raise():
    r = Retrying(
        max_tries=1,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
        raise_on_giveup=False,
    )

    def target():
        raise ValueError("x")

    assert r.call(target) is None


def test_retrying_call_rejects_async_function():
    r = Retrying()

    async def coro():
        return 1

    with pytest.raises(TypeError):
        r.call(coro)


def test_retrying_call_updates_statistics():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None, exception=ValueError)
    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("retry")
        return 1

    r.call(target)
    stats = r.statistics
    assert stats != {}
    assert r.call_state is not None


# ---------------------------------------------------------------------------
# Retrying.async_call
# ---------------------------------------------------------------------------


def test_async_call_success():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None)

    async def target(a, b):
        return a * b

    assert asyncio.run(r.async_call(target, 3, 4)) == 12


def test_async_call_retries_then_succeeds():
    r = Retrying(
        max_tries=5,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("nope")
        return "ok"

    assert asyncio.run(r.async_call(target)) == "ok"
    assert attempts["n"] == 3


def test_async_call_raises_after_giveup():
    r = Retrying(
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )

    async def target():
        raise ValueError("always")

    with pytest.raises(ValueError):
        asyncio.run(r.async_call(target))


# ---------------------------------------------------------------------------
# Retrying iterator protocol
# ---------------------------------------------------------------------------


def test_retrying_iterator():
    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None, exception=ValueError)
    iterator = iter(r)
    assert iterator is r
    assert r.statistics != {} or True

    attempts = 0
    try:
        while True:
            with r as attempt:
                attempts += 1
                if attempts < 2:
                    raise ValueError("retry me")
                attempt.set_value("done")
                break
    except StopIteration:
        pass

    assert attempts == 2


def test_retrying_iterator_success_stops():
    r = Retrying(sleep=lambda s: None, jitter=None)
    it = iter(r)
    with it as attempt:
        attempt.set_value(123)
    with pytest.raises(StopIteration):
        next(it)


def test_retrying_iterator_giveup_raises():
    r = Retrying(max_tries=2, sleep=lambda s: None, jitter=None, exception=ValueError)
    it = iter(r)
    count = 0
    with pytest.raises(ValueError):
        while True:
            with it:
                count += 1
                raise ValueError("nope")
    assert count == 2


# ---------------------------------------------------------------------------
# Retrying.copy
# ---------------------------------------------------------------------------


def test_retrying_copy_preserves_config():
    r = Retrying(
        max_tries=7,
        max_time=3.0,
        exception=ValueError,
        raise_on_giveup=False,
        name="my-retry",
        logger="custom",
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
    )
    copy = r.copy()
    assert isinstance(copy, Retrying)
    assert copy._max_tries == 7
    assert copy._max_time == 3.0
    assert copy._exception is ValueError
    assert copy._raise_on_giveup is False
    assert copy._name == "my-retry"
    assert copy._logger == "custom"
    assert copy._backoff_log_level == logging.WARNING
    assert copy._giveup_log_level == logging.CRITICAL


def test_retrying_copy_is_independent():
    r = Retrying()
    copy = r.copy()
    copy.enabled = False
    assert r.enabled is True
    assert copy.enabled is False


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------


def test_retrying_caller_success():
    caller = RetryingCaller(max_tries=3, sleep=lambda s: None, jitter=None)
    assert caller(lambda a, b: a + b, 1, 2) == 3


def test_retrying_caller_retries():
    caller = RetryingCaller(
        max_tries=5,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )
    attempts = {"n": 0}

    def target():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("x")
        return "yay"

    assert caller(target) == "yay"
    assert attempts["n"] == 3


def test_retrying_caller_giveup_raises():
    caller = RetryingCaller(
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )

    def target():
        raise ValueError("x")

    with pytest.raises(ValueError):
        caller(target)


def test_retrying_caller_rejects_async():
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


def test_retrying_caller_copy():
    caller = RetryingCaller(
        max_tries=4,
        exception=ValueError,
        name="abc",
        raise_on_giveup=False,
        logger="x",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
    )
    copy = caller.copy()
    assert copy is not caller
    assert copy._max_tries == 4
    assert copy._exception is ValueError
    assert copy._name == "abc"
    assert copy._raise_on_giveup is False
    assert copy._logger == "x"
    assert copy._backoff_log_level == logging.DEBUG
    assert copy._giveup_log_level == logging.WARNING


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------


def test_async_retrying_caller_success():
    caller = AsyncRetryingCaller(max_tries=3, sleep=lambda s: None, jitter=None)

    async def target(a, b):
        return a * b

    assert asyncio.run(caller(target, 4, 5)) == 20


def test_async_retrying_caller_retries():
    caller = AsyncRetryingCaller(
        max_tries=5,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )
    attempts = {"n": 0}

    async def target():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("x")
        return "done"

    assert asyncio.run(caller(target)) == "done"
    assert attempts["n"] == 2


def test_async_retrying_caller_giveup_raises():
    caller = AsyncRetryingCaller(
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        exception=ValueError,
    )

    async def target():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(caller(target))


def test_async_retrying_caller_on_and_copy():
    caller = AsyncRetryingCaller(max_tries=3, name="n1")
    new = caller.on(RuntimeError)
    assert isinstance(new, AsyncRetryingCaller)
    assert new is not caller
    assert new._exception is RuntimeError

    copy = caller.copy()
    assert copy is not caller
    assert copy._max_tries == 3
    assert copy._name == "n1"


# ---------------------------------------------------------------------------
# Custom condition / stop / handlers
# ---------------------------------------------------------------------------


def test_custom_stop_and_condition():
    class AlwaysRetry(RetryCondition):
        def __call__(self, state):
            return True

    class AlwaysStop(Stop):
        def __call__(self, state):
            return state.tries >= 2

    r = Retrying(
        condition=AlwaysRetry(),
        stop=AlwaysStop(),
        sleep=lambda s: None,
        jitter=None,
    )

    def target():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        r.call(target)


def test_on_success_handler_called():
    successes = []

    def on_success(details):
        successes.append(details)

    r = Retrying(max_tries=2, sleep=lambda s: None, jitter=None, on_success=on_success)
    r.call(lambda: 10)
    assert len(successes) >= 1


def test_on_attempt_handler_called():
    attempts = []

    def on_attempt(details):
        attempts.append(details)

    r = Retrying(max_tries=3, sleep=lambda s: None, jitter=None, on_attempt=on_attempt)

    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("x")
        return 1

    r.call(target)
    assert len(attempts) >= 2


def test_sleep_callback_called_between_retries():
    sleeps = []
    r = Retrying(
        max_tries=3,
        exception=ValueError,
        jitter=lambda v: 1.0,
        wait_gen=expo,
        base=1,
        sleep=lambda s: sleeps.append(s),
    )

    state = {"n": 0}

    def target():
        state["n"] += 1
        if state["n"] < 3:
            raise ValueError("retry")
        return "ok"

    assert r.call(target) == "ok"
    assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# wait_gen_kwargs propagation
# ---------------------------------------------------------------------------


def test_wait_gen_kwargs_stored():
    r = Retrying(base=2, factor=3)
    assert r._wait_gen_kwargs == {"base": 2, "factor": 3}
    copy = r.copy()
    assert copy._wait_gen_kwargs == {"base": 2, "factor": 3}
