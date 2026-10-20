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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ThinStop:
    """Stop condition that never stops (so retries continue)."""

    def __call__(self, state):
        return False


class _StopAlways:
    def __call__(self, state):
        return True


class _ConditionRetryAlways:
    def __call__(self, state):
        return True


class _ConditionNever:
    def __call__(self, state):
        return False


def _no_sleep(_seconds):
    return None


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------

def test_retry_attempt_initial_state():
    attempt = _RetryAttempt()
    assert attempt.failed is False
    assert attempt.exception is None
    assert attempt.value is None


def test_retry_attempt_success_value():
    attempt = _RetryAttempt()
    with attempt:
        attempt.set_value(42)
    assert attempt.failed is False
    assert attempt.value == 42
    assert attempt.exception is None


def test_retry_attempt_swallows_exception():
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
    # __exit__ returns False for non-Exception BaseException
    assert attempt.failed is False


def test_retry_attempt_enter_returns_self():
    attempt = _RetryAttempt()
    with attempt as entered:
        assert entered is attempt


# ---------------------------------------------------------------------------
# Retrying constructor / properties
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


def test_retrying_statistics_without_state():
    r = Retrying()
    assert r.statistics == {}


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
# Retrying iteration protocol
# ---------------------------------------------------------------------------

def test_retrying_iter_success_stops_iteration():
    r = Retrying(sleep=_no_sleep, max_tries=3)
    it = iter(r)
    attempt = next(it)
    attempt.set_value("ok")
    with pytest.raises(StopIteration):
        next(it)


def test_retrying_iter_retries_until_max_tries():
    r = Retrying(sleep=_no_sleep, max_tries=3, jitter=None)
    it = iter(r)
    attempts = 0
    try:
        while True:
            attempt = next(it)
            attempts += 1
            with attempt:
                raise ValueError("fail")
    except ValueError:
        pass
    assert attempts == 3


def test_retrying_iter_raise_on_giveup_false():
    r = Retrying(
        sleep=_no_sleep,
        max_tries=2,
        raise_on_giveup=False,
        jitter=None,
    )
    it = iter(r)
    attempt = next(it)
    with attempt:
        raise ValueError("fail")
    attempt = next(it)
    with attempt:
        raise ValueError("fail")
    # next call sees giveup -> StopIteration (not raised ValueError)
    with pytest.raises(StopIteration):
        next(it)


def test_retrying_iter_respects_sleep_calls():
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    # Use a condition that always continues and a stop never, then force giveup
    r = Retrying(
        sleep=fake_sleep,
        max_tries=3,
        jitter=None,
        wait_gen=lambda **kw: (i for i in kw["vals"]),
        vals=(1.0, 2.0),
    )
    it = iter(r)
    attempt = next(it)
    with attempt:
        raise ValueError("x")
    attempt = next(it)
    with attempt:
        raise ValueError("x")
    assert sleeps == [1.0, 2.0]


def test_retrying_iter_no_sleep_when_seconds_zero():
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    r = Retrying(
        sleep=fake_sleep,
        max_tries=2,
        jitter=None,
        wait_gen=lambda **kw: (i for i in kw["vals"]),
        vals=(0.0,),
    )
    it = iter(r)
    attempt = next(it)
    with attempt:
        raise ValueError("x")
    # seconds == 0 so sleep not called
    attempt = next(it)
    assert sleeps == []


def test_retrying_iter_uses_configured_stop_and_condition():
    calls = {"condition": 0, "stop": 0}

    class Cond:
        def __call__(self, state):
            calls["condition"] += 1
            return False

    class StopCls:
        def __call__(self, state):
            calls["stop"] += 1
            return True

    r = Retrying(
        sleep=_no_sleep,
        condition=Cond(),
        stop=StopCls(),
        raise_on_giveup=False,
        jitter=None,
    )
    it = iter(r)
    attempt = next(it)
    with attempt:
        raise ValueError("x")
    with pytest.raises(StopIteration):
        next(it)
    assert calls["condition"] == 1
    assert calls["stop"] == 1


def test_retrying_iter_state_target_name():
    r = Retrying(name="myname", sleep=_no_sleep)
    iter(r)
    assert r._iter_state.target == "myname"


def test_retrying_iter_state_default_target():
    r = Retrying(sleep=_no_sleep)
    iter(r)
    assert r._iter_state.target == "Retrying"


# ---------------------------------------------------------------------------
# Retrying.copy
# ---------------------------------------------------------------------------

def test_retrying_copy_preserves_attributes():
    r = Retrying(
        predicate=bool,
        exception=ValueError,
        max_tries=5,
        max_time=10.0,
        jitter=None,
        giveup=lambda e: True,
        raise_on_giveup=False,
        logger=None,
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        sleep=_no_sleep,
        enabled=False,
        name="n",
        rate_limit=None,
        attempt_timeout=1.5,
    )
    c = r.copy()
    assert isinstance(c, Retrying)
    assert c._predicate is bool
    assert c._exception is ValueError
    assert c._max_tries == 5
    assert c._max_time == 10.0
    assert c._jitter is None
    assert c._raise_on_giveup is False
    assert c._logger is None
    assert c._backoff_log_level == logging.WARNING
    assert c._giveup_log_level == logging.CRITICAL
    assert c._sleep is _no_sleep
    assert c._enabled is False
    assert c._name == "n"
    assert c._attempt_timeout == 1.5


def test_retrying_copy_preserves_wait_gen_kwargs():
    r = Retrying(base=2, factor=3, sleep=_no_sleep)
    c = r.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 3}


# ---------------------------------------------------------------------------
# Retrying.call
# ---------------------------------------------------------------------------

def test_retrying_call_rejects_coroutine_function():
    async def coro():
        return 1

    r = Retrying()
    with pytest.raises(TypeError):
        r.call(coro)


def test_retrying_call_success():
    r = Retrying(sleep=_no_sleep, max_tries=3)
    result = r.call(lambda a, b: a + b, 2, 3)
    assert result == 5


def test_retrying_call_forwards_args_and_kwargs():
    seen = {}

    def target(a, b=None):
        seen["a"] = a
        seen["b"] = b
        return a

    r = Retrying(sleep=_no_sleep)
    r.call(target, 10, b=20)
    assert seen == {"a": 10, "b": 20}


def test_retrying_call_sets_state_and_call_state():
    r = Retrying(sleep=_no_sleep, max_tries=2)
    assert r._state is None
    r.call(lambda: "ok")
    assert r._state is not None
    assert r.statistics != {}


def test_retrying_call_retries_and_raises():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("always")

    r = Retrying(
        sleep=_no_sleep,
        max_tries=3,
        exception=ValueError,
        jitter=None,
    )
    with pytest.raises(ValueError):
        r.call(target)
    assert calls["n"] == 3


def test_retrying_call_raise_on_giveup_false_returns_none():
    def target():
        raise ValueError("x")

    r = Retrying(
        sleep=_no_sleep,
        max_tries=2,
        exception=ValueError,
        raise_on_giveup=False,
        jitter=None,
    )
    result = r.call(target)
    assert result is None


def test_retrying_call_retry_error_callback_returns_value():
    def target():
        raise ValueError("x")

    r = Retrying(
        sleep=_no_sleep,
        max_tries=2,
        exception=ValueError,
        retry_error_callback=lambda details: "fallback",
        jitter=None,
    )
    assert r.call(target) == "fallback"


def test_retrying_call_short_circuits_when_disabled(monkeypatch):
    import backon._retry._api as api_module
    from backon import _common as common_module

    monkeypatch.setattr(api_module, "is_enabled", lambda: True)
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        return "done"

    r = Retrying(sleep=_no_sleep)
    assert r.call(target) == "done"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Retrying.async_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrying_async_call_success():
    async def target(a, b=None):
        return (a, b)

    r = Retrying(sleep=_no_sleep)
    result = await r.async_call(target, 1, b=2)
    assert result == (1, 2)


@pytest.mark.asyncio
async def test_retrying_async_call_retries_and_raises():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise ValueError("always")

    r = Retrying(
        sleep=_no_sleep,
        max_tries=3,
        exception=ValueError,
        jitter=None,
    )
    with pytest.raises(ValueError):
        await r.async_call(target)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retrying_async_call_raise_on_giveup_false():
    async def target():
        raise ValueError("x")

    r = Retrying(
        sleep=_no_sleep,
        max_tries=2,
        exception=ValueError,
        raise_on_giveup=False,
        jitter=None,
    )
    assert await r.async_call(target) is None


@pytest.mark.asyncio
async def test_retrying_async_call_sets_state():
    async def target():
        return 1

    r = Retrying(sleep=_no_sleep)
    await r.async_call(target)
    assert r._state is not None
    assert r._call_state is not None


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------

def test_sleep_using_event_waits_timeout():
    class FakeEvent:
        def __init__(self):
            self.calls = []

        def wait(self, timeout=None):
            self.calls.append(timeout)
            return False

    event = FakeEvent()
    sleep = sleep_using_event(event)
    sleep(2.5)
    assert event.calls == [2.5]


def test_sleep_using_event_returns_callable():
    class FakeEvent:
        def wait(self, timeout=None):
            return True

    fn = sleep_using_event(FakeEvent())
    assert callable(fn)


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------

def test_retrying_caller_call_rejects_coroutine_function():
    async def coro():
        return 1

    caller = RetryingCaller()
    with pytest.raises(TypeError):
        caller(coro)


def test_retrying_caller_call_success():
    caller = RetryingCaller(sleep=_no_sleep, max_tries=3)
    result = caller(lambda a, b: a * b, 4, 5)
    assert result == 20


def test_retrying_caller_call_retries_then_raises():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        raise ValueError("nope")

    caller = RetryingCaller(
        sleep=_no_sleep,
        max_tries=3,
        exception=ValueError,
        jitter=None,
    )
    with pytest.raises(ValueError):
        caller(target)
    assert calls["n"] == 3


def test_retrying_caller_raise_on_giveup_false():
    def target():
        raise ValueError("x")

    caller = RetryingCaller(
        sleep=_no_sleep,
        max_tries=2,
        exception=ValueError,
        raise_on_giveup=False,
        jitter=None,
    )
    assert caller(target) is None


def test_retrying_caller_on_returns_new_instance():
    caller = RetryingCaller(sleep=_no_sleep, exception=ValueError)
    new = caller.on(KeyError)
    assert new is not caller
    assert new._exception is KeyError
    # original unchanged
    assert caller._exception is ValueError


def test_retrying_caller_copy_preserves_attributes():
    caller = RetryingCaller(
        predicate=bool,
        exception=ValueError,
        max_tries=4,
        max_time=20.0,
        jitter=None,
        raise_on_giveup=False,
        logger=None,
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        sleep=_no_sleep,
        name="c",
        attempt_timeout=2.0,
    )
    c = caller.copy()
    assert isinstance(c, RetryingCaller)
    assert c._predicate is bool
    assert c._exception is ValueError
    assert c._max_tries == 4
    assert c._max_time == 20.0
    assert c._jitter is None
    assert c._raise_on_giveup is False
    assert c._logger is None
    assert c._name == "c"
    assert c._attempt_timeout == 2.0


def test_retrying_caller_copy_preserves_wait_gen_kwargs():
    caller = RetryingCaller(base=2, factor=5)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 5}


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_retrying_caller_success():
    async def target(a, b=None):
        return (a, b)

    caller = AsyncRetryingCaller(sleep=_no_sleep)
    assert await caller(target, 1, b=2) == (1, 2)


@pytest.mark.asyncio
async def test_async_retrying_caller_retries_then_raises():
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        raise ValueError("bad")

    caller = AsyncRetryingCaller(
        sleep=_no_sleep,
        max_tries=3,
        exception=ValueError,
        jitter=None,
    )
    with pytest.raises(ValueError):
        await caller(target)
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_async_retrying_caller_raise_on_giveup_false():
    async def target():
        raise ValueError("x")

    caller = AsyncRetryingCaller(
        sleep=_no_sleep,
        max_tries=2,
        exception=ValueError,
        raise_on_giveup=False,
        jitter=None,
    )
    assert await caller(target) is None


def test_async_retrying_caller_on_returns_new_instance():
    caller = AsyncRetryingCaller(sleep=_no_sleep, exception=ValueError)
    new = caller.on(KeyError)
    assert new is not caller
    assert new._exception is KeyError
    assert caller._exception is ValueError


def test_async_retrying_caller_copy_preserves_attributes():
    caller = AsyncRetryingCaller(
        predicate=bool,
        exception=ValueError,
        max_tries=4,
        max_time=20.0,
        jitter=None,
        raise_on_giveup=False,
        logger=None,
        backoff_log_level=logging.WARNING,
        giveup_log_level=logging.CRITICAL,
        sleep=_no_sleep,
        name="a",
        attempt_timeout=2.0,
    )
    c = caller.copy()
    assert isinstance(c, AsyncRetryingCaller)
    assert c._predicate is bool
    assert c._exception is ValueError
    assert c._max_tries == 4
    assert c._max_time == 20.0
    assert c._jitter is None
    assert c._raise_on_giveup is False
    assert c._name == "a"
    assert c._attempt_timeout == 2.0


def test_async_retrying_caller_copy_preserves_wait_gen_kwargs():
    caller = AsyncRetryingCaller(base=2, factor=5)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 5}
