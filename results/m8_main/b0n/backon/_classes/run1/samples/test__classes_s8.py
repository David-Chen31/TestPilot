"""Unit tests for backon._retry._classes."""

from __future__ import annotations

import asyncio

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)


# ---------------------------------------------------------------------------
# _RetryAttempt (helper, exercised directly)
# ---------------------------------------------------------------------------


def test_retry_attempt_success_records_value():
    a = _RetryAttempt()
    assert a.failed is False
    assert a.exception is None
    with a as entered:
        assert entered is a
        entered.set_value(42)
    assert a.failed is False
    assert a.exception is None
    assert a.value == 42


def test_retry_attempt_captures_exception():
    a = _RetryAttempt()
    with a:
        raise ValueError("boom")
    assert a.failed is True
    assert isinstance(a.exception, ValueError)
    assert str(a.exception) == "boom"


def test_retry_attempt_does_not_swallow_base_exception():
    a = _RetryAttempt()
    with pytest.raises(KeyboardInterrupt):
        with a:
            raise KeyboardInterrupt()
    assert a.failed is False
    assert a.exception is None


# ---------------------------------------------------------------------------
# Retrying - context manager / sync iterator
# ---------------------------------------------------------------------------


class _CountingSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def _make_retrying(**kwargs):
    sleep = _CountingSleep()
    retrying = Retrying(sleep=sleep, jitter=None, **kwargs)
    return retrying, sleep


def _iterate_until_success(retrying, values):
    """Run retrying iterator, returning the values produced."""
    results = []
    idx = 0
    for attempt in retrying:
        if idx < len(values):
            v = values[idx]
            idx += 1
        else:
            v = values[-1]
        attempt.set_value(v)
        results.append(v)
    return results


def test_retrying_enter_returns_self():
    r = Retrying()
    assert r.__enter__() is r
    # __exit__ returns None and does not suppress
    assert r.__exit__(None, None, None) is None


def test_retrying_success_first_try():
    retrying, sleep = _make_retrying()
    with retrying as r:
        results = []
        for attempt in r:
            attempt.set_value("ok")
            results.append(attempt.value)
    assert results == ["ok"]
    assert sleep.calls == []


def test_retrying_predicate_false_retries_until_true():
    # predicate default is operator.not_ (retry while falsy).
    retrying, sleep = _make_retrying(max_tries=5)
    with retrying as r:
        results = _iterate_until_success(r, [0, 0, 1])
    # Third attempt returns truthy -> StopIteration after.
    assert results == [0, 0, 1]
    # Two retries, both should attempt to sleep (seconds may be >0).
    assert len(sleep.calls) == 2


def test_retrying_exhausts_max_tries_raises_last_value_flow():
    class MyErr(Exception):
        pass

    retrying, _ = _make_retrying(exception=MyErr, max_tries=3)
    with pytest.raises(MyErr):
        with retrying as r:
            for attempt in r:
                attempt.set_value(None)
                raise MyErr("fail")


def test_retrying_no_raise_on_giveup_exits_cleanly():
    class MyErr(Exception):
        pass

    retrying, _ = _make_retrying(
        exception=MyErr, max_tries=2, raise_on_giveup=False
    )
    attempts = 0
    with retrying as r:
        for attempt in r:
            attempts += 1
            raise MyErr("fail")
    # Two tries attempted before giveup -> loop exits without raising.
    assert attempts == 2


def test_retrying_stops_on_success_with_value():
    retrying, _ = _make_retrying()
    count = 0
    with retrying as r:
        for attempt in r:
            count += 1
            attempt.set_value(True)
    assert count == 1


def test_retrying_statistics_empty_before_call():
    r = Retrying()
    assert r.statistics == {}


def test_retrying_call_state_none_before_call():
    r = Retrying()
    assert r.call_state is None


def test_retrying_enabled_property_setter():
    r = Retrying()
    assert r.enabled is True
    r.enabled = False
    assert r.enabled is False
    r.enabled = True
    assert r.enabled is True


# ---------------------------------------------------------------------------
# Retrying.copy
# ---------------------------------------------------------------------------


def test_retrying_copy_preserves_configuration():
    def on_success(details):
        pass

    r = Retrying(
        max_tries=3,
        max_time=10.0,
        raise_on_giveup=False,
        name="abc",
        logger="mylogger",
        backoff_log_level=10,
        giveup_log_level=20,
        on_success=on_success,
        wait_gen_kwargs=None,
        base=2,
        factor=1.5,
    )
    copy = r.copy()
    assert isinstance(copy, Retrying)
    assert copy is not r
    assert copy._wait_gen is r._wait_gen
    assert copy._max_tries == 3
    assert copy._max_time == 10.0
    assert copy._raise_on_giveup is False
    assert copy._name == "abc"
    assert copy._logger == "mylogger"
    assert copy._backoff_log_level == 10
    assert copy._giveup_log_level == 20
    assert copy._on_success is on_success
    assert copy._wait_gen_kwargs == {"base": 2, "factor": 1.5}


# ---------------------------------------------------------------------------
# Retrying.call / async_call
# ---------------------------------------------------------------------------


def test_retrying_call_returns_result():
    r = Retrying()
    result = r.call(lambda x, y: x + y, 2, 3)
    assert result == 5


def test_retrying_call_retries_until_success():
    r = Retrying(max_tries=5, sleep=lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("nope")
        return "done"

    result = r.call(fn)
    assert result == "done"
    assert len(calls) == 3


def test_retrying_call_raises_after_giveup():
    r = Retrying(max_tries=2, sleep=lambda s: None)

    def fn():
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError):
        r.call(fn)


def test_retrying_call_with_async_target_raises_typeerror():
    r = Retrying()

    async def coro():
        return 1

    with pytest.raises(TypeError) as excinfo:
        r.call(coro)
    assert "async" in str(excinfo.value).lower()


def test_retrying_call_updates_statistics():
    r = Retrying(sleep=lambda s: None)
    r.call(lambda: 42)
    assert isinstance(r.statistics, dict)
    assert "attempt_number" in r.statistics


def test_retrying_async_enter_returns_self():
    async def run():
        r = Retrying()
        result = await r.__aenter__()
        assert result is r
        assert await r.__aexit__(None, None, None) is None

    asyncio.run(run())


def test_retrying_async_call_returns_result():
    async def run():
        r = Retrying()
        return await r.async_call(_async_add, 2, 4)

    assert asyncio.run(run()) == 6


async def _async_add(a, b):
    return a + b


async def _async_flaky(counter):
    counter.append(1)
    if len(counter) < 3:
        raise ValueError("flaky")
    return "finished"


def test_retrying_async_call_retries_until_success():
    async def run():
        r = Retrying(max_tries=5, sleep=_no_sleep_async)
        counter = []
        return await r.async_call(_async_flaky, counter), counter

    result, counter = asyncio.run(run())
    assert result == "finished"
    assert len(counter) == 3


async def _no_sleep_async(seconds):
    return None


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------


def test_sleep_using_event_waits_with_timeout():
    calls = []

    class FakeEvent:
        def wait(self, timeout=None):
            calls.append(timeout)
            return True

    sleeper = sleep_using_event(FakeEvent())
    sleeper(1.5)
    assert calls == [1.5]


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------


def test_retrying_caller_calls_target():
    caller = RetryingCaller()
    assert caller(lambda x: x * 2, 5) == 10


def test_retrying_caller_retries_until_success():
    caller = RetryingCaller(max_tries=5, sleep=lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("nope")
        return "ok"

    assert caller(fn) == "ok"
    assert len(calls) == 2


def test_retrying_caller_on_returns_new_with_exception():
    class ErrA(Exception):
        pass

    class ErrB(Exception):
        pass

    caller = RetryingCaller()
    new = caller.on(ErrA)
    assert isinstance(new, RetryingCaller)
    assert new is not caller
    assert new._exception is ErrA
    assert caller._exception is None

    seq = caller.on([ErrA, ErrB])
    assert seq._exception == [ErrA, ErrB]


def test_retrying_caller_with_async_target_raises_typeerror():
    caller = RetryingCaller()

    async def coro():
        return 1

    with pytest.raises(TypeError) as excinfo:
        caller(coro)
    assert "async" in str(excinfo.value).lower()


def test_retrying_caller_copy_preserves_options():
    caller = RetryingCaller(
        max_tries=3,
        max_time=5.0,
        raise_on_giveup=False,
        name="n",
        logger="lg",
        backoff_log_level=10,
        giveup_log_level=20,
        base=2,
    )
    copy = caller.copy()
    assert isinstance(copy, RetryingCaller)
    assert copy is not caller
    assert copy._max_tries == 3
    assert copy._max_time == 5.0
    assert copy._raise_on_giveup is False
    assert copy._name == "n"
    assert copy._logger == "lg"
    assert copy._backoff_log_level == 10
    assert copy._giveup_log_level == 20
    assert copy._wait_gen_kwargs == {"base": 2}


def test_retrying_caller_on_then_call_retries_with_exception_type():
    class MyErr(Exception):
        pass

    caller = RetryingCaller(max_tries=3, sleep=lambda s: None).on(MyErr)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise MyErr("nope")
        return "ok"

    assert caller(fn) == "ok"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------


def test_async_retrying_caller_calls_coroutine():
    async def run():
        caller = AsyncRetryingCaller()
        return await caller(_async_add, 3, 4)

    assert asyncio.run(run()) == 7


def test_async_retrying_caller_retries_until_success():
    async def run():
        caller = AsyncRetryingCaller(max_tries=5, sleep=_no_sleep_async)
        counter = []
        return await caller(_async_flaky, counter), counter

    result, counter = asyncio.run(run())
    assert result == "finished"
    assert len(counter) == 3


def test_async_retrying_caller_on_returns_new_instance():
    class ErrA(Exception):
        pass

    caller = AsyncRetryingCaller()
    new = caller.on(ErrA)
    assert isinstance(new, AsyncRetryingCaller)
    assert new is not caller
    assert new._exception is ErrA
    assert caller._exception is None


def test_async_retrying_caller_copy_preserves_options():
    caller = AsyncRetryingCaller(
        max_tries=4,
        max_time=6.0,
        raise_on_giveup=False,
        name="an",
        logger="alg",
        backoff_log_level=11,
        giveup_log_level=21,
        base=3,
    )
    copy = caller.copy()
    assert isinstance(copy, AsyncRetryingCaller)
    assert copy is not caller
    assert copy._max_tries == 4
    assert copy._max_time == 6.0
    assert copy._raise_on_giveup is False
    assert copy._name == "an"
    assert copy._logger == "alg"
    assert copy._backoff_log_level == 11
    assert copy._giveup_log_level == 21
    assert copy._wait_gen_kwargs == {"base": 3}


def test_async_retrying_caller_on_with_exceptions_retries():
    class MyErr(Exception):
        pass

    async def flaky(counter):
        counter.append(1)
        if len(counter) < 2:
            raise MyErr("nope")
        return "ok"

    async def run():
        caller = AsyncRetryingCaller(max_tries=3, sleep=_no_sleep_async).on(MyErr)
        counter = []
        return await caller(flaky, counter), counter

    result, counter = asyncio.run(run())
    assert result == "ok"
    assert len(counter) == 2
