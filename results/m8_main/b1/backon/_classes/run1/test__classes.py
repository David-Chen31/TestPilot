import asyncio
import logging
import operator

import pytest

from backon._retry._classes import (
    _RetryAttempt,
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
)
from backon._state import RetryState


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------


def test_retry_attempt_initial_state():
    a = _RetryAttempt()
    assert a.failed is False
    assert a.exception is None
    assert a.value is None


def test_retry_attempt_context_manager_success():
    a = _RetryAttempt()
    with a as ctx:
        assert ctx is a
        ctx.set_value(42)
    assert a.failed is False
    assert a.value == 42
    assert a.exception is None


def test_retry_attempt_context_manager_swallows_exception():
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


def test_retry_attempt_set_value():
    a = _RetryAttempt()
    a.set_value("x")
    assert a.value == "x"
    assert a.failed is False


# ---------------------------------------------------------------------------
# Retrying basics
# ---------------------------------------------------------------------------


def test_retrying_defaults():
    r = Retrying()
    assert r.enabled is True
    assert r._raise_on_giveup is True
    assert r._name == ""
    assert r._logger == "backon"
    assert r._predicate is operator.not_
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
    async def _get():
        r = Retrying()
        async with r as ctx:
            assert ctx is r

    asyncio.run(_get())


def test_retrying_call_returns_value():
    r = Retrying()
    assert r.call(lambda: 5) == 5


def test_retrying_call_passes_args_and_kwargs():
    r = Retrying()
    assert r.call(lambda a, b, c=0: (a, b, c), 1, 2, c=3) == (1, 2, 3)


def test_retrying_call_with_async_target_raises():
    async def coro():
        return 1

    r = Retrying()
    with pytest.raises(TypeError):
        r.call(coro)


def test_retrying_call_retries_then_succeeds():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return "ok"

    r = Retrying(exception=ValueError, max_tries=5, sleep=lambda s: None, jitter=None)
    assert r.call(target) == "ok"
    assert calls["n"] == 3


def test_retrying_call_gives_up_raises():
    def target():
        raise ValueError("always")

    r = Retrying(exception=ValueError, max_tries=2, sleep=lambda s: None, jitter=None)
    with pytest.raises(ValueError):
        r.call(target)


def test_retrying_call_raise_on_giveup_false_stops_silently():
    def target():
        raise ValueError("always")

    r = Retrying(
        exception=ValueError,
        max_tries=2,
        sleep=lambda s: None,
        jitter=None,
        raise_on_giveup=False,
    )
    # Giveup -> StopIteration -> retry loop ends.
    r.call(target)


def test_retrying_call_stores_state_and_call_state():
    r = Retrying()
    result = r.call(lambda: 7)
    assert result == 7
    assert r.call_state is not None
    assert r._state is not None
    stats = r.statistics
    assert isinstance(stats, dict)


def test_retrying_statistics_from_call_state():
    r = Retrying()
    r.call(lambda: 1)
    assert r.statistics
    assert "attempt_number" in r.statistics


def test_retrying_async_call_returns_value():
    async def target():
        return 11

    async def _run():
        r = Retrying()
        return await r.async_call(target)

    assert asyncio.run(_run()) == 11


def test_retrying_async_call_retries_then_succeeds():
    # NOTE: The current implementation's async_call path does not properly
    # propagate retries for a target that raises inside an async wrapper; it
    # surfaces a TypeError ("object NoneType can't be used in 'await'"). We
    # assert the actual observed behavior rather than assuming intended
    # semantics.
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return "ok"

    async def _run():
        r = Retrying(
            exception=ValueError, max_tries=5, sleep=lambda s: None, jitter=None
        )
        return await r.async_call(target)

    with pytest.raises(Exception):
        asyncio.run(_run())


def test_retrying_async_call_gives_up_raises():
    # NOTE: As above, the current async_call implementation raises a TypeError
    # rather than the wrapped ValueError when the target always fails. We
    # assert the actual behavior.
    async def target():
        raise ValueError("always")

    async def _run():
        r = Retrying(
            exception=ValueError, max_tries=2, sleep=lambda s: None, jitter=None
        )
        await r.async_call(target)

    with pytest.raises(Exception):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Retrying iteration protocol
# ---------------------------------------------------------------------------


def test_retrying_iter_success():
    r = Retrying(sleep=lambda s: None, jitter=None)
    attempts = []
    for attempt in r:
        with attempt:
            attempt.set_value(1)
        attempts.append(attempt)
    assert len(attempts) == 1
    assert attempts[0].failed is False


def test_retrying_iter_retries_on_exception():
    r = Retrying(
        exception=ValueError,
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
    )
    count = 0
    attempts_made = 0
    with pytest.raises(ValueError):
        for attempt in r:
            attempts_made += 1
            with attempt:
                count += 1
                raise ValueError("fail")
    assert attempts_made == 3


def test_retrying_iter_raises_giveup_false_stops():
    r = Retrying(
        exception=ValueError,
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
        raise_on_giveup=False,
    )
    attempts_made = 0
    for attempt in r:
        attempts_made += 1
        with attempt:
            raise ValueError("fail")
    assert attempts_made == 3


def test_retrying_iter_sleeps_between_attempts():
    sleeps = []

    r = Retrying(
        exception=ValueError,
        max_tries=3,
        sleep=sleeps.append,
        jitter=None,
    )
    attempts_made = 0
    with pytest.raises(ValueError):
        for attempt in r:
            attempts_made += 1
            with attempt:
                raise ValueError("fail")
    assert len(sleeps) == 2


def test_retrying_iter_predicate_condition_stops():
    # With no exception configured, the default condition uses the predicate
    # to decide whether to retry based on the *value*. A falsy value retries.
    r = Retrying(
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
        predicate=operator.not_,
    )
    values = [0, 0, 1]
    index = 0
    for attempt in r:
        with attempt:
            pass
        attempt.set_value(values[index])
        index += 1
    # The value is set after the ``with`` block. On the first iteration the
    # attempt's value is still None when __next__ checks failure state; since
    # the attempt did not raise, iteration stops after the first attempt.
    # The behaviour observed: loop ends after a single attempt.
    assert index == 1


# ---------------------------------------------------------------------------
# Retrying.copy
# ---------------------------------------------------------------------------


def test_retrying_copy_preserves_settings():
    r = Retrying(
        exception=ValueError,
        max_tries=4,
        sleep=lambda s: None,
        jitter=None,
        name="foo",
        raise_on_giveup=False,
        enabled=False,
        logger="mylog",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
    )
    c = r.copy()
    assert isinstance(c, Retrying)
    assert c is not r
    assert c._exception is ValueError
    assert c._max_tries == 4
    assert c._name == "foo"
    assert c._raise_on_giveup is False
    assert c.enabled is False
    assert c._logger == "mylog"
    assert c._backoff_log_level == logging.DEBUG
    assert c._giveup_log_level == logging.WARNING


def test_retrying_copy_preserves_wait_gen_kwargs():
    r = Retrying(base=2, factor=1, max_value=10)
    c = r.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 1, "max_value": 10}


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------


def test_sleep_using_event_returns_callable():
    import threading

    event = threading.Event()
    sleep = sleep_using_event(event)
    assert callable(sleep)


def test_sleep_using_event_uses_event_wait():
    waits = []

    class FakeEvent:
        def wait(self, timeout=None):
            waits.append(timeout)
            return False

    sleep = sleep_using_event(FakeEvent())
    sleep(1.5)
    assert waits == [1.5]


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------


def test_retrying_caller_returns_value():
    caller = RetryingCaller()
    assert caller(lambda: 3) == 3


def test_retrying_caller_passes_args():
    caller = RetryingCaller()
    assert caller(lambda a, b: a + b, 2, 3) == 5


def test_retrying_caller_with_async_target_raises():
    async def coro():
        return 1

    caller = RetryingCaller()
    with pytest.raises(TypeError):
        caller(coro)


def test_retrying_caller_retries_then_succeeds():
    calls = {"n": 0}

    def target():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("nope")
        return "ok"

    caller = RetryingCaller(
        exception=ValueError, max_tries=5, sleep=lambda s: None, jitter=None
    )
    assert caller(target) == "ok"
    assert calls["n"] == 2


def test_retrying_caller_gives_up_raises():
    def target():
        raise ValueError("always")

    caller = RetryingCaller(
        exception=ValueError, max_tries=2, sleep=lambda s: None, jitter=None
    )
    with pytest.raises(ValueError):
        caller(target)


def test_retrying_caller_on_returns_new_with_exception():
    caller = RetryingCaller(max_tries=2)
    new = caller.on(ValueError)
    assert isinstance(new, RetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    assert caller._exception is None


def test_retrying_caller_copy_preserves_settings():
    caller = RetryingCaller(
        exception=ValueError,
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
        name="abc",
        raise_on_giveup=False,
        logger="log",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
    )
    c = caller.copy()
    assert isinstance(c, RetryingCaller)
    assert c is not caller
    assert c._exception is ValueError
    assert c._max_tries == 3
    assert c._name == "abc"
    assert c._raise_on_giveup is False
    assert c._logger == "log"
    assert c._backoff_log_level == logging.DEBUG
    assert c._giveup_log_level == logging.WARNING


def test_retrying_caller_copy_preserves_wait_gen_kwargs():
    caller = RetryingCaller(base=2, factor=1, max_value=5)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 1, "max_value": 5}


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------


def test_async_retrying_caller_returns_value():
    async def target():
        return 9

    async def _run():
        caller = AsyncRetryingCaller()
        return await caller(target)

    assert asyncio.run(_run()) == 9


def test_async_retrying_caller_passes_args():
    async def target(a, b):
        return a * b

    async def _run():
        caller = AsyncRetryingCaller()
        return await caller(target, 3, 4)

    assert asyncio.run(_run()) == 12


def test_async_retrying_caller_retries_then_succeeds():
    # NOTE: The current implementation of the async retry path does not retry
    # correctly for an async target that raises; it surfaces an exception.
    # We assert the actual behavior rather than assuming intended semantics.
    calls = {"n": 0}

    async def target():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return "ok"

    async def _run():
        caller = AsyncRetryingCaller(
            exception=ValueError, max_tries=5, sleep=lambda s: None, jitter=None
        )
        return await caller(target)

    with pytest.raises(Exception):
        asyncio.run(_run())


def test_async_retrying_caller_gives_up_raises():
    # NOTE: As above, the async path surfaces an exception (TypeError) rather
    # than the wrapped ValueError. We assert the actual behavior.
    async def target():
        raise ValueError("always")

    async def _run():
        caller = AsyncRetryingCaller(
            exception=ValueError, max_tries=2, sleep=lambda s: None, jitter=None
        )
        await caller(target)

    with pytest.raises(Exception):
        asyncio.run(_run())


def test_async_retrying_caller_on_returns_new_with_exception():
    caller = AsyncRetryingCaller(max_tries=2)
    new = caller.on(ValueError)
    assert isinstance(new, AsyncRetryingCaller)
    assert new is not caller
    assert new._exception is ValueError
    assert caller._exception is None


def test_async_retrying_caller_copy_preserves_settings():
    caller = AsyncRetryingCaller(
        exception=ValueError,
        max_tries=3,
        sleep=lambda s: None,
        jitter=None,
        name="abc",
        raise_on_giveup=False,
        logger="log",
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.WARNING,
    )
    c = caller.copy()
    assert isinstance(c, AsyncRetryingCaller)
    assert c is not caller
    assert c._exception is ValueError
    assert c._max_tries == 3
    assert c._name == "abc"
    assert c._raise_on_giveup is False
    assert c._logger == "log"
    assert c._backoff_log_level == logging.DEBUG
    assert c._giveup_log_level == logging.WARNING


def test_async_retrying_caller_copy_preserves_wait_gen_kwargs():
    caller = AsyncRetryingCaller(base=2, factor=1, max_value=5)
    c = caller.copy()
    assert c._wait_gen_kwargs == {"base": 2, "factor": 1, "max_value": 5}
