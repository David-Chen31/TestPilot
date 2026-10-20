import asyncio
import logging
import operator
import threading
from unittest import mock

import pytest

from backon._retry._classes import (
    Retrying,
    RetryingCaller,
    AsyncRetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)
from backon._rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------

class TestRetryAttempt:
    def test_enter_returns_self(self):
        a = _RetryAttempt()
        with a as got:
            assert got is a

    def test_success_no_exception(self):
        a = _RetryAttempt()
        with a:
            a.set_value(42)
        assert a.failed is False
        assert a.exception is None
        assert a.value == 42

    def test_exception_is_swallowed(self):
        a = _RetryAttempt()
        with a:
            raise ValueError("boom")
        assert a.failed is True
        assert isinstance(a.exception, ValueError)
        assert str(a.exception) == "boom"
        assert a.value is None

    def test_base_exception_not_swallowed(self):
        a = _RetryAttempt()
        with pytest.raises(KeyboardInterrupt):
            with a:
                raise KeyboardInterrupt()
        assert a.failed is False
        assert a.exception is None

    def test_exception_propagates_if_not_exception_subclass(self):
        a = _RetryAttempt()
        with pytest.raises(SystemExit):
            with a:
                raise SystemExit(1)
        assert a.failed is False


# ---------------------------------------------------------------------------
# Retrying: basic properties / defaults
# ---------------------------------------------------------------------------

class TestRetryingProperties:
    def test_default_statistics_empty(self):
        r = Retrying()
        assert r.statistics == {}
        assert r.call_state is None

    def test_enabled_default_true(self):
        r = Retrying()
        assert r.enabled is True

    def test_enabled_setter(self):
        r = Retrying()
        r.enabled = False
        assert r.enabled is False
        r.enabled = True
        assert r.enabled is True

    def test_context_manager_sync(self):
        r = Retrying()
        with r as got:
            assert got is r

    def test_context_manager_async(self):
        r = Retrying()

        async def go():
            async with r as got:
                assert got is r

        asyncio.run(go())


# ---------------------------------------------------------------------------
# Retrying.call (sync)
# ---------------------------------------------------------------------------

def no_sleep(_seconds):
    """No-op sleep function for tests."""
    return None


class TestRetryingCall:
    def test_returns_value_immediately(self):
        r = Retrying(sleep=no_sleep)
        assert r.call(lambda: 7) == 7

    def test_passes_args_kwargs(self):
        r = Retrying(sleep=no_sleep)

        def target(a, b, c=0):
            return a + b + c

        assert r.call(target, 1, 2, c=3) == 6

    def test_retries_on_exception_then_succeeds(self):
        calls = {"n": 0}
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=5)

        def target():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("fail")
            return "ok"

        assert r.call(target) == "ok"
        assert calls["n"] == 3

    def test_gives_up_and_raises(self):
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=3)

        def target():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            r.call(target)

    def test_raise_on_giveup_false_returns_none(self):
        r = Retrying(
            exception=ValueError,
            sleep=no_sleep,
            max_tries=2,
            raise_on_giveup=False,
        )

        def target():
            raise ValueError("always")

        # With raise_on_giveup False, retry_error_callback None -> returns None
        assert r.call(target) is None

    def test_retry_error_callback_used(self):
        r = Retrying(
            exception=ValueError,
            sleep=no_sleep,
            max_tries=2,
            retry_error_callback=lambda details: "gaveup",
        )

        def target():
            raise ValueError("always")

        assert r.call(target) == "gaveup"

    def test_async_function_raises_typeerror(self):
        r = Retrying(sleep=no_sleep)

        async def coro():
            return 1

        with pytest.raises(TypeError):
            r.call(coro)

    def test_statistics_populated_after_call(self):
        r = Retrying(sleep=no_sleep)

        def target():
            return 5

        r.call(target)
        stats = r.statistics
        assert stats["attempt_number"] == 1

    def test_holder_sets_state_and_call_state(self):
        r = Retrying(sleep=no_sleep)
        r.call(lambda: 1)
        # Both should now be set from the holder
        assert r._state is not None
        assert r.call_state is not None

    def test_sleep_called_between_attempts(self):
        sleeps = []
        calls = {"n": 0}

        def sleep_fn(s):
            sleeps.append(s)

        def target():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("x")
            return 1

        r = Retrying(exception=ValueError, sleep=sleep_fn, max_tries=5)
        r.call(target)
        # Two backoffs occurred
        assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# Retrying.async_call
# ---------------------------------------------------------------------------

class TestRetryingAsyncCall:
    def test_returns_value(self):
        r = Retrying(sleep=no_sleep)

        async def target():
            return 9

        assert asyncio.run(r.async_call(target)) == 9

    def test_passes_args_kwargs(self):
        r = Retrying(sleep=no_sleep)

        async def target(a, b=0):
            return a + b

        assert asyncio.run(r.async_call(target, 3, b=4)) == 7

    def test_retries_then_succeeds(self):
        calls = {"n": 0}
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=5)

        async def target():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("f")
            return "ok"

        assert asyncio.run(r.async_call(target)) == "ok"
        assert calls["n"] == 3

    def test_gives_up(self):
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=2)

        async def target():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            asyncio.run(r.async_call(target))

    def test_statistics_populated(self):
        r = Retrying(sleep=no_sleep)

        async def target():
            return 1

        asyncio.run(r.async_call(target))
        assert r.call_state is not None
        assert r._state is not None


# ---------------------------------------------------------------------------
# Retrying.copy
# ---------------------------------------------------------------------------

class TestRetryingCopy:
    def test_copy_preserves_configuration(self):
        handler = mock.Mock()
        cond = mock.Mock()
        stop = mock.Mock()
        rl = RateLimiter(5)
        r = Retrying(
            exception=ValueError,
            max_tries=4,
            max_time=1.0,
            jitter=None,
            giveup=lambda e: False,
            condition=cond,
            stop=stop,
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            retry_error_callback=lambda d: 1,
            raise_on_giveup=False,
            logger="mylogger",
            backoff_log_level=logging.DEBUG,
            giveup_log_level=logging.WARNING,
            sleep=no_sleep,
            enabled=False,
            name="myname",
            before=handler,
            after=handler,
            rate_limit=rl,
            attempt_timeout=2.0,
            base=3,  # wait_gen_kwargs
        )
        c = r.copy()
        assert isinstance(c, Retrying)
        assert c is not r
        assert c._exception is ValueError
        assert c._max_tries == 4
        assert c._max_time == 1.0
        assert c._jitter is None
        assert c._condition is cond
        assert c._stop is stop
        assert c._on_success is handler
        assert c._raise_on_giveup is False
        assert c._logger == "mylogger"
        assert c._backoff_log_level == logging.DEBUG
        assert c._giveup_log_level == logging.WARNING
        assert c._sleep is no_sleep
        assert c._enabled is False
        assert c._name == "myname"
        assert c._rate_limit is rl
        assert c._attempt_timeout == 2.0
        assert c._wait_gen_kwargs == {"base": 3}

    def test_copy_independent_enabled(self):
        r = Retrying()
        c = r.copy()
        c.enabled = False
        assert r.enabled is True


# ---------------------------------------------------------------------------
# Iteration protocol (__iter__ / __next__)
# ---------------------------------------------------------------------------

class TestRetryingIteration:
    def test_success_stops_iteration(self):
        r = Retrying(sleep=no_sleep, max_tries=5, exception=ValueError)
        count = 0
        for attempt in r:
            count += 1
            attempt.set_value(1)
        assert count == 1

    def test_exception_swallowed_and_retried(self):
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=3)
        attempts = 0
        with pytest.raises(ValueError):
            for attempt in r:
                attempts += 1
                raise ValueError("x")
        assert attempts == 3

    def test_raise_on_giveup_false_breaks(self):
        r = Retrying(
            exception=ValueError,
            sleep=no_sleep,
            max_tries=2,
            raise_on_giveup=False,
        )
        attempts = 0
        for attempt in r:
            attempts += 1
            raise ValueError("x")
        assert attempts == 2

    def test_retriable_then_success(self):
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=5)
        attempts = 0
        succeeded = False
        for attempt in r:
            attempts += 1
            if attempts < 3:
                raise ValueError("x")
            attempt.set_value("ok")
            succeeded = True
        assert attempts == 3
        assert succeeded is True

    def test_iteration_initializes_state(self):
        r = Retrying(exception=ValueError, sleep=no_sleep, max_tries=2)
        it = iter(r)
        assert it is r
        assert r._iter_state is not None

    def test_iteration_sleeps_between_attempts(self):
        sleeps = []
        r = Retrying(
            exception=ValueError,
            sleep=lambda s: sleeps.append(s),
            max_tries=3,
            jitter=None,
        )
        attempts = 0
        with pytest.raises(ValueError):
            for attempt in r:
                attempts += 1
                raise ValueError("x")
        assert attempts == 3
        assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------

class TestSleepUsingEvent:
    def test_waits_with_timeout(self):
        event = mock.Mock()
        sleep = sleep_using_event(event)
        sleep(1.5)
        event.wait.assert_called_once_with(timeout=1.5)

    def test_zero_timeout(self):
        event = mock.Mock()
        sleep = sleep_using_event(event)
        sleep(0)
        event.wait.assert_called_once_with(timeout=0)

    def test_uses_real_event(self):
        event = threading.Event()
        event.set()
        sleep = sleep_using_event(event)
        # Should return immediately since event set
        sleep(0.01)


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------

class TestRetryingCaller:
    def test_call_returns_value(self):
        caller = RetryingCaller(sleep=no_sleep)
        assert caller(lambda: 11) == 11

    def test_passes_args(self):
        caller = RetryingCaller(sleep=no_sleep)
        assert caller(lambda a, b: a + b, 2, 3) == 5

    def test_retries_then_succeeds(self):
        calls = {"n": 0}
        caller = RetryingCaller(exception=ValueError, sleep=no_sleep, max_tries=5)

        def target():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("x")
            return "ok"

        assert caller(target) == "ok"
        assert calls["n"] == 2

    def test_gives_up_raises(self):
        caller = RetryingCaller(exception=ValueError, sleep=no_sleep, max_tries=2)

        def target():
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            caller(target)

    def test_async_function_raises(self):
        caller = RetryingCaller(sleep=no_sleep)

        async def coro():
            return 1

        with pytest.raises(TypeError):
            caller(coro)

    def test_on_returns_new_with_exception(self):
        caller = RetryingCaller(sleep=no_sleep)
        new = caller.on(ValueError)
        assert isinstance(new, RetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        assert caller._exception is None

    def test_on_returns_retrying_caller_type(self):
        caller = RetryingCaller()
        new = caller.on([ValueError, TypeError])
        assert new._exception == [ValueError, TypeError]

    def test_copy_preserves_config(self):
        handler = mock.Mock()
        rl = RateLimiter(3)
        caller = RetryingCaller(
            exception=ValueError,
            max_tries=5,
            max_time=2.0,
            jitter=None,
            condition=mock.Mock(),
            stop=mock.Mock(),
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            retry_error_callback=lambda d: 1,
            raise_on_giveup=False,
            logger="logger",
            backoff_log_level=logging.DEBUG,
            giveup_log_level=logging.WARNING,
            sleep=no_sleep,
            name="nm",
            before=handler,
            after=handler,
            rate_limit=rl,
            attempt_timeout=1.5,
            base=2,
        )
        c = caller.copy()
        assert isinstance(c, RetryingCaller)
        assert c is not caller
        assert c._exception is ValueError
        assert c._max_tries == 5
        assert c._max_time == 2.0
        assert c._jitter is None
        assert c._on_success is handler
        assert c._raise_on_giveup is False
        assert c._logger == "logger"
        assert c._backoff_log_level == logging.DEBUG
        assert c._giveup_log_level == logging.WARNING
        assert c._sleep is no_sleep
        assert c._name == "nm"
        assert c._rate_limit is rl
        assert c._attempt_timeout == 1.5
        assert c._wait_gen_kwargs == {"base": 2}


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------

class TestAsyncRetryingCaller:
    def test_returns_value(self):
        caller = AsyncRetryingCaller(sleep=no_sleep)

        async def target():
            return 21

        assert asyncio.run(caller(target)) == 21

    def test_passes_args(self):
        caller = AsyncRetryingCaller(sleep=no_sleep)

        async def target(a, b=0):
            return a + b

        assert asyncio.run(caller(target, 1, b=2)) == 3

    def test_retries_then_succeeds(self):
        calls = {"n": 0}
        caller = AsyncRetryingCaller(
            exception=ValueError, sleep=no_sleep, max_tries=5
        )

        async def target():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("x")
            return "ok"

        assert asyncio.run(caller(target)) == "ok"
        assert calls["n"] == 2

    def test_gives_up_raises(self):
        caller = AsyncRetryingCaller(
            exception=ValueError, sleep=no_sleep, max_tries=2
        )

        async def target():
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            asyncio.run(caller(target))

    def test_on_returns_new_caller(self):
        caller = AsyncRetryingCaller(sleep=no_sleep)
        new = caller.on(ValueError)
        assert isinstance(new, AsyncRetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        assert caller._exception is None

    def test_copy_preserves_config(self):
        handler = mock.Mock()
        rl = RateLimiter(2)
        caller = AsyncRetryingCaller(
            exception=TypeError,
            max_tries=3,
            max_time=1.0,
            jitter=None,
            condition=mock.Mock(),
            stop=mock.Mock(),
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            retry_error_callback=lambda d: 1,
            raise_on_giveup=False,
            logger="lg",
            backoff_log_level=logging.DEBUG,
            giveup_log_level=logging.WARNING,
            sleep=no_sleep,
            name="name",
            before=handler,
            after=handler,
            rate_limit=rl,
            attempt_timeout=3.0,
            base=4,
        )
        c = caller.copy()
        assert isinstance(c, AsyncRetryingCaller)
        assert c is not caller
        assert c._exception is TypeError
        assert c._max_tries == 3
        assert c._max_time == 1.0
        assert c._jitter is None
        assert c._on_success is handler
        assert c._raise_on_giveup is False
        assert c._logger == "lg"
        assert c._backoff_log_level == logging.DEBUG
        assert c._giveup_log_level == logging.WARNING
        assert c._sleep is no_sleep
        assert c._name == "name"
        assert c._rate_limit is rl
        assert c._attempt_timeout == 3.0
        assert c._wait_gen_kwargs == {"base": 4}


# ---------------------------------------------------------------------------
# Constructor defaults shared by all three classes
# ---------------------------------------------------------------------------

class TestConstructorDefaults:
    @pytest.mark.parametrize("cls", [Retrying, RetryingCaller, AsyncRetryingCaller])
    def test_defaults(self, cls):
        obj = cls()
        assert obj._exception is None
        assert obj._predicate is operator.not_
        assert obj._max_tries is None
        assert obj._max_time is None
        assert obj._condition is None
        assert obj._stop is None
        assert obj._raise_on_giveup is True
        assert obj._logger == "backon"
        assert obj._backoff_log_level == logging.INFO
        assert obj._giveup_log_level == logging.ERROR
        assert obj._sleep is None
        assert obj._name == ""
        assert obj._rate_limit is None
        assert obj._attempt_timeout is None
        assert obj._wait_gen_kwargs == {}

    @pytest.mark.parametrize("cls", [Retrying, RetryingCaller, AsyncRetryingCaller])
    def test_wait_gen_kwargs_collected(self, cls):
        obj = cls(base=2, factor=3.0)
        assert obj._wait_gen_kwargs == {"base": 2, "factor": 3.0}
