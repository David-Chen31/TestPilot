import asyncio
import logging
import operator
import threading

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ListWaitGen:
    """A wait generator object exposing .next() (as _next_wait expects)."""

    def __init__(self, values):
        self._values = list(values)
        self._index = 0

    def next(self, *args, **kwargs):
        if self._index >= len(self._values):
            value = self._values[-1]
        else:
            value = self._values[self._index]
            self._index += 1
        return value


def _make_wait_gen_factory(values):
    """Return a callable suitable as a ``wait_gen`` argument.

    ``_init_wait_gen`` calls ``wait_gen(**kwargs)``, so the factory must
    accept kwargs and return an object with a ``.next`` method.
    """

    def factory(**kwargs):
        return _ListWaitGen(values)

    return factory


async def _async_noop_sleep(seconds):
    return None


# ---------------------------------------------------------------------------
# _RetryAttempt
# ---------------------------------------------------------------------------


class TestRetryAttempt:
    def test_initial_state(self):
        attempt = _RetryAttempt()
        assert attempt.failed is False
        assert attempt.exception is None
        assert attempt.value is None

    def test_set_value(self):
        attempt = _RetryAttempt()
        attempt.set_value(42)
        assert attempt.value == 42
        assert attempt.failed is False

    def test_enter_returns_self(self):
        attempt = _RetryAttempt()
        assert attempt.__enter__() is attempt

    def test_exit_no_exception(self):
        attempt = _RetryAttempt()
        assert attempt.__exit__(None, None, None) is False
        assert attempt.failed is False

    def test_exit_captures_exception(self):
        attempt = _RetryAttempt()
        exc = ValueError("boom")
        result = attempt.__exit__(ValueError, exc, None)
        assert result is True
        assert attempt.failed is True
        assert attempt.exception is exc

    def test_exit_does_not_capture_base_exception(self):
        attempt = _RetryAttempt()
        exc = KeyboardInterrupt()
        result = attempt.__exit__(KeyboardInterrupt, exc, None)
        assert result is False
        assert attempt.failed is False

    def test_context_manager_swallows_exception(self):
        attempt = _RetryAttempt()
        with attempt:
            raise RuntimeError("oops")
        assert attempt.failed is True
        assert isinstance(attempt.exception, RuntimeError)


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------


class TestSleepUsingEvent:
    def test_calls_event_wait_with_timeout(self):
        calls = []

        class FakeEvent:
            def wait(self, timeout=None):
                calls.append(timeout)
                return True

        sleep = sleep_using_event(FakeEvent())
        sleep(1.5)
        assert calls == [1.5]

    def test_returns_zero_wait(self):
        calls = []

        class FakeEvent:
            def wait(self, timeout=None):
                calls.append(timeout)

        sleep_using_event(FakeEvent())(0)
        assert calls == [0]

    def test_real_event_returns_immediately_when_set(self):
        event = threading.Event()
        event.set()
        sleep = sleep_using_event(event)
        # No exception, returns quickly since event already set.
        assert sleep(5.0) is None


# ---------------------------------------------------------------------------
# Retrying - construction & simple properties
# ---------------------------------------------------------------------------


class TestRetryingConstruction:
    def test_defaults(self):
        r = Retrying()
        assert r.enabled is True
        assert r._name == ""
        assert r._raise_on_giveup is True
        assert r._jitter is not None
        assert r.statistics == {}
        assert r.call_state is None

    def test_default_logger(self):
        r = Retrying()
        assert r._logger == "backon"

    def test_default_handlers_are_none(self):
        r = Retrying()
        assert r._on_success is None
        assert r._on_backoff is None
        assert r._on_giveup is None
        assert r._on_attempt is None
        assert r._before_sleep is None
        assert r._retry_error_callback is None
        assert r._rate_limit is None
        assert r._condition is None
        assert r._stop is None

    def test_default_giveup(self):
        r = Retrying()
        # default giveup is a callable returning False
        assert callable(r._giveup)
        assert r._giveup(ValueError("x")) is False

    def test_stores_config_values(self):
        def handler(state):
            return None

        r = Retrying(
            giveup=lambda e: True,
            condition="COND",
            stop="STOP",
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            retry_error_callback="CB",
            rate_limit="RL",
        )
        assert r._giveup is not None
        assert r._giveup(ValueError()) is True
        assert r._condition == "COND"
        assert r._stop == "STOP"
        assert r._on_backoff is handler
        assert r._on_giveup is handler
        assert r._on_attempt is handler
        assert r._before_sleep is handler
        assert r._retry_error_callback == "CB"
        assert r._rate_limit == "RL"

    def test_enabled_setter(self):
        r = Retrying()
        r.enabled = False
        assert r.enabled is False
        r.enabled = True
        assert r.enabled is True

    def test_sync_context_manager(self):
        r = Retrying()
        with r as ctx:
            assert ctx is r
        # __exit__ returns None (falsy)
        assert r.__exit__(None, None, None) is None

    def test_async_context_manager(self):
        r = Retrying()

        async def go():
            async with r as ctx:
                assert ctx is r

        asyncio.run(go())

    def test_statistics_empty_when_no_state(self):
        assert Retrying().statistics == {}


# ---------------------------------------------------------------------------
# Retrying - call
# ---------------------------------------------------------------------------


class TestRetryingCall:
    def test_success_first_try(self):
        calls = []

        def target():
            calls.append(1)
            return "ok"

        r = Retrying(sleep=lambda s: None)
        assert r.call(target) == "ok"
        assert calls == [1]
        assert r.call_state is not None
        assert r._state is not None

    def test_passes_args_kwargs(self):
        got = {}

        def target(a, b=None):
            got["a"] = a
            got["b"] = b
            return a + b

        r = Retrying(sleep=lambda s: None)
        assert r.call(target, 3, b=4) == 7
        assert got == {"a": 3, "b": 4}

    def test_retries_until_success(self):
        calls = []

        def target():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        r = Retrying(max_tries=5, exception=ValueError, sleep=lambda s: None)
        assert r.call(target) == "done"
        assert len(calls) == 3

    def test_raises_after_giveup(self):
        calls = []

        def target():
            calls.append(1)
            raise ValueError("always fail")

        r = Retrying(max_tries=2, exception=ValueError, sleep=lambda s: None)
        with pytest.raises(ValueError, match="always fail"):
            r.call(target)
        assert len(calls) == 2

    def test_raise_on_giveup_false(self):
        def target():
            raise ValueError("fail")

        r = Retrying(
            max_tries=2,
            exception=ValueError,
            raise_on_giveup=False,
            sleep=lambda s: None,
        )
        # Returns None instead of raising
        assert r.call(target) is None

    def test_async_function_raises_type_error(self):
        async def target():
            return 1

        r = Retrying()
        with pytest.raises(TypeError, match="Use await with Retrying.call"):
            r.call(target)

    def test_sleep_called_between_attempts(self):
        sleeps = []

        def target():
            raise ValueError("x")

        wait_gen = _make_wait_gen_factory([0.5, 0.5, 0.5, 0.5])

        r = Retrying(
            wait_gen,
            max_tries=3,
            exception=ValueError,
            jitter=None,
            sleep=lambda s: sleeps.append(s),
            raise_on_giveup=False,
        )
        r.call(target)
        # 2 sleeps between 3 attempts
        assert sleeps == [0.5, 0.5]

    def test_zero_sleep_not_called(self):
        sleeps = []

        def target():
            raise ValueError("x")

        wait_gen = _make_wait_gen_factory([0, 0, 0])

        r = Retrying(
            wait_gen,
            max_tries=2,
            exception=ValueError,
            jitter=None,
            sleep=lambda s: sleeps.append(s),
            raise_on_giveup=False,
        )
        r.call(target)
        assert sleeps == []

    def test_call_sets_call_state(self):
        def target():
            return 1

        r = Retrying(sleep=lambda s: None)
        r.call(target)
        assert r.call_state is not None
        assert "attempt_number" in r.call_state.statistics
        assert "start_time" in r.statistics

    def test_no_retry_on_unmatched_exception(self):
        calls = []

        def target():
            calls.append(1)
            raise KeyError("nope")

        r = Retrying(max_tries=5, exception=ValueError, sleep=lambda s: None)
        with pytest.raises(KeyError):
            r.call(target)
        assert len(calls) == 1

    def test_predicate_on_return_value(self):
        calls = []

        def target():
            calls.append(1)
            return None

        r = Retrying(
            max_tries=3,
            predicate=operator.not_,
            sleep=lambda s: None,
            raise_on_giveup=False,
        )
        result = r.call(target)
        # predicate treats None as failure for a while; eventually gives up.
        assert len(calls) == 3
        assert result is None

    def test_custom_giveup_handler_receives_state(self):
        seen = []

        def on_giveup(state):
            seen.append(state)

        def target():
            raise ValueError("x")

        r = Retrying(
            max_tries=1,
            exception=ValueError,
            on_giveup=on_giveup,
            sleep=lambda s: None,
        )
        with pytest.raises(ValueError):
            r.call(target)
        assert len(seen) == 1

    def test_on_backoff_handler_called(self):
        seen = []

        def on_backoff(state):
            seen.append(state)

        def target():
            raise ValueError("x")

        r = Retrying(
            max_tries=2,
            exception=ValueError,
            on_backoff=on_backoff,
            sleep=lambda s: None,
            raise_on_giveup=False,
        )
        r.call(target)
        assert len(seen) == 1

    def test_on_attempt_handler_called(self):
        seen = []

        def on_attempt(state):
            seen.append(state)

        def target():
            return 1

        r = Retrying(on_attempt=on_attempt, sleep=lambda s: None)
        r.call(target)
        assert len(seen) == 1

    def test_before_sleep_handler_called(self):
        seen = []

        def before_sleep(state):
            seen.append(state)

        def target():
            raise ValueError("x")

        r = Retrying(
            max_tries=2,
            exception=ValueError,
            before_sleep=before_sleep,
            sleep=lambda s: None,
            raise_on_giveup=False,
        )
        r.call(target)
        assert len(seen) == 1

    def test_retry_error_callback_used(self):
        def cb(details):
            return "cb-result"

        def target():
            raise ValueError("x")

        r = Retrying(
            max_tries=1,
            exception=ValueError,
            retry_error_callback=cb,
            raise_on_giveup=False,
            sleep=lambda s: None,
        )
        assert r.call(target) == "cb-result"

    def test_before_after_handlers(self):
        before_calls = []
        after_calls = []

        def before(state):
            before_calls.append(state)

        def after(state):
            after_calls.append(state)

        def target():
            return "ok"

        r = Retrying(before=before, after=after, sleep=lambda s: None)
        assert r.call(target) == "ok"
        assert len(before_calls) == 1
        assert len(after_calls) == 1

    def test_on_success_handler(self):
        seen = []

        def on_success(state):
            seen.append(state)

        def target():
            return "yes"

        r = Retrying(on_success=on_success, sleep=lambda s: None)
        assert r.call(target) == "yes"
        assert len(seen) == 1


# ---------------------------------------------------------------------------
# Retrying - async_call
# ---------------------------------------------------------------------------


class TestRetryingAsyncCall:
    def test_async_success(self):
        async def target():
            return "hello"

        r = Retrying(sleep=_async_noop_sleep)
        assert asyncio.run(r.async_call(target)) == "hello"
        assert r.call_state is not None

    def test_async_retries(self):
        calls = []

        async def target():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("again")
            return "done"

        r = Retrying(
            max_tries=5, exception=ValueError, sleep=_async_noop_sleep
        )
        assert asyncio.run(r.async_call(target)) == "done"
        assert len(calls) == 3

    def test_async_giveup(self):
        async def target():
            raise ValueError("boom")

        r = Retrying(
            max_tries=2, exception=ValueError, sleep=_async_noop_sleep
        )
        with pytest.raises(ValueError, match="boom"):
            asyncio.run(r.async_call(target))

    def test_async_passes_args_kwargs(self):
        async def target(a, b):
            return a * b

        r = Retrying(sleep=_async_noop_sleep)
        assert asyncio.run(r.async_call(target, 3, b=4)) == 12

    def test_async_sets_call_state(self):
        async def target():
            return 1

        r = Retrying(sleep=_async_noop_sleep)
        asyncio.run(r.async_call(target))
        assert r.call_state is not None
        assert "attempt_number" in r.statistics


# ---------------------------------------------------------------------------
# Retrying - iteration protocol
# ---------------------------------------------------------------------------


class TestRetryingIteration:
    def test_successful_iteration_stops(self):
        r = Retrying(sleep=lambda s: None)
        attempts = []
        for attempt in r:
            attempts.append(attempt)
            attempt.set_value("ok")
        assert len(attempts) == 1
        assert attempts[0].value == "ok"
        assert r._iter_state.tries == 1

    def test_iter_state_default_target_is_retrying(self):
        r = Retrying(sleep=lambda s: None)
        it = iter(r)
        assert it is r
        assert r._iter_state.target == "Retrying"

    def test_iter_state_custom_name(self):
        r = Retrying(name="myname", sleep=lambda s: None)
        iter(r)
        assert r._iter_state.target == "myname"

    def test_failure_without_retry_raises(self):
        r = Retrying(max_tries=1, exception=ValueError, sleep=lambda s: None)
        with pytest.raises(ValueError, match="fail"):
            for attempt in r:
                with attempt:
                    raise ValueError("fail")

    def test_retries_until_success(self):
        r = Retrying(max_tries=5, exception=ValueError, sleep=lambda s: None)
        attempts = []
        for attempt in r:
            attempts.append(attempt)
            with attempt:
                if len(attempts) < 3:
                    raise ValueError("not yet")
        assert len(attempts) == 3
        assert attempts[-1].failed is False

    def test_no_raise_on_giveup_stops(self):
        r = Retrying(
            max_tries=2,
            exception=ValueError,
            raise_on_giveup=False,
            sleep=lambda s: None,
        )
        attempts = []
        for attempt in r:
            attempts.append(attempt)
            with attempt:
                raise ValueError("fail")
        assert len(attempts) == 2

    def test_iteration_sleeps_between(self):
        sleeps = []

        wait_gen = _make_wait_gen_factory([0.25, 0.25, 0.25])

        r = Retrying(
            wait_gen,
            max_tries=2,
            exception=ValueError,
            jitter=None,
            sleep=lambda s: sleeps.append(s),
            raise_on_giveup=False,
        )
        for attempt in r:
            with attempt:
                raise ValueError("fail")
        assert sleeps == [0.25]

    def test_iteration_custom_condition_and_stop(self):
        condition_calls = []
        stop_calls = []

        def condition(state):
            condition_calls.append(state)
            return True

        def stop(state):
            stop_calls.append(state)
            return state.tries >= 2

        # With raise_on_giveup=False, reaching the stop condition triggers
        # StopIteration rather than re-raising the captured exception.
        r = Retrying(
            condition=condition,
            stop=stop,
            sleep=lambda s: None,
            raise_on_giveup=False,
        )
        attempts = []
        for attempt in r:
            attempts.append(attempt)
            with attempt:
                raise ValueError("fail")
        # attempt 1 (tries=1), attempt 2 evaluates condition+stop at
        # tries=1 (stop False), attempt 3 evaluates condition+stop at
        # tries=2 (stop True -> StopIteration) before yielding.
        assert len(attempts) == 2
        assert len(condition_calls) == 2
        assert len(stop_calls) == 2
        # Stop was evaluated with tries reaching the threshold.
        assert stop_calls[-1].tries == 2

    def test_iteration_custom_stop_raises_on_giveup_by_default(self):
        def stop(state):
            return state.tries >= 1

        r = Retrying(stop=stop, sleep=lambda s: None)

        with pytest.raises(ValueError, match="boom"):
            for attempt in r:
                with attempt:
                    raise ValueError("boom")

    def test_iteration_tries_increments(self):
        r = Retrying(sleep=lambda s: None)
        for attempt in r:
            attempt.set_value(1)
        assert r._iter_state.tries == 1


# ---------------------------------------------------------------------------
# Retrying - copy
# ---------------------------------------------------------------------------


class TestRetryingCopy:
    def test_copy_preserves_config(self):
        def handler(state):
            return None

        r = Retrying(
            max_tries=5,
            max_time=10,
            exception=ValueError,
            predicate=operator.not_,
            raise_on_giveup=False,
            logger="test-logger",
            backoff_log_level=logging.WARNING,
            giveup_log_level=logging.CRITICAL,
            name="my-retry",
            enabled=False,
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            before=handler,
            after=handler,
            attempt_timeout=1.5,
            multiplier=2,
        )
        c = r.copy()
        assert isinstance(c, Retrying)
        assert c is not r
        assert c._max_tries == 5
        assert c._max_time == 10
        assert c._exception is ValueError
        assert c._predicate is operator.not_
        assert c._raise_on_giveup is False
        assert c._logger == "test-logger"
        assert c._backoff_log_level == logging.WARNING
        assert c._giveup_log_level == logging.CRITICAL
        assert c._name == "my-retry"
        assert c._enabled is False
        assert c._on_success is handler
        assert c._on_backoff is handler
        assert c._on_giveup is handler
        assert c._on_attempt is handler
        assert c._before_sleep is handler
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 1.5
        assert c._wait_gen_kwargs == {"multiplier": 2}

    def test_copy_preserves_condition_and_stop(self):
        cond = lambda s: True
        stp = lambda s: True
        r = Retrying(condition=cond, stop=stp)
        c = r.copy()
        assert c._condition is cond
        assert c._stop is stp

    def test_copy_default_logger(self):
        r = Retrying()
        c = r.copy()
        assert c._logger == "backon"

    def test_copy_is_independent(self):
        r = Retrying(max_tries=3)
        c = r.copy()
        c.enabled = False
        assert r.enabled is True


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------


class TestRetryingCaller:
    def test_success(self):
        def target():
            return 5

        caller = RetryingCaller(sleep=lambda s: None)
        assert caller(target) == 5

    def test_retries_then_succeeds(self):
        calls = []

        def target():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("again")
            return "ok"

        caller = RetryingCaller(
            max_tries=3, exception=ValueError, sleep=lambda s: None
        )
        assert caller(target) == "ok"
        assert len(calls) == 2

    def test_raises_after_giveup(self):
        def target():
            raise ValueError("fail")

        caller = RetryingCaller(
            max_tries=2, exception=ValueError, sleep=lambda s: None
        )
        with pytest.raises(ValueError, match="fail"):
            caller(target)

    def test_raise_on_giveup_false(self):
        def target():
            raise ValueError("fail")

        caller = RetryingCaller(
            max_tries=2,
            exception=ValueError,
            raise_on_giveup=False,
            sleep=lambda s: None,
        )
        assert caller(target) is None

    def test_async_target_raises_type_error(self):
        async def target():
            return 1

        caller = RetryingCaller()
        with pytest.raises(TypeError, match="Use AsyncRetryingCaller"):
            caller(target)

    def test_passes_args_kwargs(self):
        def target(a, b):
            return a - b

        caller = RetryingCaller(sleep=lambda s: None)
        assert caller(target, 10, b=3) == 7

    def test_on_returns_new_caller(self):
        caller = RetryingCaller(max_tries=1)
        new = caller.on(ValueError)
        assert isinstance(new, RetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        # original untouched
        assert caller._exception is None

    def test_on_accepts_sequence(self):
        caller = RetryingCaller()
        new = caller.on([ValueError, KeyError])
        assert new._exception == [ValueError, KeyError]

    def test_on_preserves_config(self):
        caller = RetryingCaller(max_tries=7, name="caller-name", sleep=lambda s: None)
        new = caller.on(ValueError)
        assert new._max_tries == 7
        assert new._name == "caller-name"

    def test_on_handler_called(self):
        seen = []

        def on_backoff(state):
            seen.append(state)

        def target():
            raise ValueError("x")

        caller = RetryingCaller(
            max_tries=2,
            exception=ValueError,
            on_backoff=on_backoff,
            sleep=lambda s: None,
            raise_on_giveup=False,
        )
        caller(target)
        assert len(seen) == 1

    def test_on_success_handler(self):
        seen = []

        def on_success(state):
            seen.append(state)

        def target():
            return 1

        caller = RetryingCaller(on_success=on_success, sleep=lambda s: None)
        caller(target)
        assert len(seen) == 1

    def test_default_logger(self):
        caller = RetryingCaller()
        assert caller._logger == "backon"

    def test_copy(self):
        def handler(state):
            return None

        caller = RetryingCaller(
            max_tries=4,
            max_time=8,
            exception=ValueError,
            predicate=operator.not_,
            raise_on_giveup=False,
            logger="lg",
            backoff_log_level=logging.WARNING,
            giveup_log_level=logging.CRITICAL,
            sleep=lambda s: None,
            name="nm",
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            before=handler,
            after=handler,
            attempt_timeout=2.0,
            multiplier=3,
        )
        c = caller.copy()
        assert isinstance(c, RetryingCaller)
        assert c is not caller
        assert c._max_tries == 4
        assert c._max_time == 8
        assert c._exception is ValueError
        assert c._predicate is operator.not_
        assert c._raise_on_giveup is False
        assert c._logger == "lg"
        assert c._backoff_log_level == logging.WARNING
        assert c._giveup_log_level == logging.CRITICAL
        assert c._name == "nm"
        assert c._on_success is handler
        assert c._on_backoff is handler
        assert c._on_giveup is handler
        assert c._on_attempt is handler
        assert c._before_sleep is handler
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 2.0
        assert c._wait_gen_kwargs == {"multiplier": 3}

    def test_copy_is_independent(self):
        caller = RetryingCaller(max_tries=1)
        c = caller.copy()
        c._max_tries = 99
        assert caller._max_tries == 1

    def test_copy_default_logger(self):
        caller = RetryingCaller()
        c = caller.copy()
        assert c._logger == "backon"


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------


class TestAsyncRetryingCaller:
    def test_success(self):
        async def target():
            return "hi"

        caller = AsyncRetryingCaller(sleep=_async_noop_sleep)
        assert asyncio.run(caller(target)) == "hi"

    def test_retries_then_succeeds(self):
        calls = []

        async def target():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("again")
            return "ok"

        caller = AsyncRetryingCaller(
            max_tries=5, exception=ValueError, sleep=_async_noop_sleep
        )
        assert asyncio.run(caller(target)) == "ok"
        assert len(calls) == 3

    def test_giveup_raises(self):
        async def target():
            raise ValueError("fail")

        caller = AsyncRetryingCaller(
            max_tries=2, exception=ValueError, sleep=_async_noop_sleep
        )
        with pytest.raises(ValueError, match="fail"):
            asyncio.run(caller(target))

    def test_raise_on_giveup_false(self):
        async def target():
            raise ValueError("fail")

        caller = AsyncRetryingCaller(
            max_tries=2,
            exception=ValueError,
            raise_on_giveup=False,
            sleep=_async_noop_sleep,
        )
        assert asyncio.run(caller(target)) is None

    def test_passes_args_kwargs(self):
        async def target(a, b):
            return a + b

        caller = AsyncRetryingCaller(sleep=_async_noop_sleep)
        assert asyncio.run(caller(target, 2, b=5)) == 7

    def test_on_returns_new_caller(self):
        caller = AsyncRetryingCaller(max_tries=1)
        new = caller.on(ValueError)
        assert isinstance(new, AsyncRetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        assert caller._exception is None

    def test_on_accepts_sequence(self):
        caller = AsyncRetryingCaller()
        new = caller.on([ValueError, KeyError])
        assert new._exception == [ValueError, KeyError]

    def test_on_preserves_config(self):
        caller = AsyncRetryingCaller(max_tries=9, name="async-name")
        new = caller.on(ValueError)
        assert new._max_tries == 9
        assert new._name == "async-name"

    def test_on_success_handler(self):
        seen = []

        async def on_success(state):
            seen.append(state)

        async def target():
            return 1

        caller = AsyncRetryingCaller(
            on_success=on_success, sleep=_async_noop_sleep
        )
        asyncio.run(caller(target))
        assert len(seen) == 1

    def test_default_logger(self):
        caller = AsyncRetryingCaller()
        assert caller._logger == "backon"

    def test_copy(self):
        def handler(state):
            return None

        caller = AsyncRetryingCaller(
            max_tries=3,
            max_time=6,
            exception=KeyError,
            predicate=operator.not_,
            raise_on_giveup=False,
            logger="lg",
            backoff_log_level=logging.WARNING,
            giveup_log_level=logging.CRITICAL,
            sleep=lambda s: None,
            name="async-nm",
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            before=handler,
            after=handler,
            attempt_timeout=3.0,
            multiplier=4,
        )
        c = caller.copy()
        assert isinstance(c, AsyncRetryingCaller)
        assert c is not caller
        assert c._max_tries == 3
        assert c._max_time == 6
        assert c._exception is KeyError
        assert c._predicate is operator.not_
        assert c._raise_on_giveup is False
        assert c._logger == "lg"
        assert c._backoff_log_level == logging.WARNING
        assert c._giveup_log_level == logging.CRITICAL
        assert c._name == "async-nm"
        assert c._on_success is handler
        assert c._on_backoff is handler
        assert c._on_giveup is handler
        assert c._on_attempt is handler
        assert c._before_sleep is handler
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 3.0
        assert c._wait_gen_kwargs == {"multiplier": 4}

    def test_copy_is_independent(self):
        caller = AsyncRetryingCaller(max_tries=1)
        c = caller.copy()
        c._max_tries = 42
        assert caller._max_tries == 1

    def test_copy_default_logger(self):
        caller = AsyncRetryingCaller()
        c = caller.copy()
        assert c._logger == "backon"
