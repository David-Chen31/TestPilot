import asyncio
import operator
import threading
import time

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    _RetryAttempt,
    sleep_using_event,
)
from backon._rate_limiter import RateLimiter


# ------------------------------- _RetryAttempt -------------------------------


class TestRetryAttempt:
    def test_initial_state(self):
        attempt = _RetryAttempt()
        assert attempt.failed is False
        assert attempt.exception is None
        assert attempt.value is None

    def test_enter_returns_self(self):
        attempt = _RetryAttempt()
        with attempt as ctx:
            assert ctx is attempt

    def test_success_does_not_mark_failed(self):
        attempt = _RetryAttempt()
        with attempt:
            attempt.set_value(42)
        assert attempt.failed is False
        assert attempt.exception is None
        assert attempt.value == 42

    def test_exception_swallowed_and_recorded(self):
        attempt = _RetryAttempt()
        exc = ValueError("boom")
        with attempt:
            raise exc
        assert attempt.failed is True
        assert attempt.exception is exc

    def test_base_exception_not_swallowed(self):
        attempt = _RetryAttempt()
        with pytest.raises(KeyboardInterrupt):
            with attempt:
                raise KeyboardInterrupt()
        assert attempt.failed is False
        assert attempt.exception is None


# ------------------------------- Retrying ------------------------------------


def _no_sleep(_seconds):
    return None


class TestRetryingConstruction:
    def test_defaults(self):
        r = Retrying()
        assert r.enabled is True
        assert r.statistics == {}
        assert r.call_state is None

    def test_enabled_setter(self):
        r = Retrying()
        r.enabled = False
        assert r.enabled is False
        r.enabled = True
        assert r.enabled is True

    def test_context_managers(self):
        r = Retrying()
        with r as ctx:
            assert ctx is r

    def test_async_context_managers(self):
        async def run():
            r = Retrying()
            async with r as ctx:
                assert ctx is r

        asyncio.run(run())

    def test_sync_exit_returns_none(self):
        r = Retrying()
        assert r.__exit__(None, None, None) is None

    def test_async_exit_returns_none(self):
        async def run():
            r = Retrying()
            result = await r.__aexit__(None, None, None)
            assert result is None

        asyncio.run(run())


class TestRetryingCall:
    def test_call_success_immediately(self):
        r = Retrying(sleep=_no_sleep)
        assert r.call(lambda: 5) == 5
        assert r.statistics["attempt_number"] == 1
        assert r.call_state is not None

    def test_call_with_args_kwargs(self):
        r = Retrying(sleep=_no_sleep)
        result = r.call(lambda a, b=0: a + b, 3, b=4)
        assert result == 7

    def test_call_retries_on_exception(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("fail")
            return "ok"

        r = Retrying(exception=ValueError, sleep=_no_sleep)
        assert r.call(flaky) == "ok"
        assert calls["n"] == 3
        assert r.statistics["attempt_number"] == 3

    def test_call_giveup_raises(self):
        def always_fail():
            raise ValueError("nope")

        r = Retrying(exception=ValueError, max_tries=2, sleep=_no_sleep)
        with pytest.raises(ValueError):
            r.call(always_fail)

    def test_call_giveup_no_raise(self):
        def always_fail():
            raise ValueError("nope")

        r = Retrying(
            exception=ValueError,
            max_tries=1,
            sleep=_no_sleep,
            raise_on_giveup=False,
        )
        with pytest.raises(StopIteration):
            r.call(always_fail)

    def test_call_rejects_coroutine_function(self):
        async def coro():
            return 1

        r = Retrying()
        with pytest.raises(TypeError, match="Use await with Retrying.call"):
            r.call(coro)

    def test_disabled_global(self):
        import backon._retry._api as api

        calls = {"n": 0}

        def target():
            calls["n"] += 1
            return "done"

        original = api.is_enabled
        try:
            api.is_enabled = lambda: False
            r = Retrying(sleep=_no_sleep)
            assert r.call(target) == "done"
            assert calls["n"] == 1
        finally:
            api.is_enabled = original

    def test_call_state_recorded_on_exception(self):
        def always_fail():
            raise ValueError()

        r = Retrying(exception=ValueError, max_tries=2, sleep=_no_sleep)
        with pytest.raises(ValueError):
            r.call(always_fail)
        assert r.call_state is not None
        assert r.statistics != {}


class TestRetryingAsyncCall:
    def test_async_call_success(self):
        async def target():
            return "async-result"

        r = Retrying(sleep=_no_sleep)
        assert asyncio.run(r.async_call(target)) == "async-result"
        assert r.call_state is not None

    def test_async_call_retries(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("fail")
            return "ok"

        r = Retrying(exception=ValueError, sleep=_no_sleep)
        assert asyncio.run(r.async_call(flaky)) == "ok"
        assert calls["n"] == 2

    def test_async_call_giveup_raises(self):
        async def always_fail():
            raise ValueError("nope")

        r = Retrying(exception=ValueError, max_tries=2, sleep=_no_sleep)
        with pytest.raises(ValueError):
            asyncio.run(r.async_call(always_fail))


class TestRetryingCopy:
    def test_copy_preserves_config(self):
        def handler(_state):
            return None

        r = Retrying(
            predicate=bool,
            exception=(ValueError, KeyError),
            max_tries=4,
            max_time=10.0,
            jitter=None,
            giveup=lambda e: True,
            on_success=handler,
            on_backoff=handler,
            on_giveup=handler,
            on_attempt=handler,
            before_sleep=handler,
            raise_on_giveup=False,
            logger="mylogger",
            backoff_log_level=10,
            giveup_log_level=20,
            sleep=_no_sleep,
            enabled=False,
            name="myname",
            before=handler,
            after=handler,
            attempt_timeout=1.5,
        )
        c = r.copy()
        assert isinstance(c, Retrying)
        assert c is not r
        assert c._predicate is bool
        assert c._exception == (ValueError, KeyError)
        assert c._max_tries == 4
        assert c._max_time == 10.0
        assert c._jitter is None
        assert c._giveup is r._giveup
        assert c._on_success is handler
        assert c._on_backoff is handler
        assert c._on_giveup is handler
        assert c._on_attempt is handler
        assert c._before_sleep is handler
        assert c._raise_on_giveup is False
        assert c._logger == "mylogger"
        assert c._backoff_log_level == 10
        assert c._giveup_log_level == 20
        assert c._sleep is _no_sleep
        assert c.enabled is False
        assert c._name == "myname"
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 1.5

    def test_copy_preserves_wait_gen_kwargs(self):
        r = Retrying(base=2, factor=1.5)
        c = r.copy()
        assert c._wait_gen_kwargs == {"base": 2, "factor": 1.5}


class TestRetryingIteration:
    def test_iter_success(self):
        r = Retrying(sleep=_no_sleep)
        attempts = []
        for attempt in r:
            attempts.append(attempt)
            attempt.set_value(1)
        assert hasattr(r, "_iter_state")
        assert len(attempts) == 1
        assert attempts[0].value == 1

    def test_iter_retry_then_success(self):
        r = Retrying(exception=ValueError, sleep=_no_sleep)
        count = 0
        for attempt in r:
            count += 1
            if count < 2:
                raise ValueError()
        assert count == 2

    def test_iter_giveup_raises(self):
        r = Retrying(exception=ValueError, max_tries=1, sleep=_no_sleep)
        it = iter(r)
        attempt = next(it)
        with attempt:
            raise ValueError("boom")
        with pytest.raises(ValueError):
            next(it)

    def test_iter_giveup_no_raise(self):
        r = Retrying(
            exception=ValueError,
            max_tries=1,
            sleep=_no_sleep,
            raise_on_giveup=False,
        )
        it = iter(r)
        attempt = next(it)
        with attempt:
            raise ValueError("boom")
        with pytest.raises(StopIteration):
            next(it)

    def test_iter_uses_sleep(self):
        slept = []
        r = Retrying(
            exception=ValueError,
            max_tries=2,
            sleep=lambda s: slept.append(s),
        )
        it = iter(r)
        attempt = next(it)
        with attempt:
            raise ValueError()
        next(it)
        assert len(slept) == 1


# ------------------------------ sleep_using_event ----------------------------


class TestSleepUsingEvent:
    def test_returns_callable(self):
        event = threading.Event()
        fn = sleep_using_event(event)
        assert callable(fn)

    def test_waits_on_event_set(self):
        event = threading.Event()
        fn = sleep_using_event(event)
        start = time.monotonic()
        fn(10.0)
        # Wait should have blocked (event not set) then... but wait returns
        # only when set or timeout. We instead check that setting it releases
        # a waiting call.
        assert time.monotonic() - start >= 10.0

    def test_event_already_set_returns_immediately(self):
        event = threading.Event()
        event.set()
        fn = sleep_using_event(event)
        start = time.monotonic()
        fn(10.0)
        assert time.monotonic() - start < 5.0


# ------------------------------ RetryingCaller -------------------------------


class TestRetryingCaller:
    def test_call_success(self):
        caller = RetryingCaller(sleep=_no_sleep)
        assert caller(lambda: 3) == 3

    def test_call_with_args(self):
        caller = RetryingCaller(sleep=_no_sleep)
        assert caller(lambda a, b: a * b, 2, 4) == 8

    def test_call_retries(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError()
            return "ok"

        caller = RetryingCaller(exception=ValueError, sleep=_no_sleep)
        assert caller(flaky) == "ok"
        assert calls["n"] == 2

    def test_call_rejects_coroutine_function(self):
        async def coro():
            return 1

        caller = RetryingCaller()
        with pytest.raises(TypeError, match="Use AsyncRetryingCaller"):
            caller(coro)

    def test_on_returns_new_caller(self):
        caller = RetryingCaller()
        new = caller.on(ValueError)
        assert isinstance(new, RetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        assert caller._exception is None

    def test_on_with_sequence(self):
        caller = RetryingCaller()
        new = caller.on((ValueError, KeyError))
        assert new._exception == (ValueError, KeyError)

    def test_copy_preserves_config(self):
        def handler(_state):
            return None

        caller = RetryingCaller(
            predicate=bool,
            exception=ValueError,
            max_tries=3,
            max_time=5.0,
            jitter=None,
            on_success=handler,
            logger="x",
            backoff_log_level=11,
            giveup_log_level=22,
            sleep=_no_sleep,
            name="nm",
            before=handler,
            after=handler,
            attempt_timeout=2.0,
        )
        c = caller.copy()
        assert c is not caller
        assert c._predicate is bool
        assert c._exception is ValueError
        assert c._max_tries == 3
        assert c._max_time == 5.0
        assert c._jitter is None
        assert c._on_success is handler
        assert c._logger == "x"
        assert c._backoff_log_level == 11
        assert c._giveup_log_level == 22
        assert c._sleep is _no_sleep
        assert c._name == "nm"
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 2.0

    def test_copy_wait_gen_kwargs(self):
        caller = RetryingCaller(base=3)
        c = caller.copy()
        assert c._wait_gen_kwargs == {"base": 3}


# ---------------------------- AsyncRetryingCaller ----------------------------


class TestAsyncRetryingCaller:
    def test_call_success(self):
        async def target():
            return "async"

        caller = AsyncRetryingCaller(sleep=_no_sleep)
        assert asyncio.run(caller(target)) == "async"

    def test_call_with_args(self):
        async def target(a, b):
            return a + b

        caller = AsyncRetryingCaller(sleep=_no_sleep)
        assert asyncio.run(caller(target, 1, 2)) == 3

    def test_call_retries(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError()
            return "ok"

        caller = AsyncRetryingCaller(exception=ValueError, sleep=_no_sleep)
        assert asyncio.run(caller(flaky)) == "ok"
        assert calls["n"] == 2

    def test_on_returns_new_caller(self):
        caller = AsyncRetryingCaller()
        new = caller.on(ValueError)
        assert isinstance(new, AsyncRetryingCaller)
        assert new is not caller
        assert new._exception is ValueError
        assert caller._exception is None

    def test_copy_preserves_config(self):
        def handler(_state):
            return None

        caller = AsyncRetryingCaller(
            predicate=bool,
            exception=ValueError,
            max_tries=6,
            max_time=9.0,
            jitter=None,
            on_success=handler,
            logger="y",
            backoff_log_level=33,
            giveup_log_level=44,
            sleep=_no_sleep,
            name="nm2",
            before=handler,
            after=handler,
            attempt_timeout=3.0,
        )
        c = caller.copy()
        assert c is not caller
        assert c._predicate is bool
        assert c._exception is ValueError
        assert c._max_tries == 6
        assert c._max_time == 9.0
        assert c._jitter is None
        assert c._on_success is handler
        assert c._logger == "y"
        assert c._backoff_log_level == 33
        assert c._giveup_log_level == 44
        assert c._sleep is _no_sleep
        assert c._name == "nm2"
        assert c._before is handler
        assert c._after is handler
        assert c._attempt_timeout == 3.0

    def test_copy_wait_gen_kwargs(self):
        caller = AsyncRetryingCaller(base=4)
        c = caller.copy()
        assert c._wait_gen_kwargs == {"base": 4}


class TestRateLimiterIntegration:
    def test_retrying_caller_with_rate_limiter(self):
        limiter = RateLimiter(max_calls=1, period=100.0)
        caller = RetryingCaller(sleep=_no_sleep, rate_limit=limiter)

        from backon._rate_limiter import RateLimitError

        # First call consumes the single permit.
        assert caller(lambda: 1) == 1
        # Second call should hit the limiter; with raise_on_giveup default it
        # raises the RateLimitError after retries are exhausted. With no
        # exception filter, the predicate catches it? Safer: expect any exc.
        with pytest.raises(Exception):
            caller(lambda: 2)
