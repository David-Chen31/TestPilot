import asyncio
import inspect
import logging
import operator
import time
import types

import pytest

from backon._retry._classes import (
    AsyncRetryingCaller,
    Retrying,
    RetryingCaller,
    sleep_using_event,
    _RetryAttempt,
)
from backon._state import Attempt, RetryCallState, RetryState
from backon._jitter import full_jitter


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

    def test_exit_swallows_exception(self):
        attempt = _RetryAttempt()
        exc = ValueError("boom")
        result = attempt.__exit__(ValueError, exc, None)
        assert result is True
        assert attempt.failed is True
        assert attempt.exception is exc

    def test_exit_no_exception(self):
        attempt = _RetryAttempt()
        result = attempt.__exit__(None, None, None)
        assert result is False
        assert attempt.failed is False

    def test_exit_base_exception_not_swallowed(self):
        attempt = _RetryAttempt()
        exc = KeyboardInterrupt()
        result = attempt.__exit__(KeyboardInterrupt, exc, None)
        assert result is False
        assert attempt.failed is False


# ---------------------------------------------------------------------------
# sleep_using_event
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self.calls = []

    def wait(self, timeout=None):
        self.calls.append(timeout)


class TestSleepUsingEvent:
    def test_returns_callable(self):
        ev = _FakeEvent()
        fn = sleep_using_event(ev)
        assert callable(fn)

    def test_wait_called_with_timeout(self):
        ev = _FakeEvent()
        fn = sleep_using_event(ev)
        fn(2.5)
        assert ev.calls == [2.5]

    def test_returns_none(self):
        ev = _FakeEvent()
        fn = sleep_using_event(ev)
        assert fn(1.0) is None


# ---------------------------------------------------------------------------
# Retrying - construction / properties
# ---------------------------------------------------------------------------


class TestRetryingProperties:
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

    def test_context_manager(self):
        r = Retrying()
        with r as ctx:
            assert ctx is r
        # __exit__ returns None (does not suppress)

    def test_exit_does_not_suppress(self):
        r = Retrying()
        assert r.__exit__(None, None, None) is None

    def test_name_stored(self):
        r = Retrying(name="myname")
        assert r._name == "myname"


# ---------------------------------------------------------------------------
# Retrying - copy
# ---------------------------------------------------------------------------


class TestRetryingCopy:
    def test_copy_preserves_attributes(self):
        r = Retrying(
            max_tries=5,
            max_time=10.0,
            name="orig",
            enabled=False,
            raise_on_giveup=False,
            backoff_log_level=logging.DEBUG,
            giveup_log_level=logging.WARNING,
        )
        c = r.copy()
        assert c is not r
        assert isinstance(c, Retrying)
        assert c._max_tries == 5
        assert c._max_time == 10.0
        assert c._name == "orig"
        assert c._enabled is False
        assert c._raise_on_giveup is False
        assert c._backoff_log_level == logging.DEBUG
        assert c._giveup_log_level == logging.WARNING

    def test_copy_preserves_wait_gen_kwargs(self):
        r = Retrying(base=2, factor=3, max_value=100)
        c = r.copy()
        assert c._wait_gen_kwargs == {"base": 2, "factor": 3, "max_value": 100}


# ---------------------------------------------------------------------------
# Retrying - iterable protocol
# ---------------------------------------------------------------------------


def _stub_sleep_collector():
    collected = []

    def sleep(s):
        collected.append(s)

    return collected, sleep


class TestRetryingIter:
    def test_iter_returns_self(self):
        r = Retrying()
        assert iter(r) is r

    def test_single_success_then_stop(self):
        r = Retrying()
        it = iter(r)
        attempt = next(it)
        assert isinstance(attempt, _RetryAttempt)
        attempt.set_value(10)
        with pytest.raises(StopIteration):
            next(it)

    def test_retries_on_failure_then_success(self):
        r = Retrying(sleep=lambda s: None)
        it = iter(r)
        initial_stop = r._iter_state.tries
        first = next(it)
        with pytest.raises(ValueError):
            with first:
                raise ValueError("fail")

        second = next(it)
        assert r._iter_state.tries == initial_stop + 2
        second.set_value("done")
        with pytest.raises(StopIteration):
            next(it)

    def test_failure_with_raise_on_giveup_raises(self):
        r = Retrying(max_tries=1)
        it = iter(r)
        first = next(it)
        with pytest.raises(RuntimeError) as ei:
            with first:
                raise RuntimeError("boom")
        # consuming again should re-raise due to giveup + raise_on_giveup
        with pytest.raises(RuntimeError) as ei2:
            next(it)
        assert isinstance(ei2.value, RuntimeError)

    def test_failure_no_raise_on_giveup_stops(self):
        r = Retrying(max_tries=1, raise_on_giveup=False)
        it = iter(r)
        first = next(it)
        with pytest.raises(KeyError):
            with first:
                raise KeyError("x")
        with pytest.raises(StopIteration):
            next(it)

    def test_sleep_called_with_positive_wait(self):
        collected, sleep = _stub_sleep_collector()
        r = Retrying(sleep=sleep, jitter=None, max_tries=3)
        it = iter(r)
        first = next(it)
        with pytest.raises(ValueError):
            with first:
                raise ValueError()
        next(it)
        # expo default yields some positive value on second attempt
        assert len(collected) == 1
        assert collected[0] > 0

    def test_no_sleep_when_zero_wait(self):
        collected, sleep = _stub_sleep_collector()
        r = Retrying(sleep=sleep, jitter=None, max_tries=3)
        # Force zero wait via max_time=0-ish; use tiny max_time
        r2 = Retrying(sleep=sleep, jitter=None, max_tries=3, max_time=0.000001)
        it = iter(r2)
        first = next(it)
        with pytest.raises(ValueError):
            with first:
                raise ValueError()
        next(it)
        # seconds = min(value, max_time - elapsed) likely <= 0, so sleep not called
        # It either isn't called, or called with 0; assert no positive sleeps
        assert all(s <= 0 for s in collected)

    def test_iter_state_initialized(self):
        r = Retrying(name="myname", max_tries=2)
        iter(r)
        assert r._iter_state.target == "myname"
        assert r._iter_state.tries == 0
        assert isinstance(r._iter_state, RetryState)


# ---------------------------------------------------------------------------
# Retrying - call / async_call
# ---------------------------------------------------------------------------


class TestRetryingCall:
    def test_call_success_first_try(self):
        r = Retrying()
        calls = []

        def target(x, y=0):
            calls.append((x, y))
            return x + y

        result = r.call(target, 3, y=4)
        assert result == 7
        assert calls == [(3, 4)]

    def test_call_rejects_coroutine_function(self):
        r = Retrying()

        async def coro():
            return 1

        with pytest.raises(TypeError) as ei:
            r.call(coro)
        assert "async" in str(ei.value).lower()

    def test_call_retries_then_succeeds(self):
        r = Retrying(max_tries=3, sleep=lambda s: None)
        attempt_counter = {"n": 0}

        def target():
            attempt_counter["n"] += 1
            if attempt_counter["n"] < 3:
                raise ValueError("fail")
            return "ok"

        result = r.call(target)
        assert result == "ok"
        assert attempt_counter["n"] == 3

    def test_call_gives_up_raises(self):
        r = Retrying(max_tries=2, sleep=lambda s: None)

        def target():
            raise ValueError("always")

        with pytest.raises(ValueError):
            r.call(target)

    def test_call_passes_positional_and_kwargs(self):
        r = Retrying()
        captured = {}

        def target(a, b, c=None):
            captured.update(a=a, b=b, c=c)
            return True

        assert r.call(target, 1, 2, c=3) is True
        assert captured == {"a": 1, "b": 2, "c": 3}


class TestRetryingAsyncCall:
    def test_async_call_success(self):
        r = Retrying()

        async def target(x):
            return x * 2

        result = asyncio.run(r.async_call(target, 21))
        assert result == 42

    def test_async_call_retries(self):
        r = Retrying(max_tries=3, sleep=lambda s: asyncio.sleep(0))
        state = {"n": 0}

        async def target():
            state["n"] += 1
            if state["n"] < 2:
                raise ValueError("fail")
            return "ok"

        result = asyncio.run(r.async_call(target))
        assert result == "ok"
        assert state["n"] == 2


# ---------------------------------------------------------------------------
# RetryingCaller
# ---------------------------------------------------------------------------


class TestRetryingCaller:
    def test_callable_success(self):
        caller = RetryingCaller()
        result = caller(lambda x: x + 1, 5)
        assert result == 6

    def test_rejects_coroutine_function(self):
        caller = RetryingCaller()

        async def coro():
            return 1

        with pytest.raises(TypeError) as ei:
            caller(coro)
        assert "async" in str(ei.value).lower()

    def test_on_returns_new_caller_with_exception(self):
        caller = RetryingCaller()
        new = caller.on(ValueError)
        assert new is not caller
        assert isinstance(new, RetryingCaller)
        assert new._exception is ValueError
        # original unchanged
        assert caller._exception is None

    def test_on_accepts_tuple(self):
        caller = RetryingCaller()
        new = caller.on((ValueError, KeyError))
        assert new._exception == (ValueError, KeyError)

    def test_retries_with_on(self):
        caller = RetryingCaller(max_tries=3, sleep=lambda s: None).on(ValueError)
        state = {"n": 0}

        def target():
            state["n"] += 1
            if state["n"] < 2:
                raise ValueError("fail")
            return "done"

        assert caller(target) == "done"
        assert state["n"] == 2

    def test_on_does_not_retry_unlisted_exception(self):
        caller = RetryingCaller(max_tries=3, sleep=lambda s: None).on(ValueError)

        def target():
            raise KeyError("nope")

        with pytest.raises(KeyError):
            caller(target)

    def test_copy_preserves_attributes(self):
        caller = RetryingCaller(
            max_tries=4,
            name="c",
            raise_on_giveup=False,
            backoff_log_level=logging.DEBUG,
        )
        c = caller.copy()
        assert c is not caller
        assert isinstance(c, RetryingCaller)
        assert c._max_tries == 4
        assert c._name == "c"
        assert c._raise_on_giveup is False
        assert c._backoff_log_level == logging.DEBUG

    def test_copy_preserves_wait_gen_kwargs(self):
        caller = RetryingCaller(base=2, factor=4)
        c = caller.copy()
        assert c._wait_gen_kwargs == {"base": 2, "factor": 4}


# ---------------------------------------------------------------------------
# AsyncRetryingCaller
# ---------------------------------------------------------------------------


class TestAsyncRetryingCaller:
    def test_call_success(self):
        caller = AsyncRetryingCaller()

        async def target(x):
            return x + 1

        result = asyncio.run(caller(target, 1))
        assert result == 2

    def test_retries_then_succeeds(self):
        caller = AsyncRetryingCaller(
            max_tries=3, sleep=lambda s: asyncio.sleep(0)
        ).on(ValueError)
        state = {"n": 0}

        async def target():
            state["n"] += 1
            if state["n"] < 3:
                raise ValueError("fail")
            return "ok"

        assert asyncio.run(caller(target)) == "ok"
        assert state["n"] == 3

    def test_on_returns_new_async_caller(self):
        caller = AsyncRetryingCaller()
        new = caller.on(ValueError)
        assert new is not caller
        assert isinstance(new, AsyncRetryingCaller)
        assert new._exception is ValueError
        assert caller._exception is None

    def test_copy_preserves_attributes(self):
        caller = AsyncRetryingCaller(
            max_tries=5,
            name="ac",
            raise_on_giveup=False,
        )
        c = caller.copy()
        assert c is not caller
        assert isinstance(c, AsyncRetryingCaller)
        assert c._max_tries == 5
        assert c._name == "ac"
        assert c._raise_on_giveup is False

    def test_copy_preserves_wait_gen_kwargs(self):
        caller = AsyncRetryingCaller(base=3)
        c = caller.copy()
        assert c._wait_gen_kwargs == {"base": 3}


# ---------------------------------------------------------------------------
# Statistics / call_state wiring through call
# ---------------------------------------------------------------------------


class TestStatisticsWiring:
    def test_call_populates_state(self):
        r = Retrying(max_tries=1)

        def target():
            return 1

        r.call(target)
        stats = r.statistics
        assert isinstance(stats, dict)
        assert "attempt_number" in stats
        assert "elapsed" in stats

    def test_call_state_available(self):
        r = Retrying(max_tries=1)

        r.call(lambda: 1)
        cs = r.call_state
        assert isinstance(cs, RetryCallState)

    def test_statistics_empty_before_call(self):
        r = Retrying()
        assert r.statistics == {}
        assert r.call_state is None
